"""Onboard planning: compact uint8 cost map + hierarchical (coarse global -> rolling local 64x64) A*.

* ``CompactCostMap`` keeps the rover's global cost / hazard / energy as uint8 (3 B/cell instead of 32 B/cell
  for float32 cost + hazard + 5 components + energy). Terrain features (curvature, P(slope > 25 deg), entropy
  term) are computed on the fly per window from tiered-store tiles and never stored globally.
* ``CostView`` is a lazy, read-only, duck-typed CostMap: ``view.cost[idx]`` dequantises only the cells asked for
  (``evaluate_path`` works on it unchanged), with optional rover rules — driven-over cells passable, hazard hard
  limit — so no full float map is materialised for checks.
* ``hierarchical_plan``: A* on a coarse 4x grid (from the compact cost map) for the global corridor, then local
  A* inside a 64x64 window rolling forward through the corridor. Same ``astar`` / ``evaluate_path`` as the full
  planner; the local A* never enters an impassable cell.
Run: python -m onboard.planner
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from core.config import DEFAULT, Config
from core.types import N_CLASSES, RC, Belief, CostMap, Plan, TerrainLayers
from costmap.fuse import build_costmap
from onboard.quantize import COST_INF, dequantize_cost, dequantize_prob_log, quantize_cost, quantize_prob_log
from onboard.tiered_store import TieredTileStore
from planner.astar import astar, evaluate_path
from terrain.features import curvature, p_slope_exceed

_F32 = np.float32
HALO = 8                      # cells of extra DEM around a window for curvature + correlated-noise Monte-Carlo
COARSE = 4


# =========================================================================== on-the-fly window features
def window_layers(store: TieredTileStore, r0: int, r1: int, c0: int, c1: int, cfg: Config = DEFAULT,
                  boot: bool = False) -> TerrainLayers:
    """TerrainLayers for rows r0:r1, cols c0:c1: slope/roughness from tiles, curvature + P(slope>limit) on the fly."""
    H, W = store.shape
    R0, R1, C0, C1 = max(0, r0 - HALO), min(H, r1 + HALO), max(0, c0 - HALO), min(W, c1 + HALO)
    win = (store.boot_region if boot else store.get_layers_window)(R0, R1, C0, C1)
    inner = (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0))
    curv = curvature(win["dem"], store.cell_size)
    pse = p_slope_exceed(win["dem"], store.cell_size, sigma=cfg.dem_noise_sigma, limit=cfg.slope_max_deg,
                         n=cfg.mc_samples, seed=cfg.seed + 7919 * r0 + c0, corr_cells=cfg.dem_noise_corr_cells,
                         max_bytes=cfg.mc_max_bytes)
    return TerrainLayers(win["slope"][inner], win["rough"][inner], curv[inner], pse[inner])


def cost_ceiling(cfg: Config) -> float:
    """Largest finite cost the fused cost map can produce (fixes the uint8 scale; nothing is clipped)."""
    return float(cfg.w_slope + cfg.w_rough + cfg.w_class * max(cfg.class_risk.values())
                 + cfg.w_energy * cfg.energy_k_uphill * math.tan(math.radians(min(cfg.slope_max_deg, 89.0)))
                 + cfg.lambda_unc) * 1.0001


# =========================================================================== lazy views
class _Field:
    """Lazy float32 field over a uint8 array: ``f[idx]`` dequantises only ``q[idx]``; ``patch`` applies rules."""

    def __init__(self, q: np.ndarray, deq: Callable[[np.ndarray], np.ndarray],
                 patch: Callable[[tuple, np.ndarray], np.ndarray] | None = None) -> None:
        self.q, self.deq, self.patch = q, deq, patch

    @property
    def shape(self) -> tuple[int, ...]:
        return self.q.shape

    def __getitem__(self, idx):
        v = self.deq(self.q[idx])
        return self.patch(idx, v) if self.patch is not None else v

    def __array__(self, dtype=None, copy=None):          # full materialisation (rare: logging only)
        a = self[:, :]
        return a.astype(dtype) if dtype is not None else a

    def copy(self) -> np.ndarray:
        return np.array(self[:, :])


class CostView:
    """Duck-typed read-only CostMap (``cost``, ``hazard_prob``, ``energy_wh_per_m``, ``shape``, ``components``)."""

    def __init__(self, cost: _Field, hazard: _Field, energy: _Field) -> None:
        self.cost, self.hazard_prob, self.energy_wh_per_m = cost, hazard, energy
        self.components: dict[str, np.ndarray] = {}

    @property
    def shape(self) -> tuple[int, int]:
        return self.cost.shape  # type: ignore[return-value]

    def window(self, sl: tuple[slice, slice]) -> CostMap:
        """Real (float32) CostMap for a window — what the local A* consumes."""
        return CostMap(self.cost[sl], self.hazard_prob[sl], energy_wh_per_m=self.energy_wh_per_m[sl])


class CompactCostMap:
    """Global uint8 cost (255 = impassable), hazard (log scale, 1e-6..1), energy (Wh/m) — onboard cost state."""

    def __init__(self, shape: tuple[int, int], cmax: float, emax: float) -> None:
        self.cost_q = np.full(shape, COST_INF, np.uint8)
        self.haz_q = np.zeros(shape, np.uint8)
        self.energy_q = np.zeros(shape, np.uint8)
        self.cmax, self.emax = float(cmax), float(emax)

    @property
    def nbytes(self) -> int:
        return int(self.cost_q.nbytes + self.haz_q.nbytes + self.energy_q.nbytes)

    def write(self, sl: tuple[slice, slice], cm: CostMap) -> None:
        self.cost_q[sl] = quantize_cost(cm.cost, self.cmax)[0]
        self.haz_q[sl] = quantize_prob_log(cm.hazard_prob)
        if cm.energy_wh_per_m is not None:
            self.energy_q[sl] = quantize_cost(cm.energy_wh_per_m, self.emax)[0]

    @classmethod
    def build(cls, store: TieredTileStore, belief: Belief, cfg: Config) -> "CompactCostMap":
        """Boot: stream the global cost map tile by tile (features computed per tile, then discarded)."""
        emax = cfg.energy_wh_per_m_flat * (1.0 + cfg.energy_k_uphill * math.tan(math.radians(89.0)))
        ccm = cls(store.shape, cost_ceiling(cfg), min(emax, cfg.energy_wh_per_m_flat * 60.0))
        for tr in range(store.ntr):
            for tc in range(store.ntc):
                sl = store._tile_slices((tr, tc))
                L = window_layers(store, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, cfg, boot=True)
                ccm.write(sl, build_costmap(L, belief.class_probs[sl], cfg, entropy=belief.entropy[sl]))
        return ccm

    def view(self, visited: np.ndarray | None = None, hazard_limit: float | None = None,
             zero_hazard_on_visited: bool = False, cost_q: np.ndarray | None = None,
             haz_q: np.ndarray | None = None) -> CostView:
        """Lazy view with the rover's planning rules (all optional):
        visited cells -> passable at the worst finite cost; hazard > ``hazard_limit`` -> inf (except visited);
        hazard 0 on visited cells. ``cost_q`` / ``haz_q`` override the live arrays (plan-time snapshot)."""
        cq = self.cost_q if cost_q is None else cost_q
        hq = self.haz_q if haz_q is None else haz_q
        worst = self.cmax
        haz_deq = dequantize_prob_log

        def cost_patch(idx, v):
            if visited is None and hazard_limit is None:
                return v
            v = np.array(v, dtype=_F32, copy=True)
            vis = visited[idx] if visited is not None else False
            if hazard_limit is not None:
                hz = haz_deq(hq[idx])
                v = np.where((hz > hazard_limit) & ~np.asarray(vis, bool), np.inf, v)
            if visited is not None:
                v = np.where(np.asarray(vis, bool) & ~np.isfinite(v), worst, v)
            return v.astype(_F32)

        def haz_patch(idx, v):
            return np.where(visited[idx], 0.0, v).astype(_F32) if (zero_hazard_on_visited and visited is not None) else v

        return CostView(_Field(cq, lambda q: dequantize_cost(q, self.cmax), cost_patch),
                        _Field(hq, haz_deq, haz_patch),
                        _Field(self.energy_q, lambda q: dequantize_cost(q, self.emax)))

    def materialize(self) -> CostMap:
        """Full float32 CostMap (for the dashboard / evaluation after a run — not used while driving)."""
        v = self.view()
        return CostMap(v.cost[:, :], v.hazard_prob[:, :], energy_wh_per_m=v.energy_wh_per_m[:, :])


# =========================================================================== hierarchical planner
def _coarse_costmap(view: CostView, f: int = COARSE) -> CostMap:
    """Coarse grid: mean finite cost per fxf block; impassable if >= half the block is impassable.
    Computed band by band (one coarse row at a time), never materialising the full float map."""
    H, W = view.shape
    hc, wc = -(-H // f), -(-W // f)
    cost = np.empty((hc, wc), _F32)
    haz = np.empty((hc, wc), _F32)
    for i in range(hc):
        rs = slice(i * f, min((i + 1) * f, H))
        band, hband = view.cost[rs, :], view.hazard_prob[rs, :]
        padw = wc * f - W
        band = np.pad(band, ((0, f - band.shape[0]), (0, padw)), constant_values=np.inf)
        hband = np.pad(hband, ((0, f - hband.shape[0]), (0, padw)))
        blk = band.reshape(f, wc, f).transpose(1, 0, 2).reshape(wc, f * f)
        fin = np.isfinite(blk)
        frac = fin.mean(axis=1)
        mean = np.where(fin, blk, 0.0).sum(axis=1) / np.maximum(fin.sum(axis=1), 1)
        cost[i] = np.where(frac >= 0.5, mean, np.inf)
        haz[i] = hband.reshape(f, wc, f).transpose(1, 0, 2).reshape(wc, f * f).max(axis=1)
    return CostMap(cost, haz)


def _best_cell(view: CostView, block: tuple[int, int], f: int, H: int, W: int) -> RC | None:
    """Cheapest passable fine cell of a coarse block (the corridor waypoint)."""
    r0, c0 = block[0] * f, block[1] * f
    sub = view.cost[r0:min(r0 + f, H), c0:min(c0 + f, W)]
    if not np.isfinite(sub).any():
        return None
    k = int(np.argmin(np.where(np.isfinite(sub), sub, np.inf)))
    return r0 + k // sub.shape[1], c0 + k % sub.shape[1]


def hierarchical_plan(view: CostView, start: RC, goal: RC, cell_size: float = 1.0, risk_weight: float = 0.0,
                      mode: str = "balanced", connectivity: int = 8, window: int = 64, f: int = COARSE) -> Plan:
    """Coarse corridor A* + rolling local A* in ``window``x``window`` boxes. Returns a fine-grid Plan scored on
    ``view``; ``meta`` records coarse/local expansions, number of legs and whether a full-map fallback was used."""
    t0 = time.perf_counter()
    start, goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    H, W = view.shape
    meta = {"risk_weight": float(risk_weight), "hierarchical": True, "window": window, "coarse_factor": f}
    if not (np.isfinite(view.cost[start]) and np.isfinite(view.cost[goal])):
        meta.update(reason="start_or_goal_impassable", runtime_s=time.perf_counter() - t0)
        return Plan(path=[], success_prob=0.0, mode=mode, meta=meta)
    if start == goal:
        return evaluate_path(view, [start], cell_size, risk_weight, mode, 0, meta)

    coarse = _coarse_costmap(view, f)
    cs_, cg = (start[0] // f, start[1] // f), (goal[0] // f, goal[1] // f)
    coarse.cost[cs_] = coarse.cost[cs_] if np.isfinite(coarse.cost[cs_]) else float(np.nanmax(
        np.where(np.isfinite(coarse.cost), coarse.cost, np.nan)))
    coarse.cost[cg] = coarse.cost[cg] if np.isfinite(coarse.cost[cg]) else float(np.nanmax(
        np.where(np.isfinite(coarse.cost), coarse.cost, np.nan)))
    cplan = astar(coarse, cs_, cg, cell_size * f, risk_weight, mode, connectivity)
    expanded = cplan.expanded_nodes
    meta["coarse_expanded"] = cplan.expanded_nodes
    waypoints: list[RC] = []
    for blk in (cplan.path[1:-1] if cplan.found else []):
        cell = _best_cell(view, blk, f, H, W)
        if cell is not None:
            waypoints.append(cell)
    waypoints.append(goal)

    path: list[RC] = [start]
    cur, wi, legs, half = start, 0, 0, window // 2
    fallback = not cplan.found
    while cur != goal and not fallback:
        r0, c0 = max(0, cur[0] - half), max(0, cur[1] - half)
        r1, c1 = min(H, r0 + window), min(W, c0 + window)
        r0, c0 = max(0, r1 - window), max(0, c1 - window)
        inside = lambda p: r0 <= p[0] < r1 and c0 <= p[1] < c1  # noqa: E731
        j = wi
        while j + 1 < len(waypoints) and inside(waypoints[j + 1]):
            j += 1
        if not inside(waypoints[j]):
            fallback = True
            break
        local = view.window((slice(r0, r1), slice(c0, c1)))
        leg = None
        for k in range(j, wi - 1, -1):                    # farthest reachable waypoint in the window first
            tgt = waypoints[k]
            leg = astar(local, (cur[0] - r0, cur[1] - c0), (tgt[0] - r0, tgt[1] - c0), cell_size, risk_weight,
                        mode, connectivity)
            expanded += leg.expanded_nodes
            if leg.found and len(leg.path) > 1:
                path.extend((r + r0, c + c0) for r, c in leg.path[1:])
                cur, wi, legs = tgt, k + 1, legs + 1
                break
            leg = None
        if leg is None:
            fallback = True
    if fallback:                                          # corridor blocked locally: plan the rest on the full map
        full = CostMap(view.cost[:, :], view.hazard_prob[:, :], energy_wh_per_m=view.energy_wh_per_m[:, :])
        rest = astar(full, cur, goal, cell_size, risk_weight, mode, connectivity)
        expanded += rest.expanded_nodes
        if not rest.found:
            meta.update(reason="goal_unreachable", fallback_full=True, runtime_s=time.perf_counter() - t0)
            return Plan(path=[], success_prob=0.0, mode=mode, expanded_nodes=expanded, meta=meta)
        path.extend(rest.path[1:])
    meta.update(legs=legs, fallback_full=bool(fallback))
    plan = evaluate_path(view, path, cell_size, risk_weight, mode, expanded, meta)
    plan.meta["runtime_s"] = time.perf_counter() - t0
    return plan


def hierarchical_via(view: CostView, start: RC, waypoints: Sequence[RC], goal: RC, cell_size: float = 1.0,
                     risk_weight: float = 0.0, mode: str = "science", connectivity: int = 8) -> Plan:
    """Chain hierarchical legs through science waypoints (same failure meta as ``planner.pareto.plan_via``)."""
    stops = [tuple(start), *[tuple(w) for w in waypoints], tuple(goal)]
    path: list[RC] = [stops[0]]
    expanded = 0
    for a, b in zip(stops, stops[1:]):
        leg = hierarchical_plan(view, a, b, cell_size, risk_weight, mode, connectivity)
        expanded += leg.expanded_nodes
        if not leg.found:
            return Plan(path=[], success_prob=0.0, mode=mode, expanded_nodes=expanded,
                        meta={"reason": "waypoint_unreachable", "failed_leg": [list(a), list(b)],
                              "risk_weight": float(risk_weight)})
        path.extend(leg.path[1:])
    return evaluate_path(view, path, cell_size, risk_weight, mode, expanded,
                         {"risk_weight": float(risk_weight), "waypoints": [list(w) for w in waypoints],
                          "hierarchical": True})


if __name__ == "__main__":
    from belief.bayes import init_prior
    from costmap.fuse import costmap_from_belief
    from onboard.tiered_store import build_store
    from terrain.features import compute_layers
    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    L = compute_layers(w)
    b = init_prior(w)
    st = build_store(w, L, prefetch_sync=True, usage_path=None, simulate_latency=False)
    t0 = time.perf_counter()
    ccm = CompactCostMap.build(st, b, DEFAULT)
    tb = time.perf_counter() - t0
    view = ccm.view()
    hp = hierarchical_plan(view, w.start, w.goal, w.cell_size, DEFAULT.pareto_risk_balanced)
    full = astar(costmap_from_belief(L, b), w.start, w.goal, w.cell_size, DEFAULT.pareto_risk_balanced)
    print(f"compact cost map {ccm.nbytes} B (boot {tb:.2f}s); hierarchical: {hp.length_m:.1f} m P={hp.success_prob:.3f} "
          f"expanded {hp.expanded_nodes} legs {hp.meta.get('legs')} fallback {hp.meta.get('fallback_full')} "
          f"{1000 * hp.meta['runtime_s']:.0f} ms | full A*: {full.length_m:.1f} m P={full.success_prob:.3f} "
          f"expanded {full.expanded_nodes}")
    st.close()
