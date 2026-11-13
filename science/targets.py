"""Science targets: detect candidates, estimate value +/- uncertainty, rank, select under an energy budget.

Science value field (dimensionless, [0, 1]; weights from Config, sum to 1):
    value = w_ice * ice + w_rim * rim + w_bedrock * P(bedrock)
      ice      orbital subsurface-ice signal (``world.ice_signal`` — orbital data, not ground truth)
      rim      crater-rim / layering proxy = clip(curvature / p95(curvature > 0), 0, 1)
               * exp(-0.5 * ((slope - centre) / width)^2)          (convex AND moderately sloped)
      P(bedrock) from the rover's belief (never the hidden class map)

Per target (disc of ``sci_target_radius`` cells):
    value_mean / value_std  bootstrap (``sci_bootstrap_n`` draws): bedrock ~ Bernoulli(P(bedrock)),
                            ice + N(0, sci_ice_noise_std), averaged over the disc
    info_gain               value_mean * mean(entropy / ln 4) over the disc ("worth looking at because uncertain")
    utility                 value_mean - alpha * path_cost - beta * (1 - success_prob)

Selection (``select_under_budget``): greedy by utility per Wh from the current position, re-planning
from each chosen target; a target is feasible only if used + leg + (target -> goal) energy <= budget.
Every selection and skip is logged with its numbers.

Run: python -m science.targets   (prints the ranked table, writes outputs/targets.png)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter

from core.config import DEFAULT, Config
from core.types import BEDROCK, N_CLASSES, RC, Belief, CostMap, Plan, Target, TerrainLayers, World
from explain.log import DecisionLog, confidence_at, make_evidence
from planner.astar import astar

_F32 = np.float32
LN4 = float(np.log(N_CLASSES))


@dataclass
class Selection:
    """Result of ``select_under_budget``.

    Attributes:
        selected: chosen targets in visiting order (``target.plan`` = leg to it).
        skipped: targets not chosen (``target.reason`` says why, from numbers).
        legs: plans start -> t1 -> t2 ... (one per selected target).
        return_leg: last position -> goal (None if unreachable).
        energy_wh: total planned energy including the return leg.
        budget_wh: the budget used.
    """
    selected: list[Target]
    skipped: list[Target]
    legs: list[Plan] = field(default_factory=list)
    return_leg: Plan | None = None
    energy_wh: float = 0.0
    budget_wh: float = 0.0

    @property
    def waypoints(self) -> list[RC]:
        return [t.rc for t in self.selected]


# =========================================================================== value field
def rim_proxy(layers: TerrainLayers, cfg: Config = DEFAULT) -> np.ndarray:
    """(H, W) float32 in [0, 1]: convex (positive curvature) AND moderately sloped cells."""
    curv = layers.curvature
    pos = curv[curv > 0]
    ref = float(np.percentile(pos, cfg.sci_rim_curv_pct)) if pos.size else 1.0
    convex = np.clip(curv / max(ref, 1e-12), 0.0, 1.0)
    band = np.exp(-0.5 * ((layers.slope_deg - cfg.sci_slope_center_deg) / cfg.sci_slope_width_deg) ** 2)
    return (convex * band).astype(_F32)


def science_value_field(layers: TerrainLayers, belief: Belief, ice_signal: np.ndarray,
                        cfg: Config = DEFAULT) -> np.ndarray:
    """(H, W) float32 science value in [0, 1] (formula in module docstring)."""
    ice = np.clip(np.asarray(ice_signal, _F32), 0.0, 1.0)
    v = (cfg.sci_w_ice * ice + cfg.sci_w_rim * rim_proxy(layers, cfg)
         + cfg.sci_w_bedrock * belief.class_probs[..., BEDROCK])
    return np.clip(v, 0.0, 1.0).astype(_F32)


def _disc_cells(rc: RC, radius: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Row/col index arrays of the in-bounds cells within ``radius`` of ``rc``."""
    d = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(d, d, indexing="ij")
    keep = dr ** 2 + dc ** 2 <= radius ** 2
    r, c = rc[0] + dr[keep], rc[1] + dc[keep]
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    return r[ok], c[ok]


def reachable_region(passable: np.ndarray, rc: RC) -> np.ndarray:
    """(H, W) bool: the 8-connected passable component containing ``rc`` (all False if rc is blocked)."""
    from scipy.ndimage import label

    lab, _ = label(passable, structure=np.ones((3, 3), dtype=bool))
    k = lab[int(rc[0]), int(rc[1])]
    return (lab == k) if k > 0 else np.zeros_like(passable, dtype=bool)


def _nms_peaks(value: np.ndarray, passable: np.ndarray, n: int, window: int) -> list[RC]:
    """Top-n local maxima of ``value`` on passable cells, at least window//2 cells apart (Chebyshev)."""
    v = np.where(passable, value, -np.inf)
    mx = maximum_filter(v, size=window, mode="nearest")
    peaks = np.argwhere((v == mx) & np.isfinite(v))
    order = np.argsort(-v[peaks[:, 0], peaks[:, 1]], kind="stable")
    min_sep = max(window // 2, 1)
    chosen: list[RC] = []
    for r, c in peaks[order]:              # loop over peaks (tens), not cells
        if all(max(abs(r - a), abs(c - b)) > min_sep for a, b in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == n:
                break
    return chosen


# =========================================================================== candidates
def detect_candidates(world: World, layers: TerrainLayers, belief: Belief, costmap: CostMap,
                      cfg: Config = DEFAULT, seed: int | None = None, rover_rc: RC | None = None) -> list[Target]:
    """Top ``sci_n_targets`` science candidates with bootstrap value +/- std and info_gain.

    Uses only orbital data (``world.ice_signal``, DEM-derived ``layers``) and the rover's belief.
    Candidates lie on passable cells of ``costmap`` connected to ``rover_rc`` (default ``world.start``),
    so no candidate sits in a walled-off pocket. Ids are ranks by value at detection (0 = best).
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    value = science_value_field(layers, belief, world.ice_signal, cfg)
    rim = rim_proxy(layers, cfg)
    ice = np.clip(world.ice_signal, 0.0, 1.0)
    pbed = belief.class_probs[..., BEDROCK]
    region = reachable_region(costmap.passable, world.start if rover_rc is None else rover_rc)
    peaks = _nms_peaks(value, region, cfg.sci_n_targets, cfg.sci_nms_cells)

    stats = []
    n = int(cfg.sci_bootstrap_n)
    for rc in peaks:                        # <= sci_n_targets iterations, vectorised inside
        rr, cc = _disc_cells(rc, cfg.sci_target_radius, value.shape)
        ice_s = np.clip(ice[rr, cc] + rng.normal(0.0, cfg.sci_ice_noise_std, (n, rr.size)), 0.0, 1.0)
        bed_s = (rng.random((n, rr.size)) < pbed[rr, cc]).astype(np.float64)
        v_s = (cfg.sci_w_ice * ice_s + cfg.sci_w_rim * rim[rr, cc] + cfg.sci_w_bedrock * bed_s).mean(axis=1)
        h_norm = float(belief.entropy[rr, cc].mean()) / LN4
        stats.append((rc, float(v_s.mean()), float(v_s.std(ddof=1)), h_norm,
                      float(ice[rr, cc].mean()), float(rim[rr, cc].mean()), float(pbed[rr, cc].mean())))

    stats.sort(key=lambda s: -s[1])
    targets = []
    for i, (rc, vm, vs, h, ice_m, rim_m, pb) in enumerate(stats):
        t = Target(id=i, rc=rc, value_mean=vm, value_std=vs, info_gain=vm * h)
        t.reason = (f"#{i} at {rc}: value {vm:.2f} +/- {vs:.2f} (ice {ice_m:.2f}, rim {rim_m:.2f}, "
                    f"P(bedrock) {pb:.2f}); info_gain {vm * h:.2f} (entropy {h:.0%} of max)")
        targets.append(t)
    return targets


def _utility(t: Target, leg: Plan, cfg: Config) -> float:
    return t.value_mean - cfg.sci_alpha_per_cost * leg.total_cost - cfg.sci_beta_risk * (1.0 - leg.success_prob)


def plan_targets(targets: Sequence[Target], costmap: CostMap, rover_rc: RC, cfg: Config = DEFAULT,
                 cell_size: float = 1.0, risk_weight: float | None = None) -> list[Target]:
    """Plan rover -> each target; set ``plan``, ``reachable``, ``utility``; return sorted by utility (desc).

    Unreachable targets get ``utility = -inf`` and sort last.
    """
    rw = cfg.pareto_risk_balanced if risk_weight is None else risk_weight
    for t in targets:
        leg = astar(costmap, rover_rc, t.rc, cell_size, rw, mode=f"target{t.id}")
        t.plan, t.reachable = leg, leg.found
        t.utility = _utility(t, leg, cfg) if leg.found else float("-inf")
        if leg.found:
            t.reason = (f"#{t.id} at {t.rc}: value {t.value_mean:.2f} +/- {t.value_std:.2f}, info_gain "
                        f"{t.info_gain:.2f}; path {leg.length_m:.0f} m, {leg.energy_wh:.0f} Wh, P(success) "
                        f"{leg.success_prob:.1%} -> utility {t.utility:.2f}")
        else:
            t.reason = f"#{t.id} at {t.rc}: value {t.value_mean:.2f} +/- {t.value_std:.2f}; unreachable"
    return sorted(targets, key=lambda t: -t.utility)


# =========================================================================== selection
def select_under_budget(targets: Sequence[Target], costmap: CostMap, belief: Belief, start: RC, goal: RC,
                        budget_wh: float | None = None, cfg: Config = DEFAULT, cell_size: float = 1.0,
                        risk_weight: float | None = None, log: DecisionLog | None = None,
                        step: int = 0) -> Selection:
    """Greedy target selection by utility per Wh under an energy budget (see module docstring).

    Every chosen target logs ``select_target`` and every rejected one ``skip_target`` (if ``log`` given).
    """
    budget = cfg.sci_energy_budget_wh if budget_wh is None else float(budget_wh)
    rw = cfg.pareto_risk_balanced if risk_weight is None else risk_weight
    pos, used = tuple(start), 0.0
    remaining = list(targets)
    ret_cache: dict[int, Plan] = {}         # target -> goal leg does not depend on the rover position
    last: dict[int, dict] = {}              # last evaluation of each target (for skip evidence)
    selected: list[Target] = []
    legs: list[Plan] = []

    def ret_leg(t: Target) -> Plan:
        if t.id not in ret_cache:
            ret_cache[t.id] = astar(costmap, t.rc, goal, cell_size, rw, mode="return")
        return ret_cache[t.id]

    while remaining:
        best = None
        for t in remaining:
            leg = astar(costmap, pos, t.rc, cell_size, rw, mode=f"target{t.id}")
            ret = ret_leg(t) if leg.found else Plan()
            if not (leg.found and ret.found):
                last[t.id] = {"why": "reachable", "pos": pos}
                continue
            need = used + leg.energy_wh + ret.energy_wh
            util = _utility(t, leg, cfg)
            ratio = util / max(leg.energy_wh, 1e-6)
            last[t.id] = {"why": None, "pos": pos, "leg": leg, "ret": ret, "need": need, "util": util,
                          "ratio": ratio}
            if need > budget:
                last[t.id]["why"] = "budget_wh"
            elif util <= cfg.sci_min_utility:
                last[t.id]["why"] = "sci_min_utility"
            elif best is None or ratio > best[1]:
                best = (t, ratio)
        if best is None:
            break
        t = best[0]
        ev_t = last[t.id]
        leg, ret = ev_t["leg"], ev_t["ret"]
        t.plan, t.reachable, t.utility = leg, True, ev_t["util"]
        if log is not None:
            ev = make_evidence(t.rc, pos, budget, "budget_wh", target_id=t.id, value_mean=t.value_mean,
                               value_std=t.value_std, info_gain=t.info_gain, utility=ev_t["util"],
                               utility_per_wh=ev_t["ratio"], leg_length_m=leg.length_m,
                               leg_energy_wh=leg.energy_wh, leg_success_prob=leg.success_prob,
                               leg_cost=leg.total_cost, return_energy_wh=ret.energy_wh,
                               energy_used_wh=used, energy_wh=ev_t["need"], budget_wh=budget)
            t.reason = log.record(step, "target_ranked", "select_target", ev, confidence_at(belief, t.rc)).reason
        used += leg.energy_wh
        pos = t.rc
        selected.append(t)
        legs.append(leg)
        remaining.remove(t)

    skipped = []
    for t in remaining:
        info = last.get(t.id, {"why": "reachable", "pos": pos})
        why = info["why"] or "sci_min_utility"
        t.reachable = why != "reachable"
        thr = {"budget_wh": budget, "reachable": 1.0, "sci_min_utility": cfg.sci_min_utility}[why]
        nums = {"target_id": t.id, "value_mean": t.value_mean, "value_std": t.value_std,
                "info_gain": t.info_gain, "budget_wh": budget, "energy_used_wh": used}
        if "leg" in info:
            nums.update(utility=info["util"], energy_wh=info["need"], leg_length_m=info["leg"].length_m,
                        leg_energy_wh=info["leg"].energy_wh, leg_success_prob=info["leg"].success_prob)
            t.utility = info["util"]
        else:
            nums.update(utility=float("-inf"), energy_wh=float("inf"))
            t.utility = float("-inf")
        ev = make_evidence(t.rc, info["pos"], thr, why, **nums)
        if log is not None:
            t.reason = log.record(step, "target_ranked", "skip_target", ev, confidence_at(belief, t.rc)).reason
        else:
            from explain.log import build_reason
            t.reason = build_reason("skip_target", ev)
        skipped.append(t)

    return_leg = astar(costmap, pos, goal, cell_size, rw, mode="return")
    total = used + (return_leg.energy_wh if return_leg.found else float("inf"))
    return Selection(selected, skipped, legs, return_leg if return_leg.found else None, total, budget)


if __name__ == "__main__":
    import time
    from pathlib import Path

    from core.io import hillshade
    import matplotlib.pyplot as plt

    from belief.bayes import init_prior, sense
    from costmap.fuse import costmap_from_belief
    from perception.sensor_model import load_confusion
    from terrain.features import compute_layers
    from world.synthetic import make_world

    cfg = DEFAULT
    w = make_world(200, 200, seed=cfg.seed)
    L = compute_layers(w, cfg)
    C, _ = load_confusion()
    b = sense(init_prior(w, cfg), w, w.start, np.random.default_rng(cfg.seed), cfg, C)
    cm = costmap_from_belief(L, b, cfg)

    t0 = time.perf_counter()
    targets = plan_targets(detect_candidates(w, L, b, cm, cfg), cm, w.start, cfg, w.cell_size)
    t_rank = time.perf_counter() - t0
    print(f"{'id':>3s} {'rc':>11s} {'value':>12s} {'info_gain':>9s} {'path_m':>7s} {'Wh':>6s} "
          f"{'P(succ)':>8s} {'utility':>8s}")
    for t in targets:
        p = t.plan
        print(f"{t.id:3d} {str(t.rc):>11s} {t.value_mean:6.2f} +/-{t.value_std:4.2f} {t.info_gain:9.2f} "
              f"{p.length_m:7.0f} {p.energy_wh:6.0f} {p.success_prob:8.1%} {t.utility:8.2f}")
    print(f"detect + plan {len(targets)} targets: {1000 * t_rank:.0f} ms")

    log = DecisionLog()
    t0 = time.perf_counter()
    sel = select_under_budget(targets, cm, b, w.start, w.goal, cfg.sci_energy_budget_wh, cfg, w.cell_size, log=log)
    print(f"\nselect_under_budget({cfg.sci_energy_budget_wh:.0f} Wh): {len(sel.selected)} selected, "
          f"{len(sel.skipped)} skipped, planned energy {sel.energy_wh:.0f} Wh ({1000 * (time.perf_counter() - t0):.0f} ms)")
    for e in log:
        print(f"  [{e.decision:13s}] conf {e.confidence:.2f}  {e.reason}")

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.imshow(hillshade(w.dem, w.cell_size), cmap="gray")
    im = ax.imshow(science_value_field(L, b, w.ice_signal, cfg), cmap="viridis", alpha=0.6, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="science value")
    for leg in [*sel.legs, *([sel.return_leg] if sel.return_leg else [])]:
        r, c = leg.as_arrays()
        ax.plot(c, r, "-", color="white", lw=1.8)
    chosen = {t.id for t in sel.selected}
    for t in targets:
        col = "#ff5050" if t.id in chosen else "#ffd24d"
        ax.plot(t.rc[1], t.rc[0], "o", ms=10, mfc=col, mec="k")
        ax.annotate(f"#{t.id} {t.value_mean:.2f}±{t.value_std:.2f}", (t.rc[1], t.rc[0]), color="white",
                    fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.plot(w.start[1], w.start[0], "o", ms=11, mfc="lime", mec="k")
    ax.plot(w.goal[1], w.goal[0], "*", ms=15, mfc="magenta", mec="k")
    ax.set_title(f"Science targets (red = selected, budget {cfg.sci_energy_budget_wh:.0f} Wh, "
                 f"plan {sel.energy_wh:.0f} Wh)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/targets.png", dpi=95)
    plt.close(fig)
    print("saved outputs/targets.png")
