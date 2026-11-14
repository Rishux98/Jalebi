"""OnboardRover: the autonomy loop running on tiered terrain tiles + compact cost state + hierarchical A*.

A subclass of ``autonomy.loop.Rover`` — ``autonomy/loop.py`` is untouched. Decision logic (sense, Bayes update,
replan triggers, look-before-step, retreat, budget, logging) is inherited unchanged; only the data access changes:
  * terrain (slope, roughness, DEM) comes from ``TieredTileStore`` windows; curvature and P(slope > 25 deg) are
    computed on the fly per window and never stored globally;
  * the global cost / hazard / energy live in a uint8 ``CompactCostMap``; checks read lazy ``CostView``s;
  * planning is ``hierarchical_plan`` (coarse corridor + rolling 64x64 local A*);
  * every step calls ``store.update_rover(rc, remaining_path)`` -> path-lookahead prefetch, so a replan after a
    hidden hazard immediately re-prioritises tiles.
Run: python -m onboard.rover
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autonomy.loop import Rover, RunResult
from belief.bayes import init_prior
from core.config import DEFAULT, Config
from core.types import BIG_ROCK, RC, Belief, CostMap, Plan, Target, World
from costmap.fuse import build_costmap
from explain.log import DecisionLog
from onboard.planner import HALO, CompactCostMap, CostView, hierarchical_plan, hierarchical_via, window_layers
from onboard.tiered_store import TieredTileStore
from perception.sensor_model import load_confusion
from planner.astar import evaluate_path
from planner.pareto import mode_config


class _CellLayer:
    """Single-cell accessor for log evidence: ``layer[r, c]`` reads one cell through the tile store."""

    def __init__(self, rover: "OnboardRover", key: str) -> None:
        self.rover, self.key = rover, key

    def __getitem__(self, rc):
        r, c = int(rc[0]), int(rc[1])
        L = window_layers(self.rover.store, r, r + 1, c, c + 1, self.rover.cfg)
        return float({"slope": L.slope_deg, "rough": L.roughness, "curv": L.curvature,
                      "pse": L.p_slope_exceed}[self.key][0, 0])


class _TileLayers:
    """Duck-typed TerrainLayers for ``explain.log.cell_evidence`` (single-cell reads only)."""

    def __init__(self, rover: "OnboardRover") -> None:
        self.slope_deg = _CellLayer(rover, "slope")
        self.roughness = _CellLayer(rover, "rough")
        self.curvature = _CellLayer(rover, "curv")
        self.p_slope_exceed = _CellLayer(rover, "pse")


class OnboardRover(Rover):
    """Rover whose terrain + cost state live in the tiered store / compact cost map (see module docstring)."""

    def __init__(self, world: World, store: TieredTileStore, cfg: Config = DEFAULT, mode: str = "balanced",
                 seed: int | None = None, confusion: np.ndarray | None = None,
                 targets: Sequence[Target | RC] | None = None, belief: Belief | None = None,
                 budget_wh: float | None = None, window: int = 64, telemetry: bool = True) -> None:
        # ---- same state as Rover.__init__, minus the full-resolution layers / float cost map
        self.world, self.cfg, self.mode = world, cfg, mode
        self.budget_wh = budget_wh if budget_wh is not None else (cfg.sci_energy_budget_wh if targets else None)
        self.seed = cfg.seed if seed is None else int(seed)
        self.mcfg, self.risk_weight = mode_config(mode, cfg)
        self.rng = np.random.default_rng(self.seed)
        self.confusion = load_confusion()[0] if confusion is None else np.asarray(confusion, np.float32)
        self.store, self.window, self.telemetry = store, int(window), bool(telemetry)
        self.layers = _TileLayers(self)
        b = belief if belief is not None else init_prior(world, cfg, self.seed)
        self.belief = Belief(b.class_probs.copy(), b.entropy.copy(), b.observed_mask.copy())
        self.cs = world.cell_size
        self.rc: RC = world.start
        self.goal: RC = world.goal
        self.targets: list[Target] = [t if isinstance(t, Target) else Target(id=i, rc=t, value_mean=0.0, value_std=0.0)
                                      for i, t in enumerate(targets or [])]
        self.step_idx = 0
        self.log = DecisionLog()
        self.trajectory: list[RC] = [self.rc]
        self.energy_wh = 0.0
        self.distance_m = 0.0
        self.replans = 0
        self.done = False
        self.reached_goal = False
        self.plans: list[Plan] = []
        self.frames: list[dict] = []
        self.entropy_snapshots: dict[int, np.ndarray] = {}
        self.injected: list[dict] = []
        self.entered_true_hazard_cells = 0
        self.n_looks = np.zeros(world.shape, np.int32)
        self.visited = np.zeros(world.shape, bool)
        self.visited[world.start] = True
        self.extra_looks = 0
        self._keep_logged: set[tuple] = set()
        self.path: list[RC] = []
        self.path_idx = 0
        self.plan: Plan | None = None
        self.plan_cm = None
        self.tier_frames: list[np.ndarray] = []

        self._sense()
        self.reach = int(cfg.sensor_radius) + HALO + 1                     # cells a step can touch
        self.store.update_rover(self.rc, [], self.reach)
        self.ccm = CompactCostMap.build(store, self.belief, self.mcfg)       # streamed tile by tile
        self.costmap = self.ccm.view()                                       # lazy raw view (no float copy)
        self.entropy_snapshots[0] = self.belief.entropy.copy()
        self._initial_plan()
        self.store.update_rover(self.rc, self.path, self.reach)
        if self.telemetry:
            self.tier_frames.append(self.store.tier_map())
        self._frame(replanned=False)

    # ------------------------------------------------------------------ data access overrides
    def _rebuild(self, sl: tuple[slice, slice]) -> np.ndarray:
        """Window rebuild from tiles: features on the fly, result quantised into the compact cost map."""
        was = self.ccm.cost_q[sl] != 255
        L = window_layers(self.store, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, self.cfg)
        sub = build_costmap(L, self.belief.class_probs[sl], self.mcfg, entropy=self.belief.entropy[sl])
        self.ccm.write(sl, sub)
        return was & ~np.isfinite(sub.cost)

    def _planning_map(self) -> CostView:
        return self.ccm.view(visited=self.visited, hazard_limit=self.cfg.hazard_prob_max)

    def _route(self, pm, start: RC | None = None) -> Plan:
        s = self.rc if start is None else start
        conn = self.cfg.grid_connectivity
        if self.targets:
            return hierarchical_via(pm, s, self.waypoints, self.goal, self.cs, self.risk_weight, self.mode, conn)
        return hierarchical_plan(pm, s, self.goal, self.cs, self.risk_weight, self.mode, conn, self.window)

    def _adopt(self, p: Plan) -> None:
        p.meta["step"] = self.step_idx
        self.plan, self.path, self.path_idx = p, list(p.path), 0
        self.plan_cm = self.ccm.view(visited=self.visited.copy(), zero_hazard_on_visited=True,
                                     cost_q=self.ccm.cost_q.copy(), haz_q=self.ccm.haz_q.copy())
        self.plans.append(p)
        self._keep_logged.clear()

    def _check(self) -> tuple[dict | None, dict | None]:
        """Identical rules to ``Rover._check``; reads lazy views instead of a materialised float map."""
        cfg = self.cfg
        rem = self.path[self.path_idx:]
        if len(rem) < 2:
            return None, None
        ahead = np.asarray(rem[1:], dtype=np.int64)
        r, c = ahead[:, 0], ahead[:, 1]
        cm = self.ccm.view(visited=self.visited, zero_hazard_on_visited=True)
        old_cost = evaluate_path(self.plan_cm, rem, self.cs, self.risk_weight).total_cost
        new_cost = evaluate_path(cm, rem, self.cs, self.risk_weight).total_cost
        pct = (new_cost / old_cost - 1.0) if old_cost > 0 else 0.0
        costs = {"old_cost": old_cost, "new_cost": new_cost, "pct": pct}
        cost_now, cost_then = cm.cost[r, c], self.plan_cm.cost[r, c]
        fin = np.isfinite(cost_now)
        if not fin.all():
            i = int(np.argmin(fin))
            cell = (int(r[i]), int(c[i]))
            if float(self.belief.class_probs[cell[0], cell[1], BIG_ROCK]) > cfg.big_rock_impassable_p:
                name, thr = "big_rock_impassable_p", cfg.big_rock_impassable_p
            else:
                name, thr = "slope_max_deg", cfg.slope_max_deg
            return {"kind": "impassable", "cell": cell, "threshold": thr, "threshold_name": name, **costs}, None
        hp = cm.hazard_prob[r, c]
        i = int(np.argmax(hp))
        h_cell = (int(r[i]), int(c[i]))
        if hp[i] > cfg.hazard_prob_max:
            return {"kind": "hazard", "cell": h_cell, "threshold": cfg.hazard_prob_max,
                    "threshold_name": "hazard_prob_max", **costs}, None
        delta = cost_now - cost_then
        j = int(np.argmax(delta))
        c_cell = (int(r[j]), int(c[j]))
        if pct > cfg.replan_cost_increase:
            return {"kind": "cost", "cell": c_cell, "threshold": cfg.replan_cost_increase,
                    "threshold_name": "replan_cost_increase", **costs}, None
        h_ratio = float(hp[i]) / cfg.hazard_prob_max
        c_ratio = pct / cfg.replan_cost_increase
        if max(h_ratio, c_ratio) < cfg.loop_near_frac:
            return None, None
        if h_ratio >= c_ratio:
            near = {"cell": h_cell, "threshold": cfg.hazard_prob_max, "threshold_name": "hazard_prob_max",
                    "check_value": float(hp[i]), "check_ratio": h_ratio}
        else:
            near = {"cell": c_cell, "threshold": cfg.replan_cost_increase, "threshold_name": "replan_cost_increase",
                    "check_value": pct, "check_ratio": c_ratio}
        return None, {**near, **costs}

    def step(self) -> bool:
        alive = super().step()
        self.store.update_rover(self.rc, self.path[self.path_idx:], self.reach)  # lookahead prefetch
        if self.telemetry:
            self.tier_frames.append(self.store.tier_map())
        return alive

    def result(self, runtime_s: float = 0.0) -> RunResult:
        """Materialise a float CostMap for the dashboard / evaluation (after driving, not used onboard)."""
        res = super().result(runtime_s)
        res.costmap = self.ccm.materialize()
        res.meta.update(onboard=True, store=self.store.stats(), compact_cost_bytes=self.ccm.nbytes)
        return res


if __name__ == "__main__":
    import time

    from autonomy.loop import Rover as FullRover
    from onboard.tiered_store import build_store
    from terrain.features import compute_layers
    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    L = compute_layers(w)
    prior = init_prior(w, DEFAULT, DEFAULT.seed)
    st = build_store(w, L, prefetch_sync=True, usage_path=None)
    t0 = time.perf_counter()
    rv = OnboardRover(w, st, DEFAULT, "balanced", seed=DEFAULT.seed, belief=prior)
    hz = rv.inject_hazard_on_path(0.4, 3)
    res = rv.run_to_goal()
    dt = time.perf_counter() - t0
    w2 = make_world(200, 200, seed=DEFAULT.seed)
    fr = FullRover(w2, DEFAULT, "balanced", seed=DEFAULT.seed, layers=L, belief=prior)
    fr.inject_hazard(hz, 3)
    fres = fr.run_to_goal()
    s = res.meta["store"]
    print(f"ONBOARD: goal {res.reached_goal} {res.distance_m:.0f} m replans {res.replans} rocks "
          f"{res.meta['entered_true_hazard_cells']} | hit rate {s['hit_rate']:.2%} stalls {s['stalls']} prefetch "
          f"{s['prefetch_hits']}/{s['prefetch_loaded']} evictions {s['evictions_hot']} ({dt:.1f}s)")
    print(f"FULL   : goal {fres.reached_goal} {fres.distance_m:.0f} m replans {fres.replans} rocks "
          f"{fres.meta['entered_true_hazard_cells']}")
    st.close()
