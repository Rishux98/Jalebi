"""Autonomy loop: move -> sense -> Bayes update -> local cost rebuild -> check path -> replan, all logged.

The rover never reads the hidden ``world.true_class`` directly: it only sees it through the noisy
sensor model (``observe_window``). ``inject_hazard*`` writes into the world truth, so the rover only
learns about a planted boulder field when its navcam footprint reaches it.

Per step:
  1. move one cell along the current path (energy += dist * energy_wh_per_m[cell]);
  2. sense the disc of ``sensor_radius`` cells -> Bayes update of that window only;
  3. rebuild the cost map on that window only (``build_costmap`` on slices; per-cell ops);
  4. log ``hazard_detected`` for cells that just became impassable;
  5. check the remaining path, in this order:
       a cell became impassable (cost inf)          -> replan (threshold big_rock_impassable_p / slope_max_deg)
       max hazard_prob on path > hazard_prob_max    -> replan
       remaining-path cost rose > replan_cost_increase vs the map at plan time -> replan
     otherwise, if a check reached ``loop_near_frac`` of its threshold, log ``keep_path``
     (once per trigger cell per plan).
Planning treats ``hazard_prob > hazard_prob_max`` as a hard limit (cells set to inf), so a fresh
plan never re-triggers the hazard rule on the same cells.

Run: python -m autonomy.loop            (injected boulder field on the planned path)
     python -m autonomy.loop --science  (route through budget-selected science targets)
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from belief.bayes import init_prior, update
from core.config import DEFAULT, Config
from core.types import BIG_ROCK, RC, Belief, CostMap, Plan, Target, TerrainLayers, World
from costmap.fuse import build_costmap, costmap_from_belief
from explain.log import DecisionLog, cell_evidence, confidence_at, confidence_mean, make_evidence
from perception.sensor_model import load_confusion, observe_window
from planner.astar import astar, evaluate_path
from planner.pareto import mode_config, plan_via
from terrain.features import compute_layers

SQRT2 = math.sqrt(2.0)


@dataclass
class RunResult:
    """Outcome of ``Rover.run_to_goal``.

    Attributes:
        trajectory: cells actually visited, start first.
        log: every decision with evidence.
        plans: every adopted plan (initial + replans); ``plan.meta["step"]`` = step adopted.
        reached_goal: True if the rover stopped on the goal.
        steps: number of moves.
        energy_wh / distance_m: totals driven (Wh, m).
        replans: number of replans.
        frames: per-step dicts for the dashboard slider (step, rover_rc, path, replanned, energy_wh, ...).
        entropy_snapshots: step -> (H, W) float32 belief entropy (every ``loop_snapshot_every`` steps).
        belief / costmap: final belief and cost map.
        meta: injected hazards (ground truth, not rover knowledge), true boulder cells entered, runtime.
    """
    trajectory: list[RC]
    log: DecisionLog
    plans: list[Plan]
    reached_goal: bool
    steps: int
    energy_wh: float
    distance_m: float
    replans: int
    frames: list[dict] = field(default_factory=list)
    entropy_snapshots: dict[int, np.ndarray] = field(default_factory=dict)
    belief: Belief | None = None
    costmap: CostMap | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"reached_goal": self.reached_goal, "steps": self.steps, "distance_m": round(self.distance_m, 1),
                "energy_wh": round(self.energy_wh, 1), "replans": self.replans,
                "decisions": self.log.counts(), **{k: v for k, v in self.meta.items() if k != "injected"}}


def _cheb(a: RC, b: RC) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class Rover:
    """Simulated rover with its own belief, cost map, plan and decision log.

    Args:
        world: site (mutated only by ``inject_hazard*``).
        cfg: global configuration; ``mode`` picks cost weights / risk weight via ``planner.pareto.mode_config``.
        mode: "shortest" | "balanced" | "safest" | "science".
        seed: sensor-noise seed (and prior seed if ``belief`` is None).
        confusion: (4, 4) sensor model; default ``perception.sensor_model.load_confusion()``.
        targets: science targets to visit in order (``Target`` or plain (r, c) cells).
        layers / belief: precomputed terrain layers / prior belief (belief is copied, never mutated).
    """

    def __init__(self, world: World, cfg: Config = DEFAULT, mode: str = "balanced", seed: int | None = None,
                 confusion: np.ndarray | None = None, targets: Sequence[Target | RC] | None = None,
                 layers: TerrainLayers | None = None, belief: Belief | None = None,
                 budget_wh: float | None = None) -> None:
        self.world, self.cfg, self.mode = world, cfg, mode
        # energy budget (Wh) enforced on every (re)plan while targets remain; default only with targets
        self.budget_wh = budget_wh if budget_wh is not None else (cfg.sci_energy_budget_wh if targets else None)
        self.seed = cfg.seed if seed is None else int(seed)
        self.mcfg, self.risk_weight = mode_config(mode, cfg)
        self.rng = np.random.default_rng(self.seed)
        self.confusion = load_confusion()[0] if confusion is None else np.asarray(confusion, np.float32)
        self.layers = layers if layers is not None else compute_layers(world, cfg)
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
        self._keep_logged: set[tuple] = set()
        self.path: list[RC] = []
        self.path_idx = 0
        self.plan: Plan | None = None
        self.plan_cm: CostMap | None = None

        self._sense()                                                   # first look before planning
        self.costmap = costmap_from_belief(self.layers, self.belief, self.mcfg)
        self.entropy_snapshots[0] = self.belief.entropy.copy()
        self._initial_plan()
        self._frame(replanned=False)

    # ------------------------------------------------------------------ properties
    @property
    def waypoints(self) -> list[RC]:
        return [t.rc for t in self.targets]

    # ------------------------------------------------------------------ sensing + local map
    def _window(self, rc: RC, radius: int) -> tuple[tuple[slice, slice], RC]:
        h, w = self.world.shape
        r0, r1 = max(0, rc[0] - radius), min(h, rc[0] + radius + 1)
        c0, c1 = max(0, rc[1] - radius), min(w, rc[1] + radius + 1)
        return (slice(r0, r1), slice(c0, c1)), (rc[0] - r0, rc[1] - c0)

    def _sense(self) -> tuple[slice, slice]:
        """Navcam look from ``self.rc``: Bayes update of the sensor window only. Returns the window."""
        rad = int(self.cfg.sensor_radius)
        sl, local = self._window(self.rc, rad)
        obs, mask = observe_window(self.world.true_class[sl], local, rad, self.rng, self.confusion)
        sub = Belief(self.belief.class_probs[sl], self.belief.entropy[sl], self.belief.observed_mask[sl])
        post = update(sub, obs, mask, self.confusion)
        self.belief.class_probs[sl] = post.class_probs
        self.belief.entropy[sl] = post.entropy
        self.belief.observed_mask[sl] = post.observed_mask
        return sl

    def _rebuild(self, sl: tuple[slice, slice]) -> np.ndarray:
        """Rebuild the cost map on window ``sl``; return (window) bool mask of cells that just became impassable."""
        cm, L = self.costmap, self.layers
        was = np.isfinite(cm.cost[sl])
        sub_layers = TerrainLayers(L.slope_deg[sl], L.roughness[sl], L.curvature[sl], L.p_slope_exceed[sl])
        sub = build_costmap(sub_layers, self.belief.class_probs[sl], self.mcfg, entropy=self.belief.entropy[sl])
        cm.cost[sl] = sub.cost
        cm.hazard_prob[sl] = sub.hazard_prob
        for k, v in sub.components.items():
            cm.components[k][sl] = v
        if cm.energy_wh_per_m is not None and sub.energy_wh_per_m is not None:
            cm.energy_wh_per_m[sl] = sub.energy_wh_per_m
        return was & ~np.isfinite(sub.cost)

    # ------------------------------------------------------------------ planning
    def _planning_map(self) -> CostMap:
        """Cost map for planning: hazard_prob > hazard_prob_max is a hard limit; own cell kept passable."""
        cost = self.costmap.cost.copy()
        cost[self.costmap.hazard_prob > self.cfg.hazard_prob_max] = np.inf
        r, c = self.rc
        if not np.isfinite(cost[r, c]):
            fin = cost[np.isfinite(cost)]
            cost[r, c] = float(fin.max()) if fin.size else 0.0
        return CostMap(cost, self.costmap.hazard_prob, energy_wh_per_m=self.costmap.energy_wh_per_m)

    def _route(self, pm: CostMap) -> Plan:
        if self.targets:
            return plan_via(pm, self.rc, self.waypoints, self.goal, self.cs, self.risk_weight, self.mode)
        return astar(pm, self.rc, self.goal, self.cs, self.risk_weight, self.mode)

    def _find_route(self) -> Plan | None:
        """Plan to the goal through remaining targets; drop (and log) targets that became unreachable."""
        pm = self._planning_map()
        while True:
            p = self._route(pm)
            if p.found:
                return p
            if self.targets and p.meta.get("reason") == "waypoint_unreachable":
                a, b = (tuple(x) for x in p.meta["failed_leg"])
                bad = b if b != self.goal else a
                hit = [t for t in self.targets if t.rc == bad]
                if hit:
                    self._log_skip_target(hit[0])
                    self.targets.remove(hit[0])
                    continue
            return None

    def _fit_budget(self, p: Plan | None) -> Plan | None:
        """While used + planned energy > budget and targets remain, drop the lowest-value target (logged).

        The goal itself is never dropped: with no targets left the route is kept even if over budget.
        """
        while (p is not None and self.budget_wh is not None and self.targets
               and self.energy_wh + p.energy_wh > self.budget_wh):
            t = min(self.targets, key=lambda x: x.value_mean)
            ev = make_evidence(t.rc, self.rc, self.budget_wh, "budget_wh", target_id=t.id, value_mean=t.value_mean,
                               value_std=t.value_std, utility=t.utility, energy_wh=self.energy_wh + p.energy_wh,
                               energy_used_wh=self.energy_wh, route_energy_wh=p.energy_wh, budget_wh=self.budget_wh,
                               n_waypoints=len(self.targets))
            self.log.record(self.step_idx, "budget_exceeded", "skip_target", ev, confidence_at(self.belief, t.rc))
            self.targets.remove(t)
            p = self._find_route()
        return p

    def _adopt(self, p: Plan) -> None:
        p.meta["step"] = self.step_idx
        self.plan, self.path, self.path_idx = p, list(p.path), 0
        self.plan_cm = CostMap(self.costmap.cost.copy(), self.costmap.hazard_prob.copy())
        self.plans.append(p)
        self._keep_logged.clear()

    def _initial_plan(self) -> None:
        p = self._fit_budget(self._find_route())
        if p is None:
            self._stop_no_route()
            return
        self._adopt(p)
        ent = self.belief.entropy[p.as_arrays()]
        ev = make_evidence(self.goal, self.rc, self.risk_weight, "risk_weight", mode=self.mode,
                           length_m=p.length_m, energy_wh=p.energy_wh, success_prob=p.success_prob,
                           total_cost=p.total_cost, expanded_nodes=p.expanded_nodes,
                           n_waypoints=len(self.targets), path_entropy_mean=float(ent.mean()),
                           confidence_basis="path_mean")
        self.log.record(self.step_idx, "mission_start", "plan", ev, confidence_mean(self.belief, p.path))

    # ------------------------------------------------------------------ checks
    def _check(self) -> tuple[dict | None, dict | None]:
        """(trigger, near) for the remaining path; each a dict with the numbers, or None."""
        cfg = self.cfg
        rem = self.path[self.path_idx:]
        if len(rem) < 2:
            return None, None
        ahead = np.asarray(rem[1:], dtype=np.int64)
        r, c = ahead[:, 0], ahead[:, 1]
        cm = self.costmap
        old_cost = evaluate_path(self.plan_cm, rem, self.cs, self.risk_weight).total_cost
        new_cost = evaluate_path(cm, rem, self.cs, self.risk_weight).total_cost
        pct = (new_cost / old_cost - 1.0) if old_cost > 0 else 0.0
        costs = {"old_cost": old_cost, "new_cost": new_cost, "pct": pct}

        fin = np.isfinite(cm.cost[r, c])
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

        delta = cm.cost[r, c] - self.plan_cm.cost[r, c]
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

    # ------------------------------------------------------------------ decisions
    def _cell_ev(self, rc: RC) -> dict:
        return cell_evidence(self.belief, self.layers, self.costmap, rc)

    def _replan(self, trig: dict) -> bool:
        rem = self.path[self.path_idx:]
        old = evaluate_path(self.costmap, rem, self.cs, self.risk_weight)
        new = self._fit_budget(self._find_route())
        event = {"impassable": "impassable_on_path", "hazard": "hazard_on_path", "cost": "cost_increase"}[trig["kind"]]
        nums = {**self._cell_ev(trig["cell"]), "trigger": trig["kind"], "old_cost": trig["old_cost"],
                "new_cost": trig["new_cost"], "cost_increase_pct": trig["pct"],
                "old_success_prob": old.success_prob, "old_length_m": old.length_m}
        if new is None:
            self._stop_no_route(extra=nums)
            return False
        nums.update(new_success_prob=new.success_prob, new_length_m=new.length_m,
                    new_route_cost=new.total_cost, expanded_nodes=new.expanded_nodes,
                    n_waypoints=len(self.targets))
        ev = make_evidence(trig["cell"], self.rc, trig["threshold"], trig["threshold_name"], **nums)
        self.log.record(self.step_idx, event, "replan", ev, confidence_at(self.belief, trig["cell"]))
        self._adopt(new)
        self.replans += 1
        return True

    def _log_keep(self, near: dict) -> None:
        key = (near["cell"], near["threshold_name"])
        if key in self._keep_logged:
            return
        self._keep_logged.add(key)
        ev = make_evidence(near["cell"], self.rc, near["threshold"], near["threshold_name"],
                           check_value=near["check_value"], check_ratio=near["check_ratio"],
                           old_cost=near["old_cost"], new_cost=near["new_cost"], cost_increase_pct=near["pct"],
                           **self._cell_ev(near["cell"]))
        self.log.record(self.step_idx, "near_threshold", "keep_path", ev, confidence_at(self.belief, near["cell"]))

    def _log_hazard(self, sl: tuple[slice, slice], newly: np.ndarray) -> None:
        rr, cc = np.nonzero(newly)
        rows, cols = rr + sl[0].start, cc + sl[1].start
        k = int(np.argmax(self.belief.class_probs[rows, cols, BIG_ROCK]))
        cell = (int(rows[k]), int(cols[k]))
        ahead = set(self.path[self.path_idx:])
        n_on = sum((int(a), int(b)) in ahead for a, b in zip(rows, cols))
        ev = make_evidence(cell, self.rc, self.cfg.big_rock_impassable_p, "big_rock_impassable_p",
                           n_new_impassable=int(rows.size), n_on_path=int(n_on), **self._cell_ev(cell))
        self.log.record(self.step_idx, "belief_update", "hazard_detected", ev, confidence_at(self.belief, cell))

    def _log_skip_target(self, t: Target) -> None:
        ev = make_evidence(t.rc, self.rc, 1.0, "reachable", target_id=t.id, value_mean=t.value_mean,
                           value_std=t.value_std, utility=float("-inf"), energy_wh=float("inf"),
                           **self._cell_ev(t.rc))
        self.log.record(self.step_idx, "target_unreachable", "skip_target", ev, confidence_at(self.belief, t.rc))

    def _stop_no_route(self, extra: dict | None = None) -> None:
        ev = make_evidence(self.goal, self.rc, 1.0, "reachable",
                           n_impassable=int((~np.isfinite(self.costmap.cost)).sum()) if hasattr(self, "costmap") else 0,
                           **(extra or {}))
        self.log.record(self.step_idx, "no_path", "stop", ev, confidence_at(self.belief, self.goal))
        self.done = True

    def _finish(self) -> None:
        ev = make_evidence(self.goal, self.rc, 0, "goal_tolerance_cells", steps=self.step_idx,
                           distance_m=self.distance_m, energy_wh=self.energy_wh, replans=self.replans,
                           n_observed=int(self.belief.observed_mask.sum()), confidence_basis="trajectory_mean")
        self.log.record(self.step_idx, "goal_reached", "goal_reached", ev,
                        confidence_mean(self.belief, self.trajectory))
        self.done = self.reached_goal = True

    # ------------------------------------------------------------------ stepping
    def _move(self, nxt: RC) -> None:
        dr, dc = abs(nxt[0] - self.rc[0]), abs(nxt[1] - self.rc[1])
        d = self.cs * (SQRT2 if dr and dc else 1.0)
        if self.costmap.energy_wh_per_m is not None:
            self.energy_wh += d * float(self.costmap.energy_wh_per_m[nxt])
        self.distance_m += d
        self.rc = nxt
        self.path_idx += 1
        self.step_idx += 1
        self.trajectory.append(nxt)
        if int(self.world.true_class[nxt]) == BIG_ROCK:     # bookkeeping only (ground truth), never used to decide
            self.entered_true_hazard_cells += 1

    def _frame(self, replanned: bool) -> None:
        self.frames.append({"step": self.step_idx, "rover_rc": self.rc, "path": self.path[self.path_idx:],
                            "replanned": replanned, "energy_wh": self.energy_wh, "distance_m": self.distance_m,
                            "n_observed": int(self.belief.observed_mask.sum()),
                            "n_targets_left": len(self.targets), "n_log": len(self.log)})
        if self.step_idx % max(int(self.cfg.loop_snapshot_every), 1) == 0:
            self.entropy_snapshots[self.step_idx] = self.belief.entropy.copy()

    def step(self) -> bool:
        """Advance one cell and run the sense/check/replan cycle. Returns False once finished."""
        if self.done:
            return False
        if self.path_idx + 1 >= len(self.path):            # path exhausted but not at goal (should not happen)
            self._stop_no_route()
            return False
        self._move(self.path[self.path_idx + 1])

        if self.targets and self.rc == self.targets[0].rc:
            t = self.targets.pop(0)
            ev = make_evidence(t.rc, self.rc, 0, "goal_tolerance_cells", target_id=t.id, step_reached=self.step_idx,
                               n_waypoints_left=len(self.targets), energy_wh=self.energy_wh,
                               value_mean=t.value_mean, value_std=t.value_std)
            self.log.record(self.step_idx, "target_reached", "continue", ev, confidence_at(self.belief, t.rc))
        if self.rc == self.goal:
            self._finish()
            self._frame(replanned=False)
            return False

        sl = self._sense()
        newly = self._rebuild(sl)
        if newly.any():
            self._log_hazard(sl, newly)
        trig, near = self._check()
        replanned = False
        if trig is not None:
            replanned = self._replan(trig)
        elif near is not None:
            self._log_keep(near)
        self._frame(replanned)
        return not self.done

    # ------------------------------------------------------------------ hazards (ground truth)
    def inject_hazard(self, rc: RC, radius: float = 3, kind: str = "big_rock") -> RC:
        """Plant a hidden hazard into the world truth (the rover is NOT told). Recorded in RunResult.meta."""
        mask = self.world.inject_hazard(rc, radius, kind)
        self.injected.append({"rc": [int(rc[0]), int(rc[1])], "radius": float(radius), "kind": kind,
                              "step": self.step_idx, "n_cells": int(mask.sum())})
        return (int(rc[0]), int(rc[1]))

    def inject_hazard_on_path(self, frac: float = 0.4, radius: float = 3, kind: str = "big_rock") -> RC:
        """Plant a hazard on the current path, ~``frac`` of the way along and outside sensor range.

        Avoids the goal, remaining targets and the rover's current sensor footprint.

        Raises:
            ValueError: the remaining path is too short to place it out of sensor range.
        """
        rem = self.path[self.path_idx:]
        keep_clear = [self.goal, *self.waypoints]
        start_i = max(int(frac * (len(rem) - 1)), 1)
        for rc in rem[start_i:]:
            if (_cheb(rc, self.rc) > self.cfg.sensor_radius + radius + 1
                    and all(_cheb(rc, k) > radius + 1 for k in keep_clear)):
                return self.inject_hazard(rc, radius, kind)
        raise ValueError("remaining path too short to inject a hazard outside sensor range")

    # ------------------------------------------------------------------ run
    def result(self, runtime_s: float = 0.0) -> RunResult:
        self.entropy_snapshots[self.step_idx] = self.belief.entropy.copy()
        return RunResult(
            trajectory=list(self.trajectory), log=self.log, plans=list(self.plans), reached_goal=self.reached_goal,
            steps=self.step_idx, energy_wh=self.energy_wh, distance_m=self.distance_m, replans=self.replans,
            frames=self.frames, entropy_snapshots=self.entropy_snapshots, belief=self.belief, costmap=self.costmap,
            meta={"mode": self.mode, "seed": self.seed, "injected": list(self.injected),
                  "entered_true_hazard_cells": self.entered_true_hazard_cells, "runtime_s": round(runtime_s, 3)})

    def run_to_goal(self, max_steps: int | None = None) -> RunResult:
        """Step until the goal is reached, no route exists, or ``max_steps`` (default ``loop_max_steps``)."""
        limit = self.cfg.loop_max_steps if max_steps is None else int(max_steps)
        t0 = time.perf_counter()
        while not self.done:
            if self.step_idx >= limit:
                ev = make_evidence(self.goal, self.rc, limit, "loop_max_steps", steps=self.step_idx)
                self.log.record(self.step_idx, "step_limit", "stop", ev, confidence_at(self.belief, self.goal))
                self.done = True
                break
            self.step()
        return self.result(time.perf_counter() - t0)


def plot_run(world: World, res: RunResult, path: str = "outputs/traverse.png", targets: Sequence[Target] = ()) -> str:
    """Two panels: hillshade + truth boulders + initial plan + driven track + replans; final belief entropy."""
    from pathlib import Path

    from core.io import hillshade
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle

    shade = hillshade(world.dem, world.cell_size)
    rock = np.ma.masked_where(world.true_class != BIG_ROCK, np.ones(world.shape))
    tr = np.asarray(res.trajectory)
    replans = res.log.by_decision("replan")

    def draw(a, ms: float) -> None:
        a.imshow(shade, cmap="gray")
        a.imshow(rock, cmap=ListedColormap(["#e0302a"]), alpha=0.6, vmin=0, vmax=1)
        for inj in res.meta.get("injected", []):
            a.add_patch(Circle((inj["rc"][1], inj["rc"][0]), inj["radius"] + 0.5, fill=False, ec="red", lw=2, ls="--"))
        if res.plans:
            r, c = res.plans[0].as_arrays()
            a.plot(c, r, "--", color="white", lw=1.4, label="initial plan")
        a.plot(tr[:, 1], tr[:, 0], "-", color="cyan", lw=2.2, label=f"driven ({res.distance_m:.0f} m)")
        for e in res.log.by_decision("hazard_detected"):
            a.plot(e.evidence["cell"][1], e.evidence["cell"][0], "s", ms=ms * 0.45, mfc="orange", mec="none")
        for e in replans:
            rr, cc = e.evidence["rover_rc"]
            a.plot(cc, rr, "X", ms=ms, mfc="yellow", mec="k")
        for t in targets:
            a.plot(t.rc[1], t.rc[0], "D", ms=ms * 0.8, mfc="#4dc3ff", mec="k")
        a.plot(world.start[1], world.start[0], "o", ms=11, mfc="lime", mec="k")
        a.plot(world.goal[1], world.goal[0], "*", ms=15, mfc="magenta", mec="k")

    fig, ax = plt.subplots(1, 3, figsize=(22, 7.6))
    draw(ax[0], 10)
    ax[0].plot([], [], "X", ms=10, mfc="yellow", mec="k", ls="none", label=f"replan ({res.replans})")
    ax[0].plot([], [], "s", ms=6, mfc="orange", mec="none", ls="none", label="hazard detected")
    ax[0].legend(loc="lower left", fontsize=9)
    ax[0].set_title("Traverse vs ground truth (red = boulders, dashed = injected)")

    draw(ax[1], 14)
    if replans:                                         # zoom on the first replan
        r0, c0 = replans[0].evidence["rover_rc"]
        half = 22
        h, w = world.shape
        r0 = min(max(r0, half), h - 1 - half)           # keep the zoom box inside the map
        c0 = min(max(c0, half), w - 1 - half)
        ax[1].set_xlim(c0 - half - 0.5, c0 + half + 0.5)
        ax[1].set_ylim(r0 + half + 0.5, r0 - half - 0.5)
        for e in replans:
            rr, cc = e.evidence["rover_rc"]
            ax[1].annotate(f"step {e.step}", (cc, rr), color="yellow", fontsize=9, xytext=(6, -12),
                           textcoords="offset points")
        ax[1].set_title(f"Zoom: first replan at step {replans[0].step}, trigger cell "
                        f"{tuple(replans[0].evidence['cell'])}")
    else:
        ax[1].set_title("no replans")

    im = ax[2].imshow(res.belief.entropy, cmap="magma", vmin=0, vmax=np.log(4))
    ax[2].plot(tr[:, 1], tr[:, 0], "-", color="cyan", lw=1.2)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label="belief entropy after traverse (nats)")
    ax[2].set_title(f"Belief entropy after traverse ({int(res.belief.observed_mask.sum())} cells observed)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse

    from science.targets import detect_candidates, select_under_budget
    from world.synthetic import make_world

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT.seed)
    ap.add_argument("--mode", default="balanced", choices=["shortest", "balanced", "safest", "science"])
    ap.add_argument("--science", action="store_true", help="visit budget-selected science targets")
    ap.add_argument("--frac", type=float, default=0.4, help="where along the initial path to inject")
    ap.add_argument("--radius", type=float, default=3)
    ap.add_argument("--no-inject", action="store_true")
    a = ap.parse_args()

    cfg = DEFAULT
    w = make_world(a.size, a.size, seed=a.seed)
    L = compute_layers(w, cfg)
    C, src = load_confusion()
    targets: list[Target] = []
    if a.science:
        from belief.bayes import sense
        b0 = sense(init_prior(w, cfg, a.seed), w, w.start, np.random.default_rng(a.seed), cfg, C)
        cm0 = costmap_from_belief(L, b0, cfg)
        sel = select_under_budget(detect_candidates(w, L, b0, cm0, cfg, a.seed), cm0, b0, w.start, w.goal,
                                  cfg=cfg, cell_size=w.cell_size)
        targets = sel.selected
        print(f"science: {len(targets)} targets selected under {sel.budget_wh:.0f} Wh: {[t.rc for t in targets]}")

    rover = Rover(w, cfg, mode="science" if a.science else a.mode, seed=a.seed, confusion=C, targets=targets, layers=L)
    p0 = rover.plan
    print(f"world {a.size}x{a.size} seed={w.seed}, confusion={src}, mode={rover.mode}")
    print(f"initial plan: {p0.length_m:.0f} m, P(success)={p0.success_prob:.1%}")
    hz = None
    if not a.no_inject:
        hz = rover.inject_hazard_on_path(a.frac, a.radius, "big_rock")
        print(f"injected hidden boulder field r={a.radius:g} at {hz} (on the initial path; rover not told)")

    res = rover.run_to_goal()
    s = res.summary()
    print(f"\nresult: reached_goal={s['reached_goal']} steps={s['steps']} distance={s['distance_m']} m "
          f"energy={s['energy_wh']} Wh replans={s['replans']} runtime={s['runtime_s']} s "
          f"({1000 * s['runtime_s'] / max(s['steps'], 1):.1f} ms/step)")
    print(f"decisions: {s['decisions']}; true boulder cells entered: {s['entered_true_hazard_cells']}")
    print("\ndecision log (keep_path / hazard_detected abbreviated):")
    shown = {"keep_path": 0, "hazard_detected": 0}
    for e in res.log:
        if e.decision in shown:
            shown[e.decision] += 1
            if shown[e.decision] > 3:
                continue
        print(f"  s{e.step:<4d} [{e.decision:15s}] conf {e.confidence:.2f}  {e.reason}")
    for k, n in shown.items():
        if n > 3:
            print(f"  ... {n - 3} more {k} entries")

    res.log.to_json("outputs/decision_log.json")
    print("\nsaved outputs/decision_log.json,", plot_run(w, res, targets=targets))

    if hz is not None:
        replans = res.log.by_decision("replan")
        assert replans, "injected hazard on the path did not trigger a replan"
        assert any(e.evidence["hazard_prob"] > e.evidence["threshold"] or not math.isfinite(e.evidence["new_cost"])
                   or e.evidence["cost_increase_pct"] > e.evidence["threshold"] for e in replans)
    assert res.reached_goal, "rover did not reach the goal"
    print("asserts passed: replan triggered by the hidden hazard, goal reached")
