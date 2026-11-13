"""Multi-objective route options: shortest / balanced / safest / science, plus a risk-weight Pareto sweep.

Each mode is A* on a cost map built from the rover's *belief* with mode-specific weights:
    shortest  all terrain weights 0 (pure distance, impassable cells still inf), risk_weight 0
    balanced  default Config weights, risk_weight = pareto_risk_balanced
    safest    lambda_unc = pareto_lambda_unc_safest (avoid unknown terrain), risk_weight = pareto_risk_safest
    science   balanced weights, routed through science waypoints (``plan_via``; targets come in P4)

All plans are re-scored on the *same* belief cost map, so length / energy / success_prob are
comparable across modes. If a ground-truth cost map is given, each plan also gets
``meta["success_prob_true"]`` — what the belief predicted vs what reality would give.

Sweep: risk_weight over linspace(*pareto_sweep) on the balanced map -> (length_m, success_prob)
points; ``pareto_front`` keeps the non-dominated ones (shorter or safer than every other).

Run: python -m planner.pareto   (prints the mode table, writes outputs/pareto.png)
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from core.config import DEFAULT, Config
from core.types import RC, Belief, CostMap, Plan, TerrainLayers
from costmap.fuse import costmap_from_belief
from planner.astar import astar, evaluate_path

MODES: tuple[str, ...] = ("shortest", "balanced", "safest", "science")


@dataclass
class ParetoResult:
    """Output of ``run_pareto``.

    Attributes:
        modes: mode name -> Plan (science only if waypoints were given).
        sweep: one Plan per risk_weight in the sweep (``meta["risk_weight"]``).
        front: non-dominated subset of ``sweep`` + ``modes``, sorted by length.
        reference: belief cost map (balanced weights) all plans are scored on.
    """
    modes: dict[str, Plan]
    sweep: list[Plan] = field(default_factory=list)
    front: list[Plan] = field(default_factory=list)
    reference: CostMap | None = None

    def table(self) -> list[dict]:
        """One JSON-friendly row per mode (for printing / the dashboard)."""
        rows = []
        for name, p in self.modes.items():
            rows.append({"mode": name, "found": p.found, "length_m": round(p.length_m, 1),
                         "energy_wh": round(p.energy_wh, 1), "success_prob": round(p.success_prob, 4),
                         "success_prob_true": p.meta.get("success_prob_true"),
                         "path_entropy_mean": p.meta.get("path_entropy_mean"),
                         "expanded_nodes": p.expanded_nodes,
                         "risk_weight": p.meta.get("risk_weight"),
                         "runtime_s": round(p.meta.get("runtime_s", 0.0), 3)})
        return rows


def mode_config(mode: str, cfg: Config = DEFAULT) -> tuple[Config, float]:
    """(Config used to build the cost map, A* risk_weight) for a mode."""
    if mode == "shortest":
        return cfg.replace(w_slope=0.0, w_rough=0.0, w_class=0.0, w_energy=0.0, lambda_unc=0.0), 0.0
    if mode in ("balanced", "science"):
        return cfg, cfg.pareto_risk_balanced
    if mode == "safest":
        return cfg.replace(lambda_unc=cfg.pareto_lambda_unc_safest), cfg.pareto_risk_safest
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def _rescore(plan: Plan, reference: CostMap, truth: CostMap | None, cell_size: float,
             entropy: np.ndarray | None = None) -> Plan:
    """Re-score a plan's stats on the shared reference map (keeps the mode's own total_cost).

    Adds ``meta["path_entropy_mean"]`` (nats, exposure to unknown terrain) if ``entropy`` is given,
    and ``meta["success_prob_true"]`` / ``meta["truth_impassable_cells"]`` if ``truth`` is given.
    """
    if not plan.found:
        return plan
    ref = evaluate_path(reference, plan.path, cell_size)
    plan.energy_wh, plan.success_prob = ref.energy_wh, ref.success_prob
    if entropy is not None:
        rows, cols = plan.as_arrays()
        plan.meta["path_entropy_mean"] = round(float(np.asarray(entropy)[rows, cols].mean()), 4)
    if truth is not None:
        t = evaluate_path(truth, plan.path, cell_size)
        rows, cols = plan.as_arrays()
        plan.meta["success_prob_true"] = round(t.success_prob, 4)
        plan.meta["truth_impassable_cells"] = int((~np.isfinite(truth.cost[rows, cols])).sum())
    return plan


def plan_via(costmap: CostMap, start: RC, waypoints: Sequence[RC], goal: RC, cell_size: float = 1.0,
             risk_weight: float = 0.0, mode: str = "science") -> Plan:
    """Chain A* segments start -> waypoints... -> goal into one Plan.

    Returns ``Plan(path=[])`` with ``meta["reason"] = "waypoint_unreachable"`` and
    ``meta["failed_leg"]`` if any leg fails. ``expanded_nodes`` sums all legs.
    """
    t0 = time.perf_counter()
    stops = [tuple(start), *[tuple(w) for w in waypoints], tuple(goal)]
    path: list[RC] = [stops[0]]
    expanded = 0
    for i, (a, b) in enumerate(zip(stops, stops[1:])):
        leg = astar(costmap, a, b, cell_size, risk_weight, mode)
        expanded += leg.expanded_nodes
        if not leg.found:
            return Plan(path=[], success_prob=0.0, mode=mode, expanded_nodes=expanded,
                        meta={"reason": "waypoint_unreachable", "failed_leg": [list(a), list(b)],
                              "risk_weight": float(risk_weight), "waypoints": [list(w) for w in waypoints],
                              "runtime_s": time.perf_counter() - t0})
        path.extend(leg.path[1:])
    meta = {"risk_weight": float(risk_weight), "waypoints": [list(w) for w in waypoints]}
    plan = evaluate_path(costmap, path, cell_size, risk_weight, mode, expanded, meta)
    plan.meta["runtime_s"] = time.perf_counter() - t0
    return plan


def plan_modes(layers: TerrainLayers, belief: Belief, start: RC, goal: RC, cfg: Config = DEFAULT,
               cell_size: float = 1.0, science_waypoints: Sequence[RC] | None = None,
               truth: CostMap | None = None, reference: CostMap | None = None) -> dict[str, Plan]:
    """One Plan per mode (science only if ``science_waypoints`` is given), scored on ``reference``."""
    ref = reference if reference is not None else costmap_from_belief(layers, belief, cfg)
    out: dict[str, Plan] = {}
    for mode in MODES:
        if mode == "science" and not science_waypoints:
            continue
        mcfg, rw = mode_config(mode, cfg)
        cm = ref if mcfg is cfg else costmap_from_belief(layers, belief, mcfg)
        p = (plan_via(cm, start, science_waypoints, goal, cell_size, rw, mode) if mode == "science"
             else astar(cm, start, goal, cell_size, rw, mode))
        out[mode] = _rescore(p, ref, truth, cell_size, belief.entropy)
    return out


def sweep_risk(reference: CostMap, start: RC, goal: RC, cfg: Config = DEFAULT, cell_size: float = 1.0,
               truth: CostMap | None = None, entropy: np.ndarray | None = None) -> list[Plan]:
    """A* on the balanced belief map for each risk_weight in ``linspace(*cfg.pareto_sweep)``."""
    lo, hi, n = cfg.pareto_sweep
    plans = []
    for rw in np.linspace(lo, hi, int(n)):
        p = astar(reference, start, goal, cell_size, float(rw), mode=f"rw={rw:g}")
        plans.append(_rescore(p, reference, truth, cell_size, entropy))
    return plans


def pareto_front(plans: Sequence[Plan]) -> list[Plan]:
    """Non-dominated plans for (minimise length_m, maximise success_prob), sorted by length.

    A plan is dominated if another is no longer AND no less safe, and strictly better in one.
    Duplicate (length, success) points keep the first occurrence.
    """
    found = [p for p in plans if p.found]
    found.sort(key=lambda p: (p.length_m, -p.success_prob))
    front: list[Plan] = []
    best = -1.0
    for p in found:
        if p.success_prob > best + 1e-12:
            front.append(p)
            best = p.success_prob
    return front


def run_pareto(layers: TerrainLayers, belief: Belief, start: RC, goal: RC, cfg: Config = DEFAULT,
               cell_size: float = 1.0, science_waypoints: Sequence[RC] | None = None,
               truth: CostMap | None = None, do_sweep: bool = True) -> ParetoResult:
    """Modes + sweep + front in one call (what the dashboard and the demo use)."""
    ref = costmap_from_belief(layers, belief, cfg)
    modes = plan_modes(layers, belief, start, goal, cfg, cell_size, science_waypoints, truth, ref)
    sweep = sweep_risk(ref, start, goal, cfg, cell_size, truth, belief.entropy) if do_sweep else []
    front = pareto_front([*sweep, *(p for m, p in modes.items() if m != "science")])
    return ParetoResult(modes, sweep, front, ref)


def best_science_waypoint(value: np.ndarray, costmap: CostMap, belief: Belief | None = None) -> RC:
    """Stub until P4 (science/targets.py): passable cell with the highest science value."""
    v = np.where(costmap.passable, np.asarray(value, np.float32), -np.inf)
    r, c = np.unravel_index(int(np.argmax(v)), v.shape)
    return int(r), int(c)


if __name__ == "__main__":
    from pathlib import Path

    from core.io import hillshade
    import matplotlib.pyplot as plt

    from belief.bayes import init_prior, sense
    from costmap.fuse import build_costmap, one_hot_probs
    from perception.sensor_model import load_confusion
    from terrain.features import compute_layers
    from world.synthetic import make_world

    cfg = DEFAULT
    w = make_world(200, 200, seed=cfg.seed)
    L = compute_layers(w, cfg)
    C, src = load_confusion()
    belief = sense(init_prior(w, cfg), w, w.start, np.random.default_rng(cfg.seed), cfg, C)
    truth = build_costmap(L, one_hot_probs(w.true_class), cfg)
    ref = costmap_from_belief(L, belief, cfg)
    wp = best_science_waypoint(w.ice_signal, ref)

    t0 = time.perf_counter()
    res = run_pareto(L, belief, w.start, w.goal, cfg, w.cell_size, [wp], truth)
    dt = time.perf_counter() - t0
    print(f"world seed={w.seed} start={w.start} goal={w.goal}; belief = orbital prior + 1 look; "
          f"confusion={src}; science waypoint (max ice, stub) = {wp}")
    print(f"{'mode':9s} {'length_m':>9s} {'energy_wh':>10s} {'P(succ) belief':>15s} {'P(succ) truth':>14s} "
          f"{'H path':>7s} {'expanded':>9s} {'ms':>6s}")
    for row in res.table():
        print(f"{row['mode']:9s} {row['length_m']:9.1f} {row['energy_wh']:10.1f} {row['success_prob']:15.1%} "
              f"{row['success_prob_true']:14.1%} {row['path_entropy_mean']:7.3f} {row['expanded_nodes']:9d} "
              f"{1000 * row['runtime_s']:6.0f}")
    print(f"sweep risk_weight {cfg.pareto_sweep}: {len(res.sweep)} plans, front = {len(res.front)} "
          f"non-dominated; total {dt:.2f} s")
    for p in res.front:
        print(f"  front: {p.mode:9s} {p.length_m:7.1f} m  P(succ) {p.success_prob:.1%} "
              f"(truth {p.meta['success_prob_true']:.1%})")

    colors = {"shortest": "#ff4d4d", "balanced": "#ffd24d", "safest": "#4dff88", "science": "#4dc3ff"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].imshow(hillshade(w.dem, w.cell_size), cmap="gray")
    im = ax[0].imshow(belief.entropy, cmap="magma", alpha=0.55, vmin=0, vmax=np.log(4))
    ax[0].imshow(np.ma.masked_where(ref.passable, np.ones(ref.shape)), cmap="gray_r", vmin=0, vmax=1, alpha=0.8)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="belief entropy (nats); black = impassable")
    for name, p in res.modes.items():
        if p.found:
            r, c = p.as_arrays()
            ax[0].plot(c, r, "-", color=colors[name], lw=2,
                       label=f"{name}: {p.length_m:.0f} m, P={p.success_prob:.0%} (truth {p.meta['success_prob_true']:.0%})")
    ax[0].plot(w.start[1], w.start[0], "o", ms=11, mfc="lime", mec="k")
    ax[0].plot(w.goal[1], w.goal[0], "*", ms=15, mfc="magenta", mec="k")
    ax[0].plot(wp[1], wp[0], "D", ms=9, mfc="#4dc3ff", mec="k")
    ax[0].legend(loc="lower left", fontsize=8.5)
    ax[0].set_title("Route options on the rover's belief")
    ax[0].set_xticks([]); ax[0].set_yticks([])

    sl = np.array([p.length_m for p in res.sweep]); ss = np.array([p.success_prob for p in res.sweep])
    st = np.array([p.meta["success_prob_true"] for p in res.sweep])
    ax[1].scatter(sl, ss, c=[p.meta["risk_weight"] for p in res.sweep], cmap="viridis", s=50, zorder=3,
                  label="sweep (belief), colour = risk_weight")
    ax[1].scatter(sl, st, marker="x", color="grey", s=40, zorder=2, label="same routes, truth")
    fl = [p.length_m for p in res.front]; fs = [p.success_prob for p in res.front]
    ax[1].step(fl, fs, where="post", color="k", lw=1.2, label="Pareto front")
    for name, p in res.modes.items():
        if p.found:
            ax[1].plot(p.length_m, p.success_prob, "o", ms=12, mfc="none", mec=colors[name], mew=2.5)
            ax[1].annotate(name, (p.length_m, p.success_prob), textcoords="offset points", xytext=(8, -12))
    ax[1].set_xlabel("path length (m)")
    ax[1].set_ylabel("P(mission success)")
    ax[1].set_title("Length vs safety trade-off")
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/pareto.png", dpi=95)
    plt.close(fig)
    print("saved outputs/pareto.png")
