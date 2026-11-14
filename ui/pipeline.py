"""Dashboard computation: site, route options, science selection, autonomous traverse.

Plain functions taking primitives (so Streamlit can cache them by argument); no Streamlit imports.
Each traverse builds a *fresh* World, because ``inject_hazard*`` mutates the ground truth.

Run: python -m ui.pipeline   (times every stage on a 200x200 site)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autonomy.loop import Rover, RunResult
from belief.bayes import init_prior, sense
from core.config import DEFAULT, Config
from core.types import Belief, CostMap, Target, TerrainLayers, World
from costmap.fuse import build_costmap, costmap_from_belief, one_hot_probs
from explain.log import DecisionLog
from perception.sensor_model import load_confusion
from planner.pareto import ParetoResult, run_pareto
from science.targets import Selection, detect_candidates, plan_targets, select_under_budget
from terrain.features import compute_layers
from world.synthetic import make_world

MODES = ("shortest", "balanced", "safest", "science")


def make_cfg(lambda_unc: float, budget_wh: float, sensor_radius: int, connectivity: int = 8) -> Config:
    """Config with the dashboard's sidebar overrides (connectivity 8 = diagonals, 4 = horizontal/vertical only)."""
    return DEFAULT.replace(lambda_unc=float(lambda_unc), sci_energy_budget_wh=float(budget_wh),
                           sensor_radius=int(sensor_radius), grid_connectivity=int(connectivity))


@dataclass
class Site:
    """Everything known before the rover moves (plus the hidden truth, for evaluation only)."""
    world: World
    layers: TerrainLayers
    prior: Belief                 # orbital prior, nothing sensed
    belief: Belief                # prior + first navcam look at the start
    costmap: CostMap              # belief cost map (cfg weights)
    truth_costmap: CostMap        # ground-truth cost map (evaluation only)
    confusion: np.ndarray
    confusion_source: str
    cfg: Config
    direct_energy_wh: float = 0.0  # balanced start->goal plan energy (budget reference)


@dataclass
class Science:
    targets: list[Target]         # all candidates, ranked by utility
    selection: Selection
    log: DecisionLog


@dataclass
class Traverse:
    result: RunResult
    world: World                  # the world the rover drove in (with injected hazards)
    targets: list[Target] = field(default_factory=list)
    initial_plan_mode: str = "balanced"
    note: str = ""


def new_world(size: int, seed: int, cfg: Config, source: str = "synthetic") -> World:
    """Fresh World: procedural synthetic site, or the real HiRISE crop (``source="hirise"``; size ignored)."""
    if source == "hirise":
        from world.real_dem import load_world
        return load_world(cfg=cfg, seed=seed)
    return make_world(size, size, seed=seed, cfg=cfg)


def build_site(size: int, seed: int, cfg: Config, source: str = "synthetic",
               budget_factor: float | None = None) -> Site:
    """Site + belief. If ``budget_factor`` is given, the energy budget becomes
    ``budget_factor * energy of the balanced direct start->goal plan`` (site-size independent)."""
    from planner.astar import astar

    w = new_world(size, seed, cfg, source)
    L = compute_layers(w, cfg)
    C, src = load_confusion(cfg=cfg)
    prior = init_prior(w, cfg, seed)
    b = sense(prior, w, w.start, np.random.default_rng(seed), cfg, C)
    cm = costmap_from_belief(L, b, cfg)
    direct = astar(cm, w.start, w.goal, w.cell_size, cfg.pareto_risk_balanced,
                   connectivity=cfg.grid_connectivity).energy_wh
    if budget_factor is not None:
        cfg = cfg.replace(sci_energy_budget_wh=float(budget_factor) * direct)
    return Site(w, L, prior, b, cm, build_costmap(L, one_hot_probs(w.true_class), cfg), C, src, cfg, direct)


def build_science(site: Site) -> Science:
    cfg, w = site.cfg, site.world
    log = DecisionLog()
    cands = detect_candidates(w, site.layers, site.belief, site.costmap, cfg, w.seed)
    ranked = plan_targets(cands, site.costmap, w.start, cfg, w.cell_size)
    ranked_info = {t.id: (t.plan, t.utility, t.reason) for t in ranked}
    sel = select_under_budget([Target(**{**t.__dict__}) for t in ranked], site.costmap, site.belief, w.start, w.goal,
                              cfg.sci_energy_budget_wh, cfg, w.cell_size, log=log)
    for t in ranked:                       # keep the from-start leg/utility for the ranking table
        t.plan, t.utility, t.reason = ranked_info[t.id]
    return Science(ranked, sel, log)


def build_routes(site: Site, science: Science) -> ParetoResult:
    w = site.world
    return run_pareto(site.layers, site.belief, w.start, w.goal, site.cfg, w.cell_size,
                      science.selection.waypoints or None, site.truth_costmap)


def run_traverse(size: int, seed: int, cfg: Config, mode: str, inject: bool, frac: float, radius: float,
                 site: Site, science: Science, source: str = "synthetic") -> Traverse:
    """Drive the rover from start to goal in a fresh copy of the site's world."""
    w = new_world(size, seed, cfg, source)
    targets = [Target(**{**t.__dict__}) for t in science.selection.selected] if mode == "science" else []
    rover = Rover(w, cfg, mode, seed=seed, confusion=site.confusion, targets=targets, layers=site.layers,
                  belief=site.prior)
    note = ""
    if inject:
        try:
            rc = rover.inject_hazard_on_path(frac, radius, "big_rock")
            note = f"hidden boulder field r={radius:g} planted at {rc} on the initial path (rover not told)"
        except ValueError as e:
            note = f"no hazard injected: {e}"
    res = rover.run_to_goal()
    return Traverse(res, w, [Target(**{**t.__dict__}) for t in science.selection.selected] if mode == "science" else [],
                    mode, note)


if __name__ == "__main__":
    import time

    cfg = make_cfg(DEFAULT.lambda_unc, DEFAULT.sci_energy_budget_wh, DEFAULT.sensor_radius)
    t0 = time.perf_counter(); site = build_site(200, 42, cfg); t1 = time.perf_counter()
    sci = build_science(site); t2 = time.perf_counter()
    routes = build_routes(site, sci); t3 = time.perf_counter()
    tr = run_traverse(200, 42, cfg, "balanced", True, 0.4, 3, site, sci); t4 = time.perf_counter()
    print(f"site {t1 - t0:.2f}s | science {t2 - t1:.2f}s ({len(sci.selection.selected)} selected) | "
          f"routes {t3 - t2:.2f}s ({list(routes.modes)}) | traverse {t4 - t3:.2f}s "
          f"(replans {tr.result.replans}, goal {tr.result.reached_goal}) | {tr.note}")
