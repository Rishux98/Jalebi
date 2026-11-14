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
from perception.sensor_model import get_sensor_info
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
    info = get_sensor_info(cfg)                     # same matrix the UI badge reports
    C, src = info["confusion"], info["source"]
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


KIND_LABEL = {"big_rock": "boulder field", "sand": "sand trap"}


def run_traverse(size: int, seed: int, cfg: Config, mode: str, inject: bool, frac: float, radius: float,
                 site: Site, science: Science, source: str = "synthetic",
                 obstacles: tuple[tuple[int, int, float, str], ...] = ()) -> Traverse:
    """Drive the rover from start to goal in a fresh copy of the site's world.

    ``inject`` plants one boulder field on the initial path (auto). ``obstacles`` are user-placed hidden hazards
    ``(row, col, radius_cells, kind)`` with kind "big_rock" | "sand"; ones that would cover the start or goal
    (or fall off the map) are skipped and reported in ``note``. The rover is told about none of them.
    """
    w = new_world(size, seed, cfg, source)
    targets = [Target(**{**t.__dict__}) for t in science.selection.selected] if mode == "science" else []
    rover = Rover(w, cfg, mode, seed=seed, confusion=site.confusion, targets=targets, layers=site.layers,
                  belief=site.prior)
    notes = []
    if inject:
        try:
            rc = rover.inject_hazard_on_path(frac, radius, "big_rock")
            notes.append(f"hidden boulder field r={radius:g} planted at {rc} on the initial path (rover not told)")
        except ValueError as e:
            notes.append(f"no hazard injected: {e}")
    placed, skipped = [], []
    for r, c, rad, kind in obstacles:
        r, c, rad = int(r), int(c), float(rad)
        cheb = lambda a: max(abs(a[0] - r), abs(a[1] - c))  # noqa: E731
        if not w.in_bounds((r, c)) or kind not in KIND_LABEL:
            skipped.append(f"({r}, {c}) invalid")
        elif cheb(w.start) <= rad or cheb(w.goal) <= rad:
            skipped.append(f"({r}, {c}) covers start/goal")
        else:
            rover.inject_hazard((r, c), rad, kind)
            placed.append(f"{KIND_LABEL[kind]} r={rad:g} at ({r}, {c})")
    if placed:
        notes.append(f"{len(placed)} user obstacle(s), rover not told: " + "; ".join(placed))
    if skipped:
        notes.append("skipped: " + "; ".join(skipped))
    note = " · ".join(notes)
    res = rover.run_to_goal()
    return Traverse(res, w, [Target(**{**t.__dict__}) for t in science.selection.selected] if mode == "science" else [],
                    mode, note)


@dataclass
class OnboardRun:
    """Onboard-memory-mode traverse of the same site + hidden hazards as a FULL traverse."""
    result: RunResult
    world: World
    tier_frames: list
    tile: int
    store_stats: dict
    compact_cost_bytes: int
    error: dict
    layers: list
    resident_full_bytes: int
    resident_onboard_bytes: int


FULL_BYTES_PER_CELL = {  # FULL-mode rover state, bytes per cell (float32 unless noted)
    "terrain layers (slope, roughness, curvature, P(slope>25°))": 16, "elevation": 4,
    "cost + hazard + energy + 5 cost components": 32, "belief (4 class probs)": 16, "belief entropy": 4,
    "observed mask (bool) + visited (bool)": 2, "navcam look counts (int32)": 4}


def run_onboard(size: int, seed: int, cfg: Config, mode: str, site: Site, science: Science,
                injected: tuple = (), source: str = "synthetic") -> OnboardRun:
    """Drive the OnboardRover (tiered tiles + compact cost state + hierarchical A*) with the same hidden hazards
    ``(row, col, radius, kind)`` the FULL traverse had. Ground station builds + uplinks the tile store first."""
    from onboard.quantize import QuantizedDEM, pack_classes, report_error
    from onboard.rover import OnboardRover
    from onboard.tiered_store import build_store

    w = new_world(size, seed, cfg, source)
    store = build_store(w, site.layers, cfg, prefetch_sync=True, simulate_latency=False, usage_path=None)
    targets = [Target(**{**t.__dict__}) for t in science.selection.selected] if mode == "science" else []
    rover = OnboardRover(w, store, cfg, mode, seed=seed, confusion=site.confusion, targets=targets, belief=site.prior)
    for r, c, rad, kind in injected:
        rover.inject_hazard((int(r), int(c)), float(rad), kind)
    res = rover.run_to_goal()
    st = store.stats()
    H, W = w.shape
    cells = H * W
    qd = QuantizedDEM.from_float(w.dem)
    layers = [("elevation", 4 * cells, qd.nbytes, "int16 · 0.01 m · in tiles"),
              ("slope", 4 * cells, cells, "uint8 · 0.5° · in tiles"),
              ("roughness", 4 * cells, cells, "uint8 · in tiles"),
              ("curvature", 4 * cells, 0, "computed per window"),
              ("P(slope > 25°)", 4 * cells, 0, "computed per window"),
              ("cost + 5 components", 24 * cells, cells, "uint8 · 255 = impassable"),
              ("hazard_prob", 4 * cells, cells, "uint8 · log scale 1e-6…1"),
              ("energy Wh/m", 4 * cells, cells, "uint8"),
              ("terrain class (packed)", cells, int(pack_classes(w.true_class).nbytes), "2 bits / cell")]
    resident_on = (st["hot_bytes"] + st["warm_bytes"] + st["resident_coarse_bytes"] + rover.ccm.nbytes
                   + cells * (16 + 4 + 2 + 4))
    store.close()
    return OnboardRun(res, w, rover.tier_frames, store.tile, st, rover.ccm.nbytes,
                      report_error(w.dem, w.cell_size, cfg=cfg), layers, cells * sum(FULL_BYTES_PER_CELL.values()),
                      int(resident_on))


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
