"""FULL vs ONBOARD benchmark: the same traverse (same hidden boulder field) in both memory modes.

Measured per site:
  * peak RAM during rover boot + traverse (tracemalloc). FULL copies its float32 terrain layers inside the measured
    region so they count; ONBOARD counts its HOT/WARM tiers, coarse map and compact cost state. The tile "flash"
    is storage and is reported separately. Dashboard telemetry (per-step replay frames, entropy snapshots) is
    disabled in BOTH modes — it is not flight memory.
  * resident state bytes (arrays held at the end), bytes per layer (full vs compact format)
  * tier sizes, HOT hit rate, stalls, prefetch hit rate, evictions
  * path length / initial-plan P(success) differences, replans, boulder cells entered, wall time
Writes onboard/bench.json.   Run: python -m onboard.bench --sizes 200 hirise 1024
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from autonomy.loop import Rover
from belief.bayes import init_prior
from core.config import DEFAULT, Config
from core.types import TerrainLayers
from onboard.quantize import QuantizedDEM, pack_classes, report_error
from onboard.rover import OnboardRover
from onboard.tiered_store import build_store
from terrain.features import compute_layers

OUT = Path(__file__).resolve().parent / "bench.json"


class _LeanRover(Rover):
    """FULL rover without dashboard telemetry (no per-step path copies / entropy snapshots)."""

    def _frame(self, replanned: bool) -> None:
        pass


class _LeanOnboardRover(OnboardRover):
    def _frame(self, replanned: bool) -> None:
        pass


def _arrays_bytes(*objs) -> int:
    total = 0
    for o in objs:
        if isinstance(o, np.ndarray):
            total += o.nbytes
    return int(total)


def _make_world(site: str, seed: int, cfg: Config):
    if site == "hirise":
        from world.real_dem import load_world
        return lambda: load_world(cfg=cfg, seed=seed)
    n = int(site)
    from world.synthetic import make_world
    return lambda: make_world(n, n, seed=seed, cfg=cfg)


def run_case(site: str = "200", seed: int = 42, cfg: Config = DEFAULT, mode: str = "balanced",
             frac: float = 0.4, hz_radius: float = 3, measure_ram: bool = True, read_latency_s: float = 0.002,
             prefetch_sync: bool = True) -> dict:
    """Run FULL then ONBOARD on the same site + hidden hazard; return the comparison dict."""
    cfg = cfg.replace(loop_snapshot_every=10 ** 9)
    new_world = _make_world(site, seed, cfg)
    t0 = time.perf_counter()
    w_full = new_world()
    layers = compute_layers(w_full, cfg)                 # ground-station product for FULL mode
    prior = init_prior(w_full, cfg, seed)
    prep_s = time.perf_counter() - t0
    H, W = w_full.shape

    # ---------------- FULL
    if measure_ram:
        tracemalloc.start()
    t0 = time.perf_counter()
    L_onboard = TerrainLayers(layers.slope_deg.copy(), layers.roughness.copy(), layers.curvature.copy(),
                              layers.p_slope_exceed.copy())          # float32 layers held in rover RAM
    full = _LeanRover(w_full, cfg, mode, seed=seed, layers=L_onboard, belief=prior)
    hz = full.inject_hazard_on_path(frac, hz_radius, "big_rock")
    while not full.done and full.step_idx < cfg.loop_max_steps:
        full.step()
    full_s = time.perf_counter() - t0
    full_peak = tracemalloc.get_traced_memory()[1] if measure_ram else 0
    if measure_ram:
        tracemalloc.stop()
    cm = full.costmap
    full_resident = _arrays_bytes(L_onboard.slope_deg, L_onboard.roughness, L_onboard.curvature,
                                  L_onboard.p_slope_exceed, w_full.dem, cm.cost, cm.hazard_prob, cm.energy_wh_per_m,
                                  *cm.components.values(), full.belief.class_probs, full.belief.entropy,
                                  full.belief.observed_mask, full.n_looks, full.visited)
    full_res = full.result(full_s)

    # ---------------- ONBOARD (ground station builds + uplinks the tile store; flash = storage)
    w_on = new_world()
    store = build_store(w_on, layers, cfg, prefetch_sync=prefetch_sync, read_latency_s=read_latency_s,
                        usage_path=None)
    if measure_ram:
        tracemalloc.start()
    t0 = time.perf_counter()
    on = _LeanOnboardRover(w_on, store, cfg, mode, seed=seed, belief=prior, telemetry=False)
    on.inject_hazard(hz, hz_radius, "big_rock")
    while not on.done and on.step_idx < cfg.loop_max_steps:
        on.step()
    store.wait_prefetch()
    on_s = time.perf_counter() - t0
    on_peak = tracemalloc.get_traced_memory()[1] if measure_ram else 0
    if measure_ram:
        tracemalloc.stop()
    st = store.stats()
    on_resident = (st["hot_bytes"] + st["warm_bytes"] + st["resident_coarse_bytes"] + on.ccm.nbytes
                   + _arrays_bytes(on.belief.class_probs, on.belief.entropy, on.belief.observed_mask, on.n_looks,
                                   on.visited))
    on_res = on.result(on_s)
    store.close()

    cells = H * W
    qd = QuantizedDEM.from_float(w_full.dem)
    per_layer = [  # (layer, full bytes, compact/onboard bytes, how)
        ("elevation", 4 * cells, qd.nbytes, "int16 (0.01 m) in tiles"),
        ("slope", 4 * cells, cells, "uint8 0.5° in tiles"),
        ("roughness", 4 * cells, cells, "uint8 in tiles"),
        ("curvature", 4 * cells, 0, "on the fly per window"),
        ("P(slope>25°)", 4 * cells, 0, "on the fly per window"),
        ("cost + 5 components", 24 * cells, cells, "uint8, 255 = inf"),
        ("hazard_prob", 4 * cells, cells, "uint8 log-scale"),
        ("energy Wh/m", 4 * cells, cells, "uint8"),
        ("terrain class (truth / packed)", cells, int(pack_classes(w_full.true_class).nbytes), "2 bits"),
    ]
    fp, op = full_res.plans[0], on_res.plans[0]
    return {
        "site": site, "shape": [H, W], "cell_size_m": w_full.cell_size, "seed": seed, "mode": mode,
        "hazard_rc": list(hz), "prep_s": round(prep_s, 2),
        "quantisation": report_error(w_full.dem, w_full.cell_size, cfg=cfg),
        "layers": [{"layer": n, "full_bytes": f, "onboard_bytes": o, "format": how,
                    "reduction": (round(f / o, 1) if o else None)} for n, f, o, how in per_layer],
        "ram": {"full_peak_bytes": int(full_peak), "onboard_peak_bytes": int(on_peak),
                "peak_reduction": round(full_peak / on_peak, 2) if on_peak else None,
                "full_resident_bytes": full_resident, "onboard_resident_bytes": int(on_resident),
                "resident_reduction": round(full_resident / on_resident, 2) if on_resident else None},
        "tiers": {k: st[k] for k in ("tiles", "hot_tiles", "warm_tiles", "hot_bytes", "warm_bytes", "flash_bytes",
                                     "resident_coarse_bytes")},
        "cache": {k: st[k] for k in ("requests", "hits_hot", "hits_warm", "misses_cold", "stalls", "prefetch_loaded",
                                     "prefetch_hits", "evictions_hot", "evictions_warm", "flash_reads",
                                     "simulated_flash_latency_s")}
                 | {"hit_rate": round(st["hit_rate"], 4), "prefetch_hit_rate": round(st["prefetch_hit_rate"], 4)},
        "full": {"reached_goal": full_res.reached_goal, "distance_m": round(full_res.distance_m, 1),
                 "replans": full_res.replans, "initial_success_prob": round(fp.success_prob, 4),
                 "initial_length_m": round(fp.length_m, 1), "expanded_nodes": int(sum(p.expanded_nodes for p in full_res.plans)),
                 "boulder_cells_entered": full_res.meta["entered_true_hazard_cells"], "wall_s": round(full_s, 2)},
        "onboard": {"reached_goal": on_res.reached_goal, "distance_m": round(on_res.distance_m, 1),
                    "replans": on_res.replans, "initial_success_prob": round(op.success_prob, 4),
                    "initial_length_m": round(op.length_m, 1), "expanded_nodes": int(sum(p.expanded_nodes for p in on_res.plans)),
                    "boulder_cells_entered": on_res.meta["entered_true_hazard_cells"], "wall_s": round(on_s, 2)},
        "diff": {"driven_length_pct": round(100 * (on_res.distance_m / full_res.distance_m - 1), 2),
                 "initial_plan_length_pct": round(100 * (op.length_m / fp.length_m - 1), 2),
                 "initial_success_prob_pts": round(100 * (op.success_prob - fp.success_prob), 2)},
        "paths": {"full": [list(p) for p in full_res.trajectory], "onboard": [list(p) for p in on_res.trajectory]},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["200", "hirise", "1024"])
    ap.add_argument("--seed", type=int, default=DEFAULT.seed)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    results = []
    for site in a.sizes:
        t0 = time.perf_counter()
        r = run_case(site, a.seed)
        results.append(r)
        ram, d = r["ram"], r["diff"]
        print(f"[{site:>6}] {r['shape'][0]}x{r['shape'][1]}  peak RAM full {ram['full_peak_bytes'] / 1e6:.1f} MB -> "
              f"onboard {ram['onboard_peak_bytes'] / 1e6:.1f} MB ({ram['peak_reduction']}x) | resident "
              f"{ram['full_resident_bytes'] / 1e6:.1f} -> {ram['onboard_resident_bytes'] / 1e6:.1f} MB "
              f"({ram['resident_reduction']}x) | hit {r['cache']['hit_rate']:.1%} stalls {r['cache']['stalls']} "
              f"prefetch {r['cache']['prefetch_hit_rate']:.0%} | replans {r['full']['replans']}/{r['onboard']['replans']} "
              f"| driven {d['driven_length_pct']:+.1f}% | P(succ) {d['initial_success_prob_pts']:+.1f} pts | "
              f"wall {r['full']['wall_s']}s/{r['onboard']['wall_s']}s | goal {r['full']['reached_goal']}/"
              f"{r['onboard']['reached_goal']} rocks {r['full']['boulder_cells_entered']}/"
              f"{r['onboard']['boulder_cells_entered']} ({time.perf_counter() - t0:.0f}s)")
    Path(a.out).write_text(json.dumps({"generated_by": "python -m onboard.bench", "cases": results}, indent=1))
    print("saved", a.out)
