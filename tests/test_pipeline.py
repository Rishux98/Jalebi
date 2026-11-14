"""End-to-end pipeline: world -> layers -> belief -> routes -> science -> autonomous traverse, under 20 s."""
from __future__ import annotations

import time

import pytest

from ui.pipeline import build_routes, build_science, build_site, make_cfg, run_traverse
from world.real_dem import default_tile_available


@pytest.mark.parametrize("source", ["synthetic", "hirise"])
def test_full_pipeline_under_20s(source):
    if source == "hirise" and not default_tile_available():
        pytest.skip("HiRISE crop not fetched")
    cfg = make_cfg(1.0, 800.0, 6)
    t0 = time.perf_counter()
    site = build_site(200, 42, cfg, source=source, budget_factor=1.45)
    sci = build_science(site)
    routes = build_routes(site, sci)
    tr = run_traverse(200, 42, site.cfg, "science", True, 0.4, 3, site, sci, source=source)
    dt = time.perf_counter() - t0
    res = tr.result
    assert dt < 20.0, f"pipeline took {dt:.1f} s"
    assert site.cfg.sci_energy_budget_wh == pytest.approx(1.45 * site.direct_energy_wh)
    assert {"shortest", "balanced", "safest"} <= set(routes.modes)
    # residual risk is real (sensor false negatives near boulder-rich science targets): measured 2 boulder
    # cells entered over 16 runs (4 seeds x 2 sites x 2 modes); allow a small number, never a pile-up
    assert res.reached_goal and res.replans >= 1 and res.meta["entered_true_hazard_cells"] <= 2
    assert res.energy_wh <= site.cfg.sci_energy_budget_wh + 1e-6 or not sci.selection.selected


def test_user_obstacles_on_route_trigger_replans_and_bad_ones_are_skipped():
    cfg = make_cfg(1.0, 800.0, 6)
    site = build_site(100, 42, cfg, budget_factor=1.45)
    sci = build_science(site)
    path = build_routes(site, sci).modes["balanced"].path
    mid = path[len(path) // 2]
    s0 = site.world.start
    obstacles = ((mid[0], mid[1], 3, "big_rock"), (s0[0], s0[1], 2, "big_rock"), (500, 500, 2, "sand"))
    tr = run_traverse(100, 42, site.cfg, "balanced", False, 0.4, 3, site, sci, obstacles=obstacles)
    res = tr.result
    assert [i["rc"] for i in res.meta["injected"]] == [[mid[0], mid[1]]]           # only the valid one planted
    assert "1 user obstacle(s)" in tr.note and "covers start/goal" in tr.note and "invalid" in tr.note
    assert res.reached_goal and res.replans >= 1 and len(res.plans) == res.replans + 1
    assert res.meta["entered_true_hazard_cells"] <= 2
