"""Headless dashboard tests: figure builders + a full Streamlit AppTest click-through (64x64 site)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from belief.bayes import init_prior
from core.types import Plan, Target
from costmap.fuse import costmap_from_belief
from terrain.features import compute_layers
from ui import figures as F
from world.synthetic import make_world

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site():
    w = make_world(48, 48, seed=3)
    L = compute_layers(w)
    b = init_prior(w)
    return w, L, b, costmap_from_belief(L, b)


def test_cost_map_hover_and_nan(site):
    w, L, b, cm = site
    fig = F.cost_map(cm)
    hm = fig.data[0]
    assert isinstance(fig, go.Figure) and hm.type == "heatmap"
    assert np.asarray(hm.customdata).shape == (48, 48, 6)            # 5 components + hazard
    assert "uncertainty" in hm.hovertemplate and "hazard_prob" in hm.hovertemplate
    z = np.asarray(hm.z, dtype=float)
    assert not np.isinf(z).any()                                       # inf -> NaN (JSON-safe)
    fig.to_json()


def test_maps_paths_and_3d(site):
    w, L, b, cm = site
    fig = F.shaded_map(w, "t", overlay=b.entropy, overlay_name="entropy")
    F.add_path(fig, [w.start, (w.start[0] + 1, w.start[1] + 1)], "p", "#fff")
    F.add_points(fig, [w.goal], "g", "#fff", text=["goal"])
    F.add_rings(fig, [{"rc": [10, 10], "radius": 3}])
    F.add_start_goal(fig, w)
    assert len(fig.data) == 2 + 1 + 1 + 1 + 2
    f3 = F.terrain_3d(w, [("p", [w.start, w.goal], "#fff", "solid")], [("t", [w.goal], "#fff", "diamond")])
    assert [d.type for d in f3.data] == ["surface", "scatter3d", "scatter3d"]
    F.class_map(b.class_probs.argmax(-1), "c").to_json()


def test_charts():
    p = Plan(path=[(0, 0), (1, 1)], length_m=10, success_prob=0.8, meta={"success_prob_true": 0.9, "risk_weight": 1})
    fig = F.pareto_fig([p], [p], {"balanced": p})
    assert {d.name for d in fig.data} >= {"risk sweep (belief)", "Pareto front", "balanced"}
    ts = [Target(id=0, rc=(1, 1), value_mean=0.5, value_std=0.05), Target(id=1, rc=(2, 2), value_mean=0.3, value_std=0.1)]
    bar = F.targets_bar(ts, {0})
    assert [d.name for d in bar.data] == ["selected", "skipped"]
    F.confusion_fig(np.eye(4), "c").to_json()


def test_app_click_through():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert any("press" in str(m.value) and "Plan" in str(m.value) for m in at.markdown)   # empty state
    at.sidebar.selectbox(key="size").select(64).run()                  # small site for speed
    assert not at.exception
    at.sidebar.button(key="plan_btn").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    for mode in ("shortest", "safest", "science"):
        at.sidebar.selectbox(key="mode").select(mode).run()
        assert not at.exception, mode
    at.sidebar.button(key="run_btn").click().run()
    assert not at.exception
    at.sidebar.button(key="inject_btn").click().run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]                   # no tab fell into its error card
    assert any("hidden boulder field" in str(m.value) for m in at.markdown)
    log_tables = [d.value for d in at.dataframe if "reason" in d.value.columns and "decision" in d.value.columns]
    assert log_tables and len(log_tables[0]) > 0                       # filter not left empty after a run
    assert (log_tables[0]["decision"] == "plan").any()
    at.sidebar.selectbox(key="mode").select("balanced").run()
    assert not at.exception
    at.sidebar.button(key="plan_btn").click().run()
    assert not at.exception
    # 4-way movement + map view navigation
    at.sidebar.radio(key="movement").set_value("4-way (horizontal / vertical only)").run()
    assert not at.exception
    at.sidebar.select_slider(key="view_zoom").set_value(4).run()
    for b in ("pan_right", "pan_down", "pan_left", "pan_up", "ctr_goal", "ctr_start", "pan_reset"):
        at.sidebar.button(key=b).click().run()
        assert not at.exception, b
    at.sidebar.select_slider(key="view_zoom").set_value(2).run()
    at.sidebar.button(key="inject_btn").click().run()
    assert not at.exception and not at.error
    at.sidebar.toggle(key="loop").set_value(True).run()                # loop: repeated frames + JS hook
    assert not at.exception and not at.error
    at.sidebar.radio(key="movement").set_value("8-way (diagonals)").run()
    # manual obstacles: place one on the route centre, drive, then the Replans tab compares before/after
    at.sidebar.radio(key="placement").set_value("Manual (choose cells)").run()
    assert not at.exception and not at.error
    assert at.sidebar.button(key="manual_btn").disabled                # nothing placed yet
    at.sidebar.slider(key="obs_r-64").set_value(32).run()
    at.sidebar.slider(key="obs_c-64").set_value(32).run()
    at.sidebar.button(key="obs_add").click().run()
    assert at.session_state.obstacles == [(32, 32, 3, "big_rock")]
    at.sidebar.button(key="manual_btn").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any("user obstacle" in str(m.value) for m in at.markdown)
    assert any("Before R1" in str(m.value) for m in at.markdown) or \
        any("No replans" in str(m.value) for m in at.success)
    at.sidebar.button(key="obs_clear").click().run()
    at.sidebar.radio(key="placement").set_value("Auto (on planned route)").run()
    assert not at.exception
    from world.real_dem import default_tile_available
    if default_tile_available():                                       # real HiRISE terrain end-to-end
        at.sidebar.selectbox(key="source").select("HiRISE south-polar DTM").run()
        assert not at.exception
        at.sidebar.button(key="inject_btn").click().run()
        assert not at.exception and not at.error, [e.value for e in at.error]
