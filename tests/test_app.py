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
    at.sidebar.selectbox(key="size").select(64).run()                  # small site for speed
    assert not at.exception
    for mode in ("shortest", "safest", "science"):
        at.sidebar.selectbox(key="mode").select(mode).run()
        assert not at.exception, mode
    at.sidebar.button(key="run_btn").click().run()
    assert not at.exception
    at.sidebar.button(key="inject_btn").click().run()
    assert not at.exception
    assert any("Inject" in str(c.value) or "injected" in str(c.value) for c in at.caption)
    log_tables = [d.value for d in at.dataframe if "reason" in d.value.columns and "decision" in d.value.columns]
    assert log_tables and len(log_tables[0]) > 0                       # filter not left empty after a run
    assert (log_tables[0]["decision"] == "plan").any()
    at.sidebar.selectbox(key="mode").select("balanced").run()
    assert not at.exception
    at.sidebar.button(key="plan_btn").click().run()
    assert not at.exception
    from world.real_dem import default_tile_available
    if default_tile_available():                                       # real HiRISE terrain end-to-end
        at.sidebar.selectbox(key="source").select("HiRISE south-polar DTM").run()
        assert not at.exception
        at.sidebar.button(key="inject_btn").click().run()
        assert not at.exception
