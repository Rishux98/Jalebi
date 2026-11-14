"""Tests for ui/animation.py — client-side rover animation."""
from __future__ import annotations

import numpy as np
import pytest

from core.types import LogEntry
from explain.log import make_evidence
from ui import figures as F
from ui.animation import HOVER, keyframes, make_path_animation, short_reason
from world.synthetic import make_world


@pytest.fixture(scope="module")
def world():
    return make_world(48, 48, seed=3)


def _meta(w):
    h, ww = w.shape
    probs = np.zeros((h, ww, 4), np.float32)
    probs[..., 0] = 1.0
    return dict(slope=np.zeros((h, ww)), class_probs=probs, hazard=np.zeros((h, ww)),
                energy_wh_per_m=np.full((h, ww), 1.5), sensor_radius=6)


def test_keyframes_keep_start_goal_and_replans():
    k = keyframes(400, [123, 377], 150)
    assert k[0] == 0 and k[-1] == 399 and 123 in k and 377 in k and len(k) <= 150
    assert keyframes(10, [], 150) == list(range(10))


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_animation_frames_and_hover(world, mode):
    path = [(10, c) for c in range(5, 40)]
    ev = make_evidence((11, 20), (10, 18), 0.6, "big_rock_impassable_p", p_big_rock=0.81, hazard_prob=0.41,
                       slope_deg=28.0)
    rep = LogEntry(step=13, event="impassable_on_path", decision="replan", evidence=ev, confidence=0.7)
    base = F.terrain_3d(world) if mode == "3d" else F.grid_map(np.zeros(world.shape), "Inferno", "", "cost")
    fig = make_path_animation(base, path, world.dem, world.cell_size, mode, _meta(world), [rep])
    assert len(fig.frames) >= 2
    pts = next(t for t in fig.data if t.name == "path points")
    assert len(pts.customdata) == len(path) and pts.hovertemplate == HOVER
    assert pts.customdata[-1][8] == pytest.approx(len(path) - 1)        # cumulative distance (m)
    titles = [f.layout.title.text for f in fig.frames]
    assert any("Step 13 · REPLAN: P(big_rock)=0.81 > 0.60" in t for t in titles)
    assert any(t.name == "replan" for t in fig.data)
    labels = [b.label for m in fig.layout.updatemenus for b in m.buttons]
    assert {"▶ Play", "⏸ Pause", "⟲ Restart", "0.5×", "1×", "2×", "4×"} <= set(labels)
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)
    fig.to_json()


def test_loop_repeats_frames(world):
    path = [(10, c) for c in range(5, 30)]
    once = make_path_animation(F.terrain_3d(world), path, world.dem, 1.0, "3d", _meta(world))
    looped = make_path_animation(F.terrain_3d(world), path, world.dem, 1.0, "3d", _meta(world), loop=True)
    assert len(looped.frames) == 3 * len(once.frames)
    assert len(looped.layout.sliders[0].steps) == len(once.frames)      # scrubber covers one cycle


def test_short_reason_from_evidence():
    ev = make_evidence((1, 1), (0, 0), 25.0, "slope_max_deg", slope_deg=28.4, p_big_rock=0.81)
    assert short_reason(ev) == "slope 28° > 25°, P(big_rock)=0.81"
