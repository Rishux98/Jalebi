"""Contract tests for core/types.py and core/config.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.config import DEFAULT, Config
from core.types import (
    BIG_ROCK, N_CLASSES, Belief, CostMap, LogEntry, Plan, Target, TerrainLayers, World,
)


def _world(h: int = 8, w: int = 10) -> World:
    return World(dem=np.zeros((h, w)), cell_size=1, true_class=np.zeros((h, w), int),
                 ice_signal=np.zeros((h, w)), start=(0, 0), goal=(h - 1, w - 1), seed=0)


def test_world_casts_and_validates() -> None:
    w = _world()
    assert w.dem.dtype == np.float32 and w.true_class.dtype == np.int8
    assert w.shape == (8, 10) and isinstance(w.start[0], int)
    with pytest.raises(ValueError):
        World(np.zeros((4, 4)), 1.0, np.zeros((4, 5)), np.zeros((4, 4)), (0, 0), (3, 3), 0)
    with pytest.raises(ValueError):
        World(np.zeros((4, 4)), 1.0, np.zeros((4, 4)), np.zeros((4, 4)), (0, 0), (4, 3), 0)
    with pytest.raises(ValueError):
        World(np.zeros((4, 4)), 0.0, np.zeros((4, 4)), np.zeros((4, 4)), (0, 0), (3, 3), 0)


def test_terrain_layers() -> None:
    z = np.zeros((6, 6))
    t = TerrainLayers(z, z, z, z)
    assert all(getattr(t, n).dtype == np.float32 for n in ("slope_deg", "roughness", "curvature", "p_slope_exceed"))
    with pytest.raises(ValueError):
        TerrainLayers(z, np.zeros((6, 5)), z, z)


def test_belief_helpers() -> None:
    b = Belief.uniform(5, 7)
    assert b.class_probs.shape == (5, 7, N_CLASSES) and b.class_probs.dtype == np.float32
    np.testing.assert_allclose(b.entropy, np.log(4), rtol=1e-5)
    assert b.observed_mask.dtype == bool and not b.observed_mask.any()
    cls = np.full((5, 7), BIG_ROCK)
    oh = Belief.one_hot(cls)
    np.testing.assert_allclose(oh.entropy, 0.0, atol=1e-6)
    assert oh.class_probs[..., BIG_ROCK].min() == 1.0
    with pytest.raises(ValueError):
        Belief(np.zeros((5, 7, 3)), np.zeros((5, 7)), np.zeros((5, 7), bool))


def test_costmap() -> None:
    c = np.ones((4, 4))
    c[1, 1] = np.inf
    cm = CostMap(c, np.zeros((4, 4)), {"slope": np.ones((4, 4))})
    assert cm.cost.dtype == np.float32 and cm.components["slope"].dtype == np.float32
    assert not cm.passable[1, 1] and cm.passable.sum() == 15
    bd = cm.breakdown((0, 0))
    assert bd == {"slope": 1.0, "total": 1.0, "hazard_prob": 0.0}
    with pytest.raises(ValueError):
        CostMap(c, np.zeros((4, 4)), {"x": np.zeros((3, 3))})


def test_plan_defaults_and_arrays() -> None:
    p = Plan(path=[])
    assert not p.found and p.success_prob == 0.0 and p.total_cost == float("inf")
    q = Plan(path=[(np.int64(0), 1), (1, 2)], total_cost=2.8, length_m=1.41, success_prob=0.9, mode="safest")
    assert q.found and type(q.path[0][0]) is int
    rows, cols = q.as_arrays()
    assert rows.tolist() == [0, 1] and cols.tolist() == [1, 2]
    json.dumps(q.summary())


def test_target_and_logentry() -> None:
    t = Target(id=0, rc=(3, 4), value_mean=0.7, value_std=0.1)
    assert t.plan is None and not t.reachable
    e = LogEntry(step=1, event="hazard_on_path", decision="replan",
                 evidence={"cell": [3, 4], "rover_rc": [0, 0], "threshold": 0.3,
                           "threshold_name": "hazard_prob_max"}, confidence=1.7)
    assert e.confidence == 1.0
    assert json.loads(json.dumps(e.to_dict()))["evidence"]["cell"] == [3, 4]


def test_config() -> None:
    assert DEFAULT.slope_safe_deg == 15 and DEFAULT.slope_max_deg == 25
    assert DEFAULT.class_risk[BIG_ROCK] == 1.0 and DEFAULT.sensor_radius == 6
    assert DEFAULT.dem_noise_sigma == 0.3 and DEFAULT.seed == 42
    c2 = DEFAULT.replace(lambda_unc=3.0)
    assert c2.lambda_unc == 3.0 and DEFAULT.lambda_unc == 1.0
    with pytest.raises(Exception):
        DEFAULT.seed = 1  # type: ignore[misc]
    assert isinstance(Config().to_dict(), dict)
