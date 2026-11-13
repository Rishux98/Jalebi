"""Smoke tests for world/synthetic.py and World.inject_hazard."""
from __future__ import annotations

import time

import numpy as np
import pytest

from core.config import DEFAULT
from core.types import BIG_ROCK, N_CLASSES, SAND, World
from world.synthetic import bfs_reachable, make_world, passable_mask, slope_deg


@pytest.fixture(scope="module")
def world() -> World:
    return make_world(64, 64, seed=3)


def test_shapes_and_dtypes(world: World) -> None:
    assert world.dem.shape == (64, 64) and world.dem.dtype == np.float32
    assert world.ice_signal.shape == (64, 64) and world.ice_signal.dtype == np.float32
    assert world.true_class.shape == (64, 64) and world.true_class.dtype == np.int8
    assert np.isfinite(world.dem).all()
    assert set(np.unique(world.true_class)) <= set(range(N_CLASSES))
    assert 0.0 <= world.ice_signal.min() and world.ice_signal.max() <= 1.0


def test_deterministic() -> None:
    a, b = make_world(48, 48, seed=11), make_world(48, 48, seed=11)
    np.testing.assert_array_equal(a.dem, b.dem)
    np.testing.assert_array_equal(a.true_class, b.true_class)
    np.testing.assert_array_equal(a.ice_signal, b.ice_signal)
    assert (a.start, a.goal, a.seed) == (b.start, b.goal, b.seed)
    c = make_world(48, 48, seed=12)
    assert not np.array_equal(a.dem, c.dem)


@pytest.mark.parametrize("seed", range(10))
def test_start_goal_passable_and_connected(seed: int) -> None:
    w = make_world(96, 96, seed=seed)
    mask = passable_mask(w.dem, w.true_class, w.cell_size)
    sl = slope_deg(w.dem, w.cell_size)
    for rc in (w.start, w.goal):
        assert sl[rc] < DEFAULT.slope_max_deg
        assert w.true_class[rc] != BIG_ROCK
    assert w.start != w.goal
    assert bfs_reachable(mask, w.start, w.goal)


def test_start_goal_opposite_corners() -> None:
    w = make_world(200, 200, seed=DEFAULT.seed)
    assert w.start[0] < 100 and w.start[1] < 100
    assert w.goal[0] >= 100 and w.goal[1] >= 100


def test_runtime_200x200_under_1s() -> None:
    make_world(64, 64, seed=0)  # warm-up imports / caches
    t0 = time.perf_counter()
    make_world(200, 200, seed=DEFAULT.seed)
    assert time.perf_counter() - t0 < 1.0


def test_classes_and_features_present() -> None:
    w = make_world(200, 200, seed=DEFAULT.seed)
    counts = np.bincount(w.true_class.ravel(), minlength=N_CLASSES)
    assert (counts > 0).all(), counts           # all four classes appear
    assert counts[0] / counts.sum() > 0.4       # mostly soil
    assert slope_deg(w.dem, 1.0).max() > DEFAULT.slope_max_deg  # some impassable terrain
    assert w.ice_signal.max() > 0.7             # at least one ice peak


def test_bfs_reachable_basic() -> None:
    m = np.ones((5, 5), bool)
    m[:, 2] = False
    assert not bfs_reachable(m, (0, 0), (0, 4))
    m[4, 2] = True
    assert bfs_reachable(m, (0, 0), (0, 4))
    assert not bfs_reachable(m, (0, 2), (0, 4))  # start blocked


def test_small_grid_and_too_small() -> None:
    w = make_world(32, 32, seed=1)
    assert w.shape == (32, 32)
    with pytest.raises(ValueError):
        make_world(8, 8, seed=1)


# --------------------------------------------------------------------------- inject_hazard
def test_inject_hazard_modifies_true_class_only() -> None:
    w = make_world(48, 48, seed=5)
    dem0, ice0, cls0 = w.dem.copy(), w.ice_signal.copy(), w.true_class.copy()
    mask = w.inject_hazard((24, 24), radius=4, kind="big_rock")
    assert mask.dtype == bool and mask.shape == (48, 48)
    np.testing.assert_array_equal(w.dem, dem0)
    np.testing.assert_array_equal(w.ice_signal, ice0)
    assert (w.true_class[mask] == BIG_ROCK).all()
    np.testing.assert_array_equal(w.true_class[~mask], cls0[~mask])
    rr, cc = np.nonzero(mask)
    assert (np.hypot(rr - 24, cc - 24) <= 4).all()
    assert mask.sum() == 49  # discrete disc of radius 4


def test_inject_hazard_sand_and_edge() -> None:
    w = make_world(32, 32, seed=2)
    mask = w.inject_hazard((0, 0), radius=2, kind="sand")
    assert (w.true_class[mask] == SAND).all() and mask[0, 0]
    assert w.inject_hazard((5, 5), radius=0).sum() == 1


@pytest.mark.parametrize("kwargs", [
    dict(rc=(5, 5), radius=2, kind="lava"),
    dict(rc=(100, 5), radius=2, kind="sand"),
    dict(rc=(-1, 5), radius=2, kind="sand"),
    dict(rc=(5, 5), radius=-1, kind="sand"),
])
def test_inject_hazard_bad_args(kwargs: dict) -> None:
    w = make_world(32, 32, seed=2)
    with pytest.raises(ValueError):
        w.inject_hazard(**kwargs)
