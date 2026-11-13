"""Tests for costmap/fuse.py."""
from __future__ import annotations

import numpy as np
import pytest

from core.config import DEFAULT
from core.types import BIG_ROCK, N_CLASSES, SAND, SOIL, CostMap, TerrainLayers
from costmap.fuse import COMPONENT_NAMES, build_costmap, hazard_probability, one_hot_probs
from terrain.features import compute_layers
from world.synthetic import make_world


def _layers(h: int = 6, w: int = 6, slope: float = 5.0) -> TerrainLayers:
    z = np.zeros((h, w), np.float32)
    return TerrainLayers(np.full((h, w), slope, np.float32), z + 0.05, z, z)


def test_shapes_components_sum() -> None:
    w = make_world(40, 40, seed=2)
    L = compute_layers(w)
    cm = build_costmap(L, one_hot_probs(w.true_class))
    assert isinstance(cm, CostMap)
    assert cm.cost.shape == (40, 40) and cm.cost.dtype == np.float32
    assert cm.hazard_prob.dtype == np.float32 and cm.energy_wh_per_m.dtype == np.float32
    assert set(cm.components) == set(COMPONENT_NAMES)
    fin = np.isfinite(cm.cost)
    total = sum(cm.components.values())
    np.testing.assert_allclose(cm.cost[fin], total[fin], rtol=1e-5)
    assert (cm.cost[fin] >= 0).all()
    assert 0.0 <= cm.hazard_prob.min() and cm.hazard_prob.max() <= 1.0
    # impassable exactly where slope > limit or big rock (one-hot)
    expect_inf = (L.slope_deg > DEFAULT.slope_max_deg) | (w.true_class == BIG_ROCK)
    np.testing.assert_array_equal(~fin, expect_inf)


def test_big_rock_threshold_on_soft_probs() -> None:
    L = _layers(1, 3)
    p = np.zeros((1, 3, N_CLASSES), np.float32)
    p[0, 0, [SOIL, BIG_ROCK]] = [0.5, 0.5]       # below 0.6 -> passable but expensive
    p[0, 1, [SOIL, BIG_ROCK]] = [0.3, 0.7]       # above 0.6 -> impassable
    p[0, 2, SOIL] = 1.0
    cm = build_costmap(L, p)
    assert np.isfinite(cm.cost[0, 0]) and np.isinf(cm.cost[0, 1]) and np.isfinite(cm.cost[0, 2])
    assert cm.cost[0, 0] > cm.cost[0, 2]
    assert cm.hazard_prob[0, 0] > cm.hazard_prob[0, 2]


def test_steep_cell_impassable() -> None:
    L = _layers(2, 2, slope=DEFAULT.slope_max_deg + 0.5)
    cm = build_costmap(L, one_hot_probs(np.zeros((2, 2), int)))
    assert not np.isfinite(cm.cost).any()


def test_hazard_formula() -> None:
    p_exc = np.array([[0.4]], np.float32)
    probs = np.zeros((1, 1, N_CLASSES), np.float32)
    probs[0, 0, [SOIL, SAND, BIG_ROCK]] = [0.5, 0.3, 0.2]
    c = DEFAULT
    expect = 1 - (1 - c.hazard_slope_factor * 0.4) * (1 - c.hazard_rock_factor * 0.2) \
        * (1 - c.sand_hazard_factor * 0.3)
    np.testing.assert_allclose(hazard_probability(p_exc, probs), expect, rtol=1e-5)


def test_uncertainty_term_and_lambda_override() -> None:
    L = _layers(3, 3)
    uniform = np.full((3, 3, N_CLASSES), 0.25, np.float32)
    cm = build_costmap(L, uniform)
    np.testing.assert_allclose(cm.components["uncertainty"], DEFAULT.lambda_unc, rtol=1e-5)
    cm0 = build_costmap(L, uniform, lambda_unc=0.0)
    np.testing.assert_allclose(cm0.components["uncertainty"], 0.0, atol=1e-7)
    np.testing.assert_allclose(cm.cost - cm0.cost, DEFAULT.lambda_unc, rtol=1e-5)
    # one-hot -> zero uncertainty
    oh = build_costmap(L, one_hot_probs(np.zeros((3, 3), int)))
    assert np.abs(oh.components["uncertainty"]).max() < 1e-6


def test_energy_flat_and_uphill() -> None:
    flat = build_costmap(_layers(2, 2, slope=0.0), one_hot_probs(np.zeros((2, 2), int)))
    np.testing.assert_allclose(flat.energy_wh_per_m, DEFAULT.energy_wh_per_m_flat, rtol=1e-6)
    hill = build_costmap(_layers(2, 2, slope=20.0), one_hot_probs(np.zeros((2, 2), int)))
    assert (hill.energy_wh_per_m > flat.energy_wh_per_m).all()
    assert (hill.components["energy"] > flat.components["energy"]).all()


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        build_costmap(_layers(4, 4), np.full((4, 5, N_CLASSES), 0.25, np.float32))
