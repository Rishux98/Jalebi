"""Tests for planner/astar.py — edge cases, optimality vs Dijkstra, timing."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from core.types import CostMap, Plan
from costmap.fuse import build_costmap, one_hot_probs
from planner.astar import astar, evaluate_path, risk_term
from terrain.features import compute_layers
from world.synthetic import make_world


def _cm(cost: np.ndarray, hazard: np.ndarray | None = None) -> CostMap:
    cost = np.asarray(cost, np.float32)
    return CostMap(cost, np.zeros_like(cost) if hazard is None else hazard)


def _assert_valid(plan: Plan, cm: CostMap, start, goal) -> None:
    assert plan.path[0] == tuple(start) and plan.path[-1] == tuple(goal)
    rows, cols = plan.as_arrays()
    assert np.isfinite(cm.cost[rows, cols]).all()
    pas = np.isfinite(cm.cost)
    for (r0, c0), (r1, c1) in zip(plan.path, plan.path[1:]):
        dr, dc = r1 - r0, c1 - c0
        assert max(abs(dr), abs(dc)) == 1
        if dr and dc:  # diagonal: not squeezing between two blocked cells
            assert pas[r0, c1] or pas[r1, c0]


@pytest.fixture(scope="module")
def world_cm():
    w = make_world(200, 200, seed=42)
    return w, build_costmap(compute_layers(w), one_hot_probs(w.true_class))


# --------------------------------------------------------------------------- edge cases
def test_start_equals_goal() -> None:
    hz = np.full((5, 5), 0.1, np.float32)
    p = astar(_cm(np.ones((5, 5)), hz), (2, 2), (2, 2))
    assert p.path == [(2, 2)] and p.total_cost == 0.0 and p.length_m == 0.0
    assert p.success_prob == pytest.approx(0.9)


def test_start_or_goal_impassable() -> None:
    c = np.zeros((5, 5))
    c[0, 0] = np.inf
    for s, g in (((0, 0), (4, 4)), ((4, 4), (0, 0))):
        p = astar(_cm(c), s, g)
        assert p.path == [] and p.expanded_nodes == 0 and p.success_prob == 0.0 and not p.found


@pytest.mark.parametrize("s,g", [((-1, 0), (2, 2)), ((0, 0), (5, 2)), ((0, 9), (1, 1))])
def test_out_of_bounds_raises(s, g) -> None:
    with pytest.raises(ValueError, match=r"out of bounds"):
        astar(_cm(np.zeros((5, 5))), s, g)


def test_unreachable_goal() -> None:
    c = np.zeros((7, 7))
    c[:, 3] = np.inf
    p = astar(_cm(c), (0, 0), (6, 6))
    assert p.path == [] and p.success_prob == 0.0
    assert p.expanded_nodes == 21  # the whole left 7x3 component got expanded


def test_all_zero_grid_length_is_octile() -> None:
    p = astar(_cm(np.zeros((30, 40))), (2, 3), (25, 38))
    dr, dc = 23, 35
    octile = max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
    assert p.length_m == pytest.approx(octile, rel=1e-9)
    assert p.total_cost == pytest.approx(octile, rel=1e-9)
    p2 = astar(_cm(np.zeros((30, 40))), (2, 3), (25, 38), cell_size=2.5)
    assert p2.length_m == pytest.approx(2.5 * octile, rel=1e-9)


def test_no_corner_cutting() -> None:
    c = np.zeros((3, 3))
    c[0, 1] = c[1, 0] = np.inf
    cm = _cm(c)
    p = astar(cm, (0, 0), (1, 1))
    assert p.path == []  # the only move would squeeze diagonally between two blocked cells
    c2 = np.zeros((3, 3))
    c2[0, 1] = np.inf                                # one blocked neighbour -> diagonal allowed
    assert astar(_cm(c2), (0, 0), (1, 1)).path == [(0, 0), (1, 1)]


def test_avoids_inf_and_detours() -> None:
    c = np.zeros((20, 20))
    c[2:18, 10] = np.inf
    cm = _cm(c)
    p = astar(cm, (10, 2), (10, 17))
    _assert_valid(p, cm, (10, 2), (10, 17))
    assert p.length_m > 15


# --------------------------------------------------------------------------- optimality
def _dijkstra_cost(cost: np.ndarray, risk: np.ndarray, s, g) -> float:
    """Reference shortest path with identical move rules, via scipy's Dijkstra."""
    h, w = cost.shape
    pas = np.isfinite(cost)
    rows, cols, vals = [], [], []
    for r in range(h):
        for c in range(w):
            if not pas[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (dr, dc) == (0, 0) or not (0 <= nr < h and 0 <= nc < w) or not pas[nr, nc]:
                        continue
                    if dr and dc and not pas[r, nc] and not pas[nr, c]:
                        continue
                    d = math.sqrt(2) if dr and dc else 1.0
                    rows.append(r * w + c)
                    cols.append(nr * w + nc)
                    vals.append(d * (1 + float(cost[nr, nc])) + float(risk[nr, nc]))
    graph = csr_matrix((vals, (rows, cols)), shape=(h * w, h * w))
    return float(dijkstra(graph, indices=s[0] * w + s[1])[g[0] * w + g[1]])


@pytest.mark.parametrize("seed,risk_weight", [(0, 0.0), (1, 0.0), (2, 5.0), (3, 20.0)])
def test_optimal_vs_dijkstra(seed: int, risk_weight: float) -> None:
    rng = np.random.default_rng(seed)
    cost = rng.uniform(0.2, 3.0, (18, 18)).astype(np.float32)
    cost[rng.random((18, 18)) < 0.2] = np.inf
    cost[0, 0] = cost[17, 17] = 0.5
    hz = rng.uniform(0, 0.2, (18, 18)).astype(np.float32)
    cm = CostMap(cost, hz)
    p = astar(cm, (0, 0), (17, 17), risk_weight=risk_weight)
    ref = _dijkstra_cost(cost.astype(np.float64), risk_weight * risk_term(hz), (0, 0), (17, 17))
    if np.isinf(ref):
        assert p.path == []
    else:
        _assert_valid(p, cm, (0, 0), (17, 17))
        assert p.total_cost == pytest.approx(ref, rel=1e-6)


def test_risk_weight_never_lowers_success(world_cm) -> None:
    w, cm = world_cm
    # stress hazard so the trade-off is real: add a hazard band across the map
    hz = cm.hazard_prob.copy()
    hz[90:110, :] = np.maximum(hz[90:110, :], 0.02)
    cm2 = CostMap(cm.cost, hz, cm.components, cm.energy_wh_per_m)
    p0 = astar(cm2, w.start, w.goal, risk_weight=0.0)
    p1 = astar(cm2, w.start, w.goal, risk_weight=50.0)
    assert p1.success_prob >= p0.success_prob - 1e-9


# --------------------------------------------------------------------------- stats & perf
def test_evaluate_path_matches_astar(world_cm) -> None:
    w, cm = world_cm
    p = astar(cm, w.start, w.goal, w.cell_size, risk_weight=3.0)
    _assert_valid(p, cm, w.start, w.goal)
    assert p.total_cost == pytest.approx(p.meta["g_goal"], rel=1e-6)
    rows, cols = p.as_arrays()
    assert p.success_prob == pytest.approx(float(np.prod(1 - cm.hazard_prob[rows, cols].astype(np.float64))))
    assert p.energy_wh > p.length_m * 1.0  # >= flat-ground Wh/m
    e = evaluate_path(cm, p.path, w.cell_size, 3.0)
    assert e.total_cost == pytest.approx(p.total_cost)
    with pytest.raises(ValueError):
        evaluate_path(cm, [(0, 0), (0, 2)])
    assert evaluate_path(cm, []).path == []


def test_200x200_under_2s(world_cm) -> None:
    w, cm = world_cm
    t0 = time.perf_counter()
    p = astar(cm, w.start, w.goal, w.cell_size, risk_weight=10.0)
    assert time.perf_counter() - t0 < 2.0 and p.found
    # worst-ish case: noisy random costs, corner to corner
    rng = np.random.default_rng(0)
    c = rng.uniform(0.0, 4.0, (200, 200)).astype(np.float32)
    c[rng.random((200, 200)) < 0.15] = np.inf
    c[0, 0] = c[199, 199] = 0.0
    t0 = time.perf_counter()
    astar(CostMap(c, np.zeros_like(c)), (0, 0), (199, 199), risk_weight=1.0)
    assert time.perf_counter() - t0 < 2.0
