"""Tests for planner/pareto.py — mode plans, waypoint chaining, Pareto front."""
from __future__ import annotations

import numpy as np
import pytest

from belief.bayes import init_prior
from core.config import DEFAULT
from core.types import CostMap, Plan
from costmap.fuse import build_costmap, costmap_from_belief, one_hot_probs
from planner.pareto import (best_science_waypoint, mode_config, pareto_front, plan_modes, plan_via,
                            run_pareto)
from terrain.features import compute_layers
from world.synthetic import make_world


@pytest.fixture(scope="module")
def setup():
    w = make_world(32, 32, seed=3)
    L = compute_layers(w)
    b = init_prior(w)
    truth = build_costmap(L, one_hot_probs(w.true_class))
    return w, L, b, truth


def _plan(length: float, succ: float, mode: str = "x") -> Plan:
    return Plan(path=[(0, 0)], length_m=length, success_prob=succ, mode=mode)


def test_costmap_from_belief_matches_build(setup):
    w, L, b, _ = setup
    cm = costmap_from_belief(L, b)
    ref = build_costmap(L, b.class_probs, entropy=b.entropy)
    assert cm.cost.shape == (32, 32) and cm.cost.dtype == np.float32
    assert np.array_equal(cm.cost, ref.cost)
    assert cm.components["uncertainty"].max() > 0              # entropy now drives cost


def test_modes_found_and_ordered(setup):
    w, L, b, truth = setup
    plans = plan_modes(L, b, w.start, w.goal, DEFAULT, w.cell_size, truth=truth)
    assert set(plans) == {"shortest", "balanced", "safest"}       # no science without waypoints
    for p in plans.values():
        assert p.found and p.path[0] == w.start and p.path[-1] == w.goal
        assert 0.0 <= p.success_prob <= 1.0 and 0.0 <= p.meta["success_prob_true"] <= 1.0
        assert "path_entropy_mean" in p.meta
    assert plans["shortest"].length_m <= plans["safest"].length_m + 1e-9
    assert plans["safest"].success_prob >= plans["shortest"].success_prob - 1e-9


def test_mode_config():
    c, rw = mode_config("shortest")
    assert rw == 0.0 and c.w_class == 0.0 and c.lambda_unc == 0.0
    c, rw = mode_config("safest")
    assert rw == DEFAULT.pareto_risk_safest and c.lambda_unc == DEFAULT.pareto_lambda_unc_safest
    with pytest.raises(ValueError):
        mode_config("teleport")


def test_plan_via_chains_without_duplicates():
    cm = CostMap(np.zeros((10, 10), np.float32), np.zeros((10, 10), np.float32))
    p = plan_via(cm, (0, 0), [(0, 9), (9, 9)], (9, 0))
    assert p.found and p.path[0] == (0, 0) and p.path[-1] == (9, 0)
    assert (0, 9) in p.path and (9, 9) in p.path
    assert all(a != b for a, b in zip(p.path, p.path[1:]))      # no consecutive duplicates
    assert p.length_m == pytest.approx(27.0)


def test_plan_via_unreachable_waypoint():
    cost = np.zeros((10, 10), np.float32)
    cost[:, 5] = np.inf
    cm = CostMap(cost, np.zeros_like(cost))
    p = plan_via(cm, (0, 0), [(0, 9)], (9, 0))
    assert not p.found and p.meta["reason"] == "waypoint_unreachable"
    assert p.meta["failed_leg"] == [[0, 0], [0, 9]]


def test_pareto_front_non_dominated():
    pts = [_plan(10, 0.5, "a"), _plan(12, 0.7, "b"), _plan(11, 0.4, "c"),   # c dominated by a
           _plan(15, 0.7, "d"), _plan(20, 0.9, "e"), Plan(path=[], mode="nf")]
    front = pareto_front(pts)
    assert [p.mode for p in front] == ["a", "b", "e"]
    assert pareto_front([]) == []


def test_run_pareto_and_science(setup):
    w, L, b, truth = setup
    ref = costmap_from_belief(L, b)
    wp = best_science_waypoint(w.ice_signal, ref)
    assert ref.passable[wp]
    res = run_pareto(L, b, w.start, w.goal, DEFAULT, w.cell_size, [wp], truth)
    assert "science" in res.modes and wp in res.modes["science"].path
    assert len(res.sweep) == int(DEFAULT.pareto_sweep[2])
    lengths = [p.length_m for p in res.front]
    succ = [p.success_prob for p in res.front]
    assert lengths == sorted(lengths) and succ == sorted(succ)      # front is monotone
    assert {r["mode"] for r in res.table()} == {"shortest", "balanced", "safest", "science"}
