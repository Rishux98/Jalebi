"""Tests for autonomy/loop.py — hidden-hazard replanning, safety, logging, determinism."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from autonomy.loop import Rover
from core.config import DEFAULT
from core.types import Target
from perception.sensor_model import FALLBACK_CONFUSION
from terrain.features import compute_layers
from world.synthetic import make_world

N = 64


def _run(seed: int = 3, inject: bool = True):
    w = make_world(N, N, seed=seed)
    rover = Rover(w, DEFAULT, "balanced", seed=seed, confusion=FALLBACK_CONFUSION)
    hz = rover.inject_hazard_on_path(0.4, 3, "big_rock") if inject else None
    return w, rover, hz, rover.run_to_goal()


@pytest.fixture(scope="module")
def run():
    return _run()


def test_injected_hazard_triggers_logged_replan(run):
    w, rover, hz, res = run
    assert res.reached_goal
    replans = res.log.by_decision("replan")
    assert replans
    for e in replans:
        ev = e.evidence
        assert (ev["hazard_prob"] > ev["threshold"] or not math.isfinite(ev["new_cost"])
                or ev["cost_increase_pct"] > ev["threshold"])
        assert f"P(big_rock)={ev['p_big_rock']:.2f}" in e.reason
        assert ev["new_success_prob"] >= 0 and ev["old_length_m"] > 0
    # at least one replan is triggered inside the planted field (others may be natural boulders orbit missed)
    assert any(max(abs(e.evidence["cell"][0] - hz[0]), abs(e.evidence["cell"][1] - hz[1])) <= 3 + 1
               for e in replans)


def test_safety_and_trajectory(run):
    w, rover, hz, res = run
    assert res.meta["entered_true_hazard_cells"] == 0
    tr = res.trajectory
    assert tr[0] == w.start and tr[-1] == w.goal and len(tr) == res.steps + 1
    for (r0, c0), (r1, c1) in zip(tr, tr[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1
    assert res.meta["injected"][0]["rc"] == list(hz)


def test_log_contract(run):
    w, rover, hz, res = run
    assert res.log[0].decision == "plan" and res.log[-1].decision == "goal_reached"
    for e in res.log:
        r, c = e.evidence["cell"]
        assert 0 <= r < N and 0 <= c < N
        assert 0.0 <= e.confidence <= 1.0
        assert {"rover_rc", "threshold", "threshold_name"} <= set(e.evidence)
        json.dumps(e.evidence)


def test_frames_and_snapshots(run):
    w, rover, hz, res = run
    assert len(res.frames) == res.steps + 1
    assert res.frames[0]["step"] == 0 and res.frames[-1]["rover_rc"] == w.goal
    assert 0 in res.entropy_snapshots and res.steps in res.entropy_snapshots
    assert res.entropy_snapshots[0].dtype == np.float32
    assert res.entropy_snapshots[res.steps].mean() < res.entropy_snapshots[0].mean()


def test_never_enters_known_impassable_cell():
    w = make_world(N, N, seed=3)
    rover = Rover(w, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION)
    rover.inject_hazard_on_path(0.4, 3)
    while not rover.done:
        nxt = rover.path[rover.path_idx + 1]
        assert np.isfinite(rover.costmap.cost[nxt]), f"step {rover.step_idx}: next cell {nxt} known impassable"
        rover.step()
    assert rover.reached_goal


def test_deterministic():
    a = _run()[3]
    b = _run()[3]
    assert [e.reason for e in a.log] == [e.reason for e in b.log]
    assert a.trajectory == b.trajectory


def test_blocked_goal_stops_safely():
    w = make_world(N, N, seed=3)
    rover = Rover(w, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION)
    rover.inject_hazard(w.goal, 3, "big_rock")         # goal itself is now a hidden boulder field
    res = rover.run_to_goal()
    assert not res.reached_goal
    assert res.log[-1].decision == "stop" and res.log[-1].evidence["threshold_name"] == "reachable"
    assert res.meta["entered_true_hazard_cells"] == 0


def test_targets_visited_in_order():
    w = make_world(N, N, seed=3)
    L = compute_layers(w)
    mid = w.start[0] + 10, w.start[1] + 10
    rover = Rover(w, DEFAULT, "science", seed=3, confusion=FALLBACK_CONFUSION, layers=L,
                  targets=[Target(id=7, rc=mid, value_mean=0.5, value_std=0.05)])
    res = rover.run_to_goal()
    assert res.reached_goal
    cont = res.log.by_decision("continue")
    assert len(cont) == 1 and cont[0].evidence["target_id"] == 7 and mid in res.trajectory


def test_budget_drops_lowest_value_target():
    w = make_world(N, N, seed=3)
    a = (w.start[0] + 10, w.start[1] + 10)
    b = (w.start[0] + 20, w.start[1] + 5)
    ts = [Target(id=1, rc=a, value_mean=0.6, value_std=0.05), Target(id=2, rc=b, value_mean=0.2, value_std=0.05)]
    probe = Rover(w, DEFAULT, "science", seed=3, confusion=FALLBACK_CONFUSION, targets=[ts[0]], budget_wh=1e9)
    budget = probe.plan.energy_wh + 1.0              # fits target 1 only
    rover = Rover(make_world(N, N, seed=3), DEFAULT, "science", seed=3, confusion=FALLBACK_CONFUSION,
                  targets=[Target(**t.__dict__) for t in ts], budget_wh=budget)
    skips = rover.log.by_decision("skip_target")
    assert skips and skips[0].evidence["target_id"] == 2 and skips[0].evidence["threshold_name"] == "budget_wh"
    assert skips[0].evidence["energy_wh"] > budget and [t.id for t in rover.targets] == [1]
    assert rover.plan.energy_wh <= budget


def test_pause_and_sense_logged_with_numbers():
    w = make_world(N, N, seed=3)
    rover = Rover(w, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION)
    rover.inject_hazard_on_path(0.4, 3)
    res = rover.run_to_goal()
    pauses = res.log.by_decision("pause_and_sense")
    assert pauses and res.meta["extra_looks"] >= len(pauses)
    for e in pauses:
        ev = e.evidence
        assert ev["threshold_name"] in ("loop_caution_p", "loop_min_looks")
        assert 1 <= ev["extra_looks"] <= DEFAULT.loop_max_extra_looks and ev["looks_after"] > ev["looks_before"]
        assert f"{ev['p_big_rock_before']:.2f} -> {ev['p_big_rock_after']:.2f}" in e.reason


def test_retreat_out_of_dead_end():
    """Rover boxed in by boulders it discovered backs up along its own track instead of stopping."""
    w = make_world(N, N, seed=3)
    rover = Rover(w, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION)
    rover.step()
    while abs(rover.rc[0] - rover.trajectory[-2][0]) + abs(rover.rc[1] - rover.trajectory[-2][1]) != 2:
        rover.step()                                   # stop right after a diagonal move
    here, prev = rover.rc, rover.trajectory[-2]
    # belief: every neighbour except the cell it came from is a certain boulder -> no forward route
    r, c = here
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nb = (r + dr, c + dc)
            if (dr or dc) and nb != prev and 0 <= nb[0] < N and 0 <= nb[1] < N:
                rover.belief.class_probs[nb] = [0.0, 0.0, 0.0, 1.0]
                rover.costmap.cost[nb] = np.inf
                rover.costmap.hazard_prob[nb] = 0.5
    rover.costmap.cost[prev] = np.inf                  # even the way back now *looks* blocked
    # A* alone cannot leave: the only free neighbour is diagonal with both orthogonals blocked (no corner cutting)
    p = rover._find_route()
    assert p is not None and p.meta["retreat_cells"] >= 1 and p.path[1] == prev
    assert p.path[0] == here and p.path[-1] == w.goal


def test_inject_needs_room():
    w = make_world(N, N, seed=3)
    rover = Rover(w, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION)
    with pytest.raises(ValueError):
        rover.inject_hazard_on_path(0.99, 3)
