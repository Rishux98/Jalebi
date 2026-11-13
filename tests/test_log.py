"""Tests for explain/log.py — evidence contract, templates, export round-trips."""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.types import Belief, LogEntry
from explain.log import (REASONS, DecisionLog, build_reason, confidence_at, confidence_mean, make_evidence)


def _replan_ev(**over):
    ev = make_evidence((10, 12), (4, 4), 0.3, "hazard_prob_max", hazard_prob=np.float32(0.41), p_big_rock=0.78,
                       p_sand=0.02, slope_deg=np.float64(6.5), old_cost=120.0, new_cost=float("inf"),
                       cost_increase_pct=float("inf"), old_success_prob=0.12, new_success_prob=0.71,
                       old_length_m=80.0, new_length_m=92.0)
    ev.update(over)
    return ev


def test_make_evidence_plain_and_required():
    ev = _replan_ev()
    assert ev["cell"] == [10, 12] and ev["rover_rc"] == [4, 4]
    assert type(ev["hazard_prob"]) is float and type(ev["slope_deg"]) is float
    json.dumps(ev)
    with pytest.raises(TypeError):
        make_evidence((0, 0), (0, 0), 1.0, "x", nested={"a": 1})


def test_append_validates():
    log = DecisionLog()
    with pytest.raises(ValueError):
        log.append(LogEntry(0, "e", "replan", evidence={"cell": [0, 0]}))
    with pytest.raises(TypeError):                     # numpy scalar is not JSON-serialisable
        log.append(LogEntry(0, "e", "replan", evidence={**_replan_ev(), "bad": np.float32(1.0)}))


def test_reason_built_from_numbers():
    log = DecisionLog()
    e = log.record(5, "hazard_on_path", "replan", _replan_ev(), confidence=0.83)
    assert "hazard_prob=0.41 > 0.30" in e.reason and "P(big_rock)=0.78" in e.reason
    assert "12.0%" in e.reason and "71.0%" in e.reason
    assert build_reason("replan", _replan_ev(hazard_prob=0.55)).count("0.55") == 1


def test_json_round_trip_and_lookup(tmp_path):
    log = DecisionLog()
    log.record(1, "hazard_on_path", "replan", _replan_ev(), 0.8)
    log.record(2, "near_threshold", "keep_path",
               make_evidence((3, 3), (2, 2), 0.3, "hazard_prob_max", check_value=0.2, check_ratio=0.67), 0.5)
    p = log.to_json(tmp_path / "log.json")
    back = DecisionLog.from_json(p)
    assert [e.to_dict() for e in back] == [e.to_dict() for e in log]   # inf survives
    assert len(back.entries_for_cell(10, 12)) == 1 and back.entries_for_cell(9, 9) == []
    assert back.counts() == {"replan": 1, "keep_path": 1}
    df = back.to_dataframe()
    assert {"step", "decision", "reason", "ev_cell", "ev_hazard_prob"} <= set(df.columns)


def test_confidence_bounds():
    u = Belief.uniform(4, 4)
    assert confidence_at(u, (1, 1)) == pytest.approx(0.0, abs=1e-5)
    one = Belief.one_hot(np.zeros((4, 4), np.int8))
    assert confidence_at(one, (2, 2)) == pytest.approx(1.0, abs=1e-5)
    assert confidence_mean(one, [(0, 0), (1, 1)]) == pytest.approx(1.0, abs=1e-5)
    assert confidence_mean(one, []) == 0.0


def test_every_decision_has_a_template():
    for d in ("plan", "replan", "keep_path", "hazard_detected", "select_target", "skip_target",
              "continue", "goal_reached", "stop"):
        assert d in REASONS
