"""Tests for science/targets.py — value field, candidates, ranking, budgeted selection."""
from __future__ import annotations

import json

import numpy as np
import pytest

from belief.bayes import init_prior
from core.config import DEFAULT
from costmap.fuse import costmap_from_belief
from explain.log import DecisionLog
from science.targets import (detect_candidates, plan_targets, reachable_region, science_value_field,
                             select_under_budget)
from terrain.features import compute_layers
from world.synthetic import make_world

CFG = DEFAULT.replace(sci_nms_cells=9, sci_n_targets=5, sci_bootstrap_n=20)


@pytest.fixture(scope="module")
def site():
    w = make_world(64, 64, seed=3)
    L = compute_layers(w)
    b = init_prior(w)
    cm = costmap_from_belief(L, b)
    return w, L, b, cm


def test_value_field(site):
    w, L, b, _ = site
    v = science_value_field(L, b, w.ice_signal, CFG)
    assert v.shape == (64, 64) and v.dtype == np.float32
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_reachable_region():
    p = np.ones((5, 5), bool)
    p[:, 2] = False
    reg = reachable_region(p, (0, 0))
    assert reg[:, :2].all() and not reg[:, 3:].any()
    assert not reachable_region(p, (0, 2)).any()


def test_candidates(site):
    w, L, b, cm = site
    t1 = detect_candidates(w, L, b, cm, CFG, seed=1)
    t2 = detect_candidates(w, L, b, cm, CFG, seed=1)
    assert 0 < len(t1) <= CFG.sci_n_targets
    assert [(t.rc, t.value_mean, t.value_std) for t in t1] == [(t.rc, t.value_mean, t.value_std) for t in t2]
    region = reachable_region(cm.passable, w.start)
    for t in t1:
        assert region[t.rc] and t.value_std >= 0 and 0 <= t.value_mean <= 1 and t.info_gain >= 0
    vals = [t.value_mean for t in t1]
    assert vals == sorted(vals, reverse=True) and [t.id for t in t1] == list(range(len(t1)))
    for i, a in enumerate(t1):
        for c in t1[i + 1:]:
            assert max(abs(a.rc[0] - c.rc[0]), abs(a.rc[1] - c.rc[1])) > CFG.sci_nms_cells // 2


def test_plan_targets_sorted(site):
    w, L, b, cm = site
    ts = plan_targets(detect_candidates(w, L, b, cm, CFG), cm, w.start, CFG, w.cell_size)
    u = [t.utility for t in ts]
    assert u == sorted(u, reverse=True)
    assert all(t.reachable and t.plan is not None and t.plan.found for t in ts)


def test_select_respects_budget_and_logs(site):
    w, L, b, cm = site
    log = DecisionLog()
    budget = 260.0
    sel = select_under_budget(detect_candidates(w, L, b, cm, CFG), cm, b, w.start, w.goal, budget, CFG,
                              w.cell_size, log=log)
    assert sel.energy_wh <= budget + 1e-6
    assert len(sel.selected) + len(sel.skipped) == len(log)
    for e in log:
        assert 0 <= e.evidence["cell"][0] < 64 and 0 <= e.evidence["cell"][1] < 64
        assert 0.0 <= e.confidence <= 1.0
        json.dumps(e.evidence)
    for e in log.by_decision("select_target"):
        assert e.evidence["energy_wh"] <= e.evidence["budget_wh"] + 1e-6


def test_zero_budget_selects_nothing(site):
    w, L, b, cm = site
    log = DecisionLog()
    ts = detect_candidates(w, L, b, cm, CFG)
    sel = select_under_budget(ts, cm, b, w.start, w.goal, 0.0, CFG, w.cell_size, log=log)
    assert sel.selected == [] and len(sel.skipped) == len(ts)
    assert all(e.decision == "skip_target" and e.evidence["threshold_name"] == "budget_wh" for e in log)
    assert all("Wh > budget 0 Wh" in t.reason for t in sel.skipped)
