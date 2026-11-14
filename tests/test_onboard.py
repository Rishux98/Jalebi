"""Tests for onboard/: compact formats, tiered tile store, hierarchical planner, onboard rover."""
from __future__ import annotations

import numpy as np
import pytest

from belief.bayes import init_prior
from core.config import DEFAULT
from core.types import Belief
from onboard.planner import CompactCostMap, hierarchical_plan
from onboard.quantize import (COST_INF, QuantizedDEM, dequantize_cost, dequantize_prob_log, dequantize_slope,
                              dequantize_unit, pack_classes, quantize_cost, quantize_prob_log, quantize_slope,
                              quantize_unit, report_error, unpack_classes)
from onboard.rover import OnboardRover
from onboard.tiered_store import TieredTileStore, build_store, decode_tile, encode_tile
from perception.sensor_model import FALLBACK_CONFUSION
from terrain.features import compute_layers
from world.synthetic import make_world


@pytest.fixture(scope="module")
def site():
    w = make_world(128, 128, seed=3)
    return w, compute_layers(w)


def _store(w, L, **kw):
    kw.setdefault("prefetch_sync", True)
    kw.setdefault("usage_path", None)
    kw.setdefault("simulate_latency", False)
    return build_store(w, L, **kw)


# ----------------------------------------------------------------- formats
def test_int16_round_trip_within_half_scale(site):
    w, _ = site
    q = QuantizedDEM.from_float(w.dem, 0.01)
    assert q.data.dtype == np.int16
    assert np.abs(q.to_float() - w.dem).max() <= q.scale / 2 + 1e-6
    win = q.to_float_window(10, 30, 5, 25)
    assert win.shape == (20, 20) and np.allclose(win, q.to_float()[10:30, 5:25])


def test_int16_auto_widens_for_large_relief():
    dem = np.linspace(0, 2000, 400, dtype=np.float32).reshape(20, 20)   # 2 km relief > 655 m at 1 cm
    q = QuantizedDEM.from_float(dem, 0.01)
    assert q.scale > 0.01 and np.abs(q.to_float() - dem).max() <= q.scale / 2 + 1e-4
    assert report_error(dem, 1.0)["auto_widened"] is True


def test_uint8_layers_and_cost_inf_round_trip():
    a = np.random.default_rng(0).random((16, 16))
    assert np.abs(dequantize_unit(quantize_unit(a)) - a).max() <= 1 / 510 + 1e-7
    s = np.array([0.0, 10.24, 24.9, 25.1, 80.0])
    assert np.abs(dequantize_slope(quantize_slope(s)) - s).max() <= 0.25 + 1e-6
    cost = np.array([[0.0, 1.3, np.inf], [2.0, np.inf, 0.4]], np.float32)
    q, cmax = quantize_cost(cost)
    assert q.dtype == np.uint8 and (q == COST_INF).sum() == 2 and q[np.isfinite(cost)].max() <= 254
    back = dequantize_cost(q, cmax)
    assert np.array_equal(np.isinf(back), np.isinf(cost))
    assert np.abs(back[np.isfinite(cost)] - cost[np.isfinite(cost)]).max() <= cmax / 508 + 1e-6


def test_hazard_log_scale_keeps_small_probabilities():
    p = np.array([0.0, 1e-4, 1e-3, 0.0021, 0.05, 0.37, 1.0])
    back = dequantize_prob_log(quantize_prob_log(p))
    assert back[0] == 0.0
    assert np.all(np.abs(back[1:] - p[1:]) / p[1:] < 0.03)            # <= ~2.8 % relative
    assert quantize_unit(np.array([0.001]))[0] == 0                   # why linear uint8 was not enough


def test_packed_classes_exact(site):
    w, _ = site
    for cls in (w.true_class, np.random.default_rng(1).integers(0, 4, (13, 7))):
        assert np.array_equal(unpack_classes(pack_classes(cls), cls.shape), cls)
    assert pack_classes(w.true_class).nbytes == -(-w.true_class.size // 4)


# ----------------------------------------------------------------- tiles
def test_tile_decode_equals_original_crop(site):
    w, L = site
    q = QuantizedDEM.from_float(w.dem)
    crop = {"dem": q.data[:64, 64:128], "slope": quantize_slope(L.slope_deg[:64, 64:128]),
            "rough": quantize_unit(np.clip(L.roughness[:64, 64:128] / DEFAULT.rough_ref_m, 0, 1))}
    out = decode_tile(encode_tile(crop))
    for k in crop:
        assert np.array_equal(out[k], crop[k]) and out[k].dtype == crop[k].dtype
    st = _store(w, L)
    win = st.get_layers_window(0, 128, 0, 128)
    assert np.abs(win["dem"] - w.dem).max() <= st.qdem.scale / 2 + 1e-5
    assert st.flash_bytes < 3 * w.dem.nbytes                            # compressed below float32
    st.close()


def test_coarse_map_keeps_max_slope(site):
    w, L = site
    st = _store(w, L)
    f = 4
    blocks = L.slope_deg[:128, :128].reshape(32, f, 32, f).max(axis=(1, 3))
    assert np.allclose(st.coarse_slope_max[:32, :32], blocks)           # hazards never averaged away
    st.close()


def test_eviction_never_evicts_rover_tile(site):
    w, L = site
    st = _store(w, L, tile=16, max_hot=2, max_warm=2)                    # 8x8 tiles, tiny budgets
    rover = (40, 40)
    st.update_rover(rover, [], reach=0)
    rt = st.tile_of(rover)
    for r in range(0, 128, 16):                                        # touch every tile
        for c in range(0, 128, 16):
            st.get_layers_window(r, r + 1, c, c + 1)
            assert rt in st.hot, (r, c)
    assert st.counters["evictions_hot"] > 0 and len(st.hot) <= 2
    st.close()


def test_straight_traverse_with_sync_prefetch_has_no_stalls(site):
    w, L = site
    st = _store(w, L, tile=32, max_hot=12)
    path = [(64, c) for c in range(2, 126)]
    for i, rc in enumerate(path):
        st.update_rover(rc, path[i:], reach=6)
        st.get_window(rc, 6)
    s = st.stats()
    assert s["stalls"] == 0 and s["hit_rate"] == 1.0 and s["prefetch_hits"] > 0
    st.close()


def test_usage_history_persists_and_warms_next_run(site, tmp_path):
    w, L = site
    path = tmp_path / "usage.json"
    a = _store(w, L, usage_path=path, key="k")
    a.get_window((10, 10), 3)
    a.close()
    b = _store(w, L, usage_path=path, key="k")
    assert b.counters["boot_warm_preload"] >= 1 and b.access.sum() > 0
    b.close()


def test_async_prefetch_thread_safe(site):
    w, L = site
    st = _store(w, L, prefetch_sync=False, tile=32, max_hot=6)
    path = [(r, r) for r in range(2, 126)]
    for i, rc in enumerate(path):
        st.update_rover(rc, path[i:], reach=6)
        st.get_window(rc, 6)
    st.wait_prefetch()
    s = st.stats()
    assert s["requests"] == s["hits_hot"] + s["hits_warm"] + s["misses_cold"]
    assert len(st.hot) <= 6
    st.close()


# ----------------------------------------------------------------- planner + rover
def test_hierarchical_path_never_enters_impassable(site):
    w, L = site
    st = _store(w, L)
    b = init_prior(w)
    ccm = CompactCostMap.build(st, b, DEFAULT)
    view = ccm.view()
    p = hierarchical_plan(view, w.start, w.goal, w.cell_size, DEFAULT.pareto_risk_balanced, window=32)
    assert p.found and p.path[0] == w.start and p.path[-1] == w.goal
    rows, cols = p.as_arrays()
    assert np.isfinite(view.cost[rows, cols]).all()
    for (r0, c0), (r1, c1) in zip(p.path, p.path[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1
    st.close()


def test_onboard_rover_replans_around_hidden_hazard(site):
    w, L = site
    st = _store(w, L)
    rv = OnboardRover(w, st, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION, belief=init_prior(w))
    rv.inject_hazard_on_path(0.4, 3)
    res = rv.run_to_goal()
    assert res.reached_goal and res.replans >= 1 and res.meta["entered_true_hazard_cells"] <= 2
    assert res.meta["onboard"] and res.meta["store"]["stalls"] == 0
    assert len(rv.tier_frames) == res.steps + 1
    assert all(e.evidence["cell"][0] < 128 for e in res.log)
    st.close()
