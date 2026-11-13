"""Tests for terrain/features.py."""
from __future__ import annotations

import numpy as np
import pytest

from core.config import DEFAULT
from core.types import TerrainLayers
from terrain.features import compute_layers, curvature, p_slope_exceed, roughness, slope_deg
from world.synthetic import make_world


def _plane(deg: float, n: int = 32, cell: float = 1.0) -> np.ndarray:
    rr, _ = np.mgrid[:n, :n].astype(np.float32)
    return (np.tan(np.radians(deg)) * rr * cell).astype(np.float32)


def test_slope_of_planes() -> None:
    for deg in (0.0, 10.0, 30.0, 45.0):
        s = slope_deg(_plane(deg), 1.0)
        assert s.dtype == np.float32 and s.shape == (32, 32)
        np.testing.assert_allclose(s, deg, atol=1e-3)
    # same physical plane at 2 m cells
    np.testing.assert_allclose(slope_deg(_plane(20.0, cell=2.0), 2.0), 20.0, atol=1e-3)


def test_roughness_plane_zero_noise_positive() -> None:
    k = 5
    r_plane = roughness(_plane(30.0), k)
    assert r_plane.dtype == np.float32
    assert np.abs(r_plane[k:-k, k:-k]).max() < 1e-4          # tilted plane is smooth
    noisy = _plane(30.0) + np.random.default_rng(0).normal(0, 0.2, (32, 32)).astype(np.float32)
    rn = roughness(noisy, k)[k:-k, k:-k]
    assert 0.1 < float(np.median(rn)) < 0.3                  # ~ noise std, reduced by fit dof
    with pytest.raises(ValueError):
        roughness(noisy, 4)


def test_curvature_sign() -> None:
    rr, cc = np.mgrid[:21, :21].astype(np.float32)
    dome = -((rr - 10) ** 2 + (cc - 10) ** 2) * 0.01
    cv = curvature(dome, 1.0)
    assert cv.dtype == np.float32
    np.testing.assert_allclose(cv[3:-3, 3:-3], 0.04, atol=1e-5)  # -laplace(-0.01 r^2) = +0.04
    assert (curvature(-dome, 1.0)[3:-3, 3:-3] < 0).all()         # bowl is concave


def test_p_slope_exceed_limits_and_uncertain_band() -> None:
    flat = p_slope_exceed(_plane(0.0), sigma=0.05, limit=25.0, n=20, seed=1)
    steep = p_slope_exceed(_plane(40.0), sigma=0.05, limit=25.0, n=20, seed=1)
    assert flat.dtype == np.float32 and flat.max() == 0.0
    assert steep.min() == 1.0
    edge = p_slope_exceed(_plane(25.0), sigma=0.3, limit=25.0, n=40, seed=1, corr_cells=3.0)
    assert 0.2 < float(edge.mean()) < 0.8                    # right at the limit -> uncertain
    exact = p_slope_exceed(_plane(10.0), sigma=0.0, limit=25.0)
    assert exact.max() == 0.0


def test_p_slope_exceed_deterministic_and_batch_invariant() -> None:
    dem = _plane(24.0)
    a = p_slope_exceed(dem, sigma=0.3, n=12, seed=7)
    b = p_slope_exceed(dem, sigma=0.3, n=12, seed=7)
    np.testing.assert_array_equal(a, b)
    c = p_slope_exceed(dem, sigma=0.3, n=12, seed=8)
    assert not np.array_equal(a, c)
    tiny = p_slope_exceed(dem, sigma=0.3, n=12, seed=7, max_bytes=1)  # batch of 1
    np.testing.assert_allclose(a, tiny, atol=1e-6)


def test_compute_layers_world() -> None:
    w = make_world(48, 48, seed=4)
    L = compute_layers(w)
    assert isinstance(L, TerrainLayers)
    for name in ("slope_deg", "roughness", "curvature", "p_slope_exceed"):
        a = getattr(L, name)
        assert a.shape == (48, 48) and a.dtype == np.float32 and np.isfinite(a).all()
    assert 0.0 <= L.p_slope_exceed.min() and L.p_slope_exceed.max() <= 1.0
    # cells that are deterministically very steep are almost surely over the limit
    very = L.slope_deg > DEFAULT.slope_max_deg + 15
    if very.any():
        assert L.p_slope_exceed[very].mean() > 0.9
