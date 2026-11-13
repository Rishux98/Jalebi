"""Tests for core/io.py."""
from __future__ import annotations

import numpy as np

from core.io import hillshade, save_png


def test_hillshade_range_and_direction() -> None:
    yy, xx = np.mgrid[:41, :41].astype(np.float32)
    dem = 10 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 100)
    hs = hillshade(dem, 1.0, azimuth=315, altitude=45)
    assert hs.shape == (41, 41) and hs.dtype == np.float32
    assert 0.0 <= hs.min() and hs.max() <= 1.0
    # sun from the NW: the NW flank (upper-left) is brighter than the SE flank
    assert hs[15, 15] > hs[25, 25]
    flat = hillshade(np.zeros((5, 5)), 1.0, altitude=45)
    np.testing.assert_allclose(flat, np.sin(np.radians(45)), rtol=1e-5)


def test_save_png_handles_inf(tmp_path) -> None:
    a = np.random.default_rng(0).random((16, 20)).astype(np.float32)
    a[3, 3] = np.inf
    out = save_png(a, tmp_path / "sub" / "x.png", cmap="magma", title="t")
    assert out.exists() and out.stat().st_size > 0
