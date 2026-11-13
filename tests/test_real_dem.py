"""Tests for world/real_dem.py on a small synthetic GeoTIFF (no network), plus the shipped crop if present."""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from autonomy.loop import Rover  # noqa: E402
from core.config import DEFAULT  # noqa: E402
from core.types import N_CLASSES  # noqa: E402
from perception.sensor_model import FALLBACK_CONFUSION  # noqa: E402
from terrain.features import slope_deg  # noqa: E402
from world.real_dem import (DEFAULT_CROP, _block_mean, _fill_nearest, classes_from_terrain, ice_from_terrain,  # noqa: E402
                            load_dem, load_world)

NODATA = -3.4028226550889045e+38


@pytest.fixture(scope="module")
def tif(tmp_path_factory):
    """120x100 DTM at 2 m: gentle hummocks + a crater, with a nodata hole."""
    rr, cc = np.mgrid[:120, :100].astype(np.float32)
    z = 3000 + 2.0 * np.sin(rr / 7) * np.cos(cc / 9) - 8 * np.exp(-((rr - 60) ** 2 + (cc - 70) ** 2) / 200)
    z[10:14, 20:25] = NODATA
    p = tmp_path_factory.mktemp("dem") / "fake_dtm.tif"
    with rasterio.open(p, "w", driver="GTiff", height=120, width=100, count=1, dtype="float32",
                       transform=from_origin(0, 0, 2.0, 2.0), nodata=NODATA) as ds:
        ds.write(z.astype(np.float32), 1)
    return p


def test_load_dem_fill_and_downsample(tif):
    dem, cs = load_dem(tif, max_size=50)
    assert dem.dtype == np.float32 and max(dem.shape) <= 50
    assert cs == pytest.approx(2.0 * 3)                         # ceil(120 / 50) = 3
    assert np.isfinite(dem).all() and dem.min() > 2980          # nodata filled, not averaged in
    win, cs1 = load_dem(tif, window=(0, 0, 40, 30), max_size=300)
    assert win.shape == (40, 30) and cs1 == pytest.approx(2.0)


def test_helpers():
    a = np.array([[1.0, 0.0], [0.0, 4.0]])
    filled = _fill_nearest(a, a == 0)
    assert (filled != 0).all()
    assert _block_mean(np.arange(16.0).reshape(4, 4), 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    with pytest.raises(ValueError):
        _fill_nearest(a, np.ones_like(a, bool))


def test_derived_layers(tif):
    dem, cs = load_dem(tif, max_size=100)
    cls = classes_from_terrain(dem, cs, DEFAULT, seed=1)
    assert cls.dtype == np.int8 and cls.shape == dem.shape and cls.min() >= 0 and cls.max() < N_CLASSES
    assert np.array_equal(cls, classes_from_terrain(dem, cs, DEFAULT, seed=1))
    steep = slope_deg(dem, cs) > DEFAULT.slope_safe_deg
    assert (cls[steep] != 0).all()                              # steep cells never soil
    ice = ice_from_terrain(dem, cs)
    assert ice.dtype == np.float32 and ice.min() >= 0 and ice.max() <= 1 + 1e-6


def test_load_world_runs_through_rover(tif):
    w = load_world(tif, DEFAULT, seed=3, max_size=60)
    assert w.shape == (60, 50) and w.name.startswith("HiRISE")
    assert w.dem.min() == pytest.approx(0.0)                    # relative elevation
    rover = Rover(w, DEFAULT, "balanced", seed=3, confusion=FALLBACK_CONFUSION)
    res = rover.run_to_goal()
    assert res.reached_goal and res.meta["entered_true_hazard_cells"] == 0


@pytest.mark.skipif(not DEFAULT_CROP.exists(), reason="HiRISE crop not fetched (python -m world.real_dem)")
def test_shipped_hirise_crop():
    w = load_world()
    assert w.shape == (300, 300) and 3.9 < w.cell_size < 4.1
    assert 30 < float(np.ptp(w.dem)) < 200
