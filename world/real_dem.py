"""Real HiRISE terrain: windowed DTM read -> World.

The elevation is REAL (USGS Astrogeology HiRISE controlled DTM, cloud-optimised GeoTIFF on S3).
Terrain classes and the ice signal are NOT in a DTM, so they are derived procedurally from the
real geometry (documented assumptions, same rules as world/synthetic.py where possible):

  bedrock   slope > slope_safe_deg (15 deg)                     — exposed steep rock
  sand      local depressions (dem - large-scale trend below the ``real_sand_pct`` percentile)
            on gentle slopes (< slope_safe_deg)                   — wind-trapped fines
  big_rock  patchy clusters (75 % fill) on the roughest ``real_rock_pct`` % of cells
  soil      everything else
  ice       concave (negative curvature) AND shadowed under a low sun -> smoothed, scaled to [0, 1]

Data flow:
  fetch_crop(url, out)        remote windowed read (only the crop is downloaded) -> local GeoTIFF
  load_dem(path, window, max) windowed read, nodata -> nearest valid, block-mean downsample to <= max cells
  load_world(path, cfg, seed) DEM + derived classes/ice + connected start/goal -> World

Run: python -m world.real_dem    (fetches the default crop if missing, writes outputs/real_world_preview.png)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from core.config import DEFAULT, Config
from core.io import hillshade
from core.types import BEDROCK, BIG_ROCK, RC, SAND, SOIL, World
from terrain.features import curvature, roughness, slope_deg

_F32 = np.float32
ROOT = Path(__file__).resolve().parents[1]

# South-polar HiRISE controlled DTM (2 m/px); the default crop is 600x600 px around the tile centre.
DEFAULT_TILE_ID = "DTEPD_076532_0920_076203_0920_A01"
DEFAULT_URL = ("https://astrogeo-ard.s3-us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/"
               "ESP_076532_0920_ESP_076203_0920/DTEPD_076532_0920_076203_0920_A01.tif")
DEFAULT_CROP = ROOT / "data" / f"{DEFAULT_TILE_ID}_crop600.tif"


# =========================================================================== I/O
def fetch_crop(url: str = DEFAULT_URL, out: str | Path = DEFAULT_CROP, size: int = 600,
               center: tuple[int, int] | None = None) -> Path:
    """Read a ``size`` x ``size`` window of a remote COG (HTTP range requests) and save it as a GeoTIFF.

    Args:
        url: http(s) URL of a cloud-optimised GeoTIFF DTM.
        out: local output path.
        size: crop side in source pixels.
        center: (row, col) centre in source pixels; default = tile centre.
    """
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(f"/vsicurl/{url}") as ds:
        cr, cc = center if center is not None else (ds.height // 2, ds.width // 2)
        r0 = int(np.clip(cr - size // 2, 0, max(ds.height - size, 0)))
        c0 = int(np.clip(cc - size // 2, 0, max(ds.width - size, 0)))
        win = Window(c0, r0, min(size, ds.width), min(size, ds.height))
        a = ds.read(1, window=win)
        prof = ds.profile.copy()
        prof.update(height=a.shape[0], width=a.shape[1], transform=win_transform(win, ds.transform),
                    driver="GTiff", compress="deflate", predictor=3, tiled=False)
        for k in ("blockxsize", "blockysize", "interleave"):
            prof.pop(k, None)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(a, 1)
            dst.update_tags(source_url=url, window=f"row_off={r0},col_off={c0},h={win.height},w={win.width}")
    return out


def _fill_nearest(a: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Replace invalid cells with the nearest valid value (EDT indices; vectorised)."""
    if not invalid.any():
        return a
    if invalid.all():
        raise ValueError("DEM window has no valid elevation")
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Downsample by integer factor ``f`` with block means (edge-padded; anti-aliased)."""
    if f <= 1:
        return a
    h, w = a.shape
    hp, wp = -(-h // f) * f, -(-w // f) * f
    p = np.pad(a, ((0, hp - h), (0, wp - w)), mode="edge")
    return p.reshape(hp // f, f, wp // f, f).mean(axis=(1, 3))


def load_dem(path: str | Path = DEFAULT_CROP, window: tuple[int, int, int, int] | None = None,
             max_size: int = 300) -> tuple[np.ndarray, float]:
    """Read a DTM window, fill nodata, downsample to <= ``max_size`` cells per side.

    Args:
        path: local GeoTIFF (or ``/vsicurl/`` URL).
        window: (row_off, col_off, height, width) in source pixels; default = whole file.
        max_size: maximum output side in cells.

    Returns:
        (dem (H, W) float32 metres, cell_size metres) — cell_size from the transform x downsample factor.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        win = Window(window[1], window[0], window[3], window[2]) if window else None
        a = ds.read(1, window=win, masked=True).astype(np.float64)
        res = float(abs(ds.transform.a))
    data = np.ma.getdata(a)
    invalid = np.ma.getmaskarray(a) | ~np.isfinite(data) | (np.abs(data) > 1e20)
    dem = _fill_nearest(data, invalid)
    f = max(1, int(np.ceil(max(dem.shape) / max_size)))
    return _block_mean(dem, f).astype(_F32), res * f


# =========================================================================== derived layers (assumptions)
def classes_from_terrain(dem: np.ndarray, cell_size: float, cfg: Config = DEFAULT, seed: int = 0) -> np.ndarray:
    """(H, W) int8 terrain classes derived from real geometry (rules in module docstring)."""
    rng = np.random.default_rng(seed)
    slope = slope_deg(dem, cell_size)
    cls = np.full(dem.shape, SOIL, np.int8)
    cls[slope > cfg.slope_safe_deg] = BEDROCK
    trend = ndimage.gaussian_filter(dem.astype(np.float64), sigma=max(dem.shape) / 12, mode="nearest")
    resid = dem - trend
    sand = (resid < np.percentile(resid, cfg.real_sand_pct)) & (slope < cfg.slope_safe_deg)
    sand = ndimage.binary_opening(sand, iterations=1)                       # drop speckle
    cls[sand] = SAND
    rough = roughness(dem, cfg.roughness_window)
    seeds = rough > np.percentile(rough, 100 - cfg.real_rock_pct)
    rocks = ndimage.binary_dilation(seeds, iterations=1) & (rng.random(dem.shape) < 0.75)
    cls[rocks] = BIG_ROCK
    return cls


def ice_from_terrain(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """(H, W) float32 in [0, 1]: concave AND shadowed under a low sun (cold-trap proxy)."""
    concave = np.clip(-curvature(dem, cell_size), 0, None)
    concave = concave / (np.percentile(concave, 99) + 1e-12)
    shade = 1.0 - hillshade(dem, cell_size, azimuth=315.0, altitude=15.0)
    ice = ndimage.gaussian_filter(np.clip(concave, 0, 1) * shade, sigma=max(dem.shape) / 60, mode="nearest")
    return (ice / (ice.max() + 1e-12)).astype(_F32)


def _corner_pair(passable: np.ndarray, margin: int) -> tuple[RC, RC]:
    """Start/goal: cells of the largest 8-connected passable component closest to opposite corners."""
    lab, n = ndimage.label(passable, structure=np.ones((3, 3), bool))
    if n == 0:
        raise ValueError("no passable cells in this DEM window")
    sizes = np.bincount(lab.ravel())[1:]
    comp = np.argwhere(lab == int(np.argmax(sizes)) + 1)
    h, w = passable.shape

    def nearest(r: int, c: int) -> RC:
        k = int(np.argmin((comp[:, 0] - r) ** 2 + (comp[:, 1] - c) ** 2))
        return int(comp[k, 0]), int(comp[k, 1])

    return nearest(margin, margin), nearest(h - 1 - margin, w - 1 - margin)


def load_world(path: str | Path = DEFAULT_CROP, cfg: Config = DEFAULT, seed: int = 42,
               window: tuple[int, int, int, int] | None = None, max_size: int | None = None) -> World:
    """HiRISE DEM window -> World with derived classes/ice and connected start (NW) / goal (SE)."""
    dem, cs = load_dem(path, window, max_size or cfg.real_max_size)
    cls = classes_from_terrain(dem, cs, cfg, seed)
    passable = (slope_deg(dem, cs) < cfg.slope_max_deg) & (cls != BIG_ROCK)
    margin = max(1, int(round(cfg.gen_corner_margin_frac * min(dem.shape))))
    start, goal = _corner_pair(passable, margin)
    name = Path(str(path)).stem
    return World(dem=dem - float(dem.min()), cell_size=cs, true_class=cls, ice_signal=ice_from_terrain(dem, cs),
                 start=start, goal=goal, seed=seed, name=f"HiRISE {name}")


def default_tile_available() -> bool:
    return DEFAULT_CROP.exists()


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    if not DEFAULT_CROP.exists():
        t0 = time.perf_counter()
        print(f"fetching 600x600 crop of {DEFAULT_TILE_ID} (windowed HTTP read) ...")
        fetch_crop()
        print(f"saved {DEFAULT_CROP} ({DEFAULT_CROP.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    w = load_world()
    s = slope_deg(w.dem, w.cell_size)
    frac = np.bincount(w.true_class.ravel(), minlength=4) / w.true_class.size
    print(f"{w.name}: {w.shape} cells @ {w.cell_size:.2f} m ({w.shape[0] * w.cell_size:.0f} m side), relief "
          f"{np.ptp(w.dem):.1f} m, slope p50/p95/max {np.percentile(s, 50):.1f}/{np.percentile(s, 95):.1f}/{s.max():.1f} deg")
    print(f"classes soil/bedrock/sand/big_rock = {np.round(frac, 3)}; start {w.start} goal {w.goal}; "
          f"built in {time.perf_counter() - t0:.2f} s")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(hillshade(w.dem, w.cell_size), cmap="gray"); ax[0].set_title(f"{w.name} hillshade (real)")
    ax[1].imshow(w.true_class, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("derived classes (assumption)")
    ax[2].imshow(w.ice_signal, cmap="Blues"); ax[2].set_title("derived ice proxy (assumption)")
    for a in ax:
        a.plot(w.start[1], w.start[0], "go"); a.plot(w.goal[1], w.goal[0], "m*", ms=12); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/real_world_preview.png", dpi=90)
    print("saved outputs/real_world_preview.png")
