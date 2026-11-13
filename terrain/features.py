"""Geometric terrain features from a DEM: slope, roughness, curvature, P(slope > limit).

All functions take an (H, W) elevation grid in metres and return (H, W) float32 arrays.
Run: python -m terrain.features   (writes outputs/terrain_layers.png)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from core.config import DEFAULT, Config
from core.types import TerrainLayers, World

_F32 = np.float32


def slope_deg(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Terrain slope in degrees.

    Central differences (``np.gradient``; one-sided at borders):
    ``slope = atan(|grad z|)`` with the gradient in m/m.
    This is THE slope used for passability everywhere (world generator, cost map).
    """
    dz_dr, dz_dc = np.gradient(np.asarray(dem, dtype=_F32), cell_size)
    return np.degrees(np.arctan(np.hypot(dz_dr, dz_dc))).astype(_F32)


def roughness(dem: np.ndarray, k: int = 5) -> np.ndarray:
    """Local roughness in metres: RMS residual of a least-squares plane fit in a k x k window.

    A tilted plane has roughness 0, so steep-but-smooth slopes don't count twice.
    Vectorised closed form: in a symmetric window the regressors 1, dr, dc are orthogonal, so
    ``RSS/n = E[z^2] - E[z]^2 - (E[dr z]^2 + E[dc z]^2) / E[dr^2]``.
    Computed in float64 to avoid cancellation, borders use ``mode="nearest"``.

    Args:
        dem: (H, W) elevation in metres.
        k: odd window size in cells (>= 3).
    """
    if k < 3 or k % 2 == 0:
        raise ValueError(f"roughness window k must be odd and >= 3, got {k}")
    z = np.asarray(dem, dtype=np.float64)
    z = z - z.mean()
    u = np.arange(k, dtype=np.float64) - k // 2
    var_u = float((u ** 2).mean())                    # E[dr^2] over the window
    n = float(k * k)
    k_r = np.repeat(u[:, None], k, axis=1) / n        # kernel weights dr / n
    k_c = k_r.T.copy()                                # kernel weights dc / n
    m_z = ndimage.uniform_filter(z, k, mode="nearest")
    m_z2 = ndimage.uniform_filter(z * z, k, mode="nearest")
    m_rz = ndimage.correlate(z, k_r, mode="nearest")
    m_cz = ndimage.correlate(z, k_c, mode="nearest")
    rss = m_z2 - m_z ** 2 - (m_rz ** 2 + m_cz ** 2) / var_u
    return np.sqrt(np.clip(rss, 0.0, None)).astype(_F32)


def curvature(dem: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Curvature as the negative Laplacian of elevation, in 1/m.

    Positive = convex (ridge crests, crater rims), negative = concave (crater floors, basins).
    """
    lap = ndimage.laplace(np.asarray(dem, dtype=np.float64), mode="nearest") / (cell_size ** 2)
    return (-lap).astype(_F32)


def correlated_noise(shape: tuple[int, ...], sigma: float, corr_cells: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with std ``sigma`` and Gaussian spatial correlation (last two axes).

    White noise blurred with ``gaussian_filter(sigma=corr_cells)`` has std ``1/(2 sqrt(pi) s)``
    in 2-D; we rescale by that so the marginal std is ``sigma`` metres.
    """
    white = rng.standard_normal(shape).astype(_F32)
    if corr_cells <= 0:
        return (sigma * white).astype(_F32)
    filt_sigma = (0,) * (len(shape) - 2) + (corr_cells, corr_cells)
    blurred = ndimage.gaussian_filter(white, sigma=filt_sigma, mode="wrap")
    scale = 2.0 * np.sqrt(np.pi) * corr_cells
    return (sigma * scale * blurred).astype(_F32)


def p_slope_exceed(dem: np.ndarray, cell_size: float = 1.0, sigma: float = 0.3,
                   limit: float = 25.0, n: int = 30, seed: int = 0,
                   corr_cells: float = 3.0, max_bytes: float = 200e6) -> np.ndarray:
    """Monte-Carlo probability that the true slope exceeds ``limit`` degrees given DEM error.

    Draws ``n`` noisy DEMs ``dem + eps`` (``eps`` ~ correlated Gaussian, std ``sigma`` m),
    recomputes slope for each and returns the exceedance fraction. All samples of a batch are
    processed at once with shape (b, H, W). Batches are sized to stay under ``max_bytes``.

    Returns:
        (H, W) float32 in [0, 1]; multiples of 1/n.
    """
    dem = np.asarray(dem, dtype=_F32)
    h, w = dem.shape
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if sigma <= 0:  # no uncertainty: deterministic indicator
        return (slope_deg(dem, cell_size) > limit).astype(_F32)
    rng = np.random.default_rng(seed)
    tan2 = np.tan(np.radians(limit)) ** 2
    per_sample = h * w * 4 * 6                     # noise + noisy dem + 2 grads + temporaries
    batch = int(max(1, min(n, max_bytes // per_sample)))
    count = np.zeros((h, w), np.int32)
    done = 0
    while done < n:  # loop over batches, never cells
        b = min(batch, n - done)
        noisy = dem[None] + correlated_noise((b, h, w), sigma, corr_cells, rng)
        g_r, g_c = np.gradient(noisy, cell_size, axis=(1, 2))
        count += (g_r * g_r + g_c * g_c > tan2).sum(axis=0, dtype=np.int32)
        done += b
    return (count / float(n)).astype(_F32)


def compute_layers(world: World, cfg: Config = DEFAULT, seed: int | None = None) -> TerrainLayers:
    """All terrain layers for a world.

    Args:
        world: the site (uses ``dem`` and ``cell_size``).
        cfg: thresholds (slope limit, roughness window, DEM noise, MC samples).
        seed: Monte-Carlo seed; defaults to ``cfg.seed``.
    """
    cs = world.cell_size
    return TerrainLayers(
        slope_deg=slope_deg(world.dem, cs),
        roughness=roughness(world.dem, cfg.roughness_window),
        curvature=curvature(world.dem, cs),
        p_slope_exceed=p_slope_exceed(
            world.dem, cs, sigma=cfg.dem_noise_sigma, limit=cfg.slope_max_deg, n=cfg.mc_samples,
            seed=cfg.seed if seed is None else seed, corr_cells=cfg.dem_noise_corr_cells,
            max_bytes=cfg.mc_max_bytes),
    )


if __name__ == "__main__":
    import time
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    t0 = time.perf_counter()
    L = compute_layers(w)
    dt = time.perf_counter() - t0
    print(f"compute_layers(200x200, n={DEFAULT.mc_samples}) in {dt * 1000:.0f} ms")
    for name in ("slope_deg", "roughness", "curvature", "p_slope_exceed"):
        a = getattr(L, name)
        print(f"  {name:15s} min {a.min():8.3f}  median {np.median(a):8.3f}  max {a.max():8.3f}")
    unsure = (L.p_slope_exceed > 0.05) & (L.p_slope_exceed < 0.95)
    print(f"  cells with uncertain passability (0.05<p<0.95): {unsure.mean():.1%}")

    out = Path("outputs/terrain_layers.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    specs = [("slope_deg", "magma", "slope (deg)"), ("roughness", "viridis", "roughness (m)"),
             ("curvature", "coolwarm", "curvature (1/m), + = convex"),
             ("p_slope_exceed", "inferno", f"P(slope > {DEFAULT.slope_max_deg:.0f}°)")]
    for ax, (name, cmap, title) in zip(axes.ravel(), specs):
        a = getattr(L, name)
        if name == "curvature":
            v = float(np.percentile(np.abs(a), 99))
            im = ax.imshow(a, cmap=cmap, vmin=-v, vmax=v)
        else:
            im = ax.imshow(a, cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("saved", out)
