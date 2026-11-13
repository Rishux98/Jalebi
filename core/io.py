"""Quick-preview helpers: hillshade and headless PNG export."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def hillshade(dem: np.ndarray, cell_size: float = 1.0, azimuth: float = 315.0,
              altitude: float = 45.0) -> np.ndarray:
    """Lambertian hillshade of a DEM.

    Args:
        dem: (H, W) elevation in metres.
        cell_size: metres per cell.
        azimuth: sun azimuth in degrees clockwise from north (up = row 0).
        altitude: sun elevation above horizon in degrees.

    Returns:
        (H, W) float32 illumination in [0, 1].
    """
    dz_dr, dz_dc = np.gradient(np.asarray(dem, dtype=np.float32), cell_size)
    slope = np.arctan(np.hypot(dz_dr, dz_dc))
    # aspect: direction of steepest descent, measured clockwise from north.
    # north = -row direction; east = +col direction.
    aspect = np.arctan2(-dz_dc, dz_dr)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def save_png(array: np.ndarray, path: str | Path, cmap: str = "viridis", title: str | None = None,
             vmin: float | None = None, vmax: float | None = None, colorbar: bool = True,
             dpi: int = 120) -> Path:
    """Save a 2-D array as a PNG heatmap (headless). Non-finite cells are drawn grey.

    Args:
        array: (H, W) array; row 0 at the top (x = col, y = row).
        path: output file; parent dirs are created.
        cmap: matplotlib colormap name.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(array, dtype=np.float32)
    masked = np.ma.masked_invalid(a)
    cm = plt.get_cmap(cmap).with_extremes(bad="#808080")
    fig, ax = plt.subplots(figsize=(6, 6 * a.shape[0] / max(a.shape[1], 1) + 0.4))
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


if __name__ == "__main__":
    yy, xx = np.mgrid[:100, :100].astype(np.float32)
    demo = 10 * np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / 400)
    hs = hillshade(demo, 1.0)
    out = save_png(hs, Path("outputs") / "io_demo_hillshade.png", cmap="gray", title="hillshade demo")
    print(f"hillshade range [{hs.min():.2f}, {hs.max():.2f}] -> {out}")
