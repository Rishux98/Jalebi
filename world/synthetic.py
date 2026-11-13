"""Procedural Mars-like test site: DEM + ground-truth terrain classes + ice signal.

Pipeline (all vectorised, ~tens of ms for 200x200):
  1. fractal value noise base terrain (multi-octave cubic-interpolated random lattices)
  2. low basin (broad Gaussian depression) -> collects the sand field
  3. 6-10 craters: Gaussian bowl + raised Gaussian ring rim
  4. one steep, finite ridge segment roughly across the start->goal diagonal (taper at both ends
     so the map stays connected; the planner has to go around)
  5. classes: soil default; bedrock where slope > slope_safe; sand in the lower part of the basin;
     big_rock clusters seeded on crater rims
  6. ice_signal: Gaussian blobs on the floors of the 2-3 most shadowed craters (low-sun hillshade)
  7. start/goal near opposite corners, snapped to the nearest passable cell; regenerate with
     seed+1 until a 4-connected passable path joins them (4-connectivity is stricter than the
     planner's 8-connected/no-corner-cutting moves, so a path is guaranteed for A*).

Run: python -m world.synthetic   (writes outputs/world_preview.png)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from core.config import DEFAULT, Config
from core.io import hillshade
from core.types import BEDROCK, BIG_ROCK, CLASS_NAMES, N_CLASSES, RC, SAND, SOIL, World
from terrain.features import slope_deg  # canonical slope (shared with the cost map)

_F32 = np.float32


@dataclass
class _Crater:
    r: float      # centre row (cells)
    c: float      # centre col (cells)
    radius: float  # rim radius (cells)
    depth: float  # m


# --------------------------------------------------------------------------- primitives
def fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int, base_cells: int,
                  persistence: float) -> np.ndarray:
    """Multi-octave value noise in roughly [-1, 1], (H, W) float32.

    Each octave draws an (n+1, n+1) Gaussian lattice (n = base_cells * 2**o) and cubic-interpolates
    it to the full grid; amplitudes decay by ``persistence`` per octave.
    """
    out = np.zeros((h, w), _F32)
    amp, total = 1.0, 0.0
    for o in range(octaves):  # loop over octaves, not cells
        n = base_cells * (2 ** o)
        lattice = rng.standard_normal((n + 1, n + 1)).astype(_F32)
        rr = np.linspace(0, n, h, dtype=_F32)
        cc = np.linspace(0, n, w, dtype=_F32)
        coords = np.stack(np.meshgrid(rr, cc, indexing="ij"))
        out += amp * ndimage.map_coordinates(lattice, coords, order=3, mode="nearest")
        total += amp
        amp *= persistence
    return (out / total).astype(_F32)


def passable_mask(dem: np.ndarray, true_class: np.ndarray, cell_size: float,
                  cfg: Config = DEFAULT) -> np.ndarray:
    """(H, W) bool: slope < slope_max_deg and not big_rock (ground-truth traversability)."""
    return (slope_deg(dem, cell_size) < cfg.slope_max_deg) & (true_class != BIG_ROCK)


def bfs_reachable(mask: np.ndarray, start: RC, goal: RC) -> bool:
    """Plain 4-connected BFS over True cells of ``mask``. Returns True if goal reachable from start.

    Array-backed (flat indices, bool visited grid). Used as an independent cross-check of the
    connected-component test.
    """
    h, w = mask.shape
    sr, sc = start
    gr, gc = goal
    if not (mask[sr, sc] and mask[gr, gc]):
        return False
    flat = mask.ravel()
    visited = np.zeros(h * w, dtype=bool)
    s, g = sr * w + sc, gr * w + gc
    visited[s] = True
    q: deque[int] = deque([s])
    while q:
        i = q.popleft()
        if i == g:
            return True
        r, c = divmod(i, w)
        for j, ok in ((i - w, r > 0), (i + w, r < h - 1), (i - 1, c > 0), (i + 1, c < w - 1)):
            if ok and flat[j] and not visited[j]:
                visited[j] = True
                q.append(j)
    return False


def _connected(mask: np.ndarray, a: RC, b: RC) -> bool:
    """Vectorised 4-connected component check (scipy label)."""
    labels, _ = ndimage.label(mask)
    la, lb = labels[a], labels[b]
    return bool(la != 0 and la == lb)


def _snap_to_passable(mask: np.ndarray, rc: RC) -> RC | None:
    """Nearest True cell of ``mask`` to ``rc`` (Euclidean), or None if mask is empty."""
    if not mask.any():
        return None
    if mask[rc]:
        return rc
    _, (ir, ic) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return int(ir[rc]), int(ic[rc])


# --------------------------------------------------------------------------- generator
def _place_craters(h: int, w: int, rng: np.random.Generator, cfg: Config,
                   avoid: list[tuple[float, float]]) -> list[_Crater]:
    m = min(h, w)
    n = int(rng.integers(cfg.gen_n_craters[0], cfg.gen_n_craters[1] + 1))
    craters: list[_Crater] = []
    for _ in range(n):
        for _try in range(30):  # rejection sampling per crater
            radius = float(rng.uniform(*cfg.gen_crater_radius_frac)) * m
            r = float(rng.uniform(0.08 * h, 0.92 * h))
            c = float(rng.uniform(0.08 * w, 0.92 * w))
            if all(np.hypot(r - ar, c - ac) > radius * 1.6 + 0.06 * m for ar, ac in avoid):
                break
        radius = max(radius, 2.0)
        depth = float(rng.uniform(*cfg.gen_crater_depth_per_r)) * radius
        craters.append(_Crater(r, c, radius, depth))
    return craters


def _generate(h: int, w: int, seed: int, cfg: Config) -> World | None:
    """One generation attempt. Returns None if start/goal can't be connected."""
    rng = np.random.default_rng(seed)
    m = min(h, w)
    vscale = m / cfg.gen_reference_size  # keep slopes scale-invariant across grid sizes
    rr, cc = np.mgrid[:h, :w].astype(_F32)

    margin = max(1, int(round(cfg.gen_corner_margin_frac * m)))
    start0: RC = (margin, margin)
    goal0: RC = (h - 1 - margin, w - 1 - margin)

    # 1. base terrain
    dem = vscale * cfg.gen_noise_amp_m * fractal_noise(h, w, rng, cfg.gen_noise_octaves,
                                              cfg.gen_noise_base_cells, cfg.gen_noise_persistence)

    # 2. low basin, kept off the start/goal corners
    br = float(rng.uniform(0.3, 0.7) * h)
    bc = float(rng.uniform(0.3, 0.7) * w)
    bs = cfg.gen_basin_radius_frac * m
    basin_d2 = (rr - br) ** 2 + (cc - bc) ** 2
    dem -= vscale * cfg.gen_basin_depth_m * np.exp(-basin_d2 / (2 * bs ** 2))

    # 3. craters
    craters = _place_craters(h, w, rng, cfg, avoid=[start0, goal0])
    rim_frac = cfg.gen_rim_height_frac
    for k in craters:  # loop over craters (<=10), each fully vectorised
        d = np.sqrt((rr - k.r) ** 2 + (cc - k.c) ** 2)
        bowl = -k.depth * np.exp(-(d / (0.55 * k.radius)) ** 2)
        rim = rim_frac * k.depth * np.exp(-((d - k.radius) / (0.22 * k.radius)) ** 2)
        dem += bowl + rim

    # 4. finite ridge segment, roughly perpendicular to the start->goal diagonal
    theta = np.arctan2(-(h - 1), (w - 1)) + rng.uniform(-0.35, 0.35)  # anti-diagonal +- jitter
    cr = float(rng.uniform(0.4, 0.6) * h)
    ccen = float(rng.uniform(0.4, 0.6) * w)
    ur, uc = np.sin(theta), np.cos(theta)                 # unit vector along ridge (row, col)
    along = (rr - cr) * ur + (cc - ccen) * uc             # cells along ridge axis
    across = -(rr - cr) * uc + (cc - ccen) * ur           # cells across ridge axis
    half_len = float(rng.uniform(0.12, 0.2)) * np.hypot(h, w)
    sig = max(cfg.gen_ridge_width_frac * m, cfg.gen_ridge_min_sigma_cells)
    taper = np.clip(1.0 - (np.abs(along) / half_len) ** 4, 0.0, 1.0)
    dem += vscale * cfg.gen_ridge_height_m * taper * np.exp(-0.5 * (across / sig) ** 2)
    dem = dem.astype(_F32)

    # 5. classes
    slope = slope_deg(dem, cfg.cell_size)
    cls = np.full((h, w), SOIL, np.int8)
    cls[slope > cfg.slope_safe_deg] = BEDROCK

    edge = 1.0 + 0.35 * fractal_noise(h, w, rng, 3, 4, 0.5)  # ragged sand-field boundary
    in_basin = basin_d2 < (bs * edge) ** 2
    if in_basin.any():
        thr = np.percentile(dem[in_basin], cfg.gen_sand_percentile)
        sand = in_basin & (dem < thr) & (slope < cfg.slope_safe_deg)
        cls[sand] = SAND

    for k in craters:
        n_cl = cfg.gen_rock_cluster_per_crater
        ang = rng.uniform(0, 2 * np.pi, n_cl)
        rad = k.radius * rng.uniform(0.9, 1.15, n_cl)
        sizes = rng.uniform(*cfg.gen_rock_cluster_radius, n_cl)
        for a, rd, s in zip(ang, rad, sizes):
            r0, c0 = k.r + rd * np.sin(a), k.c + rd * np.cos(a)
            r_lo, r_hi = max(0, int(r0 - s - 1)), min(h, int(r0 + s + 2))
            c_lo, c_hi = max(0, int(c0 - s - 1)), min(w, int(c0 + s + 2))
            if r_lo >= r_hi or c_lo >= c_hi:
                continue
            win_r, win_c = rr[r_lo:r_hi, c_lo:c_hi], cc[r_lo:r_hi, c_lo:c_hi]
            disc = (win_r - r0) ** 2 + (win_c - c0) ** 2 <= s ** 2
            patchy = rng.random(disc.shape) < 0.75
            cls[r_lo:r_hi, c_lo:c_hi][disc & patchy] = BIG_ROCK

    # 6. ice signal in the most shadowed crater floors (low sun from the NW)
    shade = hillshade(dem, cfg.cell_size, azimuth=315.0, altitude=15.0)
    floor_shade = []
    for k in craters:
        floor = (rr - k.r) ** 2 + (cc - k.c) ** 2 <= (0.8 * k.radius) ** 2
        floor_shade.append(float(shade[floor].mean()) if floor.any() else 1.0)
    n_ice = int(rng.integers(cfg.gen_n_ice_craters[0], cfg.gen_n_ice_craters[1] + 1))
    ice = np.zeros((h, w), _F32)
    for idx in np.argsort(floor_shade)[: min(n_ice, len(craters))]:
        k = craters[int(idx)]
        peak = float(rng.uniform(0.75, 1.0))
        ice += peak * np.exp(-((rr - k.r) ** 2 + (cc - k.c) ** 2) / (2 * (0.45 * k.radius) ** 2))
    ice += 0.05 * (fractal_noise(h, w, rng, 3, 4, 0.5) + 1.0)  # faint background
    ice = np.clip(ice, 0.0, 1.0).astype(_F32)

    # 7. start/goal on passable cells, connected
    passable = (slope < cfg.slope_max_deg) & (cls != BIG_ROCK)
    start = _snap_to_passable(passable, start0)
    goal = _snap_to_passable(passable, goal0)
    if start is None or goal is None or start == goal:
        return None
    if not _connected(passable, start, goal):
        return None

    return World(dem=dem, cell_size=cfg.cell_size, true_class=cls, ice_signal=ice,
                 start=start, goal=goal, seed=seed, name="synthetic")


def make_world(H: int = 200, W: int = 200, seed: int = 42, cfg: Config | None = None) -> World:
    """Build a synthetic Mars test site.

    Args:
        H, W: grid size in cells (>= 16 each).
        seed: RNG seed. If the resulting map doesn't connect start and goal, retries with
            seed+1, seed+2, ... (``World.seed`` records the seed that succeeded).
        cfg: configuration (defaults to ``core.config.DEFAULT``).

    Returns:
        World with float32 DEM (m), int8 classes, float32 ice_signal in [0, 1], and
        start/goal on passable, mutually reachable cells.

    Raises:
        ValueError: grid too small.
        RuntimeError: no connected world within ``cfg.gen_max_regen_attempts`` seeds.
    """
    cfg = cfg or DEFAULT
    if H < 16 or W < 16:
        raise ValueError(f"grid must be at least 16x16, got {H}x{W}")
    for attempt in range(cfg.gen_max_regen_attempts):
        world = _generate(H, W, seed + attempt, cfg)
        if world is not None:
            return world
    raise RuntimeError(f"no connected world after {cfg.gen_max_regen_attempts} seeds from {seed}")


# --------------------------------------------------------------------------- demo
def save_world_preview(world: World, path: str = "outputs/world_preview.png") -> str:
    """Hillshade + semi-transparent class overlay + ice contours + start/goal. Headless."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    colors = ["#c8814a", "#6b6b6b", "#f2d16b", "#7a1f1f"]  # soil, bedrock, sand, big_rock
    hs = hillshade(world.dem, world.cell_size)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))

    ax = axes[0]
    ax.imshow(hs, cmap="gray", origin="upper")
    ax.imshow(world.true_class, cmap=ListedColormap(colors), vmin=-0.5, vmax=N_CLASSES - 0.5,
              alpha=0.45, origin="upper", interpolation="nearest")
    ax.contour(world.ice_signal, levels=[0.3, 0.6, 0.9], colors="cyan", linewidths=0.8)
    for rc, mk, lab in ((world.start, "o", "start"), (world.goal, "*", "goal")):
        ax.plot(rc[1], rc[0], mk, ms=12, mec="k", mfc="lime" if lab == "start" else "magenta")
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, CLASS_NAMES)]
              + [Patch(color="cyan", label="ice contours")], loc="lower left", fontsize=8)
    ax.set_title(f"Synthetic site (seed={world.seed}) — classes over hillshade")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    ax = axes[1]
    sl = slope_deg(world.dem, world.cell_size)
    im = ax.imshow(sl, cmap="magma", origin="upper", vmin=0, vmax=40)
    ax.contour(sl, levels=[DEFAULT.slope_max_deg], colors="white", linewidths=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="slope (deg)")
    ax.plot(world.start[1], world.start[0], "o", ms=10, mfc="lime", mec="k")
    ax.plot(world.goal[1], world.goal[0], "*", ms=12, mfc="magenta", mec="k")
    ax.set_title(f"Slope (white = {DEFAULT.slope_max_deg:.0f}° limit)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    w = make_world(200, 200, seed=DEFAULT.seed)
    dt = time.perf_counter() - t0
    sl = slope_deg(w.dem, w.cell_size)
    mask = passable_mask(w.dem, w.true_class, w.cell_size)
    print(f"make_world(200,200) in {dt * 1000:.1f} ms  (seed requested={DEFAULT.seed}, used={w.seed})")
    print(f"DEM range [{w.dem.min():.1f}, {w.dem.max():.1f}] m; slope max {sl.max():.1f} deg; "
          f"impassable {100 * (~mask).mean():.1f}%")
    fr = np.bincount(w.true_class.ravel(), minlength=N_CLASSES) / w.true_class.size
    print("class fractions: " + ", ".join(f"{n}={f:.1%}" for n, f in zip(CLASS_NAMES, fr)))
    print(f"ice_signal max {w.ice_signal.max():.2f}; start {w.start} goal {w.goal}; "
          f"BFS reachable={bfs_reachable(mask, w.start, w.goal)}")
    print("saved", save_world_preview(w))
