"""Per-cell terrain-class belief: orbital prior + Bayesian update from navcam observations.

Prior (``init_prior``) simulates a low-resolution orbital classifier:
    1. majority-vote ``true_class`` over ``prior_downsample`` x ``prior_downsample`` blocks
       (small boulder clusters vanish: orbit cannot resolve them -> hidden hazards for P4);
    2. relabel ``prior_label_noise`` of the orbital pixels uniformly at random;
    3. upsample (nearest), one-hot, Gaussian blur (``prior_blur_sigma`` cells);
    4. soften with temperature: p ** (1/T), renormalise;
    5. floor: p = floor + (1 - K*floor) * p, so every class keeps >= ``prior_floor``.

Update (``update``), per cell k = class, independent observations o_1..o_n:
    posterior(k) ∝ prior(k) * prod_i P(o_i | true = k),    P(o | k) = confusion[k, o]
Cells outside the sensor footprint get a flat likelihood (unchanged).

Run: python -m belief.bayes   (writes outputs/belief.png)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from core.config import DEFAULT, Config
from core.types import N_CLASSES, RC, Belief, World
from perception.sensor_model import FALLBACK_CONFUSION, UNOBSERVED, likelihood, observe_window

_F32 = np.float32


def _block_majority(cls: np.ndarray, f: int) -> np.ndarray:
    """(H, W) int classes -> (ceil(H/f), ceil(W/f)) majority class per f x f block (edge-padded)."""
    h, w = cls.shape
    hc, wc = -(-h // f), -(-w // f)
    padded = np.pad(cls, ((0, hc * f - h), (0, wc * f - w)), mode="edge")
    onehot = np.eye(N_CLASSES, dtype=np.int32)[padded]                        # (hc*f, wc*f, K)
    counts = onehot.reshape(hc, f, wc, f, N_CLASSES).sum(axis=(1, 3))         # (hc, wc, K)
    return counts.argmax(axis=-1)


def init_prior(world: World, cfg: Config = DEFAULT, seed: int | None = None) -> Belief:
    """Orbital prior belief (see module docstring). Deterministic for a given ``seed``.

    Args:
        world: site; only ``true_class`` is read (to simulate what orbit would see).
        cfg: ``prior_*`` parameters.
        seed: label-noise RNG seed; defaults to ``cfg.seed``.

    Returns:
        Belief with (H, W, 4) float32 probs (every class >= ``prior_floor``), entropy (nats),
        and an all-False ``observed_mask``.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    f = max(int(cfg.prior_downsample), 1)
    h, w = world.shape
    coarse = _block_majority(world.true_class.astype(np.int64), f)
    # label noise: replacement labels follow the orbital map's own class frequencies (an orbital
    # classifier rarely reports boulders it cannot resolve, so noise must not invent boulder fields)
    freq = np.bincount(coarse.ravel(), minlength=N_CLASSES).astype(np.float64)
    noisy = rng.random(coarse.shape) < cfg.prior_label_noise
    coarse = np.where(noisy, rng.choice(N_CLASSES, size=coarse.shape, p=freq / freq.sum()), coarse)
    fine = np.repeat(np.repeat(coarse, f, axis=0), f, axis=1)[:h, :w]
    p = np.eye(N_CLASSES, dtype=np.float64)[fine]                             # (H, W, K)
    if cfg.prior_blur_sigma > 0:
        p = gaussian_filter(p, sigma=(cfg.prior_blur_sigma, cfg.prior_blur_sigma, 0), mode="nearest")
    p = np.clip(p, 0.0, None) ** (1.0 / max(cfg.prior_temperature, 1e-6))
    p /= np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
    floor = float(np.clip(cfg.prior_floor, 0.0, 1.0 / N_CLASSES))
    p = floor + (1.0 - N_CLASSES * floor) * p
    return Belief.from_probs(p.astype(_F32), np.zeros((h, w), dtype=bool))


def update(belief: Belief, observed: np.ndarray, mask: np.ndarray | None = None,
           confusion: np.ndarray = FALLBACK_CONFUSION) -> Belief:
    """Bayesian update of the class belief with one or more observation maps. Returns a new Belief.

    Args:
        belief: current belief (not modified).
        observed: (H, W) or (N, H, W) int observed class per cell, ``UNOBSERVED`` (-1) where not sensed.
            N stacked maps are treated as independent observations (likelihoods multiply).
        mask: optional (H, W) bool sensor footprint; cells in it are marked observed even if every
            observation there is ``UNOBSERVED``. Defaults to "any observation present".
        confusion: (4, 4) P(pred | true) sensor model.

    Returns:
        Belief with posterior probs (float32, rows sum to 1), recomputed entropy and
        ``observed_mask = old | sensed``.
    """
    obs = np.asarray(observed)
    if obs.ndim == 2:
        obs = obs[None]
    hw = belief.shape
    if obs.ndim != 3 or obs.shape[1:] != hw:
        raise ValueError(f"observed shape {np.asarray(observed).shape} incompatible with belief {hw}")
    lik = np.prod(likelihood(obs, confusion).astype(np.float64), axis=0)       # (H, W, K)
    post = belief.class_probs.astype(np.float64) * lik
    z = post.sum(axis=-1, keepdims=True)
    bad = z[..., 0] <= 0                                                       # impossible obs: keep prior
    post = np.where(bad[..., None], belief.class_probs, post / np.where(z > 0, z, 1.0))
    sensed = (obs != UNOBSERVED).any(axis=0)
    if mask is not None:
        sensed |= np.asarray(mask, dtype=bool)
    return Belief.from_probs(post.astype(_F32), belief.observed_mask | sensed)


def sense(belief: Belief, world: World, rc: RC, rng: np.random.Generator, cfg: Config = DEFAULT,
          confusion: np.ndarray = FALLBACK_CONFUSION, radius: float | None = None) -> Belief:
    """Rover looks around from ``rc`` (navcam disc of ``cfg.sensor_radius`` cells) and updates its belief.

    The hidden ``world.true_class`` is only touched through the noisy sensor model.
    """
    r = cfg.sensor_radius if radius is None else radius
    obs, mask = observe_window(world.true_class, rc, r, rng, confusion)
    return update(belief, obs, mask, confusion)


def belief_accuracy(belief: Belief, true_class: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of cells (optionally within ``mask``) whose argmax class equals the truth."""
    hit = belief.class_probs.argmax(axis=-1) == np.asarray(true_class)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        return float(hit[m].mean()) if m.any() else float("nan")
    return float(hit.mean())


if __name__ == "__main__":
    import time
    from pathlib import Path

    from core.io import hillshade  # noqa: F401  (ensures Agg backend)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from core.types import CLASS_NAMES
    from perception.sensor_model import load_confusion
    from world.synthetic import make_world

    cfg = DEFAULT
    w = make_world(200, 200, seed=cfg.seed)
    C, src = load_confusion()
    t0 = time.perf_counter()
    prior = init_prior(w, cfg)
    t_prior = time.perf_counter() - t0
    print(f"confusion: {src}")
    print(f"prior: accuracy {belief_accuracy(prior, w.true_class):.1%}, mean entropy "
          f"{prior.entropy.mean():.3f} nats (max {np.log(N_CLASSES):.3f}), built in {1000 * t_prior:.0f} ms")
    rock = w.true_class == 3
    print(f"  true big_rock cells: {int(rock.sum())}, prior P(big_rock) on them: "
          f"mean {prior.class_probs[..., 3][rock].mean():.2f}  <- orbit misses most boulders")

    # drive along the straight start->goal line, sensing every 4 cells
    rng = np.random.default_rng(cfg.seed)
    n = int(max(abs(w.goal[0] - w.start[0]), abs(w.goal[1] - w.start[1])))
    line = [(int(round(w.start[0] + (w.goal[0] - w.start[0]) * t)), int(round(w.start[1] + (w.goal[1] - w.start[1]) * t)))
            for t in np.linspace(0, 1, n // 4 + 1)]
    post = prior
    t0 = time.perf_counter()
    for rc in line:
        post = sense(post, w, rc, rng, cfg, C)
    dt = (time.perf_counter() - t0) / len(line)
    m = post.observed_mask
    print(f"after {len(line)} sensing stops ({1000 * dt:.1f} ms/update): observed {int(m.sum())} cells")
    print(f"  accuracy on observed cells: prior {belief_accuracy(prior, w.true_class, m):.1%} -> "
          f"posterior {belief_accuracy(post, w.true_class, m):.1%}")
    print(f"  mean entropy on observed cells: {prior.entropy[m].mean():.3f} -> {post.entropy[m].mean():.3f} nats")
    if (rock & m).any():
        print(f"  P(big_rock) on observed true boulders: {prior.class_probs[..., 3][rock & m].mean():.2f} -> "
              f"{post.class_probs[..., 3][rock & m].mean():.2f}")

    cmap = ListedColormap([(0.80, 0.55, 0.30), (0.45, 0.45, 0.50), (0.95, 0.85, 0.45), (0.60, 0.10, 0.10)])
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    ax[0].imshow(w.true_class, cmap=cmap, vmin=-0.5, vmax=3.5); ax[0].set_title("truth (hidden)")
    ax[1].imshow(prior.class_probs.argmax(-1), cmap=cmap, vmin=-0.5, vmax=3.5)
    ax[1].set_title(f"orbital prior argmax ({belief_accuracy(prior, w.true_class):.0%} acc)")
    ax[2].imshow(post.class_probs.argmax(-1), cmap=cmap, vmin=-0.5, vmax=3.5)
    ax[2].set_title(f"posterior after {len(line)} navcam looks")
    im = ax[3].imshow(post.entropy, cmap="magma", vmin=0, vmax=np.log(N_CLASSES))
    ax[3].set_title("posterior entropy (nats)")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.plot([p[1] for p in line], [p[0] for p in line], ".", ms=2, color="cyan")
        a.set_xticks([]); a.set_yticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cmap(i), label=nm) for i, nm in enumerate(CLASS_NAMES)],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/belief.png", dpi=90)
    plt.close(fig)
    print("saved outputs/belief.png")
