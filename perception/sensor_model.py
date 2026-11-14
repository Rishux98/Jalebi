"""Rover "navcam" sensor model built from the perception network's confusion matrix.

The trained AI4MARS U-Net's validation confusion matrix C[t, o] = P(predicted o | true t)
*is* the sensor model: when the rover looks at a cell of true class t, it reports class o
with probability C[t, o]. The same matrix gives the Bayes likelihood P(obs | true) for P3.

If ``perception/confusion.npy`` is missing or invalid, a hard-coded plausible matrix is used
so the pipeline never blocks on training. ``load_confusion`` reports which one was used.

Torch-free: runs anywhere.  Run: python -m perception.sensor_model
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import DEFAULT, Config
from core.types import CLASS_NAMES, N_CLASSES, RC

_F32 = np.float32

# Rows = true class, cols = predicted class (soil, bedrock, sand, big_rock); rows sum to 1.
# Plausible values for an AI4MARS U-Net: sand <-> soil is the dominant confusion
# ("sand looks like soil 18% of the time"), big rocks get mistaken for bedrock.
FALLBACK_CONFUSION = np.array([
    [0.80, 0.07, 0.11, 0.02],   # true soil
    [0.10, 0.78, 0.05, 0.07],   # true bedrock
    [0.18, 0.04, 0.76, 0.02],   # true sand
    [0.05, 0.15, 0.02, 0.78],   # true big_rock
], dtype=_F32)

UNOBSERVED = -1  # value in observation maps for cells outside the sensor footprint


def _validate(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (N_CLASSES, N_CLASSES) or not np.isfinite(m).all() or (m < 0).any():
        raise ValueError(f"confusion must be a finite non-negative {N_CLASSES}x{N_CLASSES} matrix")
    rows = m.sum(axis=1, keepdims=True)
    if (rows <= 0).any():
        raise ValueError("confusion matrix has an all-zero row")
    return (m / rows).astype(_F32)


def load_confusion(path: str | Path | None = None, cfg: Config = DEFAULT) -> tuple[np.ndarray, str]:
    """Load the row-normalised 4x4 confusion matrix P(pred | true).

    Args:
        path: .npy file; defaults to ``cfg.confusion_path`` (relative to the repo root).

    Returns:
        (matrix float32 (4, 4), source) where source is the file path or ``"fallback"``.
    """
    p = Path(path) if path is not None else Path(__file__).resolve().parents[1] / cfg.confusion_path
    try:
        return _validate(np.load(p)), str(p)
    except (OSError, ValueError):
        return FALLBACK_CONFUSION.copy(), "fallback"


def get_sensor_info(cfg: Config = DEFAULT, root: str | Path | None = None) -> dict:
    """Which sensor model is active, for the UI and the pipeline. Never raises.

    "trained" only when BOTH ``perception/confusion.npy`` (valid 4x4) and ``perception/metrics.json``
    (readable JSON) exist; otherwise the documented fallback matrix.

    Returns:
        dict with ``source`` ("trained" | "fallback"), ``path`` (str | None), ``miou`` (float | None),
        ``per_class_iou`` (dict | None) and ``confusion`` ((4, 4) float32, rows = true class, P(pred | true)).
    """
    import json

    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    npy = base / cfg.confusion_path
    met = npy.parent / "metrics.json"
    try:
        if npy.exists() and met.exists():
            conf = _validate(np.load(npy))
            m = json.loads(met.read_text())
            miou = m.get("miou")
            iou = m.get("iou")
            return {"source": "trained", "path": str(npy),
                    "miou": float(miou) if isinstance(miou, (int, float)) else None,
                    "per_class_iou": dict(iou) if isinstance(iou, dict) else None, "confusion": conf}
    except Exception:  # noqa: BLE001 — UI helper must never take the dashboard down
        pass
    return {"source": "fallback", "path": None, "miou": None, "per_class_iou": None,
            "confusion": FALLBACK_CONFUSION.copy()}


def observe(true_class: np.ndarray, rng: np.random.Generator,
            confusion: np.ndarray = FALLBACK_CONFUSION) -> np.ndarray:
    """Sample noisy class observations for an array of true classes (vectorised).

    Args:
        true_class: int array of any shape with values in 0..3.
        rng: ``np.random.default_rng(seed)`` generator.
        confusion: (4, 4) P(pred | true).

    Returns:
        int8 array, same shape, sampled observed class per cell.
    """
    t = np.asarray(true_class, dtype=np.int64)
    cdf = np.cumsum(np.asarray(confusion, dtype=np.float64), axis=1)[t]   # (..., 4)
    u = rng.random(t.shape + (1,))
    obs = (u > cdf[..., :-1]).sum(axis=-1)                                # inverse-CDF sampling
    return obs.astype(np.int8)


def likelihood(observed: np.ndarray, confusion: np.ndarray = FALLBACK_CONFUSION) -> np.ndarray:
    """Bayes likelihood P(obs | true = k) for each cell: (...,) int -> (..., 4) float32.

    Unobserved cells (``UNOBSERVED``) get a flat likelihood of 1 (no information).
    """
    o = np.asarray(observed, dtype=np.int64)
    lik = np.asarray(confusion, dtype=_F32).T[np.clip(o, 0, N_CLASSES - 1)]  # row o of C^T
    lik[o == UNOBSERVED] = 1.0
    return lik.astype(_F32, copy=False)


def footprint(shape: tuple[int, int], rc: RC, radius: float) -> np.ndarray:
    """(H, W) bool disc of cells within ``radius`` cells of ``rc`` (the sensor footprint)."""
    h, w = shape
    rr, cc = np.ogrid[:h, :w]
    return (rr - rc[0]) ** 2 + (cc - rc[1]) ** 2 <= float(radius) ** 2


def observe_window(true_class: np.ndarray, rc: RC, radius: float, rng: np.random.Generator,
                   confusion: np.ndarray = FALLBACK_CONFUSION) -> tuple[np.ndarray, np.ndarray]:
    """Sense every cell within ``radius`` of the rover.

    Returns:
        (observed (H, W) int8 with ``UNOBSERVED`` outside the footprint, mask (H, W) bool).
    """
    mask = footprint(true_class.shape, rc, radius)
    obs = np.full(true_class.shape, UNOBSERVED, np.int8)
    obs[mask] = observe(true_class[mask], rng, confusion)
    return obs, mask


if __name__ == "__main__":
    from core.io import save_png  # noqa: F401  (ensures Agg backend)
    import matplotlib.pyplot as plt

    from world.synthetic import make_world

    C, src = load_confusion()
    print(f"confusion source: {src}")
    print("P(pred | true)   " + " ".join(f"{n:>9s}" for n in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, C):
        print(f"  true {name:9s} " + " ".join(f"{v:9.3f}" for v in row))

    w = make_world(200, 200, seed=DEFAULT.seed)
    rng = np.random.default_rng(DEFAULT.seed)
    obs = observe(w.true_class, rng, C)
    emp = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(emp, (w.true_class.ravel().astype(int), obs.ravel().astype(int)), 1)
    emp = emp / np.clip(emp.sum(1, keepdims=True), 1, None)
    print(f"max |empirical - C| over a 200x200 world: {np.abs(emp - C).max():.3f}")
    print(f"per-cell accuracy {np.mean(obs == w.true_class):.1%}")

    o, m = observe_window(w.true_class, w.start, DEFAULT.sensor_radius, rng, C)
    print(f"observe_window at {w.start}, r={DEFAULT.sensor_radius}: {int(m.sum())} cells sensed")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] > 0.5 else "black")
    ax.set_xticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Sensor model P(pred | true) [{'fallback' if src == 'fallback' else 'trained'}]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/confusion.png", dpi=110)
    plt.close(fig)
    print("saved outputs/confusion.png")
