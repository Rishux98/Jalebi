"""AI4MARS data loading for perception/train.py and perception/infer.py.

Images are kept as uint8 grayscale (N, S, S) and labels as uint8 (N, S, S) so 2000 images at
256 px fit in ~260 MB; normalisation to float happens per batch (``model.normalise_gray``).

* ``load_ai4mars``  streams the HF dataset ``hassanjbara/AI4MARS`` (no full download).
* ``synthetic_set`` makes a small learnable fake set (textured class regions) so the
  training/inference code can be smoke-tested on CPU with no network access.

Run: python -m perception.data            (synthetic only)
     python -m perception.data --hf 4     (stream 4 real images and print label stats)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import DEFAULT
from core.types import N_CLASSES
from perception.model import IGNORE_INDEX, map_ai4mars_mask, resize_gray

HF_DATASET = "hassanjbara/AI4MARS"
TRAIN_SPLIT = "train"
VAL_SPLIT = "test_min3"          # 322 images, labels agreed by >= 3 labellers


@dataclass
class SegSet:
    """A segmentation dataset in memory.

    Attributes:
        images: (N, S, S) uint8 grayscale.
        labels: (N, S, S) uint8 in {0..3, IGNORE_INDEX}.
        source: "hf:<split>" or "synthetic".
    """
    images: np.ndarray
    labels: np.ndarray
    source: str

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def save(self, path: str | Path) -> Path:
        """Cache to .npz (avoids re-streaming on Colab restarts)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, images=self.images, labels=self.labels, source=self.source)
        return p

    @staticmethod
    def load(path: str | Path) -> "SegSet":
        d = np.load(path)
        return SegSet(d["images"], d["labels"], str(d["source"]))


def _resize_labels(lbl: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(lbl).resize((size, size), Image.NEAREST), dtype=np.uint8)


def row_to_example(row: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    """One HF row -> (gray (S, S) uint8, labels (S, S) uint8).

    Pixels on the rover body (``rover_mask`` > 0) or beyond the labelled range (``range_mask`` > 0)
    are set to IGNORE_INDEX, matching the AI4MARS protocol.
    """
    lbl = map_ai4mars_mask(np.asarray(row["label_mask"]))
    for key in ("rover_mask", "range_mask"):
        m = row.get(key)
        if m is not None:
            m = np.asarray(m)
            m = m[..., 0] if m.ndim == 3 else m
            if m.shape == lbl.shape:
                lbl[m > 0] = IGNORE_INDEX
    return resize_gray(row["image"], size), _resize_labels(lbl, size)


def load_ai4mars(split: str, n: int, size: int = DEFAULT.percep_img_size, seed: int = DEFAULT.seed,
                 shuffle_buffer: int = 500, verbose: bool = True) -> SegSet:
    """Stream ``n`` labelled examples of ``split`` (train shuffled with ``seed``; test splits in order)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    if split == TRAIN_SPLIT:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    imgs, lbls = [], []
    t0 = time.time()
    for row in ds:
        if row.get("has_labels") is False or row.get("label_mask") is None:
            continue
        g, l = row_to_example(row, size)
        if (l == IGNORE_INDEX).all():
            continue
        imgs.append(g)
        lbls.append(l)
        if verbose and len(imgs) % 200 == 0:
            print(f"  [{split}] {len(imgs)}/{n} images ({time.time() - t0:.0f}s)")
        if len(imgs) >= n:
            break
    if not imgs:
        raise RuntimeError(f"no labelled images found in {HF_DATASET}:{split}")
    return SegSet(np.stack(imgs), np.stack(lbls), f"hf:{split}")


def synthetic_set(n: int, size: int = 64, seed: int = DEFAULT.seed) -> SegSet:
    """Fake navcam set: smooth random class regions, each class with its own brightness + texture.

    Learnable in a few CPU iterations, so it exercises train/infer end to end. Not for metrics.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.standard_normal((n, size, size)), sigma=(0, size / 10, size / 10))
    q = np.quantile(noise, [0.25, 0.5, 0.75], axis=(1, 2))                 # (3, n)
    labels = (noise[None] > q[:, :, None, None]).sum(0).astype(np.uint8)    # (n, S, S) in 0..3
    base = np.array([110, 160, 200, 60], np.float32)[labels]               # soil, bedrock, sand, big_rock
    tex = np.array([12, 6, 3, 25], np.float32)[labels]
    img = np.clip(base + tex * rng.standard_normal(labels.shape), 0, 255).astype(np.uint8)
    labels[:, :2, :] = IGNORE_INDEX                                         # a strip of "null" like AI4MARS
    return SegSet(img, labels, "synthetic")


def class_fractions(s: SegSet) -> np.ndarray:
    """Fraction of labelled pixels per class, (4,) float32."""
    c = np.bincount(s.labels[s.labels != IGNORE_INDEX].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    return (c / max(c.sum(), 1)).astype(np.float32)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=int, default=0, help="stream this many real val images")
    a = ap.parse_args()
    s = synthetic_set(8)
    print(f"synthetic: {s.images.shape} {s.images.dtype}, class fractions {np.round(class_fractions(s), 2)}")
    if a.hf:
        r = load_ai4mars(VAL_SPLIT, a.hf, verbose=False)
        ign = float((r.labels == IGNORE_INDEX).mean())
        print(f"{r.source}: {r.images.shape}, ignored {ign:.1%}, class fractions {np.round(class_fractions(r), 3)}")
