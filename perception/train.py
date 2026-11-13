"""Train the AI4MARS terrain U-Net and export the rover sensor model (confusion.npy).

Outputs (in ``--out``, default ``perception/``):
  unet.pt        checkpoint (state_dict + meta)
  confusion.npy  4x4 row-normalised P(pred | true) on the val split -> perception/sensor_model.py
  metrics.json   per-class IoU, mIoU, pixel accuracy, per-epoch history, settings

Local CPU smoke test (synthetic data, no download, writes to outputs/perception_smoke/):
  python -m perception.train --smoke
Real run (Colab GPU):
  !pip install -q segmentation-models-pytorch datasets
  !python -m perception.train                    # ~2000 train / 322 val images, 8 epochs
  # or cell by cell:  from perception.train import *; tr, va = get_data(...); run = train(...)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core.config import DEFAULT, Config
from core.types import CLASS_NAMES, N_CLASSES
from perception.data import (TRAIN_SPLIT, VAL_SPLIT, SegSet, class_fractions, load_ai4mars,
                             synthetic_set)
from perception.model import (IGNORE_INDEX, build_unet, confusion_counts, normalise_gray,
                              per_class_iou, row_normalise, save_checkpoint)


@dataclass
class TrainResult:
    """What ``train`` produced. ``counts`` = final val confusion counts C[true, pred]."""
    model: nn.Module
    counts: np.ndarray
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def set_seed(seed: int) -> np.random.Generator:
    """Seed torch (CPU + CUDA) and return a NumPy generator for batching / augmentation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def get_data(n_train: int, n_val: int, size: int, seed: int, smoke: bool,
             cache_dir: str | Path | None = None) -> tuple[SegSet, SegSet]:
    """Train/val sets: synthetic for ``smoke``, else streamed AI4MARS (cached as .npz if ``cache_dir``)."""
    if smoke:
        return synthetic_set(n_train, size, seed), synthetic_set(n_val, size, seed + 1)
    sets = []
    for split, n in ((TRAIN_SPLIT, n_train), (VAL_SPLIT, n_val)):
        cache = Path(cache_dir) / f"ai4mars_{split}_{n}_{size}.npz" if cache_dir else None
        if cache is not None and cache.exists():
            sets.append(SegSet.load(cache))
            print(f"loaded cache {cache}")
            continue
        print(f"streaming {split} ({n} images) ...")
        s = load_ai4mars(split, n, size, seed)
        if cache is not None:
            s.save(cache)
        sets.append(s)
    return sets[0], sets[1]


def class_weights(train_set: SegSet, power: float) -> torch.Tensor:
    """CE weights ∝ frequency^-power, normalised to mean 1 (helps the rare big_rock class)."""
    f = np.clip(class_fractions(train_set), 1e-4, None)
    w = f ** -power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def _batch(s: SegSet, idx: np.ndarray, dev: torch.device, flip: np.ndarray | None = None):
    g, y = s.images[idx], s.labels[idx]
    if flip is not None and flip.any():                     # horizontal flip augmentation
        g, y = g.copy(), y.copy()
        g[flip], y[flip] = g[flip][..., ::-1], y[flip][..., ::-1]
    x = torch.from_numpy(normalise_gray(g)).to(dev)
    return x, torch.from_numpy(y.astype(np.int64)).to(dev)


@torch.no_grad()
def evaluate(model: nn.Module, s: SegSet, batch: int, dev: torch.device) -> np.ndarray:
    """Deterministic (dropout off) val pass -> (4, 4) confusion counts C[true, pred]."""
    model.eval()
    counts = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for i in range(0, len(s), batch):
        idx = np.arange(i, min(i + batch, len(s)))
        x, y = _batch(s, idx, dev)
        pred = model(x).argmax(1)
        counts += confusion_counts(y.cpu().numpy(), pred.cpu().numpy())
    return counts


def summarise(counts: np.ndarray) -> dict:
    """Metrics dict from confusion counts (IoU per class, mIoU over present classes, pixel acc)."""
    iou = per_class_iou(counts)
    return {
        "iou": {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(CLASS_NAMES, iou)},
        "miou": round(float(np.nanmean(iou)), 4) if np.isfinite(iou).any() else None,
        "pixel_acc": round(float(np.trace(counts) / max(counts.sum(), 1)), 4),
    }


def train(cfg: Config, tr: SegSet, va: SegSet, epochs: int, batch: int, seed: int,
          pretrained: bool = True, device: str | None = None) -> TrainResult:
    """AdamW + (CE + Dice), AMP on CUDA. Prints loss and per-class IoU each epoch."""
    import segmentation_models_pytorch as smp

    rng = set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = dev.type == "cuda"
    model = build_unet(cfg, encoder_weights="imagenet" if pretrained else None).to(dev)
    ce = nn.CrossEntropyLoss(weight=class_weights(tr, cfg.percep_class_weight_power).to(dev),
                             ignore_index=IGNORE_INDEX)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.percep_lr, weight_decay=cfg.percep_weight_decay)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.percep_lr, total_steps=max(steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={dev} amp={use_amp} train={len(tr)} ({tr.source}) val={len(va)} "
          f"epochs={epochs} batch={batch}")
    history, t0 = [], time.time()
    counts = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x, y = _batch(tr, idx, dev, flip=rng.random(len(idx)) < 0.5)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
                loss = ce(logits, y) + dice(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        counts = evaluate(model, va, batch, dev)
        m = summarise(counts)
        history.append({"epoch": ep, "loss": round(float(np.mean(losses)), 4), **m})
        iou_txt = " ".join(f"{k}={'—' if v is None else f'{v:.3f}'}" for k, v in m["iou"].items())
        print(f"epoch {ep}/{epochs} loss={np.mean(losses):.4f} mIoU={m['miou']} acc={m['pixel_acc']} | "
              f"{iou_txt} ({time.time() - t0:.0f}s)")
    return TrainResult(model, counts, history, time.time() - t0)


def export(run: TrainResult, out: str | Path, cfg: Config, settings: dict) -> dict[str, Path]:
    """Write unet.pt, confusion.npy (P(pred|true)), metrics.json to ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    conf = row_normalise(run.counts)
    metrics = {**summarise(run.counts), "confusion_p_pred_given_true": conf.round(4).tolist(),
               "confusion_counts": run.counts.tolist(), "class_names": list(CLASS_NAMES),
               "history": run.history, "train_seconds": round(run.seconds, 1), "settings": settings}
    paths = {
        "unet": save_checkpoint(run.model, out / "unet.pt", meta={"settings": settings, **summarise(run.counts)}),
        "confusion": out / "confusion.npy",
        "metrics": out / "metrics.json",
    }
    np.save(paths["confusion"], conf)
    paths["metrics"].write_text(json.dumps(metrics, indent=2))
    return paths


def main(argv: list[str] | None = None) -> dict[str, Path]:
    cfg = DEFAULT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run on CPU (no download)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default=None, help="output dir (default perception/, smoke: outputs/perception_smoke)")
    ap.add_argument("--cache", default=None, help="dir to cache streamed data as .npz")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet encoder weights")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    smoke = a.smoke
    s = {
        "n_train": a.n_train or (32 if smoke else cfg.percep_n_train),
        "n_val": a.n_val or (8 if smoke else cfg.percep_n_val),
        "size": a.size or (64 if smoke else cfg.percep_img_size),
        "epochs": a.epochs or (2 if smoke else cfg.percep_epochs),
        "batch": a.batch or (8 if smoke else cfg.percep_batch),
        "seed": a.seed, "smoke": smoke, "pretrained": not (a.no_pretrained or smoke),
        "dropout": cfg.percep_dropout, "lr": cfg.percep_lr, "encoder": "resnet34",
        "train_split": "synthetic" if smoke else TRAIN_SPLIT, "val_split": "synthetic" if smoke else VAL_SPLIT,
    }
    out = a.out or ("outputs/perception_smoke" if smoke else str(Path(cfg.confusion_path).parent))
    tr, va = get_data(s["n_train"], s["n_val"], s["size"], s["seed"], smoke, a.cache)
    run = train(cfg, tr, va, s["epochs"], s["batch"], s["seed"], s["pretrained"], a.device)
    paths = export(run, out, cfg, s)
    print("saved " + ", ".join(str(p) for p in paths.values()))
    print("P(pred | true) on val:\n" + np.array2string(row_normalise(run.counts), precision=3))
    return paths


if __name__ == "__main__":
    main()
