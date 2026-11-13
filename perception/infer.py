"""MC-dropout inference with the trained U-Net: predictions, uncertainty maps, and an uncertainty check.

For each val image: T stochastic passes -> mean class probs, predictive entropy, mutual information.
Then checks that the uncertainty is *useful* (judging criterion "uncertainty & robustness"):
  * mean entropy on correctly vs wrongly classified pixels (wrong should be higher)
  * error rate in the 10 % most-uncertain pixels vs the 90 % least-uncertain
  * AUROC of entropy as an error detector (0.5 = useless, 1 = perfect)

Outputs (in ``--out``): sample_0..5.png (image | truth | prediction | entropy), infer_metrics.json.

  python -m perception.infer --smoke        # CPU, uses outputs/perception_smoke/unet.pt, synthetic data
  python -m perception.infer                # Colab: perception/unet.pt on AI4MARS test_min3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.config import DEFAULT, Config
from core.types import CLASS_NAMES, N_CLASSES
from perception.data import VAL_SPLIT, SegSet, load_ai4mars, synthetic_set
from perception.model import (IGNORE_INDEX, MCPrediction, confusion_counts, load_model, mc_predict,
                              normalise_gray, per_class_iou, row_normalise)

CLASS_COLORS = np.array([[0.80, 0.55, 0.30],   # soil
                         [0.45, 0.45, 0.50],   # bedrock
                         [0.95, 0.85, 0.45],   # sand
                         [0.60, 0.10, 0.10]],  # big_rock
                        np.float32)
IGNORE_COLOR = np.array([0.0, 0.0, 0.0], np.float32)


def colorize(labels: np.ndarray) -> np.ndarray:
    """(H, W) labels in {0..3, 255} -> (H, W, 3) float32 RGB."""
    out = np.broadcast_to(IGNORE_COLOR, labels.shape + (3,)).copy()
    ok = labels < N_CLASSES
    out[ok] = CLASS_COLORS[labels[ok]]
    return out


def predict_set(model, s: SegSet, n_passes: int, batch: int, seed: int) -> MCPrediction:
    """MC-dropout over a whole SegSet in batches (each batch seeded ``seed + i`` -> deterministic)."""
    parts = [mc_predict(model, normalise_gray(s.images[i:i + batch]), n_passes, seed + i)
             for i in range(0, len(s), batch)]
    return MCPrediction(*(np.concatenate([getattr(p, f) for p in parts])
                          for f in ("mean_probs", "label", "entropy", "mutual_info")), n_passes=n_passes)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U) of ``score`` for detecting ``positive``; NaN if one class is empty."""
    s = np.asarray(score, np.float64).ravel()
    y = np.asarray(positive, bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ranks[order] = np.arange(1, s.size + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)   # average ranks on ties
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def uncertainty_report(pred: MCPrediction, labels: np.ndarray, top_frac: float = 0.1) -> dict:
    """Does high entropy flag misclassified pixels? Computed on labelled pixels only."""
    valid = labels != IGNORE_INDEX
    wrong = (pred.label != labels)[valid]
    h = pred.entropy[valid]
    cut = np.quantile(h, 1 - top_frac) if h.size else 0.0
    hi = h >= cut
    counts = confusion_counts(labels, pred.label)
    iou = per_class_iou(counts)

    def _r(v: float) -> float | None:
        return None if not np.isfinite(v) else round(float(v), 4)

    return {
        "n_pixels": int(valid.sum()),
        "pixel_acc_mc": _r(1 - wrong.mean()) if wrong.size else None,
        "miou_mc": _r(np.nanmean(iou)) if np.isfinite(iou).any() else None,
        "iou_mc": {n: _r(v) for n, v in zip(CLASS_NAMES, iou)},
        "entropy_correct_mean": _r(h[~wrong].mean()) if (~wrong).any() else None,
        "entropy_wrong_mean": _r(h[wrong].mean()) if wrong.any() else None,
        f"error_rate_top{int(top_frac * 100)}pct_uncertain": _r(wrong[hi].mean()) if hi.any() else None,
        f"error_rate_bottom{int((1 - top_frac) * 100)}pct_uncertain": _r(wrong[~hi].mean()) if (~hi).any() else None,
        "entropy_error_auroc": _r(auroc(h, wrong)),
        "mutual_info_mean": _r(pred.mutual_info[valid].mean()) if valid.any() else None,
        "confusion_mc_p_pred_given_true": row_normalise(counts).round(4).tolist(),
    }


def save_samples(s: SegSet, pred: MCPrediction, out: Path, n: int) -> list[Path]:
    """Write ``n`` evenly spaced PNGs: image | truth | MC prediction | predictive entropy."""
    from core.io import save_png  # noqa: F401  (ensures Agg backend)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    paths = []
    hmax = float(np.log(N_CLASSES))
    for k, i in enumerate(np.linspace(0, len(s) - 1, min(n, len(s))).astype(int)):
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        ax[0].imshow(s.images[i], cmap="gray"); ax[0].set_title(f"navcam #{i}")
        ax[1].imshow(colorize(s.labels[i])); ax[1].set_title("truth (black = unlabelled)")
        valid = s.labels[i] != IGNORE_INDEX
        acc = float((pred.label[i] == s.labels[i])[valid].mean()) if valid.any() else float("nan")
        ax[2].imshow(colorize(pred.label[i])); ax[2].set_title(f"MC prediction (acc {acc:.0%})")
        im = ax[3].imshow(pred.entropy[i], cmap="magma", vmin=0, vmax=hmax)
        ax[3].set_title(f"entropy, T={pred.n_passes} (nats)")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        for a in ax:
            a.axis("off")
        fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)],
                   loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        p = out / f"sample_{k}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> dict:
    cfg: Config = DEFAULT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="synthetic data + smoke checkpoint, CPU")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--passes", type=int, default=cfg.percep_mc_passes)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--samples", type=int, default=cfg.percep_n_samples)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    smoke = a.smoke
    default_dir = Path("outputs/perception_smoke") if smoke else Path(cfg.confusion_path).parent
    ckpt = Path(a.ckpt) if a.ckpt else default_dir / "unet.pt"
    out = Path(a.out) if a.out else (default_dir if smoke else Path("outputs/perception"))
    out.mkdir(parents=True, exist_ok=True)
    n_val = a.n_val or (8 if smoke else cfg.percep_n_val)
    size = a.size or (64 if smoke else cfg.percep_img_size)
    passes = 5 if smoke and a.passes == cfg.percep_mc_passes else a.passes
    batch = a.batch or (8 if smoke else cfg.percep_batch)

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint {ckpt} not found; run python -m perception.train"
                         f"{' --smoke' if smoke else ''} first")
    model = load_model(ckpt, cfg, device)
    s = synthetic_set(n_val, size, a.seed + 1) if smoke else load_ai4mars(VAL_SPLIT, n_val, size, a.seed)

    pred = predict_set(model, s, passes, batch, a.seed)
    rep = {"checkpoint": str(ckpt), "data": s.source, "n_images": len(s), "mc_passes": passes,
           **uncertainty_report(pred, s.labels)}
    samples = save_samples(s, pred, out, a.samples)
    (out / "infer_metrics.json").write_text(json.dumps(rep, indent=2))

    print(f"MC-dropout T={passes} on {len(s)} images ({s.source}), device={device}")
    print(f"  pixel acc {rep['pixel_acc_mc']}, mIoU {rep['miou_mc']}")
    print(f"  entropy correct {rep['entropy_correct_mean']} vs wrong {rep['entropy_wrong_mean']} nats; "
          f"AUROC(entropy -> error) {rep['entropy_error_auroc']}")
    print(f"  error rate: 10% most uncertain {rep['error_rate_top10pct_uncertain']} vs "
          f"rest {rep['error_rate_bottom90pct_uncertain']}")
    print(f"saved {len(samples)} sample PNGs + infer_metrics.json to {out}")
    return rep


if __name__ == "__main__":
    main()
