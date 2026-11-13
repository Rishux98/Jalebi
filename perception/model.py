"""AI4MARS terrain segmentation network + MC-dropout inference (shared by train.py / infer.py).

Two layers:
  * torch-free helpers (NumPy + PIL): preprocessing, AI4MARS label mapping, entropy,
    confusion matrix, per-class IoU. These run anywhere, including the local Windows venv.
  * torch helpers: ``build_unet`` (smp U-Net/resnet34 with Dropout2d before the seg head),
    ``enable_mc_dropout``, ``mc_predict`` (T stochastic passes -> mean probs + entropy),
    checkpoint save/load. torch is imported lazily so importing this module never fails.

Uncertainty outputs of ``mc_predict`` (per pixel, nats):
  * predictive entropy  H[mean_t p_t]           (total uncertainty)
  * expected entropy    mean_t H[p_t]           (aleatoric)
  * mutual information  H[mean p] - mean H[p]   (epistemic: what MC-dropout adds)

Run: python -m perception.model   (works without torch: runs the NumPy part only)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import DEFAULT, Config
from core.types import CLASS_NAMES, N_CLASSES

try:  # torch is optional locally; required on Colab for training / inference
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - depends on environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_F32 = np.float32
IGNORE_INDEX = 255                      # AI4MARS "null" / unlabelled pixels
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], _F32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], _F32)
_EPS = 1e-8


def has_torch() -> bool:
    """True if torch is importable in this environment."""
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("perception.model: torch + segmentation-models-pytorch are required for "
                          "this function (pip install torch segmentation-models-pytorch). "
                          "The NumPy helpers and perception.sensor_model work without them.")


# =========================================================================== torch-free helpers
def map_ai4mars_mask(mask: np.ndarray) -> np.ndarray:
    """Map a raw AI4MARS label PNG to our 4 classes.

    AI4MARS ids are 0=soil, 1=bedrock, 2=sand, 3=big_rock (same as ``core.types``); 255 = null.
    Anything outside 0..3 (null, rover body, range mask) becomes ``IGNORE_INDEX``.

    Args:
        mask: (H, W) or (H, W, C) integer label image (first channel used if multi-channel).

    Returns:
        (H, W) uint8 with values in {0, 1, 2, 3, 255}.
    """
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.int64)
    out = np.full(m.shape, IGNORE_INDEX, np.uint8)
    valid = (m >= 0) & (m < N_CLASSES)
    out[valid] = m[valid]
    return out


def resize_gray(img: Any, size: int = DEFAULT.percep_img_size) -> np.ndarray:
    """Navcam image (PIL / ndarray, gray or RGB, any size) -> (size, size) uint8 grayscale, bilinear."""
    from PIL import Image

    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    return np.asarray(pil.convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def normalise_gray(gray: np.ndarray) -> np.ndarray:
    """uint8 gray (..., S, S) -> (..., 3, S, S) float32, replicated to 3 channels and ImageNet-normalised."""
    x = np.asarray(gray, dtype=_F32)[..., None, :, :] / 255.0
    x = np.repeat(x, 3, axis=-3)
    return ((x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]).astype(_F32)


def preprocess_image(img: Any, size: int = DEFAULT.percep_img_size) -> np.ndarray:
    """Navcam image -> (3, size, size) float32 network input (gray -> 3ch so the ImageNet encoder fits)."""
    return normalise_gray(resize_gray(img, size))


def preprocess_mask(mask: Any, size: int = DEFAULT.percep_img_size) -> np.ndarray:
    """Raw AI4MARS label -> (size, size) int64 targets, nearest-neighbour resize (no label blending)."""
    from PIL import Image

    m = map_ai4mars_mask(np.asarray(mask))
    m = np.asarray(Image.fromarray(m).resize((size, size), Image.NEAREST))
    return m.astype(np.int64)


def entropy(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shannon entropy in nats along ``axis`` (class axis). Max for 4 classes = ln 4 ≈ 1.386."""
    p = np.asarray(probs, dtype=_F32)
    return (-(p * np.log(p + _EPS)).sum(axis=axis)).astype(_F32)


def confusion_counts(true: np.ndarray, pred: np.ndarray, n: int = N_CLASSES,
                     ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Raw (n, n) int64 counts C[t, p], skipping pixels whose true label is ``ignore``."""
    t = np.asarray(true).ravel().astype(np.int64)
    p = np.asarray(pred).ravel().astype(np.int64)
    keep = (t != ignore) & (t >= 0) & (t < n) & (p >= 0) & (p < n)
    return np.bincount(t[keep] * n + p[keep], minlength=n * n).reshape(n, n)


def row_normalise(counts: np.ndarray) -> np.ndarray:
    """Counts -> P(pred | true) float32; empty rows become identity rows (no evidence of confusion)."""
    c = np.asarray(counts, dtype=np.float64)
    rows = c.sum(axis=1, keepdims=True)
    out = np.where(rows > 0, c / np.maximum(rows, 1), np.eye(c.shape[0]))
    return out.astype(_F32)


def per_class_iou(counts: np.ndarray) -> np.ndarray:
    """IoU per class from confusion counts; NaN for classes absent from both truth and prediction."""
    c = np.asarray(counts, dtype=np.float64)
    tp = np.diag(c)
    denom = c.sum(0) + c.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan).astype(_F32)


# =========================================================================== torch helpers
@dataclass
class MCPrediction:
    """MC-dropout output for a batch. All arrays float32 except ``label`` (uint8).

    Attributes:
        mean_probs: (B, 4, H, W) mean softmax over T passes.
        label: (B, H, W) argmax class of ``mean_probs``.
        entropy: (B, H, W) predictive entropy, nats (total uncertainty).
        mutual_info: (B, H, W) epistemic part, nats (>= 0).
        n_passes: T.
    """
    mean_probs: np.ndarray
    label: np.ndarray
    entropy: np.ndarray
    mutual_info: np.ndarray
    n_passes: int


def build_unet(cfg: Config = DEFAULT, encoder_weights: str | None = "imagenet") -> "nn.Module":
    """smp.Unet(resnet34, 4 classes) with ``Dropout2d(cfg.percep_dropout)`` inserted before the seg head.

    Pass ``encoder_weights=None`` to skip the ImageNet download (tests, loading a checkpoint).
    """
    _require_torch()
    import segmentation_models_pytorch as smp

    model = smp.Unet("resnet34", encoder_weights=encoder_weights, in_channels=3, classes=N_CLASSES)
    model.segmentation_head = nn.Sequential(nn.Dropout2d(p=cfg.percep_dropout), model.segmentation_head)
    return model


def enable_mc_dropout(model: "nn.Module") -> int:
    """Eval mode everywhere (BatchNorm frozen) except dropout layers, which stay stochastic.

    Returns:
        number of dropout modules switched on (0 means MC-dropout would be deterministic).
    """
    _require_torch()
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


def mc_predict(model: "nn.Module", images: "np.ndarray | torch.Tensor", n_passes: int = DEFAULT.percep_mc_passes,
               seed: int = DEFAULT.seed, device: str | None = None) -> MCPrediction:
    """T stochastic forward passes with dropout on -> mean probs, predictive entropy, mutual information.

    Args:
        model: network from ``build_unet``.
        images: (B, 3, H, W) or (3, H, W) preprocessed float32 (see ``preprocess_image``).
        n_passes: T >= 1.
        seed: torch RNG seed for the dropout masks (deterministic results).
        device: "cuda" / "cpu"; default = the model's device.
    """
    _require_torch()
    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")
    x = torch.as_tensor(np.asarray(images, dtype=_F32) if not isinstance(images, torch.Tensor) else images)
    if x.ndim == 3:
        x = x[None]
    dev = torch.device(device) if device else next(model.parameters()).device
    x = x.to(dev, dtype=torch.float32)

    if enable_mc_dropout(model) == 0:
        raise RuntimeError("model has no dropout layers; MC-dropout needs build_unet(...)")
    gen_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        sum_p = torch.zeros((x.shape[0], N_CLASSES, *x.shape[2:]), device=dev)
        sum_h = torch.zeros((x.shape[0], *x.shape[2:]), device=dev)
        with torch.no_grad():
            for _ in range(n_passes):
                p = torch.softmax(model(x), dim=1)
                sum_p += p
                sum_h += -(p * torch.log(p + _EPS)).sum(dim=1)
    finally:  # don't perturb the caller's RNG streams
        torch.random.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    mean_p = (sum_p / n_passes).cpu().numpy().astype(_F32)
    pred_h = entropy(mean_p, axis=1)
    exp_h = (sum_h / n_passes).cpu().numpy().astype(_F32)
    return MCPrediction(
        mean_probs=mean_p,
        label=mean_p.argmax(axis=1).astype(np.uint8),
        entropy=pred_h,
        mutual_info=np.clip(pred_h - exp_h, 0.0, None).astype(_F32),
        n_passes=n_passes,
    )


def save_checkpoint(model: "nn.Module", path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    """Save ``state_dict`` + metadata (config values, metrics) to ``path`` (e.g. perception/unet.pt)."""
    _require_torch()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, p)
    return p


def load_model(path: str | Path, cfg: Config = DEFAULT, device: str = "cpu") -> "nn.Module":
    """Rebuild the network (no ImageNet download) and load weights from ``save_checkpoint``."""
    _require_torch()
    ckpt = torch.load(Path(path), map_location=device)
    model = build_unet(cfg, encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model.to(device)


# =========================================================================== smoke demo
if __name__ == "__main__":
    rng = np.random.default_rng(DEFAULT.seed)
    S = 64

    # --- NumPy part (always runs)
    raw_img = (rng.random((120, 160)) * 255).astype(np.uint8)
    raw_mask = rng.choice(np.array([0, 1, 2, 3, 255], np.uint8), size=(120, 160))
    x = preprocess_image(raw_img, S)
    y = preprocess_mask(raw_mask, S)
    print(f"preprocess_image -> {x.shape} {x.dtype}; preprocess_mask -> {y.shape}, labels {np.unique(y)}")

    true = y
    pred = np.where(rng.random(y.shape) < 0.8, y, rng.integers(0, N_CLASSES, y.shape))
    counts = confusion_counts(true, pred)
    iou = per_class_iou(counts)
    print("toy confusion P(pred|true):\n", np.round(row_normalise(counts), 3))
    print("toy IoU: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(CLASS_NAMES, iou)))

    # --- torch part
    if not has_torch():
        print("torch not installed: skipping network demo (train/infer run on Colab).")
    else:
        model = build_unet(encoder_weights=None)
        n_drop = enable_mc_dropout(model)
        mc = mc_predict(model, np.stack([x, x]), n_passes=5)
        print(f"build_unet: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, "
              f"{n_drop} dropout layer(s)")
        print(f"mc_predict T={mc.n_passes}: probs {mc.mean_probs.shape}, entropy "
              f"[{mc.entropy.min():.3f}, {mc.entropy.max():.3f}] nats, mean MI {mc.mutual_info.mean():.4f}")

        from core.io import save_png  # noqa: F401  (ensures Agg backend)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
        ax[0].imshow(raw_img, cmap="gray"); ax[0].set_title("input (random)")
        ax[1].imshow(mc.label[0], cmap="tab10", vmin=0, vmax=9); ax[1].set_title("MC label (untrained)")
        im = ax[2].imshow(mc.entropy[0], cmap="magma", vmin=0, vmax=np.log(N_CLASSES))
        ax[2].set_title("predictive entropy (nats)")
        fig.colorbar(im, ax=ax[2], fraction=0.046)
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        Path("outputs").mkdir(exist_ok=True)
        fig.savefig("outputs/perception_model_demo.png", dpi=100)
        plt.close(fig)
        print("saved outputs/perception_model_demo.png")
