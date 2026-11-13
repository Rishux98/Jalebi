"""Smoke tests for perception/ (NumPy parts always; torch parts skipped if torch is absent)."""
from __future__ import annotations

import numpy as np
import pytest

from core.types import N_CLASSES
from perception import model as pm
from perception import sensor_model as sm


# ----------------------------------------------------------------- sensor_model
def test_load_confusion_fallback(tmp_path):
    c, src = sm.load_confusion(tmp_path / "missing.npy")
    assert src == "fallback" and c.shape == (4, 4) and c.dtype == np.float32
    assert np.allclose(c.sum(1), 1.0)


def test_load_confusion_file_normalised(tmp_path):
    p = tmp_path / "c.npy"
    np.save(p, np.eye(4) * 5 + 1)
    c, src = sm.load_confusion(p)
    assert src == str(p) and np.allclose(c.sum(1), 1.0)


def test_observe_shape_and_identity():
    t = np.random.default_rng(0).integers(0, 4, (32, 32)).astype(np.int8)
    obs = sm.observe(t, np.random.default_rng(1), np.eye(4, dtype=np.float32))
    assert obs.shape == t.shape and obs.dtype == np.int8
    assert np.array_equal(obs, t)


def test_observe_window_and_likelihood():
    t = np.zeros((32, 32), np.int8)
    obs, mask = sm.observe_window(t, (16, 16), 4, np.random.default_rng(0))
    assert mask.sum() > 0 and (obs[~mask] == sm.UNOBSERVED).all()
    lik = sm.likelihood(obs)
    assert lik.shape == (32, 32, 4) and lik.dtype == np.float32
    assert np.allclose(lik[~mask], 1.0)


# ----------------------------------------------------------------- model (NumPy helpers)
def test_map_mask_and_preprocess():
    raw = np.array([[0, 1, 2], [3, 255, 7]], np.uint8)
    assert pm.map_ai4mars_mask(raw).tolist() == [[0, 1, 2], [3, 255, 255]]
    x = pm.preprocess_image(np.zeros((20, 30), np.uint8), 32)
    assert x.shape == (3, 32, 32) and x.dtype == np.float32
    y = pm.preprocess_mask(np.full((20, 30), 2, np.uint8), 32)
    assert y.shape == (32, 32) and set(np.unique(y)) == {2}


def test_confusion_iou_entropy():
    true = np.array([0, 0, 1, 2, 3, 255])
    pred = np.array([0, 1, 1, 2, 3, 0])
    counts = pm.confusion_counts(true, pred)
    assert counts.sum() == 5                       # ignore pixel dropped
    c = pm.row_normalise(counts)
    assert np.allclose(c.sum(1), 1.0) and c.dtype == np.float32
    iou = pm.per_class_iou(counts)
    assert iou[2] == pytest.approx(1.0) and iou[0] == pytest.approx(0.5)
    h = pm.entropy(np.full((4, 2, 2), 0.25, np.float32))
    assert h.shape == (2, 2) and np.allclose(h, np.log(4), atol=1e-4)


# ----------------------------------------------------------------- data
def test_synthetic_set_and_row_to_example():
    from perception.data import row_to_example, synthetic_set

    s = synthetic_set(4, 32, seed=0)
    assert s.images.shape == (4, 32, 32) and s.images.dtype == np.uint8
    assert set(np.unique(s.labels)) <= {0, 1, 2, 3, 255}
    lbl = np.full((40, 40), 1, np.uint8)
    rover = np.zeros((40, 40), np.uint8); rover[30:] = 1
    g, l = row_to_example({"image": np.zeros((40, 40), np.uint8), "label_mask": lbl,
                           "rover_mask": rover, "range_mask": None}, 20)
    assert g.shape == l.shape == (20, 20)
    assert (l[16:] == 255).all() and (l[:14] == 1).all()


def test_auroc_and_colorize():
    from perception.infer import auroc, colorize

    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert auroc(np.ones(4), np.array([0, 1, 0, 1])) == pytest.approx(0.5)
    rgb = colorize(np.array([[0, 255]], np.uint8))
    assert rgb.shape == (1, 2, 3) and (rgb[0, 1] == 0).all()


# ----------------------------------------------------------------- model (torch)
def test_train_and_infer_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from perception import infer, train

    out = tmp_path / "p"
    paths = train.main(["--smoke", "--n-train", "8", "--n-val", "4", "--size", "32", "--epochs", "1",
                        "--batch", "4", "--out", str(out)])
    c = np.load(paths["confusion"])
    assert c.shape == (4, 4) and np.allclose(c.sum(1), 1.0, atol=1e-5)
    rep = infer.main(["--smoke", "--ckpt", str(paths["unet"]), "--n-val", "4", "--size", "32",
                      "--passes", "2", "--samples", "1", "--out", str(out)])
    assert rep["n_images"] == 4 and (out / "sample_0.png").exists()
    assert (out / "infer_metrics.json").exists()


def test_mc_predict_shapes():
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    net = pm.build_unet(encoder_weights=None)
    assert pm.enable_mc_dropout(net) >= 1
    x = np.zeros((1, 3, 64, 64), np.float32)
    a = pm.mc_predict(net, x, n_passes=3, seed=0)
    b = pm.mc_predict(net, x, n_passes=3, seed=0)
    assert a.mean_probs.shape == (1, N_CLASSES, 64, 64) and a.entropy.shape == (1, 64, 64)
    assert a.entropy.dtype == np.float32 and (a.mutual_info >= 0).all()
    assert np.allclose(a.mean_probs.sum(1), 1.0, atol=1e-4)
    assert np.allclose(a.mean_probs, b.mean_probs)   # seeded -> deterministic
