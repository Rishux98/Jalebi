"""Tests for belief/bayes.py — orbital prior and Bayesian class update."""
from __future__ import annotations

import numpy as np
import pytest

from belief.bayes import _block_majority, belief_accuracy, init_prior, sense, update
from core.config import DEFAULT
from core.types import N_CLASSES, Belief
from perception.sensor_model import FALLBACK_CONFUSION, UNOBSERVED
from world.synthetic import make_world

EYE = np.eye(N_CLASSES, dtype=np.float32)


@pytest.fixture(scope="module")
def world():
    return make_world(32, 32, seed=3)


def test_prior_shape_dtype_normalised_floor(world):
    b = init_prior(world)
    assert b.class_probs.shape == (32, 32, N_CLASSES) and b.class_probs.dtype == np.float32
    assert b.entropy.shape == (32, 32) and b.entropy.dtype == np.float32
    assert np.allclose(b.class_probs.sum(-1), 1.0, atol=1e-5)
    assert b.class_probs.min() >= DEFAULT.prior_floor - 1e-6        # every class discoverable
    assert not b.observed_mask.any()
    assert (b.entropy >= 0).all() and (b.entropy <= np.log(N_CLASSES) + 1e-5).all()


def test_prior_deterministic_and_seed_sensitive(world):
    a, b = init_prior(world, seed=1), init_prior(world, seed=1)
    assert np.array_equal(a.class_probs, b.class_probs)
    assert not np.array_equal(a.class_probs, init_prior(world, seed=2).class_probs)


def test_prior_beats_chance(world):
    assert belief_accuracy(init_prior(world), world.true_class) > 0.5


def test_block_majority():
    cls = np.array([[0, 0, 3, 3],
                    [0, 1, 3, 2],
                    [2, 2, 1, 1]])
    out = _block_majority(cls, 2)                  # 3 rows -> 2 blocks (last row edge-padded)
    assert out.tolist() == [[0, 3], [2, 1]]


def test_update_identity_confusion_gives_one_hot():
    b = Belief.uniform(8, 8)
    obs = np.full((8, 8), UNOBSERVED, np.int8)
    obs[2:4, 2:4] = 3
    post = update(b, obs, confusion=EYE)
    assert np.allclose(post.class_probs[2:4, 2:4], EYE[3])
    assert np.allclose(post.class_probs[0, 0], 0.25)                 # unobserved unchanged
    assert post.observed_mask[2:4, 2:4].all() and post.observed_mask.sum() == 4
    assert post.entropy[2, 2] == pytest.approx(0.0, abs=1e-4)


def test_update_does_not_mutate_and_reduces_entropy():
    b = Belief.uniform(6, 6)
    before = b.class_probs.copy()
    obs = np.zeros((6, 6), np.int8)
    post = update(b, obs, confusion=FALLBACK_CONFUSION)
    assert np.array_equal(b.class_probs, before) and not b.observed_mask.any()
    assert (post.entropy < b.entropy).all()
    assert post.class_probs.dtype == np.float32 and np.allclose(post.class_probs.sum(-1), 1.0, atol=1e-5)


def test_stacked_observations_equal_sequential_updates():
    rng = np.random.default_rng(0)
    b = Belief.from_probs(rng.random((5, 5, N_CLASSES)).astype(np.float32) + 0.1)
    o1 = rng.integers(0, N_CLASSES, (5, 5)).astype(np.int8)
    o2 = rng.integers(0, N_CLASSES, (5, 5)).astype(np.int8)
    o2[0, :] = UNOBSERVED
    seq = update(update(b, o1), o2)
    stk = update(b, np.stack([o1, o2]))
    assert np.allclose(seq.class_probs, stk.class_probs, atol=1e-5)


def test_mask_marks_observed_and_shape_check():
    b = Belief.uniform(4, 4)
    obs = np.full((4, 4), UNOBSERVED, np.int8)
    m = np.zeros((4, 4), bool); m[0, 0] = True
    assert update(b, obs, m).observed_mask[0, 0]
    with pytest.raises(ValueError):
        update(b, np.zeros((3, 4), np.int8))


def test_repeated_sensing_discovers_hidden_boulders(world):
    w = make_world(32, 32, seed=3)
    w.inject_hazard((16, 16), 2, "big_rock")
    b = init_prior(w)
    rng = np.random.default_rng(0)
    for _ in range(4):
        b = sense(b, w, (16, 16), rng, DEFAULT, FALLBACK_CONFUSION, radius=4)
    rock = w.true_class == 3
    assert b.class_probs[..., 3][rock & b.observed_mask].mean() > 0.5
    assert b.observed_mask[16, 16] and not b.observed_mask[0, 0]
