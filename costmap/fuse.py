"""Fuse terrain layers + class belief into a traversal cost map with a per-term breakdown.

Cost (dimensionless, per metre; A* pays ``dist_m * (1 + cost)``):
    cost = slope + rough + class + energy + uncertainty            (each already weighted)
      slope       = w_slope  * clip(slope_deg / slope_max_deg, 0, 1)
      rough       = w_rough  * clip(roughness / rough_ref_m, 0, 1)
      class       = w_class  * sum_k P(k) * class_risk[k]           (expected class risk)
      energy      = w_energy * energy_k_uphill * tan(slope)         (extra work over flat ground)
      uncertainty = lambda_unc * entropy / ln(4)                    (belief entropy, normalised)
    cost = inf  where slope_deg > slope_max_deg  or  P(big_rock) > big_rock_impassable_p

Hazard (probability that traversing the cell ends the mission):
    hazard_prob = 1 - (1 - h_slope * p_slope_exceed)
                    * (1 - h_rock  * P(big_rock))
                    * (1 - h_sand  * P(sand))

Energy model (gplanetary-nav idea, isotropic simplification): the uphill grade of a move is
direction-dependent; a cell-based map uses the worst-case grade tan(slope), so
``energy_wh_per_m = energy_wh_per_m_flat * (1 + energy_k_uphill * tan(slope))``.

All ops are per-cell, so the function works unchanged on a sub-window (P4 local rebuilds).
Run: python -m costmap.fuse   (writes outputs/costmap.png)
"""
from __future__ import annotations

import numpy as np

from core.config import DEFAULT, Config
from core.types import BIG_ROCK, N_CLASSES, SAND, Belief, CostMap, TerrainLayers

_F32 = np.float32
COMPONENT_NAMES: tuple[str, ...] = ("slope", "rough", "class", "energy", "uncertainty")


def one_hot_probs(true_class: np.ndarray) -> np.ndarray:
    """(H, W) int class map -> (H, W, 4) float32 one-hot probabilities (perfect knowledge)."""
    return np.eye(N_CLASSES, dtype=_F32)[np.asarray(true_class, dtype=np.int64)]


def hazard_probability(p_slope_exceed: np.ndarray, class_probs: np.ndarray,
                       cfg: Config = DEFAULT) -> np.ndarray:
    """Per-cell mission-loss probability in [0, 1] (formula in module docstring)."""
    p_rock = class_probs[..., BIG_ROCK]
    p_sand = class_probs[..., SAND]
    survive = ((1.0 - cfg.hazard_slope_factor * p_slope_exceed)
               * (1.0 - cfg.hazard_rock_factor * p_rock)
               * (1.0 - cfg.sand_hazard_factor * p_sand))
    return np.clip(1.0 - survive, 0.0, 1.0).astype(_F32)


def build_costmap(layers: TerrainLayers, class_probs: np.ndarray, cfg: Config = DEFAULT,
                  lambda_unc: float | None = None, entropy: np.ndarray | None = None) -> CostMap:
    """Fuse terrain + class belief into a CostMap.

    Args:
        layers: terrain features (slope deg, roughness m, p_slope_exceed).
        class_probs: (H, W, 4) per-cell class probabilities (one-hot truth before P3, belief after).
        cfg: weights and thresholds.
        lambda_unc: entropy weight; defaults to ``cfg.lambda_unc``.
        entropy: (H, W) entropy in nats; computed from ``class_probs`` if omitted.

    Returns:
        CostMap with ``cost`` (inf = impassable), ``hazard_prob``, weighted ``components``
        (which sum to ``cost`` on passable cells) and ``energy_wh_per_m``.
    """
    probs = np.asarray(class_probs, dtype=_F32)
    hw = layers.shape
    if probs.shape != (*hw, N_CLASSES):
        raise ValueError(f"class_probs shape {probs.shape} != {(*hw, N_CLASSES)}")
    lam = cfg.lambda_unc if lambda_unc is None else float(lambda_unc)
    ent = Belief.entropy_of(probs) if entropy is None else np.asarray(entropy, dtype=_F32)

    slope = layers.slope_deg
    tan_s = np.tan(np.radians(np.minimum(slope, 89.0)))
    risk = np.asarray([cfg.class_risk[k] for k in range(N_CLASSES)], dtype=_F32)

    comps = {
        "slope": cfg.w_slope * np.clip(slope / cfg.slope_max_deg, 0.0, 1.0),
        "rough": cfg.w_rough * np.clip(layers.roughness / cfg.rough_ref_m, 0.0, 1.0),
        "class": cfg.w_class * (probs @ risk),
        "energy": cfg.w_energy * cfg.energy_k_uphill * tan_s,
        "uncertainty": lam * ent / np.log(N_CLASSES),
    }
    comps = {k: v.astype(_F32) for k, v in comps.items()}
    cost = np.sum(np.stack(list(comps.values())), axis=0).astype(_F32)

    impassable = (slope > cfg.slope_max_deg) | (probs[..., BIG_ROCK] > cfg.big_rock_impassable_p)
    cost[impassable] = np.inf

    energy = (cfg.energy_wh_per_m_flat * (1.0 + cfg.energy_k_uphill * tan_s)).astype(_F32)
    return CostMap(cost=cost, hazard_prob=hazard_probability(layers.p_slope_exceed, probs, cfg),
                   components=comps, energy_wh_per_m=energy)


def costmap_from_belief(layers: TerrainLayers, belief: Belief, cfg: Config = DEFAULT,
                        lambda_unc: float | None = None) -> CostMap:
    """CostMap from the rover's belief (P3+): class terms use ``belief.class_probs``,
    the uncertainty term uses ``belief.entropy`` (nats)."""
    return build_costmap(layers, belief.class_probs, cfg, lambda_unc=lambda_unc, entropy=belief.entropy)


if __name__ == "__main__":
    import time
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from terrain.features import compute_layers
    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    L = compute_layers(w)
    t0 = time.perf_counter()
    cm = build_costmap(L, one_hot_probs(w.true_class))
    dt = time.perf_counter() - t0
    fin = np.isfinite(cm.cost)
    print(f"build_costmap(200x200) in {dt * 1000:.1f} ms; impassable {(~fin).mean():.1%}")
    print(f"  cost (finite) min {cm.cost[fin].min():.3f} median {np.median(cm.cost[fin]):.3f} "
          f"max {cm.cost[fin].max():.3f}")
    print(f"  hazard_prob max {cm.hazard_prob.max():.4f}, mean {cm.hazard_prob.mean():.5f}")
    for k, v in cm.components.items():
        print(f"  component {k:12s} mean {v[fin].mean():.3f}")
    print(f"  breakdown at start {w.start}: {cm.breakdown(w.start)}")

    out = Path("outputs/costmap.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    c = np.ma.masked_invalid(cm.cost)
    im = axes[0].imshow(c, cmap=plt.get_cmap("viridis").with_extremes(bad="black"))
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("cost (black = impassable)")
    im = axes[1].imshow(cm.hazard_prob, cmap="inferno")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("hazard_prob per cell")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("saved", out)
