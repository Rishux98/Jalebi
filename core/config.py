"""All ODYSSEY weights and thresholds in one place. Modules must not hard-code magic numbers.

Usage:
    from core.config import Config, DEFAULT
    cfg = DEFAULT.replace(lambda_unc=2.0)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


def _default_class_risk() -> dict[int, float]:
    # 0=soil, 1=bedrock, 2=sand, 3=big_rock (1.0 => impassable when confident)
    return {0: 0.05, 1: 0.10, 2: 0.45, 3: 1.0}


@dataclass(frozen=True)
class Config:
    """Every tunable of the system. Units in comments."""

    # ---- global
    seed: int = 42                      # default RNG seed
    cell_size: float = 1.0              # m per cell

    # ---- slope limits (deg)
    slope_safe_deg: float = 15.0        # below: safe
    slope_caution_deg: float = 25.0     # 15..25: caution
    slope_max_deg: float = 25.0         # above: impassable

    # ---- terrain classes
    class_risk: dict[int, float] = field(default_factory=_default_class_risk)  # per-class risk in [0,1]
    big_rock_impassable_p: float = 0.6  # P(big_rock) above this -> cost inf

    # ---- per-cell hazard model (probability that traversing ONE cell ends the mission)
    # hazard_prob = 1 - (1 - h_slope*p_slope_exceed) * (1 - h_rock*P(big_rock)) * (1 - h_sand*P(sand))
    # (ROADMAP formula with unit factors 1/1/0.5; scaled down because success_prob is a product
    #  over ~250 path cells — with 0.5 per sand cell any route touching sand would score ~0 %.)
    hazard_slope_factor: float = 0.02   # mission loss per cell if slope is truly over the limit
    hazard_rock_factor: float = 0.5     # per cell, times P(big_rock)
    sand_hazard_factor: float = 0.004   # per cell, times P(sand) (embedding risk, cf. Spirit)

    # ---- cost weights (dimensionless)
    w_slope: float = 1.0
    w_rough: float = 0.5
    w_class: float = 2.0
    w_energy: float = 0.5
    lambda_unc: float = 1.0             # weight of belief entropy in cost
    energy_k_uphill: float = 3.0        # energy/m factor: 1 + k * max(0, dz/ds)
    energy_wh_per_m_flat: float = 1.5   # Wh per metre on flat ground (rover energy model)

    # ---- terrain features
    roughness_window: int = 5           # cells, odd
    rough_ref_m: float = 0.25           # m; roughness normaliser (rough/ref clipped to [0, 1])
    dem_noise_sigma: float = 0.3        # m, orbital DEM 1-sigma error
    dem_noise_corr_cells: float = 3.0   # cells; Gaussian correlation length of DEM error
    mc_samples: int = 30                # Monte-Carlo draws for P(slope > limit)
    mc_max_bytes: float = 200e6         # memory cap per Monte-Carlo batch

    # ---- perception (P2; training runs on Colab GPU)
    confusion_path: str = "perception/confusion.npy"  # 4x4 row-normalised P(pred | true)
    percep_img_size: int = 256          # px, square resize
    percep_n_train: int = 2000          # streamed labelled train images
    percep_n_val: int = 322             # val = full AI4MARS test_min3 split (3-labeller agreement)
    percep_epochs: int = 8
    percep_batch: int = 16
    percep_lr: float = 3e-4
    percep_weight_decay: float = 1e-4   # AdamW
    percep_class_weight_power: float = 0.5  # CE weight ∝ class_freq^-power (rare big_rock up-weighted)
    percep_n_samples: int = 6           # sample PNGs written by perception/infer.py
    percep_dropout: float = 0.2         # Dropout2d before the seg head (MC-dropout)
    percep_mc_passes: int = 15          # T stochastic forward passes at inference

    # ---- orbital prior belief (P3): simulated low-res orbital classification
    prior_downsample: int = 8           # x; block size of the orbital pixel (majority vote per block)
    prior_label_noise: float = 0.2      # fraction of orbital pixels relabelled (draws follow orbital class freq)
    prior_blur_sigma: float = 4.0       # cells; Gaussian blur of the upsampled one-hot map
    prior_temperature: float = 1.5      # >1 flattens the soft probs (p ** (1/T), renormalised)
    prior_floor: float = 0.002          # min prior prob per class so Bayes can still discover hazards.
                                        # Calibrated: 0.03 -> every ~300-cell route scores P(success)~0
                                        # (0.5 * 0.03 rock hazard per cell); 0.002 -> 62-74 % on safe
                                        # routes, hidden boulders reach P(big_rock) ~0.5 after 2-3 looks.

    # ---- multi-objective planning (P3)
    pareto_risk_balanced: float = 2.0   # A* risk_weight for "balanced" / "science"
    pareto_risk_safest: float = 50.0    # A* risk_weight for "safest"
    pareto_lambda_unc_safest: float = 2.0  # entropy weight for "safest" (avoid unknown terrain; 3.0 cost
                                        # ~4 pts of modelled P(success) for little extra entropy reduction)
    pareto_sweep: tuple[float, float, int] = (0.0, 10.0, 11)  # risk_weight linspace(lo, hi, n)

    # ---- science targets (P4)
    sci_w_ice: float = 0.4              # science value = w_ice*ice + w_rim*rim + w_bedrock*P(bedrock)
    sci_w_rim: float = 0.3              #   (weights sum to 1 -> value in [0, 1])
    sci_w_bedrock: float = 0.3
    sci_slope_center_deg: float = 12.0  # deg; rim/layering proxy favours moderate slopes ...
    sci_slope_width_deg: float = 6.0    # deg; ... Gaussian band around the centre
    sci_rim_curv_pct: float = 95.0      # percentile of positive curvature used to normalise rim proxy
    sci_n_targets: int = 8              # candidates kept after non-max suppression
    sci_nms_cells: int = 15             # cells; NMS window (min spacing between targets ~ half of it)
    sci_target_radius: int = 3          # cells; disc averaged for a target's value / entropy
    sci_bootstrap_n: int = 50           # bootstrap samples for value_mean / value_std
    sci_ice_noise_std: float = 0.1      # 1-sigma noise of the orbital ice signal (bootstrap)
    sci_alpha_per_cost: float = 0.0005  # utility penalty per unit of A* path cost (~0.15 for 300)
    sci_beta_risk: float = 0.5          # utility penalty per unit of (1 - success_prob)
    sci_min_utility: float = 0.0        # targets at or below this utility are skipped
    sci_energy_budget_wh: float = 800.0  # Wh for start -> targets -> goal. Direct traverse ~545 Wh on
                                        # 200x200; 800 selects 5-6 of 8 targets on seeds 3/7/42 and the
                                        # budget (not utility) is the binding constraint. (0.001/1.0/1500:
                                        # 0-8 picks depending on seed, budget never binding.)

    # ---- sensing / autonomy
    sensor_radius: int = 6              # cells
    hazard_prob_max: float = 0.3        # replan if a path cell exceeds this (also a hard limit when planning)
    replan_cost_increase: float = 0.20  # replan if path cost rises by > 20 %
    loop_near_frac: float = 0.5         # log keep_path when a check reaches this fraction of its threshold
    loop_max_steps: int = 2000          # safety cap on autonomy-loop steps
    loop_snapshot_every: int = 10       # steps between stored belief-entropy snapshots (dashboard slider)

    # ---- synthetic world generator
    gen_noise_octaves: int = 5          # fractal value-noise octaves
    gen_noise_base_cells: int = 3       # coarsest lattice size (cells across the grid)
    gen_noise_amp_m: float = 3.0        # m, amplitude of base terrain
    gen_noise_persistence: float = 0.35 # amplitude decay per octave
    gen_n_craters: tuple[int, int] = (6, 10)          # inclusive range
    gen_crater_radius_frac: tuple[float, float] = (0.04, 0.10)  # of min(H, W)
    gen_crater_depth_per_r: tuple[float, float] = (0.12, 0.32)  # crater depth (m) per cell of radius (per-crater draw)
    gen_rim_height_frac: float = 0.35     # rim height as fraction of depth
    gen_ridge_height_m: float = 14.0      # m
    gen_ridge_width_frac: float = 0.025   # ridge Gaussian sigma as fraction of min(H, W)
    gen_basin_depth_m: float = 6.0        # m
    gen_basin_radius_frac: float = 0.22   # of min(H, W)
    gen_sand_percentile: float = 65.0     # basin cells below this DEM percentile (within basin) -> sand
    gen_rock_cluster_per_crater: int = 5  # boulder clusters seeded per crater rim
    gen_rock_cluster_radius: tuple[float, float] = (1.5, 3.5)  # cells
    gen_n_ice_craters: tuple[int, int] = (2, 3)       # inclusive range
    gen_corner_margin_frac: float = 0.06  # start/goal inset from corners
    gen_max_regen_attempts: int = 20
    gen_reference_size: int = 200       # cells; vertical amplitudes (m) are scaled by min(H, W)/this
                                        # so slopes stay scale-invariant on small test grids
    gen_ridge_min_sigma_cells: float = 2.0

    def replace(self, **kw: Any) -> "Config":
        """Return a copy with some fields changed."""
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


DEFAULT = Config()

__all__ = ["Config", "DEFAULT"]
