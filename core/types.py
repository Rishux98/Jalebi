"""ODYSSEY data contracts — shared by every module. FROZEN after P0.

Grid conventions (see CLAUDE.md):
  * maps are ``np.ndarray`` of shape ``(H, W)`` (or ``(H, W, K)``), dtype ``float32``, indexed ``[row, col]``
  * cells are ``(r, c)`` tuples of ints, never ``(x, y)``
  * ``cell_size`` in metres, elevation in metres, slope in degrees
  * impassable cost is ``np.inf``
  * terrain classes (AI4MARS): 0=soil, 1=bedrock, 2=sand, 3=big_rock

Changing a field here breaks the team. If unavoidable: add an optional field with a default,
grep all callers, and announce it at the top of the phase summary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- constants
RC = tuple[int, int]
"""A grid cell as ``(row, col)``."""

SOIL, BEDROCK, SAND, BIG_ROCK = 0, 1, 2, 3
CLASS_NAMES: tuple[str, ...] = ("soil", "bedrock", "sand", "big_rock")
N_CLASSES: int = 4

_F32 = np.float32


def _as_f32(a: np.ndarray, name: str, ndim: int) -> np.ndarray:
    """Cast to float32 (no copy if already) and check dimensionality."""
    a = np.asarray(a).astype(_F32, copy=False)
    if a.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {a.shape}")
    return a


def _check_shape(a: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} != expected {shape}")


def _to_rc(rc: Any) -> RC:
    r, c = rc
    return (int(r), int(c))


# --------------------------------------------------------------------------- World
@dataclass
class World:
    """Hidden ground truth of the simulated site. The rover never reads it directly.

    Attributes:
        dem: (H, W) float32 elevation in metres.
        cell_size: grid resolution in metres per cell.
        true_class: (H, W) int8 terrain class (0=soil, 1=bedrock, 2=sand, 3=big_rock).
        ice_signal: (H, W) float32 in [0, 1], subsurface-ice likelihood (science value proxy).
        start: rover start cell (r, c).
        goal: mission goal cell (r, c).
        seed: RNG seed actually used to build this world (may differ from the requested
            seed if the generator had to regenerate for connectivity).
        name: free-form label (e.g. "synthetic" or a HiRISE tile id).
    """

    dem: np.ndarray
    cell_size: float
    true_class: np.ndarray
    ice_signal: np.ndarray
    start: RC
    goal: RC
    seed: int
    name: str = "synthetic"

    def __post_init__(self) -> None:
        self.dem = _as_f32(self.dem, "dem", 2)
        hw = self.dem.shape
        self.true_class = np.asarray(self.true_class).astype(np.int8, copy=False)
        _check_shape(self.true_class, "true_class", hw)
        self.ice_signal = _as_f32(self.ice_signal, "ice_signal", 2)
        _check_shape(self.ice_signal, "ice_signal", hw)
        self.cell_size = float(self.cell_size)
        if self.cell_size <= 0:
            raise ValueError(f"cell_size must be > 0 m, got {self.cell_size}")
        self.start = _to_rc(self.start)
        self.goal = _to_rc(self.goal)
        for nm, rc in (("start", self.start), ("goal", self.goal)):
            if not self.in_bounds(rc):
                raise ValueError(f"{nm} {rc} out of bounds for grid {hw}")
        self.seed = int(self.seed)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (H, W)."""
        return self.dem.shape  # type: ignore[return-value]

    def in_bounds(self, rc: RC) -> bool:
        """True if cell (r, c) lies inside the grid."""
        r, c = rc
        h, w = self.dem.shape
        return 0 <= r < h and 0 <= c < w

    def inject_hazard(self, rc: RC, radius: float, kind: str = "big_rock") -> np.ndarray:
        """Plant a hidden hazard disc into ``true_class`` only (DEM and ice untouched).

        Used in P4: the rover's belief is unaware until it senses the area.

        Args:
            rc: centre cell (r, c).
            radius: disc radius in cells (>= 0; radius 0 hits only the centre cell).
            kind: ``"big_rock"`` (boulder field) or ``"sand"`` (sand trap).

        Returns:
            (H, W) bool mask of the cells that were set to the hazard class.
        """
        kinds = {"big_rock": BIG_ROCK, "sand": SAND}
        if kind not in kinds:
            raise ValueError(f"kind must be one of {sorted(kinds)}, got {kind!r}")
        rc = _to_rc(rc)
        if not self.in_bounds(rc):
            raise ValueError(f"hazard centre {rc} out of bounds for grid {self.shape}")
        if radius < 0:
            raise ValueError(f"radius must be >= 0 cells, got {radius}")
        h, w = self.shape
        rr, cc = np.ogrid[:h, :w]
        mask = (rr - rc[0]) ** 2 + (cc - rc[1]) ** 2 <= float(radius) ** 2
        self.true_class[mask] = kinds[kind]
        return mask


# --------------------------------------------------------------------------- TerrainLayers
@dataclass
class TerrainLayers:
    """Geometric terrain features derived from a DEM. All (H, W) float32.

    Attributes:
        slope_deg: terrain slope in degrees.
        roughness: local elevation std after detrending, in metres.
        curvature: Laplacian of elevation, in 1/m (positive = convex / ridge-like).
        p_slope_exceed: Monte-Carlo probability in [0, 1] that true slope exceeds the
            impassable limit, given DEM noise.
    """

    slope_deg: np.ndarray
    roughness: np.ndarray
    curvature: np.ndarray
    p_slope_exceed: np.ndarray

    def __post_init__(self) -> None:
        self.slope_deg = _as_f32(self.slope_deg, "slope_deg", 2)
        hw = self.slope_deg.shape
        for nm in ("roughness", "curvature", "p_slope_exceed"):
            a = _as_f32(getattr(self, nm), nm, 2)
            _check_shape(a, nm, hw)
            setattr(self, nm, a)

    @property
    def shape(self) -> tuple[int, int]:
        return self.slope_deg.shape  # type: ignore[return-value]


# --------------------------------------------------------------------------- Belief
@dataclass
class Belief:
    """Rover's per-cell belief over terrain classes.

    Attributes:
        class_probs: (H, W, 4) float32, each cell sums to 1 (soil, bedrock, sand, big_rock).
        entropy: (H, W) float32 Shannon entropy in nats, range [0, ln 4].
        observed_mask: (H, W) bool, True where the rover has sensed the cell at least once.
    """

    class_probs: np.ndarray
    entropy: np.ndarray
    observed_mask: np.ndarray

    def __post_init__(self) -> None:
        self.class_probs = _as_f32(self.class_probs, "class_probs", 3)
        h, w, k = self.class_probs.shape
        if k != N_CLASSES:
            raise ValueError(f"class_probs last dim must be {N_CLASSES}, got {k}")
        self.entropy = _as_f32(self.entropy, "entropy", 2)
        _check_shape(self.entropy, "entropy", (h, w))
        self.observed_mask = np.asarray(self.observed_mask, dtype=bool)
        _check_shape(self.observed_mask, "observed_mask", (h, w))

    @property
    def shape(self) -> tuple[int, int]:
        return self.entropy.shape  # type: ignore[return-value]

    @staticmethod
    def entropy_of(class_probs: np.ndarray) -> np.ndarray:
        """Shannon entropy (nats) over the last axis; (H, W, K) -> (H, W) float32."""
        p = np.clip(class_probs, 1e-12, 1.0)
        return (-(p * np.log(p)).sum(axis=-1)).astype(_F32)

    @classmethod
    def from_probs(cls, class_probs: np.ndarray, observed_mask: np.ndarray | None = None) -> "Belief":
        """Build a Belief from (H, W, 4) probs; normalises and computes entropy."""
        p = np.asarray(class_probs, dtype=_F32)
        p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
        if observed_mask is None:
            observed_mask = np.zeros(p.shape[:2], dtype=bool)
        return cls(p, cls.entropy_of(p), observed_mask)

    @classmethod
    def uniform(cls, h: int, w: int) -> "Belief":
        """Maximum-entropy belief (all classes 0.25)."""
        return cls.from_probs(np.full((h, w, N_CLASSES), 1.0 / N_CLASSES, _F32))

    @classmethod
    def one_hot(cls, true_class: np.ndarray) -> "Belief":
        """Perfect-knowledge belief from a class map (used before P3)."""
        p = np.eye(N_CLASSES, dtype=_F32)[np.asarray(true_class, dtype=np.int64)]
        return cls.from_probs(p, np.ones(p.shape[:2], dtype=bool))


# --------------------------------------------------------------------------- CostMap
@dataclass
class CostMap:
    """Fused traversal cost.

    Attributes:
        cost: (H, W) float32 dimensionless per-metre cost factor (>= 0); ``np.inf`` = impassable.
            A* step cost into cell v is ``dist_m * (1 + cost[v]) + risk_weight * -log(1 - hazard_prob[v])``.
        hazard_prob: (H, W) float32 in [0, 1], probability that entering the cell ends the mission.
        components: name -> (H, W) float32 weighted term (e.g. "slope", "rough", "class",
            "energy", "uncertainty"); sums to ``cost`` on finite cells. Kept for explainability.
        energy_wh_per_m: optional (H, W) float32 energy in Wh per metre traversed; used for
            ``Plan.energy_wh``. ``None`` until the energy model exists.
    """

    cost: np.ndarray
    hazard_prob: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)
    energy_wh_per_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.cost = _as_f32(self.cost, "cost", 2)
        hw = self.cost.shape
        self.hazard_prob = _as_f32(self.hazard_prob, "hazard_prob", 2)
        _check_shape(self.hazard_prob, "hazard_prob", hw)
        comps = {}
        for k, v in self.components.items():
            v = _as_f32(v, f"components[{k!r}]", 2)
            _check_shape(v, f"components[{k!r}]", hw)
            comps[k] = v
        self.components = comps
        if self.energy_wh_per_m is not None:
            self.energy_wh_per_m = _as_f32(self.energy_wh_per_m, "energy_wh_per_m", 2)
            _check_shape(self.energy_wh_per_m, "energy_wh_per_m", hw)

    @property
    def shape(self) -> tuple[int, int]:
        return self.cost.shape  # type: ignore[return-value]

    @property
    def passable(self) -> np.ndarray:
        """(H, W) bool, True where cost is finite."""
        return np.isfinite(self.cost)

    def breakdown(self, rc: RC) -> dict[str, float]:
        """Per-component cost at one cell, plus total and hazard_prob (plain floats)."""
        r, c = rc
        out = {k: float(v[r, c]) for k, v in self.components.items()}
        out["total"] = float(self.cost[r, c])
        out["hazard_prob"] = float(self.hazard_prob[r, c])
        return out


# --------------------------------------------------------------------------- Plan
@dataclass
class Plan:
    """A planned path and its statistics. ``Plan(path=[])`` means "no path found".

    Attributes:
        path: list of cells (r, c) from start to goal inclusive; empty if none.
        total_cost: accumulated A* cost (dimensionless metre-weighted units).
        length_m: geometric path length in metres.
        energy_wh: estimated energy in watt-hours along the path.
        success_prob: prod(1 - hazard_prob) along the path, in [0, 1].
        mode: planner mode label ("shortest", "balanced", "safest", "science", ...).
        expanded_nodes: number of nodes popped/closed by the search.
        meta: free-form extras (weights used, runtime_s, waypoints, ...); JSON-friendly.
    """

    path: list[RC] = field(default_factory=list)
    total_cost: float = float("inf")
    length_m: float = 0.0
    energy_wh: float = 0.0
    success_prob: float = 0.0
    mode: str = "balanced"
    expanded_nodes: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = [_to_rc(p) for p in self.path]
        self.total_cost = float(self.total_cost)
        self.length_m = float(self.length_m)
        self.energy_wh = float(self.energy_wh)
        self.success_prob = float(self.success_prob)
        self.expanded_nodes = int(self.expanded_nodes)

    @property
    def found(self) -> bool:
        """True if a path exists."""
        return len(self.path) > 0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(rows, cols) int arrays for fancy indexing / plotting (x=col, y=row)."""
        if not self.path:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        a = np.asarray(self.path, dtype=np.int64)
        return a[:, 0], a[:, 1]

    def summary(self) -> dict[str, Any]:
        """JSON-friendly stats (no path)."""
        return {
            "mode": self.mode,
            "found": self.found,
            "n_cells": len(self.path),
            "total_cost": self.total_cost,
            "length_m": self.length_m,
            "energy_wh": self.energy_wh,
            "success_prob": self.success_prob,
            "expanded_nodes": self.expanded_nodes,
        }


# --------------------------------------------------------------------------- Target
@dataclass
class Target:
    """A candidate science target.

    Attributes:
        id: stable integer id (rank at detection time).
        rc: target cell (r, c).
        value_mean: expected science value (dimensionless, ~[0, 1]).
        value_std: std of science value from bootstrap (same units).
        info_gain: value_mean * local mean entropy — "worth investigating because uncertain".
        reachable: True if a finite-cost path exists from the rover.
        plan: path from the rover to this target, or None if not yet planned.
        reason: template-built explanation string (numbers only, see explainable-decisions).
        utility: selection score = value - alpha*cost - beta*(1 - success_prob).
    """

    id: int
    rc: RC
    value_mean: float
    value_std: float
    info_gain: float = 0.0
    reachable: bool = False
    plan: Plan | None = None
    reason: str = ""
    utility: float = 0.0

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.rc = _to_rc(self.rc)
        self.value_mean = float(self.value_mean)
        self.value_std = float(self.value_std)
        self.info_gain = float(self.info_gain)
        self.reachable = bool(self.reachable)
        self.utility = float(self.utility)


# --------------------------------------------------------------------------- LogEntry
@dataclass
class LogEntry:
    """One traceable rover decision (see the explainable-decisions skill).

    Attributes:
        step: autonomy-loop step index (0-based).
        event: snake_case observation code (e.g. "hazard_on_path", "cost_increase").
        decision: snake_case action code (e.g. "replan", "keep_path", "select_target").
        evidence: flat dict of JSON-serialisable raw numbers; must include
            "cell" [r, c], "rover_rc" [r, c], "threshold", "threshold_name".
        confidence: 1 - entropy/ln(4) at the trigger cell, in [0, 1].
        reason: template-built human-readable string derived from evidence.
    """

    step: int
    event: str
    decision: str
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        self.step = int(self.step)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict."""
        return asdict(self)


__all__ = [
    "RC", "SOIL", "BEDROCK", "SAND", "BIG_ROCK", "CLASS_NAMES", "N_CLASSES",
    "World", "TerrainLayers", "Belief", "CostMap", "Plan", "Target", "LogEntry",
]
