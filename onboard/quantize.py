"""Compact onboard formats: int16 DEM, uint8 layers, 2-bit classes — with explicit error reporting.

| layer          | full        | compact                          | max error                       |
|----------------|-------------|----------------------------------|---------------------------------|
| elevation      | float32 (4) | int16 x scale (+offset) (2)      | scale / 2 (0.5 cm at 0.01 m)    |
| 0-1 layers     | float32 (4) | uint8, 1/255 steps (1)           | 1/510                           |
| slope (deg)    | float32 (4) | uint8, 0.5 deg steps (1)         | 0.25 deg (saturates at 127.5)   |
| cost (per m)   | float32 (4) | uint8 0..254 x cmax/254, 255=inf | cmax/508                        |
| probability    | float32 (4) | uint8 log10 scale 1e-6..1, 0=0   | ±2.8 % relative                 |
| terrain class  | int8 (1)    | 2 bits (0.25)                    | exact                           |

Precision is never changed silently: ``report_error`` measures the DEM error and the slope error it causes, and
the dashboard shows that report whenever onboard mode is on.
Run: python -m onboard.quantize
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import DEFAULT, Config

_F32 = np.float32
INT16_MAX = 32767
COST_INF = 255          # uint8 code for an impassable cell
COST_LEVELS = 254       # finite cost codes 0..254
SLOPE_STEP_DEG = 0.5


# =========================================================================== DEM: int16
@dataclass
class QuantizedDEM:
    """int16 elevation: ``z ≈ data * scale + offset`` (metres).

    Attributes:
        data: (H, W) int16 codes.
        scale: metres per code (0.01 m default, widened automatically if the relief does not fit int16).
        offset: metres added after scaling (mid-range of the DEM).
    """
    data: np.ndarray
    scale: float
    offset: float

    @classmethod
    def from_float(cls, dem: np.ndarray, scale: float = 0.01) -> "QuantizedDEM":
        z = np.asarray(dem, dtype=np.float64)
        lo, hi = float(z.min()), float(z.max())
        offset = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if half / scale > INT16_MAX:                       # auto-widen: relief must fit +-32767 codes
            scale = half / INT16_MAX * 1.000001
        q = np.clip(np.rint((z - offset) / scale), -INT16_MAX, INT16_MAX).astype(np.int16)
        return cls(q, float(scale), float(offset))

    def to_float(self) -> np.ndarray:
        return (self.data.astype(_F32) * _F32(self.scale) + _F32(self.offset)).astype(_F32)

    def to_float_window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """float32 elevation for rows r0:r1, cols c0:c1 (decodes only that window)."""
        return (self.data[r0:r1, c0:c1].astype(_F32) * _F32(self.scale) + _F32(self.offset)).astype(_F32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


# =========================================================================== uint8 layers
def quantize_unit(a: np.ndarray) -> np.ndarray:
    """[0, 1] float -> uint8 (round to 1/255)."""
    return np.rint(np.clip(np.asarray(a, np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)


def dequantize_unit(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, _F32) / _F32(255.0)).astype(_F32)


# generic aliases (spec: quantize_unit / dequantize)
dequantize = dequantize_unit

PROB_LOG_MIN = -6.0     # log10 of the smallest non-zero probability kept (1e-6)


def quantize_prob_log(p: np.ndarray) -> np.ndarray:
    """Probability -> uint8 on a log scale: 0 = exactly 0, codes 1..255 span 1e-6..1 (±2.8 % relative per step).

    Linear 1/255 steps would round a typical per-cell hazard of ~0.001 to 0, and P(success) multiplies hundreds of
    cells — so hazards need relative, not absolute, precision.
    """
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    out = np.zeros(p.shape, np.uint8)
    nz = p >= 10.0 ** PROB_LOG_MIN
    out[nz] = np.clip(np.rint((np.log10(p[nz]) - PROB_LOG_MIN) / -PROB_LOG_MIN * 254) + 1, 1, 255).astype(np.uint8)
    return out


def dequantize_prob_log(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    val = 10.0 ** ((q.astype(np.float64) - 1) / 254 * -PROB_LOG_MIN + PROB_LOG_MIN)
    return np.where(q == 0, 0.0, val).astype(_F32)


def quantize_slope(deg: np.ndarray, step: float = SLOPE_STEP_DEG) -> np.ndarray:
    """Slope degrees -> uint8 codes of ``step`` degrees (saturates at 255*step)."""
    return np.clip(np.rint(np.asarray(deg, np.float64) / step), 0, 255).astype(np.uint8)


def dequantize_slope(q: np.ndarray, step: float = SLOPE_STEP_DEG) -> np.ndarray:
    return (np.asarray(q, _F32) * _F32(step)).astype(_F32)


def quantize_cost(cost: np.ndarray, cmax: float | None = None) -> tuple[np.ndarray, float]:
    """Cost per metre -> (uint8 codes, cmax). Finite costs map to 0..254 over [0, cmax]; inf -> 255."""
    c = np.asarray(cost, np.float64)
    fin = np.isfinite(c)
    if cmax is None:
        cmax = float(c[fin].max()) if fin.any() else 1.0
    cmax = max(float(cmax), 1e-9)
    q = np.full(c.shape, COST_INF, np.uint8)
    q[fin] = np.clip(np.rint(np.clip(c[fin], 0.0, cmax) / cmax * COST_LEVELS), 0, COST_LEVELS).astype(np.uint8)
    return q, cmax


def dequantize_cost(q: np.ndarray, cmax: float) -> np.ndarray:
    q = np.asarray(q)
    out = q.astype(_F32) * _F32(cmax / COST_LEVELS)
    return np.where(q == COST_INF, np.inf, out).astype(_F32)


# =========================================================================== classes: 2 bits / cell
def pack_classes(cls: np.ndarray) -> np.ndarray:
    """(H, W) ints in 0..3 -> flat uint8, 4 cells per byte (little-end first)."""
    a = np.asarray(cls, np.uint8).ravel()
    if a.size and a.max() > 3:
        raise ValueError("classes must be in 0..3 to pack into 2 bits")
    pad = (-a.size) % 4
    a = np.concatenate([a, np.zeros(pad, np.uint8)]).reshape(-1, 4)
    return (a[:, 0] | (a[:, 1] << 2) | (a[:, 2] << 4) | (a[:, 3] << 6)).astype(np.uint8)


def unpack_classes(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p = np.asarray(packed, np.uint8)
    out = np.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], axis=1).ravel()[: shape[0] * shape[1]]
    return out.reshape(shape).astype(np.int8)


# =========================================================================== error report
def report_error(dem: np.ndarray, cell_size: float = 1.0, scale: float = 0.01, cfg: Config = DEFAULT) -> dict:
    """Quantisation error of the int16 DEM and of the slope derived from it (and of uint8 slope storage)."""
    from terrain.features import slope_deg

    dem = np.asarray(dem, _F32)
    q = QuantizedDEM.from_float(dem, scale)
    back = q.to_float()
    err = np.abs(back.astype(np.float64) - dem)
    s_full = slope_deg(dem, cell_size)
    s_from_q = slope_deg(back, cell_size)
    s_stored = dequantize_slope(quantize_slope(s_full))
    max_elev = float(err.max())
    return {
        "scale_m": q.scale,
        "auto_widened": bool(q.scale > scale * 1.0000001),
        "max_elev_err_m": max_elev,
        "mean_elev_err_m": float(err.mean()),
        "max_slope_err_deg_from_int16_dem": float(np.abs(s_from_q - s_full).max()),
        "max_slope_err_deg_uint8_storage": float(np.abs(s_stored - s_full).max()),
        "dem_noise_sigma_m": float(cfg.dem_noise_sigma),
        "noise_to_error_ratio": float(cfg.dem_noise_sigma / max_elev) if max_elev > 0 else float("inf"),
    }


if __name__ == "__main__":
    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    rep = report_error(w.dem, w.cell_size)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rep.items()})
    q = QuantizedDEM.from_float(w.dem)
    print(f"DEM float32 {w.dem.nbytes} B -> int16 {q.nbytes} B; classes int8 {w.true_class.nbytes} B -> "
          f"2-bit {pack_classes(w.true_class).nbytes} B")
