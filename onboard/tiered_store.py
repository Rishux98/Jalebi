"""Colibri-style tiered terrain tile store: COLD flash -> WARM compressed RAM -> HOT decoded RAM.

Architecture idea adapted from colibri (MoE expert offloading: hot experts on GPU, warm in RAM, cold streamed
from disk, learned cache, lookahead prefetch) -> terrain tiles. No colibri code is used.

* Tiles: 64x64. Each tile is ONE blob = row-delta encoded int16 DEM + uint8 slope (0.5 deg) + uint8 roughness
  (fraction of ``rough_ref_m``), zlib-compressed, so a tile is one flash read.
* COLD  = every blob in a "flash" dict (byte-counted; each read adds simulated latency, default 2 ms).
* WARM  = compressed blobs in RAM (budget ``max_warm`` tiles).
* HOT   = decoded numpy tiles (budget ``max_hot``, default 9 = 3x3 around the rover).
* RESIDENT always: coarse map at 4x downsample — mean elevation and MAX slope (hazards never averaged away).
* Heat(tile) = a * decayed access count + b * on-planned-path-within-N-tiles + c / (1 + tile distance to rover).
  Promotion/eviction by heat (lowest first; the rover's own tile is pinned), not plain LRU.
* Path-lookahead prefetch: next K path tiles COLD -> WARM -> HOT on a 1-worker ThreadPoolExecutor (or inline
  with ``prefetch_sync=True`` for deterministic tests). Access history persists in ``onboard/.tile_usage.json``.
* ``get_window`` assembles float32 windows from HOT tiles; a tile not HOT at request time loads synchronously
  and counts as a STALL.
Run: python -m onboard.tiered_store
"""
from __future__ import annotations

import json
import struct
import threading
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from core.config import DEFAULT, Config
from core.types import RC
from onboard.quantize import QuantizedDEM, dequantize_slope, dequantize_unit, quantize_slope, quantize_unit

_F32 = np.float32
TILE = 64
USAGE_PATH = Path(__file__).resolve().parent / ".tile_usage.json"
COARSE = 4
LAYERS = ("dem", "slope", "rough")                     # per-tile layers, in blob order
_DT = {"dem": np.int16, "slope": np.uint8, "rough": np.uint8}

Tid = tuple[int, int]


# =========================================================================== blob codec
def encode_tile(layers: dict[str, np.ndarray]) -> bytes:
    """One blob for one tile: header (h, w) + zlib(row-delta int32 of every layer, concatenated)."""
    h, w = layers["dem"].shape
    parts = []
    for k in LAYERS:
        a = np.asarray(layers[k]).astype(np.int32)
        d = a.copy()
        d[1:] -= a[:-1]                                  # row-delta: neighbouring rows are similar
        parts.append(d.tobytes())
    return struct.pack("<HH", h, w) + zlib.compress(b"".join(parts), 6)


def decode_tile(blob: bytes) -> dict[str, np.ndarray]:
    h, w = struct.unpack("<HH", blob[:4])
    raw = np.frombuffer(zlib.decompress(blob[4:]), dtype=np.int32).reshape(len(LAYERS), h, w)
    return {k: np.cumsum(raw[i], axis=0).astype(_DT[k]) for i, k in enumerate(LAYERS)}


# =========================================================================== store
class TieredTileStore:
    """Terrain tiles in three tiers plus a resident coarse map. Thread-safe (one RLock)."""

    def __init__(self, dem: np.ndarray, cell_size: float, slope_deg: np.ndarray, roughness: np.ndarray,
                 cfg: Config = DEFAULT, tile: int = TILE, max_hot: int = 9, max_warm: int = 24,
                 read_latency_s: float = 0.002, prefetch_k: int = 4, path_tiles_n: int = 6,
                 heat_weights: tuple[float, float, float] = (1.0, 2.0, 3.0), decay: float = 0.97,
                 prefetch_sync: bool = False, usage_path: str | Path | None = USAGE_PATH, key: str = "default",
                 simulate_latency: bool = True) -> None:
        self.cfg, self.tile, self.cell_size = cfg, int(tile), float(cell_size)
        self.shape: tuple[int, int] = tuple(np.shape(dem))  # type: ignore[assignment]
        H, W = self.shape
        self.ntr, self.ntc = -(-H // self.tile), -(-W // self.tile)
        self.max_hot, self.max_warm = int(max_hot), int(max_warm)
        self.read_latency_s, self.simulate_latency = float(read_latency_s), bool(simulate_latency)
        self.prefetch_k, self.path_tiles_n = int(prefetch_k), int(path_tiles_n)
        self.a, self.b, self.c = heat_weights
        self.decay, self.prefetch_sync = float(decay), bool(prefetch_sync)
        self.usage_path, self.key = (Path(usage_path) if usage_path else None), key
        self.qdem = QuantizedDEM.from_float(dem)
        self.rough_ref = float(cfg.rough_ref_m)

        # ---- COLD: one blob per tile ("flash")
        q_slope = quantize_slope(slope_deg)
        q_rough = quantize_unit(np.clip(np.asarray(roughness) / self.rough_ref, 0.0, 1.0))
        self.flash: dict[Tid, bytes] = {}
        for tr in range(self.ntr):
            for tc in range(self.ntc):
                sl = self._tile_slices((tr, tc))
                self.flash[(tr, tc)] = encode_tile({"dem": self.qdem.data[sl], "slope": q_slope[sl],
                                                    "rough": q_rough[sl]})
        del self.qdem.data                                   # the onboard copy lives only in the tiles
        self.flash_bytes = sum(len(b) for b in self.flash.values())

        # ---- RESIDENT coarse map: mean elevation, MAX slope per COARSExCOARSE block
        f = COARSE
        hp, wp = -(-H // f) * f, -(-W // f) * f
        zp = np.pad(np.asarray(dem, np.float64), ((0, hp - H), (0, wp - W)), mode="edge")
        sp = np.pad(np.asarray(slope_deg, np.float64), ((0, hp - H), (0, wp - W)), mode="edge")
        self.coarse_elev = zp.reshape(hp // f, f, wp // f, f).mean(axis=(1, 3)).astype(_F32)
        self.coarse_slope_max = sp.reshape(hp // f, f, wp // f, f).max(axis=(1, 3)).astype(_F32)

        # ---- tiers + bookkeeping
        self.hot: dict[Tid, dict[str, np.ndarray]] = {}
        self.warm: dict[Tid, bytes] = {}
        self.access = np.zeros((self.ntr, self.ntc), np.float64)
        self.rover_tile: Tid = (0, 0)
        self.path_rank: dict[Tid, int] = {}
        self._prefetched: set[Tid] = set()
        self._lock = threading.RLock()
        self._pool = None if self.prefetch_sync else ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self.counters = {k: 0 for k in ("requests", "hits_hot", "hits_warm", "misses_cold", "stalls",
                                         "prefetch_issued", "prefetch_loaded", "prefetch_hits", "evictions_hot",
                                         "evictions_warm", "flash_reads", "flash_bytes_read", "boot_warm_preload")}
        self.sim_latency_s = 0.0
        self._load_usage()

    # ------------------------------------------------------------------ geometry
    def _tile_slices(self, tid: Tid) -> tuple[slice, slice]:
        tr, tc = tid
        H, W = self.shape
        return (slice(tr * self.tile, min((tr + 1) * self.tile, H)), slice(tc * self.tile, min((tc + 1) * self.tile, W)))

    def tile_of(self, rc: RC) -> Tid:
        return int(rc[0]) // self.tile, int(rc[1]) // self.tile

    def tiles_for_box(self, r0: int, r1: int, c0: int, c1: int) -> list[Tid]:
        """Tiles intersecting rows r0:r1, cols c0:c1 (half-open)."""
        return [(tr, tc) for tr in range(r0 // self.tile, (r1 - 1) // self.tile + 1)
                for tc in range(c0 // self.tile, (c1 - 1) // self.tile + 1)]

    # ------------------------------------------------------------------ heat + tiers
    def heat(self, tid: Tid) -> float:
        rank = self.path_rank.get(tid)
        on_path = 0.0 if rank is None else (1.0 - rank / max(self.path_tiles_n, 1))
        dist = max(abs(tid[0] - self.rover_tile[0]), abs(tid[1] - self.rover_tile[1]))
        return self.a * float(self.access[tid]) + self.b * on_path + self.c / (1.0 + dist)

    def _read_flash(self, tid: Tid) -> bytes:
        blob = self.flash[tid]
        with self._lock:
            self.counters["flash_reads"] += 1
            self.counters["flash_bytes_read"] += len(blob)
            self.sim_latency_s += self.read_latency_s
        if self.simulate_latency and self.read_latency_s > 0:
            time.sleep(self.read_latency_s)               # simulated storage latency (outside the lock)
        return blob

    def _evict(self) -> None:
        """Demote lowest-heat HOT tiles to WARM, then drop lowest-heat WARM blobs. Rover tile is pinned."""
        while len(self.hot) > self.max_hot:
            cand = [t for t in self.hot if t != self.rover_tile]
            if not cand:
                break
            victim = min(cand, key=self.heat)
            self.hot.pop(victim)
            self.warm[victim] = self.flash[victim]        # compressed copy stays in RAM
            self._prefetched.discard(victim)
            self.counters["evictions_hot"] += 1
        while len(self.warm) > self.max_warm:
            victim = min(self.warm, key=self.heat)
            self.warm.pop(victim)
            self.counters["evictions_warm"] += 1

    def _make_hot(self, tid: Tid, blob: bytes | None = None) -> dict[str, np.ndarray]:
        with self._lock:
            if tid in self.hot:
                return self.hot[tid]
            if blob is None:
                blob = self.warm.get(tid)
        if blob is None:
            blob = self._read_flash(tid)
        arrs = decode_tile(blob)
        with self._lock:
            self.warm.pop(tid, None)
            self.hot[tid] = arrs
            self._evict()
            return arrs

    def _fetch(self, tid: Tid) -> dict[str, np.ndarray]:
        """Tile for an actual read: counts hit tier / stall / prefetch hit."""
        with self._lock:
            self.counters["requests"] += 1
            self.access[tid] += 1.0
            if tid in self.hot:
                self.counters["hits_hot"] += 1
                if tid in self._prefetched:
                    self.counters["prefetch_hits"] += 1
                    self._prefetched.discard(tid)
                return self.hot[tid]
            self.counters["stalls"] += 1
            self.counters["hits_warm" if tid in self.warm else "misses_cold"] += 1
        return self._make_hot(tid)

    # ------------------------------------------------------------------ public API
    def get_layers_window(self, r0: int, r1: int, c0: int, c1: int) -> dict[str, np.ndarray]:
        """float32 ``dem`` (m), ``slope`` (deg), ``rough`` (m) for rows r0:r1, cols c0:c1 (clipped to the map)."""
        H, W = self.shape
        r0, r1, c0, c1 = max(0, r0), min(H, r1), max(0, c0), min(W, c1)
        out = {"dem": np.empty((r1 - r0, c1 - c0), _F32), "slope": np.empty((r1 - r0, c1 - c0), _F32),
               "rough": np.empty((r1 - r0, c1 - c0), _F32)}
        for tid in self.tiles_for_box(r0, r1, c0, c1):
            t = self._fetch(tid)
            tsl = self._tile_slices(tid)
            ar0, ar1 = max(r0, tsl[0].start), min(r1, tsl[0].stop)
            ac0, ac1 = max(c0, tsl[1].start), min(c1, tsl[1].stop)
            src = (slice(ar0 - tsl[0].start, ar1 - tsl[0].start), slice(ac0 - tsl[1].start, ac1 - tsl[1].start))
            dst = (slice(ar0 - r0, ar1 - r0), slice(ac0 - c0, ac1 - c0))
            out["dem"][dst] = t["dem"][src].astype(_F32) * _F32(self.qdem.scale) + _F32(self.qdem.offset)
            out["slope"][dst] = dequantize_slope(t["slope"][src])
            out["rough"][dst] = dequantize_unit(t["rough"][src]) * _F32(self.rough_ref)
        return out

    def boot_region(self, r0: int, r1: int, c0: int, c1: int) -> dict[str, np.ndarray]:
        """Like ``get_layers_window`` but decodes straight from flash without touching tiers or hit stats.
        Used once at boot to stream the global compact cost map tile by tile (counted as ``boot_reads``)."""
        H, W = self.shape
        r0, r1, c0, c1 = max(0, r0), min(H, r1), max(0, c0), min(W, c1)
        out = {k: np.empty((r1 - r0, c1 - c0), _F32) for k in LAYERS}
        for tid in self.tiles_for_box(r0, r1, c0, c1):
            t = decode_tile(self.flash[tid])
            with self._lock:
                self.counters["boot_reads"] = self.counters.get("boot_reads", 0) + 1
            tsl = self._tile_slices(tid)
            ar0, ar1 = max(r0, tsl[0].start), min(r1, tsl[0].stop)
            ac0, ac1 = max(c0, tsl[1].start), min(c1, tsl[1].stop)
            src = (slice(ar0 - tsl[0].start, ar1 - tsl[0].start), slice(ac0 - tsl[1].start, ac1 - tsl[1].start))
            dst = (slice(ar0 - r0, ar1 - r0), slice(ac0 - c0, ac1 - c0))
            out["dem"][dst] = t["dem"][src].astype(_F32) * _F32(self.qdem.scale) + _F32(self.qdem.offset)
            out["slope"][dst] = dequantize_slope(t["slope"][src])
            out["rough"][dst] = dequantize_unit(t["rough"][src]) * _F32(self.rough_ref)
        return out

    def get_window(self, center_rc: RC, radius: int) -> np.ndarray:
        """float32 elevation window (2r+1 square, clipped at the map edge) assembled from HOT tiles."""
        r, c = int(center_rc[0]), int(center_rc[1])
        return self.get_layers_window(r - radius, r + radius + 1, c - radius, c + radius + 1)["dem"]

    def update_rover(self, rc: RC, remaining_path: Sequence[RC] = (), reach: int = 0) -> None:
        """Rover moved / replanned: decay history, re-rank path tiles, pin rover tile, prefetch ahead.

        ``reach`` (cells): tiles intersecting the box rc +- reach (sensor footprint + feature halo) are
        prefetched first, then the next ``prefetch_k`` tiles along the remaining path."""
        H, W = self.shape
        r, c = int(rc[0]), int(rc[1])
        near = self.tiles_for_box(max(0, r - reach), min(H, r + reach + 1), max(0, c - reach), min(W, c + reach + 1))
        order: list[Tid] = []
        for cell in remaining_path:
            t = self.tile_of(cell)
            if not order or order[-1] != t:
                if t not in order:
                    order.append(t)
            if len(order) >= self.path_tiles_n:
                break
        with self._lock:
            self.access *= self.decay
            self.rover_tile = self.tile_of(rc)
            self.path_rank = {t: i for i, t in enumerate(order)}
            seq = [self.rover_tile, *[t for t in near if t != self.rover_tile], *order[: self.prefetch_k]]
            want = [t for t in dict.fromkeys(seq) if t not in self.hot][: self.max_hot]
            self.counters["prefetch_issued"] += len(want)
        if not want:
            return
        if self._pool is None:
            self._prefetch(want)
        else:
            self._pool.submit(self._prefetch, want)

    def _prefetch(self, tids: list[Tid]) -> None:
        for tid in tids:
            with self._lock:
                if tid in self.hot:
                    continue
            self._make_hot(tid)
            with self._lock:
                if tid in self.hot:
                    self._prefetched.add(tid)
                    self.counters["prefetch_loaded"] += 1

    def wait_prefetch(self) -> None:
        """Block until queued prefetches finish (for measurements)."""
        if self._pool is not None:
            self._pool.submit(lambda: None).result()

    def tier_map(self) -> np.ndarray:
        """(ntr, ntc) uint8: 0 = COLD, 1 = WARM, 2 = HOT."""
        with self._lock:
            m = np.zeros((self.ntr, self.ntc), np.uint8)
            for t in self.warm:
                m[t] = 1
            for t in self.hot:
                m[t] = 2
            return m

    def stats(self) -> dict:
        with self._lock:
            hot_b = sum(a.nbytes for t in self.hot.values() for a in t.values())
            warm_b = sum(len(b) for b in self.warm.values())
            s = dict(self.counters)
            req = max(s["requests"], 1)
            s.update(tiles=self.ntr * self.ntc, tile_size=self.tile, hot_tiles=len(self.hot), warm_tiles=len(self.warm),
                     hot_bytes=int(hot_b), warm_bytes=int(warm_b), flash_bytes=int(self.flash_bytes),
                     resident_coarse_bytes=int(self.coarse_elev.nbytes + self.coarse_slope_max.nbytes),
                     hit_rate=s["hits_hot"] / req,
                     prefetch_hit_rate=s["prefetch_hits"] / max(s["prefetch_loaded"], 1),
                     simulated_flash_latency_s=round(self.sim_latency_s, 4))
            return s

    def ram_bytes(self) -> int:
        st = self.stats()
        return st["hot_bytes"] + st["warm_bytes"] + st["resident_coarse_bytes"]

    # ------------------------------------------------------------------ learned cache (persisted)
    def _load_usage(self) -> None:
        if not self.usage_path or not self.usage_path.exists():
            return
        try:
            data = json.loads(self.usage_path.read_text()).get(self.key)
        except (OSError, ValueError):
            return
        if not data:
            return
        for k, v in data.items():
            tr, tc = (int(x) for x in k.split(","))
            if 0 <= tr < self.ntr and 0 <= tc < self.ntc:
                self.access[tr, tc] = float(v)
        # warm-up: preload the historically hottest tiles into WARM (compressed) — repeated runs start warmer
        order = sorted(((float(self.access[t]), t) for t in self.flash), reverse=True)
        for score, tid in order[: self.max_warm]:
            if score > 0:
                self.warm[tid] = self._read_flash(tid)
                self.counters["boot_warm_preload"] += 1

    def save_usage(self) -> None:
        if not self.usage_path:
            return
        try:
            data = json.loads(self.usage_path.read_text()) if self.usage_path.exists() else {}
        except (OSError, ValueError):
            data = {}
        data[self.key] = {f"{t[0]},{t[1]}": round(float(self.access[t]), 4) for t in self.flash if self.access[t] > 0}
        self.usage_path.write_text(json.dumps(data))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.save_usage()


def build_store(world, layers, cfg: Config = DEFAULT, **kw) -> TieredTileStore:
    """Ground-station step: quantise + tile a site (DEM, slope, roughness) -> store to uplink."""
    key = kw.pop("key", f"{world.name}-{world.shape[0]}x{world.shape[1]}-s{world.seed}")
    return TieredTileStore(world.dem, world.cell_size, layers.slope_deg, layers.roughness, cfg, key=key, **kw)


if __name__ == "__main__":
    from terrain.features import compute_layers
    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    L = compute_layers(w)
    st = build_store(w, L, prefetch_sync=True, usage_path=None)
    win = st.get_window(w.start, 6)
    print(f"tiles {st.ntr}x{st.ntc}, flash {st.flash_bytes} B (float32 DEM+slope+rough = {3 * w.dem.nbytes} B); "
          f"window {win.shape}, max |err| {np.abs(win - w.dem[w.start[0] - 6:w.start[0] + 7, w.start[1] - 6:w.start[1] + 7]).max():.4f} m")
    st.update_rover(w.start, [(r, r) for r in range(12, 190)])
    print({k: v for k, v in st.stats().items() if k in ("hot_tiles", "warm_tiles", "stalls", "prefetch_loaded",
                                                          "hot_bytes", "warm_bytes", "flash_bytes")})
    st.close()
