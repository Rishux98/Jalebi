"""Risk-aware A* on an 8-connected grid + path statistics.

Step cost from u into neighbour v (dist_m = cell_size * {1, sqrt 2}):
    dist_m * (1 + cost[v])  +  risk_weight * -log(1 - hazard_prob[v])
The risk term is additive in log-survival, so minimising it maximises prod(1 - hazard).

Heuristic (admissible): octile distance * cell_size * (1 + min finite cost). The risk term
is >= 0 so it can only make the true cost larger. Diagonal moves between two impassable
orthogonal neighbours are forbidden (no corner cutting).

Implementation: flat-index heapq with lazy deletion. g/parent/closed live in flat Python
lists/bytearray (array-backed, no per-node dicts); Python lists are used instead of NumPy for
the inner loop because NumPy scalar indexing is ~10x slower there.
Run: python -m planner.astar   (writes outputs/plan_preview.png)
"""
from __future__ import annotations

import heapq
import math
import time
from collections.abc import Sequence

import numpy as np

from core.types import RC, CostMap, Plan

SQRT2 = math.sqrt(2.0)
_HAZ_CLIP = 1.0 - 1e-6


def _check_cell(rc: RC, shape: tuple[int, int], name: str) -> RC:
    r, c = int(rc[0]), int(rc[1])
    if not (0 <= r < shape[0] and 0 <= c < shape[1]):
        raise ValueError(f"{name} {(r, c)} is out of bounds for grid {shape}")
    return r, c


def risk_term(hazard_prob: np.ndarray) -> np.ndarray:
    """-log(1 - hazard_prob), float64, with hazard clipped to [0, 1-1e-6]."""
    return -np.log1p(-np.clip(np.asarray(hazard_prob, dtype=np.float64), 0.0, _HAZ_CLIP))


def evaluate_path(costmap: CostMap, path: Sequence[RC], cell_size: float = 1.0,
                  risk_weight: float = 0.0, mode: str = "balanced", expanded_nodes: int = 0,
                  meta: dict | None = None) -> Plan:
    """Score an existing path on a (possibly updated) cost map.

    Used by A* for its result and by P4 to compare old vs new path cost after a belief update.

    Returns:
        Plan with total_cost (A* objective; inf if the path enters an impassable cell),
        length_m (m), energy_wh (Wh; 0 if the map has no energy layer) and
        success_prob = prod(1 - hazard_prob) over all path cells, computed in log space.

    Raises:
        ValueError: a cell is out of bounds or consecutive cells are not 8-adjacent.
    """
    meta = dict(meta or {})
    if len(path) == 0:
        return Plan(path=[], success_prob=0.0, mode=mode, expanded_nodes=expanded_nodes, meta=meta)
    shape = costmap.shape
    pts = np.asarray([_check_cell(p, shape, "path cell") for p in path], dtype=np.int64)
    rows, cols = pts[:, 0], pts[:, 1]
    steps = np.abs(np.diff(pts, axis=0))
    if steps.size and (steps.max() > 1 or (steps.sum(axis=1) == 0).any()):
        raise ValueError("path cells must be distinct 8-neighbours")
    dist = cell_size * np.where(steps.sum(axis=1) == 2, SQRT2, 1.0) if steps.size else np.zeros(0)
    vr, vc = rows[1:], cols[1:]                                   # cells being entered
    step_cost = dist * (1.0 + costmap.cost[vr, vc].astype(np.float64))
    if risk_weight > 0:
        step_cost = step_cost + risk_weight * risk_term(costmap.hazard_prob[vr, vc])
    energy = 0.0
    if costmap.energy_wh_per_m is not None and dist.size:
        energy = float(np.sum(dist * costmap.energy_wh_per_m[vr, vc]))
    log_surv = np.sum(np.log1p(-np.clip(costmap.hazard_prob[rows, cols].astype(np.float64),
                                         0.0, _HAZ_CLIP)))
    return Plan(path=[(int(r), int(c)) for r, c in pts], total_cost=float(step_cost.sum()),
                length_m=float(dist.sum()), energy_wh=energy, success_prob=float(np.exp(log_surv)),
                mode=mode, expanded_nodes=expanded_nodes, meta=meta)


def astar(costmap: CostMap, start: RC, goal: RC, cell_size: float = 1.0,
          risk_weight: float = 0.0, mode: str = "balanced") -> Plan:
    """Minimum-cost 8-connected path from ``start`` to ``goal``.

    Args:
        costmap: per-cell cost factor (inf = impassable) and hazard_prob.
        start, goal: cells (r, c).
        cell_size: metres per cell.
        risk_weight: weight of the -log(1 - hazard) term (0 = ignore hazard probability).
        mode: label stored in the Plan.

    Returns:
        Plan. ``Plan(path=[])`` with ``success_prob=0`` if start/goal is impassable
        (``expanded_nodes=0``, no search) or the goal is unreachable (actual expansion count).

    Raises:
        ValueError: start or goal out of bounds, or negative risk_weight.
    """
    t0 = time.perf_counter()
    shape = costmap.shape
    h_, w_ = shape
    sr, sc = _check_cell(start, shape, "start")
    gr, gc = _check_cell(goal, shape, "goal")
    if risk_weight < 0:
        raise ValueError(f"risk_weight must be >= 0, got {risk_weight}")
    meta = {"risk_weight": float(risk_weight), "cell_size": float(cell_size)}
    cost = costmap.cost
    if not (np.isfinite(cost[sr, sc]) and np.isfinite(cost[gr, gc])):
        meta.update(reason="start_or_goal_impassable", runtime_s=time.perf_counter() - t0)
        return Plan(path=[], success_prob=0.0, mode=mode, expanded_nodes=0, meta=meta)
    if (sr, sc) == (gr, gc):
        meta.update(runtime_s=time.perf_counter() - t0)
        return evaluate_path(costmap, [(sr, sc)], cell_size, risk_weight, mode, 0, meta)

    finite = np.isfinite(cost)
    pas = finite.ravel().tolist()
    factor = (1.0 + cost.astype(np.float64)).ravel().tolist()
    if risk_weight > 0:
        risk = (risk_weight * risk_term(costmap.hazard_prob)).ravel().tolist()
    else:
        risk = [0.0] * (h_ * w_)
    hscale = cell_size * (1.0 + float(cost[finite].min()))      # cheapest possible metre
    diag_extra = SQRT2 - 1.0
    moves = ((-1, 0, cell_size), (1, 0, cell_size), (0, -1, cell_size), (0, 1, cell_size),
             (-1, -1, SQRT2 * cell_size), (-1, 1, SQRT2 * cell_size),
             (1, -1, SQRT2 * cell_size), (1, 1, SQRT2 * cell_size))

    n = h_ * w_
    g = [math.inf] * n
    parent = [-1] * n
    closed = bytearray(n)
    s_idx, g_idx = sr * w_ + sc, gr * w_ + gc
    g[s_idx] = 0.0
    a, b = abs(sr - gr), abs(sc - gc)
    h0 = hscale * (max(a, b) + diag_extra * min(a, b))
    heap: list[tuple[float, float, int]] = [(h0, h0, s_idx)]   # (f, h tie-break, flat idx)
    push, pop = heapq.heappush, heapq.heappop
    expanded = 0
    found = False

    while heap:
        _, _, i = pop(heap)
        if closed[i]:
            continue                                            # stale entry (lazy deletion)
        closed[i] = 1
        expanded += 1
        if i == g_idx:
            found = True
            break
        r, c = divmod(i, w_)
        gi = g[i]
        for dr, dc, d in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h_ or nc < 0 or nc >= w_:
                continue
            j = nr * w_ + nc
            if closed[j] or not pas[j]:
                continue
            if dr and dc and not pas[r * w_ + nc] and not pas[nr * w_ + c]:
                continue                                        # no corner cutting
            ng = gi + d * factor[j] + risk[j]
            if ng < g[j]:
                g[j] = ng
                parent[j] = i
                a, b = abs(nr - gr), abs(nc - gc)
                hh = hscale * (a + diag_extra * b) if a > b else hscale * (b + diag_extra * a)
                push(heap, (ng + hh, hh, j))

    meta["runtime_s"] = time.perf_counter() - t0
    if not found:
        meta["reason"] = "goal_unreachable"
        return Plan(path=[], success_prob=0.0, mode=mode, expanded_nodes=expanded, meta=meta)

    rev = [g_idx]
    while rev[-1] != s_idx:
        rev.append(parent[rev[-1]])
    path = [divmod(k, w_) for k in reversed(rev)]
    meta["g_goal"] = g[g_idx]
    plan = evaluate_path(costmap, path, cell_size, risk_weight, mode, expanded, meta)
    plan.meta["runtime_s"] = time.perf_counter() - t0
    return plan


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.config import DEFAULT
    from core.io import hillshade
    from costmap.fuse import build_costmap, one_hot_probs
    from terrain.features import compute_layers
    from world.synthetic import make_world

    w = make_world(200, 200, seed=DEFAULT.seed)
    L = compute_layers(w)
    cm = build_costmap(L, one_hot_probs(w.true_class))
    print(f"world seed={w.seed} start={w.start} goal={w.goal}")
    plans = [astar(cm, w.start, w.goal, w.cell_size, risk_weight=rw, mode=m)
             for rw, m in ((0.0, "cost_only"), (100.0, "risk_aware"))]
    print(f"{'mode':12s} {'cells':>6s} {'length_m':>9s} {'energy_wh':>10s} {'success':>8s} "
          f"{'cost':>9s} {'expanded':>9s} {'ms':>6s}")
    for p in plans:
        print(f"{p.mode:12s} {len(p.path):6d} {p.length_m:9.1f} {p.energy_wh:10.1f} "
              f"{p.success_prob:8.2%} {p.total_cost:9.1f} {p.expanded_nodes:9d} "
              f"{1000 * p.meta['runtime_s']:6.0f}")
        rows, cols = p.as_arrays()
        assert np.isfinite(cm.cost[rows, cols]).all(), "path enters an impassable cell"

    out = Path("outputs/plan_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(hillshade(w.dem, w.cell_size), cmap="gray")
    im = ax.imshow(np.ma.masked_invalid(cm.cost),
                   cmap=plt.get_cmap("viridis").with_extremes(bad="black"), alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cost factor (black = impassable)")
    for p, col in zip(plans, ("white", "red")):
        rows, cols = p.as_arrays()
        ax.plot(cols, rows, "-", color=col, lw=2,
                label=f"{p.mode}: {p.length_m:.0f} m, P(success)={p.success_prob:.1%}")
    ax.plot(w.start[1], w.start[0], "o", ms=11, mfc="lime", mec="k")
    ax.plot(w.goal[1], w.goal[0], "*", ms=14, mfc="magenta", mec="k")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("A* over fused cost map (ground-truth classes)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("saved", out)
