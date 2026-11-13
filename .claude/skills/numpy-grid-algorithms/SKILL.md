---
name: numpy-grid-algorithms
description: Rules for ODYSSEY grid code — DEM/terrain features, cost maps, A*, RRT*, Pareto sweeps, Monte-Carlo uncertainty. Use when writing or editing anything in terrain/, costmap/, planner/, belief/, world/ or any function that takes an (H, W) NumPy array.
---

# NumPy grid algorithm rules

## Array conventions (non-negotiable)
- Maps are `np.ndarray` of shape `(H, W)` (or `(H, W, K)` for per-class stacks), dtype **`float32`**.
  Cast on return: `out.astype(np.float32, copy=False)`.
- Index as `[row, col]`. Cells are `tuple[int, int]` as `(r, c)`, never `(x, y)`.
  For Plotly or matplotlib, x = col and y = row. Convert only at the plotting boundary.
- `cell_size` is in metres (default 1.0). Distances are in metres, elevation in metres, slope in degrees.
- Impassable means `np.inf` in the cost map. Test with `np.isfinite`, never `== np.inf` on sums.
- Booleans (`observed_mask`, etc.) are `dtype=bool`.

## Vectorise
- No Python loops over cells. Use NumPy broadcasting, `np.gradient`, `scipy.ndimage`
  (`uniform_filter`, `gaussian_filter`, `generic_filter` only as a last resort, `maximum_filter` for NMS,
  `distance_transform_edt`, `zoom`).
- Monte-Carlo: draw all samples at once, shape `(n, H, W)`, then reduce along `axis=0`.
  If memory is tight (n*H*W*4 bytes > ~200 MB), loop over batches, never over cells.
- Slope: `dz_dr, dz_dc = np.gradient(dem, cell_size)`; `slope = np.degrees(np.arctan(np.hypot(dz_dr, dz_dc)))`.
- Handle borders explicitly with `mode="nearest"` in ndimage filters.

## Determinism
- Every stochastic function takes `seed: int` and builds `rng = np.random.default_rng(seed)`.
  Never use `np.random.seed` or the global RNG. The same seed must give identical arrays (and a test checks this).

## A* (planner/astar.py)
- 8-connected moves: offsets `(±1,0),(0,±1)` cost `1*cell_size`, diagonals `√2*cell_size`.
- Step cost into cell v: `dist * (1 + cost[v]) + risk_weight * -log(1 - hazard_prob[v])`.
  Clip `hazard_prob` to `[0, 1-1e-6]` before the log.
- **Array-backed state:** `g = np.full((H, W), np.inf, np.float32)`, `parent = np.full((H, W), -1, np.int32)`
  storing the flat index `r*W + c`, `closed = np.zeros((H, W), bool)`. No dicts per node.
- Open set: `heapq` of `(f, tie_counter, flat_idx)`. Use lazy deletion: skip a popped node if it is `closed`.
- **Octile heuristic, kept admissible:**
  `h = cell_size * (dmax + (√2-1)*dmin) * min_step_factor`, where `min_step_factor = 1 + min(finite cost)`
  (the cheapest possible per-metre factor). The risk term is ≥ 0, so it never breaks admissibility.
- Don't step diagonally through two impassable orthogonal neighbours (no corner cutting).
- Reconstruct the path by following `parent` back from goal and reversing it. Return a `Plan`, not a list.

## Mandatory edge cases (each needs a test)
| Case | Expected |
|---|---|
| `start == goal` (and passable) | `Plan(path=[start], total_cost=0, length_m=0, success_prob=1-hazard[start])` |
| start or goal on an `inf` cell | `Plan(path=[])` immediately, no search, `expanded_nodes=0` |
| start/goal out of bounds | raise `ValueError` with the offending cell |
| goal unreachable | `Plan(path=[])`, `success_prob=0.0`, `expanded_nodes` = actual count |
| all-zero cost grid | straight/diagonal octile path, length equals the heuristic |

## Performance budget
- A 200x200 grid must plan corner-to-corner in **< 2 s** (a pytest timing test asserts this).
- Cost-map rebuilds over a local window must stay well under 100 ms. Slice the window, don't recompute the full map.
- Profile with `time.perf_counter()` in `__main__`, not with print timestamps.

## Derived path stats
- `length_m` = sum of step distances.
- `success_prob = np.prod(1 - hazard_prob[rows, cols])`.
  Compute it in log space (`np.exp(np.sum(np.log1p(-p)))`) for long paths.
- `energy_wh` uses the costmap energy component along the path. Document the units in the docstring.
