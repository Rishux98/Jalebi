# CLAUDE.md — Project ODYSSEY (MARS-09 autonomous rover)

HackInnovision 2026, Track A. 12-hour build (deadline 09:30 submission). Judges score:
technical depth, reasoning quality, **uncertainty & robustness**, **explainability/traceability**, innovation + live demo.
Every feature must make one of those visible on screen.

## Hard rules
- All project code is written during the hackathon. Open-source **libraries** (pip) are fine.
  Reference repos (PythonRobotics, gplanetary-nav, rover_pathfinder, PlanetaryPathBench) are for
  *reading the logic* — re-implement in our own modules; never copy files wholesale.
- Python 3.10+. NumPy-first, vectorised. No heavy frameworks. UI = Streamlit + Plotly.
- Every module must run headless and have a `if __name__ == "__main__":` smoke demo.
- Keep it working end-to-end at all times. Prefer a simple version that runs over a clever one that doesn't.
- Deterministic: every random process takes a `seed`.

## Grid conventions (everyone follows these)
- All maps are NumPy arrays of shape `(H, W)` (or `(H, W, K)`), dtype `float32`, indexed `[row, col]`.
- `cell_size` in metres (default 1.0). Elevation in metres. Slope in degrees.
- Impassable cells have cost `np.inf`.
- Terrain classes (AI4MARS): `0=soil, 1=bedrock, 2=sand, 3=big_rock`.
- Default limits: slope safe < 15°, caution 15–25°, impassable > 25°.

## Module layout
```
odyssey/
  core/types.py        # dataclasses: World, TerrainLayers, Belief, CostMap, Plan, Target, LogEntry
  core/config.py       # all weights/thresholds in one place
  world/synthetic.py   # procedural DEM + ground-truth terrain classes + ice signal
  world/real_dem.py    # load/crop HiRISE GeoTIFF via rasterio windowed read
  terrain/features.py  # slope, roughness, curvature, Monte-Carlo slope exceedance prob
  perception/          # AI4MARS U-Net (smp), MC-dropout, confusion-matrix sensor model
  belief/bayes.py      # per-cell class belief, Bayesian update from observations, entropy
  costmap/fuse.py      # layers -> cost + hazard_prob + per-component breakdown
  planner/astar.py     # A* (8-connected), risk-aware variant, path stats
  planner/rrt_star.py  # optional comparison
  planner/pareto.py    # weight sweep -> safest / shortest / science-max
  science/targets.py   # candidate detection, value ± std, ranking, energy-budget selection
  autonomy/loop.py     # step-sim: sense -> update belief -> check -> replan -> log
  explain/log.py       # decision log with evidence dicts
  app.py               # Streamlit dashboard
  tests/               # pytest smoke tests per module
```

## Coding style
- Type hints, short docstrings stating units.
- Functions return dataclasses from `core/types.py`, not loose tuples.
- Every decision the rover makes appends a `LogEntry(evidence={...}, confidence=...)`.
- Run `pytest -q` before saying a phase is done.
