# ODYSSEY — Roadmap, Workflow & Claude Code Prompts

## 1. Repo → module mapping (what we take from each)

| Repo / data | What we take (logic, not code) | Feeds our module |
|---|---|---|
| **AtsushiSakai/PythonRobotics** — github.com/AtsushiSakai/PythonRobotics | A* open/closed set + heuristic structure; RRT* rewiring | `planner/astar.py`, `planner/rrt_star.py` |
| **1piyalib/rover_pathfinder** — github.com/1piyalib/rover_pathfinder | DEM → reduced grid → slope-limited neighbours → A* pipeline | `terrain/features.py`, `planner/astar.py` |
| **utiasSTARS/gplanetary-nav** — github.com/utiasSTARS/gplanetary-nav | Slope-dependent velocity/energy model, solar-power idea, multi-objective (shortest vs fastest) | `costmap/fuse.py` (energy term), `planner/pareto.py` |
| **mchancan/PlanetaryPathBench** — github.com/mchancan/PlanetaryPathBench | HiRISE-derived benchmark maps to test planner on real geometry | `tests/`, demo "benchmark" tab (optional) |
| **qubvel-org/segmentation_models.pytorch** (pip `segmentation-models-pytorch`) | U-Net/ResNet34 pretrained encoder | `perception/model.py` |
| **AI4MARS** (HF: `hassanjbara/AI4MARS`) | 4 terrain classes, training data | `perception/train.py` → confusion matrix |
| **USGS HiRISE DTMs** (stac.astrogeology.usgs.gov/docs/data/mars/hirise_dtms/) | Real elevation tile (crop ~1000×1000) | `world/real_dem.py` |

**Libraries:** numpy, scipy, rasterio, opencv-python, torch, segmentation-models-pytorch, datasets, plotly, streamlit, pytest, noise (Perlin).

---

## 2. The workflow (what the system does)

```
            ┌──────────────── WORLD (hidden ground truth) ────────────────┐
            │ DEM (synthetic or HiRISE) · true terrain classes · ice layer │
            └───────────────┬─────────────────────────────┬───────────────┘
                            │ noisy orbital prior          │ local "navcam" sensing (radius r)
                            ▼                              ▼
  TERRAIN FEATURES   slope, roughness,          PERCEPTION SENSOR MODEL
  + DEM uncertainty  P(slope>25°) via MC   ←→   AI4MARS U-Net confusion matrix
                            │                    → P(observed | true class)
                            ▼                              │
                     BELIEF MAP  (per-cell class probabilities, Bayesian update, entropy)
                            │
                            ▼
                     COST MAP  = slope + roughness + class risk + energy + λ·uncertainty
                               + hazard_prob (chance constraint)
                            │
            ┌───────────────┼──────────────────┐
            ▼               ▼                  ▼
       PLANNER (A*)   PARETO: safest /    SCIENCE: targets ranked
       risk-aware     shortest / science  value ± std, energy budget
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
           AUTONOMY LOOP (no mission control): move → sense → update belief
           → path still safe? → if not, REPLAN → log decision with evidence
                           ▼
           DASHBOARD: 3D terrain + path · cost / uncertainty heatmaps ·
           ranked targets · decision log · "inject hazard" button
```

### Why this wins on the judging criteria
- **Perception tied to planning with real numbers:** the trained AI4MARS model's validation **confusion matrix becomes the rover's sensor model**. So "sand looks like soil 18% of the time" directly raises risk in the map. Few teams will connect the CV model to the planner honestly.
- **Uncertainty:** two sources, both shown: (a) terrain-class belief entropy, (b) DEM noise → Monte-Carlo probability that slope exceeds the limit.
- **Path success probability:** `P(success) = Π(1 − hazard_prob)` along the path. Show it per route ("Safest: 97.2%, Shortest: 81.5%").
- **Autonomy under comm delay:** the rover discovers hidden hazards with local sensing and replans itself. Every replan has a logged reason plus evidence.
- **Science with honesty:** targets ranked with ± uncertainty. Bonus: a "worth investigating" score = value × uncertainty (information gain).

---

## 3. Timeline (from 01:30)

| Time | Phase | Owner | Done when |
|---|---|---|---|
| 01:30–02:00 | **P0** scaffold, types, config, synthetic world | Lead | `python -m world.synthetic` saves a PNG |
| 02:00–03:15 | **P1** terrain features + cost map + A* | Planner + Terrain | Path drawn on synthetic map, avoids >25° |
| 02:00–05:00 | **P2** perception (Colab GPU, parallel) | Perception | `confusion.npy` + `unet.pt` saved |
| *02:30* | *Mentor round 1: show P1 static path + architecture diagram* | | |
| 03:15–05:30 | **P3** belief map + uncertainty + risk-aware A* + Pareto | Planner + Terrain | 3 routes with success probability |
| 04:30–06:30 | **P4** science targets + autonomy loop + decision log | Lead + Perception | Replan triggered by hidden hazard, logged |
| 04:30–07:15 | **P5** Streamlit dashboard | UI | All views live, inject-hazard button works |
| *07:30* | *Mentor round 2: full demo dry run* | | |
| 08:30–09:20 | **P6** real HiRISE swap, tests, README, demo script | All | `pytest` green, README done |
| 09:30 | **Submit** | | |

**Parallel-work rule:** after P0, everyone codes only against `core/types.py`. If someone needs a field changed, they tell the lead, the lead changes it, and everyone pulls. Each person works on their own git branch and merges every ~45 min.

---

## 4. Claude Code: step-by-step prompts

**Setup (once):**
```bash
mkdir odyssey && cd odyssey && git init
# put CLAUDE.md + ROADMAP.md here
python -m venv .venv && source .venv/bin/activate
claude
```
**How to use each prompt:** paste it, ask Claude to **show its plan first**, approve, then let it build. At the end of every phase, run the "verify" line yourself and commit.

---

### P0 — Scaffold + contracts + synthetic world
```
Read CLAUDE.md and ROADMAP.md. Phase P0.
1. Create the folder layout from CLAUDE.md with __init__.py files and requirements.txt
   (numpy scipy rasterio opencv-python torch segmentation-models-pytorch datasets plotly streamlit pytest noise matplotlib).
2. core/types.py — dataclasses:
   World(dem, cell_size, true_class, ice_signal, start, goal, seed)
   TerrainLayers(slope_deg, roughness, curvature, p_slope_exceed)
   Belief(class_probs (H,W,4), entropy (H,W), observed_mask (H,W) bool)
   CostMap(cost, hazard_prob, components: dict[str, ndarray])
   Plan(path: list[tuple[int,int]], total_cost, length_m, energy_wh, success_prob, mode, expanded_nodes)
   Target(id, rc, value_mean, value_std, info_gain, reachable, plan: Plan|None, reason)
   LogEntry(step, event, decision, evidence: dict, confidence)
3. core/config.py — one Config dataclass holding every weight/threshold
   (slope limits 15/25, class risk {soil:0.05, bedrock:0.10, sand:0.45, big_rock:1.0 impassable},
   weights w_slope, w_rough, w_class, w_energy, lambda_unc, sensor_radius=6, dem_noise_sigma=0.3 m, seed=42).
4. world/synthetic.py — make_world(H=200, W=200, seed): Perlin-noise base terrain, 6–10 Gaussian craters
   with raised rims, one steep ridge, a sand field in a low basin, big-rock clusters near crater rims,
   bedrock on high-slope areas, soil elsewhere. ice_signal: smooth field peaking in 2–3 shadowed crater floors.
   Start/goal on opposite corners guaranteed passable.
   __main__: save world_preview.png (hillshade + class overlay).
5. tests/test_world.py smoke test.
Verify: python -m world.synthetic && pytest -q
```

### P1 — Terrain features + cost map + A*
*(Reference logic: PythonRobotics grid A*, rover_pathfinder slope-limited neighbours)*
```
Phase P1. Implement:
1. terrain/features.py
   - slope_deg(dem, cell_size) using Horn's 3x3 method (np.gradient acceptable), degrees.
   - roughness(dem, k=5): local std of elevation after removing local plane fit (or std of detrended window via scipy.ndimage).
   - curvature (Laplacian).
   - p_slope_exceed(dem, cell_size, sigma, limit, n=30, seed): Monte-Carlo — add Gaussian noise to DEM n times,
     recompute slope, return fraction of samples > limit. Vectorise.
   - compute_layers(world, cfg) -> TerrainLayers.
2. costmap/fuse.py
   - build_costmap(layers, class_probs, cfg, lambda_unc) -> CostMap.
     cost = w_slope*norm(slope) + w_rough*norm(rough) + w_class*E[class_risk] + w_energy*energy_term + lambda_unc*entropy.
     hazard_prob = 1 - (1-p_slope_exceed)*(1-P(big_rock))*(1 - 0.5*P(sand)) (document the formula).
     Cells with slope>25° deterministically or P(big_rock)>0.6 -> cost inf.
     Energy term (gplanetary-nav idea): energy per metre grows with uphill slope: 1 + k*max(0, dz/ds).
     Keep each term in components{} for explainability.
   - For now, class_probs may be one-hot of world.true_class (belief comes in P3).
3. planner/astar.py
   - astar(costmap, start, goal, cell_size, risk_weight=0.0) — 8-connected, step cost = distance * (1+cell cost)
     + risk_weight * -log(1-hazard_prob). Octile heuristic (admissible: multiply by min cell factor).
     Use heapq, numpy arrays for g-score and parent; no dict-per-node overhead. Return Plan with
     length_m, energy_wh, success_prob = prod(1-hazard_prob on path), expanded_nodes. Handle no-path -> Plan(path=[]).
4. __main__ demo: build world, layers, costmap, plan; save plan_preview.png with path over cost heatmap.
5. tests: path avoids inf cells, start==goal case, unreachable goal returns empty path, 200x200 runs < 2s.
Verify: python -m planner.astar && pytest -q
```

### P2 — Perception (run on Colab/Kaggle GPU in parallel)
```
Phase P2. Create perception/train.py and perception/infer.py (must also run as a Colab notebook cell-by-cell).
1. Load HF dataset hassanjbara/AI4MARS with streaming; take ~2000 labelled train and 400 val images,
   resize 256x256 grayscale->3ch, map masks to 4 classes, ignore index 255 for null.
2. smp.Unet('resnet34', encoder_weights='imagenet', classes=4) with dropout (p=0.2) before the head.
   Train ~6–8 epochs, AdamW, CE+Dice, AMP. Print mIoU per class.
3. MC-dropout inference: enable dropout at eval, T=15 passes -> mean probs, predictive entropy map.
4. Save: unet.pt, confusion.npy (4x4 row-normalised P(pred|true) on val), metrics.json, 6 sample PNGs
   (image | prediction | entropy).
5. perception/sensor_model.py: load confusion.npy (fallback to a hard-coded plausible matrix if missing,
   so the rest of the pipeline never blocks). observe(true_class_patch, rng) -> sampled observed classes using P(pred|true).
Verify: python -m perception.sensor_model (works with fallback even without training).
```

### P3 — Belief map + uncertainty + risk-aware planning + Pareto
```
Phase P3.
1. belief/bayes.py
   - init_prior(world, cfg): orbital prior = blurred, noisy class map (simulate low-res orbital classification:
     downsample true_class x8, add 20% random label noise, upsample, convert to soft probs with temperature).
   - update(belief, observations, mask, confusion): per cell Bayes: post ∝ prior * P(obs|true) using confusion matrix,
     vectorised. Multiple observations multiply. Recompute entropy.
2. Rebuild costmap from belief instead of one-hot truth.
3. planner/pareto.py: run astar for modes
   shortest (all weights ~0 except distance), safest (risk_weight high, lambda_unc high), balanced,
   and science (via waypoints — stub until P4). Also sweep risk_weight in [0..10] to produce
   (length_m, success_prob) points -> pareto front list.
4. planner/rrt_star.py (OPTIONAL, timebox 30 min): RRT* in continuous space with collision = cost inf / hazard_prob>0.5;
   used only as a comparison metric.
5. __main__: print table mode | length | energy | success_prob | expanded nodes; save pareto.png.
Verify: python -m planner.pareto && pytest -q
```

### P4 — Science targets + autonomy loop + decision log
```
Phase P4.
1. science/targets.py
   - detect_candidates(world, layers): local maxima of a science-value field =
     0.4*ice_signal + 0.3*crater-rim/layering proxy (high curvature + moderate slope) + 0.3*bedrock prob;
     non-max suppression, top 8.
   - value_mean/std: bootstrap over belief samples + ice_signal noise (n=50).
   - info_gain = value_mean * mean entropy in radius (worth investigating because uncertain).
   - For each target: plan from rover position; reachable flag; utility = value_mean - alpha*path_cost - beta*(1-success_prob).
   - select_under_budget(targets, energy_budget): greedy by utility/energy, re-planning from each chosen target.
   - reason: human-readable string built from the numbers.
2. explain/log.py: DecisionLog with append(LogEntry), to_dataframe(), to_json().
3. autonomy/loop.py: Rover(world, cfg) with step():
   - sense: reveal true terrain within sensor_radius via perception.sensor_model.observe -> belief.update
   - every step rebuild cost on the local window (cheap) ; if any cell on remaining path now has cost inf or
     hazard_prob > threshold, or path cost rose > 20%: REPLAN with A* from current cell.
   - log every replan with evidence {cell, slope, P(big_rock), P(sand), hazard_prob, old_cost, new_cost, old/new success_prob}
     and confidence = 1 - entropy/log(4) at the trigger cell.
   - inject_hazard(rc, radius): add a hidden boulder field / sand trap into world.true_class (rover doesn't know until sensed).
   - run_to_goal(max_steps) -> trajectory + log.
4. __main__: run with an injected hazard on the initial path; print the log; assert at least one replan.
Verify: python -m autonomy.loop && pytest -q
```

### P5 — Streamlit dashboard
```
Phase P5. app.py (Streamlit, wide layout, dark theme), cached with st.cache_data/resource.
Sidebar: world source (synthetic seed | HiRISE tile), mode (shortest/balanced/safest/science), lambda_unc slider,
energy budget slider, sensor radius, buttons: "Plan", "Run autonomous traverse", "Inject hazard on path".
Main tabs:
 1. Mission: Plotly 3D surface (hillshade colouring) with planned path + traversed path + targets as markers;
    metrics row: length, energy, success probability, replans count.
 2. Cost & Hazards: heatmaps side by side — cost, hazard_prob, slope; hover shows per-component cost breakdown for a cell.
 3. Uncertainty: belief entropy map, P(slope>25°) map, observed-mask overlay; before/after traverse comparison.
 4. Routes: Pareto scatter (length vs success_prob) + table of modes.
 5. Science: ranked target table (value ± std, info_gain, reachable, reason) + bar chart with error bars.
 6. Decision Log: table of LogEntry; selecting a row highlights that cell on a map and shows the evidence JSON.
 7. Perception: sample AI4MARS predictions + entropy PNGs, confusion matrix heatmap, metrics.json.
Animate the traverse with a step slider (no heavy real-time animation).
Verify: streamlit run app.py — click through every tab with no errors.
```

### P6 — Real data + polish
```
Phase P6.
1. world/real_dem.py: load a HiRISE DTM GeoTIFF with rasterio windowed read (crop given row/col window), fill nodata
   by nearest interpolation, downsample to <=300x300, read cell_size from transform. Generate true_class/ice_signal
   procedurally from slope/roughness (document as an assumption). Wire into app sidebar.
2. tests: pytest for every module; full pipeline test under 20s.
3. README.md: problem, architecture diagram (the ASCII workflow), how uncertainty is modelled (2 sources + formulas),
   cost function, sensor model from confusion matrix, autonomy loop, assumptions, limitations, how to run, repo references used as logic inspiration.
4. DEMO.md: an 8-minute script with exact clicks.
Verify: pytest -q && streamlit run app.py
```

---

## 5. Demo script (8 min)
1. **(1 min) Problem:** 20-min comm delay means the rover must decide alone.
2. **(1.5 min) Architecture:** the workflow diagram above, showing how the CV confusion matrix becomes the sensor model.
3. **(4 min) Live:**
   1. Plan the three routes and compare success probabilities.
   2. Open the uncertainty tab.
   3. Inject a hazard and run the autonomous traverse. The rover senses it, replans, and the log shows the evidence.
   4. Show the science ranking with error bars.
4. **(1.5 min) Robustness and limits:** the real HiRISE tile, the assumptions, what we'd do next (D* Lite, real orbital spectra).

## 6. Fallbacks (if something breaks)
- **Perception training fails:** the hard-coded confusion matrix keeps the pipeline running. Say so honestly.
- **HiRISE download is slow:** stay synthetic, and show one real tile as a static screenshot.
- **Autonomy loop is buggy at 07:00:** demo static Pareto routes plus a precomputed replan log saved as JSON.
