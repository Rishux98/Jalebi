# ODYSSEY — autonomous Mars traverse planning under uncertainty

**HackInnovision 2026 · Track A · rover MARS-09**

A Mars rover is ~20 minutes of light-time from mission control, so it cannot ask before every step.
ODYSSEY plans a traverse from orbital data, drives it, **discovers hazards the orbiter could not see**,
replans on its own, and logs **every decision with the raw numbers behind it** — so a human can audit it later.

```
python -m streamlit run app.py      # dashboard → press "Inject hazard on path"
```

| What it shows | Where |
|---|---|
| Two uncertainty sources (terrain-class belief, DEM noise) propagated into a per-route **P(success)** | Uncertainty, Routes tabs |
| The perception model's **confusion matrix used as the rover's sensor model** (Bayes likelihood) | Perception tab, `perception/sensor_model.py` |
| Hidden boulder field → detected → **replan**, with evidence + confidence for each decision | Mission (replay), Decision Log |
| Science targets with **value ± bootstrap std**, selected under an energy budget | Science tab |
| Real **HiRISE** south-polar DTM (1.2 km, 4 m cells) running through the same pipeline | sidebar → World source |

---

## 1. Architecture

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

| Module | Role |
|---|---|
| `core/types.py`, `core/config.py` | Data contracts (`World`, `Belief`, `CostMap`, `Plan`, `Target`, `LogEntry`) and every threshold/weight |
| `world/synthetic.py` | Procedural site: fractal terrain, craters, ridge, sand basin, boulder clusters, ice in shadowed craters |
| `world/real_dem.py` | HiRISE DTM windowed read (HTTP range requests on a cloud-optimised GeoTIFF) → `World` |
| `terrain/features.py` | Slope, roughness, curvature, Monte-Carlo P(slope > limit) |
| `perception/` | AI4MARS U-Net (+ MC-dropout) training/inference; `sensor_model.py` turns its confusion matrix into the sensor |
| `belief/bayes.py` | Orbital prior + per-cell Bayesian update + entropy |
| `costmap/fuse.py` | Belief + terrain → cost (with per-component breakdown), hazard probability, energy |
| `planner/astar.py`, `planner/pareto.py` | Risk-aware A*, route modes, risk-weight sweep, Pareto front |
| `science/targets.py` | Candidate detection, bootstrap value ± std, utility, budgeted greedy selection |
| `autonomy/loop.py` | The rover: sense → update → local rebuild → check → replan |
| `explain/log.py` | `DecisionLog`: evidence contract, template-built reasons, JSON / DataFrame export |
| `app.py`, `ui/` | Streamlit dashboard (cached pipeline + Plotly figures) |

## 2. How uncertainty is modelled

**Source 1 — terrain class (what is the ground made of?).** Each cell holds a belief `P(class)` over
`soil, bedrock, sand, big_rock`.

* *Orbital prior* (`belief/bayes.py`): majority-vote the truth over 8×8 blocks (orbit cannot resolve
  boulders), relabel 20 % of orbital pixels (draws follow the orbital class frequencies), Gaussian-blur
  (σ = 4 cells), soften with temperature `p^(1/1.5)`, and floor every class at 0.002 so hazards stay discoverable.
* *Bayes update* from navcam observations, independent per look:
  `posterior(k) ∝ prior(k) · Π_i C[k, o_i]`, where `C[t, o] = P(observed o | true t)` is the **confusion matrix**.
* *Entropy* `H = −Σ p log p` (nats, max ln 4 ≈ 1.39) drives the uncertainty cost and every decision's confidence.

**Source 2 — elevation (how steep is it really?).** The orbital DEM has error (σ = 0.3 m, spatially
correlated over 3 cells). `terrain/features.py` draws 30 correlated noise fields, recomputes slope for
each, and stores `P(slope > 25°)` per cell (vectorised, memory-capped batches).

**Combined into a hazard probability** per cell (probability that entering it ends the mission):

```
hazard = 1 − (1 − 0.02 · P(slope>25°)) · (1 − 0.5 · P(big_rock)) · (1 − 0.004 · P(sand))
P(success of a route) = Π_cells (1 − hazard)          (computed in log space)
```

Every route reports P(success) **under the belief** and, for evaluation only, **against the hidden truth**
— the gap is exactly what the rover has to discover by sensing.

## 3. Cost function and planner

```
cost = w_slope·clip(slope/25°) + w_rough·clip(roughness/0.25 m) + w_class·Σ_k P(k)·risk_k
     + w_energy·k_up·tan(slope) + λ·H/ln4                       (per metre; inf if slope>25° or P(big_rock)>0.6)
A* step into v = dist_m · (1 + cost[v]) + risk_weight · (−log(1 − hazard[v]))
```

The risk term is additive in log-survival, so minimising it maximises P(success). The heuristic is octile
distance × the cheapest per-metre factor, so it stays admissible. A 200×200 plan takes ~30–70 ms.
Energy uses `1.5 Wh/m · (1 + 3·tan(slope))` (uphill-penalised, after gplanetary-nav's idea).

**Route modes** (`planner/pareto.py`): `shortest` (terrain weights 0), `balanced` (risk weight 2),
`safest` (risk weight 50, λ = 2 → avoids unknown terrain), `science` (balanced, through selected targets),
plus a risk-weight sweep 0…10 whose non-dominated (length, P(success)) points form the Pareto front.

**Movement:** 8-way (default, diagonals allowed, octile heuristic) or **4-way** — horizontal/vertical moves
only, Manhattan heuristic (`Config.grid_connectivity`, sidebar *Rover movement*). 4-way routes are simpler and
grid-aligned, ~10 % longer (350 m vs 320 m on the default site). Dashboard maps can be panned with the
sidebar *Map view* arrows / sliders (zoom 1–8×, centre on start/goal) or by dragging and scrolling.

## 4. Sensor model from the confusion matrix

`perception/train.py` trains `smp.Unet(resnet34)` on AI4MARS (streamed from Hugging Face, CE + Dice, AdamW,
AMP) and exports the validation confusion matrix `P(pred | true)` (`perception/confusion.npy`).
`perception/infer.py` runs **MC-dropout** (T = 15) and checks that the predictive entropy actually flags
errors (mean entropy on wrong vs correct pixels, AUROC of entropy as an error detector).

That matrix *is* the rover's sensor: `observe()` samples noisy class observations from it and
`likelihood()` gives the Bayes update. If no trained matrix exists, a documented fallback is used
(e.g. sand → soil 18 %) so the pipeline never blocks; the dashboard says which one is active.

## 5. Autonomy loop and the decision log

Each step (`autonomy/loop.py`): move one cell → sense the navcam disc (radius 6) → Bayes-update **only
that window** → rebuild the cost map **only on that window** → check the remaining route:

| Rule | Threshold | Decision |
|---|---|---|
| a route cell became impassable | P(big_rock) > 0.6 or slope > 25° | `replan` |
| max hazard on the route | > 0.3 (also a hard limit when planning) | `replan` |
| remaining route cost vs the map at plan time | > +20 % | `replan` |
| used + planned energy | > budget | drop lowest-value target (`skip_target`) |
| a check reached ≥ 50 % of its threshold | — | `keep_path` (logged once per cell) |
| next cell doubtful **before stepping** | P(big_rock) > 0.01 or < 4 looks | `pause_and_sense`: up to 3 extra looks from where it stands |
| boxed in (no route from the current cell) | — | back up ≤ 15 cells **along its own track** (driven cells are proven traversable), then replan |

Every decision is a `LogEntry(step, event, decision, evidence, confidence, reason)`:
* `evidence` is a flat dict of **raw numbers** — trigger `cell`, `rover_rc`, `threshold` + `threshold_name`,
  slope, P(slope>25°), hazard, all four class probabilities, entropy, old/new cost, old/new P(success), …
* `confidence = 1 − H/ln 4` at the trigger cell (computed, never guessed).
* `reason` is produced **only by a template** from the evidence, e.g.
  *"Replan at (17, 117): cell (20, 120) now impassable: P(big_rock)=0.75 > 0.60 (hazard_prob=0.37);
  old route P(success)=47.9% over 209 m → new route 79.3% over 209 m"*.

The rover never reads the ground truth directly — only through the sensor model. "Boulder cells entered"
is a ground-truth check computed for evaluation only: over 16 seeded runs with a hidden boulder field
(seeds 3/7/11/42 × synthetic/HiRISE × balanced/science) it is **2 cells in total**, all 16 reach the goal.

## 6. Science targets

`value = 0.4·ice + 0.3·rim/layering + 0.3·P(bedrock)`, where rim/layering = convex curvature × a band around
12° slope. Candidates are non-max-suppressed peaks inside the rover's reachable region. Each gets a
50-sample bootstrap (bedrock drawn from the belief, ice + σ 0.1 noise) → **value ± std**, an
`info_gain = value × normalised entropy`, and `utility = value − 0.0005·path_cost/cell_size − 0.5·(1 − P(success))` (cost per cell, so the penalty is site-scale independent).
Selection is greedy by utility per Wh, re-planning from each chosen target, and accepts a target only if
`used + leg + (target → goal) ≤ budget` (dashboard: budget = factor × direct-route energy).

## 7. Calibration notes (decisions made from measurements)

* **Prior floor 0.03 → 0.002.** With 0.03 every ~300-cell route scored P(success) ≈ 0 (0.5 × 0.03 boulder
  hazard per cell). 0.002 gives 62–74 % on safe routes, and a hidden boulder field still reaches
  P(big_rock) ≈ 0.5 after 2–3 looks. Orbital label noise was also switched to follow the orbital class
  frequencies — uniform noise had invented boulder fields on 5 % of the map.
* **Hazard is a hard planning limit** (cells with hazard > 0.3 set to inf when planning), otherwise A* could
  route through a 0.31 cell and the rover would replan on it every step.
* **Science weights** α = 0.0005, β = 0.5, budget 1.45× direct energy: selects 5–6 of 8 targets on seeds
  3/7/42 with the budget (not utility) as the binding constraint. Path cost in the utility is counted **per
  cell** — in metres, the 4 m/cell HiRISE site selected 0 of 8 targets.
* **Look before you step.** The low prior makes boulders in bedrock-rich terrain slow to confirm (3 rock /
  2 bedrock readings → P(big_rock) ≈ 0.09). Same 16-run benchmark, boulder cells entered: no pause 14,
  caution 0.1 → 7, **caution 0.01 → 2**, with no loss of goals.
* **Retreat along own track.** On rough HiRISE terrain the rover twice drove into a pocket and then learned
  every exit was a boulder (the diagonal it came in by is blocked by the no-corner-cutting rule). Treating
  driven-over cells as traversable and backing up along the track took the goal rate from 14/16 to 16/16.

## 7b. Onboard memory mode (tiered terrain tiles)

*Tiering idea adapted from [colibri](https://github.com/JustVugg/colibri) (MoE expert offloading: hot experts on GPU,
warm in RAM, cold streamed from disk, learned cache, lookahead prefetch) → terrain tiles. Architecture only — no
colibri code is used.* Code: `onboard/`; the regular rover (`autonomy/loop.py`) is untouched — `OnboardRover`
subclasses it and inherits every decision rule.

| piece | what it does |
|---|---|
| `onboard/quantize.py` | int16 DEM (0.01 m, auto-widened if the relief does not fit), uint8 slope (0.5°), roughness, cost (255 = impassable), **log-scale uint8 hazard** (1e-6…1, ±2.8 % relative), 2-bit classes; `report_error` |
| `onboard/tiered_store.py` | 64×64 tiles, one zlib blob each (row-delta). COLD flash (2 ms simulated read) → WARM compressed RAM → HOT decoded RAM (9 tiles). Resident 4× coarse map (mean elevation, **max** slope). Heat = decayed access + on-path + proximity; the rover tile is pinned. Path-lookahead prefetch on a 1-worker thread (sync option for tests); usage history in `onboard/.tile_usage.json` warms the next run |
| `onboard/planner.py` | uint8 `CompactCostMap` (3 B/cell instead of 32), lazy `CostView`s, `hierarchical_plan`: coarse corridor A* + rolling 64×64 local A*; curvature, P(slope>25°) and the entropy term are computed per window, never stored globally |
| `onboard/rover.py` | `OnboardRover`: terrain from tiles, compact cost state, hierarchical planning, `store.update_rover(rc, remaining_path)` every step (replans re-prioritise prefetch) |
| `onboard/bench.py` | same traverse + hidden hazard in FULL vs ONBOARD → `onboard/bench.json` |

**Precision is reported, not hidden.** Elevation error ≤ 0.5 cm (≈ 60× below the 0.3 m DEM noise); stored slope
error ≤ 0.25° (0.55° if re-derived from the int16 DEM, which is why slope is stored separately). A first version
stored hazard as linear uint8 — a typical per-cell hazard of 0.001 rounded to **0** and route P(success) jumped from
0.70 to 0.96. Hazard is therefore log-scaled; the onboard route now scores 0.687 vs 0.705 for full A*.

**What it saves (and what it does not).** Per layer: elevation 2×, slope/roughness 4×, cost state 10.7×, curvature and
P(slope>25°) are not stored at all. The rover's belief (4 float32 per cell) stays full-size in both modes because
every step updates it — so whole-rover state shrinks ≈ 2.5×, and the benefit grows with map size (a 1 m HiRISE strip is
≈ 10k × 20k cells ≈ 800 MB per float32 layer). Measured by `python -m onboard.bench` (tracemalloc peak during rover
boot + traverse; dashboard replay telemetry excluded in both modes; tile flash = storage):

| site | peak RAM full → onboard | rover state | HOT hit / stalls / prefetch hit | replans F/O | driven Δ | P(success) Δ | wall F/O |
|---|---|---|---|---|---|---|---|
| 200 (200×200) | 8.6 MB → 6.0 MB (**1.43×**) | 3.1 MB → 1.3 MB (2.48×) | 100.0% / 0 / 100% | 4/3 | -0.0 % | -1.8 pts | 11 s / 4 s |
| hirise (300×300) | 18.5 MB → 7.5 MB (**2.48×**) | 7.0 MB → 2.8 MB (2.48×) | 99.9% / 1 / 100% | 4/3 | -0.1 % | -0.1 pts | 23 s / 6 s |
| 1024 (1024×1024) | 224.1 MB → 47.7 MB (**4.7×**) | 81.8 MB → 31.2 MB (2.62×) | 99.7% / 10 / 84% | 6/4 | +0.3 % | +0.1 pts | 399 s / 38 s |

All runs reached the goal with 0 boulder cells entered in both modes. Wall times include tracemalloc overhead.

## 8. Assumptions and limitations

* **Synthetic ground truth.** Terrain classes and ice are procedural on the synthetic site. On the HiRISE
  site the **elevation is real** but classes/ice are **derived** from its slope, roughness and curvature
  (bedrock = slope > 15°, sand = local depressions on gentle slopes, boulders = roughest 1 %,
  ice = concave ∧ shadowed). This is an assumption, not mapped geology.
* **Independent observations.** Repeated navcam looks are treated as independent, so belief can become
  over-confident; the sensor noise model is the per-class confusion matrix only (no range/lighting effects).
* **Residual risk.** Sensor false negatives still let the rover onto ~1 boulder cell per 8 runs near
  boulder-rich science targets; the dashboard reports it rather than hiding it.
* **Isotropic energy.** Energy uses worst-case grade per cell, not travel direction.
* **Static world.** Hazards are injected before driving; there is no dynamic replanning cost model (D* Lite
  would make replans incremental — currently each replan is a fresh A*, ~50 ms).
* **Perception.** Until `perception/confusion.npy` is trained on Colab, the sensor uses the fallback matrix.

## 9. How to run

Python 3.10+ (developed on 3.11, Windows). The `.venv` was created with **uv**, which has no pip:

```powershell
uv venv .venv
uv pip install --python ".\.venv\Scripts\python.exe" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ".\.venv\Scripts\python.exe" -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

| Command | What it does |
|---|---|
| `python -m streamlit run app.py` | Dashboard |
| `python -m pytest -q` | Full test suite (122 tests, ~35 s) |
| `python -m autonomy.loop` / `--science` | Headless traverse with an injected hazard → `outputs/traverse.png`, `outputs/decision_log.json` |
| `python -m planner.pareto` | Route modes table + `outputs/pareto.png` |
| `python -m science.targets` | Ranked targets + budgeted selection |
| `python -m world.real_dem` | Fetch the HiRISE crop (~1 MB, windowed HTTP read) if missing |
| `python -m perception.train --smoke` / `perception.infer --smoke` | CPU smoke test of the perception pipeline |

Perception training (Colab GPU):

```
!pip install -q segmentation-models-pytorch datasets
!python -m perception.train --cache /content/cache     # → perception/unet.pt, confusion.npy, metrics.json
!python -m perception.infer                             # → outputs/perception/sample_*.png, infer_metrics.json
```

## 10. Data and references

All project code was written during the hackathon. These were read for **logic only** — no files copied:

* **AtsushiSakai/PythonRobotics** — A* open/closed-set structure, heuristic.
* **1piyalib/rover_pathfinder** — DEM → grid → slope-limited neighbours → A* pipeline.
* **utiasSTARS/gplanetary-nav** — slope-dependent energy idea, multi-objective routing.
* **mchancan/PlanetaryPathBench** — HiRISE-based planning benchmark (motivation for the real-DEM test).
* **qubvel-org/segmentation_models.pytorch** (library) — U-Net / ResNet-34 encoder.
* **AI4MARS** (Hugging Face `hassanjbara/AI4MARS`) — terrain-class training data.
* **USGS Astrogeology HiRISE controlled DTMs** — `DTEPD_076532_0920_076203_0920_A01`, a 600×600 px crop
  stored in `data/` (source URL in `world/real_dem.py` and in the GeoTIFF tags).
