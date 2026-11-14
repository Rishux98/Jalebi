# ODYSSEY — 8-minute demo script

**Before you start (T−10 min)**
1. `.\.venv\Scripts\Activate.ps1` then `python -m streamlit run app.py` → open http://localhost:8501.
2. Click through once with the defaults so everything is cached: **Inject hazard on path** (≈3 s first time),
   then **Plan**. Set World source → *HiRISE south-polar DTM*, click **Inject hazard on path**, back to *Synthetic*.
3. Sidebar defaults: Synthetic · seed 42 · 200 · balanced · λ 1.0 · budget 1.45× · sensor 6.
4. Have `outputs/traverse.png` and `outputs/decision_log.json` open in a second window (fallback).

---

### 0:00 – 1:00 · Problem (no clicks)
> "Mars is up to 20 light-minutes away. MARS-09 can't wait for mission control at every boulder — it has to
> plan, notice it was wrong, and fix it on its own. And because nobody watched it decide, every decision has
> to be auditable afterwards."

### 1:00 – 2:30 · Architecture (README diagram on screen)
> "Orbit gives us elevation and a coarse class map. Two uncertainties: what the ground is made of — a
> per-cell Bayesian belief — and how steep it really is — Monte-Carlo over DEM noise. The key link: our
> AI4MARS segmentation network's **confusion matrix is the rover's sensor model**. If the network confuses
> sand with soil 18 % of the time, the rover's belief update knows that."

### 2:30 – 3:30 · Plan the routes  — tab **🚀 Mission**, then **🧭 Routes**
1. Press **Plan** (the page starts on a "Configure the mission" card). Mission tab: point at the four routes on
   the uncertainty overlay and press **▶ Play** to show the planned route being driven.
   > "Shortest cuts straight through terrain orbit can't see well — purple is uncertainty."
2. Routes tab: Pareto chart + table.
   > "Shortest: 285 m, 62 % success under our belief. Balanced: 320 m, 71 %. The grey crosses are the same
   > routes scored against the hidden truth — shortest is really 73 %, the science route only… that gap
   > is what the rover must discover."

### 3:30 – 4:15 · Uncertainty — tab **❔ Uncertainty**
> "Left: entropy from orbit. P(slope > 25°) from 30 Monte-Carlo DEM draws. These feed the cost and the
> hazard probability of every cell." Hover a cell on **⛰️ Cost & Hazards** → per-component breakdown.

### 4:15 – 6:00 · The money shot — **💥 Inject hazard on path**
1. Click **💥 Inject hazard on path** (sidebar). Caption: *"hidden boulder field r=3 planted at (22, 122)…
   rover not told"*. Metrics: **goal reached ✅ · 4 replans · 0 boulder cells entered**.
2. Mission tab → press **▶ Play** under the 3-D terrain (or **2×**; drag the step slider to ~100, then 106, 114).
   The cyan rover drives; the ghosted planned path shows where it deviated; red ◆ = replans and the chart title
   switches to *"Step 106 · REPLAN: P(big_rock)=0.75 > 0.60"*. Toggle **🔁 Loop animation** to leave it running
   while you talk. Read the decision under the map:
   > "Step 106: cell (21, 121) now impassable — P(big_rock) 0.75 > 0.60. Keeping the old route: 48 %
   > success. New route: 76 %."
   At step 114 point out **pause_and_sense**: the next cell was 50 % boulder, so the rover stopped, took
   another look (→ 97 %) and replanned instead of stepping on it.
3. **📜 Decision Log**: click the first `replan` row → map highlights the trigger cell and the rover; the
   evidence JSON below has every raw number. Confidence = 1 − entropy/ln 4 at that cell.
   > "Nothing here is free text — every sentence is generated from those numbers."

### 6:00 – 6:45 · Science — set Planner mode → **science**, click **💥 Inject hazard on path**, tab **🔬 Science**
> "Eight candidates, value ± bootstrap uncertainty. Under a budget of 1.45× the direct route, five are
> selected; the rest are skipped with the exact energy they would have needed. During the drive the rover
> re-checks the budget at every replan and drops the lowest-value target if detours would overrun."

### 6:45 – 8:00 · Robustness & limits
1. Planner mode back to **balanced**, **World source → HiRISE south-polar DTM**, **💥 Inject hazard on path**.
   > "Real HiRISE elevation, 1.2 km at 4 m per cell, fetched by a windowed HTTP read — same pipeline,
   > same replanning, under 5 seconds."
2. **👁️ Perception** tab: confusion matrix (trained if available, else labelled *fallback*) + MC-dropout samples.
3. Limits (say them): classes on HiRISE are derived from geometry, observations treated as independent,
   each replan is a fresh A*, and sensor false negatives still let the rover onto ~2 boulder cells in 16
   benchmark runs — the dashboard shows that number instead of hiding it. Next: D* Lite for incremental replanning, real orbital spectra for the prior.

---

## If something breaks
| Problem | Do this |
|---|---|
| Dashboard won't start | Show `outputs/traverse.png` + `outputs/decision_log.json` (regenerate: `python -m autonomy.loop`) |
| A tab errors | Press **Plan**, then **Inject hazard on path** again; or switch Map size to 100 |
| HiRISE missing | `python -m world.real_dem` (~15 s, 1 MB) or stay synthetic |
| No trained perception | Perception tab already says *fallback* — say so honestly |
