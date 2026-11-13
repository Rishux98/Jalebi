---
name: explainable-decisions
description: Rules for logging every ODYSSEY rover decision as a traceable LogEntry with numeric evidence and entropy-based confidence. Use when writing autonomy/loop.py, explain/log.py, science/targets.py reasons, replanning logic, or the dashboard's Decision Log tab.
---

# Explainable decisions

Judges score explainability and traceability, so every decision the rover makes must be reproducible from the numbers it logged.

## The rule
Every decision appends exactly one `LogEntry` from `core/types.py` to the `DecisionLog`:

```python
LogEntry(step: int, event: str, decision: str, evidence: dict, confidence: float)
```

"Decision" includes plan, replan, keep_path, target_selected, target_skipped, hazard_detected,
goal_reached and abort. Logging only replans isn't enough. If the rover *considered* replanning and
kept the current path, log a `keep_path` entry too, at least whenever a check came close to a threshold.

## Field contract
- `step`: autonomy-loop step index (int, 0-based).
- `event`: what was observed, as a short snake_case code: `hazard_on_path`, `cost_increase`,
  `belief_update`, `target_ranked`, `budget_exhausted`, `goal_reached`.
- `decision`: what the rover did, as a snake_case code: `replan`, `keep_path`, `select_target`, `skip_target`, `stop`.
- `evidence`: a flat dict of JSON-serialisable **raw numbers** (cast with `float()` / `int()`, no numpy scalars).
  It must always include:
  - `cell`: `[r, c]`, the map cell that triggered the decision (this is what the dashboard highlights)
  - `rover_rc`: `[r, c]`, where the rover was
  - `threshold` + `threshold_name`: the rule that fired (e.g. `"hazard_prob_max"`, `0.3`)

  Then add every relevant raw input that applies:
  `slope_deg`, `p_slope_exceed`, `hazard_prob`, `p_soil`, `p_bedrock`, `p_sand`, `p_big_rock`,
  `entropy`, `old_cost`, `new_cost`, `cost_increase_pct`, `old_success_prob`, `new_success_prob`,
  `old_length_m`, `new_length_m`, `value_mean`, `value_std`, `energy_wh`, `budget_wh`.
- `confidence`: computed, never guessed:
  ```python
  confidence = float(1.0 - belief.entropy[r, c] / np.log(4))   # clipped to [0, 1]
  ```
  taken at the trigger `cell`. Non-cell decisions (e.g. budget) use the mean over the path or target cells, and evidence states which.

## Reason strings are templates, not prose
Build any human-readable `reason` or `decision` text **only** from evidence values via an f-string helper:

```python
def reason_replan(ev: dict) -> str:
    return (f"Replan at step: cell {tuple(ev['cell'])} hazard_prob={ev['hazard_prob']:.2f} "
            f"> {ev['threshold']:.2f} (P(big_rock)={ev['p_big_rock']:.2f}, slope={ev['slope_deg']:.1f}°); "
            f"success {ev['old_success_prob']:.1%} → {ev['new_success_prob']:.1%}")
```

Never write free text such as "terrain looked dangerous" or "chose a safer route". Every adjective must come with the number behind it. One helper per decision type lives in `explain/log.py`.

## DecisionLog API (explain/log.py)
- `append(entry)`; `__len__`; `to_dataframe()` (evidence flattened into `ev_<key>` columns); `to_json(path)`;
  `from_json(path)` so a precomputed log can back the demo (ROADMAP fallback).
- `entries_for_cell(r, c)` returns every entry whose `evidence["cell"]` matches, for map → log lookups.

## Dashboard traceability (P5, Decision Log tab)
- Any selected log row highlights `evidence["cell"]` (and `rover_rc`) on the map and shows the evidence JSON.
- Clicking a map cell lists `entries_for_cell(r, c)`.
- Display evidence as-is. Never recompute or round it differently in the UI.

## Tests (tests/test_log.py and test_loop.py)
- Every entry has `cell` within grid bounds, `0 ≤ confidence ≤ 1`, and evidence that round-trips through `json.dumps`.
- The injected-hazard run produces ≥ 1 `replan`, and its evidence has `hazard_prob` or `new_cost` above `threshold`.
- The reason string contains the formatted `hazard_prob` value (proving it was built from numbers).
