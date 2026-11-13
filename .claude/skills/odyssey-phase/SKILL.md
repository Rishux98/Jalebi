---
name: odyssey-phase
description: Workflow for running any ODYSSEY ROADMAP phase (P0–P6). Use whenever the user says "Phase P<n>", "run P3", "start the next phase", pastes a phase prompt from ROADMAP.md, or asks to implement a module listed in the CLAUDE.md layout.
---

# Running an ODYSSEY phase

Work through these steps in order and skip none. The deadline is tight, so ship a simple version that runs before you try anything clever.

## 1. Load context
- Re-read `CLAUDE.md` (hard rules, grid conventions, module layout, coding style).
- Re-read the matching phase block in `ROADMAP.md` §4, including its `Verify:` line.
- Read `core/types.py` and `core/config.py` (once they exist). They are the contract.
- Check `git status`. If the tree is dirty, say so before touching anything.

## 2. Present a plan, then wait
Show a short plan before writing code:
- Files to create or modify, with one line on what each does.
- Which dataclasses from `core/types.py` each function returns or consumes.
- New `Config` fields needed (with default values and units).
- The tests you will add.
- Anything in the phase prompt you intend to simplify or stub, and why.

Wait for approval unless the user said "go ahead" / "no plan needed".

## 3. Implement against `core/types.py` only
- Functions return dataclasses from `core/types.py`, never loose tuples.
- All thresholds and weights come from `core/config.py`. Never hard-code magic numbers in modules.
- Every module gets an `if __name__ == "__main__":` smoke demo that runs headless
  (use `matplotlib.use("Agg")` and save PNGs, never `plt.show()`).
- Every random process takes a `seed`, and uses `np.random.default_rng(seed)`.
- Grid code follows the `numpy-grid-algorithms` skill.
  Rover decisions follow the `explainable-decisions` skill.
- Read reference repos for logic only. Never paste their files.

**`core/types.py` is frozen after P0.** If a field must change:
1. Make the smallest possible change (prefer adding an optional field with a default).
2. Grep every usage and fix callers in the same change.
3. Put the change at the **top** of the final summary (see step 6).

## 4. Tests
- Add or extend `tests/test_<module>.py` with fast pytest smoke tests (< 2 s each where possible).
- Cover the edge cases listed in the phase prompt, plus at least one shape/dtype check
  (`float32`, `(H, W)`) on every returned array.
- Use small grids (e.g. 32x32) unless the test is a timing test.

## 5. Verify until green
- Run the phase's exact `Verify:` command from ROADMAP.md (run each part separately on Windows
  if `&&` chaining fails in the shell).
- Run `pytest -q`.
- Fix and re-run until both pass. Never mark a phase done with red tests or a crashing `__main__`.
  If something must be skipped, say so explicitly rather than hiding it.

## 6. Summary
End with exactly this structure:

```
⚠ core/types.py CHANGED: <what and why>        ← only if it changed; must be the first line
Phase P<n> — done | partial
Files: <created/modified list>
Verify: <command> → <result>; pytest -q → <N passed>
Demo artefacts: <PNGs / JSON written>
Simplifications / TODOs: <list, or "none">
Next: <first thing for the next phase>

Suggested commit:
feat(p<n>): <imperative summary under 60 chars>

- <bullet per module>
```

Don't commit unless the user asks. Only suggest the message.
