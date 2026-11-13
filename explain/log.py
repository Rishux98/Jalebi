"""Decision log: every rover decision as a traceable LogEntry with raw-number evidence.

Contract (see the explainable-decisions skill):
  * ``evidence`` is a flat dict of plain JSON-serialisable values and always contains
    ``cell`` [r, c] (trigger cell), ``rover_rc`` [r, c], ``threshold`` and ``threshold_name``.
  * ``confidence = 1 - entropy / ln 4`` at the trigger cell (or a stated mean, ``confidence_basis``).
  * ``reason`` is built ONLY by the template for that decision from evidence numbers.

Non-finite numbers (e.g. ``new_cost = inf`` when the old route is now blocked) are kept as floats;
``json.dumps`` writes them as ``Infinity``, which Python's ``json`` reads back unchanged.

Run: python -m explain.log
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from core.types import N_CLASSES, RC, Belief, CostMap, LogEntry, TerrainLayers

REQUIRED_KEYS: tuple[str, ...] = ("cell", "rover_rc", "threshold", "threshold_name")
LN4 = float(np.log(N_CLASSES))


# =========================================================================== evidence helpers
def _plain(v: Any) -> Any:
    """numpy scalars / arrays / tuples -> plain Python (JSON-serialisable)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        raise TypeError("evidence must be flat (no nested dicts)")
    return v


def _rc_list(rc: Any, name: str) -> list[int]:
    r, c = rc
    return [int(r), int(c)]


def make_evidence(cell: RC, rover_rc: RC, threshold: float, threshold_name: str, **nums: Any) -> dict:
    """Build a flat evidence dict with the required keys; casts everything to plain Python."""
    ev: dict[str, Any] = {"cell": _rc_list(cell, "cell"), "rover_rc": _rc_list(rover_rc, "rover_rc"),
                          "threshold": float(threshold), "threshold_name": str(threshold_name)}
    for k, v in nums.items():
        ev[k] = _plain(v)
    return ev


def cell_evidence(belief: Belief, layers: TerrainLayers, costmap: CostMap, rc: RC) -> dict[str, float]:
    """Raw per-cell inputs behind a decision: slope, P(slope over limit), hazard, cost, class probs, entropy."""
    r, c = int(rc[0]), int(rc[1])
    p = belief.class_probs[r, c]
    return {
        "slope_deg": float(layers.slope_deg[r, c]),
        "p_slope_exceed": float(layers.p_slope_exceed[r, c]),
        "hazard_prob": float(costmap.hazard_prob[r, c]),
        "cell_cost": float(costmap.cost[r, c]),
        "p_soil": float(p[0]), "p_bedrock": float(p[1]), "p_sand": float(p[2]), "p_big_rock": float(p[3]),
        "entropy": float(belief.entropy[r, c]),
    }


def confidence_at(belief: Belief, rc: RC) -> float:
    """1 - H/ln4 at one cell, clipped to [0, 1]."""
    return float(np.clip(1.0 - float(belief.entropy[int(rc[0]), int(rc[1])]) / LN4, 0.0, 1.0))


def confidence_mean(belief: Belief, cells: Iterable[RC]) -> float:
    """Mean of 1 - H/ln4 over cells (for path / target-disc decisions); 0 if no cells."""
    a = np.asarray(list(cells), dtype=np.int64).reshape(-1, 2)
    if a.size == 0:
        return 0.0
    return float(np.clip(1.0 - belief.entropy[a[:, 0], a[:, 1]].mean() / LN4, 0.0, 1.0))


# =========================================================================== reason templates
def _cell(ev: dict) -> str:
    return f"({ev['cell'][0]}, {ev['cell'][1]})"


def _pct(x: float) -> str:
    return "inf" if not math.isfinite(x) else f"{x:.1%}"


def _num(x: float, fmt: str = ".1f") -> str:
    return "inf" if not math.isfinite(x) else format(x, fmt)


def reason_plan(ev: dict) -> str:
    return (f"Plan [{ev['mode']}] {tuple(ev['rover_rc'])} -> {_cell(ev)}: {ev['length_m']:.0f} m, "
            f"{ev['energy_wh']:.0f} Wh, P(success)={ev['success_prob']:.1%}, mean entropy on path "
            f"{ev['path_entropy_mean']:.2f} nats, {ev['n_waypoints']} waypoint(s), "
            f"{ev['threshold_name']}={ev['threshold']:g}")


def _replan_trigger(ev: dict) -> str:
    name = ev["threshold_name"]
    if name == "hazard_prob_max":
        return (f"hazard_prob={ev['hazard_prob']:.2f} > {ev['threshold']:.2f} at {_cell(ev)} "
                f"(P(big_rock)={ev['p_big_rock']:.2f}, P(sand)={ev['p_sand']:.2f}, slope={ev['slope_deg']:.1f} deg)")
    if name == "big_rock_impassable_p":
        return (f"cell {_cell(ev)} now impassable: P(big_rock)={ev['p_big_rock']:.2f} > {ev['threshold']:.2f} "
                f"(hazard_prob={ev['hazard_prob']:.2f})")
    if name == "slope_max_deg":
        return f"cell {_cell(ev)} now impassable: slope={ev['slope_deg']:.1f} > {ev['threshold']:.1f} deg"
    if name == "replan_cost_increase":
        return (f"remaining path cost {_num(ev['old_cost'])} -> {_num(ev['new_cost'])} "
                f"(+{_pct(ev['cost_increase_pct'])} > {ev['threshold']:.0%}), worst cell {_cell(ev)}")
    return f"{name} threshold {ev['threshold']:g} crossed at {_cell(ev)}"


def reason_replan(ev: dict) -> str:
    back = f"; backs up {ev['retreat_cells']} cell(s) along its own track" if ev.get("retreat_cells") else ""
    return (f"Replan at {tuple(ev['rover_rc'])}: {_replan_trigger(ev)}{back}; old route P(success)="
            f"{ev['old_success_prob']:.1%} over {ev['old_length_m']:.0f} m -> new route "
            f"{ev['new_success_prob']:.1%} over {ev['new_length_m']:.0f} m")


def reason_keep_path(ev: dict) -> str:
    return (f"Keep path at {tuple(ev['rover_rc'])}: worst check {ev['threshold_name']}="
            f"{_num(ev['check_value'], '.3f')} at {_cell(ev)} is {_pct(ev['check_ratio'])} of limit "
            f"{ev['threshold']:g}")


def reason_hazard_detected(ev: dict) -> str:
    return (f"Hazard detected: {ev['n_new_impassable']} cell(s) became impassable, worst {_cell(ev)} "
            f"P(big_rock)={ev['p_big_rock']:.2f} > {ev['threshold']:.2f}, hazard_prob={ev['hazard_prob']:.2f}, "
            f"entropy={ev['entropy']:.2f} nats; {ev['n_on_path']} on current path")


def reason_select_target(ev: dict) -> str:
    return (f"Select target #{ev['target_id']} at {_cell(ev)}: value {ev['value_mean']:.2f} +/- "
            f"{ev['value_std']:.2f}, utility {ev['utility']:.2f} ({ev['utility_per_wh'] * 100:.2f}/100 Wh); "
            f"leg {ev['leg_length_m']:.0f} m P(success)={ev['leg_success_prob']:.1%}; energy "
            f"{ev['energy_used_wh']:.0f}+{ev['leg_energy_wh']:.0f}+{ev['return_energy_wh']:.0f}="
            f"{ev['energy_wh']:.0f} <= {ev['budget_wh']:.0f} Wh")


def reason_skip_target(ev: dict) -> str:
    name = ev["threshold_name"]
    head = f"Skip target #{ev['target_id']} at {_cell(ev)} (value {ev['value_mean']:.2f} +/- {ev['value_std']:.2f}): "
    if name == "budget_wh":
        return head + f"needs {_num(ev['energy_wh'], '.0f')} Wh > budget {ev['threshold']:.0f} Wh"
    if name == "reachable":
        return head + "no finite-cost path on current belief"
    if name == "sci_min_utility":
        return head + f"utility {ev['utility']:.2f} <= {ev['threshold']:.2f}"
    return head + f"{name} {ev['threshold']:g}"


def reason_continue(ev: dict) -> str:
    return (f"Reached target {_cell(ev)} at step {ev['step_reached']}; {ev['n_waypoints_left']} waypoint(s) left, "
            f"{ev['energy_wh']:.0f} Wh used")


def reason_goal_reached(ev: dict) -> str:
    return (f"Goal {_cell(ev)} reached after {ev['steps']} steps: {ev['distance_m']:.0f} m, "
            f"{ev['energy_wh']:.0f} Wh, {ev['replans']} replan(s), {ev['n_observed']} cells observed")


def reason_stop(ev: dict) -> str:
    if ev["threshold_name"] == "loop_max_steps":
        return f"Stop at {tuple(ev['rover_rc'])}: step limit {ev['threshold']:.0f} reached, goal {_cell(ev)} not reached"
    return (f"Stop at {tuple(ev['rover_rc'])}: no route to {_cell(ev)} on current belief "
            f"({ev['n_impassable']} impassable cells known)")


def reason_pause_and_sense(ev: dict) -> str:
    why = (f"P(big_rock)={ev['p_big_rock_before']:.2f} > {ev['threshold']:.2f}" if ev["threshold_name"] == "loop_caution_p"
           else f"only {ev['looks_before']} look(s) < {ev['threshold']:.0f}")
    return (f"Pause at {tuple(ev['rover_rc'])}: next cell {_cell(ev)} {why} -> {ev['extra_looks']} extra look(s): "
            f"P(big_rock) {ev['p_big_rock_before']:.2f} -> {ev['p_big_rock_after']:.2f}, looks "
            f"{ev['looks_before']} -> {ev['looks_after']}{'; route replanned' if ev['replanned'] else ''}")


REASONS: dict[str, Callable[[dict], str]] = {
    "pause_and_sense": reason_pause_and_sense,
    "plan": reason_plan,
    "replan": reason_replan,
    "keep_path": reason_keep_path,
    "hazard_detected": reason_hazard_detected,
    "select_target": reason_select_target,
    "skip_target": reason_skip_target,
    "continue": reason_continue,
    "goal_reached": reason_goal_reached,
    "stop": reason_stop,
}


def build_reason(decision: str, evidence: dict) -> str:
    """Template-built reason for a decision code (KeyError if no template exists)."""
    return REASONS[decision](evidence)


# =========================================================================== the log
class DecisionLog:
    """Append-only list of LogEntry with lookup and export helpers."""

    def __init__(self, entries: Iterable[LogEntry] | None = None) -> None:
        self.entries: list[LogEntry] = list(entries or [])

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LogEntry]:
        return iter(self.entries)

    def __getitem__(self, i: int) -> LogEntry:
        return self.entries[i]

    def append(self, entry: LogEntry) -> LogEntry:
        """Validate and append an entry (required evidence keys, JSON-serialisable evidence)."""
        missing = [k for k in REQUIRED_KEYS if k not in entry.evidence]
        if missing:
            raise ValueError(f"LogEntry evidence missing {missing}")
        json.dumps(entry.evidence)          # raises TypeError on numpy scalars / objects
        self.entries.append(entry)
        return entry

    def record(self, step: int, event: str, decision: str, evidence: dict, confidence: float) -> LogEntry:
        """Create a LogEntry with its template reason and append it."""
        return self.append(LogEntry(step=step, event=event, decision=decision, evidence=evidence,
                                    confidence=confidence, reason=build_reason(decision, evidence)))

    def entries_for_cell(self, r: int, c: int) -> list[LogEntry]:
        """All entries whose trigger ``evidence["cell"]`` is (r, c)."""
        return [e for e in self.entries if e.evidence.get("cell") == [int(r), int(c)]]

    def by_decision(self, decision: str) -> list[LogEntry]:
        return [e for e in self.entries if e.decision == decision]

    def counts(self) -> dict[str, int]:
        """Number of entries per decision code."""
        return dict(Counter(e.decision for e in self.entries))

    def to_records(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    def to_dataframe(self):
        """pandas DataFrame: step, event, decision, confidence, reason + evidence as ``ev_<key>`` columns."""
        import pandas as pd

        rows = []
        for e in self.entries:
            row = {"step": e.step, "event": e.event, "decision": e.decision,
                   "confidence": e.confidence, "reason": e.reason}
            row.update({f"ev_{k}": v for k, v in e.evidence.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"entries": self.to_records()}, indent=1))
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionLog":
        data = json.loads(Path(path).read_text())
        return cls(LogEntry(**d) for d in data["entries"])


if __name__ == "__main__":
    log = DecisionLog()
    ev = make_evidence((10, 12), (4, 4), 0.3, "hazard_prob_max", hazard_prob=np.float32(0.41),
                       p_big_rock=0.78, p_sand=0.02, slope_deg=6.5, old_cost=120.0, new_cost=float("inf"),
                       cost_increase_pct=float("inf"), old_success_prob=0.12, new_success_prob=0.71,
                       old_length_m=80.0, new_length_m=92.0)
    e = log.record(5, "hazard_on_path", "replan", ev, confidence=0.83)
    print(e.reason)
    print(log.counts(), "| entries for (10, 12):", len(log.entries_for_cell(10, 12)))
    print(log.to_dataframe().filter(like="ev_").columns.tolist()[:6], "...")
