"""Client-side animated rover trace (Plotly frames): the rover drives the path inside the browser.

No Streamlit reruns during playback — Play / Pause / Restart / speed / scrub are Plotly ``updatemenus`` and a
``sliders`` control, so it stays smooth. Looping: Plotly has no native loop, so ``loop=True`` repeats the frame
list 3x and the dashboard adds a tiny JS hook that presses "⟲ Restart" on every ``plotly_animated`` event
(see ``LOOP_JS``); if that hook cannot reach the chart, the 3 repeats still give a long loop.

Every number shown (hover, replan title) comes from arrays / log evidence passed in — nothing is invented.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from core.types import CLASS_NAMES, RC, LogEntry

CYAN = "#00E5FF"
TRAIL = "#00E5FF"
REMAIN = "rgba(0,229,255,0.38)"
GHOST = "rgba(230,230,230,0.45)"
REPLAN_RED = "#E63946"
SENSOR = "rgba(0,229,255,0.55)"
SPEEDS = {"0.5×": 120, "1×": 60, "2×": 30, "4×": 15}       # ms per frame
ANIM_META = "odyssey-anim"
TITLE_HOLD = 6                                              # frames a replan reason stays in the title

HOVER = ("<b>step %{customdata[0]}</b> · (r %{customdata[1]}, c %{customdata[2]})<br>"
         "elevation %{customdata[3]:.1f} m · slope %{customdata[4]:.1f}°<br>"
         "class %{customdata[5]} (p=%{customdata[6]:.2f}) · hazard_prob %{customdata[7]:.3f}<br>"
         "distance %{customdata[8]:.1f} m · energy %{customdata[9]:.1f} Wh<extra></extra>")

# JS for components.html: loop every chart whose layout.meta == ANIM_META and data-loop is set.
LOOP_JS = """
<script>
(function () {
  const doc = window.parent.document;
  function hook() {
    doc.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      if (!gd.layout || gd.layout.meta !== '%META%' || gd.__odysseyLoop) return;
      gd.__odysseyLoop = true;
      gd.on('plotly_animated', function () {
        const btn = Array.from(gd.querySelectorAll('.updatemenu-button'))
          .find(function (b) { return (b.textContent || '').indexOf('Restart') >= 0; });
        if (btn) { setTimeout(function () { btn.dispatchEvent(new MouseEvent('click', {bubbles: true})); }, 250); }
      });
    });
  }
  hook(); setInterval(hook, 1000);
})();
</script>
""".replace("%META%", ANIM_META)


def short_reason(ev: dict) -> str:
    """Compact replan trigger built only from evidence numbers (explainable-decisions template)."""
    name, thr = ev.get("threshold_name", ""), float(ev.get("threshold", float("nan")))
    pr = ev.get("p_big_rock")
    tail = f", P(big_rock)={pr:.2f}" if isinstance(pr, (int, float)) else ""
    if name == "slope_max_deg":
        return f"slope {ev.get('slope_deg', float('nan')):.0f}° > {thr:.0f}°{tail}"
    if name == "big_rock_impassable_p":
        return f"P(big_rock)={pr:.2f} > {thr:.2f} (hazard {ev.get('hazard_prob', float('nan')):.2f})"
    if name == "hazard_prob_max":
        return f"hazard_prob {ev.get('hazard_prob', float('nan')):.2f} > {thr:.2f}{tail}"
    if name == "replan_cost_increase":
        pct = ev.get("cost_increase_pct", float("nan"))
        return f"route cost +{pct:.0%} > {thr:.0%}" if math.isfinite(pct) else f"route blocked{tail}"
    return f"{name} {thr:g}{tail}"


def keyframes(n: int, must: Sequence[int], max_frames: int = 150) -> list[int]:
    """<= max_frames sorted path indices: evenly spaced, always including 0, n-1 and every ``must`` index."""
    if n <= 0:
        return []
    if n <= max_frames:
        return list(range(n))
    keep = {0, n - 1, *[int(i) for i in must if 0 <= int(i) < n]}
    budget = max(max_frames - len(keep), 0)          # must-keep frames are never dropped
    keep.update(int(round(x)) for x in np.linspace(0, n - 1, budget))
    return sorted(keep)


def path_customdata(path: Sequence[RC], dem: np.ndarray, cell_size: float, meta: dict) -> list[list]:
    """Per-point hover rows: step, r, c, elevation, slope, class, p(class), hazard, cum. distance, cum. energy."""
    a = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    r, c = a[:, 0], a[:, 1]
    n = len(a)
    slope = np.asarray(meta["slope"])[r, c] if "slope" in meta else np.full(n, np.nan)
    hazard = np.asarray(meta["hazard"])[r, c] if "hazard" in meta else np.full(n, np.nan)
    if "class_probs" in meta:
        probs = np.asarray(meta["class_probs"])[r, c]
        k = probs.argmax(axis=1)
        pk = probs[np.arange(n), k]
    else:
        k, pk = np.zeros(n, int), np.full(n, np.nan)
    step = np.zeros(n)
    if n > 1:
        step[1:] = cell_size * np.where(np.abs(np.diff(a, axis=0)).sum(axis=1) == 2, math.sqrt(2), 1.0)
    dist = np.cumsum(step)
    e = meta.get("energy_wh_per_m")
    energy = np.cumsum(step * (np.asarray(e)[r, c] if e is not None else 0.0))
    elev = np.asarray(dem)[r, c]
    rd = lambda v, d: round(float(v), d) if math.isfinite(float(v)) else None  # noqa: E731 (JSON-safe, compact)
    return [[int(i), int(r[i]), int(c[i]), rd(elev[i], 2), rd(slope[i], 2), CLASS_NAMES[int(k[i])], rd(pk[i], 3),
             rd(hazard[i], 4), rd(dist[i], 2), rd(energy[i], 2)] for i in range(n)]


def _circle(rc: RC, radius: float, n: int = 40) -> tuple[list[float], list[float]]:
    t = np.linspace(0, 2 * np.pi, n)
    return (rc[1] + radius * np.cos(t)).tolist(), (rc[0] + radius * np.sin(t)).tolist()


def _heading(path: Sequence[RC], i: int) -> float:
    """Marker angle (deg clockwise from screen-up) of the move into/out of path[i]; rows grow downward."""
    j0, j1 = (i - 1, i) if i > 0 else (0, min(1, len(path) - 1))
    dr, dc = path[j1][0] - path[j0][0], path[j1][1] - path[j0][1]
    return math.degrees(math.atan2(dc, -dr)) if (dr or dc) else 0.0


def make_path_animation(fig_base: go.Figure, path: Sequence[RC], dem: np.ndarray, cell_size: float,
                        mode: str = "3d", plan_meta: dict[str, Any] | None = None,
                        log_entries: Sequence[LogEntry] = (), loop: bool = False,
                        max_frames: int = 150) -> go.Figure:
    """Add an animated rover to ``fig_base`` (a 3-D terrain or 2-D map figure; not modified in place).

    Args:
        path: cells driven / planned, in order. For a traverse, index i == autonomy step i.
        dem: (H, W) elevation (m) used to drape the 3-D trace and for hover.
        plan_meta: per-cell arrays for hover — ``slope`` (deg), ``class_probs`` (H, W, 4), ``hazard``,
            ``energy_wh_per_m`` — plus optional ``ghost_path`` (planned path drawn faded underneath),
            ``sensor_radius`` (cells, 2-D circle), ``label`` (title prefix).
        log_entries: ``replan`` LogEntries; ``entry.step`` indexes ``path``; shown as red diamonds and,
            when reached, as the frame title "Step k · REPLAN: <reason from evidence>".
        loop: repeat frames 3x (the dashboard's JS hook then restarts on ``plotly_animated``).
    """
    meta = dict(plan_meta or {})
    path = [(int(r), int(c)) for r, c in path]
    n = len(path)
    if n == 0:
        return go.Figure(fig_base)
    is3d = mode == "3d"
    fig = go.Figure(fig_base)
    dem = np.asarray(dem)
    lift = 0.8
    a = np.asarray(path)
    xs, ys = a[:, 1].tolist(), a[:, 0].tolist()
    zs = np.round(dem[a[:, 0], a[:, 1]] + lift, 2).tolist()
    cd = path_customdata(path, dem, cell_size, meta)
    label = meta.get("label", "Rover")

    replans = sorted({int(e.step): e for e in log_entries if 0 <= int(e.step) < n}.items())
    rep_idx = [s for s, _ in replans]
    rep_reason = {s: short_reason(e.evidence) for s, e in replans}

    def scatter(**kw):
        return go.Scatter3d(**kw) if is3d else go.Scatter(**kw)

    def pts(idx: Sequence[int]) -> dict:
        idx = list(idx)
        d = {"x": [xs[i] for i in idx], "y": [ys[i] for i in idx]}
        if is3d:
            d["z"] = [zs[i] + (0.3 if len(idx) == 1 else 0.0) for i in idx]
        return d

    # ---- static traces: ghost (planned) path, hoverable path points, replan diamonds
    ghost = meta.get("ghost_path")
    if ghost:
        g = np.asarray(ghost)
        kw = {"x": g[:, 1].tolist(), "y": g[:, 0].tolist()}
        if is3d:
            kw["z"] = np.round(dem[g[:, 0], g[:, 1]] + lift * 0.5, 2).tolist()
        fig.add_trace(scatter(**kw, mode="lines", name="planned (before replans)", hoverinfo="skip",
                              line=dict(color=GHOST, width=4 if is3d else 2, dash="dot")))
    fig.add_trace(scatter(**pts(range(n)), mode="markers", name="path points", customdata=cd, hovertemplate=HOVER,
                          marker=dict(size=2 if is3d else 4, color="rgba(0,229,255,0.25)"), showlegend=False))
    if rep_idx:
        rk = dict(size=6, symbol="diamond", color=REPLAN_RED, line=dict(color="#111", width=1)) if is3d else \
            dict(size=13, symbol="diamond", color=REPLAN_RED, line=dict(color="#111", width=1.5))
        fig.add_trace(scatter(**pts(rep_idx), mode="markers", name="replan", marker=rk,
                              customdata=[cd[i] for i in rep_idx], text=[rep_reason[i] for i in rep_idx],
                              hovertemplate="<b>REPLAN</b> %{text}<br>" + HOVER.replace("<extra></extra>", "") +
                                            "<extra></extra>"))

    # ---- animated traces (indices recorded for frames)
    first = len(fig.data)
    fig.add_trace(scatter(**pts(range(n)), mode="lines", name="remaining", hoverinfo="skip",
                          line=dict(color=REMAIN, width=4 if is3d else 2, dash="dash")))
    fig.add_trace(scatter(**pts([0]), mode="lines", name="driven", hoverinfo="skip",
                          line=dict(color=TRAIL, width=7 if is3d else 3.5)))
    rover_marker = dict(size=10, color=CYAN, line=dict(color="#0E1117", width=2)) if not is3d else \
        dict(size=7, color=CYAN, line=dict(color="#0E1117", width=1))
    fig.add_trace(scatter(**pts([0]), mode="markers", name="rover", marker=rover_marker,
                          customdata=[cd[0]], hovertemplate="<b>ROVER</b><br>" + HOVER))
    animated = [first, first + 1, first + 2]
    if not is3d:
        rad = float(meta.get("sensor_radius", 6))
        cx, cy = _circle(path[0], rad)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=f"sensor r={rad:g}", hoverinfo="skip",
                                 line=dict(color=SENSOR, width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode="markers", name="heading", hoverinfo="skip",
                                 showlegend=False, marker=dict(symbol="arrow", size=15, color=CYAN,
                                                               angle=_heading(path, 0),
                                                               line=dict(color="#0E1117", width=1))))
        animated += [first + 3, first + 4]

    # ---- frames
    idxs = keyframes(n, rep_idx, max_frames)
    title_for: dict[int, str] = {}
    last_rep, last_k = None, -10**9
    for k, i in enumerate(idxs):
        if i in rep_reason:
            last_rep, last_k = i, k
        if last_rep is not None and k - last_k < TITLE_HOLD:
            title_for[i] = f"Step {i} · REPLAN: {rep_reason[last_rep]}"
        else:
            title_for[i] = f"{label} · step {i} / {n - 1} · {cd[i][8]:.0f} m · {cd[i][9]:.0f} Wh"

    def frame_data(i: int) -> list:
        data = [scatter(**pts(range(i, n))), scatter(**pts(range(0, i + 1))),
                scatter(**pts([i]), customdata=[cd[i]])]
        if not is3d:
            cx, cy = _circle(path[i], float(meta.get("sensor_radius", 6)))
            data += [go.Scatter(x=cx, y=cy), go.Scatter(x=[xs[i]], y=[ys[i]], marker=dict(angle=_heading(path, i)))]
        return data

    base_frames = [go.Frame(name=f"f{i}", data=frame_data(i), traces=animated,
                            layout=go.Layout(title=dict(text=title_for[i]))) for i in idxs]
    frames = list(base_frames)
    if loop:
        for rep in (1, 2):
            frames += [go.Frame(name=f"{fr.name}_{rep}", data=fr.data, traces=animated, layout=fr.layout)
                       for fr in base_frames]
    fig.frames = frames

    def play(ms: int, fromcurrent: bool = True) -> list:
        return [None, {"frame": {"duration": ms, "redraw": True}, "transition": {"duration": 0},
                       "fromcurrent": fromcurrent, "mode": "immediate"}]

    btn = dict(bgcolor="#1A1D24", bordercolor="#3a3d46", font=dict(color="#E6E6E6", size=12))
    fig.update_layout(
        meta=ANIM_META,
        title=dict(text=title_for[idxs[0]], font=dict(size=13, color="#E6E6E6"), x=0.01, xanchor="left"),
        updatemenus=[
            dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top", showactive=False,
                 pad=dict(t=4, r=4), **btn,
                 buttons=[dict(label="▶ Play", method="animate", args=play(SPEEDS["1×"])),
                          dict(label="❚❚ Pause", method="animate",
                               args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                              "transition": {"duration": 0}}]),
                          dict(label="⟲ Restart", method="animate", args=play(SPEEDS["1×"], fromcurrent=False))]),
            dict(type="buttons", direction="left", x=1.0, y=-0.02, xanchor="right", yanchor="top", showactive=False,
                 pad=dict(t=4), **btn,
                 buttons=[dict(label=s, method="animate", args=play(ms)) for s, ms in SPEEDS.items()]),
        ],
        sliders=[dict(active=0, x=0.0, y=-0.09, len=1.0, xanchor="left", yanchor="top", pad=dict(t=6, b=4),
                      currentvalue=dict(prefix="step ", font=dict(color="#E6E6E6", size=11)),
                      font=dict(color="#9aa0aa", size=9), bgcolor="#2a2d36", activebgcolor=CYAN,
                      bordercolor="#3a3d46", tickcolor="#3a3d46",
                      steps=[dict(label=str(i), method="animate",
                                  args=[[f"f{i}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                                    "transition": {"duration": 0}}]) for i in idxs])],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=110),
                      showlegend=False)          # the dashboard draws a compact HTML legend under the chart
    return fig
