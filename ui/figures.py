"""Plotly figure builders for the dashboard (data in, ``go.Figure`` out; no Streamlit).

Colour language (mission-control theme, used everywhere):
  safe = teal · caution = amber · hazard = red · unknown/uncertain = purple · path = cyan · science target = gold star
  routes (categorical, fixed order) in route comparisons: shortest blue, balanced orange, safest teal, science gold
  maps: terrain = earth-tone elevation x hillshade; cost Inferno; uncertainty Purples; hazard_prob Reds fixed 0-1
  terrain classes: fixed semantic colours (soil, bedrock, sand, big_rock)
Grid convention: x = col, y = row (row 0 at the top).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from core.io import hillshade
from core.types import CLASS_NAMES, N_CLASSES, RC, CostMap, Plan, Target, World

SURFACE = "#1A1D24"
PLOT_BG = "#0E1117"
TEXT = "#E6E6E6"
TEXT_2 = "#A9AFBA"
GRID = "#2E323C"
# semantic colour language
SAFE, CAUTION, HAZARD, UNKNOWN, PATH, SCIENCE = "#2EC4B6", "#F4A261", "#E63946", "#9D7CD8", "#00E5FF", "#FFD166"
ROUTE_COLORS = {"shortest": "#3987e5", "balanced": "#FF6B35", "safest": SAFE, "science": SCIENCE}
TRACK = PATH
REPLAN = HAZARD
TARGET = SCIENCE
TARGET_SYMBOL = "star"
EARTH = [[0.0, "#241811"], [0.25, "#4f3220"], [0.5, "#83552f"], [0.75, "#b88a5a"], [1.0, "#e9d3ae"]]
CLASS_COLORS = ["#b07a45", "#8a8a93", "#e6cf7a", "#c0392b"]   # soil, bedrock, sand, big_rock
COMPONENTS = ("slope", "rough", "class", "energy", "uncertainty")
SHADE_SCALE = [[0.0, "#0b0b0a"], [1.0, "#7d7b74"]]             # dimmed hillshade base for dark surface


def seq(hex_color: str) -> list:
    """Dark-mode sequential ramp: one hue, recessive near-surface low end -> full hue at high end."""
    return [[0.0, "#232322"], [0.5, _mix(hex_color, 0.55)], [1.0, hex_color]]


def alpha_ramp(hex_color: str) -> list:
    """Overlay ramp: transparent at 0 -> opaque hue at max (low values let the hillshade show through)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return [[0.0, f"rgba({r},{g},{b},0)"], [0.35, f"rgba({r},{g},{b},0.35)"], [1.0, f"rgba({r},{g},{b},0.92)"]]


def _mix(hex_color: str, t: float) -> str:
    """Blend ``hex_color`` with the dark surface: t=1 -> colour, t=0 -> surface."""
    base = (0x23, 0x23, 0x22)
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(b + t * (c - b)):02x}" for b, c in zip(base, rgb))


# Dark-mode variants of the requested scales: Inferno without its near-black end (black = impassable only);
# Reds / Purples start at the dark surface instead of white so low values recede on the dark theme.
_INFERNO = ["#420a68", "#6a176e", "#932667", "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"]
SEQ = {"cost": [[i / (len(_INFERNO) - 1), c] for i, c in enumerate(_INFERNO)],
       "hazard": [[0.0, "#1E2129"], [0.15, "#5a1d24"], [0.5, "#b02a35"], [1.0, "#ff5a5f"]],
       "entropy": [[0.0, "#1E2129"], [0.5, "#5b4a8f"], [1.0, "#c9b6ff"]],
       "slope": seq("#F4A261")}
OVERLAY = {"entropy": alpha_ramp("#a99ff2")}


def _layout(fig: go.Figure, height: int = 520, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=PLOT_BG, height=height,
                      margin=dict(l=10, r=10, t=40 if title else 10, b=10), font=dict(color=TEXT, size=12),
                      title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
                      legend=dict(bgcolor="rgba(26,26,25,0.85)", bordercolor=GRID, borderwidth=1, font=dict(size=11)))
    return fig


def _finite(a: np.ndarray) -> np.ndarray:
    """float array with non-finite -> NaN (Plotly JSON-safe; NaN cells render as the plot background)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a), a, np.nan)


def _xy(cells: Sequence[RC]) -> tuple[list[int], list[int]]:
    a = np.asarray(list(cells), dtype=np.int64).reshape(-1, 2)
    return a[:, 1].tolist(), a[:, 0].tolist()


def _square(fig: go.Figure, shape: tuple[int, int]) -> None:
    h, w = shape
    fig.update_xaxes(range=[-0.5, w - 0.5], showgrid=False, zeroline=False, constrain="domain", title=None)
    fig.update_yaxes(range=[h - 0.5, -0.5], showgrid=False, zeroline=False, scaleanchor="x", constrain="domain",
                     title=None)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.07, xanchor="left", x=0))


# =========================================================================== 2-D maps
def grid_map(z: np.ndarray, colorscale: str | list, title: str, colorbar: str, zmin: float | None = None,
             zmax: float | None = None, customdata: np.ndarray | None = None, hovertemplate: str | None = None,
             height: int = 520, showscale: bool = True) -> go.Figure:
    """Heatmap of an (H, W) layer; NaN/inf cells show the dark plot background (= impassable for cost)."""
    hv = hovertemplate or f"r=%{{y}} c=%{{x}}<br>{colorbar}: %{{z:.3f}}<extra></extra>"
    fig = go.Figure(go.Heatmap(z=_finite(z), colorscale=colorscale, zmin=zmin, zmax=zmax, customdata=customdata,
                               hovertemplate=hv, showscale=showscale,
                               colorbar=dict(title=dict(text=colorbar, side="right"), thickness=12, len=0.85)))
    _square(fig, np.shape(z))
    return _layout(fig, height, title)


def terrain_rgb(world: World) -> np.ndarray:
    """(H, W, 3) uint8: earth-tone elevation colours modulated by hillshade (relief you can read at a glance)."""
    z = world.dem.astype(np.float64)
    zn = (z - z.min()) / (np.ptp(z) + 1e-9)
    stops = np.array([s for s, _ in EARTH])
    cols = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in EARTH], dtype=np.float64)
    rgb = np.stack([np.interp(zn, stops, cols[:, k]) for k in range(3)], axis=-1)
    shade = hillshade(world.dem, world.cell_size)[..., None].astype(np.float64)
    return np.clip(rgb * (0.35 + 0.75 * shade), 0, 255).astype(np.uint8)


def shaded_map(world: World, title: str = "", height: int = 520, overlay: np.ndarray | None = None,
               overlay_scale: str | list | None = None, overlay_name: str = "", zmax: float | None = None) -> go.Figure:
    """Terrain-coloured hillshade base map, optionally with a transparent-to-hue magnitude overlay."""
    fig = go.Figure(go.Image(z=terrain_rgb(world), x0=0, dx=1, y0=0, dy=1, hoverinfo="skip", name="terrain"))
    if overlay is not None:
        scale = OVERLAY["entropy"] if overlay_scale in (None, "Purples") else overlay_scale
        fig.add_trace(go.Heatmap(z=_finite(overlay), colorscale=scale, zmin=0, zmax=zmax,
                                 name=overlay_name, colorbar=dict(title=dict(text=overlay_name, side="right"),
                                                                  thickness=12, len=0.85),
                                 hovertemplate=f"r=%{{y}} c=%{{x}}<br>{overlay_name}: %{{z:.3f}}<extra></extra>"))
    _square(fig, world.shape)
    return _layout(fig, height, title)


def class_map(classes: np.ndarray, title: str, height: int = 420) -> go.Figure:
    """Categorical terrain-class map (0..3) with fixed class colours."""
    scale = []
    for k, col in enumerate(CLASS_COLORS):
        scale += [[k / N_CLASSES, col], [(k + 1) / N_CLASSES, col]]
    fig = go.Figure(go.Heatmap(z=np.asarray(classes, np.float64), zmin=-0.5, zmax=N_CLASSES - 0.5, colorscale=scale,
                               customdata=np.asarray(CLASS_NAMES, dtype=object)[np.asarray(classes, np.int64)],
                               hovertemplate="r=%{y} c=%{x}<br>%{customdata}<extra></extra>",
                               colorbar=dict(tickvals=list(range(N_CLASSES)), ticktext=list(CLASS_NAMES),
                                             thickness=12, len=0.85)))
    _square(fig, np.shape(classes))
    return _layout(fig, height, title)


def add_path(fig: go.Figure, cells: Sequence[RC], name: str, color: str, dash: str = "solid", width: float = 2.5,
             showlegend: bool = True) -> go.Figure:
    if len(cells) == 0:
        return fig
    x, y = _xy(cells)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=width, dash=dash),
                             hovertemplate=f"{name}<br>r=%{{y}} c=%{{x}}<extra></extra>", showlegend=showlegend))
    return fig


def add_points(fig: go.Figure, cells: Sequence[RC], name: str, color: str, symbol: str = "circle", size: int = 11,
               text: Sequence[str] | None = None, showlegend: bool = True) -> go.Figure:
    if len(cells) == 0:
        return fig
    x, y = _xy(cells)
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name, text=list(text) if text is not None else None,
                             marker=dict(color=color, symbol=symbol, size=size, line=dict(color=SURFACE, width=2)),
                             hovertemplate=(f"{name}<br>r=%{{y}} c=%{{x}}" + ("<br>%{text}" if text is not None else "")
                                            + "<extra></extra>"), showlegend=showlegend))
    return fig


def add_start_goal(fig: go.Figure, world: World) -> go.Figure:
    add_points(fig, [world.start], "start", "#1baf7a", "circle", 13)
    return add_points(fig, [world.goal], "goal", "#e87ba4", "star", 17)


def add_rings(fig: go.Figure, injected: Sequence[dict], name: str = "injected hazard") -> go.Figure:
    for i, inj in enumerate(injected):
        r0, c0 = inj["rc"]
        rad = float(inj["radius"]) + 0.5
        t = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scatter(x=(c0 + rad * np.cos(t)).tolist(), y=(r0 + rad * np.sin(t)).tolist(), mode="lines",
                                 line=dict(color=HAZARD, width=2.5, dash="dot"), name=name, showlegend=i == 0,
                                 hovertemplate=f"{name} r={inj['radius']:g} at ({r0}, {c0})<extra></extra>"))
    return fig


def cost_map(cm: CostMap, title: str = "Cost (black = impassable)", height: int = 520,
             zmax: float | None = None) -> go.Figure:
    """Cost heatmap whose hover shows the per-component breakdown + hazard_prob of the cell."""
    names = [k for k in COMPONENTS if k in cm.components]
    cd = np.round(np.stack([_finite(cm.components[k]) for k in names] + [_finite(cm.hazard_prob)], axis=-1), 3)
    hv = ("r=%{y} c=%{x}<br><b>cost %{z:.3f}</b><br>"
          + "<br>".join(f"{k}: %{{customdata[{i}]:.3f}}" for i, k in enumerate(names))
          + f"<br>hazard_prob: %{{customdata[{len(names)}]:.3f}}<extra></extra>")
    fin = cm.cost[np.isfinite(cm.cost)]
    if zmax is None:
        zmax = float(np.percentile(fin, 99)) if fin.size else None
    return grid_map(cm.cost, SEQ["cost"], title, "cost / m", 0, zmax,
                    customdata=cd, hovertemplate=hv, height=height)


# =========================================================================== 3-D
def terrain_3d(world: World, paths: Sequence[tuple[str, Sequence[RC], str, str]] = (),
               points: Sequence[tuple[str, Sequence[RC], str, str]] = (), height: int = 620,
               max_cells: int = 160) -> go.Figure:
    """3-D DEM coloured by hillshade, with paths/points draped slightly above the surface.

    Args:
        paths: (name, cells, colour, dash) tuples.
        points: (name, cells, colour, plotly-3d symbol) tuples.
    """
    h, w = world.shape
    s = max(1, int(np.ceil(max(h, w) / max_cells)))
    dem = world.dem[::s, ::s]
    shade = hillshade(world.dem, world.cell_size)[::s, ::s]
    rows, cols = np.arange(0, h, s), np.arange(0, w, s)
    zn = (dem - dem.min()) / (np.ptp(dem) + 1e-9)
    fig = go.Figure(go.Surface(x=cols, y=rows, z=dem, surfacecolor=0.55 * zn + 0.45 * shade, colorscale=EARTH,
                               cmin=0, cmax=1, showscale=False,
                               hovertemplate="r=%{y} c=%{x}<br>z=%{z:.1f} m<extra></extra>", lighting=dict(ambient=0.9)))
    lift = 0.6
    for name, cells, color, dash in paths:
        if len(cells) == 0:
            continue
        a = np.asarray(list(cells), dtype=np.int64)
        fig.add_trace(go.Scatter3d(x=a[:, 1], y=a[:, 0], z=world.dem[a[:, 0], a[:, 1]] + lift, mode="lines", name=name,
                                   line=dict(color=color, width=6, dash=dash),
                                   hovertemplate=f"{name}<br>r=%{{y}} c=%{{x}}<extra></extra>"))
    for name, cells, color, symbol in points:
        if len(cells) == 0:
            continue
        a = np.asarray(list(cells), dtype=np.int64)
        fig.add_trace(go.Scatter3d(x=a[:, 1], y=a[:, 0], z=world.dem[a[:, 0], a[:, 1]] + 2 * lift, mode="markers",
                                   name=name, marker=dict(color=color, size=6, symbol=symbol,
                                                          line=dict(color=SURFACE, width=1)),
                                   hovertemplate=f"{name}<br>r=%{{y}} c=%{{x}}<extra></extra>"))
    zr = float(np.ptp(world.dem)) or 1.0
    fig.update_layout(scene=dict(xaxis=dict(title="col", backgroundcolor=PLOT_BG, gridcolor=GRID),
                                 yaxis=dict(title="row", autorange="reversed", backgroundcolor=PLOT_BG, gridcolor=GRID),
                                 zaxis=dict(title="z (m)", backgroundcolor=PLOT_BG, gridcolor=GRID),
                                 aspectmode="manual", aspectratio=dict(x=1, y=h / w, z=min(0.45, 3 * zr / max(h, w))),
                                 camera=dict(eye=dict(x=1.2, y=1.35, z=0.9))))
    return _layout(fig, height)


# =========================================================================== charts
def pareto_fig(sweep: Sequence[Plan], front: Sequence[Plan], modes: dict[str, Plan], height: int = 460) -> go.Figure:
    """Length vs P(success): sweep (belief), same routes vs truth, Pareto front, the 4 named modes."""
    fig = go.Figure()
    sw = [p for p in sweep if p.found]
    if sw:
        fig.add_trace(go.Scatter(x=[p.length_m for p in sw], y=[p.success_prob for p in sw], mode="markers",
                                 name="risk sweep (belief)", marker=dict(color=TEXT_2, size=9, line=dict(color=SURFACE, width=2)),
                                 text=[f"risk_weight={p.meta.get('risk_weight', 0):g}" for p in sw],
                                 hovertemplate="%{text}<br>%{x:.0f} m, P(success) %{y:.1%}<extra></extra>"))
        if all("success_prob_true" in p.meta for p in sw):
            fig.add_trace(go.Scatter(x=[p.length_m for p in sw], y=[p.meta["success_prob_true"] for p in sw],
                                     mode="markers", name="same routes vs truth",
                                     marker=dict(color=TEXT_2, size=9, symbol="x-thin", line=dict(width=2, color=TEXT_2)),
                                     hovertemplate="truth: %{x:.0f} m, P(success) %{y:.1%}<extra></extra>"))
    if front:
        fig.add_trace(go.Scatter(x=[p.length_m for p in front], y=[p.success_prob for p in front], mode="lines",
                                 line=dict(color=TEXT, width=1.5, shape="hv"), name="Pareto front", hoverinfo="skip"))
    for name, p in modes.items():
        if not p.found:
            continue
        fig.add_trace(go.Scatter(x=[p.length_m], y=[p.success_prob], mode="markers+text", name=name, text=[name],
                                 textposition="top center", textfont=dict(color=TEXT),
                                 marker=dict(color=ROUTE_COLORS.get(name, TEXT), size=14, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=(f"{name}: %{{x:.0f}} m, P(success) %{{y:.1%}} (belief)"
                                                f"<br>truth {p.meta.get('success_prob_true', float('nan')):.1%}"
                                                "<extra></extra>")))
    fig.update_xaxes(title="path length (m)", gridcolor=GRID)
    fig.update_yaxes(title="P(mission success)", tickformat=".0%", gridcolor=GRID)
    return _layout(fig, height, "Length vs safety trade-off")


def targets_bar(targets: Sequence[Target], selected_ids: set[int], height: int = 380) -> go.Figure:
    """Science value per target with ±1 std error bars; selected vs skipped."""
    fig = go.Figure()
    for label, color, keep in (("selected", ROUTE_COLORS["science"], True), ("skipped", "#5f5e5a", False)):
        ts = [t for t in targets if (t.id in selected_ids) == keep]
        if not ts:
            continue
        fig.add_trace(go.Bar(x=[f"#{t.id}" for t in ts], y=[t.value_mean for t in ts], name=label,
                             marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                             error_y=dict(type="data", array=[t.value_std for t in ts], color=TEXT_2, thickness=1.5),
                             customdata=[[t.info_gain, t.utility] for t in ts],
                             hovertemplate=("%{x}: value %{y:.2f} ± %{error_y.array:.2f}<br>info_gain %{customdata[0]:.2f}"
                                            "<br>utility %{customdata[1]:.2f}<extra></extra>")))
    fig.update_xaxes(title="target", categoryorder="array", categoryarray=[f"#{t.id}" for t in targets])
    fig.update_yaxes(title="science value (mean ± std)", gridcolor=GRID, rangemode="tozero")
    fig.update_layout(bargap=0.35)
    return _layout(fig, height, "Science value with bootstrap uncertainty")


def confusion_fig(C: np.ndarray, title: str, height: int = 440, highlight: tuple[int, int] | None = (2, 0)) -> go.Figure:
    """Sensor model heatmap: rows = true class, cols = predicted, values in %. ``highlight`` = (true, pred) cell."""
    C = np.asarray(C, np.float64)
    idx = list(range(N_CLASSES))
    fig = go.Figure(go.Heatmap(z=C * 100, x=idx, y=idx, colorscale=seq("#3987e5"), zmin=0, zmax=100,
                               text=[[f"{v:.0%}" for v in row] for row in C], texttemplate="%{text}",
                               textfont=dict(size=14, color=TEXT), colorbar=dict(title="%", thickness=12),
                               customdata=[[[CLASS_NAMES[t], CLASS_NAMES[p]] for p in idx] for t in idx],
                               hovertemplate="true %{customdata[0]} → observed %{customdata[1]}: %{z:.1f}%<extra></extra>"))
    fig.update_xaxes(title="predicted / observed class", tickvals=idx, ticktext=list(CLASS_NAMES), side="bottom")
    fig.update_yaxes(title="true class", tickvals=idx, ticktext=list(CLASS_NAMES), autorange="reversed")
    if highlight is not None:
        t, p = highlight
        fig.add_shape(type="rect", x0=p - 0.5, x1=p + 0.5, y0=t - 0.5, y1=t + 0.5, line=dict(color=CAUTION, width=3))
        fig.add_annotation(x=p + 0.5, y=t, xanchor="left", text=f"← {CLASS_NAMES[t]} seen as {CLASS_NAMES[p]}",
                           showarrow=False, font=dict(color=CAUTION, size=12), bgcolor="rgba(14,17,23,0.8)")
    return _layout(fig, height, title)


if __name__ == "__main__":
    from pathlib import Path

    from belief.bayes import init_prior
    from costmap.fuse import costmap_from_belief
    from terrain.features import compute_layers
    from world.synthetic import make_world

    w = make_world(64, 64, seed=3)
    L = compute_layers(w)
    b = init_prior(w)
    cm = costmap_from_belief(L, b)
    figs = {"cost": cost_map(cm), "classes": class_map(b.class_probs.argmax(-1), "belief"),
            "terrain3d": terrain_3d(w, [("demo", [w.start, w.goal], ROUTE_COLORS["balanced"], "solid")])}
    out = Path("outputs/figures_smoke.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                           for i, f in enumerate(figs.values())))
    print(f"{len(figs)} figures OK -> {out}")
