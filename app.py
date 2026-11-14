"""ODYSSEY · MARS-09 Autonomous Navigation — mission-control dashboard (Streamlit).

Run:  python -m streamlit run app.py
Heavy work lives in ui/pipeline.py (cached per sidebar setting), charts in ui/figures.py, the client-side
rover animation in ui/animation.py. Every tab body is wrapped so one failing panel never blanks the demo.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.config import DEFAULT
from core.types import CLASS_NAMES, N_CLASSES
from perception.sensor_model import get_sensor_info
from ui import figures as F
from ui.animation import LOOP_JS, make_path_animation
from ui.pipeline import MODES, build_routes, build_science, build_site, make_cfg, run_traverse
from world.real_dem import default_tile_available

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ODYSSEY · MARS-09", page_icon="🛰️", layout="wide")

MODE_DESC = {
    "shortest": "Pure distance — ignores terrain cost and risk (impassable cells are still avoided).",
    "balanced": "Default trade-off between distance, terrain cost and hazard risk.",
    "safest": "Strongly avoids hazards and unknown terrain; accepts a longer route.",
    "science": "Balanced weights, routed through the science targets that fit the energy budget.",
}
BADGE = {"replan": "🔴 REPLAN", "hazard_detected": "🟣 SENSE", "pause_and_sense": "🟣 SENSE",
         "keep_path": "🟢 SENSE", "select_target": "🟡 TARGET", "skip_target": "🟡 TARGET", "continue": "🟡 TARGET",
         "plan": "🔵 MISSION", "goal_reached": "🔵 MISSION", "stop": "🔴 MISSION"}

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.6rem; padding-bottom: 2rem;}}
  .ody-title {{font-size: 1.55rem; font-weight: 700; letter-spacing: .02em; margin: 0 0 .35rem 0;}}
  .ody-title span {{color: #FF6B35;}}
  .chip {{display:inline-block; padding:2px 11px; margin:0 6px 6px 0; border-radius:999px; font-size:.78rem;
          border:1px solid #2E323C; background:#1A1D24; color:#E6E6E6;}}
  .chip b {{color:#A9AFBA; font-weight:500; margin-right:4px;}}
  .chip.ok {{border-color:{F.SAFE}; color:{F.SAFE};}} .chip.warn {{border-color:{F.CAUTION}; color:{F.CAUTION};}}
  .chip.bad {{border-color:{F.HAZARD}; color:{F.HAZARD};}} .chip.info {{border-color:{F.PATH}; color:{F.PATH};}}
  .legend {{font-size:.76rem; color:#A9AFBA; margin:-.4rem 0 .6rem 0;}}
  .legend i {{display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px 0 10px; vertical-align:middle;}}
  .empty {{border:1px dashed #3a3d46; border-radius:12px; padding:2.2rem; text-align:center; background:#151821;}}
  .empty h3 {{margin:.2rem 0 .4rem 0;}}
</style>""", unsafe_allow_html=True)


# =========================================================================== cached pipeline
@st.cache_resource(show_spinner="Building site: DEM, terrain layers, orbital prior…", max_entries=8)
def get_site(src: str, size: int, seed: int, lam: float, budget: float, radius: int, conn: int):
    """``budget`` is a multiple of the direct balanced route's energy (works for any site size)."""
    return build_site(size, seed, make_cfg(lam, DEFAULT.sci_energy_budget_wh, radius, conn), source=src,
                      budget_factor=budget)


@st.cache_resource(show_spinner="Ranking science targets…", max_entries=8)
def get_science(src: str, size: int, seed: int, lam: float, budget: float, radius: int, conn: int):
    return build_science(get_site(src, size, seed, lam, budget, radius, conn))


@st.cache_resource(show_spinner="Planning route options + risk sweep…", max_entries=8)
def get_routes(src: str, size: int, seed: int, lam: float, budget: float, radius: int, conn: int):
    return build_routes(get_site(src, size, seed, lam, budget, radius, conn),
                        get_science(src, size, seed, lam, budget, radius, conn))


@st.cache_resource(show_spinner="Driving the rover…", max_entries=16)
def get_traverse(src: str, size: int, seed: int, lam: float, budget: float, radius: int, conn: int, mode: str,
                 inject: bool, frac: float, hz_radius: float):
    site = get_site(src, size, seed, lam, budget, radius, conn)
    return run_traverse(size, seed, site.cfg, mode, inject, frac, hz_radius, site,
                        get_science(src, size, seed, lam, budget, radius, conn), source=src)


@st.cache_data(show_spinner="Building rover animation…", max_entries=24)
def get_animation(args: tuple, mode: str, run: str, frac: float, hz_radius: float, view: str, cm_state: str,
                  loop: bool, cost_zmax: float):
    """Animated figure (3-D terrain or 2-D cost map) for the planned path or the driven trajectory.
    Cached on every setting that changes it (seed, size, mode, λ, budget, radius, movement, hazard, view)."""
    site, routes = get_site(*args), get_routes(*args)
    if run == "plan":
        p = routes.modes.get(mode) or routes.modes["balanced"]
        path, entries, ghost, label = p.path, (), None, f"Planned · {mode}"
        belief, cm = site.belief, site.costmap
        world = site.world
    else:
        tr = get_traverse(*args, mode, run == "inject", frac, hz_radius)
        res = tr.result
        path, entries, ghost, label = res.trajectory, res.log.by_decision("replan"), res.plans[0].path, "Driven"
        belief, cm, world = res.belief, res.costmap, tr.world
    if view == "2d" and cm_state == "before":
        belief, cm = site.belief, site.costmap
    meta = dict(slope=site.layers.slope_deg, class_probs=belief.class_probs, hazard=cm.hazard_prob,
                energy_wh_per_m=cm.energy_wh_per_m, ghost_path=ghost, sensor_radius=site.cfg.sensor_radius,
                label=label)
    if view == "3d":
        base = F.terrain_3d(world, points=[("start", [world.start], F.SAFE, "circle"),
                                           ("goal", [world.goal], "#e87ba4", "diamond")], height=640)
    else:
        base = F.cost_map(cm, "", 600, zmax=cost_zmax)
    return make_path_animation(base, path, world.dem, world.cell_size, view, meta, entries, loop=loop)


def _json_safe(d: dict) -> dict:
    return {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def chip(label: str, value: str, kind: str = "") -> str:
    return f'<span class="chip {kind}"><b>{label}</b>{value}</span>'


def legend(items: list[tuple[str, str]]) -> None:
    st.markdown('<div class="legend">' + "".join(f'<i style="background:{c}"></i>{t}' for t, c in items) + "</div>",
                unsafe_allow_html=True)


MAP_LEGEND = [("path", F.PATH), ("safe", F.SAFE), ("caution", F.CAUTION), ("hazard / replan", F.HAZARD),
              ("unknown", F.UNKNOWN), ("science target ★", F.SCIENCE)]

# =========================================================================== sidebar
st.session_state.setdefault("run", None)          # None (idle) | "plan" | "traverse" | "inject"
sensor = get_sensor_info()
with st.sidebar:
    st.markdown("### 🛰️ ODYSSEY · MARS-09")
    sources = ["Synthetic"] + (["HiRISE south-polar DTM"] if default_tile_available() else [])
    source = st.selectbox("World source", sources, key="source",
                          help="HiRISE: real elevation (USGS controlled DTM crop, 1.2 km); terrain classes and ice "
                               "are derived from its slope/roughness/curvature (documented assumption).")
    src = "hirise" if source.startswith("HiRISE") else "synthetic"
    if not default_tile_available():
        st.caption("HiRISE crop missing — run `python -m world.real_dem` to fetch it (~1 MB).")
    c1, c2 = st.columns(2)
    seed = c1.number_input("Seed", 0, 9999, DEFAULT.seed, key="seed")
    size = c2.selectbox("Map size", [64, 100, 150, 200], index=3, key="size", disabled=src == "hirise",
                        help="cells per side (1 m cells); HiRISE tile is fixed at 300×300 cells of ~4 m")
    mode = st.selectbox("Planner mode", MODES, index=1, key="mode",
                        help="science = balanced weights routed through budget-selected targets")
    st.caption(MODE_DESC[mode])
    lam = st.slider("λ uncertainty weight", 0.0, 5.0, float(DEFAULT.lambda_unc), 0.25, key="lam",
                    help="cost weight of belief entropy (avoid unknown terrain)")
    st.caption("0 = trust the map's guesses · 3+ = strongly avoid unknown terrain")
    budget = st.slider("Energy budget (× direct route)", 1.0, 3.0, 1.45, 0.05, key="budget",
                       help="Budget = this × energy of the balanced direct start→goal route; the extra pays for "
                            "science detours. 1.45× ≈ 800 Wh on the 200×200 synthetic site.")
    st.caption("1.0 = straight to goal · higher = more battery for science detours")
    radius = st.slider("Sensor radius (cells)", 3, 12, int(DEFAULT.sensor_radius), key="radius",
                       help="navcam footprint radius used for every Bayesian belief update")
    st.caption("How far onboard cameras see (cells ≈ metres)")
    movement = st.radio("Rover movement", ["8-way (diagonals)", "4-way (horizontal / vertical only)"],
                        key="movement", help="4-way: the rover moves only up/down/left/right — simpler, grid-aligned "
                                             "routes (Manhattan distance), usually a little longer.")
    conn = 4 if movement.startswith("4") else 8
    with st.expander("💥 Hidden hazard"):
        frac = st.slider("Position along route", 0.1, 0.9, 0.4, 0.05, key="frac")
        hz_radius = st.slider("Boulder-field radius (cells)", 1, 6, 3, key="hz_radius")
    b1, b2 = st.columns(2)
    if b1.button("Plan", key="plan_btn", width="stretch"):
        st.session_state.run = "plan"
    if b2.button("Run traverse", key="run_btn", type="primary", width="stretch"):
        st.session_state.run = "traverse"
    if st.button("💥 Inject hazard on path", key="inject_btn", width="stretch",
                 help="Plant a hidden boulder field on the planned route, then drive. The rover is not told."):
        st.session_state.run = "inject"
    loop = st.toggle("🔁 Loop animation", key="loop", help="Off: plays once. On: frames repeat and the player "
                                                          "restarts automatically when it reaches the end.")
    _s = get_site(src, int(size), int(seed), float(lam), float(budget), int(radius), conn)
    st.caption(f"Energy budget = {budget:.2f} × {_s.direct_energy_wh:.0f} Wh (direct route) = "
               f"**{_s.cfg.sci_energy_budget_wh:.0f} Wh**")
    if sensor["source"] == "trained":
        miou = f" · mIoU = {sensor['miou']:.2f}" if sensor["miou"] is not None else ""
        st.markdown(chip("", f"🛰️ Sensor model: trained AI4MARS U-Net{miou}", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(chip("", "⚠️ Sensor model: fallback confusion matrix (placeholder until trained model is added)",
                         "warn"), unsafe_allow_html=True)

args = (src, int(size), int(seed), float(lam), float(budget), int(radius), conn)
run = st.session_state.run
site = get_site(*args)
world, cfg = site.world, site.cfg
H, W = world.shape
sci = routes = trav = res = plan = None
if run is not None:
    sci, routes = get_science(*args), get_routes(*args)
    plan = routes.modes.get(mode) or routes.modes["balanced"]
    if run in ("traverse", "inject"):
        trav = get_traverse(*args, mode, run == "inject", float(frac), float(hz_radius))
        res = trav.result

# =========================================================================== map view (pan / zoom)
ZOOMS = [1, 2, 4, 8]
vk_r, vk_c = f"view_r-{H}", f"view_c-{W}"            # keyed by map size so stale centres never exceed bounds
st.session_state.setdefault("view_zoom", 1)
st.session_state.setdefault(vk_r, H // 2)
st.session_state.setdefault(vk_c, W // 2)


def _clamp_center(r: float, c: float) -> tuple[int, int]:
    """Clamp a view centre so the zoom window stays inside the map (every pan press then moves the view)."""
    z = max(int(st.session_state.view_zoom), 1)
    hr, hc = H / (2 * z), W / (2 * z)
    return int(round(np.clip(r, hr, H - hr))), int(round(np.clip(c, hc, W - hc)))


def _pan(dr: int, dc: int) -> None:
    """Move the view by a quarter of the visible window (callback, runs before the widgets)."""
    z = max(int(st.session_state.view_zoom), 1)
    r, c = _clamp_center(st.session_state[vk_r], st.session_state[vk_c])
    r, c = _clamp_center(r + dr * max(1, H // (4 * z)), c + dc * max(1, W // (4 * z)))
    st.session_state[vk_r], st.session_state[vk_c] = min(r, H - 1), min(c, W - 1)


def _center(rc: tuple[int, int]) -> None:
    r, c = _clamp_center(rc[0], rc[1])
    st.session_state[vk_r], st.session_state[vk_c] = min(r, H - 1), min(c, W - 1)


def _reset_view() -> None:
    st.session_state.view_zoom = 1
    _center((H // 2, W // 2))


with st.sidebar:
    with st.expander("🧭 Map view (pan / zoom)", expanded=False):
        st.select_slider("Zoom", ZOOMS, key="view_zoom", format_func=lambda z: f"{z}×")
        _, a2, _ = st.columns(3)
        a2.button("⬆", key="pan_up", on_click=_pan, args=(-1, 0), width="stretch")
        l1, l2, l3 = st.columns(3)
        l1.button("⬅", key="pan_left", on_click=_pan, args=(0, -1), width="stretch")
        l2.button("⟲", key="pan_reset", on_click=_reset_view, width="stretch", help="Full map")
        l3.button("➡", key="pan_right", on_click=_pan, args=(0, 1), width="stretch")
        _, d2, _ = st.columns(3)
        d2.button("⬇", key="pan_down", on_click=_pan, args=(1, 0), width="stretch")
        st.slider("Horizontal (column)", 0, W - 1, key=vk_c)
        st.slider("Vertical (row)", 0, H - 1, key=vk_r)
        g1, g2 = st.columns(2)
        g1.button("Start", key="ctr_start", on_click=_center, args=(world.start,), width="stretch")
        g2.button("Goal", key="ctr_goal", on_click=_center, args=(world.goal,), width="stretch")
        st.caption("Applies to every 2-D map. You can also drag to pan and scroll to zoom inside a map.")


def show(fig, key: str, window: tuple[int, int, int] | None = None) -> None:
    """Render a Plotly figure full-width. 2-D maps follow the sidebar pan/zoom window, or ``window`` =
    (row, col, half-size) to centre on one cell. Drag = pan, wheel = zoom."""
    z = int(st.session_state.view_zoom)
    if fig.layout.yaxis.scaleanchor == "x":
        if window is not None:
            r0, c0, half = window
            half = min(half, (min(H, W) - 1) // 2)
            r0, c0 = int(np.clip(r0, half, H - 1 - half)), int(np.clip(c0, half, W - 1 - half))   # stay on the map
            fig.update_xaxes(range=[c0 - half - 0.5, c0 + half + 0.5])
            fig.update_yaxes(range=[r0 + half + 0.5, r0 - half - 0.5])
        elif z > 1:
            hr, hc = H / (2 * z), W / (2 * z)
            r0 = float(np.clip(st.session_state[vk_r], hr, H - hr))
            c0 = float(np.clip(st.session_state[vk_c], hc, W - hc))
            fig.update_xaxes(range=[c0 - hc - 0.5, c0 + hc - 0.5])
            fig.update_yaxes(range=[r0 + hr - 0.5, r0 - hr - 0.5])
        fig.update_layout(dragmode="pan")
    st.plotly_chart(fig, key=key, width="stretch", config={"scrollZoom": True, "displaylogo": False})


# =========================================================================== header bar
state = ("Idle" if run is None else "Planned" if run == "plan"
         else ("Arrived" if res is not None and res.reached_goal else "Stopped"))
state_kind = {"Idle": "", "Planned": "info", "Arrived": "ok", "Stopped": "bad"}[state]
st.markdown('<div class="ody-title"><span>ODYSSEY</span> · MARS-09 Autonomous Navigation</div>', unsafe_allow_html=True)
st.markdown(chip("world", f"{world.name} · {H}×{W} @ {world.cell_size:.2f} m") + chip("mode", mode)
            + chip("movement", f"{conn}-way")
            + chip("sensor", "trained" if sensor["source"] == "trained" else "fallback",
                   "ok" if sensor["source"] == "trained" else "warn")
            + chip("mission", state, state_kind)
            + (chip("hazard", trav.note, "warn") if run == "inject" and trav is not None else ""),
            unsafe_allow_html=True)

if run is None:
    st.markdown(f"""<div class="empty"><h3>🛰️ Configure the mission and press <span style="color:#FF6B35">Plan</span></h3>
    <div style="color:#A9AFBA">Pick a world, planner mode and budget in the sidebar. <b>Plan</b> computes four route options
    from the orbital prior · <b>Run traverse</b> drives the rover · <b>💥 Inject hazard on path</b> plants a hidden boulder
    field the rover has to discover and avoid on its own.</div>
    <div style="margin-top:.8rem; color:#6f7682; font-size:.85rem">Site ready: {world.name} · start {world.start} → goal
    {world.goal} · direct route ≈ {site.direct_energy_wh:.0f} Wh</div></div>""", unsafe_allow_html=True)
    st.stop()

# =========================================================================== metric cards
shortest = routes.modes.get("shortest")
budget_wh = cfg.sci_energy_budget_wh
length = res.distance_m if res is not None else plan.length_m
energy = res.energy_wh if res is not None else plan.energy_wh
succ = (res.plans[0].success_prob if (res is not None and res.plans) else plan.success_prob)
expanded = sum(p.expanded_nodes for p in res.plans) if res is not None else plan.expanded_nodes
n_targets = len(trav.targets) if trav is not None else (len(sci.selection.selected) if mode == "science" else 0)
visited = len(res.log.by_decision("continue")) if res is not None else 0
cards = st.columns(6, gap="small")


def card(col, label: str, value: str, delta: str | None = None, delta_color: str = "normal", help_: str | None = None,
         badge: tuple[str, str] | None = None) -> None:
    with col.container(border=True):
        st.metric(label, value, delta, delta_color=delta_color, help=help_)
        if badge:
            st.markdown(chip("", badge[0], badge[1]), unsafe_allow_html=True)


card(cards[0], "Path length", f"{length:.0f} m",
     f"{length - shortest.length_m:+.0f} m vs shortest" if shortest and shortest.found else None, "inverse")
card(cards[1], "Energy", f"{energy:.0f} Wh", f"{energy / budget_wh:.0%} of {budget_wh:.0f} Wh budget", "off")
s_kind, s_txt = (("ok", "≥ 95 %: go") if succ >= 0.95 else ("warn", "80–95 %: caution") if succ >= 0.80
                 else ("bad", "< 80 %: risky"))
card(cards[2], "Success probability", f"{succ:.1%}",
     f"{(succ - shortest.success_prob) * 100:+.1f} pts vs shortest" if shortest and shortest.found else None,
     help_="Π(1 − hazard_prob) along the planned route, under the rover's belief", badge=(s_txt, s_kind))
card(cards[3], "Replans", str(res.replans) if res is not None else "—",
     f"{res.meta['extra_looks']} extra looks" if res is not None else None, "off")
card(cards[4], "Targets visited", f"{visited} / {n_targets}" if n_targets else "—",
     "science mode" if mode == "science" else "science mode only", "off")
card(cards[5], "Expanded nodes", f"{expanded:,}", f"{len(res.plans)} plan(s)" if res is not None else "A* search", "off")
if res is not None:
    rk = res.meta["entered_true_hazard_cells"]
    st.markdown(chip("goal", "reached ✅" if res.reached_goal else "not reached ⛔", "ok" if res.reached_goal else "bad")
                + chip("driven", f"{res.steps} steps") + chip("boulder cells entered (ground-truth check)", str(rk),
                                                            "ok" if rk == 0 else "bad"), unsafe_allow_html=True)

hmax = float(np.log(N_CLASSES))
fin_cost = site.costmap.cost[np.isfinite(site.costmap.cost)]
COST_ZMAX = float(np.percentile(fin_cost, 99)) if fin_cost.size else 1.0     # shared before/after colour range
anim_run = "plan" if res is None else run
if loop:
    components.html(LOOP_JS, height=0)


# =========================================================================== tabs
def tab_mission() -> None:
    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        st.markdown("##### 🚀 " + ("Planned route — press ▶ Play" if res is None else
                                   "Driven trajectory (planned path ghosted) — press ▶ Play"))
        show(get_animation(args, mode, anim_run, float(frac), float(hz_radius), "3d", "after", loop, COST_ZMAX),
             "mission3d")
        legend([("rover / driven path", F.PATH), ("remaining", "rgba(0,229,255,.4)"), ("replan", F.HAZARD),
                ("planned (ghost)", "#cfcfcf")])
    with right:
        if res is None:
            st.markdown("##### 🗺️ Route options on the orbital prior")
            fig = F.shaded_map(world, "", 560, site.belief.entropy, None, "uncertainty (nats)", hmax)
            for n, p in routes.modes.items():
                F.add_path(fig, p.path, f"{n} ({p.length_m:.0f} m, {p.success_prob:.0%})", F.ROUTE_COLORS[n],
                           "solid" if n == mode else "dash", 3.5 if n == mode else 2)
            F.add_points(fig, [t.rc for t in sci.selection.selected], "science target", F.SCIENCE, F.TARGET_SYMBOL, 13)
            F.add_start_goal(fig, world)
            show(fig, "mission2d")
            legend(MAP_LEGEND)
        else:
            st.markdown("##### 🗺️ Traverse overview")
            fig = F.shaded_map(trav.world, "", 560, res.belief.entropy, None, "uncertainty (nats)", hmax)
            F.add_rings(fig, res.meta["injected"])
            if res.plans:
                F.add_path(fig, res.plans[0].path, "planned", "#cfcfcf", "dot", 2)
            F.add_path(fig, res.trajectory, "driven", F.PATH, "solid", 3)
            reps = res.log.by_decision("replan")
            F.add_points(fig, [tuple(e.evidence["rover_rc"]) for e in reps], "replan", F.HAZARD, "diamond", 12,
                         [f"step {e.step}" for e in reps])
            F.add_points(fig, [t.rc for t in trav.targets], "science target", F.SCIENCE, F.TARGET_SYMBOL, 13)
            F.add_start_goal(fig, trav.world)
            show(fig, "mission2d")
            legend(MAP_LEGEND)
            for e in [e for e in res.log if e.decision in ("replan", "pause_and_sense", "goal_reached", "stop")][-4:]:
                st.markdown(f"`s{e.step}` **{BADGE.get(e.decision, e.decision)}** — {e.reason}")


def tab_cost() -> None:
    cm_state = "after"
    if res is not None:
        cm_state = "after" if st.radio("Map state", ["after traverse", "before traverse"], horizontal=True,
                                       key="cm_state") == "after traverse" else "before"
    cm = res.costmap if (res is not None and cm_state == "after") else site.costmap
    st.caption("Cost per metre (Inferno; black = impassable: slope > 25° or P(big_rock) > 0.6). Hover any cell for "
               "the per-component breakdown; the same colour range is used before and after the traverse.")
    st.markdown("##### 🎬 Rover on the cost map")
    show(get_animation(args, mode, anim_run, float(frac), float(hz_radius), "2d", cm_state, loop, COST_ZMAX),
         "cost_anim")
    legend([("rover / driven", F.PATH), ("remaining", "rgba(0,229,255,.4)"), ("sensor radius", "rgba(0,229,255,.55)"),
            ("replan", F.HAZARD), ("planned (ghost)", "#cfcfcf"), ("low cost", "#420a68"), ("high cost", "#fcffa4"),
            ("impassable", "#000000")])
    c2, c3 = st.columns(2, gap="medium")
    with c2:
        st.markdown("##### ⚠️ Hazard probability")
        show(F.grid_map(cm.hazard_prob, F.SEQ["hazard"], "", "hazard_prob", 0, 1, height=420), "hazard")
    with c3:
        st.markdown("##### ⛰️ Slope (from DEM)")
        show(F.grid_map(site.layers.slope_deg, F.SEQ["slope"], "", "slope (°)", 0,
                        float(np.percentile(site.layers.slope_deg, 99.5)), height=420), "slope")
    st.markdown("##### 🔎 Cell inspector")
    i1, i2, i3 = st.columns([1, 1, 4])
    r = i1.number_input("row", 0, H - 1, world.goal[0] // 2, key=f"insp_r-{src}-{size}")
    c = i2.number_input("col", 0, W - 1, world.goal[1] // 2, key=f"insp_c-{src}-{size}")
    belief_now = res.belief if (res is not None and cm is res.costmap) else site.belief
    bd = cm.breakdown((int(r), int(c)))
    bd.update({f"P({n})": float(belief_now.class_probs[int(r), int(c), k]) for k, n in enumerate(CLASS_NAMES)})
    bd.update({"entropy (nats)": float(belief_now.entropy[int(r), int(c)]),
               "slope_deg": float(site.layers.slope_deg[int(r), int(c)]),
               "P(slope > 25°)": float(site.layers.p_slope_exceed[int(r), int(c)])})
    i3.dataframe(pd.DataFrame([{k: ("inf (impassable)" if isinstance(v, float) and not math.isfinite(v) else round(v, 4))
                                for k, v in bd.items()}]), hide_index=True, width="stretch")


def tab_uncertainty() -> None:
    if res is not None:
        steps = sorted(res.entropy_snapshots)
        sk = st.select_slider("Entropy snapshot (step)", steps, value=steps[-1], key="snap")
        ent_after, after_lbl, obs = res.entropy_snapshots[sk], f"after step {sk}", res.belief.observed_mask
    else:
        sk, ent_after, after_lbl, obs = 0, site.belief.entropy, "after first look", site.belief.observed_mask
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("##### ❔ Uncertainty: orbital prior")
        show(F.grid_map(site.prior.entropy, F.SEQ["entropy"], "", "nats", 0, hmax, height=440), "ent_prior")
    with c2:
        st.markdown(f"##### ❔ Uncertainty: {after_lbl}")
        fig = F.grid_map(ent_after, F.SEQ["entropy"], "", "nats", 0, hmax, height=440)
        if res is not None:
            F.add_path(fig, res.trajectory[:sk + 1], "driven", F.PATH, "solid", 2)
        show(fig, "ent_after")
    st.caption("Same 0 – ln 4 nats colour range on both maps: the rover's navcam looks erase uncertainty along its track.")
    c3, c4, c5 = st.columns(3, gap="medium")
    with c3:
        st.markdown("##### 📐 P(slope > 25°)")
        show(F.grid_map(site.layers.p_slope_exceed, F.SEQ["slope"], "", "P", 0, 1, height=380), "pslope")
    with c4:
        st.markdown("##### 👁️ Cells observed by the navcam")
        show(F.grid_map(obs.astype(float), [[0, F.PLOT_BG], [1, F.SAFE]], "", "observed", 0, 1, height=380,
                        showscale=False), "observed")
    with c5:
        b_now = res.belief if res is not None else site.belief
        truth_now = (trav.world if trav else world).true_class
        hit_prior = site.prior.class_probs.argmax(-1) == truth_now
        hit_now = b_now.class_probs.argmax(-1) == truth_now
        seen = b_now.observed_mask
        st.markdown(f"##### 🧭 Belief: observed cells {hit_prior[seen].mean():.0%} → {hit_now[seen].mean():.0%} correct")
        show(F.class_map(b_now.class_probs.argmax(-1), "", 380), "cls_belief")
        st.caption(f"Whole map: {hit_prior.mean():.0%} → {hit_now.mean():.0%} (only {seen.mean():.0%} of cells were "
                   "observed; the rest is still the orbital prior).")
    with st.expander("Hidden ground truth (evaluation only — the rover never reads this)"):
        show(F.class_map((trav.world if trav else world).true_class, "True terrain classes", 460), "cls_truth")


def tab_routes() -> None:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(F.pareto_fig(routes.sweep, routes.front, routes.modes), "pareto")
    with c2:
        fig = F.shaded_map(world, "Route options", 460, site.belief.entropy, None, "uncertainty (nats)", hmax)
        for n, p in routes.modes.items():
            F.add_path(fig, p.path, n, F.ROUTE_COLORS[n], "solid" if n == mode else "dash")
        F.add_start_goal(fig, world)
        show(fig, "routes_map")
    tbl = pd.DataFrame(routes.table()).rename(columns={"success_prob": "P(success) belief",
                                                       "success_prob_true": "P(success) truth",
                                                       "path_entropy_mean": "mean entropy on path"})
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        "P(success) belief": st.column_config.NumberColumn(format="percent"),
        "P(success) truth": st.column_config.NumberColumn(format="percent")})
    st.caption(f"Sweep: risk_weight ∈ linspace{tuple(cfg.pareto_sweep)} on the balanced belief map. 'truth' scores the "
               "same route on the hidden ground truth — the gap is what the rover must discover by sensing.")


def tab_science() -> None:
    sel = sci.selection
    selected_ids = {t.id for t in sel.selected}
    st.caption(f"Budget {sel.budget_wh:.0f} Wh · {len(sel.selected)} selected, {len(sel.skipped)} skipped · planned "
               f"energy {sel.energy_wh:.0f} Wh. Value = {cfg.sci_w_ice:g}·ice + {cfg.sci_w_rim:g}·rim + "
               f"{cfg.sci_w_bedrock:g}·P(bedrock); ± = bootstrap std; info_gain = value × normalised entropy.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show(F.targets_bar(sci.targets, selected_ids), "sci_bar")
    with c2:
        fig = F.shaded_map(world, "Targets and selected tour", 420)
        for i, leg in enumerate([*sel.legs, *([sel.return_leg] if sel.return_leg else [])]):
            F.add_path(fig, leg.path, "selected tour", F.PATH, showlegend=i == 0)
        F.add_points(fig, [t.rc for t in sci.targets if t.id in selected_ids], "selected ★", F.SCIENCE,
                     F.TARGET_SYMBOL, 15, [f"#{t.id}" for t in sci.targets if t.id in selected_ids])
        F.add_points(fig, [t.rc for t in sci.targets if t.id not in selected_ids], "skipped", "#8a8a93",
                     F.TARGET_SYMBOL, 12, [f"#{t.id}" for t in sci.targets if t.id not in selected_ids])
        F.add_start_goal(fig, world)
        show(fig, "sci_map")
    reasons = {e.evidence.get("target_id"): e.reason for e in sci.log}
    st.dataframe(pd.DataFrame([{"id": t.id, "cell": str(t.rc), "value": round(t.value_mean, 3),
                                "± std": round(t.value_std, 3), "info_gain": round(t.info_gain, 3),
                                "utility (from start)": round(t.utility, 3) if math.isfinite(t.utility) else None,
                                "selected": t.id in selected_ids, "decision": reasons.get(t.id, t.reason)}
                               for t in sci.targets]), hide_index=True, width="stretch")


def tab_log() -> None:
    log = res.log if res is not None else sci.log
    st.caption("Traverse decisions — every row is reproducible from its evidence numbers." if res is not None else
               "Planning decisions (target selection). Run a traverse to see the rover's autonomous decisions.")
    counts = log.counts()
    all_dec = sorted(counts)
    default = [d for d in all_dec if d != "hazard_detected"] or all_dec
    lkey = f"{run}-{src}-{mode}-{seed}-{size}-{lam}-{budget}-{radius}-{conn}-{frac}-{hz_radius}"
    shown_dec = st.multiselect("Decisions", all_dec, default=default, key=f"log_filter-{lkey}",
                               format_func=lambda d: f"{d} ({counts[d]})")
    entries = [e for e in log if e.decision in shown_dec]
    df = pd.DataFrame([{"step": e.step, "event": BADGE.get(e.decision, e.decision.upper()), "decision": e.decision,
                        "confidence": float(e.confidence), "cell": str(tuple(e.evidence["cell"])), "reason": e.reason}
                       for e in entries])
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        ev = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"log_table-{lkey}-{'.'.join(shown_dec)}", height=440, width="stretch",
                          column_config={"confidence": st.column_config.ProgressColumn(
                              "confidence", help="1 − H/ln 4 at the trigger cell", min_value=0.0, max_value=1.0,
                              format="%.2f"),
                              "step": st.column_config.NumberColumn("time step", width="small"),
                              "reason": st.column_config.TextColumn("reason", width="large")}) if len(df) else None
        sel_rows = [i for i in (ev.selection.rows if ev is not None else []) if i < len(entries)]
        st.download_button("Download log (JSON)", json.dumps({"entries": log.to_records()}, indent=1),
                           "decision_log.json", "application/json", key="log_dl")
    chosen = entries[sel_rows[0]] if sel_rows else (entries[0] if entries else None)
    with c2:
        w_disp = trav.world if trav else world
        fig = F.shaded_map(w_disp, "Selected decision (map centred on its cell)" if chosen else "", 440)
        if res is not None:
            F.add_path(fig, res.trajectory, "driven", F.PATH, "solid", 2)
            F.add_rings(fig, res.meta["injected"])
        if chosen is not None:
            F.add_points(fig, [tuple(chosen.evidence["rover_rc"])], "rover at decision", "#ffffff", "triangle-up", 14)
            F.add_points(fig, [tuple(chosen.evidence["cell"])], "trigger cell", F.HAZARD, "x", 16)
        F.add_start_goal(fig, w_disp)
        win = (int(chosen.evidence["cell"][0]), int(chosen.evidence["cell"][1]), max(12, min(H, W) // 8)) \
            if chosen is not None else None
        show(fig, "log_map", window=win)
    if chosen is not None:
        st.markdown(f"**s{chosen.step} · {BADGE.get(chosen.decision, chosen.decision)}** · confidence "
                    f"{chosen.confidence:.2f} (= 1 − H/ln 4 at the trigger cell) — {chosen.reason}")
        with st.expander("Evidence (raw numbers exactly as logged)", expanded=False):
            st.json(_json_safe(chosen.evidence), expanded=True)
    st.markdown("##### 🔎 Decisions about one cell")
    l1, l2, l3 = st.columns([1, 1, 4])
    lr = l1.number_input("row", 0, H - 1, int(chosen.evidence["cell"][0]) if chosen else 0,
                         key=f"log_r-{lkey}-{id(chosen)}")
    lc = l2.number_input("col", 0, W - 1, int(chosen.evidence["cell"][1]) if chosen else 0,
                         key=f"log_c-{lkey}-{id(chosen)}")
    hits = log.entries_for_cell(int(lr), int(lc))
    l3.write(f"{len(hits)} decision(s) at ({lr}, {lc})")
    for e in hits:
        st.markdown(f"- `s{e.step}` **{BADGE.get(e.decision, e.decision)}** — {e.reason}")


def tab_perception() -> None:
    trained = sensor["source"] == "trained"
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        st.markdown("##### 👁️ Sensor model")
        show(F.confusion_fig(sensor["confusion"], ("Trained AI4MARS U-Net" if trained else "Fallback matrix")
                             + " — rows: true class · columns: observed class"), "confusion")
        st.caption("This matrix is the rover's sensor model: P(observed class | true class).")
        p = float(sensor["confusion"][2, 0])
        st.caption(f"🟧 Outlined cell — how often sand is mistaken for safe soil: **{p:.0%}**. That error rate flows "
                   "straight into every Bayesian belief update, so the rover can't simply trust a 'soil' reading.")
    with c2:
        st.markdown("##### 📊 Model quality")
        if trained:
            st.metric("Validation mIoU", f"{sensor['miou']:.3f}" if sensor["miou"] is not None else "—")
            if sensor["per_class_iou"]:
                st.dataframe(pd.DataFrame([sensor["per_class_iou"]]), hide_index=True, width="stretch")
            imet = ROOT / "outputs" / "perception" / "infer_metrics.json"
            if imet.exists():
                im = json.loads(imet.read_text())
                st.markdown(f"MC-dropout (T={im.get('mc_passes')}): entropy on correct pixels "
                            f"**{im.get('entropy_correct_mean')}** vs wrong **{im.get('entropy_wrong_mean')}** "
                            f"nats · AUROC(entropy → error) **{im.get('entropy_error_auroc')}**")
        else:
            st.warning("No trained model yet — the rover uses the fallback confusion matrix. Train on Colab:")
            st.code("!pip install -q segmentation-models-pytorch datasets\n"
                    "!python -m perception.train --cache /content/cache\n!python -m perception.infer\n"
                    "# then commit perception/confusion.npy + perception/metrics.json", language="bash")
    samples = sorted((ROOT / "perception" / "samples").glob("*.png")) or \
        sorted((ROOT / "outputs" / "perception").glob("sample_*.png"))
    if samples:
        st.markdown("##### 🖼️ MC-dropout samples (image | prediction | entropy)")
        cols = st.columns(3, gap="small")
        for i, pth in enumerate(samples[:9]):
            cols[i % 3].image(str(pth), width="stretch", caption=pth.stem)


def safe(fn) -> None:
    """Run one tab body; a bug shows as an error card instead of blanking the demo."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        st.error(f"This panel hit an error ({type(e).__name__}: {e}). The rest of the dashboard still works.")


tabs = st.tabs(["🚀 Mission", "⛰️ Cost & Hazards", "❔ Uncertainty", "🧭 Routes", "🔬 Science", "📜 Decision Log",
                "👁️ Perception"])
for tab, fn in zip(tabs, (tab_mission, tab_cost, tab_uncertainty, tab_routes, tab_science, tab_log, tab_perception)):
    with tab:
        safe(fn)
