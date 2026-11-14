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
from ui.pipeline import MODES, build_routes, build_science, build_site, make_cfg, run_onboard, run_traverse
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
  [data-testid="stMetricLabel"] p, [data-testid="stMetricDelta"] div {{white-space:normal; overflow:visible;
          text-overflow:clip;}}
  /* metric cards: one row of 6 on wide screens, 2 rows of 3 otherwise */
  .st-key-kpis [data-testid="stHorizontalBlock"] {{flex-wrap:wrap;}}
  @media (max-width: 1399px) {{
    .st-key-kpis [data-testid="stColumn"] {{flex:1 1 calc(33.333% - 1rem) !important;
          min-width:calc(33.333% - 1rem) !important;}}
  }}
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
                 inject: bool, frac: float, hz_radius: float, obstacles: tuple = ()):
    site = get_site(src, size, seed, lam, budget, radius, conn)
    return run_traverse(size, seed, site.cfg, mode, inject, frac, hz_radius, site,
                        get_science(src, size, seed, lam, budget, radius, conn), source=src, obstacles=obstacles)


@st.cache_data(show_spinner="Building rover animation…", max_entries=24)
def get_animation(args: tuple, mode: str, run: str, frac: float, hz_radius: float, view: str, cm_state: str,
                  loop: bool, cost_zmax: float, obstacles: tuple = ()):
    """Animated figure (3-D terrain or 2-D cost map) for the planned path or the driven trajectory.
    Cached on every setting that changes it (seed, size, mode, λ, budget, radius, movement, hazard, view)."""
    site, routes = get_site(*args), get_routes(*args)
    if run == "plan":
        p = routes.modes.get(mode) or routes.modes["balanced"]
        path, entries, ghost, label = p.path, (), None, f"Planned · {mode}"
        belief, cm = site.belief, site.costmap
        world = site.world
    else:
        tr = get_traverse(*args, mode, run == "inject", frac, hz_radius, obstacles)
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


@st.cache_resource(show_spinner="Onboard mode: building tiles, driving with tiered memory…", max_entries=8)
def get_onboard(src: str, size: int, seed: int, lam: float, budget: float, radius: int, conn: int, mode: str,
                injected: tuple):
    site = get_site(src, size, seed, lam, budget, radius, conn)
    return run_onboard(size, seed, site.cfg, mode, site, get_science(src, size, seed, lam, budget, radius, conn),
                       injected, source=src)


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
st.session_state.setdefault("run", None)          # None (idle) | "plan" | "traverse" | "inject" | "manual"
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
    _s = get_site(src, int(size), int(seed), float(lam), float(budget), int(radius), conn)
    Hs, Ws = _s.world.shape
    st.session_state.setdefault("obstacles", [])
    with st.expander("💥 Hidden hazard / obstacles",
                     expanded=str(st.session_state.get("placement", "")).startswith("Manual")):
        placement = st.radio("Placement", ["Auto (on planned route)", "Manual (choose cells)"], key="placement",
                             help="Auto: one boulder field on the planned route. Manual: you choose where hidden "
                                  "boulder fields / sand traps go. The rover is never told — it must discover them.")
        manual = placement.startswith("Manual")
        if not manual:
            frac = st.slider("Position along route", 0.1, 0.9, 0.4, 0.05, key="frac")
            hz_radius = st.slider("Boulder-field radius (cells)", 1, 6, 3, key="hz_radius")
        else:
            frac, hz_radius = float(st.session_state.get("frac", 0.4)), int(st.session_state.get("hz_radius", 3))
            st.selectbox("Obstacle type", ["big_rock", "sand"], key="obs_kind",
                         format_func=lambda k: "🪨 boulder field (impassable)" if k == "big_rock"
                         else "🏜️ sand trap (costly, risky)")
            st.slider("Vertical (row)", 0, Hs - 1, Hs // 2, key=f"obs_r-{Hs}")
            st.slider("Horizontal (column)", 0, Ws - 1, Ws // 2, key=f"obs_c-{Ws}")
            st.slider("Radius (cells)", 1, 8, 3, key="obs_rad")
            st.caption("Dotted amber ring on the Mission map = where it will go (press Plan to see the route).")

            def _add_obstacle() -> None:
                st.session_state.obstacles.append((int(st.session_state[f"obs_r-{Hs}"]),
                                                   int(st.session_state[f"obs_c-{Ws}"]),
                                                   int(st.session_state.obs_rad), st.session_state.obs_kind))

            o1, o2 = st.columns(2)
            o1.button("➕ Add", key="obs_add", width="stretch", on_click=_add_obstacle)
            o2.button("🗑 Clear", key="obs_clear", width="stretch", on_click=lambda: st.session_state.obstacles.clear())
            for i, (r_, c_, rad_, k_) in enumerate(st.session_state.obstacles, 1):
                st.caption(f"{i}. {'🪨 boulders' if k_ == 'big_rock' else '🏜️ sand'} at ({r_}, {c_}) r={rad_}")
            if not st.session_state.obstacles:
                st.caption("No obstacles yet — set a cell and press ➕ Add.")
    b1, b2 = st.columns(2)
    if b1.button("Plan", key="plan_btn", width="stretch"):
        st.session_state.run = "plan"
    if b2.button("Run traverse", key="run_btn", type="primary", width="stretch"):
        st.session_state.run = "traverse"
    if not manual:
        if st.button("💥 Inject hazard on path", key="inject_btn", width="stretch",
                     help="Plant a hidden boulder field on the planned route, then drive. The rover is not told."):
            st.session_state.run = "inject"
    elif st.button(f"💥 Drive with my obstacles ({len(st.session_state.obstacles)})", key="manual_btn",
                   width="stretch", disabled=not st.session_state.obstacles,
                   help="Plant your obstacles as hidden ground truth, then drive. The rover is not told."):
        st.session_state.run = "manual"
    onboard_on = st.toggle("🧠 Onboard memory mode (colibri-style tiering)", key="onboard",
                           help="Run the same traverse with int16/uint8 terrain in 64x64 tiles (HOT/WARM/COLD tiers, "
                                "path-lookahead prefetch) and hierarchical A*. Needs a traverse: a summary "
                                "appears under the metric cards, details in the 🧠 Onboard tab.")
    loop = st.toggle("🔁 Loop animation", key="loop", help="Off: plays once. On: frames repeat and the player "
                                                          "restarts automatically when it reaches the end.")
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
OBST = tuple(st.session_state.obstacles) if run == "manual" else ()      # user obstacles used by this drive
if run is not None:
    sci, routes = get_science(*args), get_routes(*args)
    plan = routes.modes.get(mode) or routes.modes["balanced"]
    if run in ("traverse", "inject", "manual"):
        trav = get_traverse(*args, mode, run == "inject", float(frac), float(hz_radius), OBST)
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
            + (chip("memory", "onboard tiers", "info") if onboard_on else "")
            + (chip("hazard", trav.note, "warn") if run in ("inject", "manual") and trav is not None else ""),
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
cards = st.container(key="kpis").columns(6, gap="small")


def card(col, label: str, value: str, delta: str | None = None, delta_color: str = "normal", help_: str | None = None,
         badge: tuple[str, str] | None = None) -> None:
    with col.container(border=True):
        st.metric(label, value, delta, delta_color=delta_color, help=help_)
        if badge:
            st.markdown(chip("", badge[0], badge[1]), unsafe_allow_html=True)


card(cards[0], "Path length", f"{length:.0f} m",
     f"{length - shortest.length_m:+.0f} m" if shortest and shortest.found else None, "inverse",
     help_="Driven (or planned) distance; delta is vs the shortest route")
card(cards[1], "Energy", f"{energy:.0f} Wh", f"{energy / budget_wh:.0%} budget", "off",
     help_=f"Energy used; delta is the share of the {budget_wh:.0f} Wh energy budget")
s_kind, s_txt = (("ok", "≥ 95 %: go") if succ >= 0.95 else ("warn", "80–95 %: caution") if succ >= 0.80
                 else ("bad", "< 80 %: risky"))
card(cards[2], "Success", f"{succ:.1%}",
     f"{(succ - shortest.success_prob) * 100:+.1f} pts" if shortest and shortest.found else None,
     help_="Success probability = Π(1 − hazard_prob) along the planned route, under the rover's belief; "
           "delta is in percentage points vs the shortest route", badge=(s_txt, s_kind))
card(cards[3], "Replans", str(res.replans) if res is not None else "—",
     f"{res.meta['extra_looks']} looks" if res is not None else None, "off",
     help_="Number of replans; delta is the extra navcam looks taken before stepping")
card(cards[4], "Targets", f"{visited} / {n_targets}" if n_targets else "—",
     "sci mode" if mode == "science" else "sci only", "off",
     help_="Science targets visited / selected (counted in science mode only)")
card(cards[5], "Nodes", f"{expanded:,}",
     (f"{len(res.plans)} plan" + ("s" if len(res.plans) != 1 else "")) if res is not None else "A* search", "off",
     help_="A* expanded nodes summed over all plans; delta is the number of plans computed")
if res is not None:
    rk = res.meta["entered_true_hazard_cells"]
    st.markdown(chip("goal", "reached ✅" if res.reached_goal else "not reached ⛔", "ok" if res.reached_goal else "bad")
                + chip("driven", f"{res.steps} steps") + chip("boulder cells entered (ground-truth check)", str(rk),
                                                            "ok" if rk == 0 else "bad"), unsafe_allow_html=True)


def onboard_strip() -> None:
    """Main-screen summary of onboard memory mode (details in the 🧠 Onboard tab)."""
    if res is None:
        st.info("🧠 Onboard memory mode is on — press **Run traverse** (or 💥 Inject hazard) to drive the same site "
                "with tiered int16/uint8 memory and compare it against the full float32 rover here.")
        return
    inj = tuple((i["rc"][0], i["rc"][1], i["radius"], i.get("kind", "big_rock")) for i in res.meta["injected"])
    ob = get_onboard(*args, mode, inj)
    ores, s_ = ob.result, ob.store_stats
    st.markdown("##### 🧠 Onboard memory mode — same traverse, tiered compact memory (details: 🧠 Onboard tab)")
    oc = st.columns(4, gap="small")
    d_len = (ores.distance_m - res.distance_m) / max(res.distance_m, 1e-9)
    ork = ores.meta["entered_true_hazard_cells"]
    for col, (lab, val, dlt, hlp) in zip(oc, [
            ("State RAM", _mb(ob.resident_onboard_bytes), f"{ob.resident_full_bytes / ob.resident_onboard_bytes:.1f}× less",
             f"Rover state RAM: onboard vs FULL float32 ({_mb(ob.resident_full_bytes)})"),
            ("HOT hits", f"{s_['hit_rate']:.1%}", f"{s_['stalls']} stalls",
             f"Tile reads served from decoded RAM ({s_['requests']} reads); stalls = synchronous flash loads"),
            ("Path Δ", f"{ores.distance_m:.0f} m", f"{d_len:+.1%}",
             f"Onboard driven distance; delta vs the FULL rover ({res.distance_m:.0f} m)"),
            ("Boulders", str(ork), "goal ✅" if ores.reached_goal else "no goal ⛔",
             "Ground-truth boulder cells entered by the onboard rover; and whether it reached the goal")]):
        with col.container(border=True):
            st.metric(lab, val, dlt, delta_color="off", help=hlp)

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
        show(get_animation(args, mode, anim_run, float(frac), float(hz_radius), "3d", "after", loop, COST_ZMAX, OBST),
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
            if manual:
                F.add_rings(fig, [{"rc": [r_, c_], "radius": rad_, "kind": k_}
                                  for r_, c_, rad_, k_ in st.session_state.obstacles], "your obstacles", dash="dash")
                F.add_rings(fig, [{"rc": [int(st.session_state[f"obs_r-{Hs}"]), int(st.session_state[f"obs_c-{Ws}"])],
                                   "radius": int(st.session_state.obs_rad), "kind": st.session_state.obs_kind}],
                            "placing (not added yet)", color=F.CAUTION, dash="dot")
            F.add_start_goal(fig, world)
            show(fig, "mission2d")
            legend(MAP_LEGEND + ([("your obstacle (boulders)", F.HAZARD), ("your obstacle (sand)", F.CAUTION)]
                                 if manual else []))
            if manual:
                st.caption(f"{len(st.session_state.obstacles)} obstacle(s) placed. Move the sliders to position the "
                           "dotted ring, press ➕ Add, then **💥 Drive with my obstacles**.")
        else:
            st.markdown("##### 🗺️ Traverse overview")
            fig = F.shaded_map(trav.world, "", 560, res.belief.entropy, None, "uncertainty (nats)", hmax)
            F.add_rings(fig, res.meta["injected"])
            if res.plans:
                F.add_path(fig, res.plans[0].path, "planned", "#cfcfcf", "dot", 2)
            F.add_path(fig, res.trajectory, "driven", F.PATH, "solid", 3)
            reps = res.log.by_decision("replan")
            F.add_labeled_points(fig, [tuple(e.evidence["rover_rc"]) for e in reps],
                                 [f"R{i}" for i in range(1, len(reps) + 1)], "replan", F.HAZARD, "diamond", 10,
                                 [f"R{i} · step {e.step} · {e.reason}" for i, e in enumerate(reps, 1)])
            F.add_points(fig, [t.rc for t in trav.targets], "science target", F.SCIENCE, F.TARGET_SYMBOL, 13)
            F.add_start_goal(fig, trav.world)
            show(fig, "mission2d")
            legend(MAP_LEGEND)
            if reps:
                st.caption(f"{len(reps)} replans (R1…R{len(reps)}). Open the **🔀 Replans** tab to compare each "
                           "route before vs after, zoomed on the trigger.")
            for e in [e for e in res.log if e.decision in ("replan", "pause_and_sense", "goal_reached", "stop")][-4:]:
                st.markdown(f"`s{e.step}` **{BADGE.get(e.decision, e.decision)}** — {e.reason}")


def _old_remaining(old_path: list, rover_rc: tuple) -> list:
    """Part of the previous plan still ahead of the rover when the replan happened."""
    return old_path[old_path.index(rover_rc):] if rover_rc in old_path else old_path


def _diff_window(paths: list, extra: list, margin: int = 8) -> tuple[int, int, int]:
    """(row, col, half) square window around the cells where ``paths`` disagree (plus ``extra`` cells)."""
    sets = [set(map(tuple, pth)) for pth in paths if len(pth)]
    diff = (set.union(*sets) - set.intersection(*sets)) if len(sets) > 1 else set()
    pts = np.asarray(list(diff) + [tuple(c) for c in extra] or [(H // 2, W // 2)])
    r0, r1, c0, c1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    half = int(max(r1 - r0, c1 - c0) // 2 + margin)
    return int((r0 + r1) // 2), int((c0 + c1) // 2), max(half, 10)


def tab_replans() -> None:
    if res is None:
        st.info("Run a traverse (▶ Run traverse, 💥 Inject hazard on path, or 💥 Drive with my obstacles) to compare "
                "how the route changed at every replan.")
        return
    reps, plans = res.log.by_decision("replan"), res.plans
    w_disp = trav.world
    h1, h2 = st.columns([3, 1])
    h1.markdown("##### 🗂️ Plan history — every adopted route, oldest grey → newest cyan")
    zoom_diff = h2.toggle("🔍 Zoom to the replan area", value=True, key="hist_zoom")
    fig = F.shaded_map(w_disp, "", 520)
    F.add_rings(fig, res.meta["injected"], "hidden obstacle")
    cols = F.gradient(len(plans))
    for k, (pl, col) in enumerate(zip(plans, cols)):
        name = "plan 0 (initial)" if k == 0 else f"plan {k} (after R{k}, step {pl.meta.get('step', '?')})"
        F.add_path(fig, pl.path, name, col, "solid" if k == len(plans) - 1 else "dash", 3 if k == len(plans) - 1 else 1.8)
    F.add_path(fig, res.trajectory, "driven", "#ffffff", "solid", 1.2)
    F.add_labeled_points(fig, [tuple(e.evidence["cell"]) for e in reps], [f"R{i}" for i in range(1, len(reps) + 1)],
                         "replan trigger cell", F.HAZARD, "x", 10,
                         [f"R{i} · step {e.step} · {e.reason}" for i, e in enumerate(reps, 1)])
    F.add_start_goal(fig, w_disp)
    focus = [tuple(e.evidence["cell"]) for e in reps] + [tuple(e.evidence["rover_rc"]) for e in reps]
    focus += [tuple(i["rc"]) for i in res.meta["injected"]]
    show(fig, "plan_history", window=_diff_window([], focus, margin=12) if (zoom_diff and reps) else None)
    legend([("initial plan", cols[0]), ("latest plan", cols[-1]), ("driven", "#ffffff"), ("trigger cell (R#)", F.HAZARD),
            ("hidden boulders", F.HAZARD), ("hidden sand", F.CAUTION)])
    st.caption("Click legend entries to hide/show individual plans.")
    if not reps:
        st.success("No replans — the first plan held all the way to the goal.")
        return

    st.markdown("##### 🔀 Before vs after one replan")
    k = st.select_slider("Replan", options=list(range(1, len(reps) + 1)), value=1, key=f"replan_pick-{len(reps)}",
                         format_func=lambda i: f"R{i} · step {reps[i - 1].step}")
    e = reps[k - 1]
    ev = e.evidence
    rover_rc, trig = tuple(ev["rover_rc"]), tuple(ev["cell"])
    old_rem = _old_remaining(plans[k - 1].path, rover_rc)
    new_path = plans[k].path
    detected = [tuple(x.evidence["cell"]) for x in res.log.by_decision("hazard_detected") if x.step <= e.step]
    driven_so_far = res.trajectory[:e.step + 1]
    near = [c for c in set(old_rem) ^ set(new_path) if max(abs(c[0] - trig[0]), abs(c[1] - trig[1])) <= 14]
    win = _diff_window([], near + [trig, rover_rc], margin=5)

    def panel(title: str, key: str, before: bool) -> None:
        f = F.shaded_map(w_disp, "", 440)
        F.add_rings(f, res.meta["injected"], "hidden obstacle")
        F.add_path(f, driven_so_far, "driven so far", "#ffffff", "solid", 2)
        if before:
            F.add_path(f, old_rem, f"old route (plan {k - 1})", F.CAUTION, "solid", 3)
        else:
            F.add_path(f, old_rem, f"old route (plan {k - 1})", F.CAUTION, "dot", 1.5)
            F.add_path(f, new_path, f"new route (plan {k})", F.PATH, "solid", 3)
        F.add_points(f, detected, "boulder detected so far", F.HAZARD, "square", 7)
        F.add_points(f, [trig], "trigger cell", F.HAZARD, "x", 16)
        F.add_points(f, [rover_rc], "rover", "#ffffff", "triangle-up", 14)
        st.markdown(f"**{title}**")
        show(f, key, window=win)

    old_set, new_set = set(old_rem), set(new_path)
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        panel(f"① Before R{k}: route being followed", "replan_before", True)
    with c2:
        panel(f"② After R{k}: new route", "replan_after", False)
    with c3:
        f = F.shaded_map(w_disp, "", 440)
        F.add_rings(f, res.meta["injected"], "hidden obstacle")
        F.add_path(f, [c for c in old_rem if c in new_set], "unchanged", "#9aa0aa", "solid", 1.5)
        F.add_points(f, sorted(old_set - new_set), "dropped (old only)", F.CAUTION, "square", 8)
        F.add_points(f, sorted(new_set - old_set), "added (new only)", F.PATH, "square", 8)
        F.add_points(f, [trig], "trigger cell", F.HAZARD, "x", 16)
        F.add_points(f, [rover_rc], "rover", "#ffffff", "triangle-up", 14)
        st.markdown(f"**③ Difference only: {len(old_set - new_set)} dropped · {len(new_set - old_set)} added**")
        show(f, "replan_diff", window=win)
    legend([("old route / dropped cells", F.CAUTION), ("new route / added cells", F.PATH), ("driven so far", "#ffffff"),
            ("trigger / detected boulders", F.HAZARD), ("unchanged", "#9aa0aa")])

    dropped, added = len(old_set - new_set), len(new_set - old_set)
    st.markdown(f"`s{e.step}` **🔴 REPLAN R{k}** · confidence {e.confidence:.2f} — {e.reason}")
    rows = [("trigger rule", f"{ev['threshold_name']} (threshold {ev['threshold']:g})", ""),
            ("P(big_rock) at trigger", f"{ev.get('p_big_rock', float('nan')):.2f}", ""),
            ("hazard_prob at trigger", f"{ev.get('hazard_prob', float('nan')):.2f}", ""),
            ("route length ahead (m)", f"{ev['old_length_m']:.0f}", f"{ev['new_length_m']:.0f}"),
            ("P(success) of route ahead", f"{ev['old_success_prob']:.1%}", f"{ev['new_success_prob']:.1%}"),
            ("cells dropped / added", f"{dropped} dropped", f"{added} added"),
            ("backed up along own track", "", f"{ev.get('retreat_cells', 0)} cell(s)")]
    st.dataframe(pd.DataFrame(rows, columns=["", "before (old route)", "after (new route)"]), hide_index=True,
                 width="stretch")
    with st.expander("Evidence (raw numbers exactly as logged)"):
        st.json(_json_safe(ev), expanded=True)


def tab_cost() -> None:
    cm_state = "after"
    if res is not None:
        cm_state = "after" if st.radio("Map state", ["after traverse", "before traverse"], horizontal=True,
                                       key="cm_state") == "after traverse" else "before"
    cm = res.costmap if (res is not None and cm_state == "after") else site.costmap
    st.caption("Cost per metre (Inferno; black = impassable: slope > 25° or P(big_rock) > 0.6). Hover any cell for "
               "the per-component breakdown; the same colour range is used before and after the traverse.")
    st.markdown("##### 🎬 Rover on the cost map")
    show(get_animation(args, mode, anim_run, float(frac), float(hz_radius), "2d", cm_state, loop, COST_ZMAX, OBST),
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
    lkey = f"{run}-{src}-{mode}-{seed}-{size}-{lam}-{budget}-{radius}-{conn}-{frac}-{hz_radius}-{hash(OBST)}"
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


def _mb(b: float) -> str:
    return f"{b / 1e6:.2f} MB" if b >= 1e5 else f"{b / 1e3:.1f} KB"


def tab_onboard() -> None:
    st.caption("Tiering idea adapted from colibri (MoE expert offloading) → terrain tiles.")
    if not onboard_on:
        st.info("Turn on **🧠 Onboard memory mode** in the sidebar to run the same traverse with compact int16/uint8 "
                "terrain in tiered 64×64 tiles (HOT / WARM / COLD) and hierarchical planning.")
        return
    if res is None:
        st.info("Run a traverse (▶ Run traverse, 💥 Inject hazard, or 💥 Drive with my obstacles): onboard mode then "
                "drives the same site with the same hidden hazards and is compared against it here.")
        return
    injected = tuple((i["rc"][0], i["rc"][1], i["radius"], i.get("kind", "big_rock")) for i in res.meta["injected"])
    ob = get_onboard(*args, mode, injected)
    ores, s_ = ob.result, ob.store_stats
    err = ob.error
    # ---- precision is never changed silently: error report always first
    with st.container(border=True):
        st.markdown(f"**📏 Quantisation error** — max elevation error **{err['max_elev_err_m'] * 100:.1f} cm** "
                    f"(int16 at {err['scale_m'] * 100:.0f} cm{', auto-widened' if err['auto_widened'] else ''}) ≈ "
                    f"**{err['noise_to_error_ratio']:.0f}× below** the {err['dem_noise_sigma_m']:.1f} m DEM noise · "
                    f"max slope error **{err['max_slope_err_deg_uint8_storage']:.2f}°** as stored "
                    f"({err['max_slope_err_deg_from_int16_dem']:.2f}° if re-derived from the int16 DEM) · hazard "
                    "probabilities on a log scale (±2.8 % relative), so tiny per-cell risks are not rounded to zero.")
    c = st.columns(4, gap="small")
    for col, (lab, val, sub) in zip(c, [("HOT hit rate", f"{s_['hit_rate']:.1%}", f"{s_['requests']} tile reads"),
                                        ("Stalls", str(s_["stalls"]), "synchronous loads"),
                                        ("Prefetch hits", f"{s_['prefetch_hits']} / {s_['prefetch_loaded']}",
                                         f"{s_['prefetch_hit_rate']:.0%} used"),
                                        ("Evictions", f"{s_['evictions_hot']} HOT · {s_['evictions_warm']} WARM",
                                         f"{s_['tiles']} tiles of {ob.tile}×{ob.tile}")]):
        with col.container(border=True):
            st.metric(lab, val, sub, delta_color="off")

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        st.markdown("##### 🧱 Tile tiers along the traverse")
        k = st.slider("Step", 0, len(ob.tier_frames) - 1, min(len(ob.tier_frames) - 1, ores.steps // 3),
                      key=f"tier_step-{len(ob.tier_frames)}")
        plan_k = [p for p in ores.plans if p.meta.get("step", 0) <= k][-1] if ores.plans else None
        rov = ores.trajectory[min(k, len(ores.trajectory) - 1)]
        ahead = plan_k.path[plan_k.path.index(rov):] if (plan_k and rov in plan_k.path) else (plan_k.path if plan_k else [])
        show(F.tier_map(ob.world, ob.tier_frames[k], ob.tile, rov, ahead, ores.trajectory[:k + 1], 520), "tier_map")
        legend([("HOT (decoded RAM)", "#FF6B35"), ("WARM (compressed RAM)", "#F4A261"), ("COLD (flash)", "#8c8c96"),
                ("resident coarse map", F.SAFE), ("rover / planned path", F.PATH)])
    with right:
        st.markdown("##### 🛣️ FULL vs ONBOARD paths")
        fig = F.shaded_map(ob.world, "", 520)
        F.add_rings(fig, ores.meta["injected"], "hidden obstacle")
        F.add_path(fig, res.trajectory, f"FULL ({res.distance_m:.0f} m)", "#ffffff", "solid", 2.5)
        F.add_path(fig, ores.trajectory, f"ONBOARD ({ores.distance_m:.0f} m)", F.PATH, "dash", 2.5)
        F.add_start_goal(fig, ob.world)
        show(fig, "onboard_paths")
        cmp_rows = [("reached goal", res.reached_goal, ores.reached_goal),
                    ("driven (m)", round(res.distance_m, 1), round(ores.distance_m, 1)),
                    ("replans", res.replans, ores.replans),
                    ("initial-plan P(success)", f"{res.plans[0].success_prob:.1%}", f"{ores.plans[0].success_prob:.1%}"),
                    ("boulder cells entered", res.meta["entered_true_hazard_cells"],
                     ores.meta["entered_true_hazard_cells"]),
                    ("A* nodes expanded", sum(p.expanded_nodes for p in res.plans),
                     sum(p.expanded_nodes for p in ores.plans))]
        st.dataframe(pd.DataFrame(cmp_rows, columns=["", "FULL", "ONBOARD"]).astype(str), hide_index=True,
                     width="stretch")

    st.markdown("##### 🧮 Memory: full float32 vs onboard compact")
    rows = [{"layer": n, "full": _mb(f), "onboard": _mb(o), "reduction": f"{f / o:.0f}×" if o else "not stored",
             "format": how} for n, f, o, how in ob.layers]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    t1, t2, t3 = st.columns(3, gap="small")
    with t1.container(border=True):
        st.metric("Rover state RAM", _mb(ob.resident_onboard_bytes),
                  f"{ob.resident_full_bytes / ob.resident_onboard_bytes:.1f}× less than FULL ({_mb(ob.resident_full_bytes)})",
                  delta_color="off", help="Belief stays full-size in both modes (it is updated every step).")
    with t2.container(border=True):
        st.metric("Tiers now", f"HOT {_mb(s_['hot_bytes'])} · WARM {_mb(s_['warm_bytes'])}",
                  f"resident coarse {_mb(s_['resident_coarse_bytes'])}", delta_color="off")
    with t3.container(border=True):
        st.metric("COLD flash (storage)", _mb(s_["flash_bytes"]), f"{s_['flash_reads']} tile reads", delta_color="off")

    bpath = ROOT / "onboard" / "bench.json"
    if bpath.exists():
        st.markdown("##### 📊 Benchmark (`python -m onboard.bench`, tracemalloc peak during boot + traverse)")
        cases = json.loads(bpath.read_text()).get("cases", [])
        st.dataframe(pd.DataFrame([{
            "site": f"{c['site']} ({c['shape'][0]}×{c['shape'][1]})",
            "peak RAM full": _mb(c["ram"]["full_peak_bytes"]), "peak RAM onboard": _mb(c["ram"]["onboard_peak_bytes"]),
            "peak ×": c["ram"]["peak_reduction"], "state ×": c["ram"]["resident_reduction"],
            "HOT hit": f"{c['cache']['hit_rate']:.1%}", "stalls": c["cache"]["stalls"],
            "prefetch hit": f"{c['cache']['prefetch_hit_rate']:.0%}",
            "replans F/O": f"{c['full']['replans']}/{c['onboard']['replans']}",
            "driven Δ": f"{c['diff']['driven_length_pct']:+.1f}%",
            "P(success) Δ": f"{c['diff']['initial_success_prob_pts']:+.1f} pts",
            "wall F/O (s)": f"{c['full']['wall_s']}/{c['onboard']['wall_s']}",
            "goal F/O": f"{c['full']['reached_goal']}/{c['onboard']['reached_goal']}"} for c in cases]),
            hide_index=True, width="stretch")
        st.caption("Peak RAM via tracemalloc while booting the rover and driving (dashboard replay telemetry excluded "
                   "in both modes; tile flash = storage). Wall times include tracemalloc overhead.")


def safe(fn) -> None:
    """Run one tab body; a bug shows as an error card instead of blanking the demo."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        st.error(f"This panel hit an error ({type(e).__name__}: {e}). The rest of the dashboard still works.")


if onboard_on:
    safe(onboard_strip)


tabs = st.tabs(["🚀 Mission", "🔀 Replans", "⛰️ Cost & Hazards", "❔ Uncertainty", "🧭 Routes", "🔬 Science", "📜 Decision Log",
                "🧠 Onboard", "👁️ Perception"])
for tab, fn in zip(tabs, (tab_mission, tab_replans, tab_cost, tab_uncertainty, tab_routes, tab_science, tab_log,
                          tab_onboard, tab_perception)):
    with tab:
        safe(fn)
