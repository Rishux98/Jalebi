"""ODYSSEY — MARS-09 autonomous rover dashboard (Streamlit).

Run:  python -m streamlit run app.py
All heavy work lives in ui/pipeline.py (cached per sidebar setting); charts in ui/figures.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.config import DEFAULT
from core.types import CLASS_NAMES, N_CLASSES
from ui import figures as F
from ui.pipeline import MODES, build_routes, build_science, build_site, make_cfg, run_traverse
from world.real_dem import default_tile_available

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ODYSSEY · MARS-09", page_icon="🛰️", layout="wide")


# =========================================================================== cached pipeline
@st.cache_resource(show_spinner="Building site: DEM, terrain layers, orbital prior…", max_entries=8)
def get_site(src: str, size: int, seed: int, lam: float, budget: float, radius: int):
    """``budget`` is a multiple of the direct balanced route's energy (works for any site size)."""
    return build_site(size, seed, make_cfg(lam, DEFAULT.sci_energy_budget_wh, radius), source=src,
                      budget_factor=budget)


@st.cache_resource(show_spinner="Ranking science targets…", max_entries=8)
def get_science(src: str, size: int, seed: int, lam: float, budget: float, radius: int):
    return build_science(get_site(src, size, seed, lam, budget, radius))


@st.cache_resource(show_spinner="Planning route options + risk sweep…", max_entries=8)
def get_routes(src: str, size: int, seed: int, lam: float, budget: float, radius: int):
    return build_routes(get_site(src, size, seed, lam, budget, radius),
                        get_science(src, size, seed, lam, budget, radius))


@st.cache_resource(show_spinner="Driving the rover…", max_entries=16)
def get_traverse(src: str, size: int, seed: int, lam: float, budget: float, radius: int, mode: str, inject: bool,
                 frac: float, hz_radius: float):
    site = get_site(src, size, seed, lam, budget, radius)
    return run_traverse(size, seed, site.cfg, mode, inject, frac, hz_radius, site,
                        get_science(src, size, seed, lam, budget, radius), source=src)


def _json_safe(d: dict) -> dict:
    return {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


# =========================================================================== sidebar
st.session_state.setdefault("run", None)          # None | "traverse" | "inject"
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
    lam = st.slider("λ uncertainty weight", 0.0, 5.0, float(DEFAULT.lambda_unc), 0.25, key="lam",
                    help="cost weight of belief entropy (avoid unknown terrain)")
    budget = st.slider("Energy budget (× direct route)", 1.0, 3.0, 1.45, 0.05, key="budget",
                       help="Budget = this × energy of the balanced direct start→goal route; the extra pays for "
                            "science detours. 1.45× ≈ 800 Wh on the 200×200 synthetic site.")
    radius = st.slider("Sensor radius (cells)", 3, 12, int(DEFAULT.sensor_radius), key="radius")
    with st.expander("Hidden hazard"):
        frac = st.slider("Position along route", 0.1, 0.9, 0.4, 0.05, key="frac")
        hz_radius = st.slider("Boulder-field radius (cells)", 1, 6, 3, key="hz_radius")
    b1, b2 = st.columns(2)
    if b1.button("Plan", key="plan_btn", width="stretch"):
        st.session_state.run = None
    if b2.button("Run traverse", key="run_btn", type="primary", width="stretch"):
        st.session_state.run = "traverse"
    if st.button("💥 Inject hazard on path", key="inject_btn", width="stretch",
                 help="Plant a hidden boulder field on the planned route, then drive. The rover is not told."):
        st.session_state.run = "inject"
    _s = get_site(src, int(size), int(seed), float(lam), float(budget), int(radius))
    st.caption(f"Energy budget = {budget:.2f} × {_s.direct_energy_wh:.0f} Wh (direct route) = "
               f"**{_s.cfg.sci_energy_budget_wh:.0f} Wh**")
    st.caption("Sensor model: " + ("trained AI4MARS confusion matrix" if (ROOT / DEFAULT.confusion_path).exists()
                                   else "fallback confusion matrix (train on Colab for the real one)"))

args = (src, int(size), int(seed), float(lam), float(budget), int(radius))
site = get_site(*args)
sci = get_science(*args)
routes = get_routes(*args)
world, cfg = site.world, site.cfg
trav = None
if st.session_state.run is not None:
    trav = get_traverse(*args, mode, st.session_state.run == "inject", float(frac), float(hz_radius))
res = trav.result if trav else None
plan = routes.modes.get(mode) or routes.modes["balanced"]

# =========================================================================== header
st.markdown("## ODYSSEY — autonomous traverse planning under uncertainty")
H, W = world.shape
site_txt = (f"{world.name} · {H}×{W} cells @ {world.cell_size:.2f} m ({H * world.cell_size:.0f}×"
            f"{W * world.cell_size:.0f} m) · seed {world.seed}")
if res is None:
    st.caption(f"{site_txt} · start {world.start} → goal {world.goal} · "
               f"mode **{mode}** · planning view (press **Run traverse** or **Inject hazard on path**)")
else:
    st.caption(f"{site_txt} · mode **{res.meta['mode']}** · "
               f"{'with injected hazard — ' + trav.note if st.session_state.run == 'inject' else 'no injected hazard'}")

m = st.columns(6)
if res is None:
    m[0].metric("Route length", f"{plan.length_m:.0f} m")
    m[1].metric("Energy", f"{plan.energy_wh:.0f} Wh", f"{plan.energy_wh - cfg.sci_energy_budget_wh:+.0f} vs budget", delta_color="inverse")
    m[2].metric("P(success) belief", f"{plan.success_prob:.1%}")
    m[3].metric("P(success) truth", f"{plan.meta.get('success_prob_true', float('nan')):.1%}",
                help="Evaluation only: the same route scored on the hidden ground truth.")
    m[4].metric("Mean entropy on route", f"{plan.meta.get('path_entropy_mean', float('nan')):.2f} nats")
    m[5].metric("Nodes expanded", f"{plan.expanded_nodes:,}")
else:
    m[0].metric("Goal", "reached ✅" if res.reached_goal else "stopped ⛔")
    m[1].metric("Driven", f"{res.distance_m:.0f} m", f"{res.steps} steps", delta_color="off")
    m[2].metric("Energy", f"{res.energy_wh:.0f} Wh", f"{res.energy_wh - cfg.sci_energy_budget_wh:+.0f} vs budget", delta_color="inverse")
    m[3].metric("Replans", res.replans)
    m[4].metric("P(success) initial plan", f"{res.plans[0].success_prob:.1%}" if res.plans else "—")
    m[5].metric("Boulder cells entered", res.meta["entered_true_hazard_cells"],
                help="Ground-truth check (the rover never sees this number).")

tabs = st.tabs(["🚀 Mission", "⛰️ Cost & Hazards", "❔ Uncertainty", "🧭 Routes", "🔬 Science", "📜 Decision Log",
                "👁️ Perception"])
selected_ids = {t.id for t in sci.selection.selected}

# =========================================================================== 1. Mission
with tabs[0]:
    left, right = st.columns([1.15, 1])
    with left:
        if res is None:
            paths = [(n, p.path, F.ROUTE_COLORS[n], "solid" if n == mode else "dash") for n, p in routes.modes.items()
                     if p.found]
            pts = [("targets", [t.rc for t in sci.targets], F.TARGET, "diamond")]
        else:
            paths = [("initial plan", res.plans[0].path if res.plans else [], F.TEXT_2, "dash"),
                     ("driven", res.trajectory, F.ROUTE_COLORS.get(res.meta["mode"], F.TRACK), "solid")]
            inj = [tuple(i["rc"]) for i in res.meta["injected"]]
            pts = [("replans", [tuple(e.evidence["rover_rc"]) for e in res.log.by_decision("replan")], F.REPLAN, "x"),
                   ("injected hazard", inj, F.HAZARD, "circle-open"),
                   ("targets", [t.rc for t in trav.targets], F.TARGET, "diamond")]
        pts += [("start", [world.start], "#1baf7a", "circle"), ("goal", [world.goal], "#e87ba4", "diamond-open")]
        st.plotly_chart(F.terrain_3d(world, paths, pts), key="mission3d")
    with right:
        if res is None:
            fig = F.shaded_map(world, "Route options on the orbital prior (entropy overlay)", 560,
                               site.belief.entropy, "Purples", "entropy (nats)", float(np.log(N_CLASSES)))
            for n, p in routes.modes.items():
                F.add_path(fig, p.path, f"{n} ({p.length_m:.0f} m, {p.success_prob:.0%})", F.ROUTE_COLORS[n],
                           "solid" if n == mode else "dash", 3 if n == mode else 2)
            F.add_start_goal(fig, world)
            st.plotly_chart(fig, key="mission2d")
            st.info("This is the plan from orbit + one navcam look. Press **Inject hazard on path** to plant a hidden "
                    "boulder field on the route and watch the rover detect it and replan.")
        else:
            k = st.slider("Replay step", 0, res.steps, res.steps, key="replay")
            fr = res.frames[min(k, len(res.frames) - 1)]
            snap = max(s for s in res.entropy_snapshots if s <= k) if any(s <= k for s in res.entropy_snapshots) else 0
            fig = F.shaded_map(trav.world, f"Step {k}: belief entropy (snapshot @ step {snap})", 560,
                               res.entropy_snapshots[snap], "Purples", "entropy (nats)", float(np.log(N_CLASSES)))
            F.add_rings(fig, res.meta["injected"])
            F.add_path(fig, fr["path"], "current plan", F.ROUTE_COLORS.get(res.meta["mode"], F.TRACK), "dash")
            F.add_path(fig, res.trajectory[:k + 1], "driven", F.TRACK, "solid", 3)
            reps = [e for e in res.log.by_decision("replan") if e.step <= k]
            F.add_points(fig, [tuple(e.evidence["rover_rc"]) for e in reps], "replan", F.REPLAN, "x", 12,
                         [f"step {e.step}" for e in reps])
            F.add_points(fig, [t.rc for t in trav.targets], "targets", F.TARGET, "diamond", 11)
            F.add_start_goal(fig, trav.world)
            F.add_points(fig, [fr["rover_rc"]], "rover", "#ffffff", "triangle-up", 15)
            st.plotly_chart(fig, key="replay_map")
            recent = [e for e in res.log if e.step <= k and e.decision != "hazard_detected"][-3:]
            st.markdown(f"**Energy** {fr['energy_wh']:.0f} Wh · **distance** {fr['distance_m']:.0f} m · "
                        f"**cells observed** {fr['n_observed']:,} · **targets left** {fr['n_targets_left']}")
            for e in reversed(recent):
                st.markdown(f"`s{e.step}` **{e.decision}** — {e.reason}")

# =========================================================================== 2. Cost & Hazards
with tabs[1]:
    cm = site.costmap
    title_state = "before traverse (orbital prior + first look)"
    if res is not None and st.radio("Map state", ["after traverse", "before traverse"], horizontal=True,
                                    key="cm_state") == "after traverse":
        cm, title_state = res.costmap, "after traverse (rover's updated belief)"
    st.caption(f"Cost map {title_state}. Hover the cost map for the per-component breakdown. "
               "Black = impassable (slope > 25° or P(big_rock) > 0.6).")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = F.cost_map(cm, "Cost (hover: breakdown)", 460)
        F.add_path(fig, (res.trajectory if res else plan.path), "route", F.ROUTE_COLORS.get(mode, F.TRACK))
        st.plotly_chart(fig, key="costmap")
    with c2:
        hz = cm.hazard_prob
        st.plotly_chart(F.grid_map(hz, F.SEQ["hazard"], "Hazard probability per cell", "hazard_prob", 0,
                                   max(float(hz.max()), 1e-3), height=460), key="hazard")
    with c3:
        st.plotly_chart(F.grid_map(site.layers.slope_deg, F.SEQ["slope"], "Slope (from DEM)", "slope (°)", 0,
                                   float(np.percentile(site.layers.slope_deg, 99.5)), height=460), key="slope")
    st.markdown("**Cell inspector**")
    i1, i2, i3 = st.columns([1, 1, 4])
    r = i1.number_input("row", 0, H - 1, world.goal[0] // 2, key=f"insp_r-{src}-{size}")
    c = i2.number_input("col", 0, W - 1, world.goal[1] // 2, key=f"insp_c-{src}-{size}")
    belief_now = res.belief if res is not None and cm is res.costmap else site.belief
    bd = cm.breakdown((int(r), int(c)))
    bd.update({f"P({n})": float(belief_now.class_probs[int(r), int(c), k]) for k, n in enumerate(CLASS_NAMES)})
    bd.update({"entropy (nats)": float(belief_now.entropy[int(r), int(c)]), "slope_deg": float(site.layers.slope_deg[int(r), int(c)]),
               "P(slope > 25°)": float(site.layers.p_slope_exceed[int(r), int(c)])})
    i3.dataframe(pd.DataFrame([{k: ("inf (impassable)" if isinstance(v, float) and not math.isfinite(v) else round(v, 4))
                                for k, v in bd.items()}]), hide_index=True)

# =========================================================================== 3. Uncertainty
with tabs[2]:
    if res is not None:
        steps = sorted(res.entropy_snapshots)
        sk = st.select_slider("Entropy snapshot (step)", steps, value=steps[-1], key="snap")
        ent_after, after_lbl = res.entropy_snapshots[sk], f"after step {sk}"
        obs = res.belief.observed_mask
    else:
        ent_after, after_lbl, obs = site.belief.entropy, "after first look", site.belief.observed_mask
    hmax = float(np.log(N_CLASSES))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(F.grid_map(site.prior.entropy, F.SEQ["entropy"], "Belief entropy: orbital prior", "nats", 0, hmax,
                                   height=440), key="ent_prior")
    with c2:
        fig = F.grid_map(ent_after, F.SEQ["entropy"], f"Belief entropy: {after_lbl}", "nats", 0, hmax, height=440)
        if res is not None:
            F.add_path(fig, res.trajectory[:sk + 1], "driven", F.TRACK, "solid", 2)
        st.plotly_chart(fig, key="ent_after")
    c3, c4, c5 = st.columns(3)
    with c3:
        st.plotly_chart(F.grid_map(site.layers.p_slope_exceed, F.SEQ["slope"], "P(slope > 25°) — Monte-Carlo DEM noise",
                                   "P", 0, 1, height=400), key="pslope")
    with c4:
        st.plotly_chart(F.grid_map(obs.astype(float), [[0, F.PLOT_BG], [1, "#3987e5"]], "Cells observed by the navcam",
                                   "observed", 0, 1, height=400, showscale=False), key="observed")
    with c5:
        b_now = res.belief if res is not None else site.belief
        truth_now = (trav.world if trav else world).true_class
        hit_prior = site.prior.class_probs.argmax(-1) == truth_now
        hit_now = b_now.class_probs.argmax(-1) == truth_now
        seen = b_now.observed_mask
        st.plotly_chart(F.class_map(b_now.class_probs.argmax(-1),
                                    f"Belief argmax — observed cells {hit_prior[seen].mean():.0%} → "
                                    f"{hit_now[seen].mean():.0%} correct", 400), key="cls_belief")
        st.caption(f"Whole map: {hit_prior.mean():.0%} → {hit_now.mean():.0%} (only {seen.mean():.0%} of cells "
                   "were observed; the rest is still the orbital prior).")
    with st.expander("Hidden ground truth (evaluation only — the rover never reads this)"):
        st.plotly_chart(F.class_map((trav.world if trav else world).true_class, "True terrain classes", 460),
                        key="cls_truth")

# =========================================================================== 4. Routes
with tabs[3]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(F.pareto_fig(routes.sweep, routes.front, routes.modes), key="pareto")
    with c2:
        fig = F.shaded_map(world, "Route options", 460, site.belief.entropy, "Purples", "entropy (nats)", hmax)
        for n, p in routes.modes.items():
            F.add_path(fig, p.path, n, F.ROUTE_COLORS[n], "solid" if n == mode else "dash")
        F.add_start_goal(fig, world)
        st.plotly_chart(fig, key="routes_map")
    tbl = pd.DataFrame(routes.table()).rename(columns={"success_prob": "P(success) belief",
                                                       "success_prob_true": "P(success) truth",
                                                       "path_entropy_mean": "mean entropy on path"})
    st.dataframe(tbl, hide_index=True, column_config={
        "P(success) belief": st.column_config.NumberColumn(format="percent"),
        "P(success) truth": st.column_config.NumberColumn(format="percent")})
    st.caption(f"Sweep: risk_weight ∈ linspace{tuple(cfg.pareto_sweep)} on the balanced belief map; the front keeps "
               "routes no other route beats on both length and P(success). 'truth' scores the same route on the hidden "
               "ground truth — the gap is what the rover must discover by sensing.")

# =========================================================================== 5. Science
with tabs[4]:
    sel = sci.selection
    st.caption(f"Budget {sel.budget_wh:.0f} Wh · {len(sel.selected)} selected, {len(sel.skipped)} skipped · "
               f"planned energy {sel.energy_wh:.0f} Wh (start → targets → goal). Value = "
               f"{cfg.sci_w_ice:g}·ice + {cfg.sci_w_rim:g}·rim/layering + {cfg.sci_w_bedrock:g}·P(bedrock); "
               "± = bootstrap std; info_gain = value × normalised entropy.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(F.targets_bar(sci.targets, selected_ids), key="sci_bar")
    with c2:
        fig = F.shaded_map(world, "Targets and selected tour", 420)
        for i, leg in enumerate([*sel.legs, *([sel.return_leg] if sel.return_leg else [])]):
            F.add_path(fig, leg.path, "selected tour", F.ROUTE_COLORS["science"], showlegend=i == 0)
        F.add_points(fig, [t.rc for t in sci.targets if t.id in selected_ids], "selected", F.ROUTE_COLORS["science"],
                     "diamond", 12, [f"#{t.id}" for t in sci.targets if t.id in selected_ids])
        F.add_points(fig, [t.rc for t in sci.targets if t.id not in selected_ids], "skipped", "#8a8a93", "diamond", 11,
                     [f"#{t.id}" for t in sci.targets if t.id not in selected_ids])
        F.add_start_goal(fig, world)
        st.plotly_chart(fig, key="sci_map")
    reasons = {e.evidence.get("target_id"): e.reason for e in sci.log}
    st.dataframe(pd.DataFrame([{"id": t.id, "cell": str(t.rc), "value": round(t.value_mean, 3),
                                "± std": round(t.value_std, 3), "info_gain": round(t.info_gain, 3),
                                "utility (from start)": round(t.utility, 3) if math.isfinite(t.utility) else None,
                                "selected": t.id in selected_ids, "decision": reasons.get(t.id, t.reason)}
                               for t in sci.targets]), hide_index=True)

# =========================================================================== 6. Decision Log
with tabs[5]:
    log = res.log if res is not None else sci.log
    st.caption("Traverse decisions" if res is not None else
               "Planning decisions (target selection). Run a traverse to see the rover's autonomous decisions.")
    counts = log.counts()
    all_dec = sorted(counts)
    default = [d for d in all_dec if d != "hazard_detected"] or all_dec
    lkey = f"{st.session_state.run}-{src}-{mode}-{seed}-{size}-{lam}-{budget}-{radius}-{frac}-{hz_radius}"  # fresh widgets per log
    show = st.multiselect("Decisions", all_dec, default=default, key=f"log_filter-{lkey}",
                          format_func=lambda d: f"{d} ({counts[d]})")
    entries = [e for e in log if e.decision in show]
    df = pd.DataFrame([{"step": e.step, "decision": e.decision, "event": e.event, "confidence": round(e.confidence, 3),
                        "cell": str(tuple(e.evidence["cell"])), "reason": e.reason} for e in entries])
    c1, c2 = st.columns([1.35, 1])
    with c1:
        ev = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"log_table-{lkey}-{'.'.join(show)}", height=440) if len(df) else None
        sel_rows = [i for i in (ev.selection.rows if ev is not None else []) if i < len(entries)]
        st.download_button("Download log (JSON)", json.dumps({"entries": log.to_records()}, indent=1),
                           "decision_log.json", "application/json", key="log_dl")
    with c2:
        chosen = entries[sel_rows[0]] if sel_rows else (entries[0] if entries else None)
        w_disp = trav.world if trav else world
        fig = F.shaded_map(w_disp, "Selected decision" if chosen else "", 440)
        if res is not None:
            F.add_path(fig, res.trajectory, "driven", F.TRACK, "solid", 2)
            F.add_rings(fig, res.meta["injected"])
        if chosen is not None:
            F.add_points(fig, [tuple(chosen.evidence["rover_rc"])], "rover at decision", "#ffffff", "triangle-up", 14)
            F.add_points(fig, [tuple(chosen.evidence["cell"])], "trigger cell", F.HAZARD, "x", 15)
        F.add_start_goal(fig, w_disp)
        st.plotly_chart(fig, key="log_map")
    if chosen is not None:
        st.markdown(f"**s{chosen.step} · {chosen.decision}** (event `{chosen.event}`, confidence "
                    f"{chosen.confidence:.2f} = 1 − H/ln4 at the trigger cell) — {chosen.reason}")
        st.json(_json_safe(chosen.evidence), expanded=True)
    st.markdown("**Decisions about one cell**")
    l1, l2, l3 = st.columns([1, 1, 4])
    lr = l1.number_input("row", 0, H - 1, int(chosen.evidence["cell"][0]) if chosen else 0,
                        key=f"log_r-{lkey}-{id(chosen)}")
    lc = l2.number_input("col", 0, W - 1, int(chosen.evidence["cell"][1]) if chosen else 0,
                        key=f"log_c-{lkey}-{id(chosen)}")
    hits = log.entries_for_cell(int(lr), int(lc))
    l3.write(f"{len(hits)} decision(s) at ({lr}, {lc})")
    for e in hits:
        st.markdown(f"- `s{e.step}` **{e.decision}** — {e.reason}")

# =========================================================================== 7. Perception
with tabs[6]:
    src = site.confusion_source
    trained = src != "fallback"
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(F.confusion_fig(site.confusion, "Sensor model P(predicted | true) — " +
                                        ("trained AI4MARS U-Net" if trained else "fallback matrix")), key="confusion")
        st.caption("This matrix *is* the rover's sensor model: it samples navcam observations and gives the Bayes "
                   "likelihood P(obs | true class) for every belief update.")
    with c2:
        mpath = ROOT / "perception" / "metrics.json"
        if mpath.exists():
            met = json.loads(mpath.read_text())
            st.metric("Validation mIoU", f"{met.get('miou', float('nan')):.3f}")
            st.dataframe(pd.DataFrame([met.get("iou", {})]), hide_index=True)
            imet = ROOT / "outputs" / "perception" / "infer_metrics.json"
            if imet.exists():
                im = json.loads(imet.read_text())
                st.markdown(f"MC-dropout (T={im.get('mc_passes')}): entropy on correct pixels "
                            f"**{im.get('entropy_correct_mean')}** vs wrong **{im.get('entropy_wrong_mean')}** nats; "
                            f"AUROC(entropy → error) **{im.get('entropy_error_auroc')}**")
        else:
            st.warning("No trained model yet: using the fallback confusion matrix. Train on Colab:")
            st.code("!pip install -q segmentation-models-pytorch datasets\n"
                    "!python -m perception.train --cache /content/cache\n!python -m perception.infer\n"
                    "# then commit perception/confusion.npy (+ metrics.json, outputs/perception/*.png)", language="bash")
    samples = sorted((ROOT / "outputs" / "perception").glob("sample_*.png"))
    if samples:
        st.markdown("**MC-dropout samples** (image | truth | prediction | predictive entropy)")
        for p in samples[:6]:
            st.image(str(p))
