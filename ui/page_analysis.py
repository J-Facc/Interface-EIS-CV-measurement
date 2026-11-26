"""Page 4 — Analyse : fit Randles (Orazem) + DRT en arrière-plan, CV, étalonnages.

Le bouton « Lancer l'analyse » démarre TOUS les calculs en arrière-plan
(``core.analysis.AnalysisRun``) ; seuls une barre de progression et un compteur s'affichent
jusqu'à la fin, puis les onglets DRT · Rct (fit Randles) · Étalonnage EIS · CV · Étalonnage CV.
Aucune physique ici : tableaux (``core.report_tables``), étalonnages (``core.calibration``),
figures (``plotting``).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import drt.engine as drt_engine
from core.analysis import AnalysisRun, DRTRecomputeRun
from core.config import config_to_dict, load_config
from core.experiment import format_concentration
from core.pipeline import InvalidAnalysisInput
from core.report_tables import drt_rows, failure_rows, group_rows, rct_rows
from core.results import WEIGHTING_LABELS
from plotting.eis_plots import drt_peak_figure, fit_nyquist_figure
from ui import nav
from ui import state as S
from ui.circuit_editor import circuit_editor

RECOMPUTE = "drt_recompute_run"


def _config() -> dict:
    return config_to_dict(load_config())


# ─────────────────────────────────────────────────────────────────────────────
# Lancement et progression
# ─────────────────────────────────────────────────────────────────────────────

def _settings(config: dict) -> dict:
    with st.expander("⚙️ Circuit et réglages de l'analyse", expanded=False):
        circuit = circuit_editor(config)
        st.markdown("**DRT** (chaque réplicat et le spectre moyen de chaque groupe)")
        dcfg = config["fit"]["drt"]
        c1, c2, c3 = st.columns(3)
        run_drt = c1.checkbox("Calculer la DRT", value=bool(dcfg.get("enabled", True)), key="an_drt_on")
        mode = c2.radio("Mode", ["optimize", "sample"], index=0 if dcfg.get("mode") == "optimize" else 1,
                        key="an_drt_mode", disabled=not run_drt, horizontal=True,
                        help="optimize : MAP, ~1 s/spectre. sample : HMC, 2 à 5 min/spectre, avec "
                             "diagnostics de convergence et intervalles de crédibilité.")
        c3.caption(f"Fenêtre du pic principal : τ ∈ {dcfg['tau_window']} s (config/default.yaml, "
                   "fit.drt.tau_window)")
        workers = st.number_input("Processus de calcul (0 = automatique : cœurs − 1)", min_value=0, max_value=64,
                                  value=0, step=1, key="an_workers")
    return {"circuit": circuit, "run_drt": run_drt, "drt_mode": mode, "workers": int(workers) or None}


def _start(exp, ms, config, inputs) -> None:
    try:
        run = AnalysisRun(exp, ms, config, circuit=inputs["circuit"], run_drt=inputs["run_drt"],
                          drt_mode=inputs["drt_mode"], max_workers=inputs["workers"])
    except InvalidAnalysisInput as exc:
        st.error(str(exc))
        return
    S.invalidate_results(st.session_state)
    st.session_state[S.ANALYSIS_RUN] = run
    st.rerun()


@st.fragment(run_every=1.0)
def _progress_fragment(state_key: str, what: str) -> None:
    run = st.session_state.get(state_key)
    if run is None:
        return
    run.poll()
    done, total = run.progress
    st.progress(done / total if total else 0.0, text=f"{what} : {done}/{total} tâche(s) terminée(s) "
                                                     f"({run.bg.running} en cours ou en attente, "
                                                     f"{run.bg.max_workers} processus, {run.bg.elapsed_s:.0f} s)")
    if st.button("⏹️ Annuler", key=f"cancel_{state_key}"):
        run.cancel()
        st.session_state[state_key] = None
        st.rerun(scope="app")
    if run.finished:
        if state_key == S.ANALYSIS_RUN:
            st.session_state[S.RESULTS] = run.results
            st.session_state[S.RESULTS_SOURCE] = "computed"
        st.session_state[state_key] = None
        st.session_state["_recompute_errors"] = getattr(run, "errors", lambda: {})()
        st.rerun(scope="app")


# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────

def _electrode_picker(results, key: str):
    elecs = results.eis.electrodes()
    if not elecs:
        return None
    return st.selectbox("Électrode", elecs, key=key, format_func=lambda ce: f"Canal {ce[0]} · électrode {ce[1]}")


def _groups_of(results, ce):
    from core.experiment import step_sort_key
    gs = [g for g in results.eis.groups.values() if (g.channel_id, g.electrode) == tuple(ce)]
    return sorted(gs, key=lambda g: step_sort_key(g.step))


def _group_label(g) -> str:
    return g.step if not g.concentration else f"{g.step} — {format_concentration(g.concentration)}"


def _render_drt_tab(results, config: dict, theme: str) -> None:
    eis = results.eis
    if eis.drt_messages:
        for m in eis.drt_messages:
            st.warning(m)
    if not any(s.drt is not None for s in eis.spectra.values()) and not eis.drt_mode:
        st.info("La DRT n'a pas été calculée (désactivée dans les réglages ou moteur indisponible).")
    st.caption("Rct_DRT = aire du pic PRINCIPAL de γ(τ) (le plus haut dans la fenêtre τ de "
               "`fit.drt.tau_window`), intégrée entre les minima qui l'encadrent. La colonne « moteur » "
               "(pic pénultième ±3 en ln τ) reste affichée à titre de contrôle.")
    ce = _electrode_picker(results, "drt_elec")
    if ce is not None:
        gs = _groups_of(results, ce)
        sel = st.multiselect("Étapes", [g.key for g in gs], default=[g.key for g in gs],
                             format_func=lambda k: _group_label(eis.groups[k]), key=f"drt_steps_{ce}")
        show = st.radio("Spectres", ["moyennes", "réplicats", "tous"], horizontal=True, key="drt_show")
        items = []
        for k in sel:
            for sp in eis.group_spectra(k):
                if (show == "moyennes" and not sp.is_mean) or (show == "réplicats" and sp.is_mean):
                    continue
                items.append((f"{eis.groups[k].step} {sp.label}", sp.drt))
        if any(d is not None for _, d in items):
            st.plotly_chart(drt_peak_figure(items, title=f"DRT — canal {ce[0]}, électrode {ce[1]}",
                                            theme_mode=theme), width="stretch", key="drt_fig")
    st.markdown("**Diagnostics par spectre**")
    st.dataframe(pd.DataFrame(drt_rows(results)), hide_index=True, width="stretch")
    _render_recompute(results, config)


def _render_recompute(results, config: dict) -> None:
    ok, why = drt_engine.engine_available()
    with st.expander("🔁 Recalculer des DRT (sample = HMC, 2 à 5 min par spectre)"):
        if not ok:
            st.error(f"Moteur DRT indisponible : {why}.")
            return
        keys = sorted(results.eis.spectra)
        chosen = st.multiselect("Spectres", keys, key="recomp_keys")
        c1, c2 = st.columns(2)
        busy = st.session_state.get(RECOMPUTE) is not None or st.session_state.get(S.ANALYSIS_RUN) is not None
        for col, mode, label in ((c1, "sample", "Recalculer en sample"), (c2, "optimize", "Revenir en optimize")):
            if col.button(label, disabled=not chosen or busy, key=f"recomp_{mode}"):
                st.session_state[RECOMPUTE] = DRTRecomputeRun(results, chosen, config, mode)
                st.rerun()
        if st.session_state.get(RECOMPUTE) is not None:
            _progress_fragment(RECOMPUTE, "Recalcul DRT")
        for k, msg in (st.session_state.pop("_recompute_errors", None) or {}).items():
            st.error(f"DRT de « {k} » non recalculée : {msg}")


def _render_rct_tab(results, theme: str) -> None:
    eis = results.eis
    circuit = eis.circuit or {}
    st.caption(f"Circuit : `{circuit.get('expression', '—')}` — paramètre cible : **{circuit.get('target_param')}**. "
               "Verdict Kramers-Kronig : SIGNALÉ, jamais excluant.")
    st.markdown("**Groupes**")
    st.dataframe(pd.DataFrame(group_rows(results)), hide_index=True, width="stretch")
    st.markdown("**Paramètres du fit et Rct_DRT, par électrode / étape / réplicat**")
    st.dataframe(pd.DataFrame(rct_rows(results)), hide_index=True, width="stretch")
    st.markdown("**Contrôle qualité : Nyquist avec la courbe du fit**")
    ce = _electrode_picker(results, "rct_elec")
    if ce is None:
        return
    gs = _groups_of(results, ce)
    c1, c2 = st.columns(2)
    gk = c1.selectbox("Étape", [g.key for g in gs], format_func=lambda k: _group_label(eis.groups[k]),
                      key=f"rct_step_{ce}")
    g = eis.groups[gk]
    sps = eis.group_spectra(gk)
    sk = c2.selectbox("Spectre", [s.key for s in sps], format_func=lambda k: eis.spectra[k].label, key=f"rct_sp_{gk}")
    sp = eis.spectra[sk]
    if g.message:
        (st.warning if g.status != "failed" else st.error)(g.message)
    if g.kk_state == "non conforme":
        st.warning(f"⚠️ Verdict Kramers-Kronig non conforme (signalé, non exclu) : {g.kk_message}")
    if sp.fit is None:
        st.error(f"Aucun fit pour ce spectre : {sp.fit_error or 'non calculé'}")
        return
    st.caption(f"Pondération : {WEIGHTING_LABELS.get(sp.fit.weighting)} — χ²ᵣ = {sp.fit.chi2_reduced:.3g}"
               + (f" (attendu ∈ [{sp.fit.chi2_ci[0]:.2f} ; {sp.fit.chi2_ci[1]:.2f}])" if sp.fit.chi2_ci else ""))
    st.plotly_chart(fit_nyquist_figure(sp, sp.fit, title=f"{sp.key} — fit du circuit", theme_mode=theme),
                    width="stretch", key="rct_fit_fig")
    for w in sp.fit.warnings:
        st.caption(f"• {w}")


def _render_results(results, config: dict) -> None:
    theme = st.session_state[S.THEME]
    if st.session_state.get(S.RESULTS_SOURCE) == "archive":
        st.info(f"📦 Résultats rechargés depuis une archive (calculés le {results.computed_at}, "
                f"v{results.app_version}) — affichage sans recalcul.")
    fails = failure_rows(results)
    if fails:
        with st.expander(f"⚠️ Spectres / courbes en échec : {len(fails)}", expanded=True):
            st.dataframe(pd.DataFrame(fails), hide_index=True, width="stretch")
    tabs = st.tabs(["DRT", "Rct (fit Randles)", "Étalonnage EIS", "CV", "Étalonnage CV"])
    with tabs[0]:
        _render_drt_tab(results, config, theme)
    with tabs[1]:
        _render_rct_tab(results, theme)
    try:
        from ui.calibration_tabs import render_cv_tab, render_eis_calibration_tab, render_cv_calibration_tab
    except ImportError:       # phase 5
        return
    with tabs[2]:
        render_eis_calibration_tab(results, st.session_state[S.EXPERIMENT], theme)
    with tabs[3]:
        render_cv_tab(results, st.session_state[S.EXPERIMENT], st.session_state.get(S.MEASUREMENTS), theme)
    with tabs[4]:
        render_cv_calibration_tab(results, st.session_state[S.EXPERIMENT], theme)


def render() -> None:
    st.title("🧮 Analyse")
    exp = st.session_state.get(S.EXPERIMENT)
    ms = st.session_state.get(S.MEASUREMENTS)
    if exp is None or not ms:
        st.warning("Aucune donnée nettoyée : importez et nettoyez d'abord les fichiers (ou chargez une archive).")
        nav.link("import", "→ Import & nettoyage")
        st.stop()
    config = _config()
    inputs = _settings(config)
    run = st.session_state.get(S.ANALYSIS_RUN)
    if run is not None:
        st.info("Calcul en cours en arrière-plan : les DRT et les fits s'afficheront quand TOUS les spectres "
                "seront calculés.")
        _progress_fragment(S.ANALYSIS_RUN, "Analyse")
        return
    results = st.session_state.get(S.RESULTS)
    can_run = inputs["circuit"] is not None
    if results is None:
        if st.button("▶️ Lancer l'analyse", type="primary", disabled=not can_run):
            _start(exp, ms, config, inputs)
        n_eis = sum(1 for m in ms if m.kind == "EIS")
        st.caption(f"{n_eis} spectre(s) EIS, {len(ms) - n_eis} courbe(s) CV à analyser.")
        return
    with st.expander("🔄 Relancer l'analyse (écrase les résultats affichés)"):
        sure = st.checkbox("Je confirme : recalculer et remplacer les résultats actuels", key="an_confirm")
        if st.button("▶️ Relancer l'analyse", disabled=not (sure and can_run)):
            _start(exp, ms, config, inputs)
    _render_results(results, config)
    nav.link("export", "→ Export")
