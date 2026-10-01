"""Streamlit tab rendering — EIS (KK / Résultats par groupe / DRT / Reconstructions /
Calibration) and CV (Visualisation / Calibration).
"""

import math

import pandas as pd
import streamlit as st

from core.models import EISSession
from core.pipeline import DRT_MODEL_NAME, recompute_drt
from core.results_table import drt_hmc_summary, group_rows, is_missing, replicate_rows
from drt import diagnostics as drt_diag
from drt import engine as drt_engine
from fits.orazem_fit import ORAZEM_MODEL_NAME
from plotting.eis_plots import (
    drt_figure,
    reconstruction_comparison_figure,
    drt_reconstruction_figure_dual,
    open_reconstruction_matplotlib_window,
    calibration_figure,
    calibration_drt_figure,
    open_calibration_matplotlib_window,
    params_table_figure,
)
from plotting.cv_plots import (
    cv_current_figure,
    cv_calibration_figure_multi,
    open_cv_calibration_matplotlib_window_multi,
)
from plotting.kk_plots import residuals_figure, validation_summary_table
from exports.exporter import (
    export_calibration_csv,
    export_cv_calibration_csv_multi,
)


# ─────────────────────────────────────────────────────────────────────────────
# EIS
# ─────────────────────────────────────────────────────────────────────────────

def _spectra_labels_for_session(session: EISSession) -> dict:
    """label -> (réplicats bruts, spectre moyen) pour une session."""
    return {lbl: (reps, mean_sp) for lbl, mean_sp, reps, _an in session.iter_groups()}


def _render_kk_tab(validations: dict) -> None:
    """Onglet 1 — Validation KK (measurement model sur les réplicats BRUTS, calculé
    AVANT les fits), dupliqué par électrode."""
    if not validations:
        st.info("Lancez une analyse pour voir les résultats de validation.")
        return

    st.caption(
        "Verdict Kramers-Kronig du measurement model, calculé sur les réplicats BRUTS de "
        "chaque groupe **avant** tout fit ; la même analyse fournit la structure d'erreur "
        "σ(ω) qui pondère ensuite le fit du circuit."
    )
    electrodes = sorted(validations.keys())
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])

    for e, elec_tab in zip(electrodes, elec_tabs):
        with elec_tab:
            validation_results = validations.get(e)
            if not validation_results:
                st.info("Aucun résultat de validation pour cette électrode.")
                continue

            st.markdown("#### Récapitulatif")
            fig_table = validation_summary_table(validation_results, theme_mode="light")
            st.plotly_chart(fig_table, width='stretch', key=f"kk_table_e{e}")

            st.markdown("#### Résidus par spectre")
            labels_kk = list(validation_results.keys())
            selected_kk = st.selectbox("Spectre", labels_kk, key=f"kk_select_e{e}")
            vr = validation_results[selected_kk]

            fig_res = residuals_figure(vr, theme_mode="light")
            st.plotly_chart(fig_res, width='stretch', key=f"kk_res_e{e}")

            if vr.error_structure_message:
                st.error(vr.error_structure_message)
            elif vr.kk_message:
                (st.success if vr.all_valid else st.error)(vr.kk_message)
            for kk in vr.replicates:
                if kk.warning:
                    st.warning(f"**{kk.label}** : {kk.warning}")
            if vr.drift_warning:
                st.error(f"**Drift inter-réplicats** : {vr.drift_warning}")
            if vr.f_min_common and vr.f_max_common < float("inf"):
                st.success(
                    f"Plage KK-valide commune : "
                    f"**{vr.f_min_common:.2f} Hz** → **{vr.f_max_common:.2f} Hz**"
                )


# ── Résultats par groupe : intra-fit vs inter-réplicats, côte à côte ─────────────

def _fmt(x, spec: str = ".4g", missing: str = "—") -> str:
    return missing if is_missing(x) else format(x, spec)


def _fit_replicate_table(rows: list) -> pd.DataFrame:
    """Circuit (Orazem) : une ligne par réplicat + la moyenne."""
    out = []
    for r in rows:
        if "fit_value" not in r:
            continue
        ci = f"[{_fmt(r['fit_chi2_ci_low'], '.2f')} ; {_fmt(r['fit_chi2_ci_high'], '.2f')}]"
        out.append({
            "Spectre": r["spectrum"] + (" (moyenne)" if r["kind"] == "moyenne" else ""),
            f"{r['target_param']}": _fmt(r["fit_value"]),
            "± intra-fit (1σ)": _fmt(r["fit_std_intra"], ".2g"),
            "χ²ᵣ": _fmt(r["fit_chi2_reduced"], ".3g"),
            "χ²ᵣ attendu (2σ)": ci,
            "Convergé": "✅" if r["fit_converged"] else "❌",
            "Alertes": r["fit_warnings"],
        })
    return pd.DataFrame(out)


def _drt_replicate_table(rows: list) -> pd.DataFrame:
    """DRT : une ligne par réplicat + la moyenne, diagnostics HMC en clair."""
    out = []
    for r in rows:
        if "drt_Rct" not in r:
            continue
        hmc = r["drt_mode"] == "sample"
        rhat = r["drt_rhat_max"]
        div = r["drt_divergences"]
        out.append({
            "Spectre": r["spectrum"] + (" (moyenne)" if r["kind"] == "moyenne" else ""),
            "Mode": "HMC (sample)" if hmc else "MAP (optimize)",
            "Rct DRT": _fmt(r["drt_Rct"]),
            "± a posteriori (1σ)": _fmt(r["drt_Rct_std_intra"], ".2g", "non calculé (MAP)" if not hmc else "—"),
            "R̂ max": ("n/a (MAP)" if not hmc else
                      (_fmt(rhat, ".4f") + (" ⚠" if not is_missing(rhat) and rhat > drt_diag.RHAT_MAX else ""))),
            "Divergences": ("n/a (MAP)" if not hmc else
                            (_fmt(div, "d") + (" ⚠" if not is_missing(div) and div > drt_diag.MAX_DIVERGENCES else ""))),
            "ESS min (bulk/tail)": ("n/a (MAP)" if not hmc else
                                    f"{_fmt(r['drt_ess_bulk_min'], '.0f')} / {_fmt(r['drt_ess_tail_min'], '.0f')}"),
            "Convergé": "✅" if r["drt_converged"] else "❌",
            "Alertes": r["drt_alerts"],
        })
    return pd.DataFrame(out)


def _render_aggregate(prefix: str, g: dict, label: str, intra_label: str) -> None:
    """Agrégat d'un groupe : moyenne, inter-réplicats et intra-fit côte à côte."""
    if f"{prefix}_mean" not in g:
        st.caption(f"{label} : aucun agrégat.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} — moyenne (n = {g[f'{prefix}_n_used']})", _fmt(g[f"{prefix}_mean"]))
    c2.metric("Inter-réplicats (s)", _fmt(g[f"{prefix}_std_between"], ".3g"))
    c3.metric(intra_label, _fmt(g[f"{prefix}_std_within"], ".3g", "non calculé"))
    c4.metric("Incertitude de la moyenne", _fmt(g[f"{prefix}_sem"], ".3g"))
    if g.get(f"{prefix}_n_excluded"):
        st.warning(f"{g[f'{prefix}_n_excluded']} réplicat(s) non convergé(s) écarté(s) de l'agrégat.")


def _render_mean_fit_diagnostics(an) -> None:
    """Diagnostics du fit du spectre moyen (docs/MEASUREMENT_MODEL.md §8) : κ, rang,
    identifiabilité, différences unilatérales, bornes actives, χ²ᵣ et son intervalle."""
    fr = an.orazem.mean_fit
    d = fr.fit_diagnostics or {}
    ci = fr.chi2_reduced_ci or (math.nan, math.nan)
    non_id = [p for p, ok in (d.get("identifiable") or {}).items() if not ok]
    st.markdown(
        f"- χ²ᵣ = **{_fmt(fr.chi2_reduced, '.3g')}** ∈ attendu [{_fmt(ci[0], '.2f')} ; {_fmt(ci[1], '.2f')}] "
        f"(ν = {d.get('dof')}, ν_σ = {d.get('dof_sigma')})\n"
        f"- conditionnement κ(J) = {_fmt(d.get('condition_number'), '.3g')} ; rang "
        f"{d.get('rank')} / {len(fr.params)}\n"
        f"- non identifiables : {', '.join(non_id) or 'aucun'} ; dérivée unilatérale : "
        f"{', '.join(d.get('jacobian_one_sided') or []) or 'aucune'}\n"
        f"- bornes actives : {', '.join(f'{p} ({side})' for p, side, _b in d.get('active_bounds') or []) or 'aucune'}\n"
        f"- départs : {d.get('n_converged')} convergé(s) sur {d.get('n_starts')}"
    )
    mm = an.validation.measurement_model if an.validation is not None else None
    es = mm.error_structure if mm is not None else None
    if es is not None:
        st.caption(
            f"Structure d'erreur caractérisée sur CE groupe ({es.n_replicates} réplicats, "
            f"ν_σ = {es.dof}) : σ = α|Zj| + β|Zr − R_sol| + γ|Z|² + δ — α = {es.alpha:.3g}, "
            f"β = {es.beta:.3g}, γ = {es.gamma:.3g} Ω⁻¹, δ = {es.delta:.3g} Ω"
            + ("" if es.equal_re_im else " (σ_r ≠ σ_j : deux structures)")
        )


def _render_results_tab(sessions: dict) -> None:
    """Onglet 2 — Résultats par groupe : statut, puis pour chaque méthode l'incertitude
    INTRA-fit (par réplicat) et la variabilité INTER-réplicats, côte à côte."""
    st.subheader("Résultats par groupe — réplicats bruts et moyenne")
    st.caption(
        "**Intra-fit** : écart-type d'UN ajustement (covariance pondérée par σ du "
        "measurement model ; pour la DRT, écart-type a posteriori HMC, non calculé en "
        "MAP). **Inter-réplicats** : dispersion entre réplicats (s, ddof = 1), qui "
        "contient le bruit ET la variabilité propre aux réplicats. **Incertitude de la "
        "moyenne** : √(max(s², v̄)/n) — la dispersion observée prime dès qu'elle dépasse "
        "ce que le bruit explique (valeur de calibration recommandée : moyenne ± cette "
        "incertitude)."
    )
    electrodes = sorted(sessions.keys())
    if not electrodes:
        return
    rep_rows = replicate_rows(sessions)
    grp_rows = group_rows(sessions)
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
    for e, tab in zip(electrodes, elec_tabs):
        with tab:
            session = sessions[e]
            for label, _mean_sp, _reps, an in session.iter_groups():
                st.markdown(f"#### {label}")
                if an is None:
                    st.info("Aucune analyse pour ce groupe.")
                    continue
                if not an.ok:
                    st.error(an.message)
                    continue
                vr = an.validation
                if vr is not None and vr.kk_message:
                    (st.success if vr.all_valid else st.warning)(vr.kk_message)
                g = next(r for r in grp_rows if r["electrode"] == e and r["group"] == label)
                rows = [r for r in rep_rows if r["electrode"] == e and r["group"] == label]
                col_fit, col_drt = st.columns(2)
                with col_fit:
                    st.markdown(f"**Circuit (fit Orazem) — {g.get('target_param', 'cible')}**")
                    st.dataframe(_fit_replicate_table(rows), hide_index=True, width='stretch')
                    _render_aggregate("fit", g, g.get("target_param", "cible"), "Intra-fit (√v̄)")
                    if g.get("fit_cochran_p") is not None and not is_missing(g.get("fit_cochran_p")):
                        st.caption(f"Q de Cochran : p = {g['fit_cochran_p']:.2g} "
                                   "(p < 0,05 : dispersion inter > intra-fit).")
                    if an.orazem is not None:
                        with st.expander("Diagnostics du fit du spectre moyen"):
                            _render_mean_fit_diagnostics(an)
                with col_drt:
                    st.markdown("**DRT — Rct (arc de transfert, Bissessur)**")
                    drt_table = _drt_replicate_table(rows)
                    if drt_table.empty:
                        st.info("DRT non calculée pour ce groupe.")
                    else:
                        st.dataframe(drt_table, hide_index=True, width='stretch')
                        _render_aggregate("drt", g, "Rct DRT", "A posteriori (√v̄)")
                    for lbl_sp, why in an.drt_failures.items():
                        st.warning(f"DRT de « {lbl_sp} » non calculée : {why}")
                for w in an.warnings:
                    if not w.startswith("DRT de «"):
                        st.warning(w)


# ── DRT ──────────────────────────────────────────────────────────────────────────

def _drt_result_for(spectrum):
    """FitResult DRT du spectre (pipeline ou recalcul), None si indisponible."""
    return (getattr(spectrum, "fit_results", {}) or {}).get(DRT_MODEL_NAME)


def _drt_spectrum_status(label: str, fr) -> None:
    """Mode DRT, diagnostics de convergence HMC et provenance du Rct, en clair.

    Rend visible qu'un Rct issu d'un repli (Rp, aire totale) n'est PAS l'arc de
    transfert de charge, et qu'une DRT MAP n'a AUCUN diagnostic de convergence.
    """
    h = drt_hmc_summary(fr)
    if h["mode"] == "sample":
        conv = (f"R̂ max = {_fmt(h['rhat_max'], '.4f')} (seuil {drt_diag.RHAT_MAX}) · "
                f"divergences = {_fmt(h['divergences'], 'd')} · ESS min = "
                f"{_fmt(h['ess_bulk_min'], '.0f')}/{_fmt(h['ess_tail_min'], '.0f')}")
        mode_txt = "bayésienne (HMC)"
    else:
        conv = "MAP : aucun diagnostic de convergence (recalculer en HMC pour R̂/divergences)"
        mode_txt = "MAP (optimize)"
    src = (getattr(fr, "params", {}) or {}).get("rct_source")
    src_txt = {"rp_fallback": "⚠ Rct par REPLI sur Rp (aire totale ≠ arc de transfert)",
               "peak_single": "Rct d'un pic unique"}.get(src, "Rct = pic pénultième (Bissessur)")
    line = f"**{label}** — DRT {mode_txt} · {conv} · {src_txt}"
    if not fr.converged or src == "rp_fallback":
        st.error(line + ("" if fr.converged else " · NON convergée"))
    elif fr.warnings:
        st.warning(line + " · " + " ; ".join(fr.warnings))
    else:
        st.caption(line)


def _render_drt_tab(sessions: dict) -> None:
    """Onglet 3 — Distribution des temps de relaxation (DRT).

    La DRT est calculée par le pipeline sur CHAQUE réplicat brut et sur la moyenne de
    chaque groupe (mode de ``fit.drt.mode``). Un bouton recalcule un spectre choisi en
    bayésien ('sample', HMC) via ``core.pipeline.recompute_drt`` (seul point d'entrée
    du recalcul) ; le Rct DRT agrégé du groupe est alors mis à jour.
    """
    st.subheader("Distribution des temps de relaxation (DRT)")

    ok, why = drt_engine.engine_available()
    if not ok:
        st.error(
            f"Moteur DRT indisponible : {why}. Installez l'extra DRT "
            "(`pip install -r requirements-drt.txt`) et la toolchain CmdStan : relancez "
            "**launch.bat** (ou `python setup_drt_bayesien.py --ensure`)."
        )
        return

    version_warning = st.session_state.get("drt_version_warning")
    if version_warning:
        st.warning(f"⚠️ {version_warning}")

    config = st.session_state.get("eis_config", {})
    electrodes = sorted(sessions.keys())
    if all(sessions[e].drt_mode is None for e in electrodes):
        st.info("La DRT n'a pas été lancée par cette analyse (désactivée dans ⚙️, ou moteur "
                "indisponible au moment de l'analyse). Recalcul possible spectre par spectre "
                "ci-dessous.")

    # (a) Recalcul bayésien ('sample') à la demande sur un spectre choisi.
    st.markdown("#### Recalcul bayésien (sample)")
    st.caption(
        "Le recalcul **bayésien (HMC)** — **2 à 5 minutes** par spectre — fournit les "
        "intervalles de crédibilité et les diagnostics de convergence (R̂, divergences, ESS)."
    )
    choices = {}
    for e in electrodes:
        # L'objet spectre est passé tel quel à recompute_drt (résolution par IDENTITÉ).
        for lbl, mean_sp, reps, _an in sessions[e].iter_groups():
            choices[f"Électrode {e} — {lbl} — moyenne"] = (e, f"{lbl} (moyenne)", mean_sp)
            for i, sp in enumerate(reps):
                choices[f"Électrode {e} — {lbl} — réplicat {i + 1} ({sp.label})"] = (e, sp.label, sp)
    if choices:
        sel = st.selectbox("Spectre à recalculer en bayésien", list(choices.keys()),
                           key="drt_sample_select")
        e_sel, lbl_sel, sp_sel = choices[sel]
        if st.button("🎲 Recalculer en bayésien (sample)", key="drt_sample_btn",
                     help="Échantillonnage HMC sur le spectre choisi ; le résultat persiste."):
            with st.spinner(f"Échantillonnage HMC de « {lbl_sel} »… (plusieurs minutes)"):
                try:
                    recompute_drt(sessions[e_sel], sp_sel, config, mode="sample")
                except (ValueError, RuntimeError) as exc:   # spectre invalide / échec de CmdStan
                    st.error(f"Échec du recalcul bayésien : {exc}")
                else:
                    st.success(f"DRT bayésienne (sample) calculée pour « {lbl_sel} ».")
                    st.rerun()

    # (b) Une figure DRT par électrode (moyennes), puis les réplicats d'un groupe choisi.
    st.markdown("#### γ(τ) des spectres moyens, par électrode")
    cols = st.columns(len(electrodes)) if electrodes else []
    for e, col in zip(electrodes, cols):
        with col:
            items = []
            for lbl, mean_sp, _reps, an in sessions[e].iter_groups():
                fr = _drt_result_for(mean_sp)
                if fr is None:
                    why = (an.message if an is not None and not an.ok
                           else (an.drt_failures.get(mean_sp.label) if an is not None else None))
                    st.info(f"DRT « {lbl} » indisponible" + (f" — {why}" if why else "."))
                    continue
                items.append((lbl, fr))
                _drt_spectrum_status(lbl, fr)
            st.plotly_chart(drt_figure(items, title=f"Électrode {e}"), width='stretch', key=f"drt_e{e}")

    st.markdown("#### γ(τ) de chaque réplicat d'un groupe")
    groups = {f"Électrode {e} — {lbl}": reps for e in electrodes
              for lbl, _m, reps, _an in sessions[e].iter_groups()}
    if groups:
        gsel = st.selectbox("Groupe", list(groups.keys()), key="drt_rep_group")
        items = []
        for i, sp in enumerate(groups[gsel]):
            fr = _drt_result_for(sp)
            if fr is not None:
                items.append((f"réplicat {i + 1}", fr))
                _drt_spectrum_status(f"réplicat {i + 1} ({sp.label})", fr)
        if items:
            st.plotly_chart(drt_figure(items, title=gsel), width='stretch', key="drt_reps")
        else:
            st.info("Aucune DRT de réplicat pour ce groupe.")


# ── Reconstructions ──────────────────────────────────────────────────────────────

def _render_reconstruction_tab(sessions: dict) -> None:
    """Onglet 4 — Reconstructions Nyquist (circuit Orazem vs DRT)."""
    st.subheader("Reconstructions Nyquist — circuit (Orazem) vs DRT")

    for e, session in sorted(sessions.items()):
        if session.circuit:
            st.caption(f"Électrode {e} — circuit : `{session.circuit['expression']}` · cible : "
                       f"**{session.circuit['target_param']}**")
        st.plotly_chart(params_table_figure(session), width='stretch', key=f"params_table_e{e}")

    st.plotly_chart(reconstruction_comparison_figure(sessions), width='stretch', key="recon_multi")

    electrodes = sorted(sessions.keys())
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
    for e, elec_tab in zip(electrodes, elec_tabs):
        with elec_tab:
            session = sessions[e]
            spectra_map = _spectra_labels_for_session(session)
            if not spectra_map:
                st.info("Aucun spectre pour cette électrode.")
                continue
            sel_label = st.selectbox("Groupe", list(spectra_map.keys()), key=f"recon_select_e{e}")
            reps, mean_sp = spectra_map[sel_label]
            st.plotly_chart(
                drt_reconstruction_figure_dual(
                    mean_sp, fr_drt=mean_sp.fit_results.get(DRT_MODEL_NAME),
                    fr_randles=mean_sp.fit_results.get(ORAZEM_MODEL_NAME),
                    label=f"{sel_label} (moyenne)",
                ),
                width='stretch', key=f"recon_avg_e{e}",
            )
            if reps:
                rep_idx = st.selectbox("Réplicat", list(range(1, len(reps) + 1)),
                                       key=f"recon_rep_select_e{e}")
                rep_sp = reps[rep_idx - 1]
                st.plotly_chart(
                    drt_reconstruction_figure_dual(
                        rep_sp, fr_drt=rep_sp.fit_results.get(DRT_MODEL_NAME),
                        fr_randles=rep_sp.fit_results.get(ORAZEM_MODEL_NAME),
                        label=f"{sel_label} — réplicat {rep_idx}",
                    ),
                    width='stretch', key=f"recon_rep_fig_e{e}",
                )

    if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="recon_matplotlib_btn"):
        open_reconstruction_matplotlib_window(sessions)


def _render_calibration_tab(sessions: dict) -> None:
    """Onglet 5 — Calibration EIS."""
    st.subheader("Courbe de calibration — Signal normalisé vs log([c])")

    electrodes = [
        e for e in sorted(sessions.keys())
        if sum(1 for g in sessions[e].groups if g.concentration > 0) >= 2
    ]
    if not electrodes:
        st.info(
            "Chargez au moins **2 spectres d'hybridation** avec "
            "des concentrations positives pour tracer la calibration."
        )
        return

    st.caption(
        "Valeur par groupe = paramètre cible du fit du spectre MOYEN (et Rct DRT du "
        "spectre moyen). Les groupes arrêtés (structure d'erreur non caractérisable) "
        "n'ont pas de fit et n'apparaissent pas."
    )
    cols = st.columns(len(electrodes)) if len(electrodes) > 1 else [st.container()]
    for e, col in zip(electrodes, cols):
        with col:
            st.markdown(f"**Électrode {e}**")
            st.plotly_chart(calibration_figure(sessions[e]), width='stretch', key=f"calib_fig_e{e}")

    st.markdown("#### Calibration DRT — log(Rct) vs log([c])")
    st.caption(
        "Rct de la DRT = arc de transfert de charge (pic pénultième). Vérifiez "
        "l'indication de mode/provenance Rct dans l'onglet **Courbes DRT**."
    )
    for e in electrodes:
        st.markdown(f"**Électrode {e}**")
        st.plotly_chart(
            calibration_drt_figure(sessions[e], model_name=DRT_MODEL_NAME),
            width='stretch', key=f"calib_drt_fig_e{e}",
        )

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="calib_matplotlib_btn"):
            open_calibration_matplotlib_window({e: sessions[e] for e in electrodes})
    with btn_cols[1]:
        st.download_button(
            "📥 Télécharger les valeurs (CSV)",
            data=export_calibration_csv({e: sessions[e] for e in electrodes}),
            file_name="eis_calibration.csv",
            mime="text/csv",
            key="calib_csv_btn",
        )


def render_eis_tabs(sessions: dict, config: dict, validations: dict) -> None:
    """Onglets d'analyse EIS : Validation KK / Résultats par groupe / Courbes DRT /
    Reconstructions / Calibration.

    Args:
        sessions: {electrode_index: EISSession} — les sessions RÉELLES de
            ``st.session_state['eis_sessions']`` (un recalcul DRT y écrit).
        config: app config dict.
        validations: {electrode_index: {label: ValidationResult}}.
    """
    del config  # la config d'analyse est relue dans st.session_state['eis_config']
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Validation KK", "2️⃣ Résultats par groupe", "3️⃣ Courbes DRT",
        "4️⃣ Reconstructions Nyquist", "5️⃣ Calibration",
    ])

    with tab1:
        _render_kk_tab(validations or {})
    with tab2:
        _render_results_tab(sessions)
    with tab3:
        _render_drt_tab(sessions)
    with tab4:
        _render_reconstruction_tab(sessions)
    with tab5:
        _render_calibration_tab(sessions)


# ─────────────────────────────────────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────────────────────────────────────

def render_cv_tabs(cv_sessions: dict) -> None:
    """Render CV analysis tabs: Visualisation I/U, Calibration.

    Args:
        cv_sessions: {electrode_index: CVSession}, une entrée par électrode disponible.
    """
    tab1, tab2 = st.tabs(["1️⃣ Visualisation I/U", "2️⃣ Calibration"])

    electrodes = sorted(cv_sessions.keys())

    with tab1:
        st.subheader("Voltampérométrie cyclique — Courant vs Potentiel")
        if not electrodes:
            st.info("Aucune donnée CV chargée.")
        else:
            elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
            for e, elec_tab in zip(electrodes, elec_tabs):
                with elec_tab:
                    session = cv_sessions[e]
                    if session.probe is not None or session.groups:
                        st.plotly_chart(cv_current_figure(session), width='stretch', key=f"cv_current_e{e}")
                    else:
                        st.info("Aucune donnée CV chargée pour cette électrode.")

    with tab2:
        st.subheader("Calibration CV — Signal normalisé")
        calibratable = {
            e: s for e, s in cv_sessions.items()
            if sum(1 for g in s.groups if g.concentration > 0) >= 2
        }
        if not calibratable:
            st.info("Ajoutez au moins 2 concentrations pour la calibration.")
        else:
            st.plotly_chart(cv_calibration_figure_multi(calibratable), width='stretch')

            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="cv_calib_matplotlib_btn"):
                    open_cv_calibration_matplotlib_window_multi(calibratable)
            with btn_cols[1]:
                st.download_button(
                    "📥 Télécharger les valeurs (CSV)",
                    data=export_cv_calibration_csv_multi(calibratable),
                    file_name="cv_calibration.csv",
                    mime="text/csv",
                    key="cv_calib_csv_btn",
                )
