"""Streamlit tab rendering — EIS (Visualisation / Measurement model & fit Orazem / DRT) and
CV (Visualisation / Calibration).

Aucune logique métier ici : les tableaux viennent de ``core/results_table.py`` (lignes
prêtes à afficher, testées sans navigateur) et les figures de ``plotting/``.
"""

import pandas as pd
import streamlit as st

from core.models import GROUP_ERROR_STRUCTURE_UNAVAILABLE
from core.pipeline import DRT_MODEL_NAME, recompute_drt
from core.results_table import (
    DRT_ABSENT,
    DRT_FAILED,
    DRT_OK,
    KK_CONFORM,
    KK_NONCONFORM,
    SEVERITY_ERROR,
    SEVERITY_OK,
    SEVERITY_WARNING,
    drt_diagnostic_rows,
    drt_replicate_envelope,
    fit_alert_lines,
    fit_diagnostic_rows,
    is_missing,
    kk_rows,
    kk_verdict_state,
    param_rows,
    worst_severity,
)
from drt import diagnostics as drt_diag
from drt import engine as drt_engine
from fits.orazem_fit import CONDITION_NUMBER_WARN
from plotting.eis_plots import (
    bode_figure,
    drt_aggregate_figure,
    drt_figure,
    fit_nyquist_figure,
    fit_residuals_figure,
    nyquist_normalized_figure,
    nyquist_replicates_figure,
)
from plotting.cv_plots import (
    cv_current_figure,
    cv_calibration_figure_multi,
    open_cv_calibration_matplotlib_window_multi,
)
from plotting.kk_plots import residuals_figure
from exports.exporter import export_cv_calibration_csv_multi


# ─────────────────────────────────────────────────────────────────────────────
# EIS — éléments communs
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x, spec: str = ".4g", missing: str = "—") -> str:
    return missing if is_missing(x) else format(x, spec)


_SEVERITY_ICON = {SEVERITY_OK: "🟢", SEVERITY_WARNING: "🟠", SEVERITY_ERROR: "🔴"}
_KK_ICON = {KK_CONFORM: "✅ conforme", KK_NONCONFORM: "❌ non conforme"}


def _kk_icon(state: str) -> str:
    return _KK_ICON.get(state, "❔ indéterminé")


def _tick(ok, na: str = "n/a (MAP)") -> str:
    """✅ / ⚠️ pour un test passé / échoué ; ``na`` quand le test ne s'applique pas."""
    return na if ok is None else ("✅" if ok else "⚠️")


def _pick_group(session, electrode: int, key: str, flagged: list):
    """Sélecteur de groupe d'une électrode. Retourne ``(label, mean_sp, reps, an)`` ou None.

    Par défaut le premier groupe qui demande l'attention de l'utilisateur (``flagged``),
    sinon le premier.
    """
    groups = list(session.iter_groups())
    if not groups:
        return None
    labels = [g[0] for g in groups]
    default = next((i for i, lbl in enumerate(labels) if lbl in flagged), 0)
    sel = st.selectbox("Groupe", labels, index=default, key=f"{key}_e{electrode}")
    return groups[labels.index(sel)]


# ─────────────────────────────────────────────────────────────────────────────
# EIS — onglet 1 : Visualisation (données mesurées, aucun fit)
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_PREPROCESSED = "Prétraitées (celles qui sont analysées)"
_SOURCE_RAW = "Brutes (avant exclusions du prétraitement)"


def _render_visualisation_tab(sessions: dict, normalized: dict = None) -> None:
    """Onglet 1 — Nyquist et Bode des spectres mesurés : réplicats superposés + moyenne,
    par électrode, avec filtre de groupes. Aucun résultat de fit ici."""
    st.subheader("Spectres d'impédance — Nyquist et Bode")
    electrodes = sorted(sessions.keys())
    if not electrodes:
        st.info("Aucun spectre EIS à afficher.")
        return
    st.caption(
        "Données mesurées uniquement (aucun fit). Chaque groupe = ses réplicats (traits fins) et "
        "leur moyenne (trait épais) ; cliquez un groupe dans la légende pour le masquer, "
        "double-cliquez pour l'isoler."
    )
    has_raw = any(sessions[e].raw_groups for e in electrodes)
    c1, c2 = st.columns([3, 2])
    with c1:
        source = st.radio("Données", [_SOURCE_PREPROCESSED] + ([_SOURCE_RAW] if has_raw else []),
                          horizontal=True, key="eis_visu_source")
    with c2:
        show_reps = st.checkbox("Afficher les réplicats", value=True, key="eis_visu_reps")
    raw = source == _SOURCE_RAW

    specs_norm = [d for d in sorted((normalized or {}).values(), key=lambda d: d["concentration"])]
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes]
                        + (["Normalisé E1 + E2"] if specs_norm else []))
    if specs_norm:
        with elec_tabs[-1]:
            st.caption("|(Z_probe − Z_c) / Z_probe| point à point, spectres moyens prétraités, moyenné "
                       "sur les électrodes disponibles (donnée dérivée, aucun fit).")
            st.plotly_chart(nyquist_normalized_figure(specs_norm, title="Normalisé E1 + E2"),
                            width='stretch', key="eis_visu_nyq_norm")
    for e, tab in zip(electrodes, elec_tabs):
        with tab:
            session = sessions[e]
            groups = session.raw_groups if raw else session.display_groups()
            if not groups:
                st.info("Aucun spectre EIS pour cette électrode.")
                continue
            labels = [g.label for g in groups]
            chosen = st.multiselect("Groupes affichés", labels, default=labels,
                                    key=f"eis_visu_groups_e{e}_{int(raw)}")
            shown = [g for g in groups if g.label in chosen]
            bare = None
            if session.bare_reference is not None and st.checkbox(
                    "Superposer l'électrode nue (référence, affichage seul)", value=True,
                    key=f"eis_visu_bare_e{e}"):
                bare = session.bare_reference
            if not shown:
                st.info("Sélectionnez au moins un groupe.")
                continue
            if raw and any(g.mean is None for g in shown):
                st.caption("Certains groupes bruts n'ont pas de moyenne (grilles de fréquences "
                           "différentes entre réplicats) : leurs réplicats sont tracés seuls.")
            col_n, col_b = st.columns(2)
            with col_n:
                st.plotly_chart(
                    nyquist_replicates_figure(shown, bare=bare, show_replicates=show_reps,
                                              title=f"Nyquist — Électrode {e}"),
                    width='stretch', key=f"eis_visu_nyq_e{e}_{int(raw)}")
            with col_b:
                st.plotly_chart(
                    bode_figure(shown, bare=bare, show_replicates=show_reps,
                                title=f"Bode — Électrode {e}"),
                    width='stretch', key=f"eis_visu_bode_e{e}_{int(raw)}")


# ─────────────────────────────────────────────────────────────────────────────
# EIS — onglet 2 : Measurement model & fit Orazem
# ─────────────────────────────────────────────────────────────────────────────

def _group_recap_rows(session) -> tuple:
    """([lignes du récapitulatif de l'électrode], [libellés des groupes à examiner])."""
    rows, flagged = [], []
    for label, _mean_sp, reps, an in session.iter_groups():
        if an is None:
            rows.append({"Groupe": label, "Réplicats": len(reps), "Verdict KK": "—",
                         "Fit Orazem": "aucune analyse", "Alertes": 0})
            continue
        state = kk_verdict_state(an)
        fit_rows = fit_diagnostic_rows(an)
        sev = worst_severity(fit_rows)
        if fit_rows:
            fit_txt = {SEVERITY_OK: "🟢 fiable", SEVERITY_WARNING: "🟠 à vérifier",
                       SEVERITY_ERROR: "🔴 incertitudes non fiables"}[sev]
        else:
            fit_txt = "⛔ arrêté — aucun fit" if not an.ok else "—"
        n_alerts = len(fit_alert_lines(an))
        if state != KK_CONFORM or sev in (SEVERITY_WARNING, SEVERITY_ERROR) or not an.ok:
            flagged.append(label)
        k_mean = next((r["n_voigt"] for r in kk_rows(an) if r["kind"] == "moyenne"), None)
        rows.append({"Groupe": label, "Réplicats": len(reps), "Verdict KK": _kk_icon(state),
                     "Éléments de Voigt (moyenne)": "—" if k_mean is None else k_mean,
                     "Fit Orazem": fit_txt, "Alertes": n_alerts})
    return rows, flagged


def _render_kk_section(an) -> None:
    """Measurement model et verdict KK d'un groupe — lus AVANT le fit."""
    vr = an.validation
    if vr is None:
        st.info("Aucun résultat de validation pour ce groupe.")
        return
    state = kk_verdict_state(an)
    if state == KK_CONFORM:
        st.success(f"✅ **Conforme Kramers-Kronig** — {vr.kk_message}")
    elif state == KK_NONCONFORM:
        st.error(f"❌ **Non conforme Kramers-Kronig** — "
                 f"{vr.error_structure_message or vr.kk_message or an.message or ''}")
    else:
        st.info(f"❔ **Verdict indéterminé** — {vr.error_structure_message or vr.kk_message}")

    table = [{
        "Spectre": ("Moyenne" if r["kind"] == "moyenne" else r["spectrum"]),
        "Méthode": r["method"],
        "Éléments de Voigt retenus": r["n_voigt"],
        "χ²ᵣ (ajustement de Im)": _fmt(r["chi2_reduced_im"], ".3g"),
        "Points hors ±2σ / tolérés": ("—" if r["n_outside"] is None
                                     else f"{r['n_outside']} / {r['n_allowed']}"),
        "Verdict": _kk_icon(KK_CONFORM if r["conform"] else
                            (KK_NONCONFORM if r["conform"] is False else "")),
    } for r in kk_rows(an)]
    st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')

    st.plotly_chart(residuals_figure(vr, theme_mode="light"), width='stretch',
                    key=f"kk_res_{vr.label}")
    for kk in vr.replicates:
        if kk.warning and kk.method != "lin_kk":     # Lin-KK : « indéterminé », déjà dit plus haut
            st.warning(f"**{kk.label}** : {kk.warning}")
    if vr.drift_warning:
        st.error(f"**Drift inter-réplicats** : {vr.drift_warning}")
    if vr.f_min_common and vr.f_max_common < float("inf"):
        st.caption(f"Plage KK-valide commune : **{vr.f_min_common:.2f} Hz** → "
                   f"**{vr.f_max_common:.2f} Hz**")
    mm = vr.measurement_model
    es = mm.error_structure if mm is not None else None
    if es is not None:
        st.caption(
            f"Structure d'erreur caractérisée sur CE groupe ({es.n_replicates} réplicats, "
            f"ν_σ = {es.dof}) : σ = α|Zj| + β|Zr − R_sol| + γ|Z|² + δ — α = {es.alpha:.3g}, "
            f"β = {es.beta:.3g}, γ = {es.gamma:.3g} Ω⁻¹, δ = {es.delta:.3g} Ω"
            + ("" if es.equal_re_im else " (σ_r ≠ σ_j : deux structures)")
        )


def _render_reliability_banner(rows: list) -> None:
    """Bandeau ROUGE quand κ dépasse le seuil ou qu'un paramètre n'est pas identifiable :
    les incertitudes affichées plus bas ne sont alors pas fiables."""
    lines = []
    for r in rows:
        if r["non_identifiable"]:
            lines.append(f"- **{r['spectrum']}** : paramètre(s) non identifiable(s) par les données — "
                         f"{', '.join(r['non_identifiable'])} (écart-type infini).")
        if r["kappa_exceeds"]:
            lines.append(f"- **{r['spectrum']}** : conditionnement κ = {_fmt(r['condition_number'], '.2e')} "
                         f"> seuil {CONDITION_NUMBER_WARN:.1e} (paramètres fortement corrélés).")
    if lines:
        st.error("⛔ **Incertitudes du fit non fiables — ne pas les exploiter telles quelles**\n\n"
                 + "\n".join(lines))


def _fit_diagnostics_table(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        if r["diagnostics_available"]:
            kappa = r["condition_number"]
            kappa_txt = _fmt(kappa, ".2e") + (" ⚠️ > seuil" if r["kappa_exceeds"] else "")
            rank_txt = f"{r['rank']} / {r['n_params']}"
            non_id = ", ".join(r["non_identifiable"]) or "aucun"
            one_sided = ", ".join(r["one_sided"]) or "aucune"
            bounds = ", ".join(f"{p} ({side})" for p, side, _b in r["active_bounds"]) or "aucune"
            starts = f"{r['n_converged']} / {r['n_starts']}"
        else:
            kappa_txt = rank_txt = non_id = one_sided = bounds = starts = "indisponible"
        out.append({
            "Spectre": r["spectrum"],
            "": _SEVERITY_ICON[r["severity"]],
            "Convergé": "✅" if r["converged"] else "❌",
            f"{r['target_param']} ± σ intra": f"{_fmt(r['target_value'])} ± {_fmt(r['target_std'], '.2g')}",
            "κ (jacobienne équilibrée)": kappa_txt,
            "Rang / P": rank_txt,
            "Non identifiables": non_id,
            "Dérivée unilatérale": one_sided,
            "Bornes actives": bounds,
            "χ²ᵣ": _fmt(r["chi2_reduced"], ".3g"),
            "χ²ᵣ attendu (2σ)": f"[{_fmt(r['chi2_ci_low'], '.2f')} ; {_fmt(r['chi2_ci_high'], '.2f')}]",
            "χ²ᵣ dans l'intervalle": _tick(r["chi2_in_ci"], na="—"),
            "Départs convergés": starts,
        })
    return pd.DataFrame(out)


def _param_table(og) -> tuple:
    """(tableau des paramètres, positions des lignes du paramètre cible)."""
    rows = param_rows(og)
    out = []
    for r in rows:
        remarks = []
        if r["one_sided"]:
            remarks.append("incertitude d'un seul côté (asymétrie numérique)")
        if r["active_bound"]:
            remarks.append(f"borne {r['active_bound']} active")
        out.append({
            "Paramètre": ("★ " if r["is_target"] else "") + r["param"],
            "Moyenne des réplicats": _fmt(r["mean"]),
            "± intra-fit (√v̄)": _fmt(r["std_within"], ".3g"),
            "± inter-réplicats (s)": _fmt(r["std_between"], ".3g", "n < 2"),
            "± incertitude de la moyenne": _fmt(r["sem"], ".3g"),
            "p de Cochran": _fmt(r["cochran_p"], ".2g"),
            "Fit du spectre moyen": _fmt(r["mean_fit_value"]),
            "± (1σ)": _fmt(r["mean_fit_std"], ".3g"),
            "Identifiable": ("—" if r["identifiable"] is None else ("oui" if r["identifiable"] else "❌ non")),
            "Remarques": " ; ".join(remarks),
        })
    return pd.DataFrame(out), [i for i, r in enumerate(rows) if r["is_target"]]


def _render_fit_section(an, mean_sp, reps, key: str) -> None:
    """Fit Orazem d'un groupe : fiabilité d'abord (bandeau, alertes), puis cible, courbe,
    paramètres et diagnostics — tout visible, rien dans un expander replié."""
    og = an.orazem
    rows = fit_diagnostic_rows(an)

    _render_reliability_banner(rows)
    for lbl, msg in fit_alert_lines(an):
        st.warning(f"**{lbl}** : {msg}")

    t = og.target
    st.markdown(f"**★ Paramètre cible : {og.target_param}** (valeur de calibration recommandée = "
                "moyenne des réplicats ± incertitude de la moyenne)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{og.target_param} — moyenne (n = {t.n})", f"{_fmt(t.mean)} ± {_fmt(t.sem, '.2g')}")
    c2.metric("Inter-réplicats (s)", _fmt(t.std_between, ".3g", "n < 2"))
    c3.metric("Intra-fit (√v̄)", _fmt(t.std_within, ".3g", "non calculé"))
    c4.metric("Incertitude de la moyenne", _fmt(t.sem, ".3g"))
    c5.metric("Fit du spectre moyen",
              f"{_fmt(og.mean_fit.target_value)} ± {_fmt(og.mean_fit.target_std, '.2g')}")
    if t.n_excluded:
        st.warning(f"{t.n_excluded} réplicat(s) non convergé(s) écarté(s) de la moyenne.")
    if not is_missing(t.q_pvalue):
        st.caption(f"Q de Cochran : p = {t.q_pvalue:.2g} (p < 0,05 : dispersion inter-réplicats "
                   "supérieure à l'incertitude intra-fit).")

    # Courbe du circuit ajusté
    model = og.mean_fit.model_name
    spectra = [("Spectre moyen", mean_sp)] + [(f"Réplicat {i + 1} ({sp.label})", sp)
                                              for i, sp in enumerate(reps)]
    sel = st.selectbox("Spectre affiché", [lbl for lbl, _ in spectra], key=f"fit_spec_{key}")
    sp = dict(spectra)[sel]
    fr = sp.fit_results.get(model)
    if fr is None or len(fr.Zfit_re) != len(sp.f):
        st.info("Pas de courbe ajustée pour ce spectre.")
    else:
        col_n, col_r = st.columns(2)
        with col_n:
            st.plotly_chart(fit_nyquist_figure(sp, fr, title=f"Nyquist — {sel}"),
                            width='stretch', key=f"fit_nyq_{key}")
        with col_r:
            st.plotly_chart(fit_residuals_figure(sp, fr, title=f"Résidus — {sel}"),
                            width='stretch', key=f"fit_res_{key}")

    # Paramètres et leurs incertitudes (intra ET inter-réplicats)
    st.markdown("**Paramètres ajustés** — intra-fit (bruit d'UN fit) et inter-réplicats "
                "(dispersion réelle entre mesures)")
    table, target_idx = _param_table(og)
    styled = table.style.apply(
        lambda r: ["background-color: rgba(250, 204, 21, 0.28); font-weight: 600"
                   if r.name in target_idx else "" for _ in r], axis=1)
    st.dataframe(styled, hide_index=True, width='stretch')

    # Diagnostics de chaque fit
    st.markdown("**Diagnostics de chaque fit**")
    st.dataframe(_fit_diagnostics_table(rows), hide_index=True, width='stretch')
    st.caption(
        f"κ = conditionnement de la jacobienne équilibrée (seuil d'alerte {CONDITION_NUMBER_WARN:.1e} = "
        "1/√ε machine). Rang < P : paramètre non identifiable. 🟢 fiable · 🟠 borne active, dérivée "
        "unilatérale ou χ²ᵣ hors intervalle · 🔴 non convergé, non identifiable ou κ au-delà du seuil."
    )


def _render_model_fit_group(an, mean_sp, reps, key: str) -> None:
    if an is None:
        st.info("Aucune analyse pour ce groupe.")
        return
    st.markdown("##### 1 · Measurement model et validation Kramers-Kronig")
    _render_kk_section(an)

    st.markdown("##### 2 · Fit Orazem du circuit")
    if an.orazem is not None:
        _render_fit_section(an, mean_sp, reps, key)
    elif an.status == GROUP_ERROR_STRUCTURE_UNAVAILABLE:
        extra = (f" Ce groupe ne compte que {len(reps)} réplicat(s) : la structure d'erreur en exige "
                 "au moins 3." if len(reps) < 3 else "")
        st.info(f"❔ **Aucun fit n'a été réalisé : verdict KK indéterminé.** La structure d'erreur de ce "
                f"groupe n'a pas pu être caractérisée, donc ni le verdict ni les poids du fit ne sont "
                f"disponibles — ce n'est pas une erreur logicielle.{extra}\n\n{an.message or ''}")
    else:
        st.error(f"Aucun fit n'a été réalisé pour ce groupe. {an.message or ''}")


def _render_model_fit_tab(sessions: dict) -> None:
    """Onglet 2 — Measurement model & fit Orazem, par électrode puis par groupe."""
    st.subheader("Measurement model & fit Orazem")
    st.caption(
        "Ordre de lecture : (1) le measurement model et le verdict Kramers-Kronig, calculés sur les "
        "réplicats bruts AVANT le fit ; (2) le fit Orazem du circuit, pondéré par la structure "
        "d'erreur du measurement model. **Intra-fit** : bruit d'UN ajustement. **Inter-réplicats** : "
        "dispersion entre réplicats (bruit ET variabilité propre). **Incertitude de la moyenne** : "
        "√(max(s², v̄)/n)."
    )
    electrodes = sorted(sessions.keys())
    if not electrodes:
        st.info("Lancez une analyse pour voir les résultats.")
        return
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
    for e, tab in zip(electrodes, elec_tabs):
        with tab:
            session = sessions[e]
            rows, flagged = _group_recap_rows(session)
            if not rows:
                st.info("Aucun groupe pour cette électrode.")
                continue
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            if flagged:
                st.warning("Groupes à examiner : " + ", ".join(f"**{g}**" for g in flagged) + ".")
            picked = _pick_group(session, e, "mm_group", flagged)
            label, mean_sp, reps, an = picked
            st.markdown(f"#### {label}")
            _render_model_fit_group(an, mean_sp, reps, key=f"e{e}_{label}")


# ─────────────────────────────────────────────────────────────────────────────
# EIS — onglet 3 : DRT
# ─────────────────────────────────────────────────────────────────────────────

#: Rappel affiché en permanence (drt/VALIDATION_REGLAGES.md §4, §6 et §7 : l'IC 95 % de Rp
#: ne contient la vraie valeur dans aucun des 18 essais HMC ; celui de Rct la contient 8 fois
#: sur 18, la grandeur extraite ayant un biais de méthode d'environ +2 % sur Randles).
_DRT_IC_NOTE = (
    "ℹ️ **Lire les intervalles** — l'IC de Rp n'est pas fiable (biais vers le haut connu) ; celui de "
    "Rct décrit la précision de l'estimation, mais pas son léger biais de méthode (≈ +2 % sur un "
    "circuit de Randles)."
)

_RCT_SOURCE_TXT = {"rp_fallback": "⚠️ repli sur Rp (aire totale ≠ arc de transfert)",
                   "peak_single": "pic unique"}


def _drt_diagnostics_table(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        if r["state"] != DRT_OK:
            out.append({"Spectre": r["spectrum"], "État": f"{'❌' if r['state'] == DRT_FAILED else '—'} "
                                                          f"{r['state']}"})
            continue
        hmc = r["mode"] == "sample"
        na = "n/a (MAP)"
        out.append({
            "Spectre": r["spectrum"],
            "Mode": "HMC (sample)" if hmc else "MAP (optimize)",
            "R̂ max": na if not hmc else f"{_fmt(r['rhat_max'], '.4f')} {_tick(r['rhat_ok'], '')}".strip(),
            "Divergences": na if not hmc else f"{_fmt(r['divergences'], 'd')} {_tick(r['divergences_ok'], '')}".strip(),
            "ESS bulk min": na if not hmc else f"{_fmt(r['ess_bulk_min'], '.0f')} {_tick(r['ess_ok'], '')}".strip(),
            "ESS tail min": na if not hmc else f"{_fmt(r['ess_tail_min'], '.0f')} {_tick(r['ess_ok'], '')}".strip(),
            "E-BFMI min": na if not hmc else _fmt(r["ebfmi_min"], ".2f"),
            "Rct (Ω)": _fmt(r["rct"]),
            "± a posteriori (1σ)": _fmt(r["rct_std"], ".2g", "non calculé (MAP)" if not hmc else "—"),
            "IC 95 % de Rct": (f"[{_fmt(r['rct_ci_low'], '.4g')} ; {_fmt(r['rct_ci_high'], '.4g')}]"
                               if hmc and not is_missing(r["rct_ci_low"]) else na if not hmc else "—"),
            "Origine de Rct": _RCT_SOURCE_TXT.get(r["rct_source"], "pic pénultième (Bissessur)"),
            "Rp (IC non fiable)": (f"{_fmt(r['rp'])} [{_fmt(r['rp_ci_low'], '.4g')} ; "
                                   f"{_fmt(r['rp_ci_high'], '.4g')}]" if hmc else _fmt(r["rp"])),
            "Convergé": "✅" if r["converged"] else "❌",
        })
    return pd.DataFrame(out).fillna("")


def _render_drt_spectrum_panel(r: dict) -> None:
    """Diagnostics HMC et Rct d'UN spectre — toujours affichés, y compris sans alerte."""
    hmc = r["mode"] == "sample"
    na = "n/a (MAP)"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R̂ max", na if not hmc else _fmt(r["rhat_max"], ".4f"),
              help=f"Seuil : ≤ {drt_diag.RHAT_MAX}")
    c2.metric("Divergences", na if not hmc else _fmt(r["divergences"], "d"),
              help=f"Seuil : ≤ {drt_diag.MAX_DIVERGENCES}")
    c3.metric("ESS bulk min", na if not hmc else _fmt(r["ess_bulk_min"], ".0f"),
              help=f"Seuil : ≥ {_fmt(r['ess_threshold'], '.0f')} ({drt_diag.ESS_MIN_PER_CHAIN:.0f} par chaîne)")
    c4.metric("ESS tail min", na if not hmc else _fmt(r["ess_tail_min"], ".0f"),
              help=f"Seuil : ≥ {_fmt(r['ess_threshold'], '.0f')}")
    c5.metric("Rct (Ω)", _fmt(r["rct"]),
              delta=None if not hmc or is_missing(r["rct_ci_low"])
              else f"IC 95 % [{_fmt(r['rct_ci_low'], '.4g')} ; {_fmt(r['rct_ci_high'], '.4g')}]",
              delta_color="off")
    if not hmc:
        st.info("MAP : aucun diagnostic de convergence ni intervalle (recalculer en HMC ci-dessous).")
    elif not r["alerts"]:
        st.success("✅ Aucune alerte : R̂, divergences et ESS sont dans les seuils.")
    for a in r["alerts"]:
        st.warning(a)
    if not r["converged"]:
        st.error("⛔ DRT NON convergée : ne pas exploiter ce Rct.")


def _render_drt_group(an, mean_sp, reps, key: str, config: dict, session, drt_ok: bool) -> None:
    rows = drt_diagnostic_rows(an, mean_sp, reps)
    if an is not None and not an.ok:
        st.info(f"⛔ Groupe arrêté : aucune DRT n'a été calculée. {an.message or ''}")
        return
    for r in rows:
        if r["state"] == DRT_FAILED:
            st.error(f"❌ **{r['spectrum']}** — DRT non calculée : {r['failure']} Le pipeline a "
                     "continué sans DRT pour ce spectre.")
        elif r["state"] == DRT_ABSENT:
            st.info(f"**{r['spectrum']}** — DRT non calculée (désactivée ou moteur indisponible "
                    "au moment de l'analyse).")

    # Diagnostics de TOUS les spectres du groupe : toujours visibles.
    st.markdown("**Diagnostics HMC et Rct de chaque spectre**")
    if any(r["state"] == DRT_OK for r in rows):
        st.dataframe(_drt_diagnostics_table(rows), hide_index=True, width='stretch')
        st.caption(f"Seuils : R̂ ≤ {drt_diag.RHAT_MAX} · divergences ≤ {drt_diag.MAX_DIVERGENCES} · "
                   f"ESS ≥ {drt_diag.ESS_MIN_PER_CHAIN:.0f} par chaîne (bulk ET tail). ✅ dans le seuil · "
                   "⚠️ hors seuil · n/a : mode MAP, pas de diagnostic.")

    drt_by_label = {r["spectrum"]: r for r in rows}
    items = [(f"Réplicat {i + 1}", sp.fit_results.get(DRT_MODEL_NAME)) for i, sp in enumerate(reps)]
    items = [(lbl, fr) for lbl, fr in items if fr is not None]
    mean_fr = mean_sp.fit_results.get(DRT_MODEL_NAME)
    envelope = drt_replicate_envelope(reps)

    views = (["Vue agrégée"] if items else []) + [lbl for lbl, _ in items] \
        + (["Spectre moyen"] if mean_fr is not None else [])
    if views:
        view = st.radio("Vue", views, horizontal=True, key=f"drt_view_{key}")
        if view == "Vue agrégée":
            if envelope is None:
                st.info("Variabilité inter-réplicats indisponible : au moins 2 réplicats avec une DRT "
                        "sont nécessaires. Les DRT des réplicats sont superposées.")
                st.plotly_chart(drt_figure(items, title="DRT des réplicats"), width='stretch',
                                key=f"drt_fig_{key}")
            else:
                st.plotly_chart(
                    drt_aggregate_figure(envelope, items,
                                         mean_item=("Spectre moyen", mean_fr) if mean_fr else None,
                                         title="DRT agrégée — moyenne et variabilité inter-réplicats"),
                    width='stretch', key=f"drt_fig_{key}")
                st.caption("Bande large = variabilité EXPÉRIMENTALE entre réplicats (min–max). La bande "
                           "fine de crédibilité HMC, propre à chaque réplicat, se lit sur la vue de "
                           "chaque réplicat.")
            d = an.drt_target if an is not None else None
            if d is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Rct DRT — moyenne (n = {d.n})", _fmt(d.mean))
                c2.metric("Inter-réplicats (s)", _fmt(d.std_between, ".3g", "n < 2"))
                c3.metric("A posteriori (√v̄)", _fmt(d.std_within, ".3g", "non calculé (MAP)"))
                c4.metric("Incertitude de la moyenne", _fmt(d.sem, ".3g"))
                if d.n_excluded:
                    st.warning(f"{d.n_excluded} réplicat(s) à DRT non convergée ou absente écarté(s) de "
                               "la moyenne de Rct.")
            for w in (an.warnings if an is not None else []):
                if w.startswith("Rct DRT agrégé sur des"):
                    st.warning(w)
        else:
            fr = mean_fr if view == "Spectre moyen" else dict(items)[view]
            row = drt_by_label["Moyenne" if view == "Spectre moyen"
                               else next(k for k in drt_by_label if k.startswith(view + " "))]
            st.plotly_chart(drt_figure([(view, fr)], title=f"DRT — {view}"), width='stretch',
                            key=f"drt_fig_{key}")
            _render_drt_spectrum_panel(row)

    # Recalcul bayésien d'un spectre du groupe
    st.markdown("**Recalcul bayésien (HMC) — 2 à 5 minutes par spectre**")
    targets = {f"Réplicat {i + 1} ({sp.label})": sp for i, sp in enumerate(reps)}
    targets["Spectre moyen"] = mean_sp
    c1, c2 = st.columns([3, 1])
    sel = c1.selectbox("Spectre à recalculer", list(targets.keys()), key=f"drt_recalc_sel_{key}")
    if c2.button("🎲 Recalculer en HMC", key=f"drt_recalc_btn_{key}", disabled=not drt_ok,
                 help="Échantillonnage HMC ; fournit R̂, divergences, ESS et intervalles. Le résultat persiste."):
        with st.spinner(f"Échantillonnage HMC de « {sel} »… (plusieurs minutes)"):
            try:
                recompute_drt(session, targets[sel], config, mode="sample")
            except (ValueError, RuntimeError) as exc:   # spectre invalide / échec de CmdStan
                st.error(f"Échec du recalcul bayésien : {exc}")
            else:
                st.success(f"DRT bayésienne calculée pour « {sel} ».")
                st.rerun()


def _render_drt_tab(sessions: dict) -> None:
    """Onglet 3 — DRT par réplicat et agrégée, diagnostics HMC toujours visibles."""
    st.subheader("Distribution des temps de relaxation (DRT)")

    drt_ok, why = drt_engine.engine_available()
    if not drt_ok:
        st.error(
            f"Moteur DRT indisponible : {why}. Installez l'extra DRT "
            "(`pip install -r requirements-drt.txt`) et la toolchain CmdStan : relancez "
            "**launch.bat** (ou `python setup_drt_bayesien.py --ensure`). Les résultats DRT déjà "
            "calculés restent affichés ; le recalcul est désactivé."
        )
    version_warning = st.session_state.get("drt_version_warning")
    if version_warning:
        st.warning(f"⚠️ {version_warning}")
    st.info(_DRT_IC_NOTE)

    electrodes = sorted(sessions.keys())
    if not electrodes:
        st.info("Lancez une analyse pour voir les résultats.")
        return
    if all(sessions[e].drt_mode is None for e in electrodes):
        st.info("La DRT n'a pas été lancée par cette analyse (désactivée dans ⚙️, ou moteur "
                "indisponible au moment de l'analyse). Un recalcul HMC reste possible spectre par spectre.")
    config = st.session_state.get("eis_config", {})
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
    for e, tab in zip(electrodes, elec_tabs):
        with tab:
            session = sessions[e]
            flagged = [lbl for lbl, mean_sp, reps, an in session.iter_groups()
                       if an is None or not an.ok or any(r["state"] != DRT_OK
                                                         for r in drt_diagnostic_rows(an, mean_sp, reps))]
            picked = _pick_group(session, e, "drt_group", flagged)
            if picked is None:
                st.info("Aucun groupe pour cette électrode.")
                continue
            label, mean_sp, reps, an = picked
            st.markdown(f"#### {label}")
            _render_drt_group(an, mean_sp, reps, key=f"e{e}_{label}", config=config, session=session,
                              drt_ok=drt_ok)


def render_eis_tabs(sessions: dict, normalized: dict = None) -> None:
    """Les trois onglets d'analyse EIS : Visualisation / Measurement model & fit Orazem / DRT.

    Args:
        sessions: {electrode_index: EISSession} — les sessions RÉELLES de
            ``st.session_state['eis_sessions']`` (un recalcul DRT y écrit). Le verdict KK et
            la structure d'erreur sont lus dans ``GroupAnalysis.validation`` de chaque groupe.
        normalized: spectres normalisés par concentration (``A_eis._build_normalized_session``),
            affichés comme un onglet de plus dans « Visualisation » ; None = pas de vue normalisée.
    """
    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Visualisation", "2️⃣ Measurement model & fit Orazem", "3️⃣ DRT",
    ])
    with tab1:
        _render_visualisation_tab(sessions, normalized)
    with tab2:
        _render_model_fit_tab(sessions)
    with tab3:
        _render_drt_tab(sessions)


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
