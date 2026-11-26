"""Tableaux de résultats prêts à afficher / exporter (v3) — aucun import Streamlit.

Source UNIQUE des tableaux de la page Analyse, des CSV de l'archive et des rapports PDF/HTML :
ils lisent tous ces fonctions, jamais les enregistrements bruts à leur façon. Rien n'est
recalculé ici : chaque valeur est lue dans ``core.results``.
"""

from __future__ import annotations

import math

from core.experiment import format_concentration, step_sort_key
from core.results import KK_NONCONFORM, WEIGHTING_LABELS, AnalysisResults


def _sort_key(rec) -> tuple:
    return (rec.channel_id, rec.electrode, step_sort_key(rec.step), rec.replicate is None, rec.replicate or 0)


def _kk_txt(conform) -> str:
    return "—" if conform is None else ("conforme" if conform else "⚠️ non conforme")


def param_names(results: AnalysisResults) -> list:
    """Paramètres du circuit, dans l'ordre de sa description."""
    return list((results.eis.circuit or {}).get("parameters") or {})


def rct_rows(results: AnalysisResults, include_means: bool = True) -> list:
    """Onglet « Rct (fit) » : une ligne par spectre — paramètres du fit, χ²ᵣ, Rct_DRT.

    Colonnes : canal, électrode, étape, concentration, réplicat (« moyenne » pour le spectre
    moyen), chaque paramètre du circuit, χ²ᵣ, convergé, pondération, verdict KK du spectre,
    Rct_DRT (aire du pic principal) et Rct DRT du moteur (pic pénultième, contrôle).
    """
    names = param_names(results)
    rows = []
    for rec in sorted(results.eis.spectra.values(), key=_sort_key):
        if rec.is_mean and not include_means:
            continue
        row = {"Canal": rec.channel_id, "Électrode": rec.electrode, "Étape": rec.step,
               "[c] (mol/L)": rec.concentration, "Réplicat": rec.label}
        fit = rec.fit
        for p in names:
            row[p] = fit.params.get(p, math.nan) if fit else math.nan
        row["χ²ᵣ"] = fit.chi2_reduced if fit else math.nan
        row["Convergé"] = (fit.converged if fit else None)
        row["Pondération"] = (fit.weighting if fit else None)
        row["KK"] = _kk_txt(rec.kk_conform)
        row["Rct_DRT (pic principal, Ω)"] = rec.drt.rct_main if rec.drt else math.nan
        row["Rct DRT moteur (contrôle, Ω)"] = rec.drt.rct_engine if rec.drt else math.nan
        row["Remarques"] = " ; ".join(([rec.fit_error] if rec.fit_error else [])
                                      + (fit.warnings if fit else [])
                                      + ([f"DRT : {rec.drt_error}"] if rec.drt_error else []))
        rows.append(row)
    return rows


def group_rows(results: AnalysisResults) -> list:
    """Récapitulatif par groupe : statut, pondération, verdict KK, Rct moyen ± s."""
    target = (results.eis.circuit or {}).get("target_param", "Rct")
    rows = []
    for g in sorted(results.eis.groups.values(), key=lambda g: (g.channel_id, g.electrode, step_sort_key(g.step))):
        agg = g.aggregate.get(target) or {}
        rows.append({"Canal": g.channel_id, "Électrode": g.electrode, "Étape": g.step,
                     "Biomarqueur": g.biomarker, "[c]": format_concentration(g.concentration)
                     if g.concentration else "—", "Réplicats": g.n_replicates,
                     "Statut": {"ok": "✅ fit Orazem", "fallback": "🟠 repli", "failed": "❌ échec"}.get(g.status,
                                                                                                       g.status),
                     "Pondération": WEIGHTING_LABELS.get(g.weighting, "—"),
                     "KK": ("⚠️ " if g.kk_state == KK_NONCONFORM else "") + g.kk_state,
                     f"{target} moyen (Ω)": agg.get("mean", math.nan),
                     f"{target} s inter (Ω)": agg.get("std_between", math.nan),
                     "Message": g.message})
    return rows


def drt_rows(results: AnalysisResults) -> list:
    """Diagnostics DRT par spectre : mode, pic principal, Rct, contrôle moteur, HMC, erreurs."""
    rows = []
    for rec in sorted(results.eis.spectra.values(), key=_sort_key):
        d = rec.drt
        row = {"Canal": rec.channel_id, "Électrode": rec.electrode, "Étape": rec.step, "Réplicat": rec.label,
               "Spectre": rec.key}
        if d is None:
            row.update({"État": "❌ " + rec.drt_error if rec.drt_error else "— non calculée"})
            rows.append(row)
            continue
        diag = d.diagnostics or {}
        row.update({
            "État": "✅" if d.peak.ok else "⚠️ pic absent", "Mode": d.mode,
            "τ pic (s)": d.peak.tau_peak, "Rct_DRT pic principal (Ω)": d.rct_main,
            "Rct moteur (pénultième, Ω)": d.rct_engine, "Rp (Ω)": d.rp,
            "R̂ max": diag.get("rhat_max"), "Divergences": diag.get("divergences"),
            "ESS bulk min": diag.get("ess_bulk_min"), "Durée (s)": diag.get("duration_s"),
            "Remarques": " ; ".join(([d.peak.message] if d.peak.message else []) + list(d.warnings)),
        })
        rows.append(row)
    return rows


def cv_rows(results: AnalysisResults) -> list:
    """Courants de pic anodique, I_probe et ΔI/I par courbe CV."""
    out = []
    for f in sorted(results.cv.values(), key=_sort_key):
        out.append({"Canal": f.channel_id, "Électrode": f.electrode, "Étape": f.step, "Réplicat": f.replicate,
                    "[c] (mol/L)": f.concentration, "E pic (V)": f.e_peak, "I pic (µA)": f.i_peak * 1e6,
                    "I_probe (µA)": f.i_probe * 1e6, "ΔI/I": f.delta_i_norm, "Remarque": f.message})
    return out


def failure_rows(results: AnalysisResults) -> list:
    stage = {"fit": "fit du circuit", "drt": "DRT", "cv": "CV", "tâche": "tâche"}
    return [{"Élément": f.key, "Étape du calcul": stage.get(f.stage, f.stage), "Cause": f.message}
            for f in results.failures]


def calibration_level_rows(curves: list) -> list:
    """Points moyens des courbes d'étalonnage (affichage et CSV)."""
    rows = []
    for c in curves:
        for lv in c.levels:
            rows.append({"Courbe": c.label, "Biomarqueur": c.biomarker, "Grandeur": c.quantity, "Source": c.source,
                         "[c] (mol/L)": lv.concentration, "log10([c])": lv.log_c, "Moyenne": lv.mean,
                         "Écart-type (ddof=1)": lv.std, "n réplicats": lv.n, "Remarque": lv.note})
    return rows


def calibration_regression_rows(curves: list) -> list:
    """Paramètres de régression de chaque courbe (y = a·log10([c]) + b)."""
    return [{"Courbe": c.label, "Biomarqueur": c.biomarker, "Grandeur": c.quantity, "Source": c.source,
             "Niveaux": len(c.levels), "Pente a": c.slope, "Ordonnée b": c.intercept, "R²": c.r2,
             "p-value": c.pvalue, "Erreur-type pente": c.stderr, "Remarque": c.message} for c in curves]


def calibration_point_rows(curves: list) -> list:
    """Chaque réplicat (inclus ou exclu, avec la raison)."""
    rows = []
    for c in curves:
        for p in c.points:
            rows.append({"Courbe": c.label, "Canal": p.channel_id, "Électrode": p.electrode, "Étape": p.step,
                         "Réplicat": p.replicate, "[c] (mol/L)": p.concentration, "Valeur": p.value,
                         "Référence probe": p.reference, "Normalisé": p.y,
                         "Inclus": "oui" if p.included else "non", "Raison": p.reason,
                         "KK": "⚠️ non conforme" if p.kk_nonconform else ""})
    return rows
