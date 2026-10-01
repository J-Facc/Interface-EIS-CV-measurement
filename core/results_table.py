"""Tables de résultats PAR RÉPLICAT et PAR GROUPE (UI et export partagent ces lignes).

Logique pure (aucun import Streamlit) : l'onglet « Résultats par groupe »
(``ui/tabs.py``) et l'export CSV (``exports/exporter.py``) affichent EXACTEMENT les
mêmes chiffres.

Deux niveaux d'incertitude, montrés CÔTE À CÔTE et jamais confondus :

* **intra-fit** — écart-type d'UN fit (covariance de la jacobienne pondérée par σ du
  measurement model pour le fit Orazem ; écart-type a posteriori HMC pour la DRT,
  NaN en MAP car non calculé) ;
* **inter-réplicats** — dispersion des valeurs entre réplicats (écart-type ddof = 1),
  qui contient le bruit ET la variabilité propre aux réplicats (dérive, remontage…).

Pour la DRT, les diagnostics de convergence HMC (R-hat max, divergences, ESS min) sont
reportés pour CHAQUE spectre — ``None`` en MAP ('optimize'), où ils n'existent pas.
"""

from __future__ import annotations

import math
from typing import Optional

from drt.engine import MODEL_NAME as DRT_MODEL_NAME

_NAN = float("nan")


def _f(x) -> float:
    """float, NaN pour None / non numérique (cellule « non calculée »)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def drt_hmc_summary(fr) -> dict:
    """Diagnostics de convergence d'un FitResult DRT (``drt/diagnostics.py``).

    Returns:
        ``mode``, ``converged``, ``rhat_max``, ``divergences``, ``ess_bulk_min``,
        ``ess_tail_min`` (None si non applicable : mode 'optimize'), ``alerts`` (codes
        des alertes, convergence ET qualité).
    """
    diag = getattr(fr, "drt_diagnostics", None) or {}
    sampler = diag.get("sampler") or {}
    hmc = getattr(fr, "drt_mode", None) == "sample"
    return {
        "mode": getattr(fr, "drt_mode", None),
        "converged": bool(getattr(fr, "converged", False)),
        "rhat_max": sampler.get("rhat_max") if hmc else None,
        "divergences": sampler.get("divergences") if hmc else None,
        "ess_bulk_min": sampler.get("ess_bulk_min") if hmc else None,
        "ess_tail_min": sampler.get("ess_tail_min") if hmc else None,
        "alerts": [a.get("code") for a in diag.get("alerts", []) if isinstance(a, dict)],
    }


def _spectrum_row(electrode, group: str, kind: str, sp, circuit_model: Optional[str]) -> dict:
    row = {"electrode": electrode, "group": group, "spectrum": sp.label, "kind": kind}
    fr = sp.fit_results.get(circuit_model) if circuit_model else None
    if fr is not None:
        ci = fr.chi2_reduced_ci or (_NAN, _NAN)
        row.update({
            "target_param": fr.target_param,
            "fit_value": _f(fr.target_value),
            "fit_std_intra": _f(fr.target_std),
            "fit_chi2_reduced": _f(fr.chi2_reduced),
            "fit_chi2_ci_low": _f(ci[0]),
            "fit_chi2_ci_high": _f(ci[1]),
            "fit_converged": bool(fr.converged),
            "fit_warnings": " | ".join(fr.warnings or []),
        })
    dr = sp.fit_results.get(DRT_MODEL_NAME)
    if dr is not None:
        h = drt_hmc_summary(dr)
        row.update({
            "drt_mode": h["mode"],
            "drt_Rct": _f(dr.target_value),
            "drt_Rct_std_intra": _f(dr.target_std),
            "drt_converged": h["converged"],
            "drt_rhat_max": h["rhat_max"],
            "drt_divergences": h["divergences"],
            "drt_ess_bulk_min": h["ess_bulk_min"],
            "drt_ess_tail_min": h["ess_tail_min"],
            "drt_alerts": ", ".join(c for c in h["alerts"] if c),
        })
    return row


def _circuit_model(session) -> Optional[str]:
    """Clé ``fit_results`` du fit Orazem de cette session (None si aucun fit)."""
    for _lbl, _sp, _reps, an in session.iter_groups():
        if an is not None and an.orazem is not None:
            return an.orazem.mean_fit.model_name
    return None


def replicate_rows(sessions: dict) -> list:
    """Une ligne par spectre : chaque réplicat BRUT puis la moyenne, pour chaque groupe.

    Args:
        sessions: {électrode: EISSession}.
    """
    rows = []
    for e, session in sorted(sessions.items()):
        model = _circuit_model(session)
        for group, mean_sp, reps, _an in session.iter_groups():
            for sp in reps:
                rows.append(_spectrum_row(e, group, "réplicat", sp, model))
            rows.append(_spectrum_row(e, group, "moyenne", mean_sp, model))
    return rows


def group_rows(sessions: dict) -> list:
    """Une ligne par groupe : statut, verdict KK, et agrégats intra/inter-réplicats.

    Colonnes ``fit_*`` (fit Orazem, paramètre cible) :
      ``fit_mean`` moyenne des réplicats convergés ; ``fit_std_between`` écart-type
      inter-réplicats ; ``fit_std_within`` incertitude intra-fit typique √v̄ ;
      ``fit_sem`` incertitude retenue pour la moyenne √(max(s², v̄)/n) ;
      ``fit_cochran_p`` p du Q de Cochran (p < 0,05 : dispersion > intra-fit) ;
      ``fit_mean_spectrum`` / ``fit_mean_spectrum_std`` : fit du spectre MOYEN (σ/√n).
    Colonnes ``drt_*`` : mêmes agrégats sur le Rct DRT (``drt_std_within`` NaN en MAP).
    """
    rows = []
    for e, session in sorted(sessions.items()):
        for group, mean_sp, reps, an in session.iter_groups():
            row = {"electrode": e, "group": group, "n_replicates": len(reps)}
            if an is None:
                rows.append(row)
                continue
            vr = an.validation
            row.update({
                "status": an.status,
                "message": an.message or "",
                "kk_conform": getattr(vr, "all_valid", None),
                "kk_message": getattr(vr, "kk_message", None) or "",
            })
            og = an.orazem
            if og is not None:
                t = og.target
                row.update({
                    "target_param": og.target_param,
                    "fit_n_used": t.n,
                    "fit_n_excluded": t.n_excluded,
                    "fit_mean": _f(t.mean),
                    "fit_std_between": _f(t.std_between),
                    "fit_std_within": _f(t.std_within),
                    "fit_sem": _f(t.sem),
                    "fit_cochran_p": _f(t.q_pvalue),
                    "fit_mean_spectrum": _f(og.mean_fit.target_value),
                    "fit_mean_spectrum_std": _f(og.mean_fit.target_std),
                })
            d = an.drt_target
            if d is not None:
                modes = sorted({str(sp.fit_results[DRT_MODEL_NAME].drt_mode) for sp in reps
                                if DRT_MODEL_NAME in sp.fit_results})
                row.update({
                    "drt_mode": "/".join(modes),
                    "drt_n_used": d.n,
                    "drt_n_excluded": d.n_excluded,
                    "drt_mean": _f(d.mean),
                    "drt_std_between": _f(d.std_between),
                    "drt_std_within": _f(d.std_within),
                    "drt_sem": _f(d.sem),
                })
            row["drt_failures"] = len(an.drt_failures)
            row["warnings"] = " | ".join(an.warnings)
            rows.append(row)
    return rows


def is_missing(x) -> bool:
    """True pour None / NaN (cellule « non calculé »)."""
    return x is None or (isinstance(x, float) and math.isnan(x))
