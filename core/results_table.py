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

import numpy as np

from core.calibration import calibration_models, calibration_points, calibration_reference
from drt import diagnostics as drt_diag
from drt.engine import MODEL_NAME as DRT_MODEL_NAME
from fits.orazem_fit import CONDITION_NUMBER_WARN

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
        des alertes, convergence ET qualité), ``duration_s`` et ``computed_at`` (durée et
        date du calcul, rangées par ``core.pipeline`` dans ``drt_diagnostics`` ; None pour un
        résultat qui n'en porte pas).
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
        "duration_s": diag.get("duration_s"),
        "computed_at": diag.get("computed_at"),
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
            "drt_duration_s": h["duration_s"],
            "drt_computed_at": h["computed_at"],
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


# ─────────────────────────────────────────────────────────────────────────────
# Onglet « Measurement model & fit Orazem » : verdict KK, diagnostics, paramètres
# ─────────────────────────────────────────────────────────────────────────────
# Lignes prêtes à afficher, construites SANS Streamlit : l'interface ne fait que les
# présenter, et les tests les vérifient sans navigateur. Rien n'est recalculé ici :
# chaque valeur est lue dans ``GroupAnalysis`` / ``FitResult`` / ``AggregatedParameter``.

#: Verdicts Kramers-Kronig d'un groupe.
KK_CONFORM = "conforme"
KK_NONCONFORM = "non conforme"
KK_UNDETERMINED = "indéterminé"

#: Gravité d'un fit, du plus bénin au plus grave (``fit_diagnostic_row``).
SEVERITY_OK = "ok"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"


def kk_verdict_state(an) -> str:
    """Verdict KK du groupe : ``KK_CONFORM``, ``KK_NONCONFORM`` ou ``KK_UNDETERMINED``.

    « Indéterminé » = aucun niveau de bruit (structure d'erreur non caractérisable,
    typiquement moins de 3 réplicats) : le pipeline n'a alors rendu AUCUN verdict.
    """
    vr = getattr(an, "validation", None)
    valid = getattr(vr, "all_valid", None)
    if valid is None:
        return KK_UNDETERMINED
    return KK_CONFORM if valid else KK_NONCONFORM


def _kk_row(label: str, kind: str, kk) -> dict:
    v = kk.verdict
    return {
        "spectrum": label, "kind": kind, "method": "measurement model",
        "n_voigt": int(kk.model_from_imag.n_elements),
        "chi2_reduced_im": _f(kk.model_from_imag.chi2_reduced),
        "n_outside": int(v.n_outside), "n_allowed": int(v.n_allowed),
        "conform": v.conform, "message": v.message,
    }


def kk_rows(an) -> list:
    """Une ligne par réplicat puis la moyenne : éléments de Voigt retenus, points hors
    bande ±2σ / tolérés, verdict. Sans measurement model (groupe indéterminé), les
    résidus Lin-KK du prétraitement, INDICATIFS (``method`` = « Lin-KK », verdict None).
    """
    vr = getattr(an, "validation", None)
    if vr is None:
        return []
    mm = vr.measurement_model
    if mm is not None:
        rows = [_kk_row(lbl, "réplicat", kk) for lbl, kk in zip(mm.replicate_labels, mm.kk_replicates)]
        rows.append(_kk_row("Moyenne", "moyenne", mm.kk_mean))
        return rows
    return [{
        "spectrum": kk.label, "kind": "réplicat", "method": "Lin-KK (indicatif)",
        "n_voigt": int(kk.n_elements), "chi2_reduced_im": _NAN,
        "n_outside": None, "n_allowed": None, "conform": None,
        "message": "Lin-KK : aucun verdict sans niveau de bruit.",
    } for kk in vr.replicates]


def fit_diagnostic_row(label: str, kind: str, fr) -> dict:
    """Diagnostics d'UN fit Orazem (``FitResult.fit_diagnostics``), prêts à afficher.

    ``diagnostics_available`` est False si le fit ne porte pas ``fit_diagnostics`` :
    les champs concernés valent alors None, jamais une valeur inventée.
    ``kappa_exceeds`` : κ > ``fits.orazem_fit.CONDITION_NUMBER_WARN`` (inf compris : une
    jacobienne non décomposable est un cas d'incertitudes indisponibles).
    ``severity`` : erreur si non convergé, paramètre non identifiable ou κ au-delà du seuil
    (les incertitudes ne sont pas fiables) ; avertissement pour une borne active, une
    dérivée unilatérale ou un χ²ᵣ hors de l'intervalle attendu.
    """
    d = getattr(fr, "fit_diagnostics", None)
    ci = getattr(fr, "chi2_reduced_ci", None) or (_NAN, _NAN)
    chi2 = _f(fr.chi2_reduced)
    in_ci = None if (math.isnan(chi2) or math.isnan(_f(ci[0]))) else bool(ci[0] <= chi2 <= ci[1])
    row = {
        "spectrum": label, "kind": kind,
        "target_param": fr.target_param, "target_value": _f(fr.target_value),
        "target_std": _f(fr.target_std), "converged": bool(fr.converged),
        "chi2_reduced": chi2, "chi2_ci_low": _f(ci[0]), "chi2_ci_high": _f(ci[1]),
        "chi2_in_ci": in_ci, "n_params": len(fr.params),
        "diagnostics_available": d is not None,
        "condition_number": None, "rank": None, "non_identifiable": [],
        "one_sided": [], "active_bounds": [], "n_starts": None, "n_converged": None,
        "dof": None, "dof_sigma": None, "kappa_exceeds": False,
    }
    if d is not None:
        kappa = d.get("condition_number")
        row.update({
            "condition_number": None if kappa is None else float(kappa),
            "rank": d.get("rank"),
            "non_identifiable": [p for p, ok in (d.get("identifiable") or {}).items() if not ok],
            "one_sided": list(d.get("jacobian_one_sided") or []),
            "active_bounds": [(p, side, float(b)) for p, side, b in d.get("active_bounds") or []],
            "n_starts": d.get("n_starts"), "n_converged": d.get("n_converged"),
            "dof": d.get("dof"), "dof_sigma": d.get("dof_sigma"),
        })
        row["kappa_exceeds"] = bool(kappa is not None and not math.isnan(float(kappa))
                                    and float(kappa) > CONDITION_NUMBER_WARN)
    if not row["converged"] or row["non_identifiable"] or row["kappa_exceeds"]:
        row["severity"] = SEVERITY_ERROR
    elif row["active_bounds"] or row["one_sided"] or in_ci is False:
        row["severity"] = SEVERITY_WARNING
    else:
        row["severity"] = SEVERITY_OK
    return row


def fit_diagnostic_rows(an) -> list:
    """Diagnostics de chaque fit du groupe : réplicats, puis spectre moyen.

    Liste vide si le groupe n'a pas de fit (``an.orazem is None``).
    """
    og = getattr(an, "orazem", None)
    if og is None:
        return []
    labels = list(getattr(og.analysis, "replicate_labels", []) or [])
    rows = [fit_diagnostic_row(labels[i] if i < len(labels) else f"réplicat {i + 1}", "réplicat", fr)
            for i, fr in enumerate(og.replicate_fits)]
    rows.append(fit_diagnostic_row("Moyenne", "moyenne", og.mean_fit))
    return rows


def worst_severity(rows: list) -> Optional[str]:
    """Gravité maximale (``SEVERITY_*``) d'une liste de ``fit_diagnostic_row``, None si vide."""
    order = (SEVERITY_OK, SEVERITY_WARNING, SEVERITY_ERROR)
    found = [r["severity"] for r in rows]
    return max(found, key=order.index) if found else None


def fit_alert_lines(an) -> list:
    """Alertes de fit à montrer d'emblée : ``[(portée, message)]``.

    Alertes de chaque ``FitResult`` (non convergence, résidu excessif, paramètre en butée,
    χ²ᵣ hors intervalle, incertitudes indisponibles…) puis alertes du groupe, hors DRT
    (affichées dans leur propre onglet). Un message identique sur plusieurs spectres n'est
    rapporté qu'une fois, avec la liste de ces spectres comme ``portée`` (« Tous les
    spectres » s'il les concerne tous) ; les alertes du groupe ont pour portée « Groupe ».
    """
    og = getattr(an, "orazem", None)
    if og is None:
        return []
    labels = list(getattr(og.analysis, "replicate_labels", []) or [])
    fits = [(labels[i] if i < len(labels) else f"réplicat {i + 1}", fr)
            for i, fr in enumerate(og.replicate_fits)] + [("Moyenne", og.mean_fit)]
    by_message: dict = {}
    for lbl, fr in fits:
        msgs = list(fr.warnings or [])
        if not fr.converged and not any("convergé" in m for m in msgs):
            msgs.append("ajustement non convergé")
        for m in dict.fromkeys(msgs):
            by_message.setdefault(m, []).append(lbl)
    out = []
    for m, scope in by_message.items():
        out.append(("Tous les spectres" if len(scope) == len(fits) and len(fits) > 1
                    else ", ".join(scope), m))
    for m in dict.fromkeys(getattr(an, "warnings", []) or []):
        if not m.startswith("DRT de «") and not m.startswith("Rct DRT agrégé"):
            out.append(("Groupe", m))
    return out


def param_rows(og) -> list:
    """Un tableau de paramètres : moyenne des réplicats et ses deux incertitudes.

    Par paramètre : ``mean`` (réplicats convergés), ``std_within`` (intra-fit typique √v̄),
    ``std_between`` (inter-réplicats s, ddof = 1), ``sem`` (incertitude retenue de la
    moyenne), ``cochran_p``, puis le fit du spectre MOYEN (valeur, écart-type, identifiable,
    dérivée unilatérale, borne active). ``is_target`` marque le paramètre cible.
    """
    mean_fit = og.mean_fit
    d = mean_fit.fit_diagnostics or {}
    ident = d.get("identifiable") or {}
    one_sided = set(d.get("jacobian_one_sided") or [])
    bounds = {p: side for p, side, _b in d.get("active_bounds") or []}
    rows = []
    for p in og.param_names:
        a = og.aggregate[p]
        rows.append({
            "param": p, "is_target": p == og.target_param, "n": a.n, "n_excluded": a.n_excluded,
            "mean": _f(a.mean), "std_within": _f(a.std_within), "std_between": _f(a.std_between),
            "sem": _f(a.sem), "cochran_p": _f(a.q_pvalue),
            "mean_fit_value": _f(mean_fit.params.get(p)), "mean_fit_std": _f(mean_fit.params_std.get(p)),
            "identifiable": ident.get(p) if ident else None,
            "one_sided": p in one_sided, "active_bound": bounds.get(p),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Onglet « DRT » : diagnostics HMC systématiques, Rct avec IC, variabilité inter-réplicats
# ─────────────────────────────────────────────────────────────────────────────

#: États d'une DRT dans ``drt_diagnostic_rows``.
DRT_OK = "calculée"
DRT_FAILED = "échec"
DRT_ABSENT = "non calculée"


def drt_diagnostic_row(label: str, kind: str, fr, failure: Optional[str] = None) -> dict:
    """Une DRT (ou son absence) en une ligne : mode, diagnostics HMC, Rct et son IC.

    Sans ``fr``, ``state`` vaut ``DRT_FAILED`` (``failure`` = motif consigné par le
    pipeline, qui a continué sans DRT pour ce spectre) ou ``DRT_ABSENT`` (jamais demandée).
    Les diagnostics HMC (``rhat_max``, ``divergences``, ``ess_*``) valent None en MAP, où
    ils n'existent pas ; ``rhat_ok``/``divergences_ok``/``ess_ok`` les comparent aux seuils de
    ``drt.diagnostics`` (None si non applicables). Rp et son IC sont fournis à part : l'IC
    de Rp n'est pas calibré (drt/VALIDATION_REGLAGES.md §4 et §7).
    """
    row = {"spectrum": label, "kind": kind, "failure": failure}
    if fr is None:
        row["state"] = DRT_FAILED if failure else DRT_ABSENT
        return row
    h = drt_hmc_summary(fr)
    diag = getattr(fr, "drt_diagnostics", None) or {}
    sampler = diag.get("sampler") or {}
    chains = sampler.get("chains")
    hmc = h["mode"] == "sample"
    ess_min = drt_diag.ESS_MIN_PER_CHAIN * chains if chains else None
    rct_ci = diag.get("Rct_ci95") or (_NAN, _NAN)
    rp_ci = diag.get("Rp_ci95") or (_NAN, _NAN)
    rhat, div = h["rhat_max"], h["divergences"]
    ess_b, ess_t = h["ess_bulk_min"], h["ess_tail_min"]

    def _ok(cond, *vals):
        return None if (not hmc or any(is_missing(v) for v in vals)) else bool(cond)

    row.update({
        "state": DRT_OK, "mode": h["mode"], "converged": h["converged"],
        "rct": _f(fr.target_value), "rct_std": _f(fr.target_std),
        "rct_ci_low": _f(rct_ci[0]), "rct_ci_high": _f(rct_ci[1]),
        "rct_source": (fr.params or {}).get("rct_source"),
        "rp": _f((fr.params or {}).get("Rp")),
        "rp_ci_low": _f(rp_ci[0]), "rp_ci_high": _f(rp_ci[1]),
        "rhat_max": rhat, "divergences": div, "ess_bulk_min": ess_b, "ess_tail_min": ess_t,
        "chains": chains, "ess_threshold": ess_min,
        "rhat_ok": _ok(rhat is not None and rhat <= drt_diag.RHAT_MAX, rhat),
        "divergences_ok": _ok(div is not None and div <= drt_diag.MAX_DIVERGENCES, div),
        "ess_ok": _ok(ess_min is not None and ess_b is not None and ess_t is not None
                      and ess_b >= ess_min and ess_t >= ess_min, ess_b, ess_t, ess_min),
        "ebfmi_min": min(sampler["ebfmi_per_chain"]) if sampler.get("ebfmi_per_chain") else None,
        "alerts": list(fr.warnings or []),
    })
    return row


def drt_diagnostic_rows(an, mean_sp, reps) -> list:
    """``drt_diagnostic_row`` de chaque réplicat puis du spectre moyen d'un groupe.

    Le motif d'un échec vient de ``GroupAnalysis.drt_failures`` (clé : label du spectre).
    """
    failures = getattr(an, "drt_failures", None) or {}
    out = []
    for i, sp in enumerate(reps):
        out.append(drt_diagnostic_row(f"Réplicat {i + 1} ({sp.label})", "réplicat",
                                      sp.fit_results.get(DRT_MODEL_NAME), failures.get(sp.label)))
    out.append(drt_diagnostic_row("Moyenne", "moyenne", mean_sp.fit_results.get(DRT_MODEL_NAME),
                                  failures.get(mean_sp.label)))
    return out


def drt_replicate_envelope(replicates: list) -> Optional[dict]:
    """Variabilité INTER-réplicats de γ(τ) : moyenne et étendue (min–max) des DRT des réplicats.

    Distincte de la bande de crédibilité HMC d'UN réplicat (incertitude du fit, fine) :
    celle-ci dit à quel point les réplicats diffèrent entre eux (variabilité
    expérimentale). Les γ sont ramenés sur la grille τ du premier réplicat (interpolation
    linéaire en ln τ) si les grilles diffèrent, et seule la plage de τ commune est
    conservée ; une DRT sans γ(τ) est ignorée.

    Returns:
        ``{"tau", "mean", "lo", "hi", "n"}`` (γ en Ω), ou None si moins de 2 réplicats portent
        une DRT.
    """
    drts = [(np.asarray(fr.drt_tau, dtype=float), np.asarray(fr.drt_gamma, dtype=float))
            for fr in (sp.fit_results.get(DRT_MODEL_NAME) for sp in replicates)
            if fr is not None and fr.drt_tau is not None and fr.drt_gamma is not None
            and len(fr.drt_tau) > 0]
    if len(drts) < 2:
        return None
    tau0 = drts[0][0]
    lo_t = max(float(t.min()) for t, _g in drts)
    hi_t = min(float(t.max()) for t, _g in drts)
    keep = (tau0 >= lo_t) & (tau0 <= hi_t)
    if not np.any(keep):
        return None
    tau = tau0[keep]
    gam = []
    for t, g in drts:
        if t.shape == tau0.shape and np.allclose(t, tau0, rtol=1e-9):
            gam.append(g[keep])
        else:
            order = np.argsort(np.log(t))
            gam.append(np.interp(np.log(tau), np.log(t)[order], g[order]))
    G = np.vstack(gam)
    return {"tau": tau, "mean": G.mean(axis=0), "lo": G.min(axis=0), "hi": G.max(axis=0), "n": len(drts)}


# ─────────────────────────────────────────────────────────────────────────────
# Onglet « Calibration » : points retenus / exclus et leur raison
# ─────────────────────────────────────────────────────────────────────────────

def calibration_rows(session, strict: bool = False) -> list:
    """Une ligne par (modèle, groupe) : le probe (référence) puis chaque concentration > 0.

    Dérivées de ``core.calibration.calibration_points`` / ``calibration_reference`` — les
    MÊMES évaluations que celles dont sortent la régression et l'export CSV. ``included`` dit
    si le point entre dans la régression ; sinon ``reasons`` en donne tous les motifs.
    ``kind`` vaut « référence » pour le probe : son exclusion empêche le signal normalisé.
    ``kk_nonconform`` : verdict KK du groupe « non conforme » — le point est alors retenu
    mais à interpréter avec prudence (exclu seulement si ``strict``, sauf le probe).
    """
    rows = []
    for model in calibration_models(session):
        ref = calibration_reference(session, model, strict)
        pts = ([("référence", ref)] if ref is not None else []) \
            + [("concentration", pt) for pt in calibration_points(session, model, strict)]
        for kind, pt in pts:
            rows.append({
                "model": model, "group": pt.label, "kind": kind,
                "concentration": pt.concentration, "value": None if pt.value is None else float(pt.value),
                "included": pt.included, "reasons": list(pt.reasons),
                "kk_nonconform": pt.kk_nonconform,
            })
    return rows
