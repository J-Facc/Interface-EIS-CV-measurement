"""Calibration EIS — logique métier pure (aucun import UI/plotting/exports).

Source **unique** des calculs de calibration EIS, afin que les figures
(`plotting/eis_plots.py`) et les exports (`exports/exporter.py`) produisent
exactement les mêmes chiffres (pente, ordonnée, R²) — et que `plotting/` ne
calcule plus rien (règle d'architecture).

Deux calibrations distinctes coexistent dans l'app :

* **signal normalisé** — `compute_calibration` : signal = |Rct_probe − Rct_c| /
  |Rct_probe| régressé sur log10([c]). C'est la calibration de
  `calibration_figure` et `export_calibration_csv`.
* **log-log Rct** — `compute_calibration_loglog` : log10(Rct) régressé sur
  log10([c]), par modèle (circuit Orazem et DRT) — `calibration_loglog_figure`.

Garde-fous communs (``_assess`` / ``calibration_points``) — UNE seule évaluation par point,
dont dérivent la régression, donc l'export CSV, et les raisons d'exclusion affichées par
l'onglet « Calibration » : un groupe n'entre dans la régression que si son fit existe, a
CONVERGÉ, a une valeur cible > 0, et si son verdict Kramers-Kronig n'est pas « non conforme »
(``GroupAnalysis.validation.all_valid is False``). Un verdict indéterminé (None) n'exclut pas :
il n'a produit aucun fit à régresser, ou n'a pas dit que les données étaient incohérentes. Le
probe, qui sert de référence au signal normalisé, est soumis aux MÊMES règles : sans référence
fiable, la calibration normalisée n'est pas calculée. Un point exclu n'est jamais perdu en
silence : ``CalibrationPoint.reasons`` dit pourquoi.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from drt.engine import MODEL_NAME as DRT_MODEL_NAME

#: Raisons d'exclusion d'un point de calibration (texte affiché tel quel par l'UI).
REASON_NOT_CONVERGED = "fit non convergé"
REASON_KK_NONCONFORM = "KK non conforme"
REASON_NO_FIT = "fit non réalisé"
REASON_DRT_MISSING = "DRT non calculée"
REASON_GROUP_STOPPED = "groupe arrêté (aucun fit)"
REASON_NONPOSITIVE = "valeur cible ≤ 0"


@dataclass
class CalibrationResult:
    """Résultat d'une régression de calibration (signal normalisé ou log-log)."""

    model: str
    log_c: np.ndarray          # log10 des concentrations retenues (> 0, fit valide)
    y: np.ndarray              # signal normalisé, ou log10(Rct) selon la variante
    concentrations: np.ndarray
    rcts: np.ndarray
    errs: np.ndarray           # reconstruction_error par point (0 si absent)
    probe_rct: float           # Rct probe (NaN pour la variante log-log)
    slope: float
    intercept: float
    r2: float
    pvalue: float
    stderr: float
    n: int


@dataclass(frozen=True)
class CalibrationPoint:
    """Un groupe candidat à la calibration d'un modèle, retenu ou exclu AVEC sa raison.

    Attributes:
        label: libellé d'affichage du groupe (« Probe », « 1.00e-09 M »).
        concentration: mol/L (0.0 pour le probe).
        value: valeur cible du fit (Rct…), None si le fit n'existe pas.
        error: erreur de reconstruction relative RMS (0 si absente).
        included: True si le point entre dans la régression.
        reasons: motifs d'exclusion (vide si retenu).
    """

    label: str
    concentration: float
    value: Optional[float]
    error: float
    included: bool
    reasons: tuple = field(default_factory=tuple)


def _assess(label: str, concentration: float, spectrum, analysis, model: str) -> CalibrationPoint:
    """Évalue un groupe pour ``model`` : retenu, ou exclu avec TOUS ses motifs."""
    fr = spectrum.fit_results.get(model) if spectrum is not None else None
    if fr is None:
        if model == DRT_MODEL_NAME:
            why = (getattr(analysis, "drt_failures", None) or {}).get(getattr(spectrum, "label", None))
            reason = REASON_DRT_MISSING + (f" : {why}" if why else "")
        elif analysis is not None and not getattr(analysis, "ok", True):
            reason = REASON_GROUP_STOPPED
        else:
            reason = REASON_NO_FIT
        return CalibrationPoint(label, concentration, None, 0.0, False, (reason,))
    reasons = []
    if not fr.converged:
        reasons.append(REASON_NOT_CONVERGED)
    vr = getattr(analysis, "validation", None)
    if getattr(vr, "all_valid", None) is False:
        reasons.append(REASON_KK_NONCONFORM)
    if not fr.target_value > 0:                    # inclut NaN
        reasons.append(REASON_NONPOSITIVE)
    return CalibrationPoint(label, concentration, float(fr.target_value),
                            float(getattr(fr, "reconstruction_error", 0.0) or 0.0),
                            not reasons, tuple(reasons))


def calibration_points(session, model: str) -> list:
    """``CalibrationPoint`` de chaque concentration > 0 de la session, dans l'ordre des groupes."""
    return [_assess(f"{g.concentration:.2e} M", float(g.concentration), g.spectrum,
                    getattr(g, "analysis", None), model)
            for g in session.groups if g.concentration > 0]


def calibration_reference(session, model: str) -> Optional[CalibrationPoint]:
    """Le probe, référence du signal normalisé, évalué comme un point (None sans probe)."""
    probe = getattr(session, "probe", None)
    if probe is None:
        return None
    return _assess("Probe", 0.0, probe, getattr(session, "probe_analysis", None), model)


def calibration_models(session) -> list:
    """Modèles à calibrer : ceux qui portent un fit (probe ou groupes), dans l'ordre
    d'apparition ; la DRT y figure dès qu'elle a été demandée, même si aucun spectre n'en
    porte (chaque point est alors exclu « DRT non calculée »)."""
    names: list = []
    spectra = [getattr(session, "probe", None)] + [g.spectrum for g in session.groups]
    for sp in spectra:
        for m in getattr(sp, "fit_results", {}) or {}:
            if m not in names:
                names.append(m)
    if getattr(session, "drt_mode", None) and DRT_MODEL_NAME not in names:
        names.append(DRT_MODEL_NAME)
    return names


def _collect_points(session, model: str):
    """Retourne (concs, rcts, errs) des points RETENUS (``calibration_points``) : fit existant,
    convergé, valeur cible > 0, verdict KK du groupe non « non conforme »."""
    kept = [pt for pt in calibration_points(session, model) if pt.included]
    return ([pt.concentration for pt in kept], [pt.value for pt in kept], [pt.error for pt in kept])


def compute_calibration(session, model: str):
    """Calibration signal normalisé pour un modèle : |ΔRct|/Rct_probe vs log10([c]).

    Args:
        session: EISSession (doit avoir un probe retenu — fit convergé, KK non « non conforme »).
        model: nom du modèle de fit (ex. "orazem", "drt_bayes").

    Returns:
        CalibrationResult, ou None si pas de fit probe valide ou < 2 points.
    """
    ref = calibration_reference(session, model)
    if ref is None or not ref.included:
        return None
    probe_rct = ref.value

    concs, rcts, errs = _collect_points(session, model)
    if len(concs) < 2:
        return None

    signal = [abs(probe_rct - r) / abs(probe_rct) for r in rcts]
    log_c = np.log10(concs)
    reg = stats.linregress(log_c, signal)
    return CalibrationResult(
        model=model, log_c=log_c, y=np.asarray(signal),
        concentrations=np.asarray(concs), rcts=np.asarray(rcts), errs=np.asarray(errs),
        probe_rct=probe_rct,
        slope=float(reg.slope), intercept=float(reg.intercept),
        r2=float(reg.rvalue ** 2), pvalue=float(reg.pvalue), stderr=float(reg.stderr),
        n=len(concs),
    )


def compute_calibration_loglog(session, model: str):
    """Calibration log-log pour un modèle : log10(Rct) vs log10([c]) (DRT).

    Args:
        session: EISSession.
        model: nom du modèle de fit.

    Returns:
        CalibrationResult (y = log10(Rct), probe_rct = NaN), ou None si < 2 points.
    """
    concs, rcts, errs = _collect_points(session, model)
    if len(concs) < 2:
        return None
    log_c = np.log10(concs)
    log_rct = np.log10(rcts)
    reg = stats.linregress(log_c, log_rct)
    return CalibrationResult(
        model=model, log_c=log_c, y=log_rct,
        concentrations=np.asarray(concs), rcts=np.asarray(rcts), errs=np.asarray(errs),
        probe_rct=float("nan"),
        slope=float(reg.slope), intercept=float(reg.intercept),
        r2=float(reg.rvalue ** 2), pvalue=float(reg.pvalue), stderr=float(reg.stderr),
        n=len(concs),
    )


@dataclass
class CVCalibrationResult:
    """Régression de calibration CV : signal ΔI normalisé moyen vs log10([c])."""

    concentrations: np.ndarray
    log_c: np.ndarray
    signals: np.ndarray        # moyenne de delta_signal par concentration
    slope: float
    intercept: float
    r2: float
    pvalue: float
    stderr: float
    n: int


def compute_cv_calibration(cv_session):
    """Calibration CV pour une session : moyenne de `delta_signal` par
    concentration (> 0, finie) régressée sur log10([c]).

    Args:
        cv_session: CVSession (ou tout objet avec `.groups`, chaque groupe ayant
            `.concentration` et `.delta_signal`).

    Returns:
        CVCalibrationResult, ou None si < 2 concentrations exploitables.
    """
    concs, signals = [], []
    for grp in cv_session.groups:
        if grp.concentration <= 0:
            continue
        mean_sig = np.nanmean(grp.delta_signal)
        if np.isfinite(mean_sig):
            concs.append(float(grp.concentration))
            signals.append(float(mean_sig))
    if len(concs) < 2:
        return None
    log_c = np.log10(concs)
    reg = stats.linregress(log_c, signals)
    return CVCalibrationResult(
        concentrations=np.asarray(concs), log_c=log_c, signals=np.asarray(signals),
        slope=float(reg.slope), intercept=float(reg.intercept),
        r2=float(reg.rvalue ** 2), pvalue=float(reg.pvalue), stderr=float(reg.stderr),
        n=len(concs),
    )


def compute_calibration_all(session):
    """Calibrations signal normalisé pour tous les modèles présents sur le probe.

    Returns:
        Liste de CalibrationResult (une par modèle avec ≥ 2 points valides),
        dans l'ordre des modèles du probe.
    """
    probe = getattr(session, "probe", None)
    probe_fr = getattr(probe, "fit_results", {}) if probe is not None else {}
    results = []
    for model in probe_fr:
        res = compute_calibration(session, model)
        if res is not None:
            results.append(res)
    return results


def compute_calibration_loglog_all(session):
    """Calibrations log-log de tous les modèles de ``calibration_models`` ayant ≥ 2 points
    retenus (ne dépend pas du probe).

    Returns:
        Liste de CalibrationResult, dans l'ordre de ``calibration_models``.
    """
    results = []
    for model in calibration_models(session):
        res = compute_calibration_loglog(session, model)
        if res is not None:
            results.append(res)
    return results
