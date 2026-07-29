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
  log10([c]). C'est la calibration DRT de `calibration_drt_figure`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


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


def _collect_points(session, model: str):
    """Retourne (concs, rcts, errs) triés d'apparition pour les concentrations
    > 0 dont le fit `model` est valide (Rct > 0)."""
    concs, rcts, errs = [], [], []
    for grp in session.groups:
        if grp.concentration <= 0:
            continue
        fr = grp.fit_results.get(model)
        if fr is None or fr.Rct <= 0:
            continue
        concs.append(float(grp.concentration))
        rcts.append(float(fr.Rct))
        errs.append(float(getattr(fr, "reconstruction_error", 0.0) or 0.0))
    return concs, rcts, errs


def compute_calibration(session, model: str):
    """Calibration signal normalisé pour un modèle : |ΔRct|/Rct_probe vs log10([c]).

    Args:
        session: EISSession (doit avoir un probe fitté par `model`).
        model: nom du modèle de fit (ex. "randles_full", "drt_bayes").

    Returns:
        CalibrationResult, ou None si pas de fit probe valide ou < 2 points.
    """
    probe = getattr(session, "probe", None)
    probe_fr = getattr(probe, "fit_results", {}) if probe is not None else {}
    probe_fit = probe_fr.get(model)
    if probe_fit is None or probe_fit.Rct <= 0:
        return None
    probe_rct = float(probe_fit.Rct)

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
