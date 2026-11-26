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
CONVERGÉ et a une valeur cible > 0 (donc calculable : DRT présente pour la courbe DRT).

Le verdict Kramers-Kronig « non conforme » (``GroupAnalysis.validation.all_valid is False``)
n'EXCLUT PLUS un point par défaut : la détection de dérive qui l'alimente a un taux de faux
positifs documenté (AUDIT.md ERR-4) qui vidait la calibration sur données réelles, probe
compris. Le point reste dans la régression, signalé (``CalibrationPoint.kk_nonconform``) pour
que l'interface le distingue. La régression « stricte » (``strict=True``) restaure l'ancienne
exclusion, pour comparer les deux. Le probe, référence du signal normalisé, n'est JAMAIS écarté sur le verdict KK, y compris en
mode strict (sans lui, plus de normalisation) : s'il a convergé il sert de référence, signalé
``kk_nonconform`` si son verdict l'est.
Un point exclu n'est jamais perdu en silence : ``CalibrationPoint.reasons`` dit pourquoi.
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

#: Nombre minimal de points retenus pour calculer une régression EIS (pente, R², p-value…).
#: Avec 2 points la droite passe exactement par eux (R² = 1 trivial, aucun degré de liberté) :
#: les points restent exposés, la régression est absente (champs ``None``).
MIN_POINTS_REGRESSION = 3
#: Message affiché (figures, UI) quand il y a trop peu de points pour la régression.
MSG_REGRESSION_MIN_POINTS = (
    f"Au moins {MIN_POINTS_REGRESSION} concentrations sont nécessaires "
    "pour la régression d'étalonnage"
)


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
    slope: Optional[float]     # None (ainsi qu'intercept, r2, pvalue, stderr) si n < MIN_POINTS_REGRESSION
    intercept: Optional[float]
    r2: Optional[float]
    pvalue: Optional[float]
    stderr: Optional[float]
    n: int
    kk_flags: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # KK non conforme

    @property
    def has_regression(self) -> bool:
        """Vrai si la régression a pu être calculée (≥ ``MIN_POINTS_REGRESSION`` points)."""
        return self.slope is not None


def _regression_fields(log_c, y) -> dict:
    """Champs de régression de ``CalibrationResult`` ; tous ``None`` sous le seuil de points."""
    if len(log_c) < MIN_POINTS_REGRESSION:
        return dict(slope=None, intercept=None, r2=None, pvalue=None, stderr=None)
    reg = stats.linregress(log_c, y)
    return dict(slope=float(reg.slope), intercept=float(reg.intercept),
                r2=float(reg.rvalue ** 2), pvalue=float(reg.pvalue), stderr=float(reg.stderr))


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
        kk_nonconform: verdict KK du groupe « non conforme » — information, pas exclusion
            (sauf ``strict=True``, où le point est alors exclu avec ``REASON_KK_NONCONFORM``).
    """

    label: str
    concentration: float
    value: Optional[float]
    error: float
    included: bool
    reasons: tuple = field(default_factory=tuple)
    kk_nonconform: bool = False


def _assess(label: str, concentration: float, spectrum, analysis, model: str,
            strict: bool = False, is_reference: bool = False) -> CalibrationPoint:
    """Évalue un groupe pour ``model`` : retenu, ou exclu avec TOUS ses motifs.

    Le verdict KK « non conforme » ne fait qu'étiqueter le point (``kk_nonconform``) ; il ne
    l'exclut que si ``strict`` et que ce n'est pas la référence (probe).
    """
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
    kk_bad = getattr(vr, "all_valid", None) is False
    if kk_bad and strict and not is_reference:
        reasons.append(REASON_KK_NONCONFORM)
    if not fr.target_value > 0:                    # inclut NaN
        reasons.append(REASON_NONPOSITIVE)
    return CalibrationPoint(label, concentration, float(fr.target_value),
                            float(getattr(fr, "reconstruction_error", 0.0) or 0.0),
                            not reasons, tuple(reasons), kk_bad)


def calibration_points(session, model: str, strict: bool = False) -> list:
    """``CalibrationPoint`` de chaque concentration > 0 de la session, dans l'ordre des groupes.

    ``strict=True`` exclut en plus les groupes au verdict KK « non conforme »."""
    return [_assess(f"{g.concentration:.2e} M", float(g.concentration), g.spectrum,
                    getattr(g, "analysis", None), model, strict)
            for g in session.groups if g.concentration > 0]


def calibration_reference(session, model: str, strict: bool = False) -> Optional[CalibrationPoint]:
    """Le probe, référence du signal normalisé, évalué comme un point (None sans probe).

    Le verdict KK n'exclut jamais la référence, ``strict`` ou non."""
    probe = getattr(session, "probe", None)
    if probe is None:
        return None
    return _assess("Probe", 0.0, probe, getattr(session, "probe_analysis", None), model,
                   strict, is_reference=True)


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


def _collect_points(session, model: str, strict: bool = False):
    """Retourne (concs, rcts, errs, kk_flags) des points RETENUS (``calibration_points``) :
    fit existant, convergé, valeur cible > 0 (et, si ``strict``, verdict KK conforme)."""
    kept = [pt for pt in calibration_points(session, model, strict) if pt.included]
    return ([pt.concentration for pt in kept], [pt.value for pt in kept], [pt.error for pt in kept],
            [pt.kk_nonconform for pt in kept])


def compute_calibration(session, model: str, strict: bool = False):
    """Calibration signal normalisé pour un modèle : |ΔRct|/Rct_probe vs log10([c]).

    Args:
        session: EISSession (doit avoir un probe retenu — fit convergé ; le verdict KK ne l'écarte jamais).
        strict: exclut aussi les concentrations au verdict KK « non conforme ».
        model: nom du modèle de fit (ex. "orazem", "drt_bayes").

    Returns:
        CalibrationResult, ou None si pas de fit probe valide ou < 2 points. Avec 2 points,
        les points sont rendus mais les champs de régression valent None (voir
        ``MIN_POINTS_REGRESSION``).
    """
    ref = calibration_reference(session, model, strict)
    if ref is None or not ref.included:
        return None
    probe_rct = ref.value

    concs, rcts, errs, kk = _collect_points(session, model, strict)
    if len(concs) < 2:
        return None

    signal = [abs(probe_rct - r) / abs(probe_rct) for r in rcts]
    log_c = np.log10(concs)
    return CalibrationResult(
        model=model, log_c=log_c, y=np.asarray(signal),
        concentrations=np.asarray(concs), rcts=np.asarray(rcts), errs=np.asarray(errs),
        probe_rct=probe_rct, kk_flags=np.asarray(kk, dtype=bool),
        **_regression_fields(log_c, signal), n=len(concs),
    )


def compute_calibration_loglog(session, model: str, strict: bool = False):
    """Calibration log-log pour un modèle : log10(Rct) vs log10([c]) (DRT).

    Args:
        session: EISSession.
        model: nom du modèle de fit.

    Returns:
        CalibrationResult (y = log10(Rct), probe_rct = NaN), ou None si < 2 points.
    """
    concs, rcts, errs, kk = _collect_points(session, model, strict)
    if len(concs) < 2:
        return None
    log_c = np.log10(concs)
    log_rct = np.log10(rcts)
    return CalibrationResult(
        model=model, log_c=log_c, y=log_rct,
        concentrations=np.asarray(concs), rcts=np.asarray(rcts), errs=np.asarray(errs),
        probe_rct=float("nan"), kk_flags=np.asarray(kk, dtype=bool),
        **_regression_fields(log_c, log_rct), n=len(concs),
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


def compute_calibration_all(session, strict: bool = False):
    """Calibrations signal normalisé pour tous les modèles présents sur le probe.

    Returns:
        Liste de CalibrationResult (une par modèle avec ≥ 2 points valides),
        dans l'ordre des modèles du probe.
    """
    probe = getattr(session, "probe", None)
    probe_fr = getattr(probe, "fit_results", {}) if probe is not None else {}
    results = []
    for model in probe_fr:
        res = compute_calibration(session, model, strict)
        if res is not None:
            results.append(res)
    return results


def compute_calibration_loglog_all(session, strict: bool = False):
    """Calibrations log-log de tous les modèles de ``calibration_models`` ayant ≥ 2 points
    retenus (ne dépend pas du probe).

    Returns:
        Liste de CalibrationResult, dans l'ordre de ``calibration_models``.
    """
    results = []
    for model in calibration_models(session):
        res = compute_calibration_loglog(session, model, strict)
        if res is not None:
            results.append(res)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# v3 — Étalonnages par réplicat : normalisation par électrode, moyenne ± écart-type
# ═════════════════════════════════════════════════════════════════════════════
#
# 1. Chaque RÉPLICAT est normalisé par la moyenne « probe » de SON électrode :
#    EIS : y = log10(Rct / Rct_probe) ; CV : y = ΔI/I = (I_probe − I) / I_probe.
# 2. Par concentration : moyenne ± écart-type ENTRE RÉPLICATS (ddof = 1) ; un seul
#    réplicat → pas d'écart-type (NaN), et c'est signalé.
# 3. Régression linéaire des moyennes sur log10([c]) (≥ MIN_POINTS_REGRESSION niveaux).
#
# Deux vues : « par électrode » (une courbe par canal/électrode) et « regroupé par
# biomarqueur » (tous les réplicats de toutes les électrodes du biomarqueur à la même
# concentration). Les deux modes de canal (une concentration par canal / plusieurs sur un
# canal) sont traités de la même façon : chaque point (canal, électrode, étape, réplicat)
# est ramené à (concentration, valeur normalisée) puis regroupé par concentration.

SOURCE_DRT = "drt"          # Rct_DRT = aire du pic principal
SOURCE_FIT = "fit"          # Rct du fit Randles (paramètre cible)
SOURCE_LABELS = {SOURCE_DRT: "DRT (aire du pic principal)", SOURCE_FIT: "fit Randles"}

VIEW_ELECTRODE = "electrode"
VIEW_BIOMARKER = "biomarker"

REASON_NO_PROBE = "aucune référence probe sur cette électrode"
REASON_NO_VALUE = "valeur non calculée"
MSG_SINGLE_REPLICATE = "1 seul réplicat : pas d'écart-type"


@dataclass
class CalPoint:
    """Un réplicat d'étalonnage (inclus ou non, avec la raison)."""

    biomarker: str
    channel_id: int
    electrode: int
    step: str
    replicate: int
    concentration: float
    value: float                 # Rct (Ω) ou I_pic (A)
    reference: float             # moyenne probe de l'électrode
    y: float                     # valeur normalisée
    included: bool = True
    reason: str = ""
    kk_nonconform: bool = False


@dataclass
class CalLevel:
    """Une concentration d'une courbe : moyenne ± écart-type (ddof = 1) des réplicats."""

    concentration: float
    log_c: float
    mean: float
    std: float                   # NaN si n < 2
    n: int
    note: str = ""


@dataclass
class CalCurve:
    """Une courbe d'étalonnage (une électrode ou un biomarqueur)."""

    biomarker: str
    label: str
    quantity: str                # « log10(Rct/Rct_probe) » ou « ΔI/I »
    source: str
    levels: list = field(default_factory=list)
    points: list = field(default_factory=list)       # tous les CalPoint, exclus compris
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r2: Optional[float] = None
    pvalue: Optional[float] = None
    stderr: Optional[float] = None
    message: str = ""

    @property
    def has_regression(self) -> bool:
        return self.slope is not None


def _probe_reference(values: list) -> float:
    vals = [v for v in values if np.isfinite(v) and v > 0]
    return float(np.mean(vals)) if vals else float("nan")


def eis_replicate_value(spectrum, source: str) -> tuple:
    """(valeur, raison si indisponible) du Rct d'un réplicat selon la source."""
    if source == SOURCE_DRT:
        if spectrum.drt is None:
            return float("nan"), "DRT non calculée"
        if not spectrum.drt.peak.ok:
            return float("nan"), "pic principal absent de la fenêtre τ"
        return float(spectrum.drt.rct_main), ""
    if spectrum.fit is None:
        return float("nan"), REASON_NO_FIT
    if not spectrum.fit.converged:
        return float("nan"), REASON_NOT_CONVERGED
    return float(spectrum.fit.target_value), ""


def eis_points(results, experiment, source: str = SOURCE_DRT) -> list:
    """Points d'étalonnage EIS (réplicats des étapes conc_i), normalisés par le probe de l'électrode."""
    spectra = [s for s in results.eis.spectra.values() if not s.is_mean]
    refs: dict = {}
    for s in spectra:
        if s.step == "probe":
            v, _ = eis_replicate_value(s, source)
            refs.setdefault((s.channel_id, s.electrode), []).append(v)
    out = []
    for s in spectra:
        if s.concentration is None or s.step in ("bare", "probe"):
            continue
        ch = experiment.channel(s.channel_id)
        ref = _probe_reference(refs.get((s.channel_id, s.electrode), []))
        v, why = eis_replicate_value(s, source)
        p = CalPoint(biomarker=ch.biomarker if ch else "", channel_id=s.channel_id, electrode=s.electrode,
                     step=s.step, replicate=s.replicate, concentration=float(s.concentration), value=v,
                     reference=ref, y=float("nan"), kk_nonconform=s.kk_conform is False)
        if why:
            p.included, p.reason = False, why
        elif not v > 0:
            p.included, p.reason = False, REASON_NONPOSITIVE
        elif not np.isfinite(ref):
            p.included, p.reason = False, REASON_NO_PROBE
        else:
            p.y = float(np.log10(v / ref))
        out.append(p)
    return out


def cv_points(results, experiment) -> list:
    """Points d'étalonnage CV : ΔI/I = (I_probe − I) / I_probe de chaque réplicat conc_i."""
    out = []
    for f in results.cv.values():
        if f.concentration is None or f.step in ("bare", "probe"):
            continue
        ch = experiment.channel(f.channel_id)
        p = CalPoint(biomarker=ch.biomarker if ch else "", channel_id=f.channel_id, electrode=f.electrode,
                     step=f.step, replicate=f.replicate, concentration=float(f.concentration), value=f.i_peak,
                     reference=f.i_probe, y=f.delta_i_norm)
        if not f.ok:
            p.included, p.reason = False, f.message or "pic anodique introuvable"
        elif not np.isfinite(f.i_probe):
            p.included, p.reason = False, REASON_NO_PROBE
        elif not np.isfinite(f.delta_i_norm):
            p.included, p.reason = False, REASON_NO_VALUE
        out.append(p)
    return out


def calibration_levels(points: list) -> list:
    """Regroupe les points INCLUS par concentration : moyenne, écart-type ddof = 1, n."""
    by_c: dict = {}
    for p in points:
        if p.included and np.isfinite(p.y):
            by_c.setdefault(p.concentration, []).append(p.y)
    levels = []
    for c in sorted(by_c):
        ys = np.asarray(by_c[c], dtype=float)
        n = int(ys.size)
        std = float(np.std(ys, ddof=1)) if n >= 2 else float("nan")
        levels.append(CalLevel(concentration=float(c), log_c=float(np.log10(c)), mean=float(ys.mean()),
                               std=std, n=n, note=MSG_SINGLE_REPLICATE if n == 1 else ""))
    return levels


def _regress(curve: CalCurve) -> CalCurve:
    if len(curve.levels) >= MIN_POINTS_REGRESSION:
        x = np.array([lv.log_c for lv in curve.levels])
        y = np.array([lv.mean for lv in curve.levels])
        reg = stats.linregress(x, y)
        curve.slope, curve.intercept = float(reg.slope), float(reg.intercept)
        curve.r2, curve.pvalue, curve.stderr = float(reg.rvalue ** 2), float(reg.pvalue), float(reg.stderr)
    else:
        curve.message = (f"{len(curve.levels)} concentration(s) exploitable(s) : "
                         f"{MSG_REGRESSION_MIN_POINTS} (points affichés sans droite).")
    if any(lv.n == 1 for lv in curve.levels):
        curve.message = (curve.message + " " if curve.message else "") + \
            "Concentration(s) à 1 seul réplicat : sans barre d'erreur."
    return curve


def build_curves(points: list, view: str = VIEW_BIOMARKER, quantity: str = "", source: str = "") -> list:
    """Une courbe par biomarqueur (``VIEW_BIOMARKER``) ou par canal/électrode (``VIEW_ELECTRODE``)."""
    groups: dict = {}
    for p in points:
        key = (p.biomarker,) if view == VIEW_BIOMARKER else (p.biomarker, p.channel_id, p.electrode)
        groups.setdefault(key, []).append(p)
    curves = []
    for key in sorted(groups):
        label = key[0] if view == VIEW_BIOMARKER else f"{key[0]} — canal {key[1]} · électrode {key[2]}"
        pts = groups[key]
        curves.append(_regress(CalCurve(biomarker=key[0], label=label, quantity=quantity, source=source,
                                        levels=calibration_levels(pts), points=pts)))
    return curves


def eis_calibration(results, experiment, source: str = SOURCE_DRT, view: str = VIEW_BIOMARKER) -> list:
    """Étalonnage EIS : log10(Rct/Rct_probe) vs log10([c]), une courbe par biomarqueur ou électrode."""
    return build_curves(eis_points(results, experiment, source), view, "log10(Rct / Rct_probe)", source)


def cv_calibration(results, experiment, view: str = VIEW_BIOMARKER) -> list:
    """Étalonnage CV : ΔI/I vs log10([c]), une courbe par biomarqueur ou électrode."""
    return build_curves(cv_points(results, experiment), view, "ΔI/I = (I_probe − I) / I_probe", "cv")


# ── Anciennes courbes (v2), recalculées sur les résultats v3 ─────────────────
# Valeur d'un groupe = paramètre du spectre MOYEN : Rct du fit (source « fit ») ou Rct DRT du
# moteur (pic pénultième, source « drt »). Signal = |R_probe − R_c| / |R_probe| (ou log10 R
# pour la vue log-log), une courbe par électrode, sans écart-type (un spectre moyen par groupe).

LEGACY_SIGNAL = "signal"
LEGACY_LOGLOG = "loglog"


def legacy_curves(results, experiment, source: str = SOURCE_FIT, kind: str = LEGACY_SIGNAL,
                  strict: bool = False) -> list:
    """Courbes d'étalonnage de la v2 (spectre moyen), une par canal/électrode."""
    curves = []
    for ch_id, el in results.eis.electrodes():
        ch = experiment.channel(ch_id)
        means = {g.step: results.eis.spectra.get(g.mean_key) for g in results.eis.groups.values()
                 if (g.channel_id, g.electrode) == (ch_id, el) and g.mean_key}
        kk = {g.step: g.kk_state for g in results.eis.groups.values() if (g.channel_id, g.electrode) == (ch_id, el)}

        def val(sp):
            if sp is None:
                return float("nan"), REASON_NO_FIT
            if source == SOURCE_DRT:
                return ((float(sp.drt.rct_engine), "") if sp.drt is not None and sp.drt.converged
                        else (float("nan"), "DRT non calculée"))
            return eis_replicate_value(sp, SOURCE_FIT)

        ref, ref_why = val(means.get("probe"))
        pts = []
        for step, sp in means.items():
            if step in ("bare", "probe") or sp is None or sp.concentration is None:
                continue
            v, why = val(sp)
            p = CalPoint(biomarker=ch.biomarker if ch else "", channel_id=ch_id, electrode=el, step=step,
                         replicate=0, concentration=float(sp.concentration), value=v, reference=ref,
                         y=float("nan"), kk_nonconform=kk.get(step) == "non conforme")
            if why:
                p.included, p.reason = False, why
            elif not v > 0:
                p.included, p.reason = False, REASON_NONPOSITIVE
            elif strict and p.kk_nonconform:
                p.included, p.reason = False, REASON_KK_NONCONFORM
            elif kind == LEGACY_LOGLOG:
                p.y = float(np.log10(v))
            elif not (np.isfinite(ref) and ref != 0):
                p.included, p.reason = False, f"{REASON_NO_PROBE}{' (' + ref_why + ')' if ref_why else ''}"
            else:
                p.y = float(abs(ref - v) / abs(ref))
            pts.append(p)
        if not pts:
            continue
        quantity = "log10(Rct)" if kind == LEGACY_LOGLOG else "|Rct_probe − Rct_c| / Rct_probe"
        label = f"{ch.biomarker if ch else ''} — canal {ch_id} · électrode {el}"
        c = _regress(CalCurve(biomarker=ch.biomarker if ch else "", label=label, quantity=quantity, source=source,
                              levels=calibration_levels(pts), points=pts))
        c.message = c.message.replace(" Concentration(s) à 1 seul réplicat : sans barre d'erreur.", "") \
                             .replace("Concentration(s) à 1 seul réplicat : sans barre d'erreur.", "")
        curves.append(c)
    return curves
