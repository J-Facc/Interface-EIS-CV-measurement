"""Étalonnages EIS et CV (v3) — logique métier pure, source UNIQUE des calculs d'étalonnage.

Les figures (``plotting/calibration_plots.py``, ``plotting/mpl_plots.py``), la page Analyse,
l'archive et les rapports lisent ces fonctions : mêmes points, mêmes moyennes, mêmes
régressions partout. ``plotting/`` ne calcule rien.

* ``eis_calibration`` : log10(Rct/Rct_probe) vs log10([c]), Rct par DRT (aire du pic
  principal) ou par le fit Randles ;
* ``cv_calibration`` : ΔI/I = (I_probe − I) / I_probe vs log10([c]) ;
* ``legacy_curves`` : courbes de la v2 (spectre moyen, |Rct_probe − Rct_c| / Rct_probe et
  log10(Rct)), conservées.

Un verdict Kramers-Kronig « non conforme » n'EXCLUT jamais un point (il est signalé :
``CalPoint.kk_nonconform``) ; seule la vue v2 « stricte » l'exclut, pour comparaison. Un point
exclu n'est jamais perdu en silence : ``CalPoint.reason`` dit pourquoi.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

#: Raisons d'exclusion d'un point de calibration (texte affiché tel quel par l'UI).
REASON_NOT_CONVERGED = "fit non convergé"
REASON_KK_NONCONFORM = "KK non conforme"
REASON_NO_FIT = "fit non réalisé"
REASON_DRT_MISSING = "DRT non calculée"
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
            return float("nan"), REASON_DRT_MISSING
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
                        else (float("nan"), REASON_DRT_MISSING))
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
