"""
core/validator.py
=================
Validation Kramers-Kronig des spectres EIS avant moyennage des réplicats.

Deux objectifs :
  1. Juger la conformité KK de chaque réplicat et du groupe — verdict affiché au
     prétraitement, donc AVANT que le fit Orazem ne soit proposé ;
  2. Détecter un drift inter-réplicats.

Méthode (une seule, AUDIT.md ERR-6) : le verdict vient TOUJOURS du measurement
model de Voigt pondéré par la structure d'erreur caractérisée sur les réplicats
du groupe (``core.measurement_model.analyze_replicates``), jugé par le critère
unique ``fits.kk_validation.kk_verdict``. Les anciens seuils sans source
(résidu < 2 % par point, paliers 10 %/25 %, µ > 0,85) sont supprimés.

Sans structure d'erreur (moins de 3 réplicats…), AUCUN verdict n'est rendu
(``is_valid=None``) : les résidus Lin-KK (Schönleber, ``fits.kk_validation.lin_kk``)
sont seulement AFFICHÉS, à titre indicatif.

Aucun import Streamlit — logique métier pure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────
# Structures de données
# ──────────────────────────────────────

@dataclass
class KKResult:
    """Résultat du test Kramers-Kronig pour un spectre unique.

    Résidus en % de |Z| (fréquences croissantes) :
      * méthode « measurement_model » : residuals_re = Re(données) − Re PRÉDITE depuis
        l'ajustement de Im seule (c'est la statistique du test) ; residuals_im = résidu
        de cet ajustement de Im ; band_re/band_im = ±2 écarts-types sous H0 (en %) ;
      * méthode « lin_kk » (affichage seul, sans verdict) : résidus Lin-KK ; pas de bande.
    """

    label: str
    frequencies: np.ndarray           # Hz, croissantes
    residuals_re: np.ndarray          # % de |Z|
    residuals_im: np.ndarray          # % de |Z|
    method: str                       # "measurement_model" | "lin_kk"
    n_elements: int                   # K (Voigt régressé) ou M (Lin-KK)
    chi2_reduced: float               # χ²ᵣ de l'ajustement de Im (NaN pour Lin-KK)
    f_min_valid: float                # Hz — bloc contigu le plus large dans la bande
    f_max_valid: float
    is_valid: Optional[bool]          # verdict ; None = indéterminé (pas de bruit caractérisé)
    warning: Optional[str] = None
    band_re: Optional[np.ndarray] = None   # ±2σ (%) de residuals_re
    band_im: Optional[np.ndarray] = None   # ±2σ (%) de residuals_im
    n_outside: int = 0
    n_allowed: int = 0


@dataclass
class ValidationResult:
    """Résultat de validation pour un groupe de réplicats (même concentration/étape)."""

    label: str                                  # ex: "bare", "hyb_1nM"
    replicates: List[KKResult] = field(default_factory=list)

    # Verdict KK du GROUPE (réplicats + moyenne, core.measurement_model) ; None si
    # la structure d'erreur n'a pas pu être caractérisée.
    all_valid: Optional[bool] = True
    drift_detected: bool = False
    drift_warning: Optional[str] = None

    # Plage fréquentielle commune (intersection des plages valides)
    f_min_common: float = 0.0
    f_max_common: float = np.inf

    # σ_r(ω), σ_j(ω) de la structure d'erreur (une mesure), sur la grille commune
    # croissante, évaluées sur le spectre moyen — SANS plancher. None si non caractérisée.
    sigma_re: Optional[np.ndarray] = None
    sigma_im: Optional[np.ndarray] = None

    # Analyse complète (structure d'erreur + tests KK) et messages pour l'utilisateur.
    measurement_model: Optional[object] = None      # MeasurementModelAnalysis
    kk_message: Optional[str] = None
    error_structure_message: Optional[str] = None


# ──────────────────────────────────────
# Paramètres par défaut
# ──────────────────────────────────────

# Seuil de divergence inter-réplicats pour détecter un drift
# (écart-type des résidus KK entre réplicats / résidu moyen)
DRIFT_CV_THRESHOLD = 0.5

_UNDETERMINED = (
    "Verdict KK indéterminé : structure d'erreur non caractérisée (résidus Lin-KK "
    "affichés à titre indicatif seulement)."
)


# ──────────────────────────────────────
# Validation d'un spectre unique
# ──────────────────────────────────────

def _kk_result_from_consistency(kk, label: str) -> KKResult:
    """KKResult (résidus en % de |Z|) depuis un ``core.measurement_model.KKConsistency``."""
    f = kk.frequencies
    mod = np.abs(kk.Zre - 1j * kk.Zim)
    v = kk.verdict
    # Spectre conforme : aucune preuve de violation → plage complète (sous H0, ~4,5 %
    # des points sortent de ±2σ par le seul bruit ; les compter fragmenterait la plage).
    # Non conforme : plus grand bloc contigu de points dans la bande.
    f_lo, f_hi = (float(f[0]), float(f[-1])) if v.conform else _find_valid_range(f, ~v.outside_re)
    return KKResult(
        label=label, frequencies=f,
        residuals_re=100.0 * kk.residual_re / mod,
        residuals_im=100.0 * kk.residual_im_fit / mod,
        method="measurement_model", n_elements=kk.model_from_imag.n_elements,
        chi2_reduced=kk.model_from_imag.chi2_reduced,
        f_min_valid=f_lo, f_max_valid=f_hi, is_valid=v.conform,
        warning=None if v.conform else v.message,
        band_re=200.0 * kk.sd_re / mod, band_im=200.0 * kk.sigma_im / mod,
        n_outside=v.n_outside, n_allowed=v.n_allowed,
    )


def validate_spectrum(
    frequencies: np.ndarray,
    z_re: np.ndarray,
    z_im: np.ndarray,
    label: str = "",
    error_structure=None,
    n_averaged: int = 1,
    reference=None,
) -> KKResult:
    """Test KK d'un spectre unique.

    Args:
        frequencies: Hz (ordre quelconque, trié ici).
        z_re, z_im: Re(Z) et Im(Z) (Ω) — convention de l'app, Z'' > 0 en BF.
        label: identifiant lisible.
        error_structure: ``core.measurement_model.ErrorStructure`` du groupe. Avec
            elle : test par measurement model et verdict. Sans elle : résidus Lin-KK
            affichés, verdict None.
        n_averaged: mesures moyennées dans ce spectre (σ/√n).
        reference: measurement model complexe de ce spectre (facultatif).

    Returns:
        KKResult. Des données inexploitables (valeurs non finies, trop peu de
        points) donnent ``is_valid=False`` avec le motif ; toute autre exception
        remonte (ce n'est plus « Lin-KK échoué » pour n'importe quelle erreur).
    """
    from fits.kk_validation import lin_kk

    f = np.asarray(frequencies, dtype=float)
    zre = np.asarray(z_re, dtype=float)
    zim = np.asarray(z_im, dtype=float)
    order = np.argsort(f)
    f, zre, zim = f[order], zre[order], zim[order]

    try:
        if error_structure is not None:
            from core.measurement_model import check_kk_consistency
            kk = check_kk_consistency(f, zre, zim, error_structure, n_averaged=n_averaged,
                                      reference=reference, label=label)
            return _kk_result_from_consistency(kk, label)
        lk = lin_kk(f, zre - 1j * zim)
    except ValueError as exc:
        logger.warning("KK impossible pour '%s' : %s", label, exc)
        n = len(f)
        lo, hi = (float(f[0]), float(f[-1])) if n else (float("nan"), float("nan"))
        return KKResult(
            label=label, frequencies=f, residuals_re=np.zeros(n), residuals_im=np.zeros(n),
            method="lin_kk", n_elements=0, chi2_reduced=float("nan"),
            f_min_valid=lo, f_max_valid=hi, is_valid=False,
            warning=f"données inexploitables pour le test KK : {exc}",
        )
    mod = np.abs(zre - 1j * zim)
    return KKResult(
        label=label, frequencies=f,
        residuals_re=100.0 * lk.res_re / mod,
        residuals_im=100.0 * (-lk.res_im) / mod,          # convention de l'app
        method="lin_kk", n_elements=lk.M, chi2_reduced=float("nan"),
        f_min_valid=float(f[0]), f_max_valid=float(f[-1]), is_valid=None,
        warning=_UNDETERMINED,
    )


def _find_valid_range(
    frequencies: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Retourne la plage [f_min, f_max] du bloc contigu de points KK-valides
    le plus large. Si aucun point n'est valide, retourne la plage complète.
    """
    if valid_mask.all():
        return float(frequencies[0]), float(frequencies[-1])
    if not valid_mask.any():
        return float(frequencies[0]), float(frequencies[-1])

    # Trouver le plus long run de True
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, v in enumerate(valid_mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    best_end = best_start + best_len - 1
    return float(frequencies[best_start]), float(frequencies[best_end])


# ──────────────────────────────────────
# Validation d'un groupe de réplicats
# ──────────────────────────────────────

def validate_replicate_group(
    frequencies_list: List[np.ndarray],
    zre_list: List[np.ndarray],
    zim_list: List[np.ndarray],
    label: str = "",
    drift_cv_threshold: float = DRIFT_CV_THRESHOLD,
    options=None,
) -> ValidationResult:
    """
    Valide un groupe de réplicats (même concentration / même étape).

    1. Measurement model : structure d'erreur des réplicats du groupe, puis test KK
       de chaque réplicat et de leur moyenne (``core.measurement_model.analyze_replicates``)
       → verdict du groupe ``all_valid`` ; σ(ω) de la structure (sans plancher) ;
    2. Plage fréquentielle commune (intersection des plages valides) ;
    3. Détection de drift inter-réplicats.

    Si la structure d'erreur n'est pas caractérisable (moins de 3 réplicats, grilles
    incompatibles, réplicats identiques…), ``all_valid`` vaut None, le motif est dans
    ``error_structure_message`` et chaque réplicat n'affiche que ses résidus Lin-KK.

    Parameters
    ----------
    frequencies_list : liste de tableaux Hz (un par réplicat)
    zre_list, zim_list : listes correspondantes de Re(Z) et Im(Z)
    label : identifiant du groupe
    options : core.measurement_model.MeasurementModelOptions (défaut : valeurs du module)

    Returns
    -------
    ValidationResult complet
    """
    from core.measurement_model import ErrorStructureUnavailable, analyze_replicates

    result = ValidationResult(label=label)

    if not frequencies_list:
        result.all_valid = False
        result.drift_warning = "Aucun réplicat fourni."
        return result

    labels = [f"{label}_rep{i+1}" for i in range(len(frequencies_list))]
    reps = [
        SimpleNamespace(f=np.asarray(f, dtype=float), Zre=np.asarray(zre, dtype=float),
                        Zim=np.asarray(zim, dtype=float), label=lab)
        for f, zre, zim, lab in zip(frequencies_list, zre_list, zim_list, labels)
    ]

    # ── 1. Measurement model : structure d'erreur puis verdict KK ──
    analysis = None
    try:
        analysis = analyze_replicates(reps, options=options, label=label)
    except ErrorStructureUnavailable as exc:
        result.error_structure_message = exc.user_message
        logger.warning("'%s' : %s", label, exc.user_message)

    if analysis is not None:
        result.measurement_model = analysis
        result.kk_message = analysis.kk_message
        result.all_valid = analysis.kk_conform
        result.replicates = [
            _kk_result_from_consistency(kk, lab) for kk, lab in zip(analysis.kk_replicates, labels)
        ]
        result.sigma_re, result.sigma_im = analysis.error_structure.sigmas(
            analysis.mean_Zre, analysis.mean_Zim)
    else:
        result.all_valid = None
        result.kk_message = _UNDETERMINED
        result.replicates = [validate_spectrum(r.f, r.Zre, r.Zim, label=r.label) for r in reps]
        if any(kk.is_valid is False for kk in result.replicates):
            result.all_valid = False           # données inexploitables : pas « indéterminé »

    # ── 2. Plage commune ──
    f_mins = [kk.f_min_valid for kk in result.replicates]
    f_maxs = [kk.f_max_valid for kk in result.replicates]
    result.f_min_common = max(f_mins)   # intersection = max des bornes basses
    result.f_max_common = min(f_maxs)   # intersection = min des bornes hautes

    if result.f_min_common >= result.f_max_common:
        result.all_valid = False
        result.drift_warning = (
            "Les plages KK-valides des réplicats ne se chevauchent pas — "
            "vérifier la stationnarité du système entre les scans."
        )
        logger.warning("'%s' : plages KK non compatibles entre réplicats", label)

    # ── 3. Détection de drift inter-réplicats ──
    if len(result.replicates) >= 2:
        result.drift_detected, result.drift_warning = _detect_drift(
            result.replicates, drift_cv_threshold
        )

    return result


def _detect_drift(
    kk_results: List[KKResult],
    cv_threshold: float,
) -> Tuple[bool, Optional[str]]:
    """
    Détecte un drift inter-réplicats en comparant les résidus KK Im.
    Si le coefficient de variation (σ/µ) des résidus dépasse cv_threshold,
    le drift est signalé.
    """
    # Interpoler les résidus Im sur une grille log commune
    f_ref = kk_results[0].frequencies
    residuals_stack = []

    for kk in kk_results:
        res_interp = np.interp(
            np.log10(f_ref),
            np.log10(kk.frequencies),
            kk.residuals_im,
        )
        residuals_stack.append(res_interp)

    residuals_stack = np.array(residuals_stack)   # shape (n_rep, n_freq)
    mean_res = np.mean(np.abs(residuals_stack), axis=0)
    std_res = np.std(residuals_stack, axis=0)

    # Éviter la division par zéro
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(mean_res > 0.1, std_res / mean_res, 0.0)

    fraction_drifted = (cv > cv_threshold).mean()

    if fraction_drifted > 0.20:   # plus de 20% des fréquences montrent un drift
        msg = (
            f"Drift détecté sur {fraction_drifted*100:.0f}% des fréquences "
            f"(CV résidus KK > {cv_threshold:.1f}). "
            f"Le système n'est peut-être pas stationnaire entre les {len(kk_results)} réplicats."
        )
        logger.warning(msg)
        return True, msg

    return False, None
