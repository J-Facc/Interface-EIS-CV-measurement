"""
core/validator.py
=================
Validation KK des spectres EIS avant moyennage des réplicats.

Deux objectifs :
  1. Diagnostiquer chaque réplicat individuellement (lin-KK)
     → résidus par fréquence, plage valide [f_min, f_max], score µ
  2. Détecter un drift inter-réplicats
     → si les résidus KK divergent systématiquement entre réplicats

Dépendance : impedance.py  (pip install impedance)
Aucun import Streamlit — logique métier pure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────

@dataclass
class KKResult:
    """Résultat du test Kramers-Kronig pour un spectre unique."""

    label: str

    # Résidus normalisés (%) sur la plage complète
    frequencies: np.ndarray          # Hz
    residuals_re: np.ndarray          # (Zre_fit - Zre_exp) / |Z_exp|  en %
    residuals_im: np.ndarray          # (Zim_fit - Zim_exp) / |Z_exp|  en %

    # Qualité globale
    mu: float                         # ratio masse RC négative / totale  (0 = parfait, <0.85 = acceptable)
    chi2_pseudo: float                # chi² pseudo normalisé

    # Plage fréquentielle valide (indices dans frequencies[])
    f_min_valid: float                # Hz — borne basse de la plage KK-valide
    f_max_valid: float                # Hz — borne haute

    # Verdict
    is_valid: bool                    # True si µ < seuil ET résidus < seuil
    warning: Optional[str] = None    # message lisible si problème détecté


@dataclass
class ValidationResult:
    """Résultat de validation pour un groupe de réplicats (même concentration/étape)."""

    label: str                                  # ex: "bare", "hyb_1nM"
    replicates: List[KKResult] = field(default_factory=list)

    # Validité globale du groupe
    all_valid: bool = True
    drift_detected: bool = False
    drift_warning: Optional[str] = None

    # Plage fréquentielle commune (intersection des plages valides)
    f_min_common: float = 0.0
    f_max_common: float = np.inf

    # Écart-type empirique inter-réplicats par fréquence (calculé dans loader.py)
    # Stocké ici pour transmission au pipeline
    sigma_re: Optional[np.ndarray] = None      # même grille que le spectre moyenné
    sigma_im: Optional[np.ndarray] = None


# ─────────────────────────────────────────────
# Paramètres par défaut
# ─────────────────────────────────────────────

# Seuil µ au-delà duquel le fit KK est considéré sur-ajusté (Schönleber 2014)
MU_THRESHOLD = 0.85

# Seuil résidu (%) au-delà duquel un point est considéré invalide
RESIDUAL_THRESHOLD_PCT = 2.0

# Fraction de points invalides tolérés avant de marquer le spectre comme douteux
INVALID_FRACTION_WARN = 0.10   # 10 %
INVALID_FRACTION_REJECT = 0.25  # 25 %

# Seuil de divergence inter-réplicats pour détecter un drift
# (écart-type des résidus KK entre réplicats / résidu moyen)
DRIFT_CV_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# Validation d'un spectre unique
# ─────────────────────────────────────────────

def validate_spectrum(
    frequencies: np.ndarray,
    z_re: np.ndarray,
    z_im: np.ndarray,
    label: str = "",
    mu_threshold: float = MU_THRESHOLD,
    residual_threshold_pct: float = RESIDUAL_THRESHOLD_PCT,
) -> KKResult:
    """
    Applique le test lin-KK (Schönleber 2014) sur un spectre unique.

    Parameters
    ----------
    frequencies : Hz, ordre HF → BF ou BF → HF (trié automatiquement)
    z_re, z_im  : parties réelle et imaginaire de Z (Ω) — convention Z'' > 0 en BF
    label       : identifiant lisible pour les messages

    Returns
    -------
    KKResult avec résidus, µ, plage valide, verdict
    """
    try:
        from impedance.validation import linKK
    except ImportError as exc:
        raise ImportError(
            "Le package 'impedance' est requis pour la validation KK.\n"
            "Ajoutez 'impedance' à requirements.txt et relancez."
        ) from exc

    # lin-KK attend les fréquences en ordre croissant et Z'' < 0 (convention impedance.py)
    sort_idx = np.argsort(frequencies)
    f_sorted = frequencies[sort_idx]
    zre_sorted = z_re[sort_idx]
    zim_sorted = -np.abs(z_im[sort_idx])   # convention : Im(Z) < 0 pour circuit R-C

    Z_complex = zre_sorted + 1j * zim_sorted

    try:
        M, mu, Z_fit, res_re, res_im = linKK(
            f_sorted,
            Z_complex,
            c=0.85,          # critère µ de Schönleber
            max_M=100,
            fit_type="complex",
            add_cap=True,
        )
    except Exception as exc:
        logger.warning("lin-KK échoué pour '%s' : %s", label, exc)
        n = len(f_sorted)
        return KKResult(
            label=label,
            frequencies=f_sorted,
            residuals_re=np.zeros(n),
            residuals_im=np.zeros(n),
            mu=1.0,
            chi2_pseudo=np.inf,
            f_min_valid=f_sorted[0],
            f_max_valid=f_sorted[-1],
            is_valid=False,
            warning=f"lin-KK échoué : {exc}",
        )

    # Résidus normalisés (%) : (fit - exp) / |Z_exp|
    Z_mod = np.abs(Z_complex)
    res_re_pct = (res_re / Z_mod) * 100.0
    res_im_pct = (res_im / Z_mod) * 100.0

    # chi² pseudo
    chi2_pseudo = float(np.mean(res_re_pct**2 + res_im_pct**2))

    # Points valides : résidu < seuil sur Re ET Im
    valid_mask = (np.abs(res_re_pct) < residual_threshold_pct) & \
                 (np.abs(res_im_pct) < residual_threshold_pct)

    invalid_fraction = 1.0 - valid_mask.mean()

    # Plage fréquentielle valide : bloc continu le plus large de points valides
    f_min_valid, f_max_valid = _find_valid_range(f_sorted, valid_mask)

    # Verdict
    if mu > mu_threshold:
        is_valid = False
        warning = (
            f"µ = {mu:.3f} > {mu_threshold} : sur-ajustement probable "
            f"(trop d'éléments RC). Spectre possiblement non-stationnaire."
        )
    elif invalid_fraction >= INVALID_FRACTION_REJECT:
        is_valid = False
        warning = (
            f"{invalid_fraction*100:.0f}% des points hors tolérance KK "
            f"(seuil résidu = {residual_threshold_pct}%). Spectre invalide."
        )
    elif invalid_fraction >= INVALID_FRACTION_WARN:
        is_valid = True   # acceptable mais signalé
        warning = (
            f"{invalid_fraction*100:.0f}% des points marginaux — "
            f"vérifier les extrémités du spectre."
        )
    else:
        is_valid = True
        warning = None

    logger.debug(
        "KK '%s' : µ=%.3f, χ²=%.4f, invalides=%.1f%%, f_valid=[%.2f, %.2f] Hz",
        label, mu, chi2_pseudo, invalid_fraction * 100,
        f_min_valid, f_max_valid,
    )

    return KKResult(
        label=label,
        frequencies=f_sorted,
        residuals_re=res_re_pct,
        residuals_im=res_im_pct,
        mu=float(mu),
        chi2_pseudo=chi2_pseudo,
        f_min_valid=f_min_valid,
        f_max_valid=f_max_valid,
        is_valid=is_valid,
        warning=warning,
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


# ─────────────────────────────────────────────
# Validation d'un groupe de réplicats
# ─────────────────────────────────────────────

def validate_replicate_group(
    frequencies_list: List[np.ndarray],
    zre_list: List[np.ndarray],
    zim_list: List[np.ndarray],
    label: str = "",
    mu_threshold: float = MU_THRESHOLD,
    residual_threshold_pct: float = RESIDUAL_THRESHOLD_PCT,
    drift_cv_threshold: float = DRIFT_CV_THRESHOLD,
) -> ValidationResult:
    """
    Valide un groupe de réplicats (même concentration / même étape).

    1. Test KK individuel sur chaque réplicat
    2. Détection de drift inter-réplicats (divergence des résidus KK)
    3. Calcul de σ_re(f) et σ_im(f) empiriques
    4. Plage fréquentielle commune (intersection des plages valides)

    Parameters
    ----------
    frequencies_list : liste de tableaux Hz (un par réplicat)
    zre_list, zim_list : listes correspondantes de Re(Z) et Im(Z)
    label : identifiant du groupe

    Returns
    -------
    ValidationResult complet
    """
    result = ValidationResult(label=label)

    if not frequencies_list:
        result.all_valid = False
        result.drift_warning = "Aucun réplicat fourni."
        return result

    # ── 1. KK individuel ──
    for i, (f, zre, zim) in enumerate(zip(frequencies_list, zre_list, zim_list)):
        kk = validate_spectrum(
            f, zre, zim,
            label=f"{label}_rep{i+1}",
            mu_threshold=mu_threshold,
            residual_threshold_pct=residual_threshold_pct,
        )
        result.replicates.append(kk)
        if not kk.is_valid:
            result.all_valid = False

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

    # ── 4. σ(f) empirique inter-réplicats ──
    #    On interpole tous les réplicats sur la grille du premier
    #    (après tri fréquentiel commun)
    if len(frequencies_list) >= 2:
        result.sigma_re, result.sigma_im = _compute_sigma(
            frequencies_list, zre_list, zim_list
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


def _compute_sigma(
    frequencies_list: List[np.ndarray],
    zre_list: List[np.ndarray],
    zim_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule σ_re(f) et σ_im(f) empiriques en interpolant les réplicats
    sur la grille fréquentielle du premier réplicat (après tri).
    """
    f_ref = np.sort(frequencies_list[0])
    zre_stack, zim_stack = [], []

    for f, zre, zim in zip(frequencies_list, zre_list, zim_list):
        sort_idx = np.argsort(f)
        zre_i = np.interp(f_ref, f[sort_idx], zre[sort_idx])
        zim_i = np.interp(f_ref, f[sort_idx], zim[sort_idx])
        zre_stack.append(zre_i)
        zim_stack.append(zim_i)

    zre_stack = np.array(zre_stack)
    zim_stack = np.array(zim_stack)

    sigma_re = np.std(zre_stack, axis=0, ddof=1)
    sigma_im = np.std(zim_stack, axis=0, ddof=1)

    # Garde-fou : σ minimum à 0.1% du module moyen pour éviter poids infinis
    Z_mean_mod = np.sqrt(np.mean(zre_stack, axis=0)**2 + np.mean(zim_stack, axis=0)**2)
    floor = 0.001 * Z_mean_mod
    sigma_re = np.maximum(sigma_re, floor)
    sigma_im = np.maximum(sigma_im, floor)

    return sigma_re, sigma_im
