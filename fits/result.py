"""Résultat d'ajustement d'un spectre — contrat commun du fit Orazem et de la DRT.

Ce module vit dans ``fits/`` (et non dans ``core/``) pour que le sens des dépendances
reste UNIQUE (AUDIT.md CPL-1) : ``core`` → ``fits``/``drt``/``circuit``, jamais
l'inverse. ``fits/orazem_fit.py`` et ``drt/engine.py`` produisent des ``FitResult``
sans rien importer de ``core`` ; ``core.models`` le ré-exporte pour l'UI et les
exports. Aucune dépendance interne : ce module ne doit importer que numpy et la
bibliothèque standard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FitResult:
    """Result of a fit model applied to an EIS spectrum.

    Zfit_im follows the same positive convention as EISSpectrum.Zim
    (i.e. -Im(Z) > 0 for a capacitive semicircle). ``residuals_*`` = données − modèle,
    dans cette même convention. Les tableaux suivent l'ordre des fréquences du
    spectre ajusté (HF → BF pour les spectres de l'application).

    Paramètre cible (signal de calibration) : le circuit étant libre, il n'y a plus
    de champ « Rct » figé. L'utilisateur DÉSIGNE le paramètre qui sert de signal
    (``target_param``, un nom de ``params``) ; ``target_value``/``target_std`` en
    sont la valeur et l'écart-type (intra-fit pour un spectre ; pour la DRT, Rct
    extrait de la distribution et son incertitude a posteriori, NaN en MAP).
    """

    model_name: str
    params: dict
    params_std: dict
    Zfit_re: np.ndarray
    Zfit_im: np.ndarray
    # χ² réduit pondéré = Σ(w·Δ²)/(2N−P), avec les poids w = 1/σ² de la structure
    # d'erreur d'Orazem. Les poids étant de vraies 1/variance (absolute_sigma TOUJOURS),
    # chi2_reduced≈1 EST un test d'adéquation modèle+erreur (voir chi2_reduced_ci).
    # NaN pour la DRT (pas de χ² pondéré, AUDIT.md DRT-5).
    chi2_reduced: float
    residuals_re: np.ndarray
    residuals_im: np.ndarray
    target_param: str
    target_value: float
    target_std: float
    converged: bool
    # Champs DRT : renseignés par drt/engine.py.
    #   drt_tau/drt_gamma      : distribution γ(τ) (τ en s, γ en Ω).
    #   drt_mode               : 'optimize' (MAP) ou 'sample' (HMC) — mode réellement
    #                            exécuté, affiché par l'UI pour ne pas comparer sans
    #                            le savoir des DRT de modes différents.
    #   drt_gamma_lo/drt_gamma_hi : bornes de crédibilité 2.5 / 97.5 % (mode 'sample'
    #                            uniquement ; None en 'optimize').
    drt_tau: Optional[np.ndarray] = None
    drt_gamma: Optional[np.ndarray] = None
    drt_mode: Optional[str] = None
    drt_gamma_lo: Optional[np.ndarray] = None
    drt_gamma_hi: Optional[np.ndarray] = None
    # Erreur de reconstruction relative RMS, √mean(|Z_fit − Z|²/|Z|²) (fit Orazem et DRT).
    reconstruction_error: Optional[float] = None
    # DRT (drt/engine.py) : erreur de reconstruction relative MAX, max_i |Z_fit − Z|/|Z|.
    # Champ DISTINCT de chi2_reduced (AUDIT.md DRT-5) : ce n'est pas un χ² pondéré.
    reconstruction_error_relative: Optional[float] = None
    # DRT (drt/engine.py) : réglages, diagnostics HMC (R-hat, ESS, divergences…),
    # alertes et notes — voir drt/diagnostics.py.
    drt_diagnostics: Optional[dict] = None
    # Fit Orazem (fits/orazem_fit.py) : méthode, conditionnement de la jacobienne,
    # identifiabilité, départs multiples, bornes actives — voir fit_spectrum().
    fit_diagnostics: Optional[dict] = None
    # Diagnostics d'ajustement remontés à l'UI (I7) : fit non convergé, χ²ᵣ hors
    # intervalle, résidu relatif élevé, paramètre en butée… Messages lisibles.
    warnings: list = field(default_factory=list)
    # Provenance de la structure d'erreur ayant pondéré CE fit (Orazem) :
    # "characterized_now" → coefficients estimés sur les réplicats du groupe, dans
    # cette analyse (aucune structure n'est plus jamais relue d'un fichier).
    error_structure_source: Optional[str] = None
    error_structure_timestamp: Optional[str] = None   # horodatage de la caractérisation utilisée
    error_structure_coeffs: Optional[dict] = None      # coefficients de la structure
    # Intervalle attendu de χ²ᵣ sous H0 (modèle et structure d'erreur corrects), au
    # niveau 95,45 % (2σ) : afficher « χ²ᵣ = 1,31 ∈ [0,71 ; 1,34] » rend le test
    # d'adéquation lisible sans analyser un texte.
    chi2_reduced_ci: Optional[tuple] = None
