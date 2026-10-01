"""Data models for EIS Analyzer."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

@dataclass
class EISSpectrum:
    """Single EIS spectrum with metadata.

    Attributes:
        label: Display name.
        f: Frequency array (Hz), sorted HF→BF.
        Zre: Real impedance (Ω).
        Zim: Imaginary impedance (Ω), positive convention (semicircle above real axis).
        concentration: Analyte concentration (mol/L); 0.0 for bare/probe.
        step: Measurement step: 'bare', 'probe', or 'hybridization'.
        n_points: Number of frequency points.
        source_files: Original filenames contributing to this spectrum.
        fit_results: Dict mapping model name to FitResult (populated by pipeline).
    """

    label: str
    f: np.ndarray
    Zre: np.ndarray
    Zim: np.ndarray
    concentration: float
    step: str
    n_points: int
    source_files: list = field(default_factory=list)
    fit_results: dict = field(default_factory=dict)
    # Validation KK — renseigné par core/validator.py
    validation: Optional[object] = None       # ValidationResult (évite import circulaire)
    sigma_re: Optional[object] = None         # np.ndarray σ_re(f) inter-réplicats (BRUT, sans plancher)
    sigma_im: Optional[object] = None         # np.ndarray σ_im(f) inter-réplicats (BRUT, sans plancher)
    n_replicates: Optional[int] = None        # nb de réplicats moyennés (caractérisation structure d'erreur)
    replicates: Optional[list] = None         # réplicats individuels (option voigt_based ; None si non conservés)
    f_min_valid: Optional[float] = None       # Hz — borne basse KK-valide
    f_max_valid: Optional[float] = None       # Hz — borne haute KK-valide

    def __post_init__(self):
        self.n_points = len(self.f)


@dataclass
class CVCurve:
    """Courbe voltammétrique légère issue du parseur EC-Lab robuste.

    Structure minimale (Ewe, I, label) destinée au routage des fichiers CV
    détectés par core.robust_loader.parse_eclab_file lorsqu'aucun traitement CV
    complet n'est requis : elle permet de stocker la courbe en session sans la
    tracer. Le courant `I` est déjà en ampères (conversion d'unité appliquée par
    le parseur). Pour l'analyse CV complète (concentration, step, réplicats,
    delta_signal), utiliser core.cv_models.CVScan.

    Attributes:
        Ewe: Potentiel appliqué (V).
        I: Courant mesuré (A) — déjà converti en ampères par le parseur.
        label: Nom d'affichage (typiquement le nom de fichier).
    """

    Ewe: np.ndarray
    I: np.ndarray
    label: str


@dataclass
class FitResult:
    """Result of a fit model applied to an EIS spectrum.

    Zfit_im follows the same positive convention as EISSpectrum.Zim
    (i.e. -Im(Z) > 0 for a capacitive semicircle). ``residuals_*`` = données − modèle,
    dans cette même convention.

    Paramètre cible (signal de calibration) : le circuit étant libre, il n'y a plus
    de champ « Rct » figé. L'utilisateur DÉSIGNE le paramètre qui sert de signal
    (``target_param``, un nom de ``params``) ; ``target_value``/``target_std`` en
    sont la valeur et l'écart-type (intra-fit pour un spectre ; pour la DRT, Rct
    extrait de la distribution et son incertitude).
    """

    model_name: str
    params: dict
    params_std: dict
    Zfit_re: np.ndarray
    Zfit_im: np.ndarray
    # χ² réduit pondéré = Σ(w·Δ²)/(2N−P), avec les poids w = 1/σ² de la structure
    # d'erreur d'Orazem. Les poids étant de vraies 1/variance (absolute_sigma TOUJOURS),
    # chi2_reduced≈1 EST un test d'adéquation modèle+erreur (voir chi2_reduced_ci).
    chi2_reduced: float
    residuals_re: np.ndarray
    residuals_im: np.ndarray
    target_param: str
    target_value: float
    target_std: float
    converged: bool
    # Champs DRT : renseignés par le plugin fits/drt_fit.py et par drt/engine.py.
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
    reconstruction_error: Optional[float] = None
    # DRT (drt/engine.py) : erreur de reconstruction relative MAX, max_i |Z_fit − Z|/|Z|.
    # Champ DISTINCT de chi2_reduced (AUDIT.md DRT-5) : ce n'est pas un χ² pondéré ;
    # le moteur DRT met chi2_reduced = NaN.
    reconstruction_error_relative: Optional[float] = None
    # DRT (drt/engine.py) : réglages, diagnostics HMC (R-hat, ESS, divergences…),
    # alertes et notes — voir drt/diagnostics.py.
    drt_diagnostics: Optional[dict] = None
    # Fit Orazem (fits/orazem_fit.py) : méthode, conditionnement de la jacobienne,
    # identifiabilité, départs multiples, bornes actives — voir fit_spectrum().
    fit_diagnostics: Optional[dict] = None
    # Verdict Lin-KK attaché par l'ANCIEN pipeline (core/pipeline._run_kk). Conservés
    # jusqu'à la bascule (étape 5) : le verdict KK de référence est désormais celui
    # du measurement model, porté par le GROUPE (MeasurementModelAnalysis), pas par
    # chaque fit.
    kk_passed: Optional[bool] = None
    kk_residuals: Optional[dict] = None
    # Diagnostics d'ajustement remontés à l'UI (I7) : fit non convergé, χ²ᵣ hors
    # intervalle, résidu relatif élevé, paramètre en butée… Messages lisibles.
    warnings: list = field(default_factory=list)
    # Provenance de la structure d'erreur ayant pondéré CE fit (Orazem) :
    #   "characterized_now"  → coefficients estimés sur les réplicats de ce jeu ;
    #   "reused_persisted"   → (ANCIEN module fits/error_structure.py uniquement)
    #                          coefficients rechargés d'une caractérisation antérieure.
    error_structure_source: Optional[str] = None
    error_structure_timestamp: Optional[str] = None   # horodatage de la caractérisation utilisée
    error_structure_coeffs: Optional[dict] = None      # coefficients de la structure
    # Intervalle attendu de χ²ᵣ sous H0 (modèle et structure d'erreur corrects), au
    # niveau 95,45 % (2σ). Conservé pour la nouvelle UI : afficher « χ²ᵣ = 1,31 ∈
    # [0,71 ; 1,34] » rend le test d'adéquation lisible sans analyser un texte.
    chi2_reduced_ci: Optional[tuple] = None


@dataclass
class ConcentrationGroup:
    """A spectrum and its fit results, grouped by analyte concentration."""

    concentration: float
    spectrum: EISSpectrum
    fit_results: dict = field(default_factory=dict)
    # NOTE (ajout hors périmètre initial — réorganisation onglets EIS) :
    # liste des spectres individuels (réplicats, avant moyenne), chacun avec
    # son propre `fit_results` rempli par core/pipeline.py. Champ optionnel,
    # vide par défaut, pour ne casser aucun code existant qui ignore ce champ.
    replicate_spectra: list = field(default_factory=list)


@dataclass
class EISSession:
    """Full analysis session state, stored in st.session_state['session']."""

    created_at: datetime = field(default_factory=datetime.now)
    bare: Optional[EISSpectrum] = None
    probe: Optional[EISSpectrum] = None
    groups: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    # NOTE (ajout hors périmètre initial) : réplicats individuels (avant moyenne)
    # pour bare/probe, avec fit_results par réplicat. Optionnel, vide par défaut.
    bare_replicate_spectra: list = field(default_factory=list)
    probe_replicate_spectra: list = field(default_factory=list)
    # Référence « électrode nue » — AFFICHAGE SEUL, JAMAIS utilisée dans les
    # calculs (ni fit, ni θ_EIS, ni normalisation, ni calibration, ni export de
    # valeurs calculées). Champ dédié et séparé de `bare`/`probe`/`groups` afin
    # que le pipeline soit structurellement incapable de la lire : elle est
    # attachée à la session APRÈS l'analyse et seulement superposée au Nyquist.
    bare_reference: Optional[EISSpectrum] = None
