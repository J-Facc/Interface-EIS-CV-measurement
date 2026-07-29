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
    (i.e. -Im(Z) > 0 for a capacitive semicircle).
    """

    model_name: str
    params: dict
    params_std: dict
    Zfit_re: np.ndarray
    Zfit_im: np.ndarray
    # χ² réduit pondéré = Σ(w·Δ²)/(2N−P), avec les poids w = 1/σ² issus de la
    # structure d'erreur d'Orazem (fits/error_structure.py). Les poids étant de
    # vraies 1/variance (absolute_sigma=True TOUJOURS), chi2_reduced≈1 EST un vrai
    # test d'adéquation modèle+erreur (cf. chi2_is_valid_test / chi2_reduced_ci).
    chi2_reduced: float
    residuals_re: np.ndarray
    residuals_im: np.ndarray
    Rct: float
    Rct_std: float
    converged: bool
# Incertitude sur Rct propagée depuis σ(f)
    # Renseigné dans un second temps (sprint 2)
    Rct_sigma: Optional[float] = None
    # Champs DRT : renseignés par le plugin fits/drt_fit.py (DRTBayesModel), qui
    # produit un FitResult standard comme les autres modèles du pipeline.
    #   drt_tau/drt_gamma      : distribution γ(τ) (τ en s, γ en Ω).
    #   drt_S/drt_lnGamma      : ln(τ) et ln(γ) précalculés (tracé ln/ln).
    #   drt_mode               : 'optimize' (MAP) ou 'sample' (HMC) — mode réellement
    #                            exécuté, affiché par l'UI pour ne pas comparer sans
    #                            le savoir des DRT de modes différents.
    #   drt_gamma_lo/drt_gamma_hi : bornes de crédibilité 2.5 / 97.5 % (mode 'sample'
    #                            uniquement ; None en 'optimize').
    drt_tau: Optional[np.ndarray] = None
    drt_gamma: Optional[np.ndarray] = None
    drt_S: Optional[np.ndarray] = None
    drt_lnGamma: Optional[np.ndarray] = None
    drt_mode: Optional[str] = None
    drt_gamma_lo: Optional[np.ndarray] = None
    drt_gamma_hi: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None
    # Validation Kramers-Kronig (fits/kk_validation.py)
    kk_passed: Optional[bool] = None
    kk_residuals: Optional[dict] = None
    # Diagnostics d'ajustement remontés à l'UI (I7) : fit non convergé, résidu
    # relatif élevé, paramètre en butée sur une borne. Liste de messages lisibles.
    warnings: list = field(default_factory=list)
    # Provenance de la structure d'erreur ayant pondéré CE fit (Orazem) :
    #   "characterized_now"  → coefficients estimés sur les réplicats de ce jeu ;
    #   "reused_persisted"   → coefficients rechargés d'une caractérisation antérieure.
    # L'UI DOIT afficher cette provenance (savoir si σ a été mesuré sur ce jeu).
    error_structure_source: Optional[str] = None
    error_structure_timestamp: Optional[str] = None   # horodatage de la caractérisation utilisée
    error_structure_coeffs: Optional[dict] = None      # {alpha,beta,gamma,delta,R_m}
    # chi2_reduced est-il un vrai test d'adéquation ? TOUJOURS True désormais
    # (poids = 1/σ² de la structure d'erreur, absolute_sigma=True).
    chi2_is_valid_test: bool = False
    # Intervalle attendu du χ²_red sous H0 : ~[1 − 2√(2/dof), 1 + 2√(2/dof)].
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
