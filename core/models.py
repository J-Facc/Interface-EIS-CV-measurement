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
    sigma_re: Optional[object] = None         # np.ndarray σ_re(f) inter-réplicats
    sigma_im: Optional[object] = None         # np.ndarray σ_im(f) inter-réplicats
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
    # χ² réduit pondéré = Σ(w·Δ²)/(2N−P), avec les poids w effectivement utilisés
    # par le fit (cf. fits/randles_full.py). ATTENTION : sous pondération modulus
    # rééchelonnée (w = 1/(alpha_noise·|Z|)², alpha_noise arbitraire), chi2_reduced≈1
    # n'est PAS un test d'adéquation statistique — juste une métrique de misfit
    # relative comparable entre spectres (voir prompt C).
    chi2_reduced: float
    residuals_re: np.ndarray
    residuals_im: np.ndarray
    Rct: float
    Rct_std: float
    converged: bool
# Incertitude sur Rct propagée depuis σ(f)
    # Renseigné dans un second temps (sprint 2)
    Rct_sigma: Optional[float] = None
    # Champs DRT hérités (Optional, laissés None) : la DRT est désormais un
    # moteur dédié (fits/drt_fit.py → DRTResult), plus un fit du pipeline.
    # Conservés pour compat des lectures getattr(..., None) existantes.
    drt_tau: Optional[np.ndarray] = None
    drt_gamma: Optional[np.ndarray] = None
    drt_S: Optional[np.ndarray] = None
    drt_lnGamma: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None
    # Validation Kramers-Kronig (fits/kk_validation.py)
    kk_passed: Optional[bool] = None
    kk_residuals: Optional[dict] = None
    # Diagnostics d'ajustement remontés à l'UI (I7) : fit non convergé, résidu
    # relatif élevé, paramètre en butée sur une borne. Liste de messages lisibles.
    warnings: list = field(default_factory=list)


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
