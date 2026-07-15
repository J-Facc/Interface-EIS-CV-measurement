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
    chi2: float
    residuals_re: np.ndarray
    residuals_im: np.ndarray
    Rct: float
    Rct_std: float
    converged: bool
# Incertitude sur Rct propagée depuis σ(f)
    # Renseigné dans un second temps (sprint 2)
    Rct_sigma: Optional[float] = None
    # DRT (Tikhonov+NNLS ou FFT Wiener) — renseigné par fits/drt_tikhonov.py / fits/drt_fft.py
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
