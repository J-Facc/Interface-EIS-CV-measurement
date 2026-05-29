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


@dataclass
class ConcentrationGroup:
    """A spectrum and its fit results, grouped by analyte concentration."""

    concentration: float
    spectrum: EISSpectrum
    fit_results: dict = field(default_factory=dict)


@dataclass
class EISSession:
    """Full analysis session state, stored in st.session_state['session']."""

    created_at: datetime = field(default_factory=datetime.now)
    bare: Optional[EISSpectrum] = None
    probe: Optional[EISSpectrum] = None
    groups: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
