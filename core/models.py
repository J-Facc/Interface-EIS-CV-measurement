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
    """

    label: str
    f: np.ndarray
    Zre: np.ndarray
    Zim: np.ndarray
    concentration: float
    step: str
    n_points: int
    source_files: list = field(default_factory=list)

    def __post_init__(self):
        self.n_points = len(self.f)


@dataclass
class FitResult:
    """Result of a fit model applied to an EIS spectrum.

    Attributes:
        model_name: Short model identifier.
        params: Fitted parameter values {name: value}.
        params_std: Standard deviations {name: std}.
        Zfit_re: Real part of fitted impedance (Ω).
        Zfit_im: Imaginary part of fitted impedance (Ω).
        chi2: Mean squared residual (Ω²).
        residuals_re: Zre - Zfit_re (Ω).
        residuals_im: Zim - Zfit_im (Ω).
        Rct: Extracted charge transfer resistance (Ω).
        Rct_std: Standard deviation of Rct (Ω).
        converged: Whether the optimiser converged.
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
    """A spectrum and its fit results, grouped by analyte concentration.

    Attributes:
        concentration: Analyte concentration (mol/L).
        spectrum: Averaged EIS spectrum for this concentration.
        fit_results: Dict mapping model name to FitResult.
    """

    concentration: float
    spectrum: EISSpectrum
    fit_results: dict = field(default_factory=dict)


@dataclass
class EISSession:
    """Full analysis session state, stored in st.session_state['session'].

    Attributes:
        created_at: Timestamp of session creation.
        bare: Bare electrode spectrum (no probe, no target).
        probe: Probe-modified electrode spectrum.
        groups: Hybridization concentration groups, sorted ascending.
        config: App config dict used for this session.
    """

    created_at: datetime = field(default_factory=datetime.now)
    bare: Optional[EISSpectrum] = None
    probe: Optional[EISSpectrum] = None
    groups: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
