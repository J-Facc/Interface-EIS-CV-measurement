"""Tests for fit models in fits/."""

import numpy as np
import pytest

from fits.physics import Z_randles_full
from fits.circular_fit import CircularFitModel
from fits.drt_fit import DRTFitModel
from core.models import EISSpectrum


# ── Synthetic spectrum factory ─────────────────────────────────────────────────

def _randles_spectrum(
    Re: float = 500.0,
    Rct: float = 5000.0,
    Qdl: float = 1e-6,
    alpha: float = 0.85,
    Re_prime: float = 50.0,
    Cb: float = 1e-9,
    ZD0: float = 300.0,
    n: int = 40,
) -> EISSpectrum:
    """Generate a noiseless synthetic Randles spectrum."""
    omega = np.logspace(1, 5, n)
    f = omega / (2.0 * np.pi)
    Z = Z_randles_full(
        omega, Re, Re_prime, Cb, Rct, Qdl, alpha, ZD0,
        xe=30e-6, D=7.2e-10, Fv=5e-10, h=60e-6, d=300e-6,
    )
    # Store HF→BF (descending frequency)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        label="synthetic",
        f=f[idx], Zre=Z.real[idx], Zim=Z.imag[idx],
        concentration=1e-9, step="hybridization",
        n_points=n,
    )


def _zarc_spectrum(R: float = 3000.0, tau0: float = 1e-3, phi: float = 0.8,
                   n: int = 60) -> EISSpectrum:
    """Generate a synthetic ZARC (R || CPE) spectrum."""
    omega = np.logspace(-1, 5, n)
    f = omega / (2.0 * np.pi)
    Z = R / (1.0 + (1j * omega * tau0) ** phi)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        label="zarc",
        f=f[idx], Zre=Z.real[idx], Zim=Z.imag[idx],
        concentration=1e-9, step="hybridization",
        n_points=n,
    )


# ── Circular fit tests ─────────────────────────────────────────────────────────

def test_circular_fit_converged():
    sp = _randles_spectrum()
    result = CircularFitModel().fit(sp, {})
    assert result.converged


def test_circular_fit_rct_within_10_percent():
    true_Rct = 5000.0
    sp = _randles_spectrum(Rct=true_Rct)
    result = CircularFitModel().fit(sp, {})
    rel_err = abs(result.Rct - true_Rct) / true_Rct
    assert rel_err < 0.10, (
        f"Circular fit Rct = {result.Rct:.0f} Ω, expected {true_Rct} Ω (±10%)"
    )


def test_circular_fit_returns_finite_arrays():
    sp = _randles_spectrum()
    result = CircularFitModel().fit(sp, {})
    assert np.all(np.isfinite(result.Zfit_re))
    assert np.all(np.isfinite(result.Zfit_im))


def test_circular_fit_params_keys():
    sp = _randles_spectrum()
    result = CircularFitModel().fit(sp, {})
    assert {"xc", "yc", "r", "Rct"} <= set(result.params.keys())


# ── DRT FFT tests ─────────────────────────────────────────────────────────────

_DRT_CONFIG = {
    "fit": {
        "drt": {
            "n_tau": 40,
            "tau_min": 1e-5,
            "tau_max": 10.0,
            "lambda_auto": True,
        }
    }
}


def test_drt_returns_positive_rct():
    sp = _zarc_spectrum()
    result = DRTFitModel().fit(sp, _DRT_CONFIG)
    assert result.Rct > 0


def test_drt_gamma_non_zero():
    sp = _zarc_spectrum()
    result = DRTFitModel().fit(sp, _DRT_CONFIG)
    gamma = np.array(result.params["gamma"])
    assert gamma.max() > 0, "DRT should have at least one non-zero value"


def test_drt_tau_length_matches_n_tau():
    sp = _zarc_spectrum()
    result = DRTFitModel().fit(sp, _DRT_CONFIG)
    assert len(result.params["tau"]) == _DRT_CONFIG["fit"]["drt"]["n_tau"]


def test_drt_fit_arrays_finite():
    sp = _zarc_spectrum()
    result = DRTFitModel().fit(sp, _DRT_CONFIG)
    assert np.all(np.isfinite(result.Zfit_re))
    assert np.all(np.isfinite(result.Zfit_im))
