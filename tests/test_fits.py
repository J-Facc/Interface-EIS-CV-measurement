"""Tests for fit models in fits/."""

import numpy as np
import pytest

from fits.physics import Z_randles_full
from fits.circular_fit import CircularFitModel
from fits.drt_fft import DRTFFTModel
from fits.drt_tikhonov import DRTTikhonovModel
from fits.kk_validation import kramers_kronig_check
from core.models import EISSpectrum


# ── Synthetic spectrum factory ──────────────────────────────────────────────────

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


# ── Circular fit tests ─────────────────────────────────────────────────

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


# ── DRT FFT tests ──────────────────────────────────────────────────────

_DRT_CONFIG = {
    "fit": {
        "drt_wiener_W": 1e-9,
        "drt_n_z": 10000,
    }
}


def test_drt_returns_positive_rct():
    sp = _zarc_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert result.Rct > 0


def test_drt_gamma_non_zero():
    sp = _zarc_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    gamma = np.array(result.drt_gamma)
    assert gamma.max() > 0, "DRT should have at least one non-zero value"


def test_drt_fit_arrays_finite():
    sp = _zarc_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert np.all(np.isfinite(result.Zfit_re))
    assert np.all(np.isfinite(result.Zfit_im))


def test_drt_fft_randles_simple():
    """Spectre Randles simple R0//C0, vérifie drt_tau/drt_gamma et chi2."""
    n = 20
    R0, C0 = 1000.0, 1e-6
    omega = np.logspace(0, 5, n)
    f = omega / (2.0 * np.pi)
    Z = R0 / (1.0 + 1j * omega * R0 * C0)
    idx = np.argsort(f)[::-1]
    sp = EISSpectrum(
        label="randles_simple",
        f=f[idx], Zre=Z.real[idx], Zim=Z.imag[idx],
        concentration=1e-9, step="hybridization",
        n_points=n,
    )
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert len(result.drt_tau) == len(result.drt_gamma)
    assert result.Rct > 0
    assert result.chi2 < 0.1


# ── New architecture: KK validation, Tikhonov DRT, stricter FFT DRT ──────────────────

def _rc_spectrum(R: float = 1000.0, C: float = 1e-6, n: int = 30) -> EISSpectrum:
    """30-point log-spaced synthetic R // C spectrum."""
    omega = np.logspace(0, 5, n)
    f = omega / (2.0 * np.pi)
    Z = R / (1.0 + 1j * omega * R * C)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        label="rc", f=f[idx], Zre=Z.real[idx], Zim=Z.imag[idx],
        concentration=1e-9, step="hybridization", n_points=n,
    )


def test_kk_validation():
    sp = _rc_spectrum()
    result = kramers_kronig_check(sp, _DRT_CONFIG)
    assert result["kk_passed"]
    assert result["max_residual"] < 0.05


def test_drt_tikhonov():
    sp = _rc_spectrum()
    result = DRTTikhonovModel().fit(sp, _DRT_CONFIG)
    assert result.Rct > 0
    assert result.reconstruction_error < 0.05


def test_drt_fft():
    sp = _rc_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert len(result.drt_tau) == _DRT_CONFIG["fit"]["drt_n_z"]
    # Le filtre Wiener FFT souffre d'artefacts de bord aux extrémités du
    # domaine log-ω (cf. fits/drt_fft.py) — tolérance plus large que
    # le modèle Tikhonov+NNLS, qui n'a pas cette limitation.
    assert result.reconstruction_error < 0.2
