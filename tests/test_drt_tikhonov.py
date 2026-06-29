"""Tests for fits/drt_tikhonov.py — DRT model-free Tikhonov(ordre2)+NNLS."""

import numpy as np
import pytest

from fits.drt_tikhonov import DRTTikhonovModel
from core.models import EISSpectrum

_CONFIG = {
    "fit": {
        "drt": {
            "n_tau": 50,
            "tau_min": 1e-6,
            "tau_max": 10.0,
            "lambda_auto": True,
        }
    }
}


def _rc_spectrum(R0: float = 50.0, R1: float = 1000.0, C1: float = 2e-6,
                  n: int = 40, noise: float = 0.0, seed: int = 0) -> EISSpectrum:
    """Synthetic R0 + (R1 // C1) spectrum — single Dirac DRT at tau0 = R1*C1."""
    rng = np.random.default_rng(seed)
    f = np.geomspace(0.1, 1e5, n)
    omega = 2.0 * np.pi * f
    tau0 = R1 * C1
    Z = R0 + R1 / (1.0 + 1j * omega * tau0)
    Zre = Z.real
    Zim = -Z.imag  # positive convention
    if noise > 0:
        Zre = Zre + rng.normal(0, noise * np.abs(Z), n)
        Zim = Zim + rng.normal(0, noise * np.abs(Z), n)
    return EISSpectrum(
        label="rc", f=f, Zre=Zre, Zim=Zim,
        concentration=1e-9, step="hybridization", n_points=n,
    )


def test_drt_tikhonov_peak_matches_analytic_tau0():
    """Single R0//C1+R1 Dirac: the gamma(tau) peak and Rct should be close
    to the theoretical tau0 = R1*C1 and R1."""
    R0, R1, C1 = 50.0, 1000.0, 2e-6
    tau0 = R1 * C1
    sp = _rc_spectrum(R0, R1, C1)
    result = DRTTikhonovModel().fit(sp, _CONFIG)

    tau_peak = np.exp(result.params["tau_Rct"])
    assert abs(np.log(tau_peak / tau0)) < 0.5, (
        f"tau_peak={tau_peak:.4g} vs tau0={tau0:.4g}"
    )
    rel_err_rct = abs(result.Rct - R1) / R1
    assert rel_err_rct < 0.10, f"Rct={result.Rct:.1f} vs R1={R1}"
    rel_err_r0 = abs(result.params["R0"] - R0) / R0
    assert rel_err_r0 < 0.10, f"R0={result.params['R0']:.1f} vs {R0}"


def test_drt_gamma_nonnegative():
    """NNLS constraint: gamma(tau) >= 0 everywhere."""
    sp = _rc_spectrum()
    result = DRTTikhonovModel().fit(sp, _CONFIG)
    assert np.all(np.asarray(result.drt_gamma) >= -1e-9)


def test_drt_reconstruction_finite_and_accurate():
    sp = _rc_spectrum()
    result = DRTTikhonovModel().fit(sp, _CONFIG)
    assert np.all(np.isfinite(result.Zfit_re))
    assert np.all(np.isfinite(result.Zfit_im))
    assert result.reconstruction_error < 0.05


@pytest.mark.parametrize("R0,R1,C1,noise", [
    (50.0, 1000.0, 2e-6, 0.0),
    (200.0, 5000.0, 5e-7, 0.01),
    (10.0, 300.0, 1e-5, 0.02),
])
def test_drt_lambda_auto_no_crash_on_noisy_data(R0, R1, C1, noise):
    """Automatic lambda selection (L-curve) should not diverge or produce
    NaN/Inf, even with added noise — model-free DRT is noise-sensitive but
    must stay numerically stable."""
    sp = _rc_spectrum(R0, R1, C1, noise=noise, seed=42)
    result = DRTTikhonovModel().fit(sp, _CONFIG)
    lam = result.params["lambda"]
    assert np.isfinite(lam) and lam > 0
    assert np.all(np.isfinite(result.drt_gamma))
    assert np.all(np.asarray(result.drt_gamma) >= -1e-9)
    assert np.isfinite(result.Rct)
    assert np.isfinite(result.reconstruction_error)
