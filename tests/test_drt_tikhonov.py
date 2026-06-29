"""Tests for fits/drt_tikhonov.py — DRT model-free Tikhonov(ordre2)+NNLS."""

import numpy as np
import pytest

from fits.drt_tikhonov import DRTTikhonovModel
from fits.physics import Z_randles_full
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


def _randles_spectrum(Re=20.0, Re_prime=5.0, Cb=1e-7, Rct=800.0, Qdl=1e-6,
                       alpha=0.9, R_D=200.0, tau_d=0.05, n=40, noise=0.02,
                       seed=0) -> EISSpectrum:
    """Synthetic 8-parameter Randles spectrum, sparsely sampled (~30-50 pts)
    with a realistic gaussian noise floor on Zre/Zim — close to a real EIS
    measurement, unlike the clean analytic R//C case above."""
    rng = np.random.default_rng(seed)
    f = np.geomspace(0.1, 1e5, n)
    omega = 2.0 * np.pi * f
    Z = Z_randles_full(omega, Re, Re_prime, Cb, Rct, Qdl, alpha, R_D, tau_d)
    Zre, Zim = Z.real, -Z.imag
    Zmag = np.abs(Z)
    Zre = Zre + rng.normal(0, noise * Zmag, n)
    Zim = Zim + rng.normal(0, noise * Zmag, n)
    return EISSpectrum(
        label="randles", f=f, Zre=Zre, Zim=Zim,
        concentration=1e-9, step="hybridization", n_points=n,
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_drt_no_periodic_comb_on_noisy_real_like_randles_spectrum(seed):
    """Non-regression: on a sparse (~40 pt), noisy (2%) Randles-derived
    spectrum — representative of a real EIS measurement, not the clean
    analytic case — gamma(tau) must stay a small number of smooth humps,
    not a periodic comb of isolated NNLS spikes (the under-regularization
    bug previously caused by an L-curve lambda selection landing in the
    near-degenerate low-lambda boundary, compounded by a tau grid much
    finer than the data could resolve)."""
    sp = _randles_spectrum(n=40, noise=0.02, seed=seed)
    result = DRTTikhonovModel().fit(sp, {"fit": {"drt": {"lambda_auto": True}}})
    gamma = np.asarray(result.drt_gamma)

    # Count contiguous runs whose amplitude exceeds 5% of the global max: a
    # comb produces many short isolated runs of comparable amplitude; a
    # well-regularized DRT produces at most 1-2 genuine humps (Rct||CPE peak,
    # possibly a diffusion peak) plus, at most, a negligible low-amplitude
    # edge ripple (<5% of the peak) that this threshold filters out.
    mask = gamma > gamma.max() * 0.05
    n_runs = int(np.sum(mask[1:] & ~mask[:-1])) + (1 if mask[0] else 0)

    assert n_runs <= 2, f"gamma(tau) a {n_runs} pics significatifs — motif en peigne suspect"
    assert np.all(gamma >= -1e-9)
    assert np.isfinite(result.reconstruction_error)
    assert result.reconstruction_error < 0.4
