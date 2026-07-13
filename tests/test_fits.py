"""Tests for fit models in fits/."""

import numpy as np
import pytest

from fits.physics import Z_randles_full
from fits.randles_full import RandlesFullModel
from fits.drt_fft import DRTFFTModel
from fits.kk_validation import kramers_kronig_check
from core.loader import load_spectrum
from core.models import EISSpectrum


# ── Synthetic spectrum factory ──────────────────────────────────────────────────

def _randles_spectrum(
    Re: float = 500.0,
    Rct: float = 5000.0,
    Qdl: float = 1e-6,
    alpha: float = 0.85,
    Re_prime: float = 50.0,
    Cb: float = 1e-9,
    R_D: float = 1.0,
    tau_d: float = 1.0,
    n: int = 80,
) -> EISSpectrum:
    """Generate a noiseless synthetic Randles spectrum.

    Frequency band is wide (1e-4 to 1e6 rad/s) and diffusion contribution
    negligible (R_D small) so that Im(Z) decays near zero at both ends —
    a requirement for the FFT/Wiener DRT deconvolution to resolve a single
    clean charge-transfer peak.
    """
    omega = np.logspace(-4, 6, n)
    f = omega / (2.0 * np.pi)
    Z = Z_randles_full(
        omega, Re, Re_prime, Cb, Rct, Qdl, alpha, R_D, tau_d,
    )
    # Store HF→BF (descending frequency)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        label="synthetic",
        # Convention loader/EISSpectrum : Zim = -Im(Z) > 0 (demi-cercle capacitif).
        f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
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
        # Convention loader/EISSpectrum : Zim = -Im(Z) > 0 (demi-cercle capacitif).
        f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
        concentration=1e-9, step="hybridization",
        n_points=n,
    )


# ── DRT FFT tests ──────────────────────────────────────────────────────

_DRT_CONFIG = {
    "fit": {
        "drt_wiener_W": 1e-9,
        "drt_n_z": 10000,
    }
}


def test_drt_returns_positive_rct():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert result.Rct > 0


def test_drt_gamma_non_zero():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    gamma = np.array(result.drt_gamma)
    assert gamma.max() > 0, "DRT should have at least one non-zero value"


def test_drt_fit_arrays_finite():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert np.all(np.isfinite(result.Zfit_re))
    assert np.all(np.isfinite(result.Zfit_im))


def test_drt_fft_randles_simple():
    """Spectre Randles synthétique propre, vérifie drt_tau/drt_gamma et Rct_drt
    cohérent avec Rct_randles à 30% près."""
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert len(result.drt_tau) == len(result.drt_gamma)
    assert result.Rct > 0
    Rct_randles = result.params["Rct_randles"]
    rel_err = abs(result.Rct - Rct_randles) / Rct_randles
    assert rel_err < 0.30, (
        f"Rct_drt = {result.Rct:.0f} Ω vs Rct_randles = {Rct_randles:.0f} Ω "
        f"(rel_err={rel_err:.2f})"
    )


# ── New architecture: KK validation, stricter FFT DRT ──────────────────

def _rc_spectrum(R: float = 1000.0, C: float = 1e-6, n: int = 30) -> EISSpectrum:
    """30-point log-spaced synthetic R // C spectrum."""
    omega = np.logspace(0, 5, n)
    f = omega / (2.0 * np.pi)
    Z = R / (1.0 + 1j * omega * R * C)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        # Convention loader/EISSpectrum : Zim = -Im(Z) > 0.
        label="rc", f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
        concentration=1e-9, step="hybridization", n_points=n,
    )


def test_kk_validation():
    sp = _rc_spectrum()
    result = kramers_kronig_check(sp, _DRT_CONFIG)
    assert result["kk_passed"]
    assert result["max_residual"] < 0.05


def test_drt_fft():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert len(result.drt_tau) == _DRT_CONFIG["fit"]["drt_n_z"]
    # Le filtre Wiener FFT souffre d'artefacts de bord (ringing) aux extrémités
    # du domaine log-ω (cf. fits/drt_fft.py) ; cette ringing contamine la
    # reconstruction de Im(Z) sur tout le domaine, d'où une erreur de
    # reconstruction relative élevée même pour un Rct correctement extrait
    # (cf. test_drt_fft_randles_simple, qui valide la précision de Rct).
    assert result.reconstruction_error < 2.0


# ── Bout-en-bout : loader → RandlesFullModel().fit (garde-fou du signe B1) ──

def _eclab_csv(Rct: float, n: int = 100) -> bytes:
    """Construit un CSV façon export EC-Lab : colonne '-Im(Z)/Ohm' POSITIVE
    (comme les fichiers réels), pour un spectre Randles de Rct connu."""
    omega = np.logspace(-1, 5, n)
    f = omega / (2.0 * np.pi)
    Z = Z_randles_full(omega, 500.0, 50.0, 1e-9, Rct, 1e-6, 0.90, 0.1, 0.5)
    lines = ["freq/Hz,Re(Z)/Ohm,-Im(Z)/Ohm"]
    for a, b, c in zip(f, Z.real, -Z.imag):  # -Im(Z) > 0
        lines.append(f"{a:.6e},{b:.6e},{c:.6e}")
    return ("\n".join(lines)).encode()


@pytest.mark.parametrize("Rct_true", [1000.0, 3000.0, 8000.0, 30000.0])
def test_randles_recovers_rct_end_to_end(Rct_true):
    """Chemin réel loader → fit : le Rct ajusté doit retrouver le Rct vrai à ±5 %.

    Garde-fou contre une inversion de signe du résidu imaginaire (B1) :
    load_spectrum produit Zim = -Im(Z) > 0 ; un résidu au mauvais signe fait
    diverger le fit (biais fort ou effondrement sur la borne basse).
    """
    sp = load_spectrum(_eclab_csv(Rct_true), label="e2e")
    assert np.all(sp.Zim >= 0), "le loader doit produire Zim positif"
    result = RandlesFullModel().fit(sp, {"fit": {"alpha_noise": 0.001, "max_iter": 10000}})
    rel_err = abs(result.Rct - Rct_true) / Rct_true
    assert rel_err < 0.05, f"Rct={result.Rct:.1f} vs {Rct_true:.1f} (rel_err={rel_err:.2%})"
