# -*- coding: utf-8 -*-
"""Tests du moteur DRT (fits/drt_fit.py — plugin bayes_drt2 / Inverter).

Deux niveaux :

(a) **Extraction Rct par pic (convention Bissessur)** — PUR, sans CmdStan ni
    bayes_drt2 : valide _local_maxima / _extract_rct_peak. Tourne en CI Linux.
(b) **Fit DRT réel** ('optimize' MAP, puis 'sample' HMC) — nécessite bayes_drt2 ET
    une installation CmdStan. Sauté proprement sinon, pour que la CI de base
    (requirements.txt léger, sans toolchain) reste verte.
"""

import numpy as np
import pytest

from core.models import EISSpectrum
import fits.drt_fit as drt

_HAVE_BAYES = drt.bayes_available()
_HAVE_CMDSTAN = _HAVE_BAYES and drt.cmdstan_available()


# ─────────────────────────────────────────────────────────────────────────────
# (a) Extraction Rct par pic — PUR (aucune dépendance runtime lourde)
# ─────────────────────────────────────────────────────────────────────────────
def _gaussian_bumps(centers, amps, sigma=0.6, n=400):
    """γ(τ) synthétique = somme de gaussiennes en ln(τ). τ croissant."""
    tau = np.logspace(-5, 2, n)          # s, croissant
    ln_tau = np.log(tau)
    gamma = np.zeros_like(ln_tau)
    for c, a in zip(centers, amps):
        gamma += a * np.exp(-0.5 * ((ln_tau - c) / sigma) ** 2)
    return tau, gamma


def test_extract_rct_penultimate_peak():
    """Deux pics → le pic PÉNULTIÈME (plus petit τ = arc de transfert) est retenu."""
    c_ct, c_diff = -6.0, 2.0            # τ_ct < τ_diff (transfert HF, diffusion BF)
    tau, gamma = _gaussian_bumps([c_ct, c_diff], [50.0, 90.0])

    Rct, tau_Rct_ln, source, warning = drt._extract_rct_peak(tau, gamma)

    assert source == "peak_penultimate"
    assert warning == ""
    assert Rct is not None and Rct > 0
    # tau_Rct (renvoyé en ln τ) doit tomber près du centre du pic de transfert.
    assert abs(tau_Rct_ln - c_ct) < 0.5
    # Rct ≈ aire du pic de transfert seul (gaussienne : a·σ·√(2π)), pas l'aire totale.
    expected = 50.0 * 0.6 * np.sqrt(2 * np.pi)
    assert abs(Rct - expected) / expected < 0.25


def test_extract_rct_single_peak_is_flagged():
    """Un seul pic → Rct extrait mais SIGNALÉ (source 'peak_single', warning)."""
    tau, gamma = _gaussian_bumps([-3.0], [40.0])

    Rct, _tau_ln, source, warning = drt._extract_rct_peak(tau, gamma)

    assert source == "peak_single"
    assert Rct is not None and Rct > 0
    assert warning  # non vide : l'UI doit pouvoir prévenir l'utilisateur


def test_extract_rct_no_peak_returns_none():
    """Aucun pic (γ ≡ 0) → pas d'extraction : (None, 'none') pour repli Rp signalé."""
    tau = np.logspace(-5, 2, 200)
    gamma = np.zeros_like(tau)

    Rct, _tau_ln, source, warning = drt._extract_rct_peak(tau, gamma)

    assert Rct is None
    assert source == "none"
    assert warning


def test_local_maxima_counts_bumps():
    """_local_maxima détecte le bon nombre de pics sur un signal propre."""
    _tau, gamma = _gaussian_bumps([-6.0, 2.0], [50.0, 90.0])
    maxima = drt._local_maxima(gamma, l=max(1, len(gamma) // 15), threshold=gamma.max() * 1e-3)
    assert len(maxima) == 2


# ─────────────────────────────────────────────────────────────────────────────
# (b) Fit DRT réel — nécessite bayes_drt2 + CmdStan
# ─────────────────────────────────────────────────────────────────────────────
def _two_peak_spectrum(tau1=1e-3, tau2=1e-1, R1=50.0, R2=80.0, R0=10.0, n=71):
    """Deux circuits RC en série (deux pics DRT nets à τ1 et τ2).

    Convention loader : ``Zim = -Im(Z) > 0``.
    """
    f = np.logspace(5, -2, n)          # HF→BF
    w = 2.0 * np.pi * f
    Z = R0 + R1 / (1.0 + 1j * w * tau1) + R2 / (1.0 + 1j * w * tau2)
    return EISSpectrum(
        label="2peaks", f=f, Zre=Z.real, Zim=-Z.imag,
        concentration=1e-9, step="probe", n_points=n,
    )


def _peak_taus(tau, gamma):
    from scipy.signal import find_peaks

    idx, _ = find_peaks(gamma)
    return np.asarray(tau)[idx]


@pytest.mark.skipif(not _HAVE_CMDSTAN, reason="bayes_drt2/CmdStan indisponible (fit DRT)")
def test_optimize_two_peaks_at_correct_tau():
    """Mode 'optimize' (MAP) : un pic près de τ1 et un près de τ2 ; FitResult cohérent."""
    tau1, tau2 = 1e-3, 1e-1
    sp = _two_peak_spectrum(tau1=tau1, tau2=tau2)

    fr = drt.DRTBayesModel().fit(sp, {"fit": {"drt": {"mode": "optimize"}}})

    assert fr.model_name == "drt_bayes"
    assert fr.drt_mode == "optimize"
    assert fr.drt_gamma_lo is None and fr.drt_gamma_hi is None  # pas d'IC en MAP
    assert fr.target_value > 0
    assert fr.params.get("rct_source") in ("peak_penultimate", "peak_single", "rp_fallback")
    assert len(fr.Zfit_re) == sp.n_points

    peaks = _peak_taus(fr.drt_tau, fr.drt_gamma)
    assert len(peaks) >= 2, "au moins deux pics attendus"
    log_peaks = np.log10(peaks)
    assert np.min(np.abs(log_peaks - np.log10(tau1))) < 0.3, "pic manquant près de τ1"
    assert np.min(np.abs(log_peaks - np.log10(tau2))) < 0.3, "pic manquant près de τ2"


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_CMDSTAN, reason="bayes_drt2/CmdStan indisponible (HMC)")
def test_sample_intervals_bracket_median():
    """Mode 'sample' (HMC) : γ_lo ≤ γ ≤ γ_hi partout, et drt_mode='sample'."""
    sp = _two_peak_spectrum()

    fr = drt.fit_drt(sp, mode="sample")

    assert fr.drt_mode == "sample"
    assert fr.drt_gamma_lo is not None and fr.drt_gamma_hi is not None
    tol = 1e-9 * (np.max(fr.drt_gamma) + 1.0)
    assert np.all(fr.drt_gamma_lo <= fr.drt_gamma + tol), "γ_lo doit minorer γ"
    assert np.all(fr.drt_gamma <= fr.drt_gamma_hi + tol), "γ_hi doit majorer γ"
