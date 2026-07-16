# -*- coding: utf-8 -*-
"""Tests du moteur DRT (fits/drt_fit.py, wrapper bayes-drt2 vendoré).

(a) Aperçu ridge sur un spectre synthétique à 2 pics → pics aux bons τ.
(b) DRT bayésienne (HMC) — marquée ``slow``, sautée si cmdstan est absent —
    vérifie que γ_lo ≤ γ ≤ γ_hi partout et que Rp ∈ [Rp_lo, Rp_hi].

Toute la suite est sautée si l'extra DRT (cvxopt/cmdstanpy) n'est pas installé,
de sorte que la CI de base (requirements.txt léger) reste verte.
"""

import numpy as np
import pytest

from core.models import EISSpectrum
import fits.drt_fit as drt

pytestmark = pytest.mark.skipif(
    not drt.bayes_available(),
    reason=f"Moteur DRT (bayes-drt2) indisponible : {drt.import_error()}",
)


def _two_peak_spectrum(tau1: float = 1e-3, tau2: float = 1e-1,
                       R1: float = 50.0, R2: float = 80.0, R0: float = 10.0,
                       n: int = 71) -> EISSpectrum:
    """Deux circuits RC en série (deux pics DRT nets à τ1 et τ2).

    Convention loader : ``Zim = -Im(Z) > 0``.
    """
    f = np.logspace(5, -2, n)
    w = 2.0 * np.pi * f
    Z = R0 + R1 / (1.0 + 1j * w * tau1) + R2 / (1.0 + 1j * w * tau2)
    return EISSpectrum(
        label="2peaks", f=f, Zre=Z.real, Zim=-Z.imag,
        concentration=1e-9, step="probe", n_points=n,
    )


def _peak_taus(tau: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """τ des maxima locaux de γ(τ)."""
    from scipy.signal import find_peaks

    idx, _ = find_peaks(gamma)
    return np.asarray(tau)[idx]


def test_ridge_two_peaks_at_correct_tau():
    """L'aperçu ridge place un pic près de τ1 et un près de τ2 (< 0.2 décade)."""
    tau1, tau2 = 1e-3, 1e-1
    sp = _two_peak_spectrum(tau1=tau1, tau2=tau2)

    res = drt.drt_preview(sp)

    assert res.engine == "ridge"
    # Le ridge ne fournit pas d'intervalles.
    assert res.gamma_lo is None and res.gamma_hi is None
    assert res.Rp > 0

    peaks = _peak_taus(res.tau, res.gamma)
    assert len(peaks) >= 2, "au moins deux pics attendus"

    log_peaks = np.log10(peaks)
    assert np.min(np.abs(log_peaks - np.log10(tau1))) < 0.2, "pic manquant près de τ1"
    assert np.min(np.abs(log_peaks - np.log10(tau2))) < 0.2, "pic manquant près de τ2"


@pytest.mark.slow
@pytest.mark.skipif(not drt.cmdstan_available(), reason="cmdstan non installé (HMC indisponible)")
def test_bayes_intervals_bracket_median():
    """HMC : γ_lo ≤ γ ≤ γ_hi partout et Rp ∈ [Rp_lo, Rp_hi]."""
    sp = _two_peak_spectrum()

    res = drt.drt_bayes(sp, use_cache=False)

    assert res.engine == "bayes"
    assert res.has_intervals
    assert res.Rp_lo is not None and res.Rp_hi is not None

    # Tolérance numérique minime pour les égalités aux bornes.
    tol = 1e-9 * (np.max(res.gamma) + 1.0)
    assert np.all(res.gamma_lo <= res.gamma + tol), "γ_lo doit minorer γ partout"
    assert np.all(res.gamma <= res.gamma_hi + tol), "γ_hi doit majorer γ partout"
    assert res.Rp_lo <= res.Rp <= res.Rp_hi, "Rp doit tomber dans [Rp_lo, Rp_hi]"
