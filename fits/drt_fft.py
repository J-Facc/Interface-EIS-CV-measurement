# -*- coding: utf-8 -*-
"""
fits/drt_fft.py — Distribution of Relaxation Times par déconvolution FFT (Wiener)
Méthode : Bissessur et al., PRE 2026 (éq. 3), traduite fidèlement depuis le
notebook de référence DRTparFFT.ipynb.

Pipeline :
  1. Grille log-ω uniforme construite directement sur les fréquences
     expérimentales (pas de modèle Randles intermédiaire).
  2. Interpolation de Re(Z) et Im(Z) sur cette grille.
  3. Déconvolution de Fredholm dans l'espace de Fourier, filtre Wiener,
     avec détection explicite des bornes spectrales utiles (limeta1/limeta2).
  4. Reconstruction de Z pour le χ² et extraction de Rct par intégrale de |γ(s)|.
"""

import numpy as np
import scipy.fftpack as fftpack

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


def _TF(x: np.ndarray, f: np.ndarray):
    RECT = np.array([1 - 2 * (i % 2) for i in range(len(x))])
    L    = fftpack.fft(f.copy()) * RECT
    L    = fftpack.fftshift(L)
    nu   = fftpack.fftshift(fftpack.fftfreq(len(x), x[1] - x[0]))
    return nu, L


def _TF_inv(x: np.ndarray, f: np.ndarray):
    RECT = np.array([1 - 2 * (i % 2) for i in range(len(x))])
    L    = fftpack.ifft(f.copy()) * RECT
    L    = fftpack.fftshift(L)
    nu   = x.copy()
    nu   = fftpack.fftshift(fftpack.fftfreq(len(nu), nu[1] - nu[0]))
    return nu, L


def _cfg_get(config, key: str, default):
    fit_cfg = config.get("fit", config) if isinstance(config, dict) else getattr(config, "fit", config)
    if isinstance(fit_cfg, dict):
        return fit_cfg.get(key, default)
    return getattr(fit_cfg, key, default)


class DRTFFTModel(BaseFitModel):
    """DRT par déconvolution FFT directe (filtre Wiener, Bissessur et al. PRE 2026)."""

    name         = "drt_fft"
    method       = "drt_fft"
    display_name = "DRT FFT Wiener"

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config, weights=None) -> FitResult:
        """Calcule la DRT par déconvolution FFT Wiener et extrait Rct."""
        f_exp   = np.asarray(spectrum.f,   dtype=float)
        Zre_exp = np.asarray(spectrum.Zre, dtype=float)
        Zim_exp = np.asarray(spectrum.Zim, dtype=float)

        W   = float(_cfg_get(config, "drt_wiener_W", 1e-8))
        n_z = int(_cfg_get(config, "drt_n_z", 10000))

        # ── 1. Grille log-ω uniforme ─────────────────────────────────────────────────────────────
        omega_exp = 2.0 * np.pi * f_exp
        z_exp     = np.log(omega_exp)
        order     = np.argsort(z_exp)
        z_sorted  = z_exp[order]

        z1 = float(z_sorted[0])  - 1.0
        z2 = float(z_sorted[-1]) + 1.0
        DELTA  = (z1 + z2) / 2.0
        Z_grid = np.linspace(z1, z2, n_z)

        # ── 2. Interpolation Re(Z) / Im(Z) — convention Im(Z) positive ────────────────────────
        Zim_pos_sorted = -Zim_exp[order]
        Zre_sorted     = Zre_exp[order]

        imZ = np.interp(Z_grid, z_sorted, Zim_pos_sorted)
        reZ = np.interp(Z_grid, z_sorted, Zre_sorted)

        # ── 3. TF + filtre Wiener ───────────────────────────────────────────────────────
        ETA, imZ_eta = _TF(Z_grid, imZ)
        n_eta = len(ETA)

        LIM     = 31.0
        eta_lim = LIM / np.pi**2

        # Détection explicite des bornes spectrales utiles (limeta1, limeta2) :
        # au-delà de ces indices le filtre Wiener est numériquement nul (1e-100).
        idx_in_range = np.where(np.abs(ETA) < eta_lim)[0]
        if idx_in_range.size > 0:
            limeta1, limeta2 = int(idx_in_range[0]), int(idx_in_range[-1])
        else:
            limeta1, limeta2 = 0, -1

        conv = np.full(n_eta, 1e-100)
        if limeta2 >= limeta1:
            sl = slice(limeta1, limeta2 + 1)
            cosh_eta = np.cosh(ETA[sl] * np.pi**2)
            conv[sl] = cosh_eta / (1.0 + W * cosh_eta**2)

        CONV = (2.0 / np.pi) * imZ_eta * conv

        # ── 4. Retour dans l'espace ln(τ) ────────────────────────────────────────────────
        S_raw, H_s_raw = _TF_inv(ETA, CONV)
        S = S_raw + DELTA

        dS  = (z2 - z1) / n_z
        Rct = float(np.sum(np.abs(H_s_raw)) * dS)

        # ── 5. Reconstruction de Z pour le χ² ───────────────────────────────────────────────
        diff           = z_sorted[:, None] - S_raw[None, :]
        Zfit_im_pos    = np.real(np.sum(H_s_raw[None, :] / (2.0 * np.cosh(diff)), axis=1)) * dS
        Zfit_im_sorted = -Zfit_im_pos
        Zfit_re_sorted = np.interp(z_sorted, Z_grid, reZ)

        inv_order = np.argsort(order)
        Zfit_re = Zfit_re_sorted[inv_order]
        Zfit_im = Zfit_im_sorted[inv_order]

        residuals_re = Zre_exp - Zfit_re
        residuals_im = Zim_exp - Zfit_im

        Zmod2 = Zre_exp**2 + Zim_exp**2 + 1e-30
        chi2 = float(np.mean((residuals_re**2 + residuals_im**2) / Zmod2))
        reconstruction_error = float(np.mean(
            np.sqrt(residuals_re**2 + residuals_im**2) / np.sqrt(Zmod2)
        ))

        # ── 6. FitResult ─────────────────────────────────────────────────────────────
        gamma   = np.abs(H_s_raw)
        tau     = np.exp(S)

        # tau_max exclut les bords (artefacts de bord du filtre Wiener)
        margin = max(1, n_z // 20)
        core_slice = slice(margin, n_z - margin)
        tau_max = float(tau[core_slice][np.argmax(gamma[core_slice])])

        params = {"Rct": Rct, "W": W, "tau_max": tau_max}

        return FitResult(
            model_name=self.name,
            params=params,
            params_std={k: 0.0 for k in params},
            Zfit_re=Zfit_re,
            Zfit_im=Zfit_im,
            chi2=chi2,
            residuals_re=residuals_re,
            residuals_im=residuals_im,
            Rct=Rct,
            Rct_std=0.0,
            converged=True,
            drt_tau=tau,
            drt_gamma=gamma,
            drt_S=S,
            drt_lnGamma=np.log(np.maximum(gamma, 1e-300)),
            reconstruction_error=reconstruction_error,
        )
