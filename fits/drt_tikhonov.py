# -*- coding: utf-8 -*-
"""
fits/drt_tikhonov.py — Distribution of Relaxation Times par régularisation
Tikhonov + NNLS (Ciucci & Chen 2015 / Hahn et al.).

Pipeline :
  1. Grille log(τ) fixe (n_z points) couvrant 1/ω_max → 1/ω_min.
  2. Matrices de noyau A_re, A_im (Re/Im de 1/(1+jωτ)) + colonne Re série.
  3. Résolution régularisée : min ||A·γ - Z||² + λ²||L·γ||²  sous γ ≥ 0
     via NNLS sur le système augmenté [A; λL] / [b; 0].
  4. Reconstruction de Z, calcul de l'erreur de reconstruction normalisée,
     extraction de Rct = Σ γ·Δ(ln τ).
"""

import numpy as np
from scipy.optimize import nnls

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


def _cfg_get(config, key: str, default):
    fit_cfg = config.get("fit", config) if isinstance(config, dict) else getattr(config, "fit", config)
    if isinstance(fit_cfg, dict):
        return fit_cfg.get(key, default)
    return getattr(fit_cfg, key, default)


class DRTTikhonovModel(BaseFitModel):
    """DRT par régularisation Tikhonov + NNLS."""

    name         = "drt_tikhonov"
    method       = "drt_tikhonov"
    display_name = "DRT Tikhonov + NNLS"

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config, weights=None) -> FitResult:
        f_exp   = np.asarray(spectrum.f,   dtype=float)
        Zre_exp = np.asarray(spectrum.Zre, dtype=float)
        Zim_exp = np.asarray(spectrum.Zim, dtype=float)

        n_z    = int(_cfg_get(config, "drt_n_z", 10000))
        n_tau  = min(n_z, 200)  # grille DRT plus grossière que la grille FFT (système dense n×n_tau)
        lam    = float(_cfg_get(config, "drt_lambda", 1e-3))

        order  = np.argsort(f_exp)
        f      = f_exp[order]
        omega  = 2.0 * np.pi * f
        Zre    = Zre_exp[order]
        Zim    = Zim_exp[order]  # EISSpectrum stocke Im(Z) brut (< 0 pour un circuit dissipatif)
        Z      = Zre + 1j * Zim

        tau = np.geomspace(1.0 / omega.max(), 1.0 / omega.min(), n_tau)
        S   = np.log(tau)

        wt     = omega[:, None] * tau[None, :]
        denom  = 1.0 + wt ** 2
        A_re   = 1.0 / denom
        A_im   = -wt / denom

        n = len(f)
        A_re_aug = np.column_stack([np.ones(n), A_re])
        A_im_aug = np.column_stack([np.zeros(n), A_im])
        A = np.vstack([A_re_aug, A_im_aug])
        b = np.concatenate([Z.real, Z.imag])

        # Régularisation Tikhonov d'ordre 0 sur gamma (NNLS sur système augmenté)
        L = np.eye(n_tau + 1)
        L[0, 0] = 0.0  # ne pas régulariser Re série
        A_reg = np.vstack([A, lam * L])
        b_reg = np.concatenate([b, np.zeros(n_tau + 1)])

        x, _ = nnls(A_reg, b_reg)
        Re_series = float(x[0])
        gamma = x[1:]

        Zfit_re_sorted = A_re_aug @ x
        Zfit_im_sorted = A_im_aug @ x

        inv_order = np.argsort(order)
        Zfit_re = Zfit_re_sorted[inv_order]
        Zfit_im = Zfit_im_sorted[inv_order]

        residuals_re = Zre_exp - Zfit_re
        residuals_im = Zim_exp - Zfit_im

        Zmod2 = Zre_exp ** 2 + Zim_exp ** 2 + 1e-30
        chi2 = float(np.mean((residuals_re ** 2 + residuals_im ** 2) / Zmod2))

        Zmod = np.sqrt(Zmod2)
        reconstruction_error = float(np.mean(
            np.sqrt(residuals_re ** 2 + residuals_im ** 2) / Zmod
        ))

        # Chaque gamma_k est directement une résistance R_k (discrétisation par
        # éléments de Voigt) — Rct = somme des résistances, pas une intégrale.
        Rct = float(np.sum(gamma))
        tau_max = float(tau[np.argmax(gamma)]) if gamma.max() > 0 else float(tau[len(tau) // 2])

        params = {"Rct": Rct, "Re_series": Re_series, "lambda": lam, "tau_max": tau_max}

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
