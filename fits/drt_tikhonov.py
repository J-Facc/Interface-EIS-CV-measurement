# -*- coding: utf-8 -*-
"""
fits/drt_tikhonov.py — DRT model-free par Tikhonov (ordre 2) + NNLS.

Méthode de référence : Wan, Saccoccio, Chen, Ciucci, Electrochim. Acta 184,
483 (2015) (DRTtools), citée par Bissessur, Man, Gamby, Phys. Rev. E 113,
025502 (2026) (DOI: 10.1103/fn2s-z364), section III.B "DRT with DRTtools".

Contrairement à fits/drt_fft.py (DRT FFT/Wiener — section III.C du papier,
appliquée à un spectre IDÉAL reconstruit depuis un fit Randles), cette
méthode est appliquée DIRECTEMENT sur les données expérimentales brutes
(spectrum.f / Zre / Zim), sans aucun fit de circuit équivalent intermédiaire.
γ(τ) est donc complètement indépendante de toute hypothèse de topologie de
circuit — c'est la méthode "model-free" du papier.

Pipeline :
  1. Grille de τ log-espacée, dimensionnée sur la plage de fréquences
     expérimentales couvertes par le spectre (PAS une grille synthétique
     dense de 10000 points : les données réelles sont éparses).
  2. Matrice de discrétisation du noyau de Fredholm (A_re, A_im), fonctions
     de base de Dirac sur la grille de τ.
  3. Régularisation de Tikhonov d'ordre 2 (pénalise la dérivée seconde de γ)
     pour favoriser une distribution lisse plutôt que sparse.
  4. Sélection automatique de λ par L-curve (courbure max en log-log entre
     résidu et rugosité de la solution).
  5. Résolution par NNLS (scipy.optimize.nnls) : garantit γ(τ) ≥ 0 et R0 ≥ 0.
  6. Extraction de Rct selon la convention Bissessur (avant-dernier pic local
     si ≥2 pics, intégrale trapèze de γ(τ) sur ±3 en ln(τ) autour du pic).
  7. Reconstruction de Z(ω) en réinjectant γ(τ) ENTIÈRE dans le noyau de
     Fredholm (pas de lien structurel avec un circuit Randles).
"""

import numpy as np
from scipy.optimize import nnls

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


def _cfg_get(config, key: str, default):
    fit_cfg = config.get("fit", config) if isinstance(config, dict) else getattr(config, "fit", config)
    if isinstance(fit_cfg, dict):
        drt_cfg = fit_cfg.get("drt", {})
    else:
        drt_cfg = getattr(fit_cfg, "drt", {})
    if isinstance(drt_cfg, dict) and key in drt_cfg:
        return drt_cfg[key]
    return getattr(drt_cfg, key, default) if not isinstance(drt_cfg, dict) else default


def _local_maxima(gamma: np.ndarray, l: int = 3, threshold: float = 0.0) -> list:
    """Detect indices of local maxima of gamma within a sliding window.

    Args:
        gamma: γ(τ) array (≥0).
        l: Half-window size (indices).
        threshold: Minimum amplitude for a maximum to be retained.

    Returns:
        Sorted list of indices of local maxima, ascending in τ.
    """
    n = len(gamma)
    maxima = []
    for i in range(n):
        if gamma[i] <= threshold:
            continue
        lo = max(0, i - l)
        hi = min(n, i + l + 1)
        if gamma[i] >= gamma[lo:hi].max():
            maxima.append(i)
    return maxima


def _build_kernel(omega: np.ndarray, tau: np.ndarray) -> tuple:
    """Discretized Fredholm kernel on Dirac basis functions.

    A_re[i,k] = 1 / (1 + (omega_i*tau_k)^2)
    A_im[i,k] = omega_i*tau_k / (1 + (omega_i*tau_k)^2)   (positive convention)
    """
    wt = omega[:, None] * tau[None, :]
    denom = 1.0 + wt ** 2
    A_re = 1.0 / denom
    A_im = wt / denom
    return A_re, A_im


def _roughness_matrix(n: int) -> np.ndarray:
    """Second-derivative (order-2 Tikhonov) operator on a vector of length n."""
    if n < 3:
        return np.zeros((0, n))
    L = np.zeros((n - 2, n))
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def _solve_nnls(A: np.ndarray, b: np.ndarray, L: np.ndarray, lam: float) -> np.ndarray:
    """Solve min ||Ax - b||^2 + lam^2 ||L x_gamma||^2, x >= 0, via augmented NNLS.

    The first column of A (R0) is left unpenalized: L is padded with a zero
    column on the left to match A's column count.
    """
    n_cols = A.shape[1]
    L_full = np.zeros((L.shape[0], n_cols))
    L_full[:, 1:] = L
    A_aug = np.vstack([A, lam * L_full])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    x, _ = nnls(A_aug, b_aug, maxiter=10000)
    return x


def _select_lambda(A: np.ndarray, b: np.ndarray, L: np.ndarray, lambdas: np.ndarray) -> float:
    """Pick lambda at the L-curve corner (max curvature in log-log residual/roughness)."""
    res_norms, rough_norms = [], []
    for lam in lambdas:
        x = _solve_nnls(A, b, L, lam)
        gamma = x[1:]
        res_norms.append(float(np.linalg.norm(A @ x - b)))
        rough_norms.append(float(np.linalg.norm(L @ gamma)) + 1e-30)

    res_norms = np.asarray(res_norms)
    rough_norms = np.asarray(rough_norms)
    valid = (res_norms > 0) & (rough_norms > 0)
    if valid.sum() < 3:
        return float(lambdas[len(lambdas) // 2])

    x_l = np.log(res_norms[valid])
    y_l = np.log(rough_norms[valid])
    lam_valid = lambdas[valid]

    # Discrete curvature of the (x_l, y_l) curve, pick max as L-curve corner.
    if len(x_l) < 3:
        return float(lam_valid[0])
    dx = np.gradient(x_l)
    dy = np.gradient(y_l)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx ** 2 + dy ** 2 + 1e-30, 1.5)
    best = int(np.argmax(curvature))
    return float(lam_valid[best])


class DRTTikhonovModel(BaseFitModel):
    """DRT model-free par Tikhonov (ordre 2) + NNLS, appliquée directement
    sur les données expérimentales (Wan, Saccoccio, Chen, Ciucci 2015 ;
    Bissessur, Man, Gamby PRE 2026, section III.B)."""

    name = "drt_tikhonov"
    label = "DRT Tikhonov + NNLS"
    method = "drt_tikhonov"
    display_name = "DRT Tikhonov + NNLS"
    description = (
        "Déconvolution model-free de la distribution des temps de relaxation "
        "par régularisation de Tikhonov (ordre 2) et NNLS, appliquée "
        "directement sur les données expérimentales (sans fit de circuit "
        "équivalent intermédiaire)."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config, weights=None) -> FitResult:
        f_exp = np.asarray(spectrum.f, dtype=float)
        Zre_exp = np.asarray(spectrum.Zre, dtype=float)
        Zim_exp = np.asarray(spectrum.Zim, dtype=float)

        order = np.argsort(f_exp)
        f_sorted = f_exp[order]
        omega = 2.0 * np.pi * f_sorted

        n_tau_cfg = int(_cfg_get(config, "n_tau", 50))
        tau_min_cfg = _cfg_get(config, "tau_min", None)
        tau_max_cfg = _cfg_get(config, "tau_max", None)

        # Tau grid sized on the experimental frequency span (not a dense
        # synthetic grid): default ~10 pts/decade, clipped to config bounds
        # if they fall inside the data-covered range.
        tau_lo_data = 1.0 / omega.max()
        tau_hi_data = 1.0 / omega.min()
        tau_lo = max(tau_lo_data, float(tau_min_cfg)) if tau_min_cfg else tau_lo_data
        tau_hi = min(tau_hi_data, float(tau_max_cfg)) if tau_max_cfg else tau_hi_data
        if tau_hi <= tau_lo:
            tau_lo, tau_hi = tau_lo_data, tau_hi_data

        n_decades = max(np.log10(tau_hi / tau_lo), 0.5)
        n_tau = max(n_tau_cfg, int(np.ceil(10 * n_decades)))
        n_tau = min(n_tau, 400)
        tau = np.geomspace(tau_lo, tau_hi, n_tau)
        ln_tau = np.log(tau)
        dln_tau = (ln_tau[-1] - ln_tau[0]) / (n_tau - 1) if n_tau > 1 else 1.0

        A_re_k, A_im_k = _build_kernel(omega, tau)
        A_re_k *= dln_tau
        A_im_k *= dln_tau

        # R0 column: contributes only to Re(Z), not Im(Z).
        A_re = np.column_stack([np.ones(len(omega)), A_re_k])
        A_im = np.column_stack([np.zeros(len(omega)), A_im_k])
        A = np.vstack([A_re, A_im])
        b = np.concatenate([Zre_exp[order], Zim_exp[order]])

        L = _roughness_matrix(n_tau)

        lambda_auto = bool(_cfg_get(config, "lambda_auto", True))
        if lambda_auto and L.shape[0] > 0:
            scale = float(np.linalg.norm(A))
            lambdas = np.geomspace(1e-6 * scale, 1.0 * scale, 25)
            lam = _select_lambda(A, b, L, lambdas)
        else:
            lam = float(_cfg_get(config, "lambda_fixed", 1e-3)) * float(np.linalg.norm(A))

        x = _solve_nnls(A, b, L, lam) if L.shape[0] > 0 else nnls(A, b, maxiter=10000)[0]
        R0 = float(x[0])
        gamma = x[1:]

        # ── Détection des maxima locaux et extraction de Rct (convention Bissessur) ──
        margin = max(1, n_tau // 20)
        core_slice = slice(margin, n_tau - margin) if n_tau > 2 * margin else slice(0, n_tau)
        gamma_core = gamma[core_slice]
        max_global = float(gamma_core.max()) if gamma_core.size else 0.0
        threshold = max_global * 1e-3
        l_window = max(1, n_tau // 15)
        maxima_idx_core = _local_maxima(gamma_core, l=l_window, threshold=threshold)
        maxima_idx = [i + margin for i in maxima_idx_core]

        if len(maxima_idx) >= 2:
            peak_idx = maxima_idx[-2]
        elif len(maxima_idx) == 1:
            peak_idx = maxima_idx[0]
        else:
            peak_idx = None

        if peak_idx is not None:
            center = ln_tau[peak_idx]
            mask = np.abs(ln_tau - center) <= 3.0
            if mask.sum() >= 2:
                x_win = ln_tau[mask]
                y_win = gamma[mask]
                ord_win = np.argsort(x_win)
                Rct = float(np.trapezoid(y_win[ord_win], x_win[ord_win]))
            else:
                Rct = float(gamma[peak_idx]) * dln_tau
            tau_Rct = float(ln_tau[peak_idx])
        else:
            Rct = 0.0
            tau_Rct = float("nan")

        # ── Reconstruction de Z(omega) sur les fréquences expérimentales ────
        Zfit_re_sorted = A_re @ x
        Zfit_im_sorted = A_im @ x
        inv_order = np.argsort(order)
        Zfit_re = Zfit_re_sorted[inv_order]
        Zfit_im = Zfit_im_sorted[inv_order]

        residuals_re = Zre_exp - Zfit_re
        residuals_im = Zim_exp - Zfit_im
        Zmod2 = Zre_exp ** 2 + Zim_exp ** 2 + 1e-30
        chi2 = float(np.mean((residuals_re ** 2 + residuals_im ** 2) / Zmod2))
        reconstruction_error = float(np.mean(
            np.sqrt(residuals_re ** 2 + residuals_im ** 2) / np.sqrt(Zmod2)
        ))

        tau_max = float(tau[core_slice][np.argmax(gamma[core_slice])]) if gamma_core.size else float("nan")

        params = {
            "Rct": Rct,
            "R0": R0,
            "lambda": lam,
            "tau_max": tau_max,
            "tau_Rct": tau_Rct,
            "n_tau": n_tau,
        }

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
            drt_S=ln_tau,
            drt_lnGamma=np.log(np.maximum(gamma, 1e-300)),
            reconstruction_error=reconstruction_error,
        )
