# -*- coding: utf-8 -*-
"""
fits/kk_validation.py — Validation Kramers-Kronig par circuits de Voigt (Lin-KK).

Remplace la dépendance au package externe `impedance` (impedance.validation.linKK)
par une implémentation directe (Boukamp 1995 / Schönleber et al. 2014) :

  Z_fit(jw) = R0 + Σ_k R_k / (1 + jw·τ_k) [+ 1/(jwC) si add_cap]

Les τ_k sont fixés sur une grille log-espacée entre 1/ω_max et 1/ω_min
(pas d'optimisation non-linéaire) ; les R_k (et R0, 1/C) sont résolus par
moindres carrés linéaires (np.linalg.lstsq) sur les parties réelle et
imaginaire empilées.

µ (Schönleber) = masse des R_k négatifs / masse totale des |R_k| — un score
élevé indique un sur-ajustement (trop d'éléments RC pour le bruit du spectre).
"""

import numpy as np


def lin_kk(
    f: np.ndarray,
    Z: np.ndarray,
    max_M: int = 100,
    add_cap: bool = True,
):
    """Ajuste un circuit de Voigt linéaire sur Z(f) et retourne le diagnostic Lin-KK.

    Args:
        f: Fréquences (Hz), triées en ordre croissant.
        Z: Impédance complexe, convention Im(Z) < 0 (circuit R-C dissipatif).
        max_M: Nombre maximal d'éléments de Voigt.
        add_cap: Ajoute un terme capacitif série 1/(jωC) au modèle.

    Returns:
        (M, mu, Z_fit, res_re, res_im)
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    omega = 2.0 * np.pi * f
    n = len(f)

    M = int(min(max_M, max(2, 2 * n // 3)))
    tau = np.geomspace(1.0 / omega.max(), 1.0 / omega.min(), M)

    A_re = np.empty((n, M + 1))
    A_im = np.empty((n, M + 1))
    A_re[:, 0] = 1.0
    A_im[:, 0] = 0.0
    for k in range(M):
        wt = omega * tau[k]
        denom = 1.0 + wt ** 2
        A_re[:, k + 1] = 1.0 / denom
        A_im[:, k + 1] = -wt / denom

    if add_cap:
        A_re = np.column_stack([A_re, np.zeros(n)])
        A_im = np.column_stack([A_im, -1.0 / omega])

    A = np.vstack([A_re, A_im])
    b = np.concatenate([Z.real, Z.imag])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    Z_fit_re = A_re @ x
    Z_fit_im = A_im @ x
    Z_fit = Z_fit_re + 1j * Z_fit_im

    res_re = Z.real - Z_fit_re
    res_im = Z.imag - Z_fit_im

    Rk = x[1:M + 1]
    total = np.sum(np.abs(Rk))
    mu = float(np.sum(np.abs(Rk[Rk < 0])) / total) if total > 0 else 0.0

    return M, mu, Z_fit, res_re, res_im


def _cfg_get(config, key: str, default):
    fit_cfg = config.get("fit", config) if isinstance(config, dict) else getattr(config, "fit", config)
    if isinstance(fit_cfg, dict):
        return fit_cfg.get(key, default)
    return getattr(fit_cfg, key, default)


def kramers_kronig_check(spectrum, config) -> dict:
    """Vérifie la consistance Kramers-Kronig d'un spectre EIS.

    Args:
        spectrum: EISSpectrum (Zim en convention positive : -Im(Z) > 0).
        config: Config app (dict ou AppSettings) — lit `fit.drt_kk_tol`.

    Returns:
        dict avec kk_passed, residuals_re, residuals_im, Z_kk_re, Z_kk_im,
        max_residual, mu.
    """
    f = np.asarray(spectrum.f, dtype=float)
    Zre = np.asarray(spectrum.Zre, dtype=float)
    Zim = np.asarray(spectrum.Zim, dtype=float)

    order = np.argsort(f)
    f_sorted = f[order]
    Z = Zre[order] - 1j * np.abs(Zim[order])

    tol = float(_cfg_get(config, "drt_kk_tol", 0.05))
    M, mu, Z_fit, res_re, res_im = lin_kk(f_sorted, Z)

    Zmod = np.abs(Z)
    Zmod = np.where(Zmod > 0, Zmod, 1e-30)
    rel_residual = np.sqrt(res_re ** 2 + res_im ** 2) / Zmod
    max_residual = float(np.max(rel_residual))
    kk_passed = bool(max_residual < tol)

    inv = np.argsort(order)
    return {
        "kk_passed": kk_passed,
        "residuals_re": res_re[inv],
        "residuals_im": res_im[inv],
        "Z_kk_re": Z_fit.real[inv],
        "Z_kk_im": -Z_fit.imag[inv],
        "max_residual": max_residual,
        "mu": mu,
    }


def linKK(f, Z, c: float = 0.85, max_M: int = 100, fit_type: str = "complex", add_cap: bool = True):
    """Interface compatible avec l'ancienne dépendance `impedance.validation.linKK`.

    `c` n'est pas utilisé (pas de recherche itérative de M) — conservé pour
    compatibilité de signature avec les appelants existants.
    """
    return lin_kk(f, Z, max_M=max_M, add_cap=add_cap)
