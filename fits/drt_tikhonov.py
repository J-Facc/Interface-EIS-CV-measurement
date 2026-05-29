"""DRT inversion via Tikhonov regularisation — v3.

Corrections vs v2 :
  • Grille τ = sort(1/(2πf)) — triée CROISSANTE, même nb de pts que les fréquences
    → DRT lisse (pas de sur-discrétisation), même résolution que les données
    → Bissessur (2026) utilise exactement cette approche via pyDRTtools
  • A_im positif (convention Zim > 0, EC-Lab) et τ croissant → ∫γ d(ln τ) > 0 ✓
  • Régularisation ordre 1 (lissage doux)
  • Champs string préfixés _str_ pour éviter le bug float() dans exporter.py

Références :
  Maradesa et al. (2024), Joule 8, 1958–1981
  Bissessur et al. (2026), Phys. Rev. E 113, 025502
  Wan et al. (2015), Electrochim. Acta 184, 483  [pyDRTtools kernel]
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


_N_LAMBDA  = 80
_LAM_MIN   = 1e-7
_LAM_MAX   = 1e1
_EDGE_FRAC = 0.10


# ── Kernel ────────────────────────────────────────────────────────────────────

def _build_kernel(omega: np.ndarray, tau_grid: np.ndarray) -> tuple:
    """Matrices A_re et A_im.

    Convention :
        Z_re - Re  ≈  A_re @ gamma
        Zim_pos    ≈  A_im @ gamma   (Zim_pos = -Im(Z) > 0)

    tau_grid DOIT être trié en ordre CROISSANT pour que ∫γ d(ln τ) > 0.
    Poids d'intégration Δ(ln τ) par différences centrées (np.gradient).
    """
    M   = len(tau_grid)
    N   = len(omega)
    dln = np.gradient(np.log(tau_grid))   # positif si tau_grid croissant

    A_re = np.zeros((N, M))
    A_im = np.zeros((N, M))
    for j in range(M):
        tau   = tau_grid[j]
        denom = 1.0 + (omega * tau) ** 2
        A_re[:, j] =          dln[j] / denom
        A_im[:, j] = omega * tau * dln[j] / denom   # positif ✓

    return A_re, A_im


# ── Régularisation ordre 1 ────────────────────────────────────────────────────

def _make_L1(n: int) -> np.ndarray:
    L = np.zeros((n - 1, n))
    for i in range(n - 1):
        L[i, i] = -1.0; L[i, i + 1] = 1.0
    return L


# ── Résolution ────────────────────────────────────────────────────────────────

def _tikhonov_solve(A, b, L, lam):
    try:
        gamma = np.linalg.solve(A.T @ A + lam * L.T @ L, A.T @ b)
    except np.linalg.LinAlgError:
        gamma = np.zeros(A.shape[1])
    return np.maximum(gamma, 0.0)


# ── Sélection λ ───────────────────────────────────────────────────────────────

def _lcurve_lambda(A, b, L, n_pts=_N_LAMBDA):
    lambdas = np.logspace(np.log10(_LAM_MIN), np.log10(_LAM_MAX), n_pts)
    rho_arr, eta_arr = [], []
    for lam in lambdas:
        g = _tikhonov_solve(A, b, L, lam)
        rho_arr.append(np.linalg.norm(A @ g - b))
        eta_arr.append(np.linalg.norm(L @ g))
    rho_arr = np.array(rho_arr)
    eta_arr = np.array(eta_arr)

    log_r = np.log10(rho_arr + 1e-30)
    log_s = np.log10(eta_arr + 1e-30)
    log_r_n = (log_r - log_r.min()) / ((log_r.max() - log_r.min()) + 1e-30)
    log_s_n = (log_s - log_s.min()) / ((log_s.max() - log_s.min()) + 1e-30)

    n   = len(lambdas)
    win = max(5, 2 * (n // 14) + 1)
    dr  = savgol_filter(np.gradient(log_r_n), win, 3)
    ds  = savgol_filter(np.gradient(log_s_n), win, 3)
    d2r = savgol_filter(np.gradient(dr),      win, 3)
    d2s = savgol_filter(np.gradient(ds),      win, 3)
    kappa = (dr * d2s - ds * d2r) / ((dr**2 + ds**2)**1.5 + 1e-30)

    edge = max(1, int(_EDGE_FRAC * n))
    kappa[:edge] = -1e30; kappa[-edge:] = -1e30
    kappa[np.log10(lambdas) < -6] = -1e30

    lam_best = float(lambdas[np.argmax(kappa)])
    return lam_best, {
        "lambdas": lambdas, "rho": rho_arr, "eta": eta_arr, "lam_lcurve": lam_best
    }


def _gcv_lambda(A, b, n_pts=_N_LAMBDA):
    lambdas = np.logspace(np.log10(_LAM_MIN), np.log10(_LAM_MAX), n_pts)
    M2 = A.shape[0]
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return float(lambdas[n_pts//2]), lambdas, np.ones(n_pts)
    Utb = U.T @ b
    scores = []
    for lam in lambdas:
        f_k   = s**2 / (s**2 + lam)
        tr_H  = float(np.sum(f_k))
        denom = (1.0 - tr_H / M2)**2
        if denom < 1e-12: scores.append(1e30); continue
        g  = np.maximum(Vt.T @ ((f_k / s) * Utb), 0.0)
        res = A @ g - b
        scores.append(float(np.dot(res, res) / (M2 * denom)))
    scores_arr = np.array(scores)
    return float(lambdas[np.argmin(scores_arr)]), lambdas, scores_arr


# ── Extraction Rct ────────────────────────────────────────────────────────────

def _extract_rct(gamma, ln_tau):
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    n = len(gamma); edge = max(2, int(_EDGE_FRAC * n))
    gm = gamma.copy(); gm[:edge] = 0.0; gm[-edge:] = 0.0
    pks, _ = find_peaks(gm, height=0.01*(gamma.max()+1e-30), distance=2)
    if len(pks) == 0:
        return float(_trapz(gm, ln_tau))
    best = pks[np.argmax(gamma[pks])]
    mask = np.abs(ln_tau - ln_tau[best]) <= 1.0
    return max(float(_trapz(gamma[mask], ln_tau[mask])), 0.0)


# ── Modèle ────────────────────────────────────────────────────────────────────

class DRTTikhonovModel(BaseFitModel):
    """DRT Tikhonov v3.

    Config fit.drt :
        lambda_auto  : True → L-curve SG   [défaut: True]
        lambda_method: 'lcurve' | 'gcv'    [défaut: 'lcurve']
    """

    name        = "drt_tikhonov"
    label       = "DRT Tikhonov"
    description = (
        "DRT par régularisation Tikhonov. "
        "Grille τ = 1/(2πf) triée croissante. "
        "λ par L-curve (Savitzky-Golay). Axe ln(τ) selon Bissessur (2026)."
    )

    def initial_guess(self, spectrum, config): return {}
    def bounds(self, config): return {}, {}

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        drt_cfg       = config.get("fit", {}).get("drt", {})
        lambda_auto   = bool(drt_cfg.get("lambda_auto",   True))
        lambda_method = str(drt_cfg.get("lambda_method",  "lcurve"))

        omega = 2.0 * np.pi * spectrum.f

        # Grille τ triée croissante = pyDRTtools / Bissessur
        tau_grid = np.sort(1.0 / omega)
        ln_tau   = np.log(tau_grid)

        # Re offset (min de Zre, convention v1)
        Re_est    = float(np.min(spectrum.Zre))
        Z_re_corr = spectrum.Zre - Re_est

        # Kernel — Zim positif (convention EC-Lab)
        A_re, A_im = _build_kernel(omega, tau_grid)
        A = np.vstack([A_re, A_im])
        b = np.concatenate([Z_re_corr, spectrum.Zim])   # Zim déjà positif

        L = _make_L1(len(tau_grid))

        # Sélection λ
        if not lambda_auto:
            lam = 1e-3
            lam_meta = {"lam_lcurve": lam,
                        "lambdas": np.array([]), "rho": np.array([]), "eta": np.array([])}
            lam_gcv, gcv_lams, gcv_scores = lam, np.array([]), np.array([])
            method_lbl = "fixe"
        elif lambda_method == "gcv":
            lam_gcv, gcv_lams, gcv_scores = _gcv_lambda(A, b)
            lam = lam_gcv
            lam_meta = {"lam_lcurve": lam,
                        "lambdas": np.array([]), "rho": np.array([]), "eta": np.array([])}
            method_lbl = "GCV"
        else:
            lam, lam_meta = _lcurve_lambda(A, b, L)
            lam_gcv, gcv_lams, gcv_scores = _gcv_lambda(A, b)
            method_lbl = "L-curve (SG)"

        gamma     = _tikhonov_solve(A, b, L, lam)
        converged = bool(np.any(gamma > 0))

        Zfit_re = Re_est + A_re @ gamma
        Zfit_im = A_im @ gamma            # positif ✓ (convention Zim > 0)

        res_re = spectrum.Zre - Zfit_re
        res_im = spectrum.Zim - Zfit_im
        chi2   = float(np.mean(res_re**2 + res_im**2))

        Rct_drt = _extract_rct(gamma, ln_tau)

        # Pics (hors bords)
        n = len(gamma); edge = max(2, int(_EDGE_FRAC * n))
        gm = gamma.copy(); gm[:edge] = 0.0; gm[-edge:] = 0.0
        pks, _ = find_peaks(gm, height=0.01*(gamma.max()+1e-30), distance=2)
        tau_peaks = tau_grid[pks].tolist() if len(pks) > 0 else []

        params = {
            # Compatibilité v1
            "Re":    Re_est,
            "Rct":   Rct_drt,
            "lambda": lam,
            "tau":   tau_grid.tolist(),
            "gamma": gamma.tolist(),
            # v3 : axe ln(τ)
            "ln_tau":        ln_tau.tolist(),
            "ln_gamma":      np.log(np.clip(gamma, 1e-30, None)).tolist(),
            "lambda_gcv":    float(lam_gcv),
            "lambda_lcurve": float(lam_meta.get("lam_lcurve", lam)),
            "tau_peaks":     tau_peaks,
            # String — préfixe _str_ pour exclure du cast float() dans exporter.py
            "_str_lambda_method": method_lbl,
            # Diagnostics (listes — exclues de l'export YAML par préfixe _lc_/_gcv_)
            "_lc_lambdas":  lam_meta["lambdas"].tolist(),
            "_lc_rho":      lam_meta["rho"].tolist(),
            "_lc_eta":      lam_meta["eta"].tolist(),
            "_gcv_lambdas": gcv_lams.tolist(),
            "_gcv_scores":  gcv_scores.tolist(),
        }

        return FitResult(
            model_name   = self.name,
            params       = params,
            params_std   = {"Re": 0.0, "Rct": 0.0, "lambda": 0.0},
            Zfit_re      = Zfit_re,
            Zfit_im      = Zfit_im,
            chi2         = chi2,
            residuals_re = res_re,
            residuals_im = res_im,
            Rct          = Rct_drt,
            Rct_std      = 0.0,
            converged    = converged,
        )
