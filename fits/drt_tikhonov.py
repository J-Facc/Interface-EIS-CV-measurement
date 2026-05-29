"""DRT inversion via Tikhonov regularisation — v2.

Améliorations vs v1 :
  • Sélection de λ par L-curve avec lissage Savitzky-Golay (plus robuste)
    + GCV calculé en parallèle (stocké pour le diagnostic)
  • Axe temporel τ = 1/(2π f) — domaine auto-calculé depuis les fréquences
  • Filtrage des artefacts de bord dans la détection des pics
  • FitResult enrichi : tau_peaks, lambda_gcv, ln_tau/ln_gamma pour le tracé

Références :
  Maradesa et al. (2024), Joule 8, 1958–1981
  Bissessur et al. (2026), Phys. Rev. E 113, 025502
  Hansen (1992), SIAM Rev. 34(4), 561–580
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


# ── Constantes ────────────────────────────────────────────────────────────────
_TAU_EXT   = 1.5    # décades d'extension au-delà du domaine fréquentiel
_N_LAMBDA  = 100    # points dans la grille λ
_LAM_MIN   = 1e-7
_LAM_MAX   = 1e1
_EDGE_FRAC = 0.12   # fraction des bords à ignorer pour la détection de pics


# ── Construction du kernel ────────────────────────────────────────────────────

def _build_kernel(omega: np.ndarray, tau_grid: np.ndarray) -> tuple:
    """Build DRT kernel matrices A_re and A_im.

    Z(ω) = Re + ∫ γ(τ) / (1 + jωτ) d(ln τ)
    Discretised with uniform log-spacing weights Δ(ln τ).

    Args:
        omega: Angular frequency array (rad/s), length N.
        tau_grid: Relaxation time grid (s), length M.

    Returns:
        (A_re, A_im) each of shape (N, M).
    """
    M = len(tau_grid)
    N = len(omega)

    ln_tau = np.log(tau_grid)
    dln = np.diff(ln_tau)
    dln = np.append(dln[0], dln)   # left-edge padding (same as v1)

    A_re = np.zeros((N, M))
    A_im = np.zeros((N, M))

    for j in range(M):
        tau  = tau_grid[j]
        denom = 1.0 + (omega * tau) ** 2
        A_re[:, j] =  dln[j] / denom
        A_im[:, j] = -omega * tau * dln[j] / denom

    return A_re, A_im


# ── Matrice de régularisation ─────────────────────────────────────────────────

def _make_L2(n_tau: int) -> np.ndarray:
    """Matrice de différences finies du 2ᵉ ordre (n_tau-2 × n_tau)."""
    L = np.zeros((max(n_tau - 2, 1), n_tau))
    for i in range(min(n_tau - 2, L.shape[0])):
        L[i, i]     =  1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] =  1.0
    return L


# ── Résolution Tikhonov ───────────────────────────────────────────────────────

def _tikhonov_solve(
    A: np.ndarray,
    b: np.ndarray,
    L: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Résout (AᵀA + λ LᵀL) γ = Aᵀb  avec γ ≥ 0."""
    ATA = A.T @ A
    ATb = A.T @ b
    LTL = L.T @ L
    try:
        gamma = np.linalg.solve(ATA + lam * LTL, ATb)
    except np.linalg.LinAlgError:
        gamma = np.zeros(A.shape[1])
    return np.maximum(gamma, 0.0)


# ── Sélection de λ ────────────────────────────────────────────────────────────

def _lcurve_lambda_sg(
    A: np.ndarray,
    b: np.ndarray,
    L: np.ndarray,
    n_pts: int = _N_LAMBDA,
) -> tuple[float, dict]:
    """Sélectionne λ par courbure maximale de la L-curve (lissage Savitzky-Golay).

    Amélioration vs v1 :
      - Axes normalisés [0,1] avant calcul de courbure (Hansen 1992)
      - Dérivées lissées par SG pour réduire le bruit numérique
      - Masquage des bords (λ < 1e-6 et ±12 % du domaine)

    Retourne (lam_best, meta_dict).
    meta contient les vecteurs bruts pour les panneaux de diagnostic.
    """
    lambdas = np.logspace(np.log10(_LAM_MIN), np.log10(_LAM_MAX), n_pts)

    rho_arr = []
    eta_arr = []

    for lam in lambdas:
        gamma = _tikhonov_solve(A, b, L, lam)
        rho_arr.append(np.linalg.norm(A @ gamma - b))
        eta_arr.append(np.linalg.norm(L @ gamma))

    rho_arr = np.array(rho_arr)
    eta_arr = np.array(eta_arr)

    # ── Courbure en log-log normalisé ──
    log_r = np.log10(rho_arr + 1e-30)
    log_s = np.log10(eta_arr + 1e-30)

    log_r_n = (log_r - log_r.min()) / ((log_r.max() - log_r.min()) + 1e-30)
    log_s_n = (log_s - log_s.min()) / ((log_s.max() - log_s.min()) + 1e-30)

    # Fenêtre SG ≈ 15 % des points
    n   = len(lambdas)
    win = max(5, 2 * (n // 14) + 1)

    dr  = savgol_filter(np.gradient(log_r_n), win, 3)
    ds  = savgol_filter(np.gradient(log_s_n), win, 3)
    d2r = savgol_filter(np.gradient(dr),      win, 3)
    d2s = savgol_filter(np.gradient(ds),      win, 3)

    kappa = (dr * d2s - ds * d2r) / ((dr**2 + ds**2)**1.5 + 1e-30)

    # Masque : ignorer bords et λ trop petits
    edge  = max(1, int(_EDGE_FRAC * n))
    valid = np.zeros(n, dtype=bool)
    valid[edge:-edge] = True
    valid[np.log10(lambdas) < -6] = False
    kappa_masked = np.where(valid, kappa, -1e30)

    idx_best   = int(np.argmax(kappa_masked))
    lam_lcurve = float(lambdas[idx_best])

    meta = {
        "lambdas":    lambdas,
        "rho":        rho_arr,
        "eta":        eta_arr,
        "kappa":      kappa,
        "lam_lcurve": lam_lcurve,
    }
    return lam_lcurve, meta


def _gcv_lambda(
    A: np.ndarray,
    b: np.ndarray,
    n_pts: int = _N_LAMBDA,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Score GCV via SVD de A (cas L = I, utilisé pour diagnostic)."""
    lambdas = np.logspace(np.log10(_LAM_MIN), np.log10(_LAM_MAX), n_pts)
    M2 = A.shape[0]

    try:
        _, s, _ = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return float(lambdas[len(lambdas)//2]), lambdas, np.ones(n_pts)

    scores = []
    for lam in lambdas:
        f     = s**2 / (s**2 + lam)
        tr_H  = float(np.sum(f))
        denom = (1.0 - tr_H / M2)**2
        if denom < 1e-12:
            scores.append(1e30)
            continue
        # Solution filtrée (projection, sans NNLS pour la vitesse)
        gamma_raw = (f / s) * (np.linalg.svd(A, full_matrices=False)[0].T @ b)
        gamma     = np.maximum(gamma_raw, 0.0)
        res       = A @ gamma - b
        scores.append(float(np.dot(res, res) / (M2 * denom)))

    scores_arr = np.array(scores)
    lam_gcv    = float(lambdas[np.argmin(scores_arr)])
    return lam_gcv, lambdas, scores_arr


# ── Extraction de Rct ─────────────────────────────────────────────────────────

def _extract_rct(gamma: np.ndarray, ln_tau: np.ndarray) -> float:
    """Intègre le pic principal de γ(τ) pour estimer Rct.

    Stratégie (Bissessur 2026) :
      1. Ignore les bords (artefacts de discrétisation)
      2. Prend le pic de plus haute amplitude
      3. Intègre sur ±1 unité ln(τ) autour du pic
    """
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

    n    = len(gamma)
    edge = max(3, int(_EDGE_FRAC * n))
    g_mid = gamma.copy()
    g_mid[:edge]  = 0.0
    g_mid[-edge:] = 0.0

    peaks_idx, _ = find_peaks(g_mid, height=0.01 * (gamma.max() + 1e-30), distance=3)

    if len(peaks_idx) == 0:
        return float(_trapz(g_mid, ln_tau))

    best  = peaks_idx[np.argmax(gamma[peaks_idx])]
    ln_t0 = ln_tau[best]
    mask  = np.abs(ln_tau - ln_t0) <= 1.0
    Rct   = float(_trapz(gamma[mask], ln_tau[mask]))
    return max(Rct, 0.0)


# ── Modèle principal ──────────────────────────────────────────────────────────

class DRTTikhonovModel(BaseFitModel):
    """DRT inversion via Tikhonov regularisation — v2.

    Paramètres config (section fit.drt) :
        n_tau        : nb de points de collocation τ     [défaut: 50]
        tau_min      : borne basse du domaine τ (s)      [défaut: auto]
        tau_max      : borne haute du domaine τ (s)      [défaut: auto]
        lambda_auto  : True → L-curve SG, False → 1e-3  [défaut: True]
        lambda_method: 'lcurve' | 'gcv'                 [défaut: 'lcurve']
    """

    name        = "drt_tikhonov"
    label       = "DRT Tikhonov"
    description = (
        "Distribution des temps de relaxation par régularisation Tikhonov. "
        "λ sélectionné par L-curve (lissage Savitzky-Golay). "
        "Tracé en ln(τ) selon Bissessur (2026)."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        drt_cfg = config.get("fit", {}).get("drt", {})
        n_tau         = int(drt_cfg.get("n_tau",         50))
        lambda_auto   = bool(drt_cfg.get("lambda_auto",  True))
        lambda_method = str(drt_cfg.get("lambda_method", "lcurve"))

        omega = 2.0 * np.pi * spectrum.f

        # ── Domaine τ = 1/(2π f), étendu de ±TAU_EXT décades ──
        log_tau_min = -np.log10(spectrum.f.max()) - _TAU_EXT
        log_tau_max = -np.log10(spectrum.f.min()) + _TAU_EXT
        tau_grid = np.logspace(log_tau_min, log_tau_max, n_tau)
        ln_tau   = np.log(tau_grid)          # axe Bissessur : ln(τ)

        # ── Re offset ──
        Re_est    = float(np.min(spectrum.Zre))
        Z_re_corr = spectrum.Zre - Re_est

        # ── Kernel et vecteur d'observations ──
        A_re, A_im = _build_kernel(omega, tau_grid)
        A = np.vstack([A_re, A_im])
        b = np.concatenate([Z_re_corr, spectrum.Zim])

        # ── Matrice de régularisation (ordre 2, comme v1) ──
        L = _make_L2(n_tau)

        # ── Sélection λ ──
        if not lambda_auto:
            lam       = 1e-3
            lam_meta  = {"lam_lcurve": lam, "lambdas": np.array([]), "rho": np.array([]), "eta": np.array([])}
            lam_gcv   = lam
            gcv_lams  = np.array([])
            gcv_scores = np.array([])
            lam_method_label = "fixe"
        elif lambda_method == "gcv":
            lam_gcv, gcv_lams, gcv_scores = _gcv_lambda(A, b)
            lam      = lam_gcv
            lam_meta = {"lam_lcurve": lam, "lambdas": np.array([]), "rho": np.array([]), "eta": np.array([])}
            lam_method_label = "GCV"
        else:  # 'lcurve' (défaut)
            lam, lam_meta = _lcurve_lambda_sg(A, b, L)
            lam_gcv, gcv_lams, gcv_scores = _gcv_lambda(A, b)
            lam_method_label = "L-curve (SG)"

        # ── Résolution ──
        gamma     = _tikhonov_solve(A, b, L, lam)
        converged = bool(np.any(gamma > 0))

        # ── Impédance reconstruite ──
        Zfit_re = Re_est + A_re @ gamma
        Zfit_im = -(A_im @ gamma)            # convention positive (comme v1)

        # ── χ² ──
        res_re = spectrum.Zre - Zfit_re
        res_im = spectrum.Zim - Zfit_im
        chi2   = float(np.mean(res_re**2 + res_im**2))

        # ── Rct ──
        Rct_drt = _extract_rct(gamma, ln_tau)

        # ── Pics τ (pour MAD Bissessur et marqueurs graphe) ──
        n    = len(gamma)
        edge = max(3, int(_EDGE_FRAC * n))
        g_mid = gamma.copy()
        g_mid[:edge] = 0.0; g_mid[-edge:] = 0.0
        peaks_idx, _ = find_peaks(g_mid, height=0.01 * (gamma.max() + 1e-30), distance=3)
        tau_peaks = tau_grid[peaks_idx].tolist() if len(peaks_idx) > 0 else []

        params = {
            # Compatibilité v1
            "Re":     Re_est,
            "Rct":    Rct_drt,
            "lambda": lam,
            "tau":    tau_grid.tolist(),      # axe τ en secondes (compatibilité v1)
            "gamma":  gamma.tolist(),

            # Nouveaux champs v2 (utilisés par drt_figure_v2 dans eis_plots.py)
            "ln_tau":        ln_tau.tolist(),           # axe Bissessur : ln(τ)
            "ln_gamma":      np.log(np.clip(gamma, 1e-30, None)).tolist(),
            "lambda_method": lam_method_label,
            "tau_peaks":     tau_peaks,
            "lambda_gcv":    float(lam_gcv),
            "lambda_lcurve": float(lam_meta.get("lam_lcurve", lam)),
            # Données brutes L-curve et GCV pour les panneaux de diagnostic
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
