"""DRT inversion via Tikhonov regularisation with L-curve lambda selection."""

import numpy as np
from scipy.signal import find_peaks

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


def _build_kernel(omega: np.ndarray, tau_grid: np.ndarray) -> tuple:
    """Build DRT kernel matrices A_re and A_im.

    The DRT model: Z(ω) = Re + ∫ γ(τ) / (1 + jωτ) d(ln τ)
    Discretised with trapezoidal weights on a log-τ grid.

    Args:
        omega: Angular frequency array (rad/s), length N.
        tau_grid: Relaxation time grid (s), length M.

    Returns:
        (A_re, A_im) each of shape (N, M).
    """
    M = len(tau_grid)
    N = len(omega)

    # Log-spacing weights for integration d(ln τ)
    ln_tau = np.log(tau_grid)
    dln = np.diff(ln_tau)
    dln = np.append(dln[0], dln)  # left-edge padding

    A_re = np.zeros((N, M))
    A_im = np.zeros((N, M))

    for j in range(M):
        tau = tau_grid[j]
        denom = 1.0 + (omega * tau) ** 2
        A_re[:, j] = dln[j] / denom
        A_im[:, j] = -omega * tau * dln[j] / denom

    return A_re, A_im


def _lcurve_lambda(A: np.ndarray, b: np.ndarray, L: np.ndarray, n_pts: int = 40) -> float:
    """Select Tikhonov lambda by maximum L-curve curvature.

    Args:
        A: Full kernel matrix (2N x M).
        b: Observation vector (2N,).
        L: Regularisation matrix (M-2 x M, second-order finite differences).
        n_pts: Number of lambda values to scan.

    Returns:
        Optimal lambda (float).
    """
    lambdas = np.logspace(-6, 2, n_pts)
    ATA = A.T @ A
    ATb = A.T @ b
    LTL = L.T @ L

    rho_log = []
    eta_log = []

    for lam in lambdas:
        try:
            x = np.linalg.solve(ATA + lam * LTL, ATb)
            x = np.maximum(x, 0.0)
            rho_log.append(np.log(np.linalg.norm(A @ x - b) + 1e-30))
            eta_log.append(np.log(np.linalg.norm(L @ x) + 1e-30))
        except np.linalg.LinAlgError:
            rho_log.append(30.0)
            eta_log.append(30.0)

    rho_arr = np.array(rho_log)
    eta_arr = np.array(eta_log)

    drho = np.gradient(rho_arr)
    deta = np.gradient(eta_arr)
    d2rho = np.gradient(drho)
    d2eta = np.gradient(deta)

    num = drho * d2eta - deta * d2rho
    den = (drho ** 2 + deta ** 2) ** 1.5
    kappa = np.abs(num / np.maximum(den, 1e-30))

    return float(lambdas[np.argmax(kappa)])


class DRTTikhonovModel(BaseFitModel):
    """DRT inversion via Tikhonov regularisation.

    Extracts γ(τ), the distribution of relaxation times, from EIS data.
    Lambda is selected automatically by L-curve curvature maximisation.
    Rct is estimated as ∫ γ(τ) d(ln τ) (total area under the DRT).
    """

    name = "drt_tikhonov"
    label = "DRT Tikhonov"
    description = (
        "Distribution des temps de relaxation par régularisation Tikhonov. "
        "λ sélectionné par L-curve. Rct = intégrale sous le pic principal."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Invert EIS spectrum to DRT.

        Args:
            spectrum: EIS spectrum.
            config: App config dict (drt sub-section read for grid parameters).

        Returns:
            FitResult with tau and gamma stored in params dict.
        """
        drt_cfg = config.get("fit", {}).get("drt", {})
        n_tau = int(drt_cfg.get("n_tau", 50))
        tau_min = float(drt_cfg.get("tau_min", 1e-5))
        tau_max = float(drt_cfg.get("tau_max", 100.0))
        lambda_auto = bool(drt_cfg.get("lambda_auto", True))

        omega = 2.0 * np.pi * spectrum.f

        tau_grid = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

        # Subtract HF Re offset
        Re_est = float(np.min(spectrum.Zre))
        Z_re_corr = spectrum.Zre - Re_est

        A_re, A_im = _build_kernel(omega, tau_grid)
        A = np.vstack([A_re, A_im])
        b = np.concatenate([Z_re_corr, spectrum.Zim])

        # Second-order derivative regularisation matrix
        L = np.zeros((max(n_tau - 2, 1), n_tau))
        for i in range(min(n_tau - 2, L.shape[0])):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0

        lam = _lcurve_lambda(A, b, L) if lambda_auto else 1e-3

        ATA = A.T @ A
        ATb = A.T @ b
        LTL = L.T @ L

        try:
            gamma = np.linalg.solve(ATA + lam * LTL, ATb)
            gamma = np.maximum(gamma, 0.0)
            converged = True
        except np.linalg.LinAlgError:
            gamma = np.zeros(n_tau)
            converged = False

        Zfit_re = Re_est + A_re @ gamma
        Zfit_im = -(A_im @ gamma)

        res_re = spectrum.Zre - Zfit_re
        res_im = spectrum.Zim - Zfit_im
        chi2 = float(np.mean(res_re ** 2 + res_im ** 2))

        # Rct = total area under gamma (∫ γ d ln τ already included via dln weights)
        Rct_drt = float(np.sum(gamma))

        params = {
            "Re": Re_est,
            "Rct": Rct_drt,
            "lambda": lam,
            "tau": tau_grid.tolist(),
            "gamma": gamma.tolist(),
        }

        return FitResult(
            model_name=self.name,
            params=params,
            params_std={"Re": 0.0, "Rct": 0.0, "lambda": 0.0},
            Zfit_re=Zfit_re,
            Zfit_im=Zfit_im,
            chi2=chi2,
            residuals_re=res_re,
            residuals_im=res_im,
            Rct=Rct_drt,
            Rct_std=0.0,
            converged=converged,
        )
