"""DRT by Tikhonov regularisation with automatic lambda selection via L-curve.

The DRT maps the imaginary part of Z(omega) onto a distribution gamma(tau):
    Zim(omega) = integral of gamma(tau) * omega*tau / (1+(omega*tau)^2) d(tau)

Rct is estimated as the integral under the dominant DRT peak.
Reference: Wan, Saccoccio, Chen, Ciucci, Electrochim. Acta, 2015.
"""

import numpy as np
from scipy.signal import find_peaks

from core.models import EISSpectrum, FitResult
from fits.base import BaseFitModel
from fits.registry import register
from core.logger import get_logger

log = get_logger("drt_tikhonov")


def _build_A_matrix(omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Build the DRT kernel matrix A[i,k] = omega_i*tau_k / (1+(omega_i*tau_k)^2).

    Args:
        omega: Angular frequencies (rad/s), shape (N,).
        tau: Relaxation times (s), shape (M,).

    Returns:
        Matrix A of shape (N, M).
    """
    ot = np.outer(omega, tau)      # (N, M)
    return ot / (1.0 + ot ** 2)   # element-wise


def _tikhonov_solve(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Solve (A^T A + lam*I) x = A^T b and project to non-negative.

    Args:
        A: Kernel matrix (N, M).
        b: Target vector (N,).
        lam: Regularisation parameter.

    Returns:
        Non-negative solution vector (M,).
    """
    M = A.shape[1]
    AtA = A.T @ A + lam * np.eye(M)
    Atb = A.T @ b
    x = np.linalg.solve(AtA, Atb)
    return np.maximum(x, 0.0)


def _select_lambda_lcurve(A: np.ndarray, b: np.ndarray, n_pts: int = 40) -> float:
    """Select regularisation lambda by maximum L-curve curvature.

    Args:
        A: Kernel matrix.
        b: Target vector.
        n_pts: Number of lambda candidates on a log-scale grid.

    Returns:
        Optimal lambda.
    """
    lambdas = np.logspace(-8, 0, n_pts)
    rho, eta = [], []
    for lam in lambdas:
        x = _tikhonov_solve(A, b, lam)
        rho.append(np.linalg.norm(A @ x - b))
        eta.append(np.linalg.norm(x))

    log_rho = np.log(np.array(rho) + 1e-20)
    log_eta = np.log(np.array(eta) + 1e-20)

    d1r = np.gradient(log_rho)
    d2r = np.gradient(d1r)
    d1e = np.gradient(log_eta)
    d2e = np.gradient(d1e)

    num = d1r * d2e - d2r * d1e
    den = (d1r ** 2 + d1e ** 2) ** 1.5
    curvature = num / (den + 1e-20)

    return float(lambdas[int(np.argmax(curvature))])


@register
class DRTTikhonov(BaseFitModel):
    name = "drt_tikhonov"
    label = "DRT Tikhonov"
    description = (
        "Distribution des temps de relaxation par régularisation Tikhonov. "
        "Lambda sélectionné par L-curve. Rct = intégrale sous le pic dominant."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Compute DRT via Tikhonov regularisation and extract Rct from dominant peak.

        Args:
            spectrum: EIS data.
            config: App config dict.

        Returns:
            FitResult; params contains 'tau', 'gamma', 'lambda', 'Re'.
        """
        drt_cfg = config.get("fit", {}).get("drt", {})
        n_tau = int(drt_cfg.get("n_tau", 50))
        tau_min = float(drt_cfg.get("tau_min", 1e-5))
        tau_max = float(drt_cfg.get("tau_max", 100.0))
        lambda_auto = bool(drt_cfg.get("lambda_auto", True))

        omega = 2.0 * np.pi * spectrum.f
        Zre = spectrum.Zre
        Zim = np.abs(spectrum.Zim)   # DRT uses positive imaginary part

        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        A = _build_A_matrix(omega, tau)

        lam = _select_lambda_lcurve(A, Zim) if lambda_auto else 1e-4
        gamma = _tikhonov_solve(A, Zim, lam)

        Zim_fit = A @ gamma
        Re_est = float(np.mean(Zre[:max(3, len(Zre) // 10)]))
        Zre_fit = np.full_like(Zim_fit, Re_est)

        res_re = Zre - Zre_fit
        res_im = Zim - Zim_fit
        chi2 = float(np.mean(res_re ** 2 + res_im ** 2))

        # Extract Rct: integrate under the highest-prominence peak
        Rct = 0.0
        peaks, props = find_peaks(gamma, prominence=0.0)
        if len(peaks) > 0:
            main_idx = peaks[int(np.argmax(props["prominences"]))]
            log_tau = np.log10(tau)
            log_tau_pk = log_tau[main_idx]
            region = np.abs(log_tau - log_tau_pk) < 2.0
            Rct = float(np.trapz(gamma[region], tau[region]))

        return FitResult(
            model_name=self.name,
            params={
                "tau": tau.tolist(),
                "gamma": gamma.tolist(),
                "lambda": float(lam),
                "Re": Re_est,
            },
            params_std={},
            Zfit_re=Zre_fit,
            Zfit_im=Zim_fit,
            chi2=chi2,
            residuals_re=res_re,
            residuals_im=res_im,
            Rct=Rct,
            Rct_std=0.0,
            converged=True,
        )
