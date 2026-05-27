"""Constrained Randles fit: Re fixed from Nyquist HF intercept, 3 free parameters.

Free parameters: Rct, Qdl, alpha.
Re is pinned to the HF real-axis intercept; ZD0 is estimated from the
impedance range (max Zre - Re).
"""

import numpy as np
from scipy.optimize import curve_fit

from core.models import EISSpectrum, FitResult
from fits.base import BaseFitModel
from fits.physics import Z_randles_full
from fits.registry import register
from core.logger import get_logger

log = get_logger("randles_constrained")


def _estimate_Re(Zre: np.ndarray, Zim: np.ndarray) -> float:
    """Estimate electrolyte resistance as the HF Nyquist real-axis intercept.

    Uses the point with the smallest |Zim| in the first fifth of data (HF region).

    Args:
        Zre: Real impedance array (Ω), sorted HF→BF.
        Zim: Imaginary impedance array (Ω).

    Returns:
        Estimated Re (Ω).
    """
    n_hf = max(3, len(Zre) // 5)
    idx = int(np.argmin(np.abs(Zim[:n_hf])))
    return float(Zre[idx])


@register
class RandlesConstrained(BaseFitModel):
    name = "randles_constrained"
    label = "Randles contraint"
    description = (
        "Randles avec Re fixé (lecture Nyquist HF). "
        "3 paramètres libres: Rct, Qdl, alpha. "
        "ZD0 estimé depuis la plage d'impédance."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        Re = _estimate_Re(spectrum.Zre, spectrum.Zim)
        Rct_est = max(2.0 * (float(np.max(spectrum.Zre)) - Re), 500.0)
        return {"Re": Re, "Rct": Rct_est, "Qdl": 1e-6, "alpha": 0.8}

    def bounds(self, config: dict) -> tuple:
        b = config.get("fit", {}).get("bounds_randles_full", {})
        lower = {
            "Rct": b.get("Rct", [100.0, 1e9])[0],
            "Qdl": b.get("Qdl", [1e-12, 1e-4])[0],
            "alpha": b.get("alpha", [0.6, 1.0])[0],
        }
        upper = {
            "Rct": b.get("Rct", [100.0, 1e9])[1],
            "Qdl": b.get("Qdl", [1e-12, 1e-4])[1],
            "alpha": b.get("alpha", [0.6, 1.0])[1],
        }
        return lower, upper

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Fit constrained Randles with Re fixed from HF Nyquist intercept.

        Args:
            spectrum: EIS data.
            config: App config dict.

        Returns:
            FitResult with converged Rct, Qdl, alpha.
        """
        omega = 2.0 * np.pi * spectrum.f
        Zre = spectrum.Zre
        Zim = spectrum.Zim
        Z_obs = Zre + 1j * Zim

        g = self.initial_guess(spectrum, config)
        Re_fixed = g["Re"]

        geom = config.get("geometry", {})
        cond = config.get("conditions", {})
        phys = config.get("physics", {})
        fit_cfg = config.get("fit", {})

        xe = geom.get("xe", 30e-6)
        h = geom.get("h", 60e-6)
        d = geom.get("d", 300e-6)
        Fv = cond.get("Fv", 5e-10)
        D = phys.get("D_FeII", 6.5e-10)
        alpha_noise = fit_cfg.get("alpha_noise", 0.001)
        max_iter = fit_cfg.get("max_iter", 10000)

        ZD0_est = max(float(np.max(Zre)) - Re_fixed, 50.0)
        weights = 1.0 / (alpha_noise * np.abs(Z_obs) + 1.0)

        lower, upper = self.bounds(config)
        p0 = [g["Rct"], g["Qdl"], g["alpha"]]
        lb = [lower["Rct"], lower["Qdl"], lower["alpha"]]
        ub = [upper["Rct"], upper["Qdl"], upper["alpha"]]

        def model_fn(omega_arr, Rct, Qdl, alpha):
            Z = Z_randles_full(
                omega_arr, Re_fixed, 0.0, 0.0,
                Rct, Qdl, alpha, ZD0_est,
                xe, D, Fv, h, d,
            )
            return np.concatenate([Z.real, Z.imag])

        Z_data = np.concatenate([Zre, Zim])
        sigma_w = np.concatenate([1.0 / (weights + 1e-12), 1.0 / (weights + 1e-12)])

        converged = False
        perr = [np.inf, np.inf, np.inf]
        try:
            popt, pcov = curve_fit(
                model_fn, omega, Z_data,
                p0=p0, bounds=(lb, ub),
                sigma=sigma_w, absolute_sigma=True,
                max_nfev=max_iter, method="trf",
            )
            perr_raw = np.sqrt(np.diag(pcov))
            perr = [float(e) if np.isfinite(e) else np.inf for e in perr_raw]
            converged = True
        except Exception as e:
            log.warning(f"Constrained Randles did not converge: {e}")
            popt = p0

        Rct_fit, Qdl_fit, alpha_fit = popt
        Z_fit = Z_randles_full(
            omega, Re_fixed, 0.0, 0.0,
            Rct_fit, Qdl_fit, alpha_fit, ZD0_est,
            xe, D, Fv, h, d,
        )

        res_re = Zre - Z_fit.real
        res_im = Zim - Z_fit.imag
        chi2 = float(np.mean(res_re ** 2 + res_im ** 2))

        return FitResult(
            model_name=self.name,
            params={"Re": Re_fixed, "Rct": float(Rct_fit), "Qdl": float(Qdl_fit), "alpha": float(alpha_fit)},
            params_std={"Re": 0.0, "Rct": perr[0], "Qdl": perr[1], "alpha": perr[2]},
            Zfit_re=Z_fit.real,
            Zfit_im=Z_fit.imag,
            chi2=chi2,
            residuals_re=res_re,
            residuals_im=res_im,
            Rct=float(Rct_fit),
            Rct_std=perr[0],
            converged=converged,
        )
