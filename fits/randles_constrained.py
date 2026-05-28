"""Constrained Randles fit: Re fixed to HF intercept, ZD0 scaled by Fv^(-1/3).

Free parameters: Rct, Qdl, alpha (3 degrees of freedom).
"""

import numpy as np
from scipy.optimize import least_squares

from fits.base import BaseFitModel
from fits.physics import Z_randles_full
from core.models import EISSpectrum, FitResult


def _estimate_re(spectrum: EISSpectrum) -> float:
    """Estimate Re as the minimum Zre in the HF region (first 5 points)."""
    n_hf = min(5, len(spectrum.f))
    return float(np.min(spectrum.Zre[:n_hf]))


class RandlesConstrainedModel(BaseFitModel):
    """Randles fit with Re fixed and ZD0 scaled to Fv^(-1/3).

    Re is pinned to the HF Nyquist intercept. ZD0 is estimated from the
    flow-rate scaling law. Only Rct, Qdl, and alpha are optimised.
    """

    name = "randles_constrained"
    label = "Randles contraint"
    description = (
        "Randles avec Re fixé sur l'axe réel HF et ZD0 ∝ Fv^(−1/3). "
        "3 paramètres libres : Rct, Qdl, α."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        """Estimate Rct from Nyquist diameter, Qdl and alpha from defaults.

        Args:
            spectrum: EIS spectrum.
            config: App config dict.

        Returns:
            Dict with keys Rct, Qdl, alpha.
        """
        Re = _estimate_re(spectrum)
        Rct_est = max(float(np.max(spectrum.Zre)) - Re, 100.0)
        return {"Rct": Rct_est, "Qdl": 1e-6, "alpha": 0.85}

    def bounds(self, config: dict) -> tuple:
        """Return bounds from config for the 3 free parameters.

        Args:
            config: App config dict.

        Returns:
            (lower_dict, upper_dict).
        """
        b = config.get("fit", {}).get("bounds_randles_full", {})
        lo = {
            "Rct": b.get("Rct", [100.0, 1e9])[0],
            "Qdl": b.get("Qdl", [1e-12, 1e-4])[0],
            "alpha": b.get("alpha", [0.6, 1.0])[0],
        }
        hi = {
            "Rct": b.get("Rct", [100.0, 1e9])[1],
            "Qdl": b.get("Qdl", [1e-12, 1e-4])[1],
            "alpha": b.get("alpha", [0.6, 1.0])[1],
        }
        return lo, hi

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Fit constrained Randles model.

        Re is fixed; ZD0 is scaled to Fv^(-1/3); Rct, Qdl, alpha are optimised.

        Args:
            spectrum: EIS spectrum.
            config: App config dict (must include geometry and conditions).

        Returns:
            FitResult with all Randles parameters (fixed ones included in params).
        """
        omega = 2.0 * np.pi * spectrum.f

        Re = _estimate_re(spectrum)
        geom = config.get("geometry", {})
        cond = config.get("conditions", {})
        phys = config.get("physics", {})

        xe = float(geom.get("xe", 30e-6))
        h = float(geom.get("h", 60e-6))
        d = float(geom.get("d", 300e-6))
        Fv = float(cond.get("Fv", 5e-10))
        D = float(phys.get("D_FeIII", 7.2e-10))

        # ZD0 baseline at Fv_ref = 5e-10 m³/s, scaled by Fv^(-1/3)
        Fv_ref = 5e-10
        ZD0 = 500.0 * (Fv_ref / Fv) ** (1.0 / 3.0)

        Re_prime = Re * 0.05
        Cb = 1e-9

        guess = self.initial_guess(spectrum, config)
        lo, hi = self.bounds(config)

        x0 = [guess["Rct"], guess["Qdl"], guess["alpha"]]
        blo = [lo["Rct"], lo["Qdl"], lo["alpha"]]
        bhi = [hi["Rct"], hi["Qdl"], hi["alpha"]]

        alpha_noise = float(config.get("fit", {}).get("alpha_noise", 0.001))
        Z_data = spectrum.Zre + 1j * spectrum.Zim
        weight = 1.0 / np.maximum(alpha_noise * np.abs(Z_data), 1.0)

        def residuals(x):
            Rct, Qdl, alpha_p = x
            Z = Z_randles_full(omega, Re, Re_prime, Cb, Rct, Qdl, alpha_p,
                               ZD0, xe, D, Fv, h, d)
            return np.concatenate([
                (Z.real - spectrum.Zre) * weight,
                (Z.imag - spectrum.Zim) * weight,
            ])

        max_iter = int(config.get("fit", {}).get("max_iter", 10000))

        try:
            result = least_squares(
                residuals, x0, bounds=(blo, bhi),
                max_nfev=max_iter, method="trf",
            )
            converged = result.success or result.cost < 1.0
            Rct_fit, Qdl_fit, alpha_fit = result.x
        except Exception:
            Rct_fit, Qdl_fit, alpha_fit = x0
            converged = False

        Z_fit = Z_randles_full(omega, Re, Re_prime, Cb, Rct_fit, Qdl_fit,
                               alpha_fit, ZD0, xe, D, Fv, h, d)

        res_re = spectrum.Zre - Z_fit.real
        res_im = spectrum.Zim - Z_fit.imag
        chi2 = float(np.mean(res_re ** 2 + res_im ** 2))

        params = {
            "Re": Re, "Re_prime": Re_prime, "Cb": Cb,
            "Rct": Rct_fit, "Qdl": Qdl_fit, "alpha": alpha_fit,
            "ZD0": ZD0, "D_eff": D,
        }

        return FitResult(
            model_name=self.name,
            params=params,
            params_std={k: 0.0 for k in params},
            Zfit_re=Z_fit.real,
            Zfit_im=Z_fit.imag,
            chi2=chi2,
            residuals_re=res_re,
            residuals_im=res_im,
            Rct=float(Rct_fit),
            Rct_std=0.0,
            converged=converged,
        )
