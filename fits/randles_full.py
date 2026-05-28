"""Full Randles circuit fit: 8 free parameters, modulus weighting, lmfit.

Circuit topology: Re - [R'e // Cb] - [Rct // CPE(Qdl, alpha)] - ZD
Free parameters: Re, Re_prime, Cb, Rct, Qdl, alpha, ZD0, D_eff
"""

import numpy as np
import lmfit

from core.models import EISSpectrum, FitResult
from fits.base import BaseFitModel
from fits.physics import Z_randles_full
from fits.registry import register
from core.logger import get_logger

log = get_logger("randles_full")


@register
class RandlesFull(BaseFitModel):
    name = "randles_full"
    label = "Randles complet"
    description = (
        "Circuit Randles complet: Re, R’e, Cb, Rct, CPE(Qdl, alpha), ZD0, D_eff. "
        "8 paramètres libres, pondération Modulus."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        Zre = spectrum.Zre
        Re_est = max(float(Zre[0]), 100.0)
        Rct_est = max(2.0 * (float(np.max(Zre)) - Re_est), 500.0)
        ZD0_est = max(float(np.max(Zre) - np.min(Zre)), 100.0)
        phys = config.get("physics", {})
        D_est = (phys.get("D_FeIII", 7.2e-10) + phys.get("D_FeII", 6.5e-10)) / 2.0
        return {
            "Re": Re_est,
            "Re_prime": 10.0,
            "Cb": 1e-9,
            "Rct": Rct_est,
            "Qdl": 1e-6,
            "alpha": 0.8,
            "ZD0": ZD0_est,
            "D_eff": D_est,
        }

    def bounds(self, config: dict) -> tuple:
        b = config.get("fit", {}).get("bounds_randles_full", {})
        defaults = {
            "Re":       [100.0,   100000.0],
            "Re_prime": [1.0,     100000.0],
            "Cb":       [1e-12,   1e-4],
            "Rct":      [100.0,   1e9],
            "Qdl":      [1e-12,   1e-4],
            "alpha":    [0.6,     1.0],
            "ZD0":      [10.0,    1e6],
            "D_eff":    [1e-11,   1e-8],
        }
        lower = {k: b.get(k, v)[0] for k, v in defaults.items()}
        upper = {k: b.get(k, v)[1] for k, v in defaults.items()}
        return lower, upper

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Fit the full Randles model with modulus weighting via lmfit.

        Args:
            spectrum: EIS data.
            config: App config dict.

        Returns:
            FitResult with 8 fitted parameters and reconstructed impedance.
        """
        omega = 2.0 * np.pi * spectrum.f
        Zre = spectrum.Zre
        Zim = spectrum.Zim
        Z_obs = Zre + 1j * Zim

        geom = config.get("geometry", {})
        cond = config.get("conditions", {})
        fit_cfg = config.get("fit", {})

        xe = geom.get("xe", 30e-6)
        h = geom.get("h", 60e-6)
        d = geom.get("d", 300e-6)
        Fv = cond.get("Fv", 5e-10)
        alpha_noise = fit_cfg.get("alpha_noise", 0.001)
        max_iter = fit_cfg.get("max_iter", 10000)

        g0 = self.initial_guess(spectrum, config)
        lower, upper = self.bounds(config)

        params = lmfit.Parameters()
        for pname, val in g0.items():
            params.add(pname, value=val, min=lower[pname], max=upper[pname])

        weights = 1.0 / (alpha_noise * np.abs(Z_obs) + 1.0)

        def residuals(p):
            Z_model = Z_randles_full(
                omega,
                p["Re"], p["Re_prime"], p["Cb"],
                p["Rct"], p["Qdl"], p["alpha"],
                p["ZD0"], xe, p["D_eff"], Fv, h, d,
            )
            dre = (Zre - Z_model.real) * weights
            dim = (Zim - Z_model.imag) * weights
            return np.concatenate([dre, dim])

        converged = False
        result = None
        try:
            result = lmfit.minimize(residuals, params, method="leastsq", max_nfev=max_iter)
            converged = result.success or bool(result.errorbars)
        except Exception as e:
            log.warning(f"Full Randles fit failed: {e}")

        if result is not None and converged:
            p_opt = result.params
            p_err = {k: (float(result.params[k].stderr) if result.params[k].stderr else np.inf) for k in g0}
        else:
            p_opt = params
            p_err = {k: np.inf for k in g0}

        Z_fit = Z_randles_full(
            omega,
            float(p_opt["Re"]), float(p_opt["Re_prime"]), float(p_opt["Cb"]),
            float(p_opt["Rct"]), float(p_opt["Qdl"]), float(p_opt["alpha"]),
            float(p_opt["ZD0"]), xe, float(p_opt["D_eff"]), Fv, h, d,
        )

        res_re = Zre - Z_fit.real
        res_im = Zim - Z_fit.imag
        chi2 = float(np.mean(res_re ** 2 + res_im ** 2))
        params_out = {k: float(p_opt[k]) for k in g0}

        return FitResult(
            model_name=self.name,
            params=params_out,
            params_std=p_err,
            Zfit_re=Z_fit.real,
            Zfit_im=Z_fit.imag,
            chi2=chi2,
            residuals_re=res_re,
            residuals_im=res_im,
            Rct=float(p_opt["Rct"]),
            Rct_std=p_err.get("Rct", np.inf),
            converged=converged,
        )
