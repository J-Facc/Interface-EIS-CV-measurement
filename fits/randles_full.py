"""Full Randles circuit fit: 8 free parameters, Orazem error-structure weighting.

Méthode de pondération UNIQUE : structure d'erreur d'Orazem & Tribollet
(fits/error_structure.py, via fits/weighting.py). Les poids valent 1/σ²
(absolute_sigma=True TOUJOURS) → la covariance des paramètres n'est jamais
rééchelonnée et χ²_red est un vrai test d'adéquation.
"""

import numpy as np
from scipy.optimize import least_squares

from fits.base import BaseFitModel
from fits.physics import Z_randles_full
from fits.weighting import resolve_weights
from core.models import EISSpectrum, FitResult

_PARAM_NAMES = ["Re", "Re_prime", "Cb", "Rct", "Qdl", "alpha", "R_D", "tau_d"]

# Seuils de diagnostic (I7) — rendre visibles les échecs d'ajustement muets.
_REL_RESIDUAL_WARN = 0.10   # résidu relatif RMS > 10 % → ajustement médiocre
_BOUND_PROXIMITY = 0.01     # paramètre à < 1 % d'une borne → contraint par la borne


class RandlesFullModel(BaseFitModel):
    """Full Randles fit with 8 free parameters.

    Parameters: Re, R'e, Cb, Rct, Qdl, α, R_D, tau_d.
    Diffusion element: bounded diffusion Z_D = R_D · tanh(√(jω·τ_d)) / √(jω·τ_d).
    Weighting: Orazem error structure w = 1/σ² (σ = α|Z_re|+β|Z_im|+γ|Z|²/R_m+δ).
    Optimiser: scipy.optimize.least_squares with TRF algorithm.
    """

    name = "randles_full"
    label = "Randles complet"
    description = (
        "Circuit Randles complet avec 8 paramètres libres "
        "(Re, R'e, Cb, Rct, Qdl, α, R_D, τ_d). Pondération : structure d'erreur "
        "d'Orazem (measurement model)."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        """Estimate initial values from spectrum extrema.

        Args:
            spectrum: EIS spectrum.
            config: App config dict.

        Returns:
            Dict {param_name: initial_value}.
        """
        Re_est = max(float(np.min(spectrum.Zre)), 100.0)
        Rct_est = max(float(np.max(spectrum.Zre)) - Re_est, 500.0)

        # tau_d initial guess: inverse of the low-frequency knee frequency
        f_min = float(np.min(spectrum.f)) if len(spectrum.f) else 1.0
        tau_d_est = 1.0 / (2.0 * np.pi * f_min) if f_min > 0 else 1.0

        return {
            "Re": Re_est,
            "Re_prime": max(Re_est * 0.05, 1.0),
            "Cb": 1e-9,
            "Rct": Rct_est,
            "Qdl": 1e-6,
            "alpha": 0.85,
            "R_D": 0.2 * Rct_est,
            "tau_d": tau_d_est,
        }

    def bounds(self, config: dict) -> tuple:
        """Return per-parameter bounds from config YAML.

        Args:
            config: App config dict.

        Returns:
            (lower_dict, upper_dict).
        """
        b = config.get("fit", {}).get("bounds_randles_full", {})
        defaults_lo = {
            "Re": 100.0, "Re_prime": 1.0, "Cb": 1e-12,
            "Rct": 100.0, "Qdl": 1e-12, "alpha": 0.6,
            "R_D": 10.0, "tau_d": 1e-4,
        }
        defaults_hi = {
            "Re": 1e5, "Re_prime": 1e5, "Cb": 1e-4,
            "Rct": 1e9, "Qdl": 1e-4, "alpha": 1.0,
            "R_D": 1e6, "tau_d": 1e3,
        }

        lo, hi = {}, {}
        for k in _PARAM_NAMES:
            v = b.get(k, None)
            lo[k] = v[0] if v else defaults_lo[k]
            hi[k] = v[1] if v else defaults_hi[k]
        return lo, hi

    def fit(self, spectrum: EISSpectrum, config: dict, weights=None) -> FitResult:
        """Optimise full Randles model against spectrum using TRF least-squares.

        Args:
            spectrum: EIS spectrum.
            config: App config dict.

        Returns:
            FitResult with 8 parameters and approximate standard deviations.
        """
        omega = 2.0 * np.pi * spectrum.f

        guess = self.initial_guess(spectrum, config)
        lo, hi = self.bounds(config)

        x0 = [guess[k] for k in _PARAM_NAMES]
        blo = [lo[k] for k in _PARAM_NAMES]
        bhi = [hi[k] for k in _PARAM_NAMES]

        # Pondération UNIQUE : structure d'erreur d'Orazem, w = 1/σ² (identique sur
        # réel et imaginaire sous equal_re_im). resolve_weights lève
        # ErrorStructureUnavailable si aucune structure n'est disponible (ni
        # réplicats, ni coefficients persistés) : le fit est alors REFUSÉ (on ne
        # rattrape PAS — pas de repli sur un σ arbitraire).
        error_structure = None
        if weights is not None:
            # Poids explicites injectés (tests / appels avancés) : la structure
            # d'erreur n'est alors pas résolue (provenance inconnue).
            if isinstance(weights, (tuple, list)) and len(weights) == 2:
                w_re = np.asarray(weights[0], dtype=float)
                w_im = np.asarray(weights[1], dtype=float)
            else:  # tableau unique hérité → même poids sur les deux composantes
                w_re = w_im = np.asarray(weights, dtype=float)
        else:
            w_re, w_im, error_structure = resolve_weights(spectrum, config)

        sw_re = np.sqrt(w_re)
        sw_im = np.sqrt(w_im)

        def residuals(x):
            Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d = x
            Z = Z_randles_full(omega, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d)
            return np.concatenate([
                (Z.real - spectrum.Zre) * sw_re,
                # spectrum.Zim suit la convention positive du loader (-Im(Z) > 0),
                # alors que Z.imag est l'imaginaire physique (< 0) : le résidu doit
                # comparer -Z.imag à spectrum.Zim, cohérent avec le χ² plus bas.
                (-Z.imag - spectrum.Zim) * sw_im,
            ])

        max_iter = int(config.get("fit", {}).get("max_iter", 10000))

        try:
            result = least_squares(
                residuals, x0, bounds=(blo, bhi),
                max_nfev=max_iter, method="trf",
                ftol=1e-10, xtol=1e-10,
            )
            converged = result.success
            x_fit = result.x

            # Covariance des paramètres depuis la jacobienne pondérée J (résidus
            # pondérés r = √w·Δ, cf. residuals()). Les poids étant TOUJOURS de
            # vraies 1/variance (structure d'erreur d'Orazem, absolute_sigma=True),
            # la covariance est cov = (JᵀJ)⁻¹ SANS AUCUN rééchelonnement par le χ²
            # réduit. La branche « modulus » (cov · 2·cost/dof) a été SUPPRIMÉE :
            # elle n'a plus lieu d'être puisqu'il n'existe plus qu'une seule
            # pondération. σ_param = sqrt(diag((JᵀJ)⁻¹)).
            J = result.jac
            try:
                cov = np.linalg.inv(J.T @ J)
                std = np.sqrt(np.abs(np.diag(cov)))
            except np.linalg.LinAlgError:
                std = np.zeros(len(_PARAM_NAMES))

        except Exception:
            x_fit = x0
            std = np.zeros(len(_PARAM_NAMES))
            converged = False

        params = dict(zip(_PARAM_NAMES, x_fit))
        params_std = dict(zip(_PARAM_NAMES, std))

        Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d = x_fit
        Z_fit = Z_randles_full(omega, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d)

        res_re = spectrum.Zre - Z_fit.real
        res_im = spectrum.Zim + Z_fit.imag

        # χ² réduit pondéré effectivement minimisé : Σ(w_re·Δre² + w_im·Δim²)/(2N−P),
        # avec les poids par point du fit (w_re, w_im, cf. residuals()).
        # Cohérence de signe du résidu imaginaire : residuals() empile
        # (−Z.imag − Zim) tandis que res_im = Zim + Z_fit.imag ci-dessus. Ces deux
        # quantités sont OPPOSÉES ((−Z.imag − Zim) = −(Zim + Z.imag)) → carrés
        # identiques, même contribution au χ² (cf. test d'équivalence).
        # Les poids étant 1/σ² (structure d'erreur d'Orazem), chi2_reduced≈1 EST un
        # vrai test d'adéquation modèle+erreur.
        dof = max(2 * len(spectrum.f) - len(_PARAM_NAMES), 1)
        chi2_reduced = float(np.sum(w_re * res_re ** 2 + w_im * res_im ** 2) / dof)

        # Sous H0 (modèle + structure d'erreur corrects), χ²_red ~ χ²(dof)/dof,
        # d'espérance 1 et d'écart-type √(2/dof) ; intervalle indicatif ~2σ.
        half = 2.0 * np.sqrt(2.0 / dof)
        chi2_reduced_ci = (max(1.0 - half, 0.0), 1.0 + half)
        chi2_warning = None
        if not (chi2_reduced_ci[0] <= chi2_reduced <= chi2_reduced_ci[1]):
            chi2_warning = (
                f"χ²ᵣ = {chi2_reduced:.2f} hors de l'intervalle attendu "
                f"[{chi2_reduced_ci[0]:.2f}, {chi2_reduced_ci[1]:.2f}] — "
            )
            chi2_warning += (
                "sous-ajustement : bruit réel > σ (modèle ou structure d'erreur "
                "sous-estimée)." if chi2_reduced > chi2_reduced_ci[1]
                else "sur-ajustement : σ surestimé ou réplicats corrélés."
            )

        # ── Diagnostics d'ajustement (I7) : les 3 gardes qui auraient crié B1 ──
        Zmod2 = spectrum.Zre ** 2 + spectrum.Zim ** 2 + 1e-30
        rel_residual = float(np.sqrt(np.mean((res_re ** 2 + res_im ** 2) / Zmod2)))

        warnings: list = []
        if not converged:
            warnings.append("ajustement non convergé")
        if chi2_warning is not None:
            warnings.append(chi2_warning)
        if rel_residual > _REL_RESIDUAL_WARN:
            warnings.append(
                f"résidu relatif élevé ({rel_residual * 100:.0f} %) — "
                f"ajustement médiocre, résultat peu fiable"
            )
        for name, value in params.items():
            lo_b, hi_b = lo[name], hi[name]
            if abs(value - lo_b) <= _BOUND_PROXIMITY * max(abs(lo_b), 1e-30):
                warnings.append(f"{name} collé à la borne basse ({lo_b:.3g})")
            elif abs(value - hi_b) <= _BOUND_PROXIMITY * max(abs(hi_b), 1e-30):
                warnings.append(f"{name} collé à la borne haute ({hi_b:.3g})")

        return FitResult(
            model_name=self.name,
            params=params,
            params_std=params_std,
            Zfit_re=Z_fit.real,
            Zfit_im=-Z_fit.imag,
            chi2_reduced=chi2_reduced,
            residuals_re=res_re,
            residuals_im=res_im,
            target_param="Rct",
            target_value=float(Rct),
            target_std=float(params_std.get("Rct", 0.0)),
            converged=converged,
            reconstruction_error=rel_residual,
            warnings=warnings,
            error_structure_source=(
                error_structure.source if error_structure is not None else None
            ),
            error_structure_timestamp=(
                error_structure.timestamp if error_structure is not None else None
            ),
            error_structure_coeffs=(
                {
                    "alpha": error_structure.alpha,
                    "beta": error_structure.beta,
                    "gamma": error_structure.gamma,
                    "delta": error_structure.delta,
                    "R_m": error_structure.R_m,
                }
                if error_structure is not None else None
            ),
            chi2_reduced_ci=chi2_reduced_ci,
        )
