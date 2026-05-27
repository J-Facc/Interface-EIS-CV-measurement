"""Geometric circular fit on the Nyquist HF semicircle.

Rct is read as the chord of the fitted circle at Zim = 0:
    Rct = 2 * sqrt(r^2 - yc^2)
where (xc, yc) is the circle centre and r its radius.
No physical parameters are required — the estimate is purely geometric.
"""

import numpy as np

from core.models import EISSpectrum, FitResult
from fits.base import BaseFitModel
from fits.registry import register


def _fit_circle_algebraic(x: np.ndarray, y: np.ndarray) -> tuple:
    """Algebraic least-squares circle fit.

    Solves 2*xc*x + 2*yc*y + c = x^2 + y^2 as a linear system.

    Args:
        x: X-coordinates (Zre, Ω).
        y: Y-coordinates (Zim, Ω).

    Returns:
        (xc, yc, radius) of the best-fit circle.

    Raises:
        np.linalg.LinAlgError: If the system is singular.
    """
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = float(result[0]), float(result[1])
    r = float(np.sqrt(max(result[2] + xc ** 2 + yc ** 2, 0.0)))
    return xc, yc, r


@register
class CircularFit(BaseFitModel):
    name = "circular"
    label = "Circulaire (géométrique)"
    description = (
        "Ajustement d'un cercle sur le demi-cercle HF du Nyquist. "
        "Rct = longueur de la corde à Zim = 0. "
        "Robuste, sans paramètre physique."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Fit a circle to the HF Nyquist arc and extract Rct geometrically.

        Uses the first 67 % of data points (highest frequencies) to avoid
        contamination by low-frequency diffusion tails.

        Args:
            spectrum: EIS data.
            config: App config dict (unused).

        Returns:
            FitResult with Rct and circle arc as the fitted impedance.
        """
        Zre = spectrum.Zre
        Zim = spectrum.Zim
        n_hf = max(5, int(len(Zre) * 0.67))
        x = Zre[:n_hf]
        y = Zim[:n_hf]

        fail = FitResult(
            model_name=self.name,
            params={},
            params_std={},
            Zfit_re=Zre.copy(),
            Zfit_im=Zim.copy(),
            chi2=np.inf,
            residuals_re=np.zeros_like(Zre),
            residuals_im=np.zeros_like(Zim),
            Rct=np.nan,
            Rct_std=0.0,
            converged=False,
        )

        try:
            xc, yc, r = _fit_circle_algebraic(x, y)
        except Exception:
            return fail

        if r < 1.0:
            return fail

        # Chord at Zim = 0
        disc = r ** 2 - yc ** 2
        if disc < 0:
            Rct = 2.0 * r
            Re_fit = xc - r
        else:
            half_chord = float(np.sqrt(disc))
            Rct = 2.0 * half_chord
            Re_fit = xc - half_chord

        # Reconstruct upper semicircle arc for display
        theta = np.linspace(0.0, np.pi, 300)
        Zfit_re = xc - r * np.cos(theta)
        Zfit_im = yc + r * np.sin(theta)

        # Point-wise residuals at measured Zre positions (approximate)
        Zfit_re_pts = np.interp(Zre, Zfit_re, Zfit_re)
        Zfit_im_pts = np.interp(Zre, Zfit_re, Zfit_im)
        res_re = Zre - Zfit_re_pts
        res_im = Zim - Zfit_im_pts
        chi2 = float(np.mean(res_re ** 2 + res_im ** 2))

        return FitResult(
            model_name=self.name,
            params={"Re": Re_fit, "Rct": Rct, "xc": xc, "yc": yc, "r": r},
            params_std={"Re": 0.0, "Rct": 0.0},
            Zfit_re=Zfit_re,
            Zfit_im=Zfit_im,
            chi2=chi2,
            residuals_re=res_re,
            residuals_im=res_im,
            Rct=float(Rct),
            Rct_std=0.0,
            converged=True,
        )
