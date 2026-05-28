"""Geometric circular fit on the Nyquist plot to extract Rct without physical parameters."""

import numpy as np

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


def _fit_circle_algebraic(x: np.ndarray, y: np.ndarray) -> tuple:
    """Algebraic least-squares circle fit.

    Solves: (x-xc)² + (y-yc)² = r²  →  2xc·x + 2yc·y + k = x² + y²
    where k = r² - xc² - yc².

    Args:
        x: Real part of impedance (Ω).
        y: Imaginary part of impedance (Ω).

    Returns:
        (xc, yc, r) — circle centre and radius (Ω).
    """
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0], sol[1]
    r = float(np.sqrt(np.maximum(sol[2] + xc ** 2 + yc ** 2, 0.0)))
    return float(xc), float(yc), r


class CircularFitModel(BaseFitModel):
    """Geometric circle fit on the Nyquist semicircle.

    Fits a circle to the upper arc of the Nyquist plot and reads Rct as
    the chord length at Zim = 0:  Rct = 2 * sqrt(r² - yc²).

    No physical parameters required — robust alternative to circuit fitting.
    """

    name = "circular"
    label = "Fit circulaire"
    description = (
        "Ajustement géométrique d'un cercle sur le demi-cercle HF du diagramme "
        "de Nyquist. Rct = corde à Zim = 0. Aucun paramètre physique requis."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        """Estimate circle parameters from spectrum extent.

        Args:
            spectrum: EIS spectrum.
            config: Unused.

        Returns:
            Dict with keys xc, yc, r.
        """
        xc = 0.5 * (spectrum.Zre.max() + spectrum.Zre.min())
        r = 0.5 * (spectrum.Zre.max() - spectrum.Zre.min())
        return {"xc": xc, "yc": 0.0, "r": r}

    def bounds(self, config: dict) -> tuple:
        """No bounds needed for geometric fit."""
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Fit circle to Nyquist data and extract Rct from chord length.

        Args:
            spectrum: EIS spectrum (Zim positive convention expected).
            config: Unused.

        Returns:
            FitResult with Rct, xc, yc, r in params.
        """
        x = spectrum.Zre
        y = spectrum.Zim

        # Prefer upper-arc points (Zim >= 0) if enough exist
        mask = y >= 0
        if mask.sum() < 4:
            mask = np.ones(len(x), dtype=bool)

        xc, yc, r = _fit_circle_algebraic(x[mask], y[mask])

        # Rct = chord length at Zim = 0
        disc = r ** 2 - yc ** 2
        Rct = 2.0 * float(np.sqrt(np.maximum(disc, 0.0)))

        # Project each data point onto the circle for residuals
        theta_data = np.arctan2(y - yc, x - xc)
        Zfit_re = xc + r * np.cos(theta_data)
        Zfit_im = yc + r * np.sin(theta_data)

        dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        chi2 = float(np.mean((dist - r) ** 2))

        params = {
            "xc": xc,
            "yc": yc,
            "r": r,
            "Rct": Rct,
        }

        return FitResult(
            model_name=self.name,
            params=params,
            params_std={k: 0.0 for k in params},
            Zfit_re=Zfit_re,
            Zfit_im=Zfit_im,
            chi2=chi2,
            residuals_re=x - Zfit_re,
            residuals_im=y - Zfit_im,
            Rct=Rct,
            Rct_std=0.0,
            converged=True,
        )
