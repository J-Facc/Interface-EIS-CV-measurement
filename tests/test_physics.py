"""Tests for fits/physics.py (Z_randles_full supprimé à l'étape 5 : le Randles complet
est désormais une expression de circuit, testée par test_circuit_parser.py)."""

import numpy as np

from fits.physics import Z_D


def test_Z_D_dc_limit():
    """At very small omega, Z_D should approach R_D."""
    omega = np.array([1e-8])
    R_D = 1000.0
    Z = Z_D(omega, R_D, tau_d=1.0)
    assert abs(Z.real[0] - R_D) / R_D < 0.05, (
        f"DC limit: got {Z.real[0]:.1f} Ω, expected ~{R_D} Ω"
    )


def test_Z_D_returns_finite():
    omega = np.logspace(-2, 5, 50)
    Z = Z_D(omega, R_D=500.0, tau_d=1.0)
    assert np.all(np.isfinite(Z))
