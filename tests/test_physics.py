"""Tests for fits/physics.py (Z_randles_full supprimé à l'étape 5 : le Randles complet
est désormais une expression de circuit, testée par test_circuit_parser.py)."""

import numpy as np
import pytest

from fits.physics import (
    Z_D,
    theta_EIS,
    Cdl_brug,
)


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


def test_theta_EIS_known_value():
    result = theta_EIS(1700.0, 50000.0)
    assert abs(result - (1.0 - 1700.0 / 50000.0)) < 1e-12
    assert abs(result - 0.966) < 0.001


def test_theta_EIS_zero_when_equal():
    assert theta_EIS(1000.0, 1000.0) == pytest.approx(0.0)


def test_cdl_brug_positive():
    result = Cdl_brug(Qdl=1e-6, alpha=0.85, Re=500.0, Re_prime=50.0, Rct=5000.0)
    assert result > 0
