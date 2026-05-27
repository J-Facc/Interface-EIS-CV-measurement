"""Tests for fits/physics.py."""

import numpy as np
import pytest

from fits.physics import (
    alpha_h,
    sigma_reduced,
    ZD_complex,
    Z_randles_full,
    theta_EIS,
    Cdl_brug,
)


def test_alpha_h_formula():
    Fv, h, d = 5e-10, 60e-6, 300e-6
    result = alpha_h(Fv, h, d)
    expected = 6.0 * Fv / (h ** 2 * d)
    assert abs(result - expected) / expected < 1e-12


def test_alpha_h_known_value():
    result = alpha_h(5e-10, 60e-6, 300e-6)
    # 6 * 5e-10 / (3.6e-9 * 300e-6) = 3e-9 / 1.08e-12 ≈ 2777.8
    assert abs(result - 2777.8) < 1.0


def test_ZD_complex_dc_limit():
    """At very low omega (sigma << 1), |ZD| should approach ZD0."""
    omega = np.array([1e-4])
    ZD0 = 1000.0
    Z = ZD_complex(omega, ZD0, xe=30e-6, D=7.2e-10, Fv=5e-10, h=60e-6, d=300e-6)
    assert abs(Z.real[0] - ZD0) / ZD0 < 0.05, (
        f"DC limit: got {Z.real[0]:.1f} Ω, expected ~{ZD0} Ω"
    )


def test_ZD_complex_returns_finite():
    omega = np.logspace(-2, 5, 50)
    Z = ZD_complex(omega, 500.0, xe=30e-6, D=7.2e-10, Fv=5e-10, h=60e-6, d=300e-6)
    assert np.all(np.isfinite(Z))


def test_theta_EIS_known_value():
    result = theta_EIS(1700.0, 50000.0)
    assert abs(result - (1.0 - 1700.0 / 50000.0)) < 1e-12
    assert abs(result - 0.966) < 0.001


def test_theta_EIS_zero_when_equal():
    assert theta_EIS(1000.0, 1000.0) == pytest.approx(0.0)


def test_Z_randles_full_shape_and_finite():
    omega = np.logspace(1, 5, 40)
    Z = Z_randles_full(
        omega,
        Re=500.0, Re_prime=50.0, Cb=1e-9,
        Rct=5000.0, Qdl=1e-6, alpha=0.85,
        ZD0=300.0, xe=30e-6, D=7.2e-10, Fv=5e-10, h=60e-6, d=300e-6,
    )
    assert Z.shape == (40,)
    assert np.all(np.isfinite(Z))


def test_Z_randles_full_re_at_hf():
    """At very high frequency, Z should converge to Re."""
    omega = np.array([1e8])
    Re = 500.0
    Z = Z_randles_full(
        omega,
        Re=Re, Re_prime=50.0, Cb=1e-9,
        Rct=5000.0, Qdl=1e-6, alpha=0.85,
        ZD0=300.0, xe=30e-6, D=7.2e-10, Fv=5e-10, h=60e-6, d=300e-6,
    )
    assert abs(Z.real[0] - Re) / Re < 0.05, (
        f"HF real limit: got {Z.real[0]:.1f} Ω, expected ~{Re} Ω"
    )


def test_cdl_brug_positive():
    result = Cdl_brug(Qdl=1e-6, alpha=0.85, Re=500.0, Re_prime=50.0, Rct=5000.0)
    assert result > 0
