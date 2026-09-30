"""Tests de circuit/elements.py et de sa cohérence avec le registre et le parseur."""

import inspect

import numpy as np
import pytest

from circuit import elements
from circuit.parser import ALLOWED_FUNCTIONS
from circuit.registry_elements import ELEMENTS
from fits.physics import Z_D

W = 2.0 * np.pi * np.logspace(-4, 7, 80)


# --- cohérence registre / signatures / liste blanche -------------------------

def test_registry_matches_whitelist():
    assert set(ELEMENTS) == set(ALLOWED_FUNCTIONS)


@pytest.mark.parametrize("name", sorted(set(ELEMENTS) - {"parallel"}))
def test_element_signature_matches_registry(name):
    params = list(inspect.signature(getattr(elements, name)).parameters)
    assert params[0] == "w"
    spec = ELEMENTS[name]
    assert len(params) - 1 == spec["n_params"]
    assert len(spec["params"]) == len(spec["units"]) == len(spec["default_bounds"]) \
        == spec["n_params"]
    for lo, hi in spec["default_bounds"]:
        assert lo is None or hi is None or lo < hi
    assert spec["description"] and spec["formula"] and spec["bounds_note"]


@pytest.mark.parametrize("name", sorted(set(ELEMENTS) - {"parallel"}))
def test_element_returns_complex_array_shaped_like_w(name):
    args = [0.5] * ELEMENTS[name]["n_params"]
    out = getattr(elements, name)(W, *args)
    assert out.shape == W.shape and np.iscomplexobj(out)


def test_parallel_signature_is_variadic_without_w():
    (param,) = inspect.signature(elements.parallel).parameters.values()
    assert param.kind is inspect.Parameter.VAR_POSITIONAL
    assert ELEMENTS["parallel"]["n_params"] is None


# --- formules ----------------------------------------------------------------

def test_R_C_L():
    np.testing.assert_array_equal(elements.R(W, 50.0), np.full(W.shape, 50.0 + 0j))
    np.testing.assert_allclose(elements.C(W, 1e-6), 1.0 / (1j * W * 1e-6))
    np.testing.assert_allclose(elements.L(W, 1e-6), 1j * W * 1e-6)


def test_Q_alpha_one_is_capacitor_and_phase_is_constant():
    np.testing.assert_allclose(elements.Q(W, 1e-6, 1.0), elements.C(W, 1e-6), rtol=1e-12)
    for alpha in (0.5, 0.8):
        phase = np.degrees(np.angle(elements.Q(W, 1e-6, alpha)))
        np.testing.assert_allclose(phase, -90.0 * alpha, atol=1e-9)


def test_W_semi_infinite():
    z = elements.W(W, 10.0)
    np.testing.assert_allclose(np.degrees(np.angle(z)), -45.0, atol=1e-9)
    np.testing.assert_allclose(np.abs(z), 10.0 * np.sqrt(2.0 / W), rtol=1e-12)
    np.testing.assert_allclose(z, 10.0 * np.sqrt(2.0) / np.sqrt(1j * W), rtol=1e-12)


def test_ZD_bounded_is_exactly_fits_physics_Z_D():
    """Reprise exacte : égalité bit à bit, y compris dans la branche L'Hôpital."""
    omega = np.concatenate([[0.0, 1e-20, 1e-17], W])
    for R_D, tau_d in [(1000.0, 1.0), (10.0, 1e-3), (5e5, 300.0)]:
        np.testing.assert_array_equal(
            elements.ZD_bounded(omega, R_D, tau_d), Z_D(omega, R_D, tau_d))


def test_Ws_equals_ZD_bounded():
    np.testing.assert_allclose(elements.Ws(W, 300.0, 2.0),
                               elements.ZD_bounded(W, 300.0, 2.0), rtol=1e-15)


def test_finite_warburgs_limits():
    r, tau = 200.0, 1.5
    # Basse fréquence : Ws → r ; Wo → r/3 en partie réelle (divergence capacitive).
    low = np.array([1e-7])
    assert elements.Ws(low, r, tau)[0] == pytest.approx(r, rel=1e-5)
    assert elements.Wo(low, r, tau)[0].real == pytest.approx(r / 3.0, rel=1e-5)
    assert elements.Wo(low, r, tau)[0].imag < -1e6
    # Haute fréquence : les deux tendent vers le Warburg semi-infini σ = r/√(2τ).
    high = np.array([1e6, 1e8])
    sigma = r / np.sqrt(2.0 * tau)
    np.testing.assert_allclose(elements.Wo(high, r, tau), elements.W(high, sigma), rtol=1e-9)
    np.testing.assert_allclose(elements.Ws(high, r, tau), elements.W(high, sigma), rtol=1e-9)


def test_finite_warburgs_have_no_overflow_at_extreme_frequency():
    omega = np.array([1e12, 1e15])
    assert np.all(np.isfinite(elements.Wo(omega, 1.0, 1e3)))
    assert np.all(np.isfinite(elements.Ws(omega, 1.0, 1e3)))


def test_Wo_at_zero_frequency_is_infinite():
    assert np.isinf(elements.Wo(np.array([0.0]), 1.0, 1.0)[0].real)


def test_parallel():
    np.testing.assert_allclose(elements.parallel(100.0, 100.0), 50.0 + 0j)
    np.testing.assert_allclose(
        elements.parallel(elements.R(W, 1e3), elements.C(W, 1e-6)),
        1e3 / (1.0 + 1j * W * 1e-3), rtol=1e-12)
    # Court-circuit : 0 (et non nan).
    out = elements.parallel(np.array([0.0, 10.0]), np.array([5.0, 10.0]))
    np.testing.assert_array_equal(out, np.array([0.0 + 0j, 5.0 + 0j]))
    with pytest.raises(ValueError):
        elements.parallel(1.0)
