"""Rct par DRT = aire du pic principal de γ(τ) (core/drt_peak.py)."""

import numpy as np
import pytest

from core.drt_peak import MainPeak, main_peak


def _gauss(ln_tau, center, width, area):
    """Pic gaussien en ln τ d'aire ``area`` (Ω) : ∫ γ d(ln τ) = area."""
    return area / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((ln_tau - center) / width) ** 2)


TAU = np.logspace(-7, 2, 400)
LN = np.log(TAU)


def test_area_of_an_isolated_peak():
    g = _gauss(LN, np.log(3e-3), 0.6, 3000.0)
    pk = main_peak(TAU, g, [1e-5, 1e-1])
    assert pk.ok and pk.rct == pytest.approx(3000.0, rel=1e-3)
    assert pk.tau_peak == pytest.approx(3e-3, rel=0.03)
    assert pk.idx_lo < pk.idx_peak < pk.idx_hi and pk.other_candidates == []


def test_highest_peak_in_the_window_is_kept_and_others_reported():
    g = (_gauss(LN, np.log(1e-6), 0.5, 5000.0)          # hors fenêtre (plus haut) : ignoré
         + _gauss(LN, np.log(1e-3), 0.4, 2000.0)        # retenu
         + _gauss(LN, np.log(5e-2), 0.4, 500.0))        # candidat plus bas : signalé
    pk = main_peak(TAU, g, [1e-5, 1e-1])
    assert pk.ok and pk.tau_peak == pytest.approx(1e-3, rel=0.05)
    assert pk.rct == pytest.approx(2000.0, rel=0.02)    # intégré entre les minima qui l'encadrent
    assert len(pk.other_candidates) == 1 and "autre(s) pic(s)" in pk.message


def test_integration_stops_at_the_local_minima():
    g = _gauss(LN, np.log(1e-3), 0.5, 1200.0) + _gauss(LN, np.log(3e-2), 0.5, 1000.0)
    pk = main_peak(TAU, g, [1e-5, 1e-1])
    assert pk.ok and pk.rct < 1200.0 * 1.05          # pas l'aire des deux pics
    assert pk.tau_lo < pk.tau_peak < pk.tau_hi < 3e-2


def test_descending_tau_order_is_supported():
    g = _gauss(LN, np.log(3e-3), 0.6, 3000.0)
    pk = main_peak(TAU[::-1], g[::-1], [1e-5, 1e-1])
    assert pk.ok and pk.rct == pytest.approx(3000.0, rel=1e-3)
    assert TAU[::-1][pk.idx_peak] == pytest.approx(pk.tau_peak)


@pytest.mark.parametrize("tau, gamma, fragment", [
    (TAU, _gauss(LN, np.log(1e-6), 0.5, 100.0), "Aucun pic"),
    (TAU, np.zeros_like(TAU), "nulle"),
    (np.array([]), np.array([]), "invalide"),
])
def test_no_peak_gives_a_clear_message(tau, gamma, fragment):
    pk = main_peak(tau, gamma, [1e-5, 1e-1])
    assert not pk.ok and np.isnan(pk.rct) and fragment in pk.message


def test_roundtrip_dict():
    pk = main_peak(TAU, _gauss(LN, np.log(3e-3), 0.6, 3000.0), [1e-5, 1e-1])
    assert MainPeak.from_dict(pk.to_dict()) == pk


def test_rct_drt_within_5_percent_on_a_synthetic_rc_spectrum():
    """Spectre synthétique R_s + (R_ct ∥ C) de R_ct connu → Rct_DRT (pic principal) à ±5 %."""
    import drt.engine as eng
    ok, why = eng.engine_available()
    if not ok:
        pytest.skip(f"moteur DRT indisponible : {why}")
    from core.models import EISSpectrum
    rs, rct, c = 100.0, 3000.0, 1e-6
    f = np.logspace(5, -1, 61)
    w = 2 * np.pi * f
    Z = rs + rct / (1 + 1j * w * rct * c)
    rng = np.random.default_rng(0)
    zre = Z.real * (1 + 0.002 * rng.standard_normal(f.size))
    zim = -Z.imag * (1 + 0.002 * rng.standard_normal(f.size))
    sp = EISSpectrum("rc", f, zre, zim, 0.0, "probe", f.size)
    fr = eng.fit_drt(sp, mode="optimize")
    pk = main_peak(fr.drt_tau, fr.drt_gamma, [1e-5, 1e-1])
    assert pk.ok
    assert pk.rct == pytest.approx(rct, rel=0.05), pk
    assert pk.tau_peak == pytest.approx(rct * c, rel=0.5)
