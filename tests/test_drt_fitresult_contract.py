"""Contrat DRT ↔ nouveau ``FitResult`` (étape 4), isolé du pipeline.

``FitResult`` n'a plus de champ ``Rct``/``Rct_std`` figé : la DRT doit produire
``target_param = "Rct"``, ``target_value``, ``target_std``, et ne plus produire les
champs retirés (``drt_S``, ``drt_lnGamma``, ``Rct_sigma``, ``chi2_is_valid_test``).

Deux niveaux :

  A. ``drt/engine.py`` avec un ``Inverter`` FACTICE dont γ(τ) est connu (deux pics
     log-normaux normalisés, donc Rct attendu exact) : vérifie ce que le moteur
     CONSOMME (spectrum en convention Zim = −Im(Z) > 0 → Z physique, HF → BF) et
     PRODUIT. Tourne partout, sans cvxopt ni CmdStan (job CI « validate »).
  B. Le moteur DRT RÉEL en mode MAP (``drt/engine.py``, seul moteur DRT depuis
     l'étape 5 — l'ancien plugin ``fits/drt_fit.py`` est supprimé) : sauté sans
     CmdStan, exécuté par le job CI « drt ».
"""

from dataclasses import fields

import numpy as np
import pytest

from core.calibration import compute_calibration_loglog
from core.models import ConcentrationGroup, EISSession, EISSpectrum, FitResult
from drt import engine
from tests.synthetic_data import N_POINTS_AUDIT, noisy_arrays

_REMOVED = ("Rct", "Rct_std", "Rct_sigma", "drt_S", "drt_lnGamma", "chi2_is_valid_test",
            "kk_passed", "kk_residuals")   # verdict KK porté par le GROUPE depuis l'étape 5

# γ(τ) factice : R1 à τ1 (transfert de charge), R2 à τ2 (diffusion), largeur s en ln τ.
_R0, _R1, _TAU1, _R2, _TAU2, _S = 150.0, 2500.0, 1e-3, 800.0, 1.0, 0.4
_TAU = np.geomspace(1e-7, 1e2, 120)


def _gamma(tau):
    """γ (Ω, par unité de ln τ) : ∫ γ dlnτ = R1 + R2."""
    ln = np.log(tau)
    peak = lambda r, t: r * np.exp(-0.5 * ((ln - np.log(t)) / _S) ** 2) / (_S * np.sqrt(2 * np.pi))
    return peak(_R1, _TAU1) + peak(_R2, _TAU2)


def _z_of(f):
    """Z(ω) = R0 + ∫ γ(τ)/(1 + jωτ) dlnτ — l'impédance EXACTE de cette DRT."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    g = _gamma(_TAU)
    return _R0 + np.trapezoid(g[None, :] / (1 + 1j * w[:, None] * _TAU[None, :]), np.log(_TAU), axis=1)


class _FakeMLE:
    converged = True


class _FakeInverter:
    """Sous-ensemble de ``bayes_drt2.Inverter`` utilisé par ``drt/engine.py`` (mode MAP)."""

    calls = []
    stan_model_name = "Series (factice)"

    def __init__(self, distributions):           # dict neuf exigé (engine._fresh_distributions)
        self.distributions = {engine.DIST_NAME: {"tau": _TAU}}
        self.stan_mle = _FakeMLE()

    def fit(self, f, Z, **kw):
        _FakeInverter.calls.append((np.array(f), np.array(Z), kw))

    def predict_distribution(self, name, tau, percentile=None):
        assert name == engine.DIST_NAME and percentile is None
        return _gamma(np.asarray(tau))

    def predict_Rp(self, percentile=None):
        return float(np.trapezoid(_gamma(_TAU), np.log(_TAU)))

    def predict_Z(self, f):
        return _z_of(f)


def _spectrum(f=None):
    f = np.logspace(5, -1, 40) if f is None else f            # HF → BF, comme le loader
    z = _z_of(f)
    return EISSpectrum(label="drt-contrat", f=f, Zre=z.real, Zim=-z.imag,
                       concentration=1e-9, step="hybridization", n_points=len(f))


@pytest.fixture
def fake_engine(monkeypatch):
    _FakeInverter.calls = []
    monkeypatch.setattr(engine, "engine_available", lambda: (True, None))
    monkeypatch.setattr(engine, "_import_inverter", lambda: _FakeInverter)
    return _FakeInverter


def _assert_new_fitresult_contract(fr):
    assert isinstance(fr, FitResult)
    names = {fl.name for fl in fields(FitResult)}
    assert {"target_param", "target_value", "target_std"} <= names
    for gone in _REMOVED:
        assert gone not in names, gone
        assert not hasattr(fr, gone), gone
    assert fr.target_param == "Rct"
    assert fr.target_value == fr.params["Rct"]
    assert fr.drt_tau is not None and fr.drt_tau.shape == fr.drt_gamma.shape
    assert fr.Zfit_re.shape == fr.Zfit_im.shape == fr.residuals_re.shape == fr.residuals_im.shape


# ═════════════════════════════════════════════════════════════════════════════
# A. drt/engine.py avec Inverter factice (toujours exécuté)
# ═════════════════════════════════════════════════════════════════════════════

def test_engine_consumes_the_app_convention_and_feeds_inverter_physical_z(fake_engine):
    sp = _spectrum(np.random.default_rng(0).permutation(np.logspace(5, -1, 40)))   # ordre quelconque
    engine.fit_drt(sp, mode="optimize")
    (f_in, Z_in, kw), = fake_engine.calls
    assert np.all(np.diff(f_in) < 0)                         # HF → BF
    assert np.all(Z_in.imag < 0)                             # Z physique = Zre − j·Zim
    np.testing.assert_allclose(Z_in, _z_of(f_in), rtol=1e-12)
    assert kw["mode"] == "optimize" and kw["random_seed"] == engine.DEFAULT_RANDOM_SEED


def test_engine_produces_the_new_fitresult_fields(fake_engine):
    sp = _spectrum()
    fr = engine.fit_drt(sp, mode="optimize")
    _assert_new_fitresult_contract(fr)
    # Rct = aire du pic PÉNULTIÈME (τ1), dans la fenêtre de τ mesurée.
    assert fr.target_value == pytest.approx(_R1, rel=1e-3)
    assert fr.params["tau_Rct"] == pytest.approx(_TAU1, rel=0.1)         # secondes (DRT-4)
    assert fr.params["Rp"] == pytest.approx(_R1 + _R2, rel=1e-6)
    assert fr.drt_diagnostics["rct_source"] == "peak_penultimate"
    # MAP : pas d'incertitude a posteriori — NaN explicite, jamais un 0 inventé.
    assert np.isnan(fr.target_std) and np.isnan(fr.params_std["Rct"])
    assert np.isnan(fr.chi2_reduced)                                      # DRT-5
    assert fr.drt_mode == "optimize" and fr.converged is True
    assert fr.drt_gamma_lo is None and fr.drt_gamma_hi is None
    # Retour dans la convention de l'app et dans l'ordre du spectre.
    np.testing.assert_allclose(fr.Zfit_im, sp.Zim, rtol=1e-9)
    np.testing.assert_allclose(fr.residuals_im, sp.Zim - fr.Zfit_im, atol=1e-9)
    assert fr.reconstruction_error_relative < 1e-9                       # max |Ẑ − Z|/|Z|
    assert fr.fit_diagnostics is None and fr.chi2_reduced_ci is None     # champs propres au fit Orazem


def test_the_calibration_reads_the_drt_target_value(fake_engine):
    """Consommateur principal du champ : la calibration lit ``target_value``."""
    session = EISSession()
    for conc, scale in ((1e-9, 1.0), (1e-8, 1.2), (1e-7, 1.5)):
        sp = _spectrum()
        fr = engine.fit_drt(sp, mode="optimize")
        fr.target_value *= scale
        session.groups.append(ConcentrationGroup(concentration=conc, spectrum=sp,
                                                 fit_results={engine.MODEL_NAME: fr}))
    cal = compute_calibration_loglog(session, engine.MODEL_NAME)
    np.testing.assert_allclose(cal.rcts, [_R1, 1.2 * _R1, 1.5 * _R1], rtol=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# B. Moteurs DRT réels, mode MAP (job CI « drt »)
# ═════════════════════════════════════════════════════════════════════════════

_HAVE_ENGINE, _WHY = engine.engine_available()
_NEEDS_ENGINE = pytest.mark.skipif(not _HAVE_ENGINE, reason=f"DRT réelle : {_WHY} (job CI « drt »)")


def _randles_spectrum(rct=3000.0):
    f, zre, zim = noisy_arrays(rct, 0.005, seed=3, n_points=N_POINTS_AUDIT)
    return EISSpectrum(label="randles", f=f, Zre=zre, Zim=zim, concentration=0.0,
                       step="probe", n_points=len(f))


@_NEEDS_ENGINE
def test_real_engine_map_produces_the_new_fitresult_contract():
    fr = engine.fit_drt(_randles_spectrum(), mode="optimize")
    _assert_new_fitresult_contract(fr)
    assert fr.target_value == pytest.approx(3000.0, rel=0.05)    # RCT_REL_TOL de test_drt_engine (mesuré : +2,2 %)
    assert np.isnan(fr.target_std)                               # MAP : non calculée
    assert fr.converged is True and fr.drt_mode == "optimize"
    assert fr.drt_diagnostics["settings"]["mode"] == "optimize"
