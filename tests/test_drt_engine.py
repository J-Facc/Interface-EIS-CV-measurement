# -*- coding: utf-8 -*-
"""Tests du moteur DRT durci (drt/engine.py) et de ses gardes qualité (drt/diagnostics.py).

Trois niveaux :

(a) **Gardes pures** (drt/diagnostics.py) — sans CmdStan ; tournent dans le job CI
    « validate ».
(b) **Briques pures du moteur** (réglages, entrée, Rct / τ en secondes) — sans CmdStan.
(c) **DRT réelle** sur les cas où AUDIT.md a mesuré l'échec silencieux DRT-1 — exigent
    cvxopt + cmdstanpy + CmdStan (job CI « drt ») ; sautés proprement sinon.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.models import EISSpectrum
from drt import diagnostics as dg
from drt import engine

_HAVE_ENGINE, _WHY = engine.engine_available()
_NEEDS_ENGINE = pytest.mark.skipif(
    not _HAVE_ENGINE,
    reason=f"DRT réelle : exige cvxopt + cmdstanpy + CmdStan (job CI « drt ») — {_WHY}",
)


# ═════════════════════════════════════════════════════════════════════════════
# (a) Gardes pures
# ═════════════════════════════════════════════════════════════════════════════
def _good_sampler_diag(chains=4):
    return {
        "chains": chains, "draws_per_chain": 500, "divergences": 0, "max_treedepth_hits": 0,
        "rhat_max": 1.01, "rhat_max_variable": "x[3]", "ess_bulk_min": 800.0, "ess_tail_min": 700.0,
        "ebfmi_per_chain": [0.8] * chains,
    }


def test_check_sampler_accepts_clean_diagnostics():
    assert dg.check_sampler(_good_sampler_diag()) == []


@pytest.mark.parametrize("key, value, code", [
    ("rhat_max", 1.0501, "rhat_high"),
    ("divergences", 1, "divergences"),
    ("ess_bulk_min", 399.0, "ess_bulk_low"),     # 100 × 4 chaînes
    ("ess_tail_min", 10.0, "ess_tail_low"),
    ("ebfmi_per_chain", [0.8, 0.29, 0.8, 0.8], "ebfmi_low"),
])
def test_check_sampler_flags_each_threshold(key, value, code):
    diag = _good_sampler_diag()
    diag[key] = value
    alerts = dg.check_sampler(diag)
    assert dg.codes(alerts) == [code]
    assert all(a.category == "convergence" for a in alerts)


def test_rhat_threshold_is_the_documented_user_decision():
    """Seuils = décision utilisateur (R-hat > 1,05 ou divergences > 0 ⇒ alerte)."""
    assert dg.RHAT_MAX == 1.05
    assert dg.MAX_DIVERGENCES == 0
    diag = _good_sampler_diag()
    diag["rhat_max"] = 1.05                       # égal au seuil : toléré
    assert dg.check_sampler(diag) == []


@pytest.mark.parametrize("key", ["rhat_max", "divergences", "ess_bulk_min", "ess_tail_min"])
def test_missing_diagnostic_is_an_alert_never_a_success(key):
    diag = _good_sampler_diag()
    diag[key] = None
    assert dg.check_sampler(diag), f"{key}=None doit produire une alerte"


def test_treedepth_saturation_is_reported_but_not_an_alert():
    """Efficacité, pas validité (doc. Stan) : ses effets sont gardés via R-hat et l'ESS."""
    diag = _good_sampler_diag()
    diag["max_treedepth_hits"] = 2000
    assert dg.check_sampler(diag) == []
    diag["ess_bulk_min"] = 50.0                    # … mais une exploration lente se voit ici
    assert dg.codes(dg.check_sampler(diag)) == ["ess_bulk_low"]


def test_ebfmi_matches_definition():
    rng = np.random.default_rng(0)
    e = rng.standard_normal((500, 2))
    expected = [np.sum(np.diff(e[:, c]) ** 2) / np.sum((e[:, c] - e[:, c].mean()) ** 2) for c in range(2)]
    assert np.allclose(dg.ebfmi(e), expected)
    # Chaîne très autocorrélée (marche aléatoire) → E-BFMI ≪ 0,3
    walk = np.cumsum(rng.standard_normal(500))
    assert dg.ebfmi(walk)[0] < 0.3


def test_reconstruction_error_relative_definition():
    Z = np.array([100 - 10j, 50 - 5j, 20 - 1j])
    Zf = Z * np.array([1.01, 1.0, 0.9])
    rec = dg.reconstruction_error_relative(Z, Zf)
    assert rec["max"] == pytest.approx(0.1)
    assert rec["mean"] == pytest.approx((0.01 + 0 + 0.1) / 3)
    assert rec["rms"] == pytest.approx(np.sqrt((0.01 ** 2 + 0.1 ** 2) / 3))


def test_check_quality_flags_the_audit_failure_values():
    """Valeurs mesurées par AUDIT.md §4.4 (DRT-1) → alertes, jamais le silence (DRT-2)."""
    alerts = dg.check_quality(rp=-159.4, recon={"max": 3.63, "mean": 2.0, "rms": 2.5})
    assert dg.codes(alerts) == ["rp_nonpositive", "reconstruction_error"]
    assert all(a.category == "qualite" for a in alerts)
    assert dg.check_quality(rp=float("nan"), recon={"max": 0.01})[0].code == "rp_nonpositive"
    assert dg.check_quality(rp=130.2, recon={"max": 0.0076, "mean": 0.004, "rms": 0.004}) == []


def test_negative_area_fraction():
    tau = np.logspace(-6, 2, 200)
    lt = np.log(tau)
    pos = 50 * np.exp(-0.5 * ((lt - np.log(1e-3)) / 0.5) ** 2)
    assert dg.negative_area_fraction(tau, pos) == 0.0
    neg = pos - 60 * np.exp(-0.5 * ((lt - np.log(1e-1)) / 0.5) ** 2)
    assert dg.negative_area_fraction(tau, neg) == pytest.approx(60 / 110, rel=1e-3)   # 55 % > 25 %
    alerts = dg.check_quality(rp=10.0, recon={"max": 0.01}, gamma=neg, tau=tau)
    assert dg.codes(alerts) == ["gamma_negative"]


# ═════════════════════════════════════════════════════════════════════════════
# (b) Briques pures du moteur
# ═════════════════════════════════════════════════════════════════════════════
def test_defaults_are_explicit_documented_constants():
    s = engine.DRTSettings()
    assert s.mode == engine.DEFAULT_MODE == "sample"          # HMC par défaut (décision utilisateur)
    assert s.random_seed == engine.DEFAULT_RANDOM_SEED
    assert s.nonneg is engine.DEFAULT_NONNEG
    assert s.init_from_ridge is engine.DEFAULT_INIT_FROM_RIDGE


@pytest.mark.parametrize("kw", [
    {"mode": "map"}, {"random_seed": -1}, {"random_seed": 2 ** 32}, {"random_seed": 1.5},
    {"random_seed": True}, {"chains": 0}, {"warmup": 0}, {"samples": 0}, {"max_iter": 0},
    {"adapt_delta": 1.0}, {"adapt_delta": 0.0}, {"adapt_delta": True},
])
def test_settings_reject_invalid_values(kw):
    with pytest.raises(ValueError):
        engine.DRTSettings(**kw)


def _gauss_drt(centers_s, amps, sigma=0.5, n=300):
    tau = np.logspace(-7, 3, n)
    lt = np.log(tau)
    g = sum(a * np.exp(-0.5 * ((lt - np.log(c)) / sigma) ** 2) for c, a in zip(centers_s, amps))
    return tau, g


def test_rct_window_selects_penultimate_peak_and_tau_is_in_seconds():
    """DRT-4 : τ_Rct est un temps (s), pas ln τ."""
    tau, g = _gauss_drt([1e-3, 1e-1], [50.0, 90.0])
    mask, idx, source, note = engine.rct_window(tau, g)
    assert source == "peak_penultimate" and note == ""
    assert tau[idx] == pytest.approx(1e-3, rel=0.05)          # secondes
    area = engine._integrate(tau, g, mask, idx)[0]
    assert area == pytest.approx(50 * 0.5 * np.sqrt(2 * np.pi), rel=0.02)


def test_rct_peak_search_is_restricted_to_the_measured_window():
    """Mécanisme mesuré (VALIDATION_REGLAGES.md §5) : une bosse de 0,2 % du max HORS fenêtre
    (τ = 6 s > 1/(2π·0,1 Hz)) devenait le « dernier » pic → pénultième = diffusion."""
    tau, g = _gauss_drt([3e-3, 2e-1, 6.3], [2300.0, 1200.0, 4.5], sigma=0.4)
    f = np.logspace(5, -1, 60)
    bounds = engine.measured_tau_window(f)
    assert bounds == pytest.approx((1 / (2 * np.pi * 1e5), 1 / (2 * np.pi * 0.1)))
    _m, idx_old, *_ = engine.rct_window(tau, g)                 # règle historique
    assert tau[idx_old] == pytest.approx(2e-1, rel=0.1)          # → diffusion (faux)
    _m, idx_new, source, _n = engine.rct_window(tau, g, bounds)
    assert source == "peak_penultimate"
    assert tau[idx_new] == pytest.approx(3e-3, rel=0.1)          # → transfert de charge
    assert engine.RCT_PEAKS_IN_MEASURED_WINDOW is True


def test_rct_window_single_and_no_peak():
    tau, g = _gauss_drt([1e-2], [40.0])
    assert engine.rct_window(tau, g)[2] == "peak_single"
    mask, idx, source, note = engine.rct_window(tau, np.zeros_like(tau))
    assert mask is None and source == "none" and note


def test_integrate_is_linear_over_draws():
    """La MÊME fenêtre appliquée à chaque tirage : moyenne des aires = aire de la moyenne."""
    tau, g = _gauss_drt([1e-3, 1e-1], [50.0, 90.0])
    mask, idx, _s, _n = engine.rct_window(tau, g)
    rng = np.random.default_rng(1)
    draws = g[None, :] * (1 + 0.05 * rng.standard_normal((200, 1)))
    per_draw = engine._integrate(tau, draws, mask, idx)
    assert per_draw.shape == (200,)
    assert per_draw.mean() == pytest.approx(engine._integrate(tau, draws.mean(axis=0), mask, idx)[0])


@pytest.mark.parametrize("mutate, msg", [
    (lambda f, zr, zi: (f[:5], zr[:5], zi[:5]), "trop court"),
    (lambda f, zr, zi: (f, np.where(np.arange(f.size) == 3, np.nan, zr), zi), "non finies"),
    (lambda f, zr, zi: (np.where(np.arange(f.size) == 0, -1.0, f), zr, zi), "≤ 0"),
    (lambda f, zr, zi: (np.where(np.arange(f.size) == 1, f[0], f), zr, zi), "dupliquées"),
])
def test_invalid_spectra_are_rejected_before_any_fit(mutate, msg):
    f = np.logspace(5, -1, 30)
    Z = 10 + 50 / (1 + 1j * 2 * np.pi * f * 1e-3)
    f2, zr, zi = mutate(f, Z.real, -Z.imag)
    sp = EISSpectrum(label="bad", f=f2, Zre=zr, Zim=zi, concentration=0.0, step="probe", n_points=len(f2))
    with pytest.raises(ValueError, match=msg):
        engine.fit_drt(sp)


# ═════════════════════════════════════════════════════════════════════════════
# (c) DRT réelle — cas où AUDIT.md §4.4 a mesuré l'échec silencieux DRT-1
# ═════════════════════════════════════════════════════════════════════════════
# L'ancien moteur (fits/drt_fit.py, réglages amont) rendait sur CES spectres Rp < 0
# (−159 Ω ; −5678…−9278 Ω) avec converged=True et warnings=[] — comportement figé par
# tests/test_pipeline.py::test_drt_default_map_is_silently_wrong_on_randles_like_spectra.
# Ici, le moteur durci avec ses réglages PAR DÉFAUT doit donner le bon résultat SANS alerte.

#: Tolérances — marges au-dessus des écarts mesurés (drt/VALIDATION_REGLAGES.md §6).
RP_REL_TOL = 0.02     # biais mesuré de la moyenne a posteriori sous nonneg : +0,6 à +1,5 %
RCT_REL_TOL = 0.05    # 2 RC : ≤ 0,5 % ; Randles : biais de la convention ±3 ln τ ≈ +2 %


def _two_rc_spectrum(n=60, f0=5, f1=-1, noise=0.0, seed=0):
    """AUDIT.md Annexe A.3 : R0=10 + 50/(1+jω·1 ms) + 80/(1+jω·0,1 s) ; Rp = 130 Ω."""
    f = np.logspace(f0, f1, n)
    w = 2 * np.pi * f
    Z = 10 + 50 / (1 + 1j * w * 1e-3) + 80 / (1 + 1j * w * 1e-1)
    r = np.random.default_rng(seed)
    Z = Z * (1 + noise * (r.standard_normal(n) + 1j * r.standard_normal(n)))
    return EISSpectrum(label="2RC", f=f, Zre=Z.real, Zim=-Z.imag, concentration=0.0, step="probe", n_points=n)


def _randles_spectrum(Rct):
    """AUDIT.md Annexe A.4 : Randles complet, 60 pts 1e5→1e-1 Hz, bruit 0,5 %, graine = Rct."""
    from tests.synthetic_data import noisy_arrays

    f, zre, zim = noisy_arrays(Rct, noise=0.005, seed=int(Rct), n_points=60)
    return EISSpectrum(label=f"Randles{Rct}", f=f, Zre=zre, Zim=zim, concentration=0.0, step="probe",
                       n_points=len(f))


# (id, fabrique, graine Stan, Rp vrai, Rct visé, τ visé [s] (bornes))
DRT1_CASES = [
    ("A3-60pts-sans-bruit", lambda: _two_rc_spectrum(), 1234, 130.0, 50.0, (0.7e-3, 1.4e-3)),
    ("A3-60pts-bruit-graine1234", lambda: _two_rc_spectrum(noise=0.005), 1234, 130.0, 50.0, (0.7e-3, 1.4e-3)),
    ("A3-60pts-bruit-graine3", lambda: _two_rc_spectrum(noise=0.005), 3, 130.0, 50.0, (0.7e-3, 1.4e-3)),
    ("A4-Randles-Rct3000", lambda: _randles_spectrum(3000), 1234, 20 + 1.3 * 3000, 3000.0, (1e-3, 1e-2)),
    ("A4-Randles-Rct3500", lambda: _randles_spectrum(3500), 1234, 20 + 1.3 * 3500, 3500.0, (1e-3, 1e-2)),
    ("A4-Randles-Rct4200", lambda: _randles_spectrum(4200), 1234, 20 + 1.3 * 4200, 4200.0, (1e-3, 1e-2)),
    ("A4-Randles-Rct5200", lambda: _randles_spectrum(5200), 1234, 20 + 1.3 * 5200, 5200.0, (1e-3, 1e-2)),
]
_IDS = [c[0] for c in DRT1_CASES]


def _assert_correct_and_silent(fr, rp_true, rct_target, tau_bounds):
    d = fr.drt_diagnostics
    rp = fr.params["Rp"]
    assert rp > 0, "Rp doit être positif (DRT-1)"
    assert abs(rp / rp_true - 1) <= RP_REL_TOL, f"Rp={rp:.1f} vs {rp_true:.1f}"
    assert d["alerts"] == [], f"alertes inattendues : {d['alerts']}"
    assert fr.warnings == [], f"avertissements inattendus : {fr.warnings}"
    assert fr.converged is True
    assert d["quality_ok"] is True
    # DRT-5 : pas de χ² pour la DRT ; l'erreur de reconstruction a son propre champ
    assert np.isnan(fr.chi2_reduced)
    assert 0 < fr.reconstruction_error_relative < dg.RECONSTRUCTION_MAX_REL_MAX
    # DRT-4 : τ_Rct en SECONDES, sur l'arc de transfert de charge
    assert fr.params["rct_source"] == "peak_penultimate"
    assert tau_bounds[0] <= fr.params["tau_Rct"] <= tau_bounds[1], fr.params["tau_Rct"]
    assert abs(fr.Rct / rct_target - 1) <= RCT_REL_TOL, f"Rct={fr.Rct:.1f} vs {rct_target:.1f}"


@_NEEDS_ENGINE
@pytest.mark.parametrize("cid, make, seed, rp_true, rct_target, tau_bounds", DRT1_CASES, ids=_IDS)
def test_default_map_preview_is_correct_on_drt1_cases(cid, make, seed, rp_true, rct_target, tau_bounds):
    """mode='optimize' (aperçu) avec les réglages par défaut du moteur : correct, sans alerte."""
    fr = engine.fit_drt(make(), mode="optimize", random_seed=seed)
    assert fr.drt_mode == "optimize"
    assert fr.drt_gamma_lo is None and fr.drt_gamma_hi is None
    _assert_correct_and_silent(fr, rp_true, rct_target, tau_bounds)


@_NEEDS_ENGINE
@pytest.mark.slow
@pytest.mark.parametrize("cid, make, seed, rp_true, rct_target, tau_bounds", DRT1_CASES, ids=_IDS)
def test_default_hmc_is_correct_and_converged_on_drt1_cases(cid, make, seed, rp_true, rct_target, tau_bounds):
    """Réglage PAR DÉFAUT (HMC) : correct, convergé (R-hat, divergences, ESS…) et sans alerte."""
    fr = engine.fit_drt(make(), random_seed=seed)
    d = fr.drt_diagnostics
    assert fr.drt_mode == "sample" and d["settings"]["mode"] == "sample"
    s = d["sampler"]
    assert s["rhat_max"] <= dg.RHAT_MAX and s["divergences"] == 0
    assert s["chains"] == engine.DEFAULT_CHAINS and s["draws_per_chain"] == engine.DEFAULT_SAMPLES
    _assert_correct_and_silent(fr, rp_true, rct_target, tau_bounds)
    # Incertitude a posteriori (HMC) : IC cohérents, jamais un écart-type nul inventé
    lo, hi = d["Rct_ci95"]
    assert lo < fr.Rct < hi and fr.Rct_std > 0
    lo, hi = d["Rp_ci95"]
    assert lo < fr.params["Rp"] < hi
    assert np.all(fr.drt_gamma_lo <= fr.drt_gamma + 1e-9) and np.all(fr.drt_gamma <= fr.drt_gamma_hi + 1e-9)


@_NEEDS_ENGINE
def test_upstream_default_settings_are_now_flagged_instead_of_silent():
    """DRT-2 : les réglages amont (nonneg=False, init_from_ridge=False) donnent toujours
    l'optimum dégénéré en MAP — mais il est désormais SIGNALÉ (Rp ≤ 0, reconstruction, γ < 0)."""
    fr = engine.fit_drt(_two_rc_spectrum(), mode="optimize", nonneg=False, init_from_ridge=False)
    codes = [a["code"] for a in fr.drt_diagnostics["alerts"]]
    assert fr.params["Rp"] < 0                                   # même optimum que l'audit
    assert {"rp_nonpositive", "reconstruction_error", "gamma_negative"} <= set(codes)
    assert fr.warnings and fr.drt_diagnostics["quality_ok"] is False


@_NEEDS_ENGINE
def test_explicit_seed_makes_the_map_reproducible():
    sp = _two_rc_spectrum(noise=0.005)
    a = engine.fit_drt(sp, mode="optimize", random_seed=7)
    b = engine.fit_drt(sp, mode="optimize", random_seed=7)
    assert np.array_equal(a.drt_gamma, b.drt_gamma)
    assert a.drt_diagnostics["settings"]["random_seed"] == 7


@_NEEDS_ENGINE
@pytest.mark.slow
def test_explicit_seed_makes_the_hmc_reproducible():
    """Même graine ⇒ mêmes tirages. Budget réduit (propriété indépendante du budget)."""
    sp = _two_rc_spectrum(noise=0.005)
    budget = dict(chains=2, warmup=150, samples=150)
    a = engine.fit_drt(sp, random_seed=11, **budget)
    b = engine.fit_drt(sp, random_seed=11, **budget)
    assert np.array_equal(a.drt_gamma, b.drt_gamma)
    assert a.Rct == b.Rct and a.drt_diagnostics["sampler"]["rhat_max"] == b.drt_diagnostics["sampler"]["rhat_max"]
