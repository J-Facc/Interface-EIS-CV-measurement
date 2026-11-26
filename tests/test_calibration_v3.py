"""Étalonnages v3 (core/calibration.py) et CV ΔI/I (core/cv.py) — tests numériques de la phase 5."""

from pathlib import Path

import numpy as np
import pytest

from core.calibration import (
    LEGACY_LOGLOG,
    LEGACY_SIGNAL,
    MSG_SINGLE_REPLICATE,
    REASON_NO_PROBE,
    SOURCE_DRT,
    SOURCE_FIT,
    VIEW_BIOMARKER,
    VIEW_ELECTRODE,
    cv_calibration,
    eis_calibration,
    eis_points,
    legacy_curves,
)
from core.cv import anodic_peak, cv_features, normalized_delta
from core.cv_loader import load_cv_measurement
from core.drt_peak import MainPeak
from core.experiment import Channel, Experiment, Measurement
from core.results import (
    AnalysisResults,
    CVFeature,
    DRTRecord,
    EISResults,
    FitRecord,
    GroupRecord,
    SpectrumRecord,
    group_key,
    mean_key,
)

ARR = np.zeros(3)


def _fit(rct, converged=True):
    return FitRecord("Rct", {"Rct": rct}, {"Rct": 1.0}, 1.0, converged, "orazem", ARR, ARR)


def _drt(rct):
    pk = MainPeak(ok=True, rct=rct, tau_peak=1e-3, gamma_peak=1.0, tau_window=[1e-5, 1e-1])
    return DRTRecord("optimize", ARR, ARR, pk, rct * 1.01, float("nan"), "peak_penultimate", rct, True, ARR, ARR)


def _results(spec: dict, experiment: Experiment) -> AnalysisResults:
    """``spec`` : {(canal, électrode, étape): [Rct des réplicats]} → AnalysisResults (fit ET DRT = Rct)."""
    eis = EISResults(circuit={"target_param": "Rct", "parameters": {"Rct": {}}})
    for (ch, el, step), values in spec.items():
        conc = experiment.channel(ch).concentration_of(step)
        gk = group_key(ch, el, step)
        keys = []
        for r, v in enumerate(values, start=1):
            k = f"EIS_C{ch}_E{el}_{step}_r{r}"
            keys.append(k)
            eis.spectra[k] = SpectrumRecord(k, gk, ch, el, step, r, conc, ARR, ARR, ARR, fit=_fit(v), drt=_drt(v))
        mk = mean_key(ch, el, step)
        m = float(np.mean(values))
        eis.spectra[mk] = SpectrumRecord(mk, gk, ch, el, step, None, conc, ARR, ARR, ARR, fit=_fit(m), drt=_drt(m))
        eis.groups[gk] = GroupRecord(gk, ch, el, step, conc, experiment.channel(ch).biomarker, len(values),
                                     replicate_keys=keys, mean_key=mk)
    return AnalysisResults(eis=eis)


# ── (a) une concentration par canal ──────────────────────────────────────────

def test_one_concentration_per_channel_grouped_by_biomarker():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9]), Channel(2, "A", [1e-8]), Channel(3, "A", [1e-7]),
                             Channel(4, "A", [1e-8])])
    spec = {}
    for ch, probe, val in ((1, 1000, 1200), (2, 2000, 3000), (3, 1000, 2000), (4, 500, 800)):
        for el in (1, 2):
            spec[(ch, el, "probe")] = [probe * 0.99, probe * 1.01]
            spec[(ch, el, "conc_1")] = [val, val]
    res = _results(spec, exp)
    [curve] = eis_calibration(res, exp, SOURCE_FIT, VIEW_BIOMARKER)
    assert [lv.concentration for lv in curve.levels] == [1e-9, 1e-8, 1e-7]
    # 1e-8 : canaux 2 et 4 regroupés (2 électrodes × 2 réplicats chacun = 8 points).
    lv = curve.levels[1]
    ys = [np.log10(3000 / 2000)] * 4 + [np.log10(800 / 500)] * 4
    assert lv.n == 8 and lv.mean == pytest.approx(np.mean(ys))
    assert lv.std == pytest.approx(np.std(ys, ddof=1))
    assert curve.has_regression and 0 <= curve.r2 <= 1
    # Vue par électrode : 8 courbes d'un seul niveau, sans régression.
    per_el = eis_calibration(res, exp, SOURCE_FIT, VIEW_ELECTRODE)
    assert len(per_el) == 8 and all(len(c.levels) == 1 and not c.has_regression for c in per_el)


# ── (b) plusieurs concentrations sur un canal ────────────────────────────────

def test_several_concentrations_on_one_channel_normalized_per_electrode():
    exp = Experiment(["B"], [Channel(1, "B", [1e-10, 1e-9, 1e-8])])
    spec = {(1, 1, "probe"): [1000, 1100, 1200], (1, 2, "probe"): [2000, 2000],
            (1, 1, "conc_1"): [1200, 1300], (1, 1, "conc_2"): [1500], (1, 1, "conc_3"): [2000, 2200, 2100],
            (1, 2, "conc_1"): [2400, 2500], (1, 2, "conc_2"): [3000, 3100], (1, 2, "conc_3"): [4000, 4200]}
    res = _results(spec, exp)
    pts = eis_points(res, exp, SOURCE_FIT)
    e1 = [p for p in pts if p.electrode == 1 and p.step == "conc_1"]
    assert e1[0].reference == pytest.approx(1100.0)                      # moyenne des probe de l'électrode 1
    assert e1[0].y == pytest.approx(np.log10(1200 / 1100))
    e2 = [p for p in pts if p.electrode == 2 and p.step == "conc_1"]
    assert e2[0].reference == pytest.approx(2000.0)                      # PAS celle de l'électrode 1
    curves = {c.label: c for c in eis_calibration(res, exp, SOURCE_FIT, VIEW_ELECTRODE)}
    c1 = curves["B — canal 1 · électrode 1"]
    assert [lv.n for lv in c1.levels] == [2, 1, 3]
    one = c1.levels[1]                                                   # cas à 1 réplicat
    assert np.isnan(one.std) and one.note == MSG_SINGLE_REPLICATE and "1 seul réplicat" in c1.message
    third = [np.log10(v / 1100) for v in (2000, 2200, 2100)]
    assert c1.levels[2].mean == pytest.approx(np.mean(third))
    assert c1.levels[2].std == pytest.approx(np.std(third, ddof=1))
    x = [lv.log_c for lv in c1.levels]
    y = [lv.mean for lv in c1.levels]
    slope, intercept = np.polyfit(x, y, 1)
    assert c1.slope == pytest.approx(slope) and c1.intercept == pytest.approx(intercept)


# ── (c) mélange des deux modes pour un même biomarqueur ──────────────────────

def test_mixed_modes_are_regrouped_by_concentration():
    exp = Experiment(["C"], [Channel(1, "C", [1e-9, 1e-8]), Channel(2, "C", [1e-8])])
    spec = {(1, 1, "probe"): [1000], (1, 1, "conc_1"): [1500], (1, 1, "conc_2"): [2000],
            (2, 1, "probe"): [500], (2, 1, "conc_1"): [1000]}
    res = _results(spec, exp)
    [curve] = eis_calibration(res, exp, SOURCE_DRT, VIEW_BIOMARKER)
    lv = {l.concentration: l for l in curve.levels}
    assert lv[1e-8].n == 2
    assert lv[1e-8].mean == pytest.approx(np.mean([np.log10(2000 / 1000), np.log10(1000 / 500)]))
    assert lv[1e-9].n == 1 and not curve.has_regression


def test_points_without_probe_or_value_are_excluded_with_a_reason():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9, 1e-8, 1e-7])])
    res = _results({(1, 1, "conc_1"): [1000], (1, 2, "probe"): [1000], (1, 2, "conc_1"): [2000],
                    (1, 2, "conc_2"): [3000]}, exp)
    res.eis.spectra["EIS_C1_E2_conc_2_r1"].fit = _fit(3000, converged=False)
    pts = {(p.electrode, p.step): p for p in eis_points(res, exp, SOURCE_FIT)}
    assert not pts[(1, "conc_1")].included and pts[(1, "conc_1")].reason == REASON_NO_PROBE
    assert not pts[(2, "conc_2")].included and "non convergé" in pts[(2, "conc_2")].reason
    assert pts[(2, "conc_1")].included


def test_drt_and_fit_sources_are_both_available():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    res = _results({(1, 1, "probe"): [1000], (1, 1, "conc_1"): [2000]}, exp)
    res.eis.spectra["EIS_C1_E1_conc_1_r1"].drt = _drt(4000)
    fit = eis_points(res, exp, SOURCE_FIT)[0]
    drt = eis_points(res, exp, SOURCE_DRT)[0]
    assert fit.y == pytest.approx(np.log10(2)) and drt.y == pytest.approx(np.log10(4))


def test_legacy_curves_use_the_mean_spectrum():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9, 1e-8, 1e-7])])
    res = _results({(1, 1, "probe"): [1000, 1000], (1, 1, "conc_1"): [1500, 1700], (1, 1, "conc_2"): [2000],
                    (1, 1, "conc_3"): [3000]}, exp)
    [sig] = legacy_curves(res, exp, SOURCE_FIT, LEGACY_SIGNAL)
    assert [round(lv.mean, 6) for lv in sig.levels] == [0.6, 1.0, 2.0]   # |1000 − 1600|/1000…
    assert sig.has_regression
    [ll] = legacy_curves(res, exp, SOURCE_DRT, LEGACY_LOGLOG)
    assert ll.levels[0].mean == pytest.approx(np.log10(1600 * 1.01))     # Rct DRT moteur du spectre moyen


# ── CV ───────────────────────────────────────────────────────────────────────

def test_anodic_peak_on_a_synthetic_curve():
    """Courbe synthétique à pic connu : I_pic exact, sur la branche anodique seulement."""
    E = np.concatenate([np.linspace(0, -0.2, 50), np.linspace(-0.2, 0.4, 150), np.linspace(0.4, 0, 100)])
    I = 1e-6 * np.exp(-((E - 0.15) / 0.03) ** 2)
    I[200:] *= -3.0                     # branche cathodique (retour) : courants différents
    I[60] = 5e-6                        # point aberrant sur la branche anodique… hors fenêtre
    i_pk, e_pk, ok, _ = anodic_peak(E, I, [0.0, 0.3])
    k = np.argmin(np.abs(E[50:200] - 0.15)) + 50
    assert ok and i_pk == pytest.approx(I[k]) and e_pk == pytest.approx(E[k])
    assert anodic_peak(E, I, [0.5, 0.6])[2] is False


def test_normalized_current_by_hand():
    assert normalized_delta(4e-6, 3e-6) == pytest.approx(0.25)
    assert normalized_delta(4e-6, 5e-6) == pytest.approx(-0.25)
    assert np.isnan(normalized_delta(0.0, 1e-6))


def _cv(ch, el, step, rep, amp):
    E = np.concatenate([np.linspace(0, 0.25, 60), np.linspace(0.25, 0, 60)])
    I = amp * np.concatenate([np.linspace(0, 1, 60), np.linspace(1, -1, 60)])
    return Measurement(ch, el, step, rep, "CV", "x", {"E": E, "I": I, "cycle": np.ones_like(E)})


def test_cv_features_and_calibration_per_electrode():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9, 1e-8, 1e-7])])
    ms = [_cv(1, 1, "probe", 1, 4e-6), _cv(1, 1, "probe", 2, 6e-6),       # I_probe(E1) = 5 µA
          _cv(1, 2, "probe", 1, 10e-6),                                    # I_probe(E2) = 10 µA
          _cv(1, 1, "conc_1", 1, 4e-6), _cv(1, 1, "conc_1", 2, 3e-6),
          _cv(1, 1, "conc_2", 1, 2.5e-6), _cv(1, 1, "conc_3", 1, 1e-6),
          _cv(1, 2, "conc_1", 1, 8e-6)]
    feats = cv_features(ms, exp, [0.0, 0.25])
    f = feats["CV_C1_E1_conc_1_r2"]
    assert f.i_peak == pytest.approx(3e-6) and f.i_probe == pytest.approx(5e-6)
    assert f.delta_i_norm == pytest.approx((5e-6 - 3e-6) / 5e-6)
    assert feats["CV_C1_E2_conc_1_r1"].delta_i_norm == pytest.approx(0.2)   # sa propre référence
    res = AnalysisResults(cv=feats)
    [curve] = cv_calibration(res, exp, VIEW_ELECTRODE)[:1]
    lv = curve.levels[0]
    assert lv.n == 2 and lv.mean == pytest.approx(np.mean([0.2, 0.4]))
    assert lv.std == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert curve.has_regression


def test_cv_peak_on_the_real_file_and_missing_probe():
    d = load_cv_measurement((Path(__file__).parent / "data" / "cv_bare_2cycles.txt").read_bytes(), "x")
    i_pk, e_pk, ok, _ = anodic_peak(d["E"], d["I"], [0.0, 0.25])
    assert ok and 3.0e-6 < i_pk < 3.5e-6 and 0.15 < e_pk <= 0.2
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    feats = cv_features([_cv(1, 1, "conc_1", 1, 1e-6)], exp, [0.0, 0.25])
    assert "aucun pic « probe »" in feats["CV_C1_E1_conc_1_r1"].message
    assert not cv_calibration(AnalysisResults(cv=feats), exp)[0].levels
