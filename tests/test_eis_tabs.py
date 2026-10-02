"""Les quatre onglets EIS (ui/tabs.py::render_eis_tabs) : Visualisation / Measurement model &
fit Orazem / DRT / Calibration.

Organisation
    A. Lignes de données (core/results_table.py), sans Streamlit : verdict KK, diagnostics de
       fit (κ, rang, identifiabilité, bornes, χ²ᵣ), alertes, paramètres, diagnostics DRT,
       enveloppe inter-réplicats
    B. Figures (plotting/eis_plots.py) : smoke-tests de rendu sur de vrais résultats du pipeline
    D. Calibration : filtre (convergence, KK), raisons d'exclusion, cohérence avec l'export
    C. Rendu de la vraie page EIS (Streamlit AppTest) : structure à trois onglets, groupe
       « indéterminé », avertissement κ visuellement distinct, diagnostics DRT toujours visibles
       (DRT factice : ni CmdStan ni cvxopt requis)
"""

import copy
import io
from pathlib import Path

import numpy as np
import pytest

import core.results_table as rt
import drt.engine as drt_engine
from core.models import DisplayGroup, FitResult
from core.pipeline import DRT_MODEL_NAME, run_pipeline
from fits.orazem_fit import CONDITION_NUMBER_WARN
from plotting.eis_plots import (
    bode_figure,
    drt_aggregate_figure,
    drt_figure,
    fit_nyquist_figure,
    fit_residuals_figure,
    nyquist_replicates_figure,
)
from tests.synthetic_data import make_config, randles_file, replicate_assignments

REPO = Path(__file__).resolve().parents[1]
_RCT_PROBE, _RCT_C1, _C1 = 3000.0, 3500.0, 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Outils
# ─────────────────────────────────────────────────────────────────────────────

class _FakeDRT:
    """Remplace ``drt.engine.fit_drt`` : DRT plausible, différente d'un spectre à l'autre
    (sinon l'enveloppe inter-réplicats serait dégénérée), ou exception pour un label."""

    def __init__(self, fail=None, alerts=None):
        self.fail = dict(fail or {})
        self.alerts = list(alerts or [])
        self.n = 0

    def __call__(self, spectrum, *, mode, **kwargs):
        if spectrum.label in self.fail:
            raise self.fail[spectrum.label]
        self.n += 1
        n = len(spectrum.f)
        hmc = mode == "sample"
        tau = np.geomspace(1e-6, 10.0, 30)
        gamma = (1.0 + 0.1 * self.n) * np.exp(-0.5 * ((np.log(tau) + 3.0) / 1.5) ** 2) + 1e-3
        rct = float(np.max(spectrum.Zre) - np.min(spectrum.Zre))
        sampler = ({"rhat_max": 1.002, "divergences": 0, "ess_bulk_min": 900.0, "ess_tail_min": 800.0,
                    "chains": 4, "ebfmi_per_chain": [0.9, 0.8, 0.9, 0.8]}
                   if hmc else {"optimizer_converged": True})
        return FitResult(
            model_name=DRT_MODEL_NAME, params={"Rct": rct, "Rp": rct * 1.02, "drt_mode": mode},
            params_std={"Rct": 5.0 if hmc else float("nan")}, Zfit_re=np.asarray(spectrum.Zre, float),
            Zfit_im=np.asarray(spectrum.Zim, float), chi2_reduced=float("nan"),
            residuals_re=np.zeros(n), residuals_im=np.zeros(n), target_param="Rct",
            target_value=rct, target_std=5.0 if hmc else float("nan"), converged=True, drt_mode=mode,
            drt_tau=tau, drt_gamma=gamma,
            drt_gamma_lo=0.9 * gamma if hmc else None, drt_gamma_hi=1.1 * gamma if hmc else None,
            drt_diagnostics={"sampler": sampler, "alerts": [], "Rct_ci95": [rct - 10, rct + 10],
                             "Rp_ci95": [rct, rct * 1.05], "rct_source": "peak_penultimate"},
            warnings=list(self.alerts),
        )


def _experiment_clean(n_rep):
    probe = [io.BytesIO(randles_file(_RCT_PROBE, seed=k)) for k in range(n_rep)]
    c1 = [io.BytesIO(randles_file(_RCT_C1, seed=10 + k)) for k in range(n_rep)]
    return {"mode": "eis_only", "n_electrodes": 1, "concentrations": [_C1],
            "probe": {"eis": {"electrode_1": probe}}, "calibration": {"eis": {"electrode_1": [c1]}}}


@pytest.fixture(scope="module")
def nominal():
    """probe + 1 concentration, 3 réplicats, structure d'erreur caractérisable, sans DRT."""
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(enabled=False))
    return session


@pytest.fixture(scope="module")
def stopped():
    """Un seul fichier par condition : groupes ARRÊTÉS (verdict KK indéterminé)."""
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 1, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 1, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(enabled=False))
    return session


@pytest.fixture(scope="module")
def with_drt():
    """Même run avec la DRT factice en HMC ('sample')."""
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1"))
    fake = _FakeDRT(fail={"probe_r1.txt": RuntimeError("compilation Stan échouée (simulée)")})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(drt_engine, "fit_drt", fake)
        mp.setattr(drt_engine, "engine_available", lambda: (True, None))
        session, _ = run_pipeline(fa, make_config(enabled=True, mode="sample"))
    return session


def _fit_with_diagnostics(**diag):
    base = dict(condition_number=1e3, rank=2, identifiable={"a": True, "b": True},
                active_bounds=[], jacobian_one_sided=[], n_starts=8, n_converged=8, dof=70, dof_sigma=None)
    base.update(diag)
    z = np.ones(10)
    return FitResult(
        model_name="orazem", params={"a": 1.0, "b": 2.0}, params_std={"a": 0.1, "b": 0.2},
        Zfit_re=z, Zfit_im=z, chi2_reduced=1.0, residuals_re=z, residuals_im=z, target_param="a",
        target_value=1.0, target_std=0.1, converged=True, chi2_reduced_ci=(0.7, 1.3), fit_diagnostics=base)


# ═════════════════════════════════════════════════════════════════════════════
# A. Lignes de données
# ═════════════════════════════════════════════════════════════════════════════

def test_kk_verdict_states(nominal, stopped):
    assert rt.kk_verdict_state(nominal.probe_analysis) in (rt.KK_CONFORM, rt.KK_NONCONFORM)
    assert rt.kk_verdict_state(stopped.probe_analysis) == rt.KK_UNDETERMINED
    assert rt.kk_verdict_state(None) == rt.KK_UNDETERMINED


def test_kk_rows_report_the_voigt_elements_and_the_mean(nominal):
    rows = rt.kk_rows(nominal.probe_analysis)
    assert [r["kind"] for r in rows] == ["réplicat"] * 3 + ["moyenne"]
    assert all(r["n_voigt"] >= 1 and r["method"] == "measurement model" for r in rows)
    assert all(r["n_allowed"] >= 0 and r["n_outside"] >= 0 for r in rows)


def test_kk_rows_of_an_undetermined_group_are_indicative_lin_kk(stopped):
    rows = rt.kk_rows(stopped.probe_analysis)
    assert rows and all(r["conform"] is None and "Lin-KK" in r["method"] for r in rows)


def test_fit_diagnostic_rows_expose_every_requested_field(nominal):
    rows = rt.fit_diagnostic_rows(nominal.probe_analysis)
    assert [r["kind"] for r in rows] == ["réplicat"] * 3 + ["moyenne"]
    r = rows[-1]
    assert r["diagnostics_available"] and r["condition_number"] > 0 and r["rank"] is not None
    assert r["n_params"] == len(nominal.probe_analysis.orazem.mean_fit.params)
    for key in ("non_identifiable", "one_sided", "active_bounds", "chi2_ci_low", "chi2_ci_high",
                "chi2_in_ci", "target_value", "target_std", "n_starts", "n_converged"):
        assert key in r
    assert rt.fit_diagnostic_rows(None) == []


def test_kappa_above_the_orazem_threshold_is_an_error_severity():
    ok = rt.fit_diagnostic_row("s", "réplicat", _fit_with_diagnostics())
    assert ok["severity"] == rt.SEVERITY_OK and not ok["kappa_exceeds"]
    big = rt.fit_diagnostic_row("s", "réplicat",
                                _fit_with_diagnostics(condition_number=2 * CONDITION_NUMBER_WARN))
    assert big["kappa_exceeds"] and big["severity"] == rt.SEVERITY_ERROR
    inf = rt.fit_diagnostic_row("s", "réplicat", _fit_with_diagnostics(condition_number=float("inf")))
    assert inf["kappa_exceeds"] and inf["severity"] == rt.SEVERITY_ERROR


def test_the_threshold_is_the_one_of_orazem_fit_not_a_copy():
    import fits.orazem_fit as of
    assert CONDITION_NUMBER_WARN == of._COND_WARN
    assert 6e7 < CONDITION_NUMBER_WARN < 7e7                      # ≈ 6,7·10⁷


def test_non_identifiable_one_sided_and_active_bounds_are_reported():
    r = rt.fit_diagnostic_row("s", "réplicat", _fit_with_diagnostics(
        identifiable={"a": True, "b": False}, rank=1,
        jacobian_one_sided=["a"], active_bounds=[("b", "basse", 0.0)]))
    assert r["non_identifiable"] == ["b"] and r["one_sided"] == ["a"]
    assert r["active_bounds"] == [("b", "basse", 0.0)] and r["severity"] == rt.SEVERITY_ERROR
    w = rt.fit_diagnostic_row("s", "réplicat", _fit_with_diagnostics(jacobian_one_sided=["a"]))
    assert w["severity"] == rt.SEVERITY_WARNING


def test_a_fit_without_diagnostics_says_so_instead_of_inventing_values():
    fr = _fit_with_diagnostics()
    fr.fit_diagnostics = None
    r = rt.fit_diagnostic_row("s", "réplicat", fr)
    assert r["diagnostics_available"] is False and r["condition_number"] is None and r["rank"] is None


def test_chi2_outside_the_expected_interval_is_flagged():
    fr = _fit_with_diagnostics()
    fr.chi2_reduced = 2.5
    r = rt.fit_diagnostic_row("s", "réplicat", fr)
    assert r["chi2_in_ci"] is False and r["severity"] == rt.SEVERITY_WARNING


def test_fit_alert_lines_group_identical_messages_and_keep_group_alerts(nominal):
    lines = rt.fit_alert_lines(nominal.probe_analysis)
    msgs = [m for _s, m in lines]
    assert len(msgs) == len(set(msgs))                            # un message n'apparaît qu'une fois
    # Les butées (Re_prime, Cb) de ce jeu synthétique touchent tous les spectres.
    assert any(scope == "Tous les spectres" and "butée" in m for scope, m in lines)
    assert rt.fit_alert_lines(None) == []


def test_param_rows_mark_the_target_and_carry_both_uncertainties(nominal):
    og = nominal.probe_analysis.orazem
    rows = rt.param_rows(og)
    assert [r["param"] for r in rows] == og.param_names
    assert [r["param"] for r in rows if r["is_target"]] == [og.target_param]
    t = next(r for r in rows if r["is_target"])
    assert t["mean"] == og.target.mean and t["sem"] == og.target.sem
    assert t["std_within"] == og.target.std_within and t["std_between"] == og.target.std_between


def test_worst_severity():
    assert rt.worst_severity([]) is None
    assert rt.worst_severity([{"severity": "ok"}, {"severity": "error"}, {"severity": "warning"}]) == "error"


def test_drt_rows_state_for_ok_failed_and_absent(with_drt, nominal):
    _label, mean_sp, reps, an = next(iter(with_drt.iter_groups()))
    rows = rt.drt_diagnostic_rows(an, mean_sp, reps)
    by_label = {r["spectrum"]: r for r in rows}
    failed = by_label["Réplicat 2 (probe_r1.txt)"]
    assert failed["state"] == rt.DRT_FAILED and "compilation Stan" in failed["failure"]
    ok = by_label["Réplicat 1 (probe_r0.txt)"]
    assert ok["state"] == rt.DRT_OK and ok["mode"] == "sample"
    assert ok["rhat_ok"] and ok["divergences_ok"] and ok["ess_ok"] and ok["ess_threshold"] == 400.0
    assert ok["rct_ci_low"] < ok["rct"] < ok["rct_ci_high"] and ok["ebfmi_min"] == 0.8
    _l, m2, r2, an2 = next(iter(nominal.iter_groups()))
    assert {r["state"] for r in rt.drt_diagnostic_rows(an2, m2, r2)} == {rt.DRT_ABSENT}


def test_drt_rows_in_map_have_no_hmc_diagnostics():
    fr = _FakeDRT()(type("S", (), {"label": "x", "f": np.ones(40), "Zre": np.arange(40.0),
                                   "Zim": np.ones(40)})(), mode="optimize")
    r = rt.drt_diagnostic_row("x", "réplicat", fr)
    assert r["mode"] == "optimize" and r["rhat_max"] is None and r["rhat_ok"] is None
    assert r["divergences"] is None and r["ess_bulk_min"] is None and r["ess_ok"] is None


def test_hmc_thresholds_flag_a_bad_run():
    fr = _FakeDRT()(type("S", (), {"label": "x", "f": np.ones(40), "Zre": np.arange(40.0),
                                   "Zim": np.ones(40)})(), mode="sample")
    fr.drt_diagnostics["sampler"].update(rhat_max=1.2, divergences=3, ess_bulk_min=50.0)
    r = rt.drt_diagnostic_row("x", "réplicat", fr)
    assert r["rhat_ok"] is False and r["divergences_ok"] is False and r["ess_ok"] is False


def test_replicate_envelope_is_wider_than_none_and_needs_two_drts(with_drt):
    _l, mean_sp, reps, _an = next(iter(with_drt.iter_groups()))
    env = rt.drt_replicate_envelope(reps)               # 2 réplicats sur 3 portent une DRT
    assert env["n"] == 2 and np.all(env["hi"] >= env["lo"]) and np.any(env["hi"] > env["lo"])
    assert np.all(env["lo"] <= env["mean"]) and np.all(env["mean"] <= env["hi"])
    assert rt.drt_replicate_envelope(reps[:1]) is None
    assert rt.drt_replicate_envelope([]) is None


def test_replicate_envelope_interpolates_a_different_tau_grid(with_drt):
    _l, _m, reps, _an = next(iter(with_drt.iter_groups()))
    reps = copy.deepcopy(reps)
    fr = reps[2].fit_results[DRT_MODEL_NAME]
    fr.drt_tau = np.geomspace(1e-5, 1.0, 17)
    fr.drt_gamma = np.full(17, 0.5)
    env = rt.drt_replicate_envelope(reps)
    assert env is not None and env["n"] == 2
    assert env["tau"].min() >= 1e-5 and env["tau"].max() <= 1.0     # plage commune seulement


# ═════════════════════════════════════════════════════════════════════════════
# B. Figures
# ═════════════════════════════════════════════════════════════════════════════

def test_nyquist_and_bode_draw_replicates_and_mean_per_group(nominal):
    groups = nominal.display_groups()
    assert [g.label for g in groups] == ["Probe", "1.00e-09 M"]
    nyq = nyquist_replicates_figure(groups)
    assert len(nyq.data) == 2 * (3 + 1)                            # 3 réplicats + 1 moyenne / groupe
    assert {t.legendgroup for t in nyq.data} == {"Probe", "1.00e-09 M"}
    only_mean = nyquist_replicates_figure(groups, show_replicates=False)
    assert len(only_mean.data) == 2
    bode = bode_figure(groups)
    assert len(bode.data) == 2 * 2 * (3 + 1)                       # deux panneaux (|Z| et phase)
    assert bode.layout.xaxis.type == "log" and bode.layout.yaxis.type == "log"
    assert bode.layout.xaxis2.type == "log"


def test_bode_values_are_modulus_and_capacitive_phase(nominal):
    g = nominal.display_groups()[0]
    bode = bode_figure([g], show_replicates=False)
    mod, phase = bode.data[0], bode.data[1]
    assert np.allclose(mod.y, np.hypot(g.mean.Zre, g.mean.Zim))
    assert np.all(np.asarray(phase.y) > 0)                         # −φ > 0 : spectre capacitif


def test_visualisation_figures_accept_the_bare_reference_and_a_group_without_mean(nominal):
    g = nominal.display_groups()[0]
    no_mean = DisplayGroup(label=g.label, concentration=g.concentration, step=g.step,
                           mean=None, replicates=g.replicates)
    ref = nominal.display_groups()[1].mean
    for fig in (nyquist_replicates_figure([no_mean], bare=ref), bode_figure([no_mean], bare=ref)):
        assert any(t.name == "Électrode nue (réf.)" for t in fig.data)
        assert any(t.showlegend for t in fig.data if t.legendgroup == g.label)   # légende présente


def test_fit_figures_draw_experiment_and_fitted_curve(nominal):
    _l, mean_sp, reps, an = next(iter(nominal.iter_groups()))
    fr = mean_sp.fit_results[an.orazem.mean_fit.model_name]
    nyq = fit_nyquist_figure(mean_sp, fr)
    assert [t.name for t in nyq.data] == ["Expérience", "Circuit ajusté"]
    assert np.allclose(nyq.data[1].x, fr.Zfit_re)
    res = fit_residuals_figure(reps[0], reps[0].fit_results[an.orazem.mean_fit.model_name])
    assert len(res.data) == 2


def test_drt_aggregate_figure_has_the_wide_band_replicates_and_mean(with_drt):
    _l, mean_sp, reps, _an = next(iter(with_drt.iter_groups()))
    env = rt.drt_replicate_envelope(reps)
    items = [(f"Réplicat {i + 1}", sp.fit_results[DRT_MODEL_NAME]) for i, sp in enumerate(reps)
             if DRT_MODEL_NAME in sp.fit_results]
    fig = drt_aggregate_figure(env, items, mean_item=("m", mean_sp.fit_results[DRT_MODEL_NAME]))
    names = [t.name for t in fig.data]
    assert any(n and "Variabilité inter-réplicats" in n for n in names)
    assert "Moyenne des réplicats" in names and "DRT du spectre moyen" in names
    assert next(t for t in fig.data if t.name == "DRT du spectre moyen").visible == "legendonly"
    # La bande de crédibilité HMC (fine) vit sur la vue du réplicat, pas sur l'agrégat.
    single = drt_figure([items[0]])
    assert any("IC 95 %" in (t.name or "") for t in single.data)


# ═════════════════════════════════════════════════════════════════════════════
# D. Calibration (core/calibration.py, source unique de l'onglet ET de l'export CSV)
# ═════════════════════════════════════════════════════════════════════════════

def _cal_fit(model, rct, converged=True):
    z = np.ones(5)
    return FitResult(model_name=model, params={}, params_std={}, Zfit_re=z, Zfit_im=z, chi2_reduced=1.0,
                     residuals_re=z, residuals_im=z, target_param="Rct", target_value=rct,
                     target_std=1.0, converged=converged)


def _cal_session(*, orazem=None, drt=None, kk=None, drt_mode="optimize", probe_kk=None,
                 probe_converged=True):
    """probe + 4 concentrations. ``orazem``/``drt`` : {indice de groupe: dict(rct, converged)}
    pour surcharger ; ``drt`` None = pas de DRT ; ``kk`` : {indice: all_valid}."""
    from types import SimpleNamespace
    from core.models import ConcentrationGroup, EISSession, EISSpectrum

    def spectrum(conc, step, label):
        f = np.logspace(5, -1, 10)
        return EISSpectrum(label=label, f=f, Zre=f * 0 + 1.0, Zim=f * 0 + 1.0, concentration=conc,
                           step=step, n_points=10)

    def analysis(valid):
        return SimpleNamespace(ok=True, validation=SimpleNamespace(all_valid=valid), drt_failures={})

    s = EISSession()
    s.probe = spectrum(0.0, "probe", "probe (avg)")
    s.probe.fit_results["orazem"] = _cal_fit("orazem", 3000.0, probe_converged)
    s.probe_analysis = analysis(probe_kk)
    if drt is not None:
        s.drt_mode = drt_mode
        s.probe.fit_results["drt_bayes"] = _cal_fit("drt_bayes", 3000.0)
    for i, (conc, rct) in enumerate([(1e-12, 3300.0), (1e-11, 3900.0), (1e-10, 4800.0), (1e-9, 5700.0)]):
        sp = spectrum(conc, "hybridization", f"c{i} (avg)")
        o = {"rct": rct, "converged": True, **(orazem or {}).get(i, {})}
        sp.fit_results["orazem"] = _cal_fit("orazem", o["rct"], o["converged"])
        if drt is not None and drt.get(i) is not False:          # drt[i] = False : pas de DRT
            d = {"rct": rct * 1.02, "converged": True, **(drt.get(i) or {})}
            sp.fit_results["drt_bayes"] = _cal_fit("drt_bayes", d["rct"], d["converged"])
        grp = ConcentrationGroup(concentration=conc, spectrum=sp)
        grp.analysis = analysis((kk or {}).get(i))
        s.groups.append(grp)
    return s


def test_calibration_keeps_every_group_when_all_fits_are_sound():
    from core.calibration import calibration_points, compute_calibration, compute_calibration_loglog
    s = _cal_session()
    assert all(p.included and p.reasons == () for p in calibration_points(s, "orazem"))
    assert compute_calibration(s, "orazem").n == 4 and compute_calibration_loglog(s, "orazem").n == 4


def test_a_non_converged_fit_is_excluded_from_both_regressions_with_its_reason():
    from core.calibration import (REASON_NOT_CONVERGED, calibration_points, compute_calibration,
                                  compute_calibration_loglog)
    s = _cal_session(orazem={1: dict(converged=False)})
    pts = calibration_points(s, "orazem")
    assert [p.included for p in pts] == [True, False, True, True]
    assert pts[1].reasons == (REASON_NOT_CONVERGED,)
    for cal in (compute_calibration(s, "orazem"), compute_calibration_loglog(s, "orazem")):
        assert cal.n == 3 and 1e-11 not in cal.concentrations


def test_a_kk_non_conform_group_is_kept_but_flagged_by_default():
    from core.calibration import calibration_points, compute_calibration, compute_calibration_loglog
    s = _cal_session(kk={0: False, 2: None, 3: True})
    pts = calibration_points(s, "orazem")
    assert [p.included for p in pts] == [True, True, True, True]
    assert [p.kk_nonconform for p in pts] == [True, False, False, False]
    assert all(p.reasons == () for p in pts)
    for cal in (compute_calibration(s, "orazem"), compute_calibration_loglog(s, "orazem")):
        assert cal.n == 4 and list(cal.kk_flags) == [True, False, False, False]


def test_strict_regression_excludes_kk_non_conform_groups_but_not_undetermined_ones():
    from core.calibration import (REASON_KK_NONCONFORM, calibration_points, compute_calibration,
                                  compute_calibration_loglog)
    s = _cal_session(kk={0: False, 2: None, 3: True})
    pts = calibration_points(s, "orazem", strict=True)
    assert [p.included for p in pts] == [False, True, True, True]
    assert pts[0].reasons == (REASON_KK_NONCONFORM,)
    for cal in (compute_calibration(s, "orazem", strict=True),
                compute_calibration_loglog(s, "orazem", strict=True)):
        assert cal.n == 3 and 1e-12 not in cal.concentrations


def test_every_reason_is_reported_when_several_apply():
    from core.calibration import REASON_KK_NONCONFORM, REASON_NOT_CONVERGED, calibration_points
    s = _cal_session(orazem={0: dict(converged=False)}, kk={0: False})
    assert calibration_points(s, "orazem")[0].reasons == (REASON_NOT_CONVERGED,)    # KK ne motive plus
    assert calibration_points(s, "orazem", strict=True)[0].reasons == (REASON_NOT_CONVERGED,
                                                                       REASON_KK_NONCONFORM)


def test_every_group_kk_non_conform_probe_included_still_calibrates():
    """Cas réel : tous les verdicts KK « non conforme », probe compris."""
    from core.calibration import calibration_reference, compute_calibration, compute_calibration_loglog
    s = _cal_session(kk={0: False, 1: False, 2: False, 3: False}, probe_kk=False)
    ref = calibration_reference(s, "orazem")
    assert ref.included and ref.kk_nonconform
    assert compute_calibration(s, "orazem").n == 4 and compute_calibration_loglog(s, "orazem").n == 4


def test_a_missing_drt_is_reported_with_its_recorded_failure():
    from core.calibration import REASON_DRT_MISSING, calibration_points
    s = _cal_session(drt={2: False})                     # pas de DRT sur le groupe 2
    s.groups[2].analysis.drt_failures["c2 (avg)"] = "échec CmdStan simulé"
    pts = calibration_points(s, "drt_bayes")
    assert [p.included for p in pts] == [True, True, False, True]
    assert pts[2].reasons[0].startswith(REASON_DRT_MISSING) and "échec CmdStan simulé" in pts[2].reasons[0]


def test_a_requested_drt_with_no_result_excludes_every_point_instead_of_vanishing():
    from core.calibration import REASON_DRT_MISSING, calibration_models, calibration_points, \
        compute_calibration_loglog
    s = _cal_session()
    assert calibration_models(s) == ["orazem"]
    s.drt_mode = "optimize"                               # DRT demandée, mais aucun spectre n'en porte
    assert calibration_models(s) == ["orazem", "drt_bayes"]
    pts = calibration_points(s, "drt_bayes")
    assert not any(p.included for p in pts) and all(p.reasons == (REASON_DRT_MISSING,) for p in pts)
    assert compute_calibration_loglog(s, "drt_bayes") is None


def test_an_unusable_probe_blocks_the_normalised_signal_but_not_the_loglog():
    from core.calibration import calibration_reference, compute_calibration, compute_calibration_loglog
    s = _cal_session(probe_converged=False)
    assert not calibration_reference(s, "orazem").included
    assert compute_calibration(s, "orazem") is None
    assert compute_calibration_loglog(s, "orazem").n == 4          # la référence n'y sert pas


def test_the_probe_is_never_excluded_on_kk_alone_even_in_strict_mode():
    from core.calibration import calibration_reference, compute_calibration
    s = _cal_session(probe_kk=False)
    for strict in (False, True):
        assert calibration_reference(s, "orazem", strict).included
        assert compute_calibration(s, "orazem", strict).n == 4


def test_methods_are_filtered_independently():
    from core.calibration import compute_calibration_all
    s = _cal_session(orazem={1: dict(converged=False)}, drt={})
    by_model = {c.model: c.n for c in compute_calibration_all(s)}
    assert by_model == {"orazem": 3, "drt_bayes": 4}


def test_the_csv_export_applies_the_same_filter_as_the_tab():
    import csv
    from exports.exporter import export_calibration_csv
    s = _cal_session(orazem={1: dict(converged=False)}, kk={3: False})
    rows = list(csv.DictReader(io.StringIO(export_calibration_csv(s).decode())))
    assert sorted(float(r["concentration_M"]) for r in rows if r["model"] == "orazem") == [1e-12, 1e-10, 1e-9]
    assert [r["kk_non_conforme"] for r in rows if float(r["concentration_M"]) == 1e-9] == ["True"]
    strict = list(csv.DictReader(io.StringIO(export_calibration_csv(s, strict=True).decode())))
    assert sorted(float(r["concentration_M"]) for r in strict) == [1e-12, 1e-10]


def test_calibration_rows_list_included_and_excluded_points_with_reasons():
    s = _cal_session(orazem={1: dict(converged=False)}, kk={0: False}, drt={})
    rows = rt.calibration_rows(s)
    assert {r["model"] for r in rows} == {"orazem", "drt_bayes"}
    assert [r["kind"] for r in rows if r["model"] == "orazem"] == ["référence"] + ["concentration"] * 4
    out = {(r["model"], r["group"]): r["reasons"] for r in rows if not r["included"]}
    # Le verdict KK (par GROUPE) ne retire plus le point : il le signale pour les deux méthodes ;
    # seule la non-convergence, propre au fit Orazem, exclut.
    assert out == {("orazem", "1.00e-11 M"): ["fit non convergé"]}
    flagged = {(r["model"], r["group"]) for r in rows if r["included"] and r["kk_nonconform"]}
    assert flagged == {("orazem", "1.00e-12 M"), ("drt_bayes", "1.00e-12 M")}
    strict_out = {(r["model"], r["group"]) for r in rt.calibration_rows(s, strict=True) if not r["included"]}
    assert strict_out == {("orazem", "1.00e-12 M"), ("drt_bayes", "1.00e-12 M"), ("orazem", "1.00e-11 M")}


def test_the_calibration_tab_does_no_regression_of_its_own():
    source = (REPO / "ui" / "tabs.py").read_text(encoding="utf-8")
    assert "linregress" not in source and "polyfit" not in source
    assert "export_calibration_csv" in source and "calibration_rows" in source
    assert "calib_strict" in source and "retenu (KK non conforme)" in source


def test_loglog_figure_draws_one_regression_per_method_and_only_kept_points():
    from plotting.eis_plots import calibration_loglog_figure
    s = _cal_session(orazem={1: dict(converged=False)}, drt={})
    fig = calibration_loglog_figure(s)
    pts = {t.name: t for t in fig.data if t.mode == "markers" and "résidus" not in t.name}
    assert set(pts) == {"orazem", "drt_bayes"}
    assert len(pts["orazem"].x) == 3 and len(pts["drt_bayes"].x) == 4
    empty = calibration_loglog_figure(_cal_session(orazem={i: dict(converged=False) for i in range(4)}))
    assert not any(t.type == "scatter" for t in empty.data)


# ═════════════════════════════════════════════════════════════════════════════
# C. Rendu de la page EIS réelle (Streamlit AppTest)
# ═════════════════════════════════════════════════════════════════════════════

def _run_page(monkeypatch, n_rep, *, drt=None, raw=True):
    import streamlit
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(streamlit, "page_link", lambda *a, **k: None)
    if drt is not None:
        monkeypatch.setattr(drt_engine, "fit_drt", drt)
        monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    at = AppTest.from_file(str(REPO / "pages" / "A_eis.py"), default_timeout=300)
    at.session_state["preprocessing_done"] = True
    at.session_state["experiment_clean"] = _experiment_clean(n_rep)
    if raw:
        at.session_state["experiment"] = _experiment_clean(n_rep)
    at.session_state["eis_drt_enabled"] = drt is not None
    if drt is not None:
        at.session_state["eis_drt_mode"] = "sample"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _top_tabs(at):
    return [t.label for t in at.tabs if t.label.startswith(("1️⃣", "2️⃣", "3️⃣", "4️⃣"))]


def test_page_has_exactly_the_four_tabs_in_order(monkeypatch):
    at = _run_page(monkeypatch, 3)
    assert _top_tabs(at) == ["1️⃣ Visualisation", "2️⃣ Measurement model & fit Orazem", "3️⃣ DRT",
                             "4️⃣ Calibration"]
    assert len([t for t in at.tabs if t.label[:2] in ("1️", "2️", "3️", "4️", "5️")]) == 4


def test_page_offers_raw_data_only_when_the_raw_experiment_exists(monkeypatch):
    at = _run_page(monkeypatch, 3, raw=True)
    sessions = at.session_state["eis_sessions"]
    assert [g.label for g in sessions[1].raw_groups] == ["Probe", "1.00e-09 M"]
    assert all(len(g.replicates) == 3 for g in sessions[1].raw_groups)
    assert [r.label for r in at.radio if r.key == "eis_visu_source"][0] == "Données"
    at2 = _run_page(monkeypatch, 3, raw=False)
    radio = next(r for r in at2.radio if r.key == "eis_visu_source")
    assert len(radio.options) == 1


def test_model_fit_tab_shows_verdict_then_fit_with_visible_alerts(monkeypatch):
    at = _run_page(monkeypatch, 3)
    text = " ".join(m.value for m in at.markdown)
    assert text.index("1 · Measurement model") < text.index("2 · Fit Orazem")
    # Verdict clair (succès OU erreur, jamais silencieux) et avertissements de fit à la surface.
    assert any("Kramers-Kronig" in x.value for x in list(at.success) + list(at.error))
    assert any("butée" in w.value for w in at.warning)
    assert not any(e.label.startswith("Diagnostics") for e in at.expander)     # rien d'enterré
    labels = [m.label for m in at.metric]
    assert "Inter-réplicats (s)" in labels and "Intra-fit (√v̄)" in labels and "Fit du spectre moyen" in labels


def test_a_high_condition_number_raises_a_red_banner(monkeypatch):
    monkeypatch.setattr(rt, "CONDITION_NUMBER_WARN", 1.0)           # tout κ dépasse le seuil
    at = _run_page(monkeypatch, 3)
    banners = [e.value for e in at.error if "Incertitudes du fit non fiables" in e.value]
    assert banners and "conditionnement κ" in banners[0]


def test_an_undetermined_group_explains_itself_instead_of_an_error_or_an_empty_table(monkeypatch):
    at = _run_page(monkeypatch, 1)
    infos = [i.value for i in at.info]
    assert any("Verdict indéterminé" in i for i in infos)
    assert any("Aucun fit n'a été réalisé : verdict KK indéterminé" in i and "au moins 3" in i
               for i in infos)
    assert any("Groupe arrêté : aucune DRT" in i for i in infos)


def test_drt_tab_always_shows_hmc_diagnostics_and_the_ic_note(monkeypatch):
    at = _run_page(monkeypatch, 3, drt=_FakeDRT())
    assert any("l'IC de Rp n'est pas fiable" in i.value for i in at.info)
    # Vue agrégée (par défaut) : le tableau des diagnostics de TOUS les spectres est déjà là.
    df = next(d for d in at.dataframe if "R̂ max" in d.value.columns)
    assert {"Divergences", "ESS bulk min", "ESS tail min", "IC 95 % de Rct"} <= set(df.value.columns)
    assert len(df.value) == 4                                          # 3 réplicats + la moyenne
    # Vue d'un spectre : mêmes diagnostics en grand, et l'absence d'alerte est DITE.
    view = next(r for r in at.radio if r.key.startswith("drt_view_"))
    at = view.set_value("Réplicat 1").run()
    labels = [m.label for m in at.metric]
    for needed in ("R̂ max", "Divergences", "ESS bulk min", "ESS tail min", "Rct (Ω)"):
        assert needed in labels
    assert any("Aucune alerte" in s.value for s in at.success)


def test_drt_tab_names_a_spectrum_whose_computation_failed(monkeypatch):
    fake = _FakeDRT(fail={"probe_e1_r2.csv": RuntimeError("échec CmdStan simulé")})
    at = _run_page(monkeypatch, 3, drt=fake)
    errors = [e.value for e in at.error]
    assert any("probe_e1_r2.csv" in e and "échec CmdStan simulé" in e and "continué sans DRT" in e
               for e in errors)


def test_drt_alerts_are_displayed_not_buried(monkeypatch):
    at = _run_page(monkeypatch, 3, drt=_FakeDRT(alerts=["ESS tail faible (simulé)"]))
    # Vue d'un réplicat (la vue agrégée est la vue par défaut) : alertes immédiatement visibles.
    view = next(r for r in at.radio if r.key.startswith("drt_view_"))
    at = view.set_value("Réplicat 1").run()
    assert any("ESS tail faible (simulé)" in w.value for w in at.warning)


def test_calibration_tab_shows_both_methods_and_names_the_excluded_points(monkeypatch):
    # 1 concentration seulement : la régression est impossible, le tableau dit pourquoi et rien ne disparaît.
    at = _run_page(monkeypatch, 3, drt=_FakeDRT())
    assert any(w.value.startswith("**orazem** : pas de régression") for w in at.warning)
    table = next(d for d in at.dataframe if "Raison de l'exclusion" in d.value.columns)
    df = table.value
    assert set(df["Méthode"]) == {"orazem", "drt_bayes"}
    assert {"référence (normalisation)", "point de calibration"} == set(df["Rôle"])
    assert any(b.label.startswith("📥") for b in at.get("download_button"))
    assert len(df) == 4                                    # (probe + 1 concentration) × 2 méthodes


def test_calibration_tab_without_drt_says_so(monkeypatch):
    at = _run_page(monkeypatch, 3)
    assert any("Aucune DRT calculée pour cette électrode" in i.value for i in at.info)


# ─────────────────────────────────────────────────────────────────────────────
# Clés Streamlit uniques entre électrodes partageant les mêmes labels de groupe
# ─────────────────────────────────────────────────────────────────────────────

def _render_tabs_script():
    import streamlit as st
    from ui.tabs import render_eis_tabs
    render_eis_tabs(st.session_state["sessions"])


def test_two_electrodes_sharing_group_labels_raise_no_duplicate_key(monkeypatch, with_drt):
    """Régression : e1 et e2 ont chacune un groupe « Probe » (et la même concentration).
    Toute clé construite sur le seul label (ex. ``kk_res_probe``) lève
    StreamlitDuplicateElementKey. Rendu des 4 onglets avec DRT, KK, fit et visualisation."""
    import streamlit
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(streamlit, "page_link", lambda *a, **k: None)
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    sessions = {1: copy.deepcopy(with_drt), 2: copy.deepcopy(with_drt)}
    labels = [[g[0] for g in s.iter_groups()] for s in sessions.values()]
    assert labels[0] == labels[1] and "Probe" in labels[0]       # précondition : labels partagés

    at = AppTest.from_function(_render_tabs_script, default_timeout=300)
    at.session_state["sessions"] = sessions
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    keys = [c.key for c in at.get("plotly_chart")]
    assert any(k and k.startswith("kk_res_") for k in keys)
    assert len(keys) == len(set(keys))
