"""Calcul en arrière-plan (core/background.py, core/analysis.py) — phase 4."""

import math
import threading

import numpy as np
import pytest

import core.analysis as analysis_mod
from core.analysis import AnalysisRun, DRTRecomputeRun, build_group_payloads, eis_group_task
from core.background import BackgroundRun, default_workers
from core.experiment import Channel, Experiment, Measurement
from core.report_tables import drt_rows, failure_rows, group_rows, rct_rows
from core.results import GROUP_FALLBACK, GROUP_OK, KK_UNDETERMINED, WEIGHTING_PROPORTIONAL
from tests.synthetic_data import make_config, orazem_noisy_arrays


def _meas(ch, el, step, rep, rct, seed):
    f, zr, zi, _ = orazem_noisy_arrays(rct, seed)
    return Measurement(ch, el, step, rep, "EIS", f"{step}_{rep}.txt", {"f": f, "Zre": zr, "Zim": zi})


@pytest.fixture
def setup():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    ms = ([_meas(1, 1, "probe", r, 3000, r) for r in (1, 2, 3)]
          + [_meas(1, 1, "conc_1", r, 4000, 10 + r) for r in (1, 2)])
    return exp, ms


# ── BackgroundRun ────────────────────────────────────────────────────────────

def _boom(x):
    if x == 2:
        raise ValueError("donnée invalide")
    return math.sqrt(x)


@pytest.mark.parametrize("kind", ["sync", "thread", "process"])
def test_one_failing_task_does_not_stop_the_others(kind):
    run = BackgroundRun(kind, max_workers=2)
    ok, err = {}, {}
    for x in (1, 2, 4, 9):
        run.submit(str(x), _boom, x, on_result=lambda r, x=x: ok.__setitem__(x, r),
                   on_error=lambda m, x=x: err.__setitem__(x, m))
    run.total = 4
    assert run.wait(timeout=120)
    assert ok == {1: 1.0, 4: 2.0, 9: 3.0}
    assert list(err) == [2] and "donnée invalide" in err[2]
    assert run.done == 4


def test_cancel_drops_pending_tasks():
    gate = threading.Event()
    run = BackgroundRun("thread", max_workers=1)
    got = []
    for x in range(5):
        run.submit(str(x), lambda _x: gate.wait(5), x, on_result=got.append, on_error=got.append)
    run.cancel()
    gate.set()
    assert run.finished and run.cancelled and run.poll() and got == []


def test_default_workers():
    assert default_workers(3) == 3
    assert default_workers(None) >= 1


# ── Analyse : fit Orazem, repli, résultats publiés seulement à la fin ───────

def test_groups_fit_with_orazem_or_fallback(setup):
    exp, ms = setup
    run = AnalysisRun(exp, ms, make_config(enabled=False), executor_kind="sync")
    assert run.wait(timeout=300)
    res = run.results
    probe, conc = res.eis.groups["C1_E1_probe"], res.eis.groups["C1_E1_conc_1"]
    assert probe.status == GROUP_OK and probe.weighting == "orazem"
    assert probe.aggregate["Rct"]["mean"] == pytest.approx(3000, rel=0.02) and probe.aggregate["Rct"]["n"] == 3
    # 2 réplicats : fit QUAND MÊME calculé, en repli signalé, verdict KK indéterminé.
    assert conc.status == GROUP_FALLBACK and conc.weighting == WEIGHTING_PROPORTIONAL
    assert conc.kk_state == KK_UNDETERMINED and "REPLI" in conc.message
    assert conc.aggregate["Rct"]["mean"] == pytest.approx(4000, rel=0.02)
    keys = set(res.eis.spectra)
    assert {"EIS_C1_E1_probe_r1", "EIS_C1_E1_probe_mean", "EIS_C1_E1_conc_1_r2", "EIS_C1_E1_conc_1_mean"} <= keys
    for sp in res.eis.spectra.values():
        assert sp.fit is not None and sp.fit.converged and len(sp.fit.Zfit_re) == len(sp.f)
    assert res.failures == []
    rows = rct_rows(res)
    assert len(rows) == 7 and {"Rct", "Re", "χ²ᵣ", "Rct_DRT (pic principal, Ω)"} <= set(rows[0])
    assert len(group_rows(res)) == 2 and len(drt_rows(res)) == 7


def test_nothing_is_published_before_every_task_is_done(setup, monkeypatch):
    exp, ms = setup
    gate = threading.Event()
    real = analysis_mod.eis_group_task

    def slow(payload):
        if payload["step"] == "conc_1":
            gate.wait(30)
        return real(payload)

    monkeypatch.setattr(analysis_mod, "eis_group_task", slow)
    run = AnalysisRun(exp, ms, make_config(enabled=False), executor_kind="thread", max_workers=2)
    for _ in range(200):                       # le groupe probe finit, conc_1 reste bloqué
        run.poll()
        if run.progress[0] >= 1:
            break
        threading.Event().wait(0.05)
    assert run.progress[0] == 1 and run.results is None and not run.finished
    gate.set()
    assert run.wait(timeout=300) and run.results is not None


def test_an_isolated_failure_is_listed_and_the_batch_completes(setup, monkeypatch):
    exp, ms = setup
    real = analysis_mod.eis_group_task

    def flaky(payload):
        if payload["step"] == "probe":
            raise RuntimeError("worker mort")
        return real(payload)

    monkeypatch.setattr(analysis_mod, "eis_group_task", flaky)
    run = AnalysisRun(exp, ms, make_config(enabled=False), executor_kind="sync")
    assert run.wait(timeout=300)
    res = run.results
    assert "C1_E1_conc_1" in res.eis.groups and "C1_E1_probe" not in res.eis.groups
    rows = failure_rows(res)
    assert rows == [{"Élément": "C1_E1_probe", "Étape du calcul": "fit du circuit",
                     "Cause": "RuntimeError: worker mort"}]


def test_payloads_group_by_channel_electrode_step(setup):
    exp, ms = setup
    ms = ms + [_meas(1, 2, "probe", 1, 3000, 99)]
    p = build_group_payloads(exp, ms, make_config(), None)
    assert [x["gkey"] for x in p] == ["C1_E1_probe", "C1_E1_conc_1", "C1_E2_probe"]
    assert p[1]["concentration"] == 1e-9 and [s["replicate"] for s in p[0]["spectra"]] == [1, 2, 3]


def test_group_task_is_picklable_end_to_end(setup):
    import pickle
    exp, ms = setup
    payload = build_group_payloads(exp, ms, make_config(), None)[1]
    out = eis_group_task(pickle.loads(pickle.dumps(payload)))
    assert pickle.loads(pickle.dumps(out))["group"].key == "C1_E1_conc_1"


# ── DRT réelle (sautée sans CmdStan) ─────────────────────────────────────────

def _engine():
    import drt.engine as eng
    ok, why = eng.engine_available()
    if not ok:
        pytest.skip(f"moteur DRT indisponible : {why}")


def test_full_run_in_processes_with_drt_and_recompute(setup):
    _engine()
    exp, ms = setup
    run = AnalysisRun(exp, ms, make_config(enabled=True, mode="optimize"), executor_kind="process", max_workers=2)
    assert run.progress[1] == 2 + 5 + 2                 # 2 groupes + 5 réplicats + 2 moyennes
    assert run.wait(timeout=900)
    res = run.results
    assert res.eis.drt_mode == "optimize"
    for sp in res.eis.spectra.values():
        assert sp.drt is not None and sp.drt.peak.ok, (sp.key, sp.drt_error)
    probe_mean = res.eis.spectra["EIS_C1_E1_probe_mean"]
    assert probe_mean.drt.rct_main == pytest.approx(3000, rel=0.08)
    # Recalcul d'UN spectre : seul lui change.
    before = {k: s.drt for k, s in res.eis.spectra.items()}
    rr = DRTRecomputeRun(res, ["EIS_C1_E1_probe_r1"], make_config(), "optimize", executor_kind="sync")
    assert rr.wait(timeout=300) and rr.errors() == {}
    assert res.eis.spectra["EIS_C1_E1_probe_r1"].drt is not before["EIS_C1_E1_probe_r1"]
    assert all(res.eis.spectra[k].drt is before[k] for k in before if k != "EIS_C1_E1_probe_r1")
    assert np.allclose(res.eis.spectra["EIS_C1_E1_probe_r1"].drt.gamma, before["EIS_C1_E1_probe_r1"].gamma)
