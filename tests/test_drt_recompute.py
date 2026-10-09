"""core/drt_recompute.py — recalcul DRT d'un spectre, de plusieurs, retour à ``optimize``.

Organisation
    A. Chronométrage : mode, durée et date rangés dans ``drt_diagnostics`` (moteur factice)
    B. Recalcul d'UN spectre : n'écrase que lui, un échec est rendu (jamais levé) et n'écrit rien
    C. Registre à clés stables et retour à ``optimize``
    D. Recalcul de plusieurs spectres : progression n/N, un échec n'arrête pas la suite
    E. Exports : durée et mode dans le YAML de session et le CSV par spectre
    F. Panneau de l'onglet DRT (Streamlit AppTest, moteur factice)
    G. Moteur DRT RÉEL (sautés sans CmdStan ; le HMC est marqué ``slow``)

Les sections A à F n'exigent ni CmdStan ni cvxopt : ``drt.engine.fit_drt`` est remplacé.
"""

import copy
import csv
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import core.pipeline as pipeline
import drt.engine as drt_engine
from core import drt_recompute as dr
from core.app_state import ANALYSIS_RESULT_KEYS
from core.pipeline import DRT_MODEL_NAME, InvalidAnalysisInput, run_pipeline
from exports.exporter import export_replicate_results_csv, export_session_yaml
from tests.synthetic_data import make_config, replicate_assignments
from tests.test_pipeline import _FakeDRT

REPO = Path(__file__).resolve().parents[1]
_RCT_PROBE, _RCT_C1, _C1 = 3000.0, 3500.0, 1e-9

_HAVE_CMDSTAN, _WHY_NOT = drt_engine.engine_available()
_NEEDS_CMDSTAN = pytest.mark.skipif(
    not _HAVE_CMDSTAN, reason=f"DRT réelle : {_WHY_NOT} (job CI « drt »)")


def _assignments(n_rep=3):
    return (replicate_assignments("probe", 0.0, _RCT_PROBE, n_rep, 0, "probe")
            + replicate_assignments("hybridization", _C1, _RCT_C1, n_rep, 10, "c1"))


@pytest.fixture(scope="module")
def _base_run():
    """probe + 1 concentration, 3 réplicats, DRT factice calculée en MAP par le pipeline."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(drt_engine, "fit_drt", _FakeDRT())
        mp.setattr(drt_engine, "engine_available", lambda: (True, None))
        cfg = make_config(enabled=True, mode="optimize")
        session, _ = run_pipeline(_assignments(), cfg)
    return SimpleNamespace(session=session, cfg=cfg)


@pytest.fixture
def run(_base_run, monkeypatch):
    """Copie profonde du run + moteur factice actif ; ``run.fake`` enregistre les appels."""
    fake = _FakeDRT()
    monkeypatch.setattr(drt_engine, "fit_drt", fake)
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    c = copy.deepcopy(_base_run)
    return SimpleNamespace(session=c.session, cfg=c.cfg, fake=fake, store={})


def _target(run, session_label):
    return next(t for t in dr.iter_targets({1: run.session}) if t.session_label == session_label)


def _active(target):
    return target.spectrum.fit_results[DRT_MODEL_NAME]


def _fake_clock(monkeypatch, step=2.5):
    """Horloge déterministe pour ``core.pipeline`` : chaque lecture avance de ``step`` secondes."""
    ticks = iter(range(10_000))
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=lambda: next(ticks) * step))


# ═════════════════════════════════════════════════════════════════════════════
# A. Chronométrage
# ═════════════════════════════════════════════════════════════════════════════

def test_the_initial_analysis_stamps_mode_duration_and_date_on_every_drt(run):
    targets = dr.iter_targets({1: run.session})
    assert len(targets) == 8                                   # (probe + 3 réplicats) × 2 groupes
    for t in targets:
        diag = _active(t).drt_diagnostics
        assert diag["mode"] == "optimize"
        assert isinstance(diag["duration_s"], float) and diag["duration_s"] >= 0
        assert diag["computed_at"][:2] == "20"                 # ISO 8601
        assert dr.run_info(_active(t)).mode == "optimize"


def test_the_duration_is_the_wall_time_of_fit_drt_and_keeps_the_engine_diagnostics(run, monkeypatch):
    _fake_clock(monkeypatch, step=2.5)
    t = _target(run, "probe#1")
    out = dr.recalculate(t, run.cfg, "sample", run.store)
    info = dr.run_info(_active(t))
    assert out.duration_s == info.duration_s == 2.5
    assert info.mode == "sample"
    assert "sampler" in _active(t).drt_diagnostics             # le dict du moteur n'est pas remplacé


def test_a_result_without_stamp_has_unknown_duration_not_zero():
    assert dr.run_info(None) is None
    fr = _FakeDRT()(SimpleNamespace(label="x", f=[1.0] * 3, Zre=[1.0, 2.0, 3.0], Zim=[1.0] * 3), mode="optimize")
    info = dr.run_info(fr)
    assert info.mode == "optimize" and info.duration_s is None and info.computed_at is None


# ═════════════════════════════════════════════════════════════════════════════
# B. Recalcul d'UN spectre
# ═════════════════════════════════════════════════════════════════════════════

def test_recalculating_one_spectrum_overwrites_only_that_one(run):
    before = {t.key: _active(t) for t in dr.iter_targets({1: run.session})}
    t = _target(run, "1.00e-09#2")

    out = dr.recalculate(t, run.cfg, "sample", run.store)

    assert out.ok and out.mode == "sample" and out.error is None
    assert run.fake.calls == [(t.spectrum.label, "sample")]                 # un seul calcul
    assert _active(t) is not before[t.key] and _active(t).drt_mode == "sample"
    for other in dr.iter_targets({1: run.session}):
        if other.key != t.key:
            assert _active(other) is before[other.key]                      # intacts, par identité


def test_a_failure_is_returned_with_the_label_and_leaves_the_old_result_in_place(run):
    t = _target(run, "probe#0")
    run.fake.fail[t.spectrum.label] = RuntimeError("compilation Stan échouée (simulée)")
    previous = _active(t)

    out = dr.recalculate(t, run.cfg, "sample", run.store)

    assert not out.ok and "compilation Stan échouée" in out.error
    assert t.spectrum.label in out.display and "probe#0" in out.display
    assert _active(t) is previous and run.store == {}


def test_an_invalid_spectrum_is_a_failure_outcome_too(run):
    t = _target(run, "probe#0")
    run.fake.fail[t.spectrum.label] = ValueError("Spectre trop court pour une DRT")
    assert not dr.recalculate(t, run.cfg, "optimize", run.store).ok


def test_an_unknown_mode_is_a_caller_bug_and_raises(run):
    with pytest.raises(InvalidAnalysisInput, match="Réglages DRT invalides"):
        dr.recalculate(_target(run, "probe"), run.cfg, "nuts", run.store)
    assert run.fake.calls == []


def test_target_of_resolves_by_identity_and_refuses_a_stranger(run):
    sp = run.session.probe_replicate_spectra[1]
    t = dr.target_of(1, run.session, sp)
    assert t.session_label == "probe#1" and t.key == dr.drt_store_key(1, "probe#1") == "e1|probe#1"
    stranger = copy.deepcopy(sp)
    with pytest.raises(ValueError, match="introuvable"):
        dr.target_of(1, run.session, stranger)


# ═════════════════════════════════════════════════════════════════════════════
# C. Registre et retour à optimize
# ═════════════════════════════════════════════════════════════════════════════

def test_sample_keeps_the_optimize_result_in_the_store_under_a_stable_key(run):
    t = _target(run, "probe#0")
    optimize = _active(t)

    dr.recalculate(t, run.cfg, "sample", run.store)

    assert list(run.store) == ["e1|probe#0"]
    assert run.store["e1|probe#0"]["optimize"] is optimize
    assert run.store["e1|probe#0"]["sample"] is _active(t)


def test_a_new_sample_click_is_the_only_thing_that_overwrites_the_stored_sample(run):
    t = _target(run, "probe#0")
    dr.recalculate(t, run.cfg, "sample", run.store)
    first = _active(t)
    dr.revert_to_optimize(t, run.cfg, run.store)
    assert run.store[t.key]["sample"] is first                  # le retour ne l'écrase pas
    dr.recalculate(t, run.cfg, "sample", run.store)
    assert run.store[t.key]["sample"] is _active(t) and _active(t) is not first


def test_reverting_reuses_the_stored_optimize_without_computing_again(run):
    t = _target(run, "probe#0")
    optimize = _active(t)
    dr.recalculate(t, run.cfg, "sample", run.store)
    calls = len(run.fake.calls)
    an = run.session.probe_analysis
    assert any(w.startswith("Rct DRT agrégé sur des") for w in an.warnings)   # modes mêlés

    out = dr.revert_to_optimize(t, run.cfg, run.store)

    assert out.ok and out.reused and out.mode == "optimize"
    assert _active(t) is optimize and len(run.fake.calls) == calls            # aucun calcul
    assert not any(w.startswith("Rct DRT agrégé sur des") for w in an.warnings)  # agrégat rafraîchi


def test_reverting_without_a_stored_optimize_recomputes_it_and_keeps_the_sample(run):
    t = _target(run, "1.00e-09")
    dr.recalculate(t, run.cfg, "sample", None)                  # pas de registre : rien de gardé
    sample = _active(t)

    out = dr.revert_to_optimize(t, run.cfg, run.store)

    assert out.ok and not out.reused and run.fake.calls[-1] == (t.spectrum.label, "optimize")
    assert _active(t).drt_mode == "optimize"
    assert run.store[t.key]["sample"] is sample                 # le résultat quitté est conservé


def test_reverting_a_spectrum_already_in_optimize_does_nothing(run):
    t = _target(run, "probe")
    out = dr.revert_to_optimize(t, run.cfg, run.store)
    assert out.ok and out.reused and run.fake.calls == [] and run.store == {}


def test_the_store_is_reset_with_the_other_analysis_results():
    assert dr.STORE_KEY in ANALYSIS_RESULT_KEYS


# ═════════════════════════════════════════════════════════════════════════════
# D. Plusieurs spectres
# ═════════════════════════════════════════════════════════════════════════════

def test_recalculating_many_reports_n_over_N_and_continues_after_a_failure(run):
    targets = dr.iter_targets({1: run.session})
    bad = _target(run, "probe#1")
    run.fake.fail[bad.spectrum.label] = RuntimeError("HMC divergé (simulé)")
    seen = []

    outcomes = dr.recalculate_many(targets, run.cfg, "sample", run.store,
                                   on_progress=lambda i, n, lbl: seen.append((i, n, lbl)))

    assert [o.key for o in outcomes] == [t.key for t in targets]
    assert [o.key for o in outcomes if not o.ok] == [bad.key]
    assert sum(o.ok for o in outcomes) == 7
    assert [(i, n) for i, n, _l in seen] == [(i, 8) for i in range(8)] + [(8, 8)]
    assert seen[0][2] == targets[0].display and seen[-1][2] == ""
    assert _active(bad).drt_mode == "optimize"                  # l'échec laisse l'ancien résultat
    assert all(_active(t).drt_mode == "sample" for t in targets if t.key != bad.key)


def test_reverting_many_returns_everything_to_optimize(run):
    targets = dr.iter_targets({1: run.session})
    dr.recalculate_many(targets, run.cfg, "sample", run.store)
    calls = len(run.fake.calls)

    outcomes = dr.revert_many(targets, run.cfg, run.store)

    assert all(o.ok and o.reused for o in outcomes) and len(run.fake.calls) == calls
    assert all(_active(t).drt_mode == "optimize" for t in targets)


def test_stopped_groups_are_not_offered_for_recalculation(monkeypatch):
    monkeypatch.setattr(drt_engine, "fit_drt", _FakeDRT())
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    session, _ = run_pipeline(_assignments(n_rep=1), make_config(enabled=True, mode="optimize"))
    assert session.failed_groups()                              # un seul fichier : KK indéterminé
    assert dr.iter_targets({1: session}) == []


def test_estimate_follows_the_documented_two_to_five_minutes():
    assert dr.estimate_sample_minutes(1) == (2, 5) and dr.estimate_sample_minutes(8) == (16, 40)


# ═════════════════════════════════════════════════════════════════════════════
# E. Exports
# ═════════════════════════════════════════════════════════════════════════════

def test_the_session_yaml_carries_mode_and_duration_of_each_drt(run, monkeypatch):
    _fake_clock(monkeypatch, step=4.0)
    dr.recalculate(_target(run, "1.00e-09"), run.cfg, "sample", run.store)

    data = yaml.safe_load(export_session_yaml(run.session))

    probe, c1 = data["groups"]
    assert probe["fits"][DRT_MODEL_NAME]["drt_mode"] == "optimize"
    assert set(probe["fits"][DRT_MODEL_NAME]["drt_run"]) == {"mode", "duration_s", "computed_at"}
    assert c1["fits"][DRT_MODEL_NAME]["drt_mode"] == "sample"
    assert c1["fits"][DRT_MODEL_NAME]["drt_run"]["duration_s"] == 4.0
    assert "drt_run" not in probe["fits"]["orazem"]             # le fit Orazem n'est pas touché


def test_the_per_spectrum_csv_has_duration_and_date_columns(run):
    dr.recalculate(_target(run, "probe#0"), run.cfg, "sample", run.store)

    rows = list(csv.DictReader(io.StringIO(export_replicate_results_csv({1: run.session}).decode())))

    assert {"drt_mode", "drt_duration_s", "drt_computed_at"} <= set(rows[0])
    sample = [r for r in rows if r["drt_mode"] == "sample"]
    assert len(sample) == 1 and float(sample[0]["drt_duration_s"]) >= 0 and sample[0]["drt_computed_at"]


# ═════════════════════════════════════════════════════════════════════════════
# F. Onglet DRT (Streamlit AppTest)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def page(monkeypatch):
    """Vraie page EIS lancée avec la DRT factice en OPTIMIZE (le défaut de l'application)."""
    import streamlit
    from streamlit.testing.v1 import AppTest

    from tests.test_eis_tabs import _experiment_clean, _FakeDRT as PageFakeDRT

    fake = PageFakeDRT()
    monkeypatch.setattr(streamlit, "page_link", lambda *a, **k: None)
    monkeypatch.setattr(drt_engine, "fit_drt", fake)
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    at = AppTest.from_file(str(REPO / "pages" / "A_eis.py"), default_timeout=300)
    at.session_state["preprocessing_done"] = True
    at.session_state["experiment_clean"] = _experiment_clean(3)
    at.session_state["eis_drt_enabled"] = True
    at.session_state["eis_drt_mode"] = "optimize"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return SimpleNamespace(at=at, fake=fake)


def test_the_tab_shows_one_row_per_spectrum_with_mode_duration_and_buttons(page):
    at = page.at
    keys = {b.key for b in at.button}
    for i in range(3):
        assert f"drt_row_sample_e1|probe#{i}" in keys
    assert {"drt_row_sample_e1|probe", "drt_all_sample", "drt_all_optimize"} <= keys
    warnings = " ".join(w.value for w in at.warning)
    assert "2 à 5 minutes" in warnings and "instable" in warnings
    assert "pour 8 spectres" in warnings                        # le global annonce le total
    assert at.session_state["eis_drt_store"] == {}              # créé à l'analyse, vide


def test_clicking_a_row_recalculates_that_spectrum_in_sample_and_the_result_persists(page):
    at = page.at
    n_before = page.fake.n
    at = at.button(key="drt_row_sample_e1|probe#0").click().run()
    assert not at.exception, [e.value for e in at.exception]

    sessions = at.session_state["eis_sessions"]
    assert sessions[1].probe_replicate_spectra[0].fit_results[DRT_MODEL_NAME].drt_mode == "sample"
    assert sessions[1].probe_replicate_spectra[1].fit_results[DRT_MODEL_NAME].drt_mode == "optimize"
    assert page.fake.n == n_before + 1
    assert "e1|probe#0" in at.session_state["eis_drt_store"]
    assert any("DRT sample recalculée" in s.value for s in at.success)

    # Un rerun quelconque ne touche à rien : le résultat n'est écrasé que par un nouveau clic.
    at = at.run()
    stored = at.session_state["eis_drt_store"]["e1|probe#0"]["sample"]
    assert at.session_state["eis_sessions"][1].probe_replicate_spectra[0].fit_results[DRT_MODEL_NAME] is stored


def test_a_failing_spectrum_gives_a_warning_with_its_label_and_the_others_still_run(page):
    at = page.at
    label = at.session_state["eis_sessions"][1].probe_replicate_spectra[1].label
    page.fake.fail[label] = RuntimeError("Stan a échoué (simulé)")

    at = at.button(key="drt_all_sample").click().run()
    assert not at.exception, [e.value for e in at.exception]

    texts = [w.value for w in at.warning]
    assert any(label in w and "Stan a échoué" in w and "conservé" in w for w in texts)
    sessions = at.session_state["eis_sessions"]
    modes = [s.fit_results[DRT_MODEL_NAME].drt_mode
             for _lbl, mean_sp, reps, _an in sessions[1].iter_groups() for s in [*reps, mean_sp]]
    assert modes.count("sample") == 7 and modes.count("optimize") == 1


def test_revert_to_optimize_from_the_tab_restores_the_map_result(page):
    at = page.at
    optimize = at.session_state["eis_sessions"][1].probe_replicate_spectra[0].fit_results[DRT_MODEL_NAME]
    at = at.button(key="drt_row_sample_e1|probe#0").click().run()
    calls = page.fake.n
    at = at.button(key="drt_row_optimize_e1|probe#0").click().run()
    assert not at.exception
    restored = at.session_state["eis_sessions"][1].probe_replicate_spectra[0].fit_results[DRT_MODEL_NAME]
    assert restored is optimize and page.fake.n == calls


def test_a_new_analysis_empties_the_store(page):
    at = page.at
    at = at.button(key="drt_row_sample_e1|probe#0").click().run()
    assert at.session_state["eis_drt_store"]
    at = at.button(key="eis_rerun_btn").click().run()
    assert at.session_state["eis_drt_store"] == {}
    assert at.session_state["eis_sessions"][1].probe_replicate_spectra[0] \
        .fit_results[DRT_MODEL_NAME].drt_mode == "optimize"    # relancée avec le mode de la page


# ═════════════════════════════════════════════════════════════════════════════
# G. Moteur DRT réel
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_session():
    """probe seul, 3 réplicats, SANS DRT : on la calcule ensuite via le recalcul."""
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(enabled=False))
    return session


@_NEEDS_CMDSTAN
def test_real_optimize_recalculation_stamps_duration_and_mode(real_session):
    session = copy.deepcopy(real_session)
    t = dr.target_of(1, session, session.probe_replicate_spectra[0])
    store = {}

    out = dr.recalculate(t, make_config(enabled=True, mode="optimize"), "optimize", store)

    assert out.ok, out.error
    info = dr.run_info(_active(t))
    assert info.mode == "optimize" and info.duration_s > 0 and info.computed_at
    assert _active(t).converged and store[t.key]["optimize"] is _active(t)


@pytest.mark.slow
@_NEEDS_CMDSTAN
def test_real_sample_recalculation_then_revert_to_optimize(real_session):
    session = copy.deepcopy(real_session)
    t = dr.target_of(1, session, session.probe_replicate_spectra[0])
    cfg, store = make_config(enabled=True, mode="optimize"), {}
    assert dr.recalculate(t, cfg, "optimize", store).ok
    optimize = _active(t)

    out = dr.recalculate(t, cfg, "sample", store)

    assert out.ok, out.error
    info = dr.run_info(_active(t))
    assert info.mode == "sample" and info.duration_s > 1.0
    assert _active(t).drt_diagnostics["sampler"]["rhat_max"] is not None
    assert dr.revert_to_optimize(t, cfg, store).reused and _active(t) is optimize
    assert store[t.key]["sample"].drt_mode == "sample"
