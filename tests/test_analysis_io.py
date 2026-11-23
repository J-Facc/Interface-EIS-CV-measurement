"""core/analysis_io.py et calcul DRT différé (core.drt_recompute.compute_pending_drt).

Moteur DRT factice : aucun CmdStan requis.
"""

import io
import pickle
import zipfile

import numpy as np
import pytest

import drt.engine as drt_engine
from core import analysis_io as aio
from core import drt_recompute as dr
from core.pipeline import DRT_MODEL_NAME, run_pipeline
from tests.synthetic_data import make_config, replicate_assignments
from tests.test_pipeline import _FakeDRT


@pytest.fixture
def state(monkeypatch):
    """Analyse SANS DRT (comme la page EIS : DRT différée), dans un faux session_state."""
    fake = _FakeDRT()
    monkeypatch.setattr(drt_engine, "fit_drt", fake)
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    cfg = make_config(enabled=False, mode="optimize")
    fas = (replicate_assignments("probe", 0.0, 3000.0, 3, 0, "probe")
           + replicate_assignments("hybridization", 1e-9, 3500.0, 3, 10, "c1"))
    session, val = run_pipeline(fas, cfg, run_drt=False)
    return {"eis_sessions": {1: session}, "eis_validations": {1: val}, "eis_config": cfg,
            "eis_drt_store": {}, "eis_normalized": {}, "eis_inputs": {"run_drt": True},
            "eis_failures": {}, "fake": fake}


def test_drt_is_deferred_then_computed_for_every_spectrum(state):
    sessions = state["eis_sessions"]
    assert aio.count_drt(sessions) == (0, 8)          # probe + 1 conc : (3 réplicats + moyenne) × 2
    assert state["fake"].calls == []                 # run_drt=False : aucun calcul DRT au pipeline
    out = dr.compute_pending_drt(sessions, state["eis_config"], "optimize", state["eis_drt_store"])
    assert len(out) == 8 and all(o.ok for o in out)
    assert aio.count_drt(sessions) == (8, 8)
    assert sessions[1].drt_mode == "optimize"
    assert all(an.drt_target is not None for *_x, an in sessions[1].iter_groups())


def test_deferred_drt_is_resumable_and_records_failures(state):
    sessions = state["eis_sessions"]
    first = sessions[1].probe_replicate_spectra[0]
    state["fake"].fail = {first.label: ValueError("spectre invalide")}
    out = dr.compute_pending_drt(sessions, state["eis_config"], "optimize")
    assert sum(not o.ok for o in out) == 1
    assert first.label in sessions[1].probe_analysis.drt_failures
    n_calls = len(state["fake"].calls)
    assert dr.pending_targets(sessions) == []        # tout est traité (réussi ou en échec)
    assert dr.compute_pending_drt(sessions, state["eis_config"], "optimize") == []
    assert len(state["fake"].calls) == n_calls       # rien n'est recalculé


def test_interrupted_computation_resumes_where_it_stopped(state):
    sessions = state["eis_sessions"]
    some = dr.pending_targets(sessions)[:3]
    dr.recalculate_many(some, state["eis_config"], "optimize")
    calls = len(state["fake"].calls)
    out = dr.compute_pending_drt(sessions, state["eis_config"], "optimize")
    assert len(out) == 5 and len(state["fake"].calls) == calls + 5


def test_save_and_load_roundtrip_keeps_drt_without_recompute(state):
    sessions = state["eis_sessions"]
    dr.compute_pending_drt(sessions, state["eis_config"], "optimize", state["eis_drt_store"])
    data = aio.save_analysis(state)
    calls = len(state["fake"].calls)
    loaded = aio.load_analysis(data)
    assert len(state["fake"].calls) == calls                    # rechargement = zéro calcul
    assert loaded["manifest"]["drt_spectra_computed"] == 8
    assert loaded["manifest"]["electrodes"] == [1]
    sp0, sp1 = sessions[1].probe, loaded["eis_sessions"][1].probe
    f0, f1 = sp0.fit_results[DRT_MODEL_NAME], sp1.fit_results[DRT_MODEL_NAME]
    assert np.array_equal(f0.drt_gamma, f1.drt_gamma) and f0.target_value == f1.target_value
    assert np.array_equal(sp0.Zre, sp1.Zre)
    assert aio.count_drt(loaded["eis_sessions"]) == (8, 8)
    assert loaded["eis_sessions"][1].groups[0].analysis.drt_target.n == 3


def test_save_without_analysis_is_refused():
    with pytest.raises(aio.AnalysisFileError):
        aio.save_analysis({"eis_sessions": None})


def test_load_rejects_foreign_or_corrupt_files():
    with pytest.raises(aio.AnalysisFileError):
        aio.load_analysis(b"pas un zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("experiment.yaml", "mode: both")        # ZIP d'expérience, pas d'analyse
    with pytest.raises(aio.AnalysisFileError, match="pas une analyse"):
        aio.load_analysis(buf.getvalue())


def test_load_refuses_arbitrary_code_in_pickle(tmp_path):
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned"),))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", '{"format": "eis-analysis", "version": 1}')
        zf.writestr("analysis.pkl", pickle.dumps({"eis_sessions": {1: Evil()}}))
    with pytest.raises(aio.AnalysisFileError, match="non autorisé"):
        aio.load_analysis(buf.getvalue())
    assert not (tmp_path / "pwned").exists()
