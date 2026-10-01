"""B-STATE / B-STATE-b (AUDIT.md §2.5) — réinitialisation des résultats d'analyse.

Avant : les pages remettaient à None des clés MORTES (``eis_session``,
``eis_validation``) ; les vraies (``eis_sessions``, ``eis_validations``,
``eis_normalized``, ``cv_sessions``) survivaient à un rechargement de ZIP, à un nouvel
import ou à une re-validation du prétraitement → l'ANCIENNE analyse était réaffichée et
exportée. Et un nouvel import ne remettait pas ``preprocessing_done`` à False → la page
EIS levait ``KeyError`` sur ``experiment_clean`` (B-STATE-b).
"""

import ast
from pathlib import Path

import pytest

from core.app_state import (
    ANALYSIS_RESULT_KEYS,
    SHARED_DEFAULTS,
    init_shared_state,
    preprocessing_ready,
    reset_analysis_results,
    reset_for_new_experiment,
)

REPO = Path(__file__).resolve().parents[1]
_STALE = object()


def _state_after_an_analysis():
    state = {}
    init_shared_state(state)
    state.update(experiment={"name": "ancienne"}, experiment_clean={"name": "ancienne"},
                 preprocessing_done=True, validation_results={"probe": _STALE})
    for key in ANALYSIS_RESULT_KEYS:
        state[key] = _STALE
    return state


def test_the_real_result_keys_are_reset_not_dead_ones():
    assert set(ANALYSIS_RESULT_KEYS) == {"eis_sessions", "eis_validations", "eis_normalized", "cv_sessions"}
    assert not {"eis_session", "eis_validation"} & set(SHARED_DEFAULTS)


def test_init_creates_missing_keys_without_overwriting():
    state = {"experiment_clean": "gardé"}
    init_shared_state(state)
    assert state["experiment_clean"] == "gardé"
    assert all(state[k] is None for k in ANALYSIS_RESULT_KEYS)
    assert state["preprocessing_done"] is False
    state["exclusions"]["x"] = 1                          # pas de dict partagé entre sessions
    assert SHARED_DEFAULTS["exclusions"] == {}


def test_bstate_fixed_preprocessing_revalidation_invalidates_every_analysis():
    state = _state_after_an_analysis()
    reset_analysis_results(state)
    assert all(state[k] is None for k in ANALYSIS_RESULT_KEYS)
    # Le prétraitement qui vient d'être validé, lui, est conservé.
    assert state["experiment_clean"] == {"name": "ancienne"} and state["preprocessing_done"] is True
    assert state["validation_results"] == {"probe": _STALE}       # verdict KK juste recalculé


def test_bstate_fixed_a_new_experiment_invalidates_analysis_and_preprocessing():
    state = _state_after_an_analysis()
    reset_for_new_experiment(state)
    assert all(state[k] is None for k in ANALYSIS_RESULT_KEYS)
    assert state["validation_results"] is None
    assert state["preprocessing_done"] is False and "experiment_clean" not in state
    assert preprocessing_ready(state) is False                    # B-STATE-b : garde des pages


def test_preprocessing_ready_requires_both_the_flag_and_the_data():
    assert preprocessing_ready({"preprocessing_done": True, "experiment_clean": {}}) is True
    assert preprocessing_ready({"preprocessing_done": True}) is False      # cas B-STATE-b
    assert preprocessing_ready({"preprocessing_done": False, "experiment_clean": {}}) is False


def _calls(path):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    return [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]


@pytest.mark.parametrize("path, expected", [
    ("pages/0_import.py", "reset_for_new_experiment"),            # ZIP rechargé ET import validé
    ("pages/1_pretraitement.py", "reset_analysis_results"),       # re-validation du prétraitement
])
def test_pages_call_the_reset_on_every_data_change(path, expected):
    calls = _calls(path)
    assert calls.count(expected) == (2 if expected == "reset_for_new_experiment" else 1)
    source = (REPO / path).read_text(encoding="utf-8")
    assert '"eis_session"' not in source and '"eis_validation"' not in source


def test_analysis_pages_guard_on_preprocessing_ready():
    for path in ("pages/A_eis.py", "pages/B_cv.py"):
        assert "preprocessing_ready" in _calls(path), path


@pytest.mark.parametrize("page", ["A_eis.py", "B_cv.py"])
def test_bstate_b_fixed_no_keyerror_after_a_reimport(monkeypatch, page):
    """CORRIGÉ (B-STATE-b) : ``preprocessing_done`` resté à True sans ``experiment_clean``
    (ancien ré-import) faisait lever ``KeyError`` à la page ; elle affiche maintenant la
    garde « Aucune donnée disponible »."""
    import streamlit
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(streamlit, "page_link", lambda *a, **k: None)   # page lancée hors navigation
    at = AppTest.from_file(str(REPO / "pages" / page), default_timeout=60)
    at.session_state["preprocessing_done"] = True                       # ancien état incohérent
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Aucune donnée disponible" in w.value for w in at.warning)
