"""Assignation manuelle, contrôles de cohérence et nettoyage EIS + CV (core/assignment.py, v3)."""

from pathlib import Path

import numpy as np
import pytest

from core.assignment import (
    LEVEL_ERROR,
    LEVEL_WARNING,
    AssignmentRow,
    check_assignment,
    clean_files,
    detect_kind,
    guess_from_filename,
    guess_rows,
    has_errors,
    step_options,
)
from core.cv_loader import load_cv_measurement
from core.experiment import Channel, Experiment
from tests.synthetic_data import make_config, randles_file

DATA = Path(__file__).parent / "data"
CV_FILE = DATA / "cv_bare_2cycles.txt"


@pytest.fixture
def exp():
    return Experiment(["A", "B"], [Channel(1, "A", [1e-9, 1e-8]), Channel(2, "B", [1e-9])])


def _row(fid, kind="EIS", ch=1, el=1, step="probe", rep=1, **kw):
    return AssignmentRow(file_id=fid, filename=f"{fid}.txt", kind=kind, channel=ch, electrode=el,
                         step=step, replicate=rep, detected_kind=kw.pop("detected", kind), **kw)


# ── Détection et pré-remplissage ─────────────────────────────────────────────

def test_detect_kind_from_content():
    assert detect_kind(randles_file(1000.0)) == "EIS"
    assert detect_kind(CV_FILE.read_bytes()) == "CV"
    assert detect_kind(b"foo;bar\n1;2\n") is None


@pytest.mark.parametrize("name, expected", [
    ("w4_bare_07_CV_C01.mpr", {"kind": "CV", "step": "bare"}),
    ("canal2_E2_probe_r3.txt", {"channel": 2, "electrode": 2, "step": "probe", "replicate": 3}),
    ("ch1_e1_conc_2_rep1_PEIS.txt", {"channel": 1, "electrode": 1, "step": "conc_2", "replicate": 1,
                                     "kind": "EIS"}),
    ("mesure.txt", {}),
])
def test_guess_from_filename(name, expected):
    got = guess_from_filename(name)
    assert {k: got.get(k) for k in expected} == expected


def test_guess_rows_numbers_replicates_and_respects_channels(exp):
    files = [("a", "probe_x.txt", "EIS"), ("b", "probe_y.txt", "EIS"), ("c", "canal2_conc_2.txt", "EIS"),
             ("d", "canal9_probe.txt", "CV")]
    rows = guess_rows(files, exp)
    assert [(r.channel, r.step, r.replicate) for r in rows[:2]] == [(1, "probe", 1), (1, "probe", 2)]
    assert rows[2].channel == 2 and rows[2].step is None          # conc_2 n'existe pas sur le canal 2
    assert rows[3].channel == 1 and rows[3].kind == "CV"          # canal 9 inexistant → premier canal
    assert step_options(exp) == ["bare", "probe", "conc_1", "conc_2"]


# ── Contrôles de cohérence : chaque cas déclenche un message clair ───────────

def _messages(issues, level=None):
    return [i.message for i in issues if level is None or i.level == level]


def test_valid_assignment_has_no_error(exp):
    rows = [_row("p1"), _row("p2", rep=2), _row("c1", step="conc_1"), _row("q1", el=2),
            _row("cv", kind="CV")]
    issues = check_assignment(rows, exp)
    assert not has_errors(issues), _messages(issues)


def test_unassigned_file(exp):
    issues = check_assignment([_row("x", step=None, rep=None)], exp)
    assert any("non assigné" in m and "étape" in m and "réplicat" in m for m in _messages(issues, LEVEL_ERROR))


def test_duplicate(exp):
    issues = check_assignment([_row("a"), _row("b")], exp)
    msgs = _messages(issues, LEVEL_ERROR)
    assert any("Doublon" in m and "« a.txt »" in m and "« b.txt »" in m for m in msgs), msgs


def test_conc_step_beyond_the_channel(exp):
    issues = check_assignment([_row("p", ch=2), _row("x", ch=2, step="conc_2")], exp)
    assert any("conc_2 inexistante pour le canal 2" in m for m in _messages(issues, LEVEL_ERROR))


def test_electrode_without_probe_is_reported(exp):
    issues = check_assignment([_row("p"), _row("x", el=2, step="conc_1")], exp)
    assert any("électrode 2" in m and "aucune mesure « probe »" in m for m in _messages(issues, LEVEL_WARNING))


@pytest.mark.parametrize("row, fragment", [
    (dict(ch=7), "canal 7 inexistant"),
    (dict(el=3), "électrode 3 invalide"),
    (dict(step="hyb"), "étape « hyb » invalide"),
    (dict(kind="EIS", detected="CV"), "contenu est un fichier CV"),
    (dict(rep=0), "réplicat 0 invalide"),
])
def test_other_errors(exp, row, fragment):
    issues = check_assignment([_row("x", **row)], exp)
    assert any(fragment in m for m in _messages(issues, LEVEL_ERROR)), _messages(issues)


def test_excluded_rows_are_ignored_and_empty_set_is_an_error(exp):
    assert has_errors(check_assignment([_row("a", exclude=True)], exp))
    issues = check_assignment([_row("a"), _row("b", exclude=True)], exp)
    assert not has_errors(issues)


# ── Nettoyage ────────────────────────────────────────────────────────────────

def test_clean_keeps_every_replicate_and_removes_excluded_points(exp):
    contents = {f"p{k}": randles_file(1500.0, seed=k) for k in range(3)}
    contents["cv"] = CV_FILE.read_bytes()
    rows = [_row(f"p{k}", rep=k + 1) for k in range(3)] + [_row("cv", kind="CV")]
    ms, issues = clean_files(rows, contents, exp, make_config(), excluded_points={"p0": [0, 1]})
    assert not has_errors(issues)
    eis = [m for m in ms if m.kind == "EIS"]
    assert [m.replicate for m in eis] == [1, 2, 3]                 # aucun moyennage
    n = len(eis[1].data["f"])
    assert len(eis[0].data["f"]) == n - 2
    assert np.all(np.diff(eis[1].data["f"]) < 0)                   # HF → BF
    cv = next(m for m in ms if m.kind == "CV")
    assert set(cv.data) == {"E", "I", "cycle"} and np.all(cv.data["cycle"] == 2)  # dernier cycle
    assert any("cycle 2 conservé" in w for w in cv.warnings)


def test_clean_blocks_on_assignment_errors_and_reports_unreadable_files(exp):
    ms, issues = clean_files([_row("a"), _row("b")], {"a": b"", "b": b""}, exp)
    assert ms == [] and has_errors(issues)
    ms, issues = clean_files([_row("a")], {"a": b"freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm\n1\t2\t3\n"}, exp)
    assert ms == [] and any("moins de 5 points" in i.message for i in issues if i.level == LEVEL_ERROR)


# ── Loader CV ────────────────────────────────────────────────────────────────

def test_cv_loader_on_a_real_two_cycle_file():
    d = load_cv_measurement(CV_FILE.read_bytes(), "cv", "last")
    assert d["cycles"] == [1, 2] and d["cycle_kept"] == 2
    assert abs(d["I"]).max() < 1e-5                       # mA → A (≈ 3,3 µA)
    assert d["E"][0] == pytest.approx(0.0, abs=1e-3)      # ordre chronologique (départ à 0 V)
    assert d["E"].min() == pytest.approx(-0.2, abs=2e-3)
    first = load_cv_measurement(CV_FILE.read_bytes(), "cv", 1)
    assert first["cycle_kept"] == 1 and np.all(first["cycle"] == 1)
    with pytest.raises(ValueError, match="cycle 5 absent"):
        load_cv_measurement(CV_FILE.read_bytes(), "cv", 5)


def test_cv_loader_reads_the_archive_csv_format():
    E = np.linspace(-0.2, 0.2, 20)
    text = "E,I,cycle\n" + "\n".join(f"{e},{1e-6 * e},1" for e in E)
    d = load_cv_measurement(text.encode(), "x")
    assert np.allclose(d["E"], E) and np.allclose(d["I"], 1e-6 * E)
    with pytest.raises(ValueError, match="EIS"):
        load_cv_measurement(randles_file(1000.0), "eis")
