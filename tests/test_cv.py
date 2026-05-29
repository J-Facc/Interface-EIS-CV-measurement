"""Tests for CV loader, pipeline, and plot functions."""

import numpy as np
import pytest

from core.cv_loader import load_cv_file, average_cv_replicates
from core.cv_pipeline import run_cv_pipeline
from core.cv_models import CVScan, CVSession


# ── Fixtures ──────────────────────────────────────────────────────────────────

CSV_BASIC = b"Ewe,I\n-0.5,-1e-6\n0.0,0.0\n0.5,1e-6\n"
CSV_MA = b"E/V,I/mA\n-0.5,-0.001\n0.0,0.0\n0.5,0.001\n"
CSV_UA = b"potential,current/uA\n-0.5,-1.0\n0.0,0.0\n0.5,1.0\n"
CSV_SEMICOLON = b"Ewe;I\n-0.5;-1e-6\n0.0;0.0\n0.5;1e-6\n"
CSV_TAB = b"Ewe\tI\n-0.5\t-1e-6\n0.0\t0.0\n0.5\t1e-6\n"


# ── load_cv_file ──────────────────────────────────────────────────────────────

def test_load_basic_csv():
    scan = load_cv_file(CSV_BASIC, "test", 1e-9, "probe")
    assert len(scan.E) == 3
    assert scan.E[0] == pytest.approx(-0.5)
    assert scan.I[0] == pytest.approx(-1e-6)


def test_load_milliamp_conversion():
    scan = load_cv_file(CSV_MA, "test_ma", 1e-9, "probe")
    assert scan.I[2] == pytest.approx(1e-6)


def test_load_microamp_conversion():
    scan = load_cv_file(CSV_UA, "test_ua", 1e-9, "probe")
    assert scan.I[2] == pytest.approx(1e-6)


def test_load_semicolon_separator():
    scan = load_cv_file(CSV_SEMICOLON, "test_semi", 0.0, "probe")
    assert len(scan.E) == 3


def test_load_tab_separator():
    scan = load_cv_file(CSV_TAB, "test_tab", 0.0, "probe")
    assert len(scan.E) == 3


def test_sorted_by_potential():
    csv = b"Ewe,I\n0.5,1e-6\n-0.5,-1e-6\n0.0,0.0\n"
    scan = load_cv_file(csv, "unsorted", 0.0, "probe")
    assert scan.E[0] < scan.E[1] < scan.E[2]


def test_missing_e_column_raises():
    csv = b"voltage_x,current\n0.0,0.0\n"
    with pytest.raises(ValueError, match="Colonne potentiel"):
        load_cv_file(csv, "bad", 0.0, "probe")


def test_missing_i_column_raises():
    csv = b"Ewe,amp\n0.0,0.0\n"
    with pytest.raises(ValueError, match="Colonne courant"):
        load_cv_file(csv, "bad", 0.0, "probe")


# ── average_cv_replicates ─────────────────────────────────────────────────────

def test_average_single_returns_same():
    scan = load_cv_file(CSV_BASIC, "s", 0.0, "probe")
    avg = average_cv_replicates([scan])
    np.testing.assert_array_equal(avg.E, scan.E)


def test_average_two_scans():
    s1 = load_cv_file(CSV_BASIC, "s1", 0.0, "probe")
    csv2 = b"Ewe,I\n-0.5,-3e-6\n0.0,0.0\n0.5,3e-6\n"
    s2 = load_cv_file(csv2, "s2", 0.0, "probe")
    avg = average_cv_replicates([s1, s2])
    assert avg.I[2] == pytest.approx(2e-6)


# ── run_cv_pipeline ───────────────────────────────────────────────────────────

def _make_assignment(csv: bytes, step: str, conc: float, name: str = "f.csv") -> dict:
    return {"content": csv, "filename": name, "step": step, "concentration": conc}


def test_pipeline_returns_cv_session():
    assignments = [
        _make_assignment(CSV_BASIC, "probe", 0.0),
        _make_assignment(CSV_BASIC, "hybridization", 1e-9),
        _make_assignment(CSV_BASIC, "hybridization", 1e-8),
    ]
    cv = run_cv_pipeline(assignments)
    assert isinstance(cv, CVSession)
    assert cv.probe is not None
    assert len(cv.groups) == 2


def test_pipeline_delta_signal_shape():
    probe_csv = b"Ewe,I\n-0.5,-2e-6\n0.0,-1e-6\n0.5,0.0\n"
    hyb_csv = b"Ewe,I\n-0.5,-1e-6\n0.0,-0.5e-6\n0.5,0.5e-6\n"
    assignments = [
        _make_assignment(probe_csv, "probe", 0.0),
        _make_assignment(hyb_csv, "hybridization", 1e-9),
    ]
    cv = run_cv_pipeline(assignments)
    assert cv.groups[0].delta_signal.shape == cv.groups[0].scan.E.shape


def test_pipeline_no_probe():
    assignments = [
        _make_assignment(CSV_BASIC, "hybridization", 1e-9),
    ]
    cv = run_cv_pipeline(assignments)
    assert cv.probe is None
    assert np.all(np.isnan(cv.groups[0].delta_signal))


# ── cv_plots (import only — no display) ──────────────────────────────────────

def test_cv_current_figure_importable():
    from plotting.cv_plots import cv_current_figure, cv_calibration_figure
    assignments = [
        _make_assignment(CSV_BASIC, "probe", 0.0),
        _make_assignment(b"Ewe,I\n-0.5,-2e-6\n0.0,0.0\n0.5,2e-6\n", "hybridization", 1e-9),
        _make_assignment(b"Ewe,I\n-0.5,-3e-6\n0.0,0.0\n0.5,3e-6\n", "hybridization", 1e-8),
    ]
    cv = run_cv_pipeline(assignments)
    fig1 = cv_current_figure(cv)
    fig2 = cv_calibration_figure(cv)
    assert fig1 is not None
    assert fig2 is not None
