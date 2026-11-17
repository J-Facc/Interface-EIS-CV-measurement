"""Calibration avec 2 concentrations : points affichés, aucune régression, aucun plantage.

Sous ``MIN_POINTS_REGRESSION`` (3) points retenus, ``core.calibration`` rend les points avec des
champs de régression ``None`` ; figures et export CSV le consomment sans exception. Au-delà, le
comportement est inchangé (non-régression).
"""

import csv
import io

import numpy as np
import pytest
from scipy import stats

from core.calibration import (
    MIN_POINTS_REGRESSION,
    MSG_REGRESSION_MIN_POINTS,
    compute_calibration,
    compute_calibration_all,
    compute_calibration_loglog,
    compute_calibration_loglog_all,
)
from core.models import ConcentrationGroup, EISSession, EISSpectrum, FitResult
from exports.exporter import export_calibration_csv
from plotting.eis_plots import calibration_figure, calibration_loglog_figure

MODEL = "orazem"
POINTS = [(1e-13, 3900.0), (1e-11, 4800.0), (1e-9, 5700.0), (1e-7, 6600.0)]


def _fr(rct: float) -> FitResult:
    z = np.array([])
    return FitResult(
        model_name=MODEL, params={}, params_std={},
        Zfit_re=z, Zfit_im=z, chi2_reduced=0.0, residuals_re=z, residuals_im=z,
        target_param="Rct", target_value=rct, target_std=0.0, converged=True,
    )


def _spectrum(conc: float, step: str) -> EISSpectrum:
    f = np.logspace(5, -1, 10)
    return EISSpectrum(label=step, f=f, Zre=f * 0 + 1.0, Zim=f * 0 + 1.0,
                       concentration=conc, step=step, n_points=len(f))


def _session(n: int) -> EISSession:
    s = EISSession()
    s.probe = _spectrum(0.0, "probe")
    s.probe.fit_results = {MODEL: _fr(3000.0)}
    for conc, rct in POINTS[:n]:
        sp = _spectrum(conc, "hybridization")
        sp.fit_results = {MODEL: _fr(rct)}
        s.groups.append(ConcentrationGroup(concentration=conc, spectrum=sp,
                                           fit_results={MODEL: _fr(rct)}))
    return s


def _csv_rows(session):
    return list(csv.DictReader(io.StringIO(export_calibration_csv(session).decode())))


def _annotations(fig) -> str:
    return " ".join(a.text for a in fig.layout.annotations)


@pytest.mark.parametrize("compute", [compute_calibration, compute_calibration_loglog])
def test_two_points_no_regression(compute):
    cal = compute(_session(2), MODEL)
    assert cal is not None and cal.n == 2 and len(cal.log_c) == 2
    assert not cal.has_regression
    assert (cal.slope, cal.intercept, cal.r2, cal.pvalue, cal.stderr) == (None,) * 5


def test_one_point_still_returns_nothing():
    assert compute_calibration(_session(1), MODEL) is None
    assert compute_calibration_loglog(_session(1), MODEL) is None


def test_two_points_figures_show_points_and_message():
    session = _session(2)
    fig = calibration_figure(session)
    scatter = [t for t in fig.data if t.type == "scatter"]
    assert [t.mode for t in scatter] == ["markers"]          # points, pas de droite
    assert len(scatter[0].x) == 2
    table = next(t for t in fig.data if t.type == "table")
    assert list(table.cells.values[1]) == ["—"]               # R² absent, pas de plantage
    assert MSG_REGRESSION_MIN_POINTS in _annotations(fig)

    fig_ll = calibration_loglog_figure(session)
    assert all(t.mode == "markers" for t in fig_ll.data)
    assert MSG_REGRESSION_MIN_POINTS in _annotations(fig_ll)


def test_two_points_exports_do_not_crash():
    session = _session(2)
    rows = _csv_rows(session)
    assert len(rows) == 2
    for r in rows:
        assert all(r[c] == "" for c in ("slope", "intercept", "r2", "p_value", "std_err"))
        assert r["signal_norm"] != ""
    # PNG / HTML de la figure : sérialisation sans exception
    fig = calibration_figure(session)
    assert "<html" in fig.to_html().lower()
    assert fig.to_json()


def test_three_points_regression_unchanged():
    assert MIN_POINTS_REGRESSION == 3
    session = _session(3)
    cal = compute_calibration(session, MODEL)
    assert cal.has_regression and cal.n == 3
    ref = stats.linregress(cal.log_c, cal.y)
    assert cal.slope == float(ref.slope) and cal.r2 == float(ref.rvalue ** 2)
    assert cal.stderr == float(ref.stderr) and cal.pvalue == float(ref.pvalue)

    fig = calibration_figure(session)
    assert [t.mode for t in fig.data if t.type == "scatter"] == ["markers", "lines"]
    assert MSG_REGRESSION_MIN_POINTS not in _annotations(fig)
    assert MSG_REGRESSION_MIN_POINTS not in _annotations(calibration_loglog_figure(session))
    rows = _csv_rows(session)
    assert len(rows) == 3 and float(rows[0]["slope"]) == cal.slope


def test_four_points_unchanged():
    cal = compute_calibration_all(_session(4))[0]
    assert cal.has_regression and cal.n == 4
    assert compute_calibration_loglog_all(_session(4))[0].has_regression


def test_real_pipeline_with_two_concentrations_end_to_end():
    """Pipeline réel (probe + exactement 2 concentrations, DRT off) → calibration, figures, CSV."""
    from core.pipeline import run_pipeline
    from tests import test_pipeline as tp
    from tests.synthetic_data import make_config

    session, _ = run_pipeline(tp._full_assignments(), make_config(**tp._NO_DRT))
    assert len(session.groups) == 2
    cals = compute_calibration_all(session)
    assert cals and all(c.n == 2 and not c.has_regression for c in cals)
    assert calibration_figure(session) is not None
    assert calibration_loglog_figure(session) is not None
    assert len(_csv_rows(session)) == 2 * len(cals)
