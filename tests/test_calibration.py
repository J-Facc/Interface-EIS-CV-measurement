"""I5 — la calibration a une source unique (core/calibration.py).

Garantit que la figure (plotting) et l'export (exports) affichent exactement les
mêmes pente / ordonnée / R² pour une même session, et qu'ils dérivent tous deux
de core.calibration (plus de calcul dans plotting/).
"""

import csv
import io

import numpy as np

from core.models import EISSession, EISSpectrum, FitResult, ConcentrationGroup
from core.cv_models import CVSession, CVConcentrationGroup, CVScan
from core.calibration import compute_calibration, compute_cv_calibration
from plotting.eis_plots import calibration_figure
from exports.exporter import (
    export_calibration_csv,
    export_cv_calibration_csv,
    export_cv_calibration_csv_multi,
)

MODEL = "randles_full"


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


def _session() -> EISSession:
    s = EISSession()
    s.probe = _spectrum(0.0, "probe")
    s.probe.fit_results = {MODEL: _fr(3000.0)}
    for conc, rct in [(1e-13, 3900.0), (1e-11, 4800.0), (1e-9, 5700.0), (1e-7, 6600.0)]:
        sp = _spectrum(conc, "hybridization")
        sp.fit_results = {MODEL: _fr(rct)}
        s.groups.append(ConcentrationGroup(concentration=conc, spectrum=sp,
                                           fit_results={MODEL: _fr(rct)}))
    return s


def _export_regression(session):
    reader = csv.reader(io.StringIO(export_calibration_csv(session).decode()))
    rows = list(reader)
    header = rows[0]
    i_model, i_slope, i_int, i_r2 = (header.index(c) for c in
                                     ("model", "slope", "intercept", "r2"))
    for r in rows[1:]:
        if r[i_model] == MODEL:
            return float(r[i_slope]), float(r[i_int]), float(r[i_r2])
    raise AssertionError("modèle absent de l'export")


def _figure_table_regression(fig):
    table = next(t for t in fig.data if t.type == "table")
    cols = table.cells.values  # [modèle, R², pente, intercept, p-value, std err]
    idx = list(cols[0]).index(MODEL)
    return cols[2][idx], cols[3][idx], cols[1][idx]  # slope, intercept, r2 (str)


def test_export_matches_core_exactly():
    session = _session()
    cal = compute_calibration(session, MODEL)
    slope, intercept, r2 = _export_regression({1: session})
    assert slope == cal.slope
    assert intercept == cal.intercept
    assert r2 == cal.r2


def test_figure_matches_core_and_export():
    session = _session()
    cal = compute_calibration(session, MODEL)
    s_str, i_str, r2_str = _figure_table_regression(calibration_figure(session))
    # La figure formate en %.4f : on compare au même formatage de la source.
    assert s_str == f"{cal.slope:.4f}"
    assert i_str == f"{cal.intercept:.4f}"
    assert r2_str == f"{cal.r2:.4f}"
    # ... et l'export porte exactement les mêmes chiffres.
    slope, intercept, r2 = _export_regression({1: session})
    assert (f"{slope:.4f}", f"{intercept:.4f}", f"{r2:.4f}") == (s_str, i_str, r2_str)


# ── I5b : calibration CV ────────────────────────────────────────────────────

def _cv_session() -> CVSession:
    s = CVSession()
    s.probe = CVScan(label="probe", E=np.linspace(-0.5, 0.5, 5),
                     I=np.ones(5), concentration=0.0, step="probe")
    for conc, sig in [(1e-13, 0.10), (1e-11, 0.25), (1e-9, 0.40), (1e-7, 0.55)]:
        scan = CVScan(label=f"c{conc}", E=np.linspace(-0.5, 0.5, 5),
                      I=np.ones(5), concentration=conc, step="hybridization")
        s.groups.append(CVConcentrationGroup(
            concentration=conc, scan=scan, delta_signal=np.full(5, sig)))
    return s


def _cv_export_regression(csv_bytes, model_col_absent=True):
    reader = csv.reader(io.StringIO(csv_bytes.decode()))
    rows = list(reader)
    header = rows[0]
    i_slope, i_int, i_r2 = (header.index(c) for c in ("slope", "intercept", "r2"))
    r = rows[1]
    return float(r[i_slope]), float(r[i_int]), float(r[i_r2])


def test_cv_export_matches_core_exactly():
    s = _cv_session()
    cal = compute_cv_calibration(s)
    for exporter in (lambda: export_cv_calibration_csv(s),
                     lambda: export_cv_calibration_csv_multi({1: s})):
        slope, intercept, r2 = _cv_export_regression(exporter())
        assert slope == cal.slope
        assert intercept == cal.intercept
        assert r2 == cal.r2
