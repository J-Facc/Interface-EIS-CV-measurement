"""Garde-fou « display only » — la référence « électrode nue » (bare_reference)
ne doit JAMAIS influencer une sortie calculée.

On construit une même session d'exemple (EIS et CV), on calcule les sorties
(Rct par modèle, pente/ordonnée/R² de calibration), puis on renseigne
`bare_reference` avec une courbe volontairement aberrante et on recalcule :
tout chiffre qui bouge signalerait une fuite de la bare dans un calcul.

On vérifie en outre côté rendu que la superposition bare n'ajoute qu'UNE trace
de référence, sans altérer les données des autres courbes.
"""

import numpy as np

from core.models import EISSession, EISSpectrum, FitResult, ConcentrationGroup
from core.cv_models import CVSession, CVConcentrationGroup, CVScan
from core.calibration import (
    compute_calibration,
    compute_calibration_all,
    compute_calibration_loglog,
    compute_cv_calibration,
)
from plotting.eis_plots import nyquist_figure_electrode
from plotting.cv_plots import cv_current_figure

MODEL = "randles_full"


# ── Fixtures EIS (mêmes conventions que tests/test_calibration.py) ────────────

def _fr(rct: float) -> FitResult:
    z = np.array([])
    return FitResult(
        model_name=MODEL, params={}, params_std={},
        Zfit_re=z, Zfit_im=z, chi2_reduced=0.0, residuals_re=z, residuals_im=z,
        Rct=rct, Rct_std=0.0, converged=True,
    )


def _spectrum(conc: float, step: str) -> EISSpectrum:
    f = np.logspace(5, -1, 10)
    return EISSpectrum(label=step, f=f, Zre=f * 0 + 1.0, Zim=f * 0 + 1.0,
                       concentration=conc, step=step, n_points=len(f))


def _eis_session() -> EISSession:
    s = EISSession()
    s.probe = _spectrum(0.0, "probe")
    s.probe.fit_results = {MODEL: _fr(3000.0)}
    for conc, rct in [(1e-13, 3900.0), (1e-11, 4800.0), (1e-9, 5700.0), (1e-7, 6600.0)]:
        sp = _spectrum(conc, "hybridization")
        sp.fit_results = {MODEL: _fr(rct)}
        s.groups.append(ConcentrationGroup(concentration=conc, spectrum=sp,
                                           fit_results={MODEL: _fr(rct)}))
    return s


def _aberrant_bare_spectrum() -> EISSpectrum:
    """Référence bare volontairement extrême (Rct énorme) : si elle fuitait dans
    un calcul, les chiffres bougeraient de façon flagrante."""
    sp = _spectrum(0.0, "bare")
    sp.fit_results = {MODEL: _fr(1e9)}
    return sp


def _eis_calc_snapshot(session):
    """(Rct par groupe, calibration signal, calibration loglog, calib_all)."""
    rcts = tuple(g.fit_results[MODEL].Rct for g in session.groups)
    cal = compute_calibration(session, MODEL)
    cal_ll = compute_calibration_loglog(session, MODEL)
    cal_all = compute_calibration_all(session)
    return (
        rcts,
        (cal.slope, cal.intercept, cal.r2, cal.probe_rct),
        (cal_ll.slope, cal_ll.intercept, cal_ll.r2),
        tuple((c.model, c.slope, c.intercept, c.r2) for c in cal_all),
    )


def test_eis_bare_reference_does_not_change_calculations():
    session = _eis_session()
    before = _eis_calc_snapshot(session)

    session.bare_reference = _aberrant_bare_spectrum()
    after = _eis_calc_snapshot(session)

    assert before == after, "bare_reference a fui dans un calcul EIS"


# ── Fixtures CV ──────────────────────────────────────────────────────────────

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


def _aberrant_bare_scan() -> CVScan:
    return CVScan(label="bare", E=np.linspace(-0.5, 0.5, 5),
                  I=np.full(5, 1e6), concentration=0.0, step="bare")


def test_cv_bare_reference_does_not_change_calculations():
    s = _cv_session()
    cal_before = compute_cv_calibration(s)
    before = (cal_before.slope, cal_before.intercept, cal_before.r2)

    s.bare_reference = _aberrant_bare_scan()
    cal_after = compute_cv_calibration(s)
    after = (cal_after.slope, cal_after.intercept, cal_after.r2)

    assert before == after, "bare_reference a fui dans un calcul CV"


# ── Rendu : la superposition bare n'ajoute qu'UNE trace, sans altérer le reste ─

def _eis_display_spectra(session):
    spectra = [{"label": "Probe", "Zre": session.probe.Zre,
                "Zim": session.probe.Zim, "concentration": 0.0}]
    for g in session.groups:
        spectra.append({"label": g.spectrum.label, "Zre": g.spectrum.Zre,
                        "Zim": g.spectrum.Zim, "concentration": g.concentration})
    return spectra


def test_eis_nyquist_bare_overlay_is_additive_only():
    session = _eis_session()
    spectra = _eis_display_spectra(session)

    fig_no = nyquist_figure_electrode(spectra, title="t")
    fig_bare = nyquist_figure_electrode(spectra, title="t",
                                        bare=_aberrant_bare_spectrum())

    assert len(fig_bare.data) == len(fig_no.data) + 1
    ref = [t for t in fig_bare.data if t.name == "Électrode nue (réf.)"]
    assert len(ref) == 1
    # Les traces de données préexistantes sont inchangées (mêmes x/y).
    for t0, t1 in zip(fig_no.data, fig_bare.data):
        assert np.array_equal(t0.x, t1.x)
        assert np.array_equal(t0.y, t1.y)


def test_cv_current_bare_overlay_is_additive_only():
    s = _cv_session()
    fig_no = cv_current_figure(s)

    s.bare_reference = _aberrant_bare_scan()
    fig_bare = cv_current_figure(s)

    assert len(fig_bare.data) == len(fig_no.data) + 1
    ref = [t for t in fig_bare.data if t.name == "Électrode nue (réf.)"]
    assert len(ref) == 1
    for t0, t1 in zip(fig_no.data, fig_bare.data):
        assert np.array_equal(t0.x, t1.x)
        assert np.array_equal(t0.y, t1.y)
