"""Archive .zip v3 (exports/archive.py) : aller-retour exact, aucun recalcul, archives invalides."""

import io
import json
import zipfile

import numpy as np
import pytest

from core.calibration import SOURCE_DRT, SOURCE_FIT, VIEW_BIOMARKER, VIEW_ELECTRODE, cv_calibration, eis_calibration
from exports.archive import MANIFEST, ArchiveError, is_v3_archive, read_archive, write_archive
from tests.synthetic_data import make_config
from tests.v3_data import build_session


@pytest.fixture(scope="module")
def session():
    return build_session()


def _same_array(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.asarray(a).shape == np.asarray(b).shape and np.allclose(a, b, equal_nan=True, rtol=0, atol=0)


def _assert_same(res, back):
    assert set(back.eis.spectra) == set(res.eis.spectra) and set(back.eis.groups) == set(res.eis.groups)
    for k, g in res.eis.groups.items():
        assert back.eis.groups[k].meta() == g.meta()
    for k, s in res.eis.spectra.items():
        b = back.eis.spectra[k]
        assert b.meta() == s.meta(), k
        for attr in ("f", "Zre", "Zim"):
            assert _same_array(getattr(s, attr), getattr(b, attr)), (k, attr)
        if s.fit:
            assert _same_array(s.fit.Zfit_re, b.fit.Zfit_re) and _same_array(s.fit.Zfit_im, b.fit.Zfit_im)
        if s.drt:
            for attr in ("tau", "gamma", "gamma_lo", "gamma_hi", "Zfit_re", "Zfit_im"):
                assert _same_array(getattr(s.drt, attr), getattr(b.drt, attr)), (k, attr)
            assert b.drt.rct_main == s.drt.rct_main
    assert {k: f.meta() for k, f in res.cv.items()} == {k: f.meta() for k, f in back.cv.items()}
    assert [f.meta() for f in res.failures] == [f.meta() for f in back.failures]
    from core.results import _jsonable
    assert back.eis.circuit == _jsonable(res.eis.circuit)


def test_roundtrip_is_exact_and_calls_no_fit(session, monkeypatch):
    exp, ms, res = session
    data = write_archive(exp, ms, res, make_config())
    assert is_v3_archive(data)

    def forbidden(*a, **k):
        raise AssertionError("un calcul a été relancé au rechargement")

    import core.analysis
    import core.pipeline
    import drt.engine
    import fits.orazem_fit
    for mod, name in ((fits.orazem_fit, "fit_spectrum"), (fits.orazem_fit, "fit_replicate_group"),
                      (drt.engine, "fit_drt"), (core.pipeline, "_analyze_group"),
                      (core.analysis, "eis_group_task"), (core.analysis, "drt_task"),
                      (core.pipeline, "analyze_replicates")):
        monkeypatch.setattr(mod, name, forbidden)
    exp2, ms2, back = read_archive(data)
    assert exp2.to_dict() == exp.to_dict()
    by_id = {m.id: m for m in ms}
    assert {m.id: m.meta_dict() for m in ms2} == {k: m.meta_dict() for k, m in by_id.items()}
    for m in ms2:
        for k, v in m.data.items():
            assert _same_array(v, by_id[m.id].data[k]), (m.id, k)
    _assert_same(res, back)
    # Étalonnages identiques (mêmes points, mêmes régressions).
    for src in (SOURCE_DRT, SOURCE_FIT):
        for view in (VIEW_BIOMARKER, VIEW_ELECTRODE):
            a, b = eis_calibration(res, exp, src, view), eis_calibration(back, exp2, src, view)
            assert [(c.label, c.slope, c.r2) for c in a] == [(c.label, c.slope, c.r2) for c in b]
            assert np.allclose([lv.mean for c in a for lv in c.levels], [lv.mean for c in b for lv in c.levels])
    assert [c.slope for c in cv_calibration(res, exp)] == [c.slope for c in cv_calibration(back, exp2)]


def test_archive_layout_and_manifest(session):
    exp, ms, res = session
    data = write_archive(exp, ms, res, make_config())
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        man = json.loads(zf.read(MANIFEST))
    assert man["schema_version"] == 3 and man["app_version"] and man["experiment"]["name"] == "essai"
    assert "fit.drt.tau_window" in man["config_flat"]
    for p in ("data/eis/C1_E1_probe_r1.csv", "data/cv/C1_E2_conc_3_r1.csv", "results/fits_randles.csv",
              "results/cv_features.csv", "results/calibration_eis.csv", "results/calibration_cv.csv",
              "results/drt_peaks.csv", "results/drt/EIS_C1_E1_probe_r1.csv", "results/analysis.json"):
        assert p in names, p
    assert {f["path"] for f in man["files"]} == names - {MANIFEST}
    assert all(len(f["sha256"]) == 64 for f in man["files"])


def test_archive_without_results(session):
    exp, ms, _ = session
    exp2, ms2, res2 = read_archive(write_archive(exp, ms, None, make_config()))
    assert res2 is None and len(ms2) == len(ms)


def _mutate(data: bytes, fn) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            content = fn(name, src.read(name))
            if content is not None:
                dst.writestr(name, content)
    return out.getvalue()


@pytest.mark.parametrize("mutation, fragment", [
    (lambda n, b: b.replace(b"1", b"2", 1) if n == "data/eis/C1_E1_probe_r1.csv" else b, "Somme de contrôle"),
    (lambda n, b: None if n == "results/drt/EIS_C1_E1_probe_r1.csv" else b, "manquant"),
    (lambda n, b: None if n == MANIFEST else b, "manifest"),
    (lambda n, b: json.dumps(dict(json.loads(b), schema_version=99)).encode() if n == MANIFEST else b,
     "Version de schéma"),
    (lambda n, b: b"{pas du json" if n == MANIFEST else b, "illisible"),
])
def test_corrupted_archives_raise_a_clear_error(session, mutation, fragment):
    exp, ms, res = session
    bad = _mutate(write_archive(exp, ms, res, make_config()), mutation)
    with pytest.raises(ArchiveError, match=fragment):
        read_archive(bad)


def test_not_a_zip():
    with pytest.raises(ArchiveError, match="archive .zip"):
        read_archive(b"ceci n'est pas un zip")
    assert not is_v3_archive(b"xx")
