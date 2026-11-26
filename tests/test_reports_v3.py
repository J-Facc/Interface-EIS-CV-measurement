"""Rapports PDF et HTML (exports/report_pdf.py, exports/report_html.py), import d'archives v2."""

import io
import re

import pytest

from core.experiment_io import save_experiment
from core.legacy_import import import_v2
from exports.archive import read_archive, write_archive
from exports.report_html import build_html
from exports.report_pdf import build_pdf, pdf_plan
from tests.synthetic_data import make_config, randles_file
from tests.v3_data import build_session


@pytest.fixture(scope="module")
def session():
    return build_session()


def _pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


def test_pdf_opens_and_has_the_planned_pages(session):
    exp, ms, res = session
    pdf = build_pdf(exp, ms, res)
    assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    plan = pdf_plan(exp, ms, res)
    kinds = [k for k, _ in plan]
    # garde, CV (1 canal), EIS (2 canaux), DRT (2), Rct (2 pages), étalonnage EIS (2 biomarqueurs), CV, échecs
    assert kinds == ["cover", "cv", "eis", "eis", "drt", "drt", "rct", "rct", "cal_eis", "cal_eis", "cal_cv",
                     "failures"]
    assert _pages(pdf) == len(plan)


def test_pdf_without_results_has_only_data_pages(session):
    exp, ms, _ = session
    pdf = build_pdf(exp, ms, None)
    assert _pages(pdf) == len(pdf_plan(exp, ms, None)) == 1 + 1 + 2


def test_reports_from_a_reloaded_archive_are_identical_in_structure(session):
    exp, ms, res = session
    exp2, ms2, res2 = read_archive(write_archive(exp, ms, res, make_config()))
    assert pdf_plan(exp2, ms2, res2) == pdf_plan(exp, ms, res)
    assert _pages(build_pdf(exp2, ms2, res2)) == _pages(build_pdf(exp, ms, res))


def test_html_is_standalone_and_contains_the_expected_figures(session):
    exp, ms, res = session
    page = build_html(exp, ms, res)
    assert page.startswith("<!doctype html>") and "Sommaire" in page
    assert not re.search(r"<script[^>]+src=", page)          # aucune dépendance externe : plotly.js embarqué
    assert page.count("* plotly.js v") == 1                    # … et une seule fois
    n_figs = page.count("Plotly.newPlot")
    # CV : 2 ; EIS : (2 + 1 électrodes) × (Nyquist + Bode) = 6 ; DRT : 3 ; étalonnage EIS : 2 sources × 2 ; CV : 1
    assert n_figs == 2 + 6 + 3 + 4 + 1
    for sid in ("experience", "cv-c1", "eis-c1", "eis-c2", "drt-c1", "rct", "cal-eis", "cv-features", "cal-cv",
                "echecs"):
        assert f'id="{sid}"' in page and f'href="#{sid}"' in page


def test_v2_archive_is_imported_as_configuration_and_assigned_files():
    bio = lambda data, name: (lambda b: (setattr(b, "name", name), b)[1])(io.BytesIO(data))  # noqa: E731
    exp2 = {"name": "ancienne", "date": "2026-01-01", "mode": "eis_only", "concentrations": [1e-9, 1e-8],
            "n_electrodes": 2, "n_replicats": 2,
            "probe": {"eis": {f"electrode_{e}": [bio(randles_file(1000.0, seed=e * 10 + r), f"p{e}{r}.txt")
                                                for r in range(2)] for e in (1, 2)}},
            "calibration": {"eis": {f"electrode_{e}": [[bio(randles_file(1500.0, seed=e * 100 + c * 10 + r),
                                                            f"c{e}{c}{r}.txt") for r in range(2)] for c in range(2)]
                                    for e in (1, 2)}, "cv": None},
            "validation": None}
    z = save_experiment(exp2, exclusions={"e1": {"eis": {"probe": [False, True]}}},
                        deleted_points={"e2_eis_c1_r0": [0, 3]})
    out = import_v2(z)
    exp = out["experiment"]
    assert exp.biomarkers == ["ancienne"] and len(exp.channels) == 1
    assert exp.channels[0].concentrations == [1e-9, 1e-8]
    rows = out["rows"]
    assert len(rows) == 12
    key = {(r.electrode, r.step, r.replicate): r for r in rows}
    assert key[(1, "probe", 2)].exclude and not key[(1, "probe", 1)].exclude
    assert all(r.channel == 1 and r.kind == "EIS" for r in rows)
    assert out["excluded_points"][key[(2, "conc_2", 1)].file_id] == [0, 3]
    from core.assignment import check_assignment, clean_files, has_errors
    assert not has_errors(check_assignment(rows, exp))
    contents = {fid: v["content"] for fid, v in out["raw_files"].items()}
    ms, issues = clean_files(rows, contents, exp, make_config(), out["excluded_points"])
    assert len(ms) == 11 and not has_errors(issues)
