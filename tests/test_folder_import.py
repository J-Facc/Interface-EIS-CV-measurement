"""Import d'un dossier par canal (core/folder_import.py) : noms de fichiers, arborescence, .mpr."""

from pathlib import Path

import pytest

import core.folder_import as fi
from core.assignment import LEVEL_ERROR, LEVEL_WARNING, check_assignment, clean_files, has_errors
from core.experiment import Channel, Experiment
from core.folder_import import import_channel_folder, match_concentration, parse_path, reassign_steps
from tests.synthetic_data import make_config, randles_file

CV = (Path(__file__).parent / "data" / "cv_bare_2cycles.txt").read_bytes()
CONCS = [1e-16, 1e-14, 1e-12]


def _tree(root="canal1", electrodes=(1, 2), concs=CONCS, reps=(1, 2, 3), ext="txt"):
    """Arborescence du poste de mesure : probe/ et calibration/, EIS et CV."""
    files = []
    for e in electrodes:
        for r in reps:
            files.append((f"{root}/probe/eis/electrode_{e}/probe_eis_e{e}_r{r}.{ext}", randles_file(1500.0, seed=e * 10 + r)))
            files.append((f"{root}/probe/cv/electrode_{e}/probe_cv_e{e}_r{r}.{ext}", CV))
            for i, c in enumerate(concs):
                files.append((f"{root}/calibration/eis/electrode_{e}/eis_e{e}_{c:.2e}M_r{r}.{ext}",
                              randles_file(1500.0 + 500 * (i + 1), seed=100 * e + 10 * i + r)))
                files.append((f"{root}/calibration/cv/electrode_{e}/cv_e{e}_{c:.2e}M_r{r}.{ext}", CV))
    return files


@pytest.fixture
def exp():
    return Experiment(["miR-30b"], [Channel(1, "miR-30b", list(CONCS)), Channel(2, "miR-30b", [1e-12])])


# ── Noms de fichiers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("path, expected", [
    ("probe/eis/electrode_1/probe_eis_e1_r3.txt", ("probe", "EIS", 1, 3, None)),
    ("calibration/cv/electrode_2/cv_e2_1.00e-08M_r1.txt", ("calibration", "CV", 2, 1, 1e-8)),
    ("CALIBRATION/EIS/electrode_1/EIS_e1_5e-10M_r2.MPR", ("calibration", "EIS", 1, 2, 5e-10)),
    ("bare/eis/electrode_1/bare_eis_e1_r1.csv", ("bare", "EIS", 1, 1, None)),
    ("probe_eis_e1_r1.mpt", ("probe", "EIS", 1, 1, None)),          # sans dossiers : le nom suffit
    ("canal1\\probe\\cv\\electrode_1\\probe_cv_e1_r2.txt", ("probe", "CV", 1, 2, None)),
])
def test_names_are_parsed(path, expected):
    p = parse_path(path)
    assert (p.dataset, p.kind, p.electrode, p.replicate, p.concentration) == expected
    assert p.warnings == []


@pytest.mark.parametrize("path", ["probe/eis/electrode_1/mesure.txt", "probe_eis_e1.txt", "eis_e1_r1.txt",
                                  "probe/eis/electrode_1/probe_eis_e1_r1.png"])
def test_unexpected_names(path):
    assert parse_path(path) is None


@pytest.mark.parametrize("path, fragment", [
    ("probe/cv/electrode_1/probe_eis_e1_r1.txt", "« cv/ »"),
    ("probe/eis/electrode_2/probe_eis_e1_r1.txt", "electrode_2"),
    ("calibration/eis/electrode_1/probe_eis_e1_r1.txt", "« calibration/ »"),
])
def test_folder_name_inconsistencies_are_reported(path, fragment):
    p = parse_path(path)
    assert p is not None and any(fragment in w for w in p.warnings)


def test_concentration_matching_tolerates_rounded_names():
    assert match_concentration(1.00e-8, [1e-9, 1.0e-8]) == 2
    assert match_concentration(3.33e-9, [3.3333e-9]) == 1
    assert match_concentration(2e-8, [1e-8]) is None


# ── Dossier complet d'un canal ───────────────────────────────────────────────

def test_full_channel_folder_is_assigned_and_cleaned(exp):
    res = import_channel_folder(_tree(), exp, 1)
    assert not [i for i in res.issues], [i.message for i in res.issues]
    assert res.concentrations_found == pytest.approx(CONCS)
    rows = {(r.kind, r.electrode, r.step, r.replicate) for r in res.rows}
    assert len(res.rows) == 2 * 3 * 2 * 4 and len(rows) == len(res.rows)
    assert ("EIS", 2, "conc_3", 3) in rows and ("CV", 1, "probe", 1) in rows
    assert all(r.channel == 1 and not r.exclude for r in res.rows)
    assert not has_errors(check_assignment(res.rows, exp))
    contents = {k: v["content"] for k, v in res.raw_files.items()}
    ms, issues = clean_files(res.rows, contents, exp, make_config())
    assert len(ms) == 48 and not has_errors(issues)


def test_unknown_concentration_is_an_error_until_adopted(exp):
    res = import_channel_folder(_tree(concs=[1e-12, 1e-10]), exp, 2)
    errs = [i.message for i in res.issues if i.level == LEVEL_ERROR]
    assert len(errs) == 1 and "Canal 2 : concentration 1.00e-10 M absente" in errs[0] and "12 fichier(s)" in errs[0]
    assert res.concentrations_found == pytest.approx([1e-12, 1e-10])
    exp2 = Experiment(exp.biomarkers, [exp.channels[0], Channel(2, "miR-30b", res.concentrations_found)])
    rows = reassign_steps(res.rows, exp2)
    assert {r.step for r in rows if r.file_concentration == 1e-10} == {"conc_2"}
    assert not has_errors(check_assignment(rows, exp2))


def test_global_electrode_numbers_are_renumbered_within_the_channel(exp):
    res = import_channel_folder(_tree(electrodes=(3, 4), concs=[1e-12]), exp, 2)
    assert {r.electrode for r in res.rows} == {1, 2}
    assert any(i.level == LEVEL_WARNING and "e3 → électrode 1" in i.message for i in res.issues)
    res = import_channel_folder(_tree(electrodes=(1, 2, 3), concs=[1e-12]), exp, 2)
    assert any(i.level == LEVEL_ERROR and "3 électrodes" in i.message for i in res.issues)


def test_unexpected_file_is_added_excluded_and_reported(exp):
    files = _tree(electrodes=(1,), concs=[], reps=(1,)) + [("canal1/notes/mesure.txt", randles_file(1000.0))]
    res = import_channel_folder(files, exp, 1)
    odd = next(r for r in res.rows if r.filename.endswith("mesure.txt"))
    assert odd.exclude and odd.step is None
    assert any("nom inattendu" in i.message for i in res.issues)


def test_mpr_files_are_converted(exp, monkeypatch):
    seen = []

    def fake(raw):
        seen.append(raw)
        return randles_file(1500.0)
    monkeypatch.setattr(fi, "mpr_to_csv_bytes", fake)
    res = import_channel_folder([("canal1/probe/eis/electrode_1/probe_eis_e1_r1.mpr", b"BIO-LOGIC binary")], exp, 1)
    assert seen == [b"BIO-LOGIC binary"]
    [row] = res.rows
    assert (row.kind, row.detected_kind, row.step) == ("EIS", "EIS", "probe")
    assert next(iter(res.raw_files.values()))["content"].startswith(b"freq/Hz")

    def broken(raw):
        raise RuntimeError("format inconnu")
    monkeypatch.setattr(fi, "mpr_to_csv_bytes", broken)
    res = import_channel_folder([("probe_eis_e1_r1.mpr", b"x")], exp, 1)
    assert res.rows == [] and "conversion .mpr impossible" in res.issues[0].message
