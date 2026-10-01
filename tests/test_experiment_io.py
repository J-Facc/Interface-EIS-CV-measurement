"""Tests for core/experiment_io.py.

* apply_point_exclusions (Bug 3) — les deux premiers tests, historiques ;
* tests de CARACTÉRISATION (filet de sécurité avant refonte, cf. AUDIT.md §8) : aller-retour
  save_experiment / load_experiment (formats v1 et v2), apply_exclusions et cas
  supplémentaires d'apply_point_exclusions. Ils figent le comportement RÉEL, y compris
  les pièges signalés par « COMPORTEMENT ACTUEL » (positions de réplicats non conservées,
  fichiers manquants ignorés en silence…).
"""

import datetime
import io
import re
import zipfile

import numpy as np
import pytest
import yaml

import core.experiment_io as experiment_io
from core.experiment_io import (
    _read_bytesio,
    _sig_types_for_mode,
    apply_exclusions,
    apply_point_exclusions,
    load_experiment,
    save_experiment,
)
from core.loader import load_spectrum


def _make_csv_bio(n: int = 10) -> io.BytesIO:
    """Build an in-memory CSV (positive Zim, frequencies clear of 50/100 Hz)."""
    freqs = np.logspace(np.log10(200), 4, n)
    lines = ["frequency_hz,zreal_ohm,zimag_ohm"]
    for i, f in enumerate(freqs):
        zre = 500 + i * 100
        zim = 200 - i * 5
        lines.append(f"{f:.4f},{zre:.2f},{zim:.2f}")
    bio = io.BytesIO("\n".join(lines).encode())
    bio.seek(0)
    return bio


def _make_experiment(bio: io.BytesIO) -> dict:
    return {
        "mode": "eis_only",
        "n_electrodes": 1,
        "concentrations": [],
        "probe": {"eis": {"electrode_1": [bio]}},
        "calibration": {"eis": {"electrode_1": []}, "cv": {"electrode_1": []}},
    }


def test_apply_point_exclusions_removes_correct_points():
    bio = _make_csv_bio(10)
    original = load_spectrum(bio.getvalue(), "orig.csv")
    bio.seek(0)

    experiment = _make_experiment(bio)
    deleted_idx = [2, 5]
    point_exclusions = {"e1_eis_cprobe_r0": deleted_idx}

    exp_clean = apply_point_exclusions(experiment, point_exclusions)

    new_bio = exp_clean["probe"]["eis"]["electrode_1"][0]
    new_sp = load_spectrum(new_bio.getvalue(), "new.csv")

    assert new_sp.n_points == original.n_points - len(deleted_idx)

    keep = [i for i in range(original.n_points) if i not in deleted_idx]
    np.testing.assert_allclose(new_sp.f, original.f[keep])
    np.testing.assert_allclose(new_sp.Zre, original.Zre[keep])
    np.testing.assert_allclose(new_sp.Zim, original.Zim[keep])


def test_apply_point_exclusions_leaves_unaffected_replicates_untouched():
    bio = _make_csv_bio(10)
    experiment = _make_experiment(bio)

    exp_clean = apply_point_exclusions(experiment, {})

    assert exp_clean["probe"]["eis"]["electrode_1"][0] is not None
    new_sp = load_spectrum(
        exp_clean["probe"]["eis"]["electrode_1"][0].getvalue(), "untouched.csv"
    )
    assert new_sp.n_points == 10


# ═════════════════════════════════════════════════════════════════════════════
# Caractérisation — save_experiment / load_experiment
# ═════════════════════════════════════════════════════════════════════════════

def _bio(text, name=None):
    b = io.BytesIO(text.encode() if isinstance(text, str) else text)
    if name is not None:
        b.name = name
    return b


def _txt(bios):
    """Contenu texte (et non les objets) d'une liste de BytesIO."""
    return [b.getvalue().decode() for b in bios]


def _full_experiment():
    """Expérience complète : 2 électrodes, EIS + CV, 2 concentrations, validation EIS."""
    return {
        "name": "Expérience é", "date": "2026-01-02", "mode": "both",
        "concentrations": [1e-9, 1e-8], "n_electrodes": 2, "n_replicats": 2,
        "probe": {
            "eis": {"electrode_1": [_bio("p1"), _bio("p2")], "electrode_2": [_bio("q1")]},
            "cv": {"electrode_1": [_bio("cvp1")], "electrode_2": []},
        },
        "calibration": {
            "eis": {"electrode_1": [[_bio("a1"), _bio("a2")], [_bio("b1")]], "electrode_2": []},
            "cv": {"electrode_1": [[_bio("k1")], [_bio("k2")]], "electrode_2": []},
        },
        "validation": {"eis": {"electrode_1": [[_bio("v1")]]}, "cv": {}},
    }


def _zip_of(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _meta_of(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return yaml.safe_load(zf.read("experiment.yaml"))


def test_zip_layout_and_file_naming():
    z = save_experiment(_full_experiment())

    expected = [
        "calibration/cv/electrode_1/cv_e1_1.00e-09M_r1.txt",
        "calibration/cv/electrode_1/cv_e1_1.00e-08M_r1.txt",
        "calibration/eis/electrode_1/eis_e1_1.00e-09M_r1.txt",
        "calibration/eis/electrode_1/eis_e1_1.00e-09M_r2.txt",
        "calibration/eis/electrode_1/eis_e1_1.00e-08M_r1.txt",
        "experiment.yaml",
        "probe/cv/electrode_1/probe_cv_e1_r1.txt",
        "probe/eis/electrode_1/probe_eis_e1_r1.txt",
        "probe/eis/electrode_1/probe_eis_e1_r2.txt",
        "probe/eis/electrode_2/probe_eis_e2_r1.txt",
        "validation/eis/electrode_1/eis_e1_val_1.00e-09M_r1.txt",
    ]
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        assert sorted(zf.namelist()) == sorted(expected)
    meta = _meta_of(z)
    assert meta["file_map"]["probe"]["eis"]["electrode_1"] == [
        "probe/eis/electrode_1/probe_eis_e1_r1.txt", "probe/eis/electrode_1/probe_eis_e1_r2.txt"]
    assert meta["file_map"]["calibration"]["eis"]["electrode_1"] == [
        ["calibration/eis/electrode_1/eis_e1_1.00e-09M_r1.txt",
         "calibration/eis/electrode_1/eis_e1_1.00e-09M_r2.txt"],
        ["calibration/eis/electrode_1/eis_e1_1.00e-08M_r1.txt"],
    ]


def test_round_trip_v2_preserves_content_and_metadata():
    exp = _full_experiment()
    out = load_experiment(save_experiment(exp, preprocessing_done=True))

    assert {k: out[k] for k in ("name", "date", "mode", "concentrations", "n_electrodes",
                                "n_replicats", "preprocessing_done")} == {
        "name": "Expérience é", "date": "2026-01-02", "mode": "both",
        "concentrations": [1e-9, 1e-8], "n_electrodes": 2, "n_replicats": 2,
        "preprocessing_done": True,
    }
    assert _txt(out["probe"]["eis"]["electrode_1"]) == ["p1", "p2"]
    assert _txt(out["probe"]["eis"]["electrode_2"]) == ["q1"]
    assert _txt(out["probe"]["cv"]["electrode_1"]) == ["cvp1"]
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["a1", "a2"], ["b1"]]
    assert [_txt(r) for r in out["calibration"]["cv"]["electrode_1"]] == [["k1"], ["k2"]]
    assert [_txt(r) for r in out["validation"]["eis"]["electrode_1"]] == [["v1"]]
    # Électrodes sans données : listes vides (jamais absentes).
    assert out["probe"]["cv"]["electrode_2"] == [] and out["calibration"]["eis"]["electrode_2"] == []
    assert out["exclusions"] == {} and out["deleted_points"] == {}


def test_loaded_files_are_bytesio_at_position_zero_named_after_the_archive_member():
    out = load_experiment(save_experiment(_full_experiment()))
    bio = out["probe"]["eis"]["electrode_1"][0]

    assert isinstance(bio, io.BytesIO)
    assert bio.tell() == 0
    assert bio.name == "probe_eis_e1_r1.txt"
    assert out["calibration"]["eis"]["electrode_1"][1][0].name == "eis_e1_1.00e-08M_r1.txt"


def test_save_leaves_the_caller_streams_at_position_zero():
    exp = _full_experiment()
    save_experiment(exp)
    assert exp["probe"]["eis"]["electrode_1"][0].tell() == 0
    assert exp["probe"]["eis"]["electrode_1"][0].read() == b"p1"


def test_round_trip_drops_missing_and_empty_replicates_and_shifts_positions():
    """COMPORTEMENT ACTUEL (piège, observation non listée dans l'audit).

    Un réplicat ``None`` ou vide n'est pas écrit et la position n'est PAS conservée :
    [p1, None, p3] revient sous la forme [p1, p3] (le nom de fichier « r3 » garde la
    trace de la position d'origine, mais l'indice de liste, lui, se décale). Or les
    exclusions et les points supprimés sont indexés par POSITION : appliqués après un
    aller-retour, ils visent un autre réplicat.
    """
    exp = {
        "mode": "eis_only", "concentrations": [1e-9], "n_electrodes": 1,
        "probe": {"eis": {"electrode_1": [_bio("p1"), None, _bio("p3")]}},
        "calibration": {"eis": {"electrode_1": [[_bio("a1"), _bio(""), _bio("a3")]]}},
    }
    out = load_experiment(save_experiment(exp))

    probe = out["probe"]["eis"]["electrode_1"]
    assert _txt(probe) == ["p1", "p3"]
    assert [b.name for b in probe] == ["probe_eis_e1_r1.txt", "probe_eis_e1_r3.txt"]
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["a1", "a3"]]

    # Conséquence : une exclusion posée sur la position 1 (ex-« p3 ») ne vise plus le même fichier.
    kept = apply_exclusions(out, {"e1": {"eis": {"probe": [False, True]}}})
    assert _txt(kept["probe"]["eis"]["electrode_1"]) == ["p1"]


def test_round_trip_exclusions_and_deleted_points_are_normalised():
    exclusions = {
        "e1": {"eis": {"probe": [True, False], 0: [False, True], 1: []}, "cv": {}},
        "e2": {},
    }
    deleted = {"e1_eis_cprobe_r0": [3, 1], "e1_eis_c0_r0": []}
    z = save_experiment(_full_experiment(), exclusions=exclusions, deleted_points=deleted)
    out = load_experiment(z)

    # Modalités/électrodes vides écartées ; clés de concentration rendues entières ;
    # « probe » conservé ; listes vides conservées telles quelles.
    assert out["exclusions"] == {"e1": {"eis": {"probe": [True, False], 0: [False, True], 1: []}}}
    assert all(type(k) in (int, str) for k in out["exclusions"]["e1"]["eis"])
    assert out["deleted_points"] == {"e1_eis_cprobe_r0": [1, 3]}       # trié, vides écartés


def test_empty_exclusions_are_omitted_but_empty_deleted_points_are_written_as_an_empty_dict():
    """COMPORTEMENT ACTUEL (incohérence bénigne) : ``exclusions`` n'est écrit que s'il reste
    quelque chose après filtrage ; ``deleted_points`` l'est dès que le dict d'entrée est
    non vide, même si toutes ses listes le sont (il ressort alors vide à la relecture)."""
    z = save_experiment(_full_experiment(), exclusions={"e1": {}}, deleted_points={"x": []})
    meta = _meta_of(z)
    assert "exclusions" not in meta
    assert meta["deleted_points"] == {}
    assert load_experiment(z)["deleted_points"] == {}

    meta = _meta_of(save_experiment(_full_experiment(), exclusions={}, deleted_points={}))
    assert "exclusions" not in meta and "deleted_points" not in meta


def test_single_mode_experiments_only_carry_their_signal_type():
    exp = _full_experiment()
    eis = load_experiment(save_experiment(dict(exp, mode="eis_only")))
    assert list(eis["probe"]) == ["eis"]
    assert list(eis["calibration"]) == ["eis", "cv"] and eis["calibration"]["cv"] is None
    assert list(eis["validation"]) == ["eis"]

    exp = _full_experiment()
    cv = load_experiment(save_experiment(dict(exp, mode="cv_only")))
    assert list(cv["probe"]) == ["cv"]
    assert cv["calibration"]["eis"] is None and cv["calibration"]["cv"] is not None
    # La validation fournie était EIS seule, mais ``validation`` n'est pas None pour autant :
    # un dict de listes vides est « vrai » (voir test_validation_is_only_none_when_absent).
    assert cv["validation"] == {"cv": {"electrode_1": [], "electrode_2": []}}


def test_validation_is_only_none_when_absent_not_when_empty():
    """COMPORTEMENT ACTUEL (piège, observation non listée dans l'audit).

    ``validation`` revient ``None`` seulement si la clé est absente, ``None`` ou ``{}``.
    Un dict sans fichier (``{"eis": {}, "cv": {}}``) est « vrai » : il est écrit dans le
    file_map, donc relu comme un dict de listes vides — un appelant qui teste
    ``if experiment["validation"]`` croit alors à des données de validation.
    """
    for absent in (None, {}):
        exp = _full_experiment()
        exp["validation"] = absent
        assert load_experiment(save_experiment(exp))["validation"] is None
    exp = _full_experiment()
    del exp["validation"]
    assert load_experiment(save_experiment(exp))["validation"] is None

    exp = _full_experiment()
    exp["validation"] = {"eis": {}, "cv": {}}
    out = load_experiment(save_experiment(exp))["validation"]
    assert out == {
        "eis": {"electrode_1": [], "electrode_2": []},
        "cv": {"electrode_1": [], "electrode_2": []},
    }


def test_minimal_experiment_gets_documented_defaults():
    out = load_experiment(save_experiment({}))

    assert out["mode"] == "both" and out["n_electrodes"] == 2 and out["n_replicats"] == 3
    assert out["name"] == "" and out["concentrations"] == [] and out["preprocessing_done"] is False
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", out["date"])           # date du jour si absente
    datetime.date.fromisoformat(out["date"])
    assert {s: sorted(d) for s, d in out["probe"].items()} == {
        "eis": ["electrode_1", "electrode_2"], "cv": ["electrode_1", "electrode_2"]}
    assert out["calibration"]["eis"] == {"electrode_1": [], "electrode_2": []}


def test_more_calibration_groups_than_concentrations_use_zero_in_the_file_name():
    exp = _full_experiment()
    exp["concentrations"] = [1e-9]                    # 2 groupes de calibration, 1 concentration
    z = save_experiment(exp)

    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        names = zf.namelist()
    assert "calibration/eis/electrode_1/eis_e1_0.00e+00M_r1.txt" in names
    out = load_experiment(z)                          # le file_map garde les données accessibles
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["a1", "a2"], ["b1"]]
    assert out["concentrations"] == [1e-9]


def test_metadata_types_are_coerced_on_save():
    exp = _full_experiment()
    exp.update(concentrations=[1, 2], n_replicats="4", n_electrodes="2")
    meta = _meta_of(save_experiment(exp, preprocessing_done=1))
    assert meta["concentrations"] == [1.0, 2.0] and all(isinstance(c, float) for c in meta["concentrations"])
    assert meta["n_replicats"] == 4 and meta["n_electrodes"] == 2
    assert meta["preprocessing_done"] is True


def test_load_rejects_non_zip_and_zip_without_experiment_yaml():
    with pytest.raises(zipfile.BadZipFile):
        load_experiment(b"pas un zip")
    with pytest.raises(ValueError, match="'experiment.yaml' introuvable"):
        load_experiment(_zip_of({"probe/eis/electrode_1/a.txt": "x"}))


def _v1_zip():
    return _zip_of({
        "experiment.yaml": yaml.dump({
            "name": "ancien", "mode": "eis_only", "concentrations": [1e-9, 1e-8], "n_electrodes": 1}),
        "probe/eis/electrode_1/rep_0.bin": "P0",
        "probe/eis/electrode_1/rep_1.bin": "P1",
        "probe/eis/electrode_1/rep_3.bin": "P3-orphelin",           # après un trou : jamais lu
        "calibration/eis/electrode_1/conc_0/rep_0.bin": "C00",
        "calibration/eis/electrode_1/conc_1/rep_0.bin": "C10",
        "calibration/eis/electrode_1/conc_1/rep_1.bin": "C11",
        "validation/eis/electrode_1/conc_0/rep_0.bin": "V00",
        "validation/eis/electrode_1/conc_1/rep_0.bin": "V10",
    })


def test_load_v1_format_reads_rep_n_bin_until_the_first_gap():
    out = load_experiment(_v1_zip())

    assert out["name"] == "ancien" and out["mode"] == "eis_only"
    assert out["n_replicats"] == 3 and out["preprocessing_done"] is False     # défauts
    assert out["exclusions"] == {} and out["deleted_points"] == {}
    probe = out["probe"]["eis"]["electrode_1"]
    assert _txt(probe) == ["P0", "P1"]                        # rep_3.bin ignoré (trou en rep_2)
    assert [b.name for b in probe] == ["rep_0.bin", "rep_1.bin"]
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["C00"], ["C10", "C11"]]
    assert out["calibration"]["cv"] is None
    assert [_txt(r) for r in out["validation"]["eis"]["electrode_1"]] == [["V00"], ["V10"]]


def test_load_v1_without_any_data_file():
    z = _zip_of({"experiment.yaml": yaml.dump({"mode": "cv_only", "concentrations": [1], "n_electrodes": 1})})
    out = load_experiment(z)

    assert out["probe"] == {"cv": {"electrode_1": []}}
    assert out["calibration"] == {"eis": None, "cv": {"electrode_1": [[]]}}   # 1 concentration → 1 liste vide
    assert out["validation"] is None


def _v2_zip_with_odd_content():
    meta = {
        "mode": "eis_only", "concentrations": [1e-9], "n_electrodes": 1,
        "exclusions": {"e1": [True, False]},                      # ancien format à plat
        "file_map": {
            "probe": {"eis": {"electrode_1": [
                "probe/eis/electrode_1/a.txt",
                "probe/eis/electrode_1/manquant.txt",
                "probe/eis/electrode_1/b.mpr",
            ]}},
            "calibration": {"eis": {"electrode_1": [["calibration/eis/electrode_1/absent.txt"]]}},
            "validation": {"eis": {}, "cv": {}},
        },
    }
    return _zip_of({
        "experiment.yaml": yaml.dump(meta),
        "probe/eis/electrode_1/a.txt": "A",
        "probe/eis/electrode_1/b.mpr": "MPR-BRUT",
    })


def test_load_v2_silently_drops_files_listed_in_the_file_map_but_missing_from_the_zip(monkeypatch):
    """COMPORTEMENT ACTUEL : un fichier annoncé par le file_map mais absent de l'archive
    est ignoré sans avertissement ; la liste de réplicats raccourcit (ou devient vide)."""
    monkeypatch.setattr(experiment_io, "mpr_to_csv_bytes", lambda data: b"CSV")
    out = load_experiment(_v2_zip_with_odd_content())

    assert [b.name for b in out["probe"]["eis"]["electrode_1"]] == ["a.txt", "b.csv"]
    assert out["calibration"]["eis"]["electrode_1"] == [[]]


def test_load_v2_converts_mpr_members_to_csv(monkeypatch):
    seen = []
    monkeypatch.setattr(experiment_io, "mpr_to_csv_bytes", lambda data: seen.append(data) or b"CSV")
    out = load_experiment(_v2_zip_with_odd_content())

    probe = out["probe"]["eis"]["electrode_1"]
    assert seen == [b"MPR-BRUT"]
    assert (probe[1].name, probe[1].getvalue()) == ("b.csv", b"CSV")
    assert probe[1].tell() == 0


def test_load_v2_keeps_the_raw_mpr_when_the_conversion_fails(monkeypatch):
    def boom(data):
        raise RuntimeError("mpr illisible")

    monkeypatch.setattr(experiment_io, "mpr_to_csv_bytes", boom)
    probe = load_experiment(_v2_zip_with_odd_content())["probe"]["eis"]["electrode_1"]

    assert (probe[1].name, probe[1].getvalue()) == ("b.mpr", b"MPR-BRUT")


def test_load_ignores_the_legacy_flat_exclusion_format():
    """COMPORTEMENT ACTUEL : {"e1": [True, False]} (liste au lieu d'un dict) est ignoré
    en silence — ``exclusions`` revient vide."""
    assert load_experiment(_v2_zip_with_odd_content())["exclusions"] == {}


def test_read_bytesio_tolerates_missing_and_broken_streams():
    assert _read_bytesio(None) == b""
    assert _read_bytesio(object()) == b""            # ni seek ni read
    b = _bio("abc")
    b.read(2)
    assert _read_bytesio(b) == b"abc" and b.tell() == 0


def test_sig_types_for_mode_and_dead_zip_alias():
    assert _sig_types_for_mode("eis_only") == ["eis"]
    assert _sig_types_for_mode("cv_only") == ["cv"]
    assert _sig_types_for_mode("both") == ["eis", "cv"]
    assert _sig_types_for_mode("n'importe quoi") == ["eis", "cv"]      # repli silencieux


# ═════════════════════════════════════════════════════════════════════════════
# Caractérisation — apply_exclusions
# ═════════════════════════════════════════════════════════════════════════════

def _exclusion_experiment(mode="both"):
    return {
        "mode": mode, "n_electrodes": 2,
        "probe": {
            "eis": {"electrode_1": [_bio("p0"), _bio("p1"), _bio("p2")],
                    "electrode_2": [_bio("q0"), _bio("q1")]},
            "cv": {"electrode_1": [_bio("cp0"), _bio("cp1")]},
        },
        "calibration": {
            "eis": {"electrode_1": [[_bio("a0"), _bio("a1")], [_bio("b0"), _bio("b1"), _bio("b2")]],
                    "electrode_2": []},
            "cv": {"electrode_1": [[_bio("k0"), _bio("k1")]]},
        },
    }


_EXCLUSIONS = {
    "e1": {"eis": {"probe": [False, True], 0: [True], 1: [False, False, True]},
           "cv": {"probe": [True, True], 0: [False, True]}},
    "e2": {"eis": {"probe": [True, False, True]}},
}


def test_apply_exclusions_removes_flagged_replicates_by_position():
    out = apply_exclusions(_exclusion_experiment(), _EXCLUSIONS)

    assert _txt(out["probe"]["eis"]["electrode_1"]) == ["p0", "p2"]
    assert _txt(out["probe"]["eis"]["electrode_2"]) == ["q1"]          # liste d'exclusion plus longue : ignorée
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["a1"], ["b0", "b1"]]
    assert _txt(out["probe"]["cv"]["electrode_1"]) == []
    assert [_txt(r) for r in out["calibration"]["cv"]["electrode_1"]] == [["k0"]]


def test_apply_exclusions_returns_a_deep_copy_and_never_mutates_the_input():
    exp = _exclusion_experiment()
    out = apply_exclusions(exp, _EXCLUSIONS)

    assert _txt(exp["probe"]["eis"]["electrode_1"]) == ["p0", "p1", "p2"]
    assert out["probe"]["eis"]["electrode_1"][0] is not exp["probe"]["eis"]["electrode_1"][0]
    assert out is not exp


def test_apply_exclusions_only_touches_the_signal_types_of_the_mode():
    eis = apply_exclusions(_exclusion_experiment("eis_only"), _EXCLUSIONS)
    assert _txt(eis["probe"]["eis"]["electrode_1"]) == ["p0", "p2"]
    assert _txt(eis["probe"]["cv"]["electrode_1"]) == ["cp0", "cp1"]

    cv = apply_exclusions(_exclusion_experiment("cv_only"), _EXCLUSIONS)
    assert _txt(cv["probe"]["eis"]["electrode_1"]) == ["p0", "p1", "p2"]
    assert _txt(cv["probe"]["cv"]["electrode_1"]) == []


def test_apply_exclusions_with_no_exclusions_keeps_everything():
    out = apply_exclusions(_exclusion_experiment(), {})
    assert _txt(out["probe"]["eis"]["electrode_1"]) == ["p0", "p1", "p2"]
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["a0", "a1"], ["b0", "b1", "b2"]]


def test_apply_exclusions_matches_concentration_indices_as_integers_only():
    """COMPORTEMENT ACTUEL : les indices de concentration sont des ``int`` ; une clé
    ``"0"`` (chaîne) ne vise aucun groupe et est ignorée sans erreur."""
    out = apply_exclusions(_exclusion_experiment(), {"e1": {"eis": {"0": [True, True]}}})
    assert [_txt(r) for r in out["calibration"]["eis"]["electrode_1"]] == [["a0", "a1"], ["b0", "b1", "b2"]]


def test_apply_exclusions_defaults_to_two_electrodes_and_mode_both():
    exp = {"probe": {"eis": {"electrode_1": [_bio("z")]}}, "calibration": {}}
    out = apply_exclusions(exp, {"e1": {"eis": {"probe": [True]}}})
    assert out["probe"]["eis"]["electrode_1"] == []


def test_apply_exclusions_tolerates_missing_probe_and_calibration_sections():
    exp = {"mode": "eis_only", "n_electrodes": 1}
    assert apply_exclusions(exp, {"e1": {"eis": {"probe": [True]}}}) == exp


# ═════════════════════════════════════════════════════════════════════════════
# Caractérisation — apply_point_exclusions (compléments)
# ═════════════════════════════════════════════════════════════════════════════

_POINT_FREQS = [10000.0, 5000.0, 2000.0, 1000.0, 500.0, 300.0, 200.0, 150.0]


def _point_csv(freqs=_POINT_FREQS):
    lines = ["freq/Hz,Re,-Im"] + [f"{f},{100 + i},{50 - i}" for i, f in enumerate(freqs)]
    return "\n".join(lines)


def _point_experiment():
    return {
        "mode": "both", "n_electrodes": 1,
        "probe": {"eis": {"electrode_1": [_bio(_point_csv(), "orig.csv"), _bio(_point_csv(), "r1.csv")]}},
        "calibration": {"eis": {"electrode_1": [[_bio(_point_csv(), "c0r0.csv"), _bio(_point_csv(), "c0r1.csv")]]}},
    }


def test_point_exclusions_target_probe_and_calibration_replicates_by_label():
    exp = _point_experiment()
    raw = _point_csv().encode()
    out = apply_point_exclusions(exp, {
        "e1_eis_cprobe_r1": [0, 99],           # 99 hors plage : ignoré
        "e1_eis_c0_r1": [1],
        "e1_eis_c0_r0": [],                    # liste vide : replicat laissé tel quel
    })

    probe = out["probe"]["eis"]["electrode_1"]
    assert probe[0].getvalue() == raw and probe[0].name == "orig.csv"    # non concerné : octets intacts
    assert load_spectrum(probe[1].getvalue(), "x").n_points == 7          # 8 − le point 0
    assert probe[1].name == "r1.csv"
    cal = out["calibration"]["eis"]["electrode_1"][0]
    assert cal[0].getvalue() == raw
    assert load_spectrum(cal[1].getvalue(), "x").n_points == 7
    # L'entrée n'est pas modifiée.
    assert exp["probe"]["eis"]["electrode_1"][1].getvalue() == raw


def test_point_exclusions_rewrite_a_normalised_csv_in_loader_order():
    out = apply_point_exclusions(_point_experiment(), {"e1_eis_cprobe_r1": [0]})
    text = out["probe"]["eis"]["electrode_1"][1].getvalue().decode()

    lines = text.split("\n")
    assert lines[0] == "frequency_Hz,Zreal_Ohm,Zimag_Ohm"
    assert lines[1].startswith("5000.0,101,49")                # point 0 (10 kHz) retiré
    assert len(lines) == 8


def test_point_exclusions_output_is_sorted_hf_to_bf_even_for_ascending_input():
    asc = _point_csv(_POINT_FREQS[::-1])
    exp = {"mode": "eis_only", "n_electrodes": 1, "probe": {"eis": {"electrode_1": [_bio(asc)]}}}
    out = apply_point_exclusions(exp, {"e1_eis_cprobe_r0": [0]})

    bio = out["probe"]["eis"]["electrode_1"][0]
    new = load_spectrum(bio.getvalue(), "n")
    # L'indice 0 vise le 1er point APRÈS nettoyage du loader (le plus haute fréquence : 10 kHz).
    np.testing.assert_allclose(new.f, _POINT_FREQS[1:])
    assert bio.name == "edited.csv"                              # nom par défaut si l'original n'en a pas


def test_point_exclusions_of_every_point_leave_a_header_only_file():
    exp = {"mode": "eis_only", "n_electrodes": 1, "probe": {"eis": {"electrode_1": [_bio(_point_csv())]}}}
    out = apply_point_exclusions(exp, {"e1_eis_cprobe_r0": list(range(8))})
    assert out["probe"]["eis"]["electrode_1"][0].getvalue().decode() == "frequency_Hz,Zreal_Ohm,Zimag_Ohm"


def test_point_exclusions_are_ignored_for_cv_only_and_for_empty_requests():
    raw = _point_csv().encode()
    cv_only = dict(_point_experiment(), mode="cv_only")
    out = apply_point_exclusions(cv_only, {"e1_eis_cprobe_r1": [0]})
    assert out is not cv_only and out["probe"]["eis"]["electrode_1"][1].getvalue() == raw

    for empty in (None, {}):
        out = apply_point_exclusions(_point_experiment(), empty)
        assert out["probe"]["eis"]["electrode_1"][1].getvalue() == raw
