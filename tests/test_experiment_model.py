"""Modèle de données v3 (core/experiment.py) et persistance des biomarqueurs (core/user_data.py)."""

import json

import numpy as np
import pytest

from core.config import user_data_dir
from core.experiment import (
    KIND_EIS,
    MODE_MULTI,
    MODE_SINGLE,
    Channel,
    Experiment,
    Measurement,
    concentration_from_parts,
    concentration_parts,
    conc_index,
    conc_step,
    default_biomarker_for_channel,
    is_valid_step_name,
    sort_measurements,
    step_sort_key,
    validate_experiment,
)
from core.user_data import add_biomarkers, biomarkers_path, load_biomarkers, save_biomarkers


# ── Étapes et concentrations ─────────────────────────────────────────────────

def test_conc_steps_roundtrip_and_order():
    assert conc_step(3) == "conc_3" and conc_index("conc_3") == 3
    assert conc_index("probe") is None and conc_index("conc_0") is None
    assert is_valid_step_name("bare") and is_valid_step_name("conc_12") and not is_valid_step_name("hyb")
    steps = ["conc_10", "probe", "conc_2", "bare"]
    assert sorted(steps, key=step_sort_key) == ["bare", "probe", "conc_2", "conc_10"]
    with pytest.raises(ValueError):
        conc_step(0)


@pytest.mark.parametrize("c", [1e-12, 2.5e-9, 9.99e-7, 1.0])
def test_concentration_parts_roundtrip(c):
    m, e = concentration_parts(c)
    assert concentration_from_parts(m, e) == pytest.approx(c, rel=1e-9)
    assert 1.0 <= m < 10.0


def test_mode_is_deduced_per_channel():
    assert Channel(1, "A", [1e-9]).mode == MODE_SINGLE
    assert Channel(2, "A", [1e-9, 1e-8]).mode == MODE_MULTI
    ch = Channel(1, "A", [1e-9, 1e-8])
    assert ch.steps() == ["bare", "probe", "conc_1", "conc_2"]
    assert ch.concentration_of("conc_2") == 1e-8 and ch.concentration_of("conc_3") is None
    assert ch.concentration_of("probe") is None
    assert ch.has_step("conc_2") and not ch.has_step("conc_3")


def test_default_biomarker_rule():
    bios = ["A", "B", "C"]
    assert [default_biomarker_for_channel(k, bios, 3) for k in range(3)] == bios
    assert [default_biomarker_for_channel(k, bios, 4) for k in range(4)] == ["A"] * 4
    assert default_biomarker_for_channel(0, [], 2) is None


# ── Validation ───────────────────────────────────────────────────────────────

def test_validation_accepts_a_valid_experiment():
    exp = Experiment(["A", "B"], [Channel(1, "A", [1e-9]), Channel(2, "B", [1e-9, 1e-8])])
    assert validate_experiment(exp) == []


@pytest.mark.parametrize("exp, fragment", [
    (Experiment([], [Channel(1, "A", [1e-9])]), "au moins un biomarqueur"),
    (Experiment(["A", ""], [Channel(1, "A", [1e-9])]), "non vide"),
    (Experiment(["A", "a"], [Channel(1, "A", [1e-9])]), "double"),
    (Experiment(["A"], []), "au moins un canal"),
    (Experiment(["A"], [Channel(1, "Z", [1e-9])]), "non déclaré"),
    (Experiment(["A"], [Channel(1, "A", [])]), "au moins une concentration"),
    (Experiment(["A"], [Channel(1, "A", [0.0])]), "> 0"),
    (Experiment(["A"], [Channel(1, "A", [-1e-9])]), "> 0"),
])
def test_validation_messages(exp, fragment):
    errors = validate_experiment(exp)
    assert errors and any(fragment in e for e in errors), errors


def test_experiment_json_roundtrip():
    exp = Experiment(["A", "B"], [Channel(1, "A", [1e-9]), Channel(2, "B", [1e-9, 1e-8])], name="x")
    back = Experiment.from_dict(json.loads(json.dumps(exp.to_dict())))
    assert back.to_dict() == exp.to_dict()
    with pytest.raises(ValueError):
        Experiment.from_dict({"foo": 1})


def test_measurement_to_spectrum_maps_steps_and_concentration():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9, 1e-8])])
    data = {"f": np.array([1e3, 1e2]), "Zre": np.array([1.0, 2.0]), "Zim": np.array([0.5, 0.7])}
    m = Measurement(1, 2, "conc_2", 3, KIND_EIS, "f.txt", data)
    sp = m.to_spectrum(exp)
    assert m.id == "EIS_C1_E2_conc_2_r3"
    assert sp.step == "hybridization" and sp.concentration == 1e-8 and sp.label == m.id
    assert Measurement(1, 1, "probe", 1, KIND_EIS, "p", data).to_spectrum(exp).step == "probe"
    ms = sort_measurements([Measurement(1, 1, "conc_1", 1, KIND_EIS, "", data),
                            Measurement(1, 1, "probe", 2, KIND_EIS, "", data),
                            Measurement(1, 1, "probe", 1, KIND_EIS, "", data)])
    assert [(x.step, x.replicate) for x in ms] == [("probe", 1), ("probe", 2), ("conc_1", 1)]


# ── Persistance des biomarqueurs ─────────────────────────────────────────────

def test_user_data_dir_is_outside_the_launcher_app_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("EIS_USER_DATA_DIR", raising=False)
    launcher = tmp_path / "EIS"
    app = launcher / "eis_app"
    app.mkdir(parents=True)
    assert user_data_dir(app) == launcher / "user_data"          # à côté de launch.bat
    assert user_data_dir(tmp_path / "clone") == tmp_path / "clone" / "user_data"   # repli dev
    monkeypatch.setenv("EIS_USER_DATA_DIR", str(tmp_path / "forced"))
    assert user_data_dir(app) == tmp_path / "forced"


def test_biomarkers_persist_and_survive_an_app_update(tmp_path):
    """Les noms survivent à un « redémarrage » (relecture) et à un remplacement de eis_app/."""
    launcher = tmp_path / "EIS"
    app = launcher / "eis_app"
    app.mkdir(parents=True)
    d = user_data_dir(app)
    assert load_biomarkers(d) == []
    assert add_biomarkers(["miR-21", " ARN-X ", "mir-21", ""], d) == ["miR-21", "ARN-X"]
    # Mise à jour simulée de launch.bat : eis_app est remplacé de fond en comble.
    import shutil
    shutil.rmtree(app)
    (launcher / "eis_app").mkdir()
    assert load_biomarkers(user_data_dir(launcher / "eis_app")) == ["miR-21", "ARN-X"]
    assert add_biomarkers(["CEA"], d) == ["miR-21", "ARN-X", "CEA"]
    raw = json.loads(biomarkers_path(d).read_text(encoding="utf-8"))
    assert raw == {"version": 1, "biomarkers": ["miR-21", "ARN-X", "CEA"]}


def test_corrupted_biomarkers_file_gives_an_empty_list(tmp_path):
    biomarkers_path(tmp_path).write_text("{pas du json", encoding="utf-8")
    assert load_biomarkers(tmp_path) == []
    assert save_biomarkers(["A"], tmp_path) == ["A"]
    assert load_biomarkers(tmp_path) == ["A"]
