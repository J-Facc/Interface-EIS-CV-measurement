"""Pages v3 rendues sans navigateur (Streamlit AppTest) : aucune exception, flux de base."""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _app(body: str) -> AppTest:
    script = f"import sys; sys.path.insert(0, {str(REPO)!r})\n" + body
    return AppTest.from_string(script, default_timeout=120)


@pytest.fixture(autouse=True)
def _isolated_user_data(tmp_path, monkeypatch):
    monkeypatch.setenv("EIS_USER_DATA_DIR", str(tmp_path / "user_data"))
    monkeypatch.chdir(REPO)


def test_full_app_starts_on_configuration_page():
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Configuration" in t.value for t in at.title)


def test_configuration_page_validates_and_persists_biomarkers(tmp_path):
    at = _app("from ui import state as S, page_config\nimport streamlit as st\n"
              "S.init_state(st.session_state)\npage_config.render()\n")
    at.run()
    assert not at.exception
    # Liste vide : le menu propose « + Ajouter… » et un champ de saisie.
    at.text_input(key="cfg_bio_new_0").input("miR-21").run()
    assert not at.exception
    from core.user_data import load_biomarkers
    assert load_biomarkers(Path(os.environ["EIS_USER_DATA_DIR"])) == ["miR-21"]
    assert at.selectbox(key="cfg_ch_bio_sel_0").value == "miR-21"
    at.button[0].click().run()          # « Valider la configuration »
    assert not at.exception
    exp = at.session_state["experiment"]
    assert exp.biomarkers == ["miR-21"] and exp.channels[0].concentrations == [1e-12]
    assert any("validée" in s.value for s in at.success)
