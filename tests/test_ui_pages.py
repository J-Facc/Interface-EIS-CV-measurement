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
    at = _app("from core import app_state as S\nfrom ui import page_config\nimport streamlit as st\n"
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


def _uploaded(name, data):
    class _F:
        def __init__(self):
            self.name = name
        def getvalue(self):
            return data
    return _F()


def test_import_page_assigns_checks_and_cleans():
    """Page Import : dépôt → assignation pré-remplie → contrôle → nettoyage → mesures."""
    from tests.synthetic_data import randles_file
    body = (
        "import streamlit as st\nfrom core import app_state as S\nfrom ui import page_import\n"
        "from core.experiment import Experiment, Channel\n"
        "S.init_state(st.session_state)\n"
        "if st.session_state[S.EXPERIMENT] is None:\n"
        "    st.session_state[S.EXPERIMENT] = Experiment(['A'], [Channel(1, 'A', [1e-9])])\n"
        "page_import.render()\n"
    )
    at = _app(body)
    at.run()
    assert not at.exception
    # Simule le dépôt : on passe par la fonction d'ingestion avec de faux fichiers.
    import ui.page_import as pi
    files = [_uploaded("probe_r1.txt", randles_file(1500.0, seed=1)),
             _uploaded("probe_r2.txt", randles_file(1500.0, seed=2)),
             _uploaded("conc_1_r1.txt", randles_file(2500.0, seed=3)),
             _uploaded("cv_probe.txt", (REPO / "tests" / "data" / "cv_bare_2cycles.txt").read_bytes())]
    state = {}
    from core import app_state as S
    S.init_state(state)
    from core.experiment import Experiment, Channel
    state[S.EXPERIMENT] = Experiment(['A'], [Channel(1, 'A', [1e-9])])
    import streamlit as st
    real = st.session_state
    try:
        pi.st.session_state = state      # l'ingestion n'utilise que session_state
        assert pi._ingest(files) == []
    finally:
        pi.st.session_state = real
    rows = state[S.ASSIGNMENT]
    assert [(r["kind"], r["step"], r["replicate"]) for r in rows] == [
        ("EIS", "probe", 1), ("EIS", "probe", 2), ("EIS", "conc_1", 1), ("CV", "probe", 1)]
    for k in (S.RAW_FILES, S.ASSIGNMENT):
        at.session_state[k] = state[k]
    at.run()
    assert not at.exception and not at.error
    clean = next(b for b in at.button if "Nettoyer" in b.label)
    clean.click().run()
    assert not at.exception
    ms = at.session_state[S.MEASUREMENTS]
    assert len(ms) == 4 and {m.kind for m in ms} == {"EIS", "CV"}


def _measurements():
    """Mesures nettoyées synthétiques : canal 1, 2 électrodes, probe + conc_1, EIS + CV."""
    from core.assignment import AssignmentRow, clean_files
    from core.experiment import Channel, Experiment
    from tests.synthetic_data import make_config, randles_file
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    cv = (REPO / "tests" / "data" / "cv_bare_2cycles.txt").read_bytes()
    rows, contents = [], {}
    for el in (1, 2):
        for step, rct in (("probe", 1500.0), ("conc_1", 2500.0)):
            for rep in (1, 2):
                fid = f"e{el}{step}{rep}"
                contents[fid] = randles_file(rct, seed=el * 10 + rep)
                rows.append(AssignmentRow(fid, fid, "EIS", 1, el, step, rep, detected_kind="EIS"))
                contents[fid + "cv"] = cv
                rows.append(AssignmentRow(fid + "cv", fid + "cv", "CV", 1, el, step, rep, detected_kind="CV"))
    ms, issues = clean_files(rows, contents, exp, make_config())
    assert ms, issues
    return exp, ms


def test_viz_page_synthesis_by_default_then_electrode_view():
    exp, ms = _measurements()
    at = _app("import streamlit as st\nfrom core import app_state as S\nfrom ui import page_viz\n"
              "S.init_state(st.session_state)\npage_viz.render()\n")
    at.session_state["experiment"] = exp
    at.session_state["measurements"] = ms
    at.run()
    assert not at.exception
    sel = at.selectbox(key="viz_electrode")
    assert sel.value == "synthese" and sel.options[1:] == ["Canal 1 — A"]
    assert len(at.get("plotly_chart")) == 2                            # CV + EIS (moyennes par électrode)
    sel.set_value("1").run()
    assert not at.exception
    assert at.multiselect(key="viz_steps_1").value == ["probe", "conc_1"]
    assert len(at.get("plotly_chart")) == 6                            # (E1, E2, les deux) × (CV, EIS)
    at.radio(key="viz_eis_mode").set_value("Bode").run()
    assert not at.exception
    assert not any("normalis" in (t.label or "").lower() for t in at.toggle)


def test_viz_page_imports_no_fit_and_no_drt():
    """Critère de la phase 3 : la page Visualisation (et sa mise en forme) n'importe aucun fit ni DRT."""
    import ast
    for path in (REPO / "ui" / "page_viz.py", REPO / "core" / "display.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        bad = {m for m in mods if m and (m.startswith(("fits", "drt", "core.pipeline", "core.analysis",
                                                       "core.measurement_model", "core.drt")))}
        assert not bad, (path.name, bad)


def test_display_groups_are_ordered_and_filtered():
    from core.display import cv_display_groups, eis_display_groups
    exp, ms = _measurements()
    g = eis_display_groups(ms, exp, 1, 2, steps=["conc_1", "probe"], replicates=[2])
    assert [x.label for x in g] == ["Probe", "conc_1 — 1.00e-09 M"]
    assert [len(x.replicates) for x in g] == [1, 1] and g[0].mean is None
    c = cv_display_groups(ms, exp, 1, 1)
    assert [x["step"] for x in c] == ["probe", "conc_1"] and len(c[0]["curves"]) == 2


def test_analysis_page_runs_in_background_then_shows_tabs(monkeypatch):
    """Lancer l'analyse → progression → onglets DRT / Rct (exécuteur synchrone, sans DRT)."""
    monkeypatch.setenv("EIS_EXECUTOR", "sync")
    exp, ms = _measurements()
    at = _app("import streamlit as st\nfrom core import app_state as S\nfrom ui import page_analysis\nS.init_state(st.session_state)\n"
              "page_analysis.render()\n")
    at.session_state["experiment"] = exp
    at.session_state["measurements"] = ms
    at.run()
    assert not at.exception
    at.checkbox(key="an_drt_on").uncheck().run()
    launch = next(b for b in at.button if "Lancer" in b.label)
    launch.click().run()
    assert not at.exception
    for _ in range(5):                       # le fragment de progression récolte puis relance l'app
        if at.session_state["results"] is not None:
            break
        at.run()
    res = at.session_state["results"]
    assert res is not None and len(res.eis.groups) == 4
    assert not at.exception
    assert [t.label for t in at.tabs][:5] == ["DRT", "Rct (fit Randles)", "Étalonnage EIS", "CV", "Étalonnage CV"]
    # Les onglets d'étalonnage sont rendus (2 réplicats/étape → moyennes ± écart-type).
    assert len(at.get("plotly_chart")) >= 4      # DRT, Rct, étalonnage EIS, CV, étalonnage CV
    assert len(res.cv) == 8


def test_export_page_generates_the_three_files():
    from tests.v3_data import build_session
    exp, ms, res = build_session()
    at = _app("import streamlit as st\nfrom core import app_state as S\nfrom ui import page_export\nS.init_state(st.session_state)\n"
              "page_export.render()\n")
    at.session_state["experiment"] = exp
    at.session_state["measurements"] = ms
    at.session_state["results"] = res
    at.run()
    assert not at.exception
    for kind in ("zip", "pdf", "html"):
        at.button(key=f"gen_{kind}").click().run()
        assert not at.exception and not at.error, kind
    cache = at.session_state["_export_cache"]
    assert set(cache) == {"zip", "pdf", "html"}
    assert cache["pdf"][1].startswith(b"%PDF") and cache["zip"][1][:2] == b"PK"


def test_archive_reload_shows_results_without_recomputing(monkeypatch):
    """Archive v3 chargée (page Configuration) → page Analyse affichée sans aucun calcul."""
    from tests.v3_data import build_session
    from exports.archive import read_archive, write_archive
    from tests.synthetic_data import make_config
    exp, ms, res = build_session()
    exp2, ms2, res2 = read_archive(write_archive(exp, ms, res, make_config()))
    import core.analysis

    def forbidden(*a, **k):
        raise AssertionError("recalcul interdit")
    monkeypatch.setattr(core.analysis.AnalysisRun, "__init__", forbidden)
    at = _app("import streamlit as st\nfrom core import app_state as S\nfrom ui import page_analysis\nS.init_state(st.session_state)\n"
              "page_analysis.render()\n")
    from core import app_state as S
    state = {}
    S.init_state(state)
    S.load_archive_state(state, exp2, ms2, res2)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception
    assert any("archive" in i.value for i in at.info)
    assert [t.label for t in at.tabs][:5] == ["DRT", "Rct (fit Randles)", "Étalonnage EIS", "CV", "Étalonnage CV"]


def test_import_page_with_one_folder_per_channel():
    """Dossier d'un canal (arborescence du poste de mesure) → tableau rempli → nettoyage."""
    from tests.test_folder_import import _tree
    import ui.page_import as pi
    from core import app_state as S
    from core.experiment import Experiment, Channel
    state = {}
    S.init_state(state)
    state[S.EXPERIMENT] = Experiment(["A"], [Channel(1, "A", [1e-16, 1e-14, 1e-12])])
    import streamlit as st
    real = st.session_state
    try:
        pi.st.session_state = state
        pi._ingest_folder(1, [_uploaded(n, b) for n, b in _tree(electrodes=(1,), reps=(1, 2))])
    finally:
        pi.st.session_state = real
    assert len(state[S.ASSIGNMENT]) == 2 * 2 * 4 and state["folder_issues"][1] == []
    at = _app("import streamlit as st\nfrom core import app_state as S\nfrom ui import page_import\n"
              "S.init_state(st.session_state)\npage_import.render()\n")
    for k in (S.EXPERIMENT, S.RAW_FILES, S.ASSIGNMENT, "folder_issues", "folder_concentrations"):
        at.session_state[k] = state[k]
    at.run()
    assert not at.exception and not at.error
    # Revue graphe par graphe : « Valider » sans sélection passe au suivant, Précédent/Suivant naviguent.
    at.button(key="imp_walk_ok").click().run()
    assert at.session_state["imp_walk_idx"] == 1 and not at.exception
    at.button(key="imp_walk_next").click().run()
    assert at.session_state["imp_walk_idx"] == 2
    at.button(key="imp_walk_prev").click().run()
    assert at.session_state["imp_walk_idx"] == 1 and not at.exception
    next(b for b in at.button if "Nettoyer" in b.label).click().run()
    assert not at.exception
    assert len(at.session_state[S.MEASUREMENTS]) == 16
