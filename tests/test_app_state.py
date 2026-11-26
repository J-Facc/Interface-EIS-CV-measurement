"""Clés partagées v3 et invalidations (core/app_state.py) — jamais de résultat périmé affiché."""

from pathlib import Path

from core import app_state as S


class _Run:
    cancelled = False

    def cancel(self):
        self.cancelled = True


def _state():
    st = {}
    S.init_state(st)
    return st


def test_init_does_not_overwrite_and_creates_fresh_containers():
    a, b = _state(), _state()
    a[S.RAW_FILES]["x"] = 1
    assert b[S.RAW_FILES] == {}
    a[S.THEME] = "dark"
    S.init_state(a)
    assert a[S.THEME] == "dark" and a[S.RAW_FILES] == {"x": 1}


def test_new_experiment_invalidates_measurements_results_and_running_analysis():
    st = _state()
    run = _Run()
    st.update({S.MEASUREMENTS: ["m"], S.RESULTS: "r", S.RESULTS_SOURCE: "computed", S.ANALYSIS_RUN: run,
               S.RAW_FILES: {"f": {}}, S.ASSIGNMENT: [{"file_id": "f"}]})
    S.new_experiment(st, "exp")
    assert st[S.EXPERIMENT] == "exp" and st[S.MEASUREMENTS] is None and st[S.RESULTS] is None
    assert run.cancelled and st[S.ANALYSIS_RUN] is None
    assert st[S.RAW_FILES] and st[S.ASSIGNMENT]          # fichiers déposés conservés


def test_archive_load_replaces_everything():
    st = _state()
    st.update({S.RAW_FILES: {"f": {}}, S.ASSIGNMENT: [1], S.EXCLUDED_POINTS: {"f": [1]}})
    S.load_archive_state(st, "e", ["m"], "r")
    assert (st[S.EXPERIMENT], st[S.MEASUREMENTS], st[S.RESULTS]) == ("e", ["m"], "r")
    assert st[S.RESULTS_SOURCE] == "archive" and st[S.RAW_FILES] == {} and st[S.EXCLUDED_POINTS] == {}
    assert S.has_results(st) and S.has_measurements(st) and S.has_experiment(st)


def test_launcher_sentinel_files_still_exist():
    """launch.bat refuse une mise à jour dont il manque l'un de ces fichiers (ligne « for %%F in »)."""
    repo = Path(__file__).resolve().parent.parent
    line = next(l for l in (repo / "launch.bat").read_text().splitlines() if "for %%F in (app.py" in l)
    names = line.split("(", 1)[1].split(")", 1)[0].split()
    for n in names:
        if "!" in n:
            continue
        assert (repo / n.replace("\\", "/")).exists(), n
