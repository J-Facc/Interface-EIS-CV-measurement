"""Moyennes de réplicats pour l'affichage (core/display.py) et cache du nettoyage."""

import numpy as np
import pytest

from core.assignment import AssignmentRow, clean_files
from core.display import channel_mean_view, mean_cv, mean_eis, mean_groups, synthesis_view
from core.experiment import Channel, Experiment, Measurement
from tests.synthetic_data import make_config, randles_file


def _eis(ch, el, step, rep, scale):
    f = np.logspace(5, -1, 30)
    return Measurement(ch, el, step, rep, "EIS", "x", {"f": f, "Zre": scale * np.ones(30), "Zim": 2 * scale * np.ones(30)})


def _cv(ch, el, step, rep, scale, n=50):
    E = np.linspace(-0.2, 0.2, n)
    return Measurement(ch, el, step, rep, "CV", "x", {"E": E, "I": scale * E, "cycle": np.ones(n)})


def test_mean_eis_same_grid_and_interpolated():
    f = np.logspace(5, -1, 20)
    m = mean_eis([(f, np.full(20, 1.0), np.full(20, 2.0)), (f, np.full(20, 3.0), np.full(20, 4.0))])
    assert np.allclose(m["Zre"], 2.0) and np.allclose(m["sd_re"], np.std([1, 3], ddof=1)) and m["n"] == 2
    assert np.allclose(m["mod"], np.mean([np.hypot(1, 2), np.hypot(3, 4)]))
    f2 = np.logspace(4, -1, 15)
    m = mean_eis([(f, np.log10(f), np.ones(20)), (f2, np.log10(f2), np.ones(15))])
    assert m["f"].max() <= 1e4 * 1.0001 and np.allclose(m["Zre"], np.log10(m["f"]), atol=1e-9)
    assert np.allclose(mean_eis([(f, np.ones(20), np.ones(20))])["sd_re"], 0)


def test_mean_cv_same_length_and_resampled():
    E = np.linspace(0, 1, 10)
    m = mean_cv([(E, E), (E, 3 * E)])
    assert np.allclose(m["I"], 2 * E) and np.allclose(m["sd"], np.std([E, 3 * E], axis=0, ddof=1))
    m = mean_cv([(E, E), (np.linspace(0, 1, 20), np.linspace(0, 1, 20))])
    assert len(m["E"]) == 10 and np.allclose(m["I"], E)


@pytest.fixture
def data():
    exp = Experiment(["A", "B"], [Channel(1, "A", [1e-9]), Channel(2, "B", [1e-9])])
    ms = [_eis(1, el, st, r, el * 100 + r) for el in (1, 2) for st in ("probe", "conc_1") for r in (1, 2)]
    ms += [_cv(1, el, "probe", r, el + r) for el in (1, 2) for r in (1, 2)] + [_eis(2, 1, "probe", 1, 7)]
    return exp, ms


def test_channel_view_has_e1_e2_and_both(data):
    exp, ms = data
    items = channel_mean_view(ms, exp, "EIS", 1)
    assert [i["label"] for i in items] == ["Probe — E1", "conc_1 — 1.00e-09 M — E1", "Probe — E2",
                                           "conc_1 — 1.00e-09 M — E2", "Probe — E1+E2", "conc_1 — 1.00e-09 M — E1+E2"]
    both = items[4]["mean"]
    assert both["n"] == 4 and np.allclose(both["Zre"], np.mean([101, 102, 201, 202]))
    assert items[0]["mean"]["n"] == 2 and np.allclose(items[0]["mean"]["Zre"], 101.5)
    assert {i["dash"] for i in items} == {"dash", "dot", "solid"}


def test_synthesis_pools_both_electrodes_with_band(data):
    exp, ms = data
    items = synthesis_view(ms, exp, "EIS")
    assert [i["label"] for i in items] == ["C1 A · Probe", "C1 A · conc_1 — 1.00e-09 M", "C2 B · Probe"]
    assert all(i["band"] for i in items) and items[0]["mean"]["n"] == 4
    assert [i["label"] for i in synthesis_view(ms, exp, "EIS", ["B"])] == ["C2 B · Probe"]
    cv = synthesis_view(ms, exp, "CV")
    assert len(cv) == 1 and cv[0]["mean"]["n"] == 4
    assert mean_groups(ms, exp, "CV", 1, (1,), replicates=[2])[0]["mean"]["n"] == 1


def test_cleaning_reuses_the_cache():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    rows = [AssignmentRow("a", "a.txt", "EIS", 1, 1, "probe", 1, detected_kind="EIS")]
    contents = {"a": randles_file(1500.0)}
    cache = {}
    ms, _ = clean_files(rows, contents, exp, make_config(), {"a": [0, 1]}, cache)
    assert "a" in cache and len(ms[0].data["f"]) == len(cache["a"][0]["f"]) - 2
    ms2, _ = clean_files(rows, {"a": b"illisible"}, exp, make_config(), {}, cache)   # lu depuis le cache
    assert len(ms2[0].data["f"]) == len(cache["a"][0]["f"])
