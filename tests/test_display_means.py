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


# ── Normalisation par le probe de la même électrode ──────────────────────────

def _cvpk(ch, el, step, rep, peak, n=60):
    """CV dont le pic anodique (branche E croissante) vaut ``peak`` A à E = 0,2 V."""
    E = np.concatenate([np.linspace(0, 0.2, n), np.linspace(0.2, 0, n)])
    I = np.concatenate([np.linspace(0, peak, n), np.linspace(peak, -peak, n)])
    return Measurement(ch, el, step, rep, "CV", "x", {"E": E, "I": I, "cycle": np.ones(2 * n)})


def test_eis_normalized_by_the_probe_of_the_same_electrode(data):
    exp, ms = data
    items = channel_mean_view(ms, exp, "EIS", 1, normalize=True)
    lab = {i["label"]: i["mean"] for i in items}
    # E1 : probe moyen Re = 101,5 ; conc_1 r1/r2 : Re = 101, 102 → |101,5 − Re|/101,5 moyenné
    e1 = lab["conc_1 — 1.00e-09 M — E1"]
    assert np.allclose(e1["Zre"], np.mean([0.5 / 101.5, 0.5 / 101.5]))
    e2 = lab["conc_1 — 1.00e-09 M — E2"]                       # référence = probe de E2 (201,5)
    assert np.allclose(e2["Zre"], 0.5 / 201.5)
    both = lab["conc_1 — 1.00e-09 M — E1+E2"]                  # chaque réplicat par SON électrode
    assert np.allclose(both["Zre"], np.mean([0.5 / 101.5] * 2 + [0.5 / 201.5] * 2))
    assert np.allclose(lab["Probe — E1"]["Zre"], np.mean([0.5 / 101.5] * 2))   # probe → écart à sa moyenne


def test_cv_normalized_by_the_probe_peak_of_the_same_electrode():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    ms = [_cvpk(1, 1, "probe", 1, 4e-6), _cvpk(1, 1, "probe", 2, 6e-6), _cvpk(1, 2, "probe", 1, 10e-6),
          _cvpk(1, 1, "conc_1", 1, 3e-6), _cvpk(1, 2, "conc_1", 1, 8e-6)]
    g = {x["electrodes"]: x for x in mean_groups(ms, exp, "CV", 1, (1, 2), normalize=True, cv_window=(0, 0.25))
         if x["step"] == "conc_1"}
    m = g[(1, 2)]["mean"]
    assert np.max(m["I"]) == pytest.approx(np.mean([3 / 5, 8 / 10]))     # 3 µA/5 µA et 8 µA/10 µA
    one = mean_groups(ms, exp, "CV", 1, (1,), steps=["conc_1"], normalize=True, cv_window=(0, 0.25))[0]
    assert np.max(one["mean"]["I"]) == pytest.approx(0.6)


def test_electrode_without_probe_is_skipped_when_normalizing():
    exp = Experiment(["A"], [Channel(1, "A", [1e-9])])
    ms = [_eis(1, 1, "probe", 1, 100), _eis(1, 1, "conc_1", 1, 110), _eis(1, 2, "conc_1", 1, 300)]
    g = mean_groups(ms, exp, "EIS", 1, normalize=True)
    conc = next(x for x in g if x["step"] == "conc_1")
    assert conc["mean"]["n"] == 1 and conc["skipped"] == [2]


def test_synthesis_step_filter(data):
    exp, ms = data
    steps = lambda show: [i["step"] for i in synthesis_view(ms, exp, "EIS", show=show)]  # noqa: E731
    assert steps("probe") == ["probe", "probe"] and steps("conc") == ["conc_1"] and steps("bare") == []
    assert steps("all") == ["probe", "conc_1", "probe"]
    assert all(i["normalized"] for i in synthesis_view(ms, exp, "EIS", normalize=True))
