"""Chargeur CV (core/cv_loader.py) : formats, unités, ordre de mesure, branches, cycles."""

import numpy as np
import pytest

from core.cv_loader import load_cv_measurement, split_branches

CSV_BASIC = b"Ewe,I\n-0.5,-1e-6\n0.0,0.0\n0.5,1e-6\n-0.2,-2e-7\n0.1,3e-7\n"
CSV_MA = b"E/V,I/mA\n-0.5,-0.001\n0.0,0.0\n0.5,0.001\n0.2,0.0002\n0.1,0.0001\n"
CSV_UA = b"potential,current/uA\n-0.5,-1.0\n0.0,0.0\n0.5,1.0\n0.2,0.2\n0.1,0.1\n"
CSV_SEMICOLON = b"Ewe;I\n-0.5;-1e-6\n0.0;0.0\n0.5;1e-6\n0.2;2e-7\n0.1;1e-7\n"
CSV_TAB = b"Ewe\tI\n-0.5\t-1e-6\n0.0\t0.0\n0.5\t1e-6\n0.2\t2e-7\n0.1\t1e-7\n"

_N_BRANCH = 120


def _loop_arrays():
    """Boucle aller (pic anodique à 0,25 V) puis retour (pic cathodique à 0,19 V)."""
    fwd = np.linspace(-0.2, 0.6, _N_BRANCH)
    ret = np.linspace(0.6, -0.2, _N_BRANCH)
    i_f = 5e-6 * np.exp(-((fwd - 0.25) / 0.05) ** 2)
    i_r = -4e-6 * np.exp(-((ret - 0.19) / 0.05) ** 2)
    return np.concatenate([fwd, ret]), np.concatenate([i_f, i_r])


def _loop_file(E, I, cycles=None) -> bytes:
    head = "Ewe/V\t<I>/mA" + ("\tcycle number" if cycles is not None else "")
    rows = [f"{e:.6f}\t{i * 1e3:.9f}" + (f"\t{c}" if cycles is not None else "")
            for e, i, c in zip(E, I, cycles if cycles is not None else [None] * len(E))]
    return ("\n".join([head] + rows)).encode()


@pytest.mark.parametrize("data", [CSV_BASIC, CSV_SEMICOLON, CSV_TAB])
def test_separators(data):
    d = load_cv_measurement(data, "x")
    assert len(d["E"]) == 5 and d["E"][0] == pytest.approx(-0.5) and d["I"][0] == pytest.approx(-1e-6)


@pytest.mark.parametrize("data", [CSV_MA, CSV_UA])
def test_current_units_are_converted_to_ampere(data):
    d = load_cv_measurement(data, "x")
    assert d["I"][2] == pytest.approx(1e-6)


def test_measurement_order_is_kept_never_sorted():
    E, I = _loop_arrays()
    d = load_cv_measurement(_loop_file(E, I), "loop")
    np.testing.assert_allclose(d["E"], E, atol=1e-6)
    np.testing.assert_allclose(d["I"], I, atol=1e-9)
    assert not np.all(np.diff(d["E"]) >= 0)
    unsorted = b"Ewe,I\n0.5,1e-6\n-0.5,-1e-6\n0.0,0.0\n0.2,1e-7\n0.1,0\n"
    assert list(load_cv_measurement(unsorted, "u")["E"]) == [0.5, -0.5, 0.0, 0.2, 0.1]


def test_split_branches_loop_and_sweep():
    E, _ = _loop_arrays()
    assert len(split_branches(E)) == 2
    assert len(split_branches(E[:_N_BRANCH])) == 1
    assert len(split_branches(np.concatenate([E, E]))) == 4
    noisy = E[:_N_BRANCH] + np.random.default_rng(1).normal(0, 1e-3, _N_BRANCH)
    assert len(split_branches(noisy)) == 1


def test_cycle_selection_with_and_without_cycle_column():
    E, I = _loop_arrays()
    E2, I2 = np.concatenate([E, E]), np.concatenate([I, 2 * I])
    cyc = [1] * len(E) + [2] * len(E)
    last = load_cv_measurement(_loop_file(E2, I2, cyc), "c")
    assert last["cycles"] == [1, 2] and last["cycle_kept"] == 2
    np.testing.assert_allclose(last["I"], 2 * I, atol=1e-9)
    first = load_cv_measurement(_loop_file(E2, I2, cyc), "c", "first")
    np.testing.assert_allclose(first["I"], I, atol=1e-9)
    no_col = load_cv_measurement(_loop_file(E2, I2), "n")
    assert no_col["cycles"] == [1] and len(no_col["E"]) == len(E2)
    assert any("plusieurs cycles" in w for w in no_col["warnings"])


@pytest.mark.parametrize("data, fragment", [
    (b"voltage_x,current\n0.0,0.0\n", "introuvables"),
    (b"Ewe,amp\n0.0,0.0\n", "introuvables"),
    (b"Ewe,I\n0,0\n1,1\n", "moins de 5 points"),
])
def test_invalid_files(data, fragment):
    with pytest.raises(ValueError, match=fragment):
        load_cv_measurement(data, "bad")
