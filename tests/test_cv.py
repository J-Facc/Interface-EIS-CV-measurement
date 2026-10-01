"""Tests for CV loader, pipeline, and plot functions."""

import numpy as np
import pytest

from core.cv_loader import (
    average_cv_replicates,
    interp_on_reference,
    load_cv_file,
    split_branches,
)
from core.cv_pipeline import run_cv_pipeline
from core.cv_models import CVSession


# ── Fixtures ──────────────────────────────────────────────────────────────────

CSV_BASIC = b"Ewe,I\n-0.5,-1e-6\n0.0,0.0\n0.5,1e-6\n"
CSV_MA = b"E/V,I/mA\n-0.5,-0.001\n0.0,0.0\n0.5,0.001\n"
CSV_UA = b"potential,current/uA\n-0.5,-1.0\n0.0,0.0\n0.5,1.0\n"
CSV_SEMICOLON = b"Ewe;I\n-0.5;-1e-6\n0.0;0.0\n0.5;1e-6\n"
CSV_TAB = b"Ewe\tI\n-0.5\t-1e-6\n0.0\t0.0\n0.5\t1e-6\n"


# ── load_cv_file ──────────────────────────────────────────────────────────────

def test_load_basic_csv():
    scan = load_cv_file(CSV_BASIC, "test", 1e-9, "probe")
    assert len(scan.E) == 3
    assert scan.E[0] == pytest.approx(-0.5)
    assert scan.I[0] == pytest.approx(-1e-6)


def test_load_milliamp_conversion():
    scan = load_cv_file(CSV_MA, "test_ma", 1e-9, "probe")
    assert scan.I[2] == pytest.approx(1e-6)


def test_load_microamp_conversion():
    scan = load_cv_file(CSV_UA, "test_ua", 1e-9, "probe")
    assert scan.I[2] == pytest.approx(1e-6)


def test_load_semicolon_separator():
    scan = load_cv_file(CSV_SEMICOLON, "test_semi", 0.0, "probe")
    assert len(scan.E) == 3


def test_load_tab_separator():
    scan = load_cv_file(CSV_TAB, "test_tab", 0.0, "probe")
    assert len(scan.E) == 3


def test_sorted_by_potential():
    csv = b"Ewe,I\n0.5,1e-6\n-0.5,-1e-6\n0.0,0.0\n"
    scan = load_cv_file(csv, "unsorted", 0.0, "probe")
    assert scan.E[0] < scan.E[1] < scan.E[2]


# ── CV-1 : boucle voltammétrique (AUDIT.md, Annexe A.6) ───────────────────────

_N_BRANCH = 200


def _loop_arrays():
    """Boucle synthétique de l'Annexe A.6 : aller pic +5 µA à 0,25 V, retour
    pic −4 µA à 0,19 V."""
    n = _N_BRANCH
    E = np.concatenate([np.linspace(-0.2, 0.6, n), np.linspace(0.6, -0.2, n)])

    def peak(x, e0, w):
        return np.exp(-(((x - e0) / w) ** 2))

    I = np.concatenate([
        5e-6 * peak(E[:n], 0.25, 0.06) + 1e-7 * E[:n],
        -4e-6 * peak(E[n:], 0.19, 0.06) - 1e-7 * E[n:],
    ])
    return E, I


def _loop_file(E, I) -> bytes:
    txt = "Ewe/V\t<I>/mA\n" + "\n".join(f"{e:.6f}\t{i * 1e3:.9f}" for e, i in zip(E, I))
    return txt.encode()


def _sign_changes(y) -> int:
    d = np.diff(y)
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def test_loop_keeps_measurement_order_not_sorted():
    E, I = _loop_arrays()
    scan = load_cv_file(_loop_file(E, I), "loop", 0.0, "probe")
    n = _N_BRANCH
    assert len(scan.E) == 2 * n
    # Pas de tri global : E n'est PAS monotone, il monte puis redescend.
    assert not np.all(np.diff(scan.E) >= 0)
    np.testing.assert_allclose(scan.E, E, atol=1e-6)
    np.testing.assert_allclose(scan.I, I, atol=1e-9)
    assert np.all(np.diff(scan.E[:n]) > 0)   # branche aller
    assert np.all(np.diff(scan.E[n:]) < 0)   # branche retour


def test_loop_branches_stay_coherent_no_interleaving():
    E, I = _loop_arrays()
    scan = load_cv_file(_loop_file(E, I), "loop", 0.0, "probe")
    n = _N_BRANCH
    # Avant correctif : 223 changements de signe de dI/dE (≈ 4 attendus).
    assert _sign_changes(scan.I) <= 6
    # Chaque branche garde son propre pic, avec le bon signe et au bon potentiel.
    fwd_E, fwd_I = scan.E[:n], scan.I[:n]
    ret_E, ret_I = scan.E[n:], scan.I[n:]
    assert fwd_I.max() == pytest.approx(5e-6, rel=0.05)
    assert fwd_E[np.argmax(fwd_I)] == pytest.approx(0.25, abs=0.01)
    assert ret_I.min() == pytest.approx(-4e-6, rel=0.05)
    assert ret_E[np.argmin(ret_I)] == pytest.approx(0.19, abs=0.01)
    assert fwd_I.min() > -1e-6 and ret_I.max() < 1e-6


def test_simple_sweep_is_still_sorted():
    # Balayage simple désordonné (pas de cycle) : comportement historique.
    E = np.linspace(-0.2, 0.6, 40)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(E))
    csv = ("Ewe,I\n" + "\n".join(f"{E[i]},{1e-6 * E[i]}" for i in perm)).encode()
    scan = load_cv_file(csv, "sweep", 0.0, "probe")
    assert np.all(np.diff(scan.E) >= 0)
    np.testing.assert_allclose(scan.I, 1e-6 * scan.E)


def test_split_branches_loop_and_sweep():
    E, _ = _loop_arrays()
    assert len(split_branches(E)) == 2
    assert len(split_branches(E[:_N_BRANCH])) == 1
    # Deux cycles : aller, retour, aller, retour.
    assert len(split_branches(np.concatenate([E, E]))) == 4
    # Bruit de mesure < 5 % de l'étendue : pas de faux rebroussement.
    noisy = E[:_N_BRANCH] + np.random.default_rng(1).normal(0, 1e-3, _N_BRANCH)
    assert len(split_branches(noisy)) == 1


def test_interp_on_reference_matches_branches():
    E, I = _loop_arrays()
    scan = load_cv_file(_loop_file(E, I), "loop", 0.0, "probe")
    # Rééchantillonner la boucle sur elle-même doit la restituer (np.interp
    # global sur un E non monotone ne le ferait pas).
    np.testing.assert_allclose(interp_on_reference(scan.E, scan.E, scan.I), scan.I, atol=1e-9)
    # Une boucle ne s'interpole pas sur un balayage simple.
    with pytest.raises(ValueError, match="branches"):
        interp_on_reference(scan.E, scan.E[:_N_BRANCH], scan.I[:_N_BRANCH])


def test_loop_average_and_pipeline_keep_loop():
    E, I = _loop_arrays()
    data = _loop_file(E, I)
    data2 = _loop_file(E, 2 * I)
    s1 = load_cv_file(data, "r1", 0.0, "probe")
    s2 = load_cv_file(data2, "r2", 0.0, "probe")
    avg = average_cv_replicates([s1, s2])
    np.testing.assert_allclose(avg.I, 1.5 * I, atol=1e-9)

    cv = run_cv_pipeline([
        _make_assignment(data, "probe", 0.0),
        _make_assignment(_loop_file(E, 0.5 * I), "hybridization", 1e-9),
    ])
    delta = cv.groups[0].delta_signal
    # |I_probe − 0,5·I_probe| / |I_probe| = 0,5 là où le courant est non nul,
    # sur les deux branches (un entrelacement donnerait n'importe quoi).
    big = np.abs(I) > 1e-7
    np.testing.assert_allclose(delta[big], 0.5, atol=1e-4)  # arrondi du fichier


def test_missing_e_column_raises():
    csv = b"voltage_x,current\n0.0,0.0\n"
    with pytest.raises(ValueError, match="Colonne potentiel"):
        load_cv_file(csv, "bad", 0.0, "probe")


def test_missing_i_column_raises():
    csv = b"Ewe,amp\n0.0,0.0\n"
    with pytest.raises(ValueError, match="Colonne courant"):
        load_cv_file(csv, "bad", 0.0, "probe")


# ── average_cv_replicates ─────────────────────────────────────────────────────

def test_average_single_returns_same():
    scan = load_cv_file(CSV_BASIC, "s", 0.0, "probe")
    avg = average_cv_replicates([scan])
    np.testing.assert_array_equal(avg.E, scan.E)


def test_average_two_scans():
    s1 = load_cv_file(CSV_BASIC, "s1", 0.0, "probe")
    csv2 = b"Ewe,I\n-0.5,-3e-6\n0.0,0.0\n0.5,3e-6\n"
    s2 = load_cv_file(csv2, "s2", 0.0, "probe")
    avg = average_cv_replicates([s1, s2])
    assert avg.I[2] == pytest.approx(2e-6)


# ── run_cv_pipeline ───────────────────────────────────────────────────────────

def _make_assignment(csv: bytes, step: str, conc: float, name: str = "f.csv") -> dict:
    return {"content": csv, "filename": name, "step": step, "concentration": conc}


def test_pipeline_returns_cv_session():
    assignments = [
        _make_assignment(CSV_BASIC, "probe", 0.0),
        _make_assignment(CSV_BASIC, "hybridization", 1e-9),
        _make_assignment(CSV_BASIC, "hybridization", 1e-8),
    ]
    cv = run_cv_pipeline(assignments)
    assert isinstance(cv, CVSession)
    assert cv.probe is not None
    assert len(cv.groups) == 2


def test_pipeline_delta_signal_shape():
    probe_csv = b"Ewe,I\n-0.5,-2e-6\n0.0,-1e-6\n0.5,0.0\n"
    hyb_csv = b"Ewe,I\n-0.5,-1e-6\n0.0,-0.5e-6\n0.5,0.5e-6\n"
    assignments = [
        _make_assignment(probe_csv, "probe", 0.0),
        _make_assignment(hyb_csv, "hybridization", 1e-9),
    ]
    cv = run_cv_pipeline(assignments)
    assert cv.groups[0].delta_signal.shape == cv.groups[0].scan.E.shape


def test_pipeline_no_probe():
    assignments = [
        _make_assignment(CSV_BASIC, "hybridization", 1e-9),
    ]
    cv = run_cv_pipeline(assignments)
    assert cv.probe is None
    assert np.all(np.isnan(cv.groups[0].delta_signal))


# ── cv_plots (import only — no display) ──────────────────────────────────────

def test_cv_current_figure_importable():
    from plotting.cv_plots import cv_current_figure, cv_calibration_figure_multi
    assignments = [
        _make_assignment(CSV_BASIC, "probe", 0.0),
        _make_assignment(b"Ewe,I\n-0.5,-2e-6\n0.0,0.0\n0.5,2e-6\n", "hybridization", 1e-9),
        _make_assignment(b"Ewe,I\n-0.5,-3e-6\n0.0,0.0\n0.5,3e-6\n", "hybridization", 1e-8),
    ]
    cv = run_cv_pipeline(assignments)
    fig1 = cv_current_figure(cv)
    fig2 = cv_calibration_figure_multi({1: cv})
    assert fig1 is not None
    assert fig2 is not None
