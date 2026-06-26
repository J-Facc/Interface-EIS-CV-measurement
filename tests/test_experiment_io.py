"""Tests for core/experiment_io.py — apply_point_exclusions (Bug 3)."""

import io

import numpy as np

from core.experiment_io import apply_point_exclusions
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
