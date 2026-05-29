"""Tests for CV loader, pipeline, and plot functions."""

from __future__ import annotations

import numpy as np
import pytest

from core.cv_loader import load_cv_file, average_cv_replicates
from core.cv_models import CVScan, CVSession, CVConcentrationGroup
from core.cv_pipeline import run_cv_pipeline


def _make_csv(E_vals, I_vals, sep=","):
    header = f"Ewe/V{sep}<I>/A\n"
    rows = "".join(f"{e}{sep}{i}\n" for e, i in zip(E_vals, I_vals))
    return header + rows


def _make_ua_csv(E_vals, I_uA_vals, sep=","):
    header = f"Ewe/V{sep}I/uA\n"
    rows = "".join(f"{e}{sep}{i}\n" for e, i in zip(E_vals, I_uA_vals))
    return header + rows


class TestLoadCVFile:
    def test_basic_load(self):
        E = [0.0, 0.1, 0.2, 0.3, 0.4]
        I = [1e-6, 2e-6, 3e-6, 2e-6, 1e-6]
        content = _make_csv(E, I)
        scan = load_cv_file(content, "test.csv", concentration=1e-9, step="probe")
        assert scan.label == "test.csv"
        assert scan.step == "probe"
        assert len(scan.E) == 5
        np.testing.assert_allclose(scan.E, sorted(E))

    def test_ua_conversion(self):
        E = [0.0, 0.1, 0.2, 0.3, 0.4]
        I_uA = [1.0, 2.0, 3.0, 2.0, 1.0]
        content = _make_ua_csv(E, I_uA)
        scan = load_cv_file(content, "test_ua.csv", concentration=0.0, step="probe")
        np.testing.assert_allclose(scan.I, np.array(I_uA) * 1e-6)

    def test_sorted_by_potential(self):
        E = [0.4, 0.2, 0.0, 0.3, 0.1]
        I = [1e-6, 3e-6, 5e-6, 2e-6, 4e-6]
        content = _make_csv(E, I)
        scan = load_cv_file(content, "unsorted.csv", concentration=0.0, step="probe")
        assert np.all(np.diff(scan.E) >= 0)

    def test_too_few_points_raises(self):
        content = "Ewe/V,<I>/A\n0.1,1e-6\n0.2,2e-6\n"
        with pytest.raises(ValueError, match="moins de 3 points"):
            load_cv_file(content, "short.csv", 0.0, "probe")

    def test_bytes_input(self):
        E = [0.0, 0.1, 0.2, 0.3]
        I = [1e-6, 2e-6, 3e-6, 2e-6]
        content = _make_csv(E, I).encode("utf-8")
        scan = load_cv_file(content, "bytes.csv", 0.0, "probe")
        assert len(scan.E) == 4


class TestAverageCVReplicates:
    def test_single_scan_passthrough(self):
        E = np.linspace(0, 1, 10)
        I = np.ones(10) * 1e-6
        sc = CVScan("a", E, I, 0.0, "probe", ["a"])
        result = average_cv_replicates([sc])
        assert result is sc

    def test_average_same_grid(self):
        E = np.linspace(0, 1, 10)
        I1 = np.ones(10) * 1e-6
        I2 = np.ones(10) * 3e-6
        sc1 = CVScan("a", E.copy(), I1, 1e-9, "hybridization", ["a"])
        sc2 = CVScan("b", E.copy(), I2, 1e-9, "hybridization", ["b"])
        avg = average_cv_replicates([sc1, sc2])
        np.testing.assert_allclose(avg.I, np.ones(10) * 2e-6)

    def test_average_different_grid(self):
        E1 = np.linspace(0, 1, 10)
        E2 = np.linspace(0, 1, 20)
        I1 = np.ones(10) * 1e-6
        I2 = np.ones(20) * 3e-6
        sc1 = CVScan("a", E1, I1, 1e-9, "hybridization", ["a"])
        sc2 = CVScan("b", E2, I2, 1e-9, "hybridization", ["b"])
        avg = average_cv_replicates([sc1, sc2])
        np.testing.assert_allclose(avg.I, np.ones(10) * 2e-6, rtol=1e-5)


class TestRunCVPipeline:
    def _make_assignment(self, E, I, step, conc, label="f.csv"):
        content = _make_csv(E, I).encode()
        return {"content": content, "filename": label, "step": step, "concentration": conc}

    def test_probe_only(self):
        E = np.linspace(-0.5, 0.5, 20).tolist()
        I = (np.ones(20) * 1e-6).tolist()
        assignments = [self._make_assignment(E, I, "probe", 0.0)]
        session = run_cv_pipeline(assignments)
        assert session.probe is not None
        assert len(session.groups) == 0

    def test_probe_and_hybridization(self):
        E = np.linspace(-0.5, 0.5, 20).tolist()
        I_probe = (np.ones(20) * 2e-6).tolist()
        I_hyb = (np.ones(20) * 1e-6).tolist()
        assignments = [
            self._make_assignment(E, I_probe, "probe", 0.0, "probe.csv"),
            self._make_assignment(E, I_hyb, "hybridization", 1e-9, "hyb.csv"),
        ]
        session = run_cv_pipeline(assignments)
        assert session.probe is not None
        assert len(session.groups) == 1
        grp = session.groups[0]
        assert grp.concentration == 1e-9
        assert len(grp.delta_signal) > 0
        assert not np.all(np.isnan(grp.delta_signal))

    def test_delta_signal_values(self):
        E = np.linspace(0, 1, 20).tolist()
        I_probe = (np.ones(20) * 4e-6).tolist()
        I_hyb = (np.ones(20) * 2e-6).tolist()
        assignments = [
            self._make_assignment(E, I_probe, "probe", 0.0, "probe.csv"),
            self._make_assignment(E, I_hyb, "hybridization", 1e-9, "hyb.csv"),
        ]
        session = run_cv_pipeline(assignments)
        delta = session.groups[0].delta_signal
        np.testing.assert_allclose(delta, np.ones_like(delta) * 0.5, rtol=1e-4)

    def test_empty_assignments(self):
        session = run_cv_pipeline([])
        assert session.probe is None
        assert session.groups == []


class TestCVPlots:
    def _make_session(self):
        E = np.linspace(-0.5, 0.5, 50)
        probe = CVScan("probe", E, np.sin(E) * 1e-5, 0.0, "probe", ["probe"])
        scan1 = CVScan("hyb1", E, np.sin(E) * 0.8e-5, 1e-9, "hybridization", ["hyb1"])
        scan2 = CVScan("hyb2", E, np.sin(E) * 0.6e-5, 1e-8, "hybridization", ["hyb2"])
        grp1 = CVConcentrationGroup(1e-9, scan1, np.abs(np.sin(E)) * 0.2)
        grp2 = CVConcentrationGroup(1e-8, scan2, np.abs(np.sin(E)) * 0.4)
        return CVSession(probe=probe, groups=[grp1, grp2])

    def test_cv_current_figure_has_traces(self):
        from plotting.cv_plots import cv_current_figure
        session = self._make_session()
        fig = cv_current_figure(session)
        assert len(fig.data) == 3  # probe + 2 concentrations

    def test_cv_calibration_figure_has_fit(self):
        from plotting.cv_plots import cv_calibration_figure
        session = self._make_session()
        fig = cv_calibration_figure(session)
        # Points trace + fit line trace
        assert len(fig.data) == 2

    def test_cv_calibration_empty_session(self):
        from plotting.cv_plots import cv_calibration_figure
        session = CVSession()
        fig = cv_calibration_figure(session)
        assert len(fig.data) == 0
