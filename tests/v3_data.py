"""Session v3 complète et synthétique (expérience, mesures, résultats) partagée par les tests
d'archive et de rapports. Pas un module de tests (nom sans ``test_``)."""

from pathlib import Path

import numpy as np

from core.analysis import AnalysisRun
from core.drt_peak import main_peak
from core.experiment import Channel, Experiment, Measurement
from core.results import DRTRecord, Failure
from tests.synthetic_data import make_config, orazem_noisy_arrays

CV = (Path(__file__).parent / "data" / "cv_bare_2cycles.txt").read_bytes()


def _eis(ch, el, step, rep, rct, seed):
    f, zr, zi, _ = orazem_noisy_arrays(rct, seed)
    return Measurement(ch, el, step, rep, "EIS", f"{step}_{rep}.txt", {"f": f, "Zre": zr, "Zim": zi})


def _cv(ch, el, step, rep, scale):
    from core.cv_loader import load_cv_measurement
    d = load_cv_measurement(CV, "x")
    return Measurement(ch, el, step, rep, "CV", f"cv_{step}_{rep}.txt",
                       {"E": d["E"], "I": d["I"] * scale, "cycle": d["cycle"]}, warnings=["cycle 2 conservé"])


def _fake_drt(sp, mode="sample"):
    tau = np.logspace(-6, 1, 80)
    rct = float(sp.fit.target_value) if sp.fit else 1000.0
    gamma = rct / (0.5 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((np.log(tau) - np.log(3e-3)) / 0.5) ** 2)
    return DRTRecord(mode, tau, gamma, main_peak(tau, gamma, [1e-5, 1e-1]), rct * 1.02, 12.5, "peak_penultimate",
                     rct * 1.3, True, sp.Zre * 1.001, sp.Zim * 0.999, gamma * 0.9, gamma * 1.1,
                     {"rhat_max": 1.003, "divergences": 0, "alerts": []}, ["note"])


def build_session():
    exp = Experiment(["miR-21", "ARN-X"], [Channel(1, "miR-21", [1e-9, 1e-8, 1e-7]), Channel(2, "ARN-X", [1e-8])],
                     name="essai")
    ms = []
    for el in (1, 2):
        ms += [_eis(1, el, "probe", r, 3000, 10 * el + r) for r in (1, 2, 3)]
        for i, rct in enumerate((3500, 4500, 6000), start=1):
            ms += [_eis(1, el, f"conc_{i}", r, rct, 100 * el + 10 * i + r) for r in (1, 2)]
        ms += [_cv(1, el, "probe", r, 1.0) for r in (1, 2)]
        ms += [_cv(1, el, f"conc_{i}", 1, 1 - 0.1 * i) for i in (1, 2, 3)]
    ms += [_eis(2, 1, "probe", 1, 2000, 7), _eis(2, 1, "conc_1", 1, 2600, 8)]
    run = AnalysisRun(exp, ms, make_config(enabled=False), executor_kind="sync")
    assert run.wait(timeout=600)
    res = run.results
    for k, sp in res.eis.spectra.items():         # DRT synthétiques (IC compris), sans CmdStan
        if not k.endswith("r2"):
            sp.drt = _fake_drt(sp)
    res.eis.spectra["EIS_C1_E1_conc_1_r2"].drt_error = "échec CmdStan simulé"
    res.failures.append(Failure("EIS_C1_E1_conc_1_r2", "drt", "échec CmdStan simulé"))
    return exp, ms, res


