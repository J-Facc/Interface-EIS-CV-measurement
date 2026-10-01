"""Tests de exports/exporter.py.

B-EXP (AUDIT.md §8.4) CORRIGÉ à l'étape 5 : le CSV de paramètres écrivait son en-tête
une seule fois, d'après le PREMIER modèle rencontré, puis alignait par POSITION les
lignes des autres modèles (Rp de la DRT sous « Re »…). Chaque modèle écrit désormais
SES PROPRES colonnes ``<modèle>.<paramètre>``, alignées par NOM ; les tests de la
section A vérifient ce comportement corrigé (ils figeaient auparavant le défaut).

Corrigés aussi : DRT et reconstructions n'omettent plus les groupes de concentration
(``ConcentrationGroup.fit_results`` est un alias de ``spectrum.fit_results``).

Le reste du fichier fige le comportement des autres exports (calibration, DRT,
normalisation, CV, session YAML, archive ZIP).
"""

import csv
import io
import logging
import zipfile

import numpy as np
import pytest
import yaml

from core.cv_models import CVConcentrationGroup, CVScan, CVSession
from core.models import ConcentrationGroup, EISSession
from core.pipeline import run_pipeline
from exports import exporter as E
from tests.synthetic_data import (
    make_config,
    make_fit_result,
    make_spectrum,
    randles_file,
    replicate_assignments,
)

# Paramètres tels que les écrivent le fit Orazem (circuit Randles) et la DRT.
_CIRCUIT = {"Re": 1, "Re_prime": 2, "Cb": 3, "Rct": 4, "Qdl": 5, "alpha": 6, "R_D": 7, "tau_d": 8}
_DRT = {"Rct": 10.0, "Rp": 11.0, "tau_Rct": 5.5e-3, "n_tau": 80,
        "rct_source": "peak_penultimate", "drt_mode": "optimize"}
_BASE = ["electrode", "group", "concentration", "spectrum", "kind", "model",
         "target_param", "target_value", "target_std", "chi2_reduced", "converged"]
_CIRCUIT_COLS = [f"orazem.{k}" for k in _CIRCUIT]
_DRT_COLS = [f"drt_bayes.{k}" for k in _DRT]


def _circuit_fit(rct=4, std=0.1):
    return make_fit_result("orazem", dict(_CIRCUIT), rct, std)


def _drt_fit(**kwargs):
    return make_fit_result("drt_bayes", dict(_DRT), 10.0, **kwargs)


def _group_session(models, concentration=1e-9):
    """Une session à un groupe portant les fits ``models`` (ordre = ordre d'insertion)."""
    fits = {"orazem": _circuit_fit, "drt_bayes": _drt_fit}
    session = EISSession()
    session.groups = [ConcentrationGroup(
        concentration=concentration, spectrum=make_spectrum(),
        fit_results={m: fits[m]() for m in models})]
    return session


def _rows(csv_bytes):
    return list(csv.reader(io.StringIO(csv_bytes.decode())))


def _dicts(csv_bytes):
    return list(csv.DictReader(io.StringIO(csv_bytes.decode())))


# ═════════════════════════════════════════════════════════════════════════════
# A. B-EXP corrigé — export_params_csv
# ═════════════════════════════════════════════════════════════════════════════

def test_params_csv_with_a_single_model_is_well_formed():
    out = E.export_params_csv({1: _group_session(["orazem"])})

    rows = _rows(out)
    assert rows[0] == _BASE + _CIRCUIT_COLS
    assert rows[1] == ["1", "1.00e-09 M", "1e-09", "x", "moyenne", "orazem", "Rct", "4", "0.1",
                       "1.0", "True", "1", "2", "3", "4", "5", "6", "7", "8"]
    assert {len(r) for r in rows} == {len(_BASE) + 8}


@pytest.mark.parametrize("order", [["orazem", "drt_bayes"], ["drt_bayes", "orazem"]])
def test_bexp_fixed_each_model_writes_its_own_columns_aligned_by_name(order):
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ — B-EXP »).

    Avant : en-tête du premier modèle, lignes des autres alignées par position — Rp
    sous « Re », τ sous « Cb », colonnes manquantes ou en trop selon l'ordre. Après :
    une colonne par (modèle, paramètre), chaque ligne ne remplit que les siennes, et
    toutes les lignes ont la longueur de l'en-tête QUEL QUE SOIT l'ordre des modèles.
    """
    out = E.export_params_csv({1: _group_session(order)})

    rows = _rows(out)
    cols = {"orazem": _CIRCUIT_COLS, "drt_bayes": _DRT_COLS}
    assert rows[0] == _BASE + cols[order[0]] + cols[order[1]]
    assert {len(r) for r in rows} == {len(rows[0])}                  # plus de ligne tronquée/débordante

    by_model = {r["model"]: r for r in _dicts(out)}
    circuit, drt = by_model["orazem"], by_model["drt_bayes"]
    assert [circuit[c] for c in _CIRCUIT_COLS] == ["1", "2", "3", "4", "5", "6", "7", "8"]
    assert all(circuit[c] == "" for c in _DRT_COLS)
    assert drt["drt_bayes.Rp"] == "11.0" and drt["drt_bayes.tau_Rct"] == "0.0055"
    assert drt["drt_bayes.rct_source"] == "peak_penultimate" and drt["drt_bayes.drt_mode"] == "optimize"
    assert all(drt[c] == "" for c in _CIRCUIT_COLS)
    # Le « Rct » du circuit et celui de la DRT (grandeurs différentes) ne partagent pas de colonne.
    assert circuit["orazem.Rct"] == "4" and drt["drt_bayes.Rct"] == "10.0"
    assert drt["target_param"] == "Rct" and drt["target_value"] == "10.0"


def test_params_csv_writes_a_std_column_next_to_each_parameter_that_has_one():
    session = _group_session(["orazem"])
    fr = session.groups[0].fit_results["orazem"]
    fr.params_std = {"Rct": 0.25, "alpha": 0.01}
    row = _dicts(E.export_params_csv({1: session}))[0]
    assert row["orazem.Rct_std"] == "0.25" and row["orazem.alpha_std"] == "0.01"
    assert "orazem.Re_std" not in row


def test_params_csv_exports_every_replicate_and_the_mean_of_every_group():
    session = _group_session(["orazem"])
    reps = []
    for k in range(3):
        r = make_spectrum(f"r{k}")
        r.fit_results["orazem"] = _circuit_fit(rct=4 + k)
        reps.append(r)
    session.groups[0].replicate_spectra = reps
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    session.probe.fit_results["orazem"] = _circuit_fit(rct=2)

    rows = _dicts(E.export_params_csv({1: session}))
    assert [(r["group"], r["spectrum"], r["kind"], r["target_value"]) for r in rows] == [
        ("Probe", "probe", "moyenne", "2"),
        ("1.00e-09 M", "r0", "réplicat", "4"), ("1.00e-09 M", "r1", "réplicat", "5"),
        ("1.00e-09 M", "r2", "réplicat", "6"), ("1.00e-09 M", "x", "moyenne", "4"),
    ]


def test_bexp_the_zip_parameter_file_now_carries_every_model_aligned():
    z = E.export_full_zip(None, {1: _group_session(["orazem", "drt_bayes"])}, {})

    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        out = zf.read("export/fits/parametres_electrode_1.csv")
    rows = _rows(out)
    assert rows[0] == _BASE + _CIRCUIT_COLS + _DRT_COLS
    assert [r[5] for r in rows[1:]] == ["orazem", "drt_bayes"]
    assert {len(r) for r in rows} == {len(rows[0])}


def test_params_csv_is_header_only_without_any_fit_and_accepts_a_single_session():
    assert _rows(E.export_params_csv({})) == [_BASE]
    assert _rows(E.export_params_csv({1: EISSession()})) == [_BASE]
    single = E.export_params_csv(_group_session(["orazem"]))         # EISSession seule → électrode 1
    assert _rows(single)[1][0] == "1"


def test_params_csv_lists_electrodes_in_increasing_order():
    out = E.export_params_csv({2: _group_session(["orazem"]), 1: _group_session(["orazem"])})
    assert [r[0] for r in _rows(out)[1:]] == ["1", "2"]


# ═════════════════════════════════════════════════════════════════════════════
# Session YAML, calibration, reconstructions, normalisation
# ═════════════════════════════════════════════════════════════════════════════

def test_session_yaml_serialises_fits_with_float_coercion():
    session = _group_session(["orazem", "drt_bayes"])
    session.groups[0].fit_results["orazem"].params["Cb"] = np.float64(3.5)

    data = yaml.safe_load(E.export_session_yaml(session))

    assert list(data) == ["created_at", "circuit", "drt_mode", "messages", "load_errors", "groups"]
    group = data["groups"][0]
    assert group["group"] == "1.00e-09 M"
    assert group["concentration"] == 1e-9 and group["n_points"] == 40
    assert group["replicates"] == [] and group["status"] is None
    fits = group["fits"]
    assert list(fits) == ["orazem", "drt_bayes"]
    assert fits["orazem"]["params"]["Cb"] == 3.5
    assert fits["orazem"]["converged"] is True
    # Les paramètres numériques deviennent des flottants ; les chaînes restent des chaînes.
    assert fits["drt_bayes"]["params"] == {
        "Rct": 10.0, "Rp": 11.0, "tau_Rct": 5.5e-3, "n_tau": 80.0,
        "rct_source": "peak_penultimate", "drt_mode": "optimize"}


def _calibration_session():
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    session.probe.fit_results["orazem"] = _circuit_fit(rct=1000.0)
    for conc, rct in ((1e-9, 1200.0), (1e-8, 1500.0), (1e-7, 2000.0)):
        session.groups.append(ConcentrationGroup(
            concentration=conc, spectrum=make_spectrum(f"c{conc:g}"),
            fit_results={"orazem": _circuit_fit(rct=rct)}))
    return session


def test_calibration_csv_columns_and_values():
    out = E.export_calibration_csv({1: _calibration_session()})

    rows = _dicts(out)
    assert list(rows[0]) == ["electrode", "model", "concentration_M", "log10_concentration",
                             "signal_norm", "Rct_Ohm", "Rct_probe_Ohm",
                             "slope", "intercept", "r2", "p_value", "std_err"]
    assert [r["concentration_M"] for r in rows] == ["1e-09", "1e-08", "1e-07"]
    assert [float(r["log10_concentration"]) for r in rows] == [-9.0, -8.0, -7.0]
    # signal = |Rct_probe − Rct| / Rct_probe
    assert [float(r["signal_norm"]) for r in rows] == pytest.approx([0.2, 0.5, 1.0])
    assert {r["Rct_probe_Ohm"] for r in rows} == {"1000.0"}
    # Régression : comparée à un calcul indépendant (numpy), identique sur toutes les lignes.
    slope, intercept = np.polyfit([-9.0, -8.0, -7.0], [0.2, 0.5, 1.0], 1)
    r2 = np.corrcoef([-9.0, -8.0, -7.0], [0.2, 0.5, 1.0])[0, 1] ** 2
    for r in rows:
        assert float(r["slope"]) == pytest.approx(slope)
        assert float(r["intercept"]) == pytest.approx(intercept)
        assert float(r["r2"]) == pytest.approx(r2)


def test_calibration_csv_is_header_only_without_a_probe_fit():
    out = E.export_calibration_csv({1: EISSession()})
    assert len(_rows(out)) == 1
    assert out.endswith(b"\r\n")                                       # le module csv écrit \r\n


def test_reconstruction_csv_rows_and_columns():
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    fr = _circuit_fit()
    fr.reconstruction_error = 0.5
    session.probe.fit_results["orazem"] = fr

    rows = _dicts(E.export_reconstruction_csv({1: session}))

    assert len(rows) == 40
    assert list(rows[0]) == ["electrode", "label", "concentration", "model", "f_Hz", "Zre_mesure",
                             "Zim_mesure", "Zre_reconstruit", "Zim_reconstruit", "erreur_reconstruction"]
    assert rows[0]["label"] == "probe" and rows[0]["model"] == "orazem"
    assert rows[0]["erreur_reconstruction"] == "0.5"


def test_drt_and_reconstruction_csv_include_concentration_groups():
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ ») : les fits d'un groupe étaient rangés
    dans ``group.fit_results`` alors que ces exports lisent ``group.spectrum.fit_results``
    — vide : les groupes disparaissaient. ``group.fit_results`` est désormais un alias
    du dictionnaire du spectre moyen."""
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    drt_fit = _drt_fit(drt_tau=np.array([1e-3, 1e-2]), drt_gamma=np.array([1.0, 2.0]))
    session.probe.fit_results = {"drt_bayes": drt_fit, "orazem": _circuit_fit()}
    session.groups = [ConcentrationGroup(
        concentration=1e-9, spectrum=make_spectrum("c1e-9"),
        fit_results={"drt_bayes": drt_fit, "orazem": _circuit_fit()})]
    assert session.groups[0].spectrum.fit_results is session.groups[0].fit_results

    drt_labels = {r["label"] for r in _dicts(E.export_drt_csv({1: session}))}
    recon_labels = {r["label"] for r in _dicts(E.export_reconstruction_csv({1: session}))}

    assert drt_labels == {"probe", "c1e-9"}
    assert recon_labels == {"probe", "c1e-9"}


def test_reconstruction_csv_from_a_real_pipeline_run_has_every_group():
    """Sur une vraie session de run_pipeline : bare, probe ET la concentration ; les
    tableaux reconstruits sont alignés sur les fréquences du spectre (HF → BF)."""
    fa = (replicate_assignments("bare", 0.0, 2500.0, 3, 100, "bare")
          + replicate_assignments("probe", 0.0, 3000.0, 3, 0, "probe")
          + replicate_assignments("hybridization", 1e-9, 3500.0, 3, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(enabled=False))
    assert "orazem" in session.groups[0].fit_results

    rows = _dicts(E.export_reconstruction_csv({1: session}))
    assert {r["label"] for r in rows} == {"bare", "probe", "c1_r0.txt (avg)"}
    probe = [r for r in rows if r["label"] == "probe"]
    f = [float(r["f_Hz"]) for r in probe]
    assert f == sorted(f, reverse=True)
    rel = [abs(float(r["Zre_reconstruit"]) - float(r["Zre_mesure"])) / float(r["Zre_mesure"]) for r in probe]
    assert max(rel) < 0.05                                             # mêmes fréquences, même ordre


def test_drt_csv_rows_mode_default_and_log_floor():
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    session.probe.fit_results["drt_bayes"] = _drt_fit(
        drt_tau=np.array([1e-3, 1e-2]), drt_gamma=np.array([1.0, 2.0]), drt_mode=None)
    replicate = make_spectrum("r0")
    replicate.fit_results["drt_bayes"] = _drt_fit(
        drt_tau=np.array([1.0]), drt_gamma=np.array([0.0]), drt_mode="sample")
    session.probe_replicate_spectra = [replicate]
    no_drt = make_spectrum("r1")                                       # sans DRT : simplement omis
    session.probe_replicate_spectra.append(no_drt)

    rows = _dicts(E.export_drt_csv({1: session}))

    assert [(r["label"], r["replicate_idx"], r["drt_mode"]) for r in rows] == [
        ("probe", "avg", "optimize"), ("probe", "avg", "optimize"), ("probe", "0", "sample")]
    assert float(rows[0]["ln_tau"]) == pytest.approx(np.log(1e-3))
    assert float(rows[1]["ln_gamma"]) == pytest.approx(np.log(2.0))
    # γ = 0 : le log est protégé par un plancher de 1e-300 (ln = −690,78).
    assert float(rows[2]["ln_gamma"]) == pytest.approx(np.log(1e-300))


def test_drt_csv_skips_fits_without_a_distribution():
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    session.probe.fit_results["drt_bayes"] = _drt_fit()                # ni drt_tau ni drt_gamma
    assert _dicts(E.export_drt_csv({1: session})) == []


def test_normalization_csv_falls_back_to_an_index_when_frequencies_are_missing():
    normalized = {
        1e-9: {"Zre_norm": [1, 2], "Zim_norm": [3, 4]},
        1e-8: {"Zre_norm": [5], "Zim_norm": [6], "f": [100.0]},
    }
    rows = _rows(E.export_normalization_csv(normalized))

    assert rows[0] == ["concentration", "frequency_Hz", "Zre_norm", "Zim_norm"]
    assert rows[1:] == [["1e-09", "0", "1", "3"], ["1e-09", "1", "2", "4"], ["1e-08", "100.0", "5", "6"]]
    assert _rows(E.export_normalization_csv(None)) == [rows[0]]


# ═════════════════════════════════════════════════════════════════════════════
# Calibration CV
# ═════════════════════════════════════════════════════════════════════════════

def _cv_session():
    def group(conc, delta):
        scan = CVScan(label=f"c{conc:g}", E=np.array([0.0, 1.0]), I=np.array([1.0, 2.0]),
                      concentration=conc, step="hybridization")
        return CVConcentrationGroup(concentration=conc, scan=scan, delta_signal=np.array(delta))
    return CVSession(groups=[group(1e-9, [0.1, 0.3]), group(1e-8, [0.4, 0.6]), group(0.0, [9.0])])


def test_cv_calibration_csv_uses_the_mean_signal_per_positive_concentration():
    rows = _dicts(E.export_cv_calibration_csv(_cv_session()))

    assert list(rows[0]) == ["concentration_M", "log10_concentration", "signal_norm",
                             "slope", "intercept", "r2", "p_value", "std_err"]
    assert [float(r["signal_norm"]) for r in rows] == pytest.approx([0.2, 0.5])   # la conc. 0 est écartée
    assert float(rows[0]["slope"]) == pytest.approx(0.3)


def test_cv_calibration_csv_multi_has_one_block_per_electrode_and_skips_unusable_sessions():
    out = E.export_cv_calibration_csv_multi({2: _cv_session(), 1: _cv_session(), 3: CVSession()})

    rows = _dicts(out)
    assert list(rows[0]) == ["electrode", "concentration_M", "log10_concentration",
                             "delta_signal_moyen", "slope", "intercept", "r2", "p_value", "std_err"]
    assert [r["electrode"] for r in rows] == ["1", "1", "2", "2"]       # 3 : < 2 concentrations → omis


def test_cv_calibration_csv_is_header_only_with_fewer_than_two_concentrations():
    assert len(_rows(E.export_cv_calibration_csv(CVSession()))) == 1


def test_cv_calibration_csv_from_result_is_reachable_but_has_no_producer():
    """COMPORTEMENT ACTUEL (AUDIT.md §2.1 #8 / §2.6) : cette variante ne sert qu'à un dict
    ``{"groups": [...]}`` que plus aucune page ne produit ; elle reste pourtant testable
    et atteignable depuis export_full_zip. Ne retient que les groupes à conc > 0 et
    signal fini."""
    result = {"groups": [
        {"concentration": 1e-9, "delta_I_norm_mean": 0.2},
        {"concentration": 1e-8, "delta_I_norm_mean": float("nan")},
        {"concentration": 0.0, "delta_I_norm_mean": 0.9},
        {"concentration": 1e-7, "delta_I_norm_mean": 0.8},
    ]}
    rows = _dicts(E.export_cv_calibration_csv_from_result(result, {"slope": 0.3, "r2": 0.9}))

    assert [r["concentration_M"] for r in rows] == ["1e-09", "1e-07"]
    assert rows[0]["slope"] == "0.3" and rows[0]["intercept"] == ""     # clé absente → champ vide
    assert len(_rows(E.export_cv_calibration_csv_from_result({}))) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Archive ZIP
# ═════════════════════════════════════════════════════════════════════════════

def _cv_file(peak, seed):
    rng = np.random.default_rng(seed)
    E_ = np.linspace(-0.2, 0.6, 30)
    I_ = peak * np.exp(-((E_ - 0.25) / 0.1) ** 2) + 1e-7 * rng.standard_normal(30)
    return ("Ewe/V\t<I>/mA\n" + "\n".join(f"{e:.6f}\t{i * 1e3:.9f}" for e, i in zip(E_, I_))).encode()


def _experiment_clean():
    return {
        "mode": "both", "n_electrodes": 1, "concentrations": [1e-9],
        "probe": {
            "eis": {"electrode_1": [io.BytesIO(randles_file(3000.0, seed=1)),
                                    io.BytesIO(randles_file(3000.0, seed=2))]},
            "cv": {"electrode_1": [io.BytesIO(_cv_file(5e-6, 1))]},
        },
        "calibration": {
            "eis": {"electrode_1": [[io.BytesIO(randles_file(3500.0, seed=3))]]},
            "cv": {"electrode_1": [[io.BytesIO(_cv_file(4e-6, 2))]]},
        },
    }


def _zip_names(z):
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        return sorted(zf.namelist())


def test_full_zip_layout_with_every_source_available():
    session = _calibration_session()
    z = E.export_full_zip(
        _experiment_clean(), {1: session}, {1e-9: {"Zre_norm": [1], "Zim_norm": [2]}},
        cv_session={1: _cv_session()})

    assert _zip_names(z) == sorted([
        "export/data_pretraitees/eis_electrode_1.csv",
        "export/data_pretraitees/cv_electrode_1.csv",
        "export/drt/drt_values.csv",
        "export/normalisation/nyquist_normalise.csv",
        "export/fits/parametres_electrode_1.csv",
        "export/fits/resultats_par_replicat.csv",
        "export/fits/resultats_par_groupe.csv",
        "export/reconstructions/reconstruction_values.csv",
        "export/calibration/eis_calibration.csv",
        "export/calibration/cv_calibration.csv",
        "export/session/eis_session_electrode_1.yaml",
    ])


def test_full_zip_preprocessed_data_averages_replicates_per_condition():
    z = E.export_full_zip(_experiment_clean(), {}, {})
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        eis = _rows(zf.read("export/data_pretraitees/eis_electrode_1.csv"))
        cv = _rows(zf.read("export/data_pretraitees/cv_electrode_1.csv"))

    assert eis[0] == ["label", "concentration", "f_Hz", "Zre_Ohm", "Zim_Ohm"]
    assert [r[0] for r in eis[1:]] == ["probe"] * 40 + ["c1"] * 40       # 2 réplicats moyennés → 40 points
    assert {r[1] for r in eis[41:]} == {"1e-09"}
    assert cv[0] == ["label", "concentration", "E_V", "I_A"]
    assert {r[0] for r in cv[1:]} == {"probe", "c1"}


def test_full_zip_always_writes_a_drt_file_even_without_any_drt():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    La docstring promet qu'aucun fichier vide n'est créé. Or ``export_drt_csv`` renvoie
    toujours au moins l'en-tête, donc ``drt_bytes.strip()`` est toujours vrai :
    ``export/drt/drt_values.csv`` est écrit, sans une seule ligne de données.
    """
    z = E.export_full_zip(None, {1: _calibration_session()}, {})
    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        drt = _rows(zf.read("export/drt/drt_values.csv"))
    assert drt == [["electrode", "label", "concentration", "replicate_idx", "drt_mode", "ln_tau", "ln_gamma"]]


def test_full_zip_omits_everything_that_is_absent():
    for args in ((None, {}, {}), (None, None, None)):
        assert _zip_names(E.export_full_zip(*args)) == []
    # Une session sans aucun fit : pas de fichier de paramètres (en-tête seul).
    names = _zip_names(E.export_full_zip(None, {1: EISSession()}, {}))
    assert "export/fits/parametres_electrode_1.csv" not in names


def test_full_zip_picks_the_cv_calibration_exporter_from_the_shape_of_cv_session():
    def cv_csv(cv_session):
        z = E.export_full_zip(None, None, None, cv_session=cv_session)
        with zipfile.ZipFile(io.BytesIO(z)) as zf:
            return _rows(zf.read("export/calibration/cv_calibration.csv"))[0]

    assert cv_csv(_cv_session())[0] == "concentration_M"                       # CVSession seule
    assert cv_csv({1: _cv_session()})[0] == "electrode"                         # dict d'électrodes
    assert cv_csv({"groups": [{"concentration": 1e-9, "delta_I_norm_mean": 0.2}]})[2] == "delta_I_norm_mean"
    # Un CVSession sans concentration exploitable : le fichier est écrit quand même, réduit à
    # son en-tête (même défaut que le drt_values.csv vide ci-dessous).
    assert len(cv_csv(CVSession())) == 8
    assert _zip_names(E.export_full_zip(None, None, None, cv_session=CVSession())) == [
        "export/calibration/cv_calibration.csv"]


def test_full_zip_logs_nothing_at_error_level(caplog):
    with caplog.at_level(logging.ERROR):
        E.export_full_zip(_experiment_clean(), {1: _calibration_session()}, {})
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
