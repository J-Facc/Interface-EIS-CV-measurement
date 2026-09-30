"""Tests de CARACTÉRISATION de exports/exporter.py (filet de sécurité avant refonte).

Le cœur de ce fichier est le bug B-EXP (AUDIT.md §8.4) reproduit TEL QUEL : le CSV de
paramètres écrit son en-tête une seule fois, d'après la première ligne, puis aligne
dessus des lignes de modèles différents. Les tests qui le figent doivent CHANGER de
comportement le jour où il est corrigé — c'est le signal voulu (voir l'en-tête de
tests/test_pipeline.py pour la philosophie).

Le reste du fichier fige, en un endroit, le comportement des autres exports (CSV de
calibration, DRT, reconstructions, normalisation, CV, session YAML, archive ZIP),
dont trois défauts SUPPLÉMENTAIRES non listés dans l'audit : DRT et reconstructions
omettent les groupes de concentration, le ZIP écrit toujours un ``drt_values.csv``
même sans DRT, et la branche « résultat CV sous forme de dict » est atteignable mais
sans producteur.
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

# Paramètres tels que les écrivent les deux plugins (cf. AUDIT.md Annexe A.1).
_RANDLES = {"Re": 1, "Re_prime": 2, "Cb": 3, "Rct": 4, "Qdl": 5, "alpha": 6, "R_D": 7, "tau_d": 8}
_DRT = {"Rct": 10.0, "Rp": 11.0, "tau_Rct": -5.2, "n_tau": 80,
        "rct_source": "peak_penultimate", "drt_mode": "optimize"}
# FitResult ne porte plus de champ « Rct » figé : les colonnes communes sont le
# paramètre cible désigné (target_param/target_value/target_std) ; « Rct » n'est plus
# qu'un paramètre parmi d'autres (colonne propre au jeu de paramètres).
_BASE = ["electrode", "concentration", "model", "target_param", "target_value", "target_std",
         "chi2_reduced", "converged"]
_RANDLES_HEADER = _BASE + ["Re", "Re_prime", "Cb", "Rct", "Qdl", "alpha", "R_D", "tau_d"]
_DRT_HEADER = _BASE + ["Rct", "Rp", "tau_Rct", "n_tau", "rct_source", "drt_mode"]


def _randles_fit(rct=4, std=0.1):
    return make_fit_result("randles_full", dict(_RANDLES), rct, std)


def _drt_fit(**kwargs):
    return make_fit_result("drt_bayes", dict(_DRT), 10.0, **kwargs)


def _group_session(models, concentration=1e-9):
    """Une session à un groupe portant les fits ``models`` (ordre = ordre d'insertion)."""
    fits = {"randles_full": _randles_fit, "drt_bayes": _drt_fit}
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
# B-EXP — export_params_csv
# ═════════════════════════════════════════════════════════════════════════════

def test_params_csv_with_a_single_model_is_well_formed():
    """Contrôle : tant qu'un seul modèle est exporté, le CSV est cohérent."""
    out = E.export_params_csv({1: _group_session(["randles_full"])})

    rows = _rows(out)
    assert rows[0] == _RANDLES_HEADER
    assert rows[1] == ["1", "1e-09", "randles_full", "Rct", "4", "0.1", "1.0", "True",
                       "1", "2", "3", "4", "5", "6", "7", "8"]
    assert {len(r) for r in rows} == {16}


def test_bexp_params_csv_misaligns_a_second_model_under_the_first_models_header():
    """COMPORTEMENT ACTUEL BOGUÉ — bug B-EXP (AUDIT.md §8.4), à corriger à l'étape 5.

    Reproduit l'Annexe A.1. ``export_params_csv`` écrit l'en-tête UNE fois, d'après
    la première ligne (ici randles_full : Re, Re_prime, Cb, Qdl, alpha, R_D, tau_d), puis
    y aligne la ligne de drt_bayes, dont les paramètres sont tout autres (Rct, Rp,
    tau_Rct, n_tau, rct_source, drt_mode). Résultat sur la 2ᵉ ligne :

        Rct=10.0       se lit sous « Re »
        Rp=11.0        sous « Re_prime »
        tau_Rct=−5.2   (un LN τ, pas un τ) sous « Cb »
        n_tau=80       sous « Rct »
        rct_source     sous « Qdl »,  drt_mode sous « alpha »
        R_D, tau_d     absentes : 14 champs pour 16 colonnes.

    (Depuis le retrait du champ FitResult.Rct, la colonne « Rct » n'est plus partagée
    par construction : la valeur cible commune est « target_value », alignée.)

    C'est ce que télécharge le bouton « Paramètres fit Randles (CSV) » de la page
    Export, qui passe les sessions NON filtrées (pages/E_export.py). Le jour où ce
    test échoue, B-EXP est corrigé : le mettre à jour en même temps que le correctif.
    """
    out = E.export_params_csv({1: _group_session(["randles_full", "drt_bayes"])})

    rows = _rows(out)
    assert rows[0] == _RANDLES_HEADER
    assert [len(r) for r in rows] == [16, 16, 14]                     # ligne DRT tronquée

    randles, drt = _dicts(out)
    assert randles["model"] == "randles_full" and randles["Re"] == "1" and randles["tau_d"] == "8"
    assert drt["model"] == "drt_bayes"
    assert drt["target_param"] == "Rct" and drt["target_value"] == "10.0"   # colonnes communes : correctes
    assert drt["Re"] == "10.0"                                        # Rct sous « Re »
    assert drt["Re_prime"] == "11.0"                                  # Rp sous « Re_prime »
    assert drt["Cb"] == "-5.2"                                        # ln τ sous « Cb »
    assert drt["Rct"] == "80"                                         # n_tau sous « Rct »
    assert drt["Qdl"] == "peak_penultimate"                           # rct_source sous « Qdl »
    assert drt["alpha"] == "optimize"                                 # drt_mode sous « alpha »
    assert drt["R_D"] is None and drt["tau_d"] is None                # colonnes manquantes


def test_bexp_the_misalignment_flips_when_the_drt_row_comes_first():
    """COMPORTEMENT ACTUEL BOGUÉ — bug B-EXP, à corriger à l'étape 5.

    L'ordre des modèles décide de quel jeu de paramètres fournit l'en-tête. Le registre
    découvre ``drt_bayes`` AVANT ``randles_full`` (ordre alphabétique des modules), donc un
    ``run_pipeline(active_models=None)`` produit ce cas : l'en-tête est celui de la DRT,
    et la ligne Randles (16 champs) DÉBORDE de l'en-tête (14 colonnes).
    """
    out = E.export_params_csv({1: _group_session(["drt_bayes", "randles_full"])})

    rows = _rows(out)
    assert rows[0] == _DRT_HEADER
    assert [len(r) for r in rows] == [14, 14, 16]

    drt, randles = _dicts(out)
    assert drt["Rp"] == "11.0" and drt["drt_mode"] == "optimize"      # DRT : alignée
    assert randles["Rct"] == "1"                                      # Re sous « Rct »
    assert randles["Rp"] == "2"                                       # Re_prime sous « Rp »
    assert randles["tau_Rct"] == "3"                                  # Cb sous « tau_Rct »
    assert randles["n_tau"] == "4"                                    # Rct sous « n_tau »
    assert randles["rct_source"] == "5" and randles["drt_mode"] == "6"
    assert randles[None] == ["7", "8"]                                # R_D, tau_d sans colonne


def test_bexp_the_zip_export_filters_randles_only_and_is_therefore_aligned():
    """Contre-épreuve : ``export_full_zip`` filtre ``randles_full`` avant d'appeler
    ``export_params_csv`` (exporter.py:479-485) — son fichier est cohérent. Seul le
    bouton de la page, qui ne filtre pas, est touché par B-EXP."""
    z = E.export_full_zip(None, {1: _group_session(["randles_full", "drt_bayes"])}, {})

    with zipfile.ZipFile(io.BytesIO(z)) as zf:
        out = zf.read("export/fit_randles/parametres_electrode_1.csv")
    rows = _rows(out)
    assert rows[0] == _RANDLES_HEADER
    assert [r[2] for r in rows[1:]] == ["randles_full"]
    assert {len(r) for r in rows} == {16}


def test_params_csv_is_empty_without_any_fit_and_accepts_a_single_session():
    assert E.export_params_csv({}) == b""                              # même pas d'en-tête
    assert E.export_params_csv({1: EISSession()}) == b""
    single = E.export_params_csv(_group_session(["randles_full"]))     # EISSession seule → électrode 1
    assert _rows(single)[1][0] == "1"


def test_params_csv_lists_electrodes_in_increasing_order():
    out = E.export_params_csv({2: _group_session(["randles_full"]), 1: _group_session(["randles_full"])})
    assert [r[0] for r in _rows(out)[1:]] == ["1", "2"]


# ═════════════════════════════════════════════════════════════════════════════
# Session YAML, calibration, reconstructions, normalisation
# ═════════════════════════════════════════════════════════════════════════════

def test_session_yaml_serialises_fits_with_float_coercion_and_drops_legacy_keys():
    session = _group_session(["randles_full", "drt_bayes"])
    session.groups[0].fit_results["drt_bayes"].params.update({"_lc_lambda": 1.0, "_gcv_lambda": 2.0})
    session.groups[0].fit_results["randles_full"].params["Cb"] = np.float64(3.5)

    data = yaml.safe_load(E.export_session_yaml(session))

    assert list(data) == ["created_at", "groups"]
    group = data["groups"][0]
    assert group["concentration"] == 1e-9 and group["n_points"] == 40
    fits = group["fits"]
    assert list(fits) == ["randles_full", "drt_bayes"]
    assert fits["randles_full"]["params"]["Cb"] == 3.5
    assert fits["randles_full"]["converged"] is True
    # Les paramètres numériques deviennent des flottants ; les chaînes restent des chaînes ;
    # les préfixes « _lc_ » / « _gcv_ » (ancien DRT Tikhonov) sont filtrés.
    assert fits["drt_bayes"]["params"] == {
        "Rct": 10.0, "Rp": 11.0, "tau_Rct": -5.2, "n_tau": 80.0,
        "rct_source": "peak_penultimate", "drt_mode": "optimize"}


def _calibration_session():
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    session.probe.fit_results["randles_full"] = _randles_fit(rct=1000.0)
    for conc, rct in ((1e-9, 1200.0), (1e-8, 1500.0), (1e-7, 2000.0)):
        session.groups.append(ConcentrationGroup(
            concentration=conc, spectrum=make_spectrum(f"c{conc:g}"),
            fit_results={"randles_full": _randles_fit(rct=rct)}))
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
    fr = _randles_fit()
    fr.reconstruction_error = 0.5
    session.probe.fit_results["randles_full"] = fr

    rows = _dicts(E.export_reconstruction_csv({1: session}))

    assert len(rows) == 40
    assert list(rows[0]) == ["electrode", "label", "concentration", "model", "f_Hz", "Zre_mesure",
                             "Zim_mesure", "Zre_reconstruit", "Zim_reconstruit", "erreur_reconstruction"]
    assert rows[0]["label"] == "probe" and rows[0]["model"] == "randles_full"
    assert rows[0]["erreur_reconstruction"] == "0.5"


def test_drt_and_reconstruction_csv_omit_concentration_groups_whose_fits_live_on_the_group():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    Le pipeline range les fits d'un groupe dans ``group.fit_results`` (cf.
    test_pipeline::test_group_fits_live_on_the_group_not_on_its_spectrum), mais
    ``export_drt_csv`` et ``export_reconstruction_csv`` lisent
    ``group.spectrum.fit_results`` — vide : les groupes de concentration disparaissent
    de ces deux exports, sans erreur ni avertissement. Mesuré avec la DRT réelle du
    pipeline : seules les lignes « probe » sortent.
    """
    session = EISSession()
    session.probe = make_spectrum("probe", step="probe", concentration=0.0)
    drt_fit = _drt_fit(drt_tau=np.array([1e-3, 1e-2]), drt_gamma=np.array([1.0, 2.0]))
    session.probe.fit_results = {"drt_bayes": drt_fit, "randles_full": _randles_fit()}
    session.groups = [ConcentrationGroup(
        concentration=1e-9, spectrum=make_spectrum("c1e-9"),
        fit_results={"drt_bayes": drt_fit, "randles_full": _randles_fit()})]   # disposition du pipeline

    drt_labels = {r["label"] for r in _dicts(E.export_drt_csv({1: session}))}
    recon_labels = {r["label"] for r in _dicts(E.export_reconstruction_csv({1: session}))}

    assert drt_labels == {"probe"}
    assert recon_labels == {"probe"}


def test_reconstruction_csv_from_a_real_pipeline_run_only_has_bare_and_probe(tmp_path):
    """Même défaut, sur une vraie session produite par run_pipeline (Randles)."""
    fa = (replicate_assignments("bare", 0.0, 2500.0, 3, 100, "bare")
          + replicate_assignments("probe", 0.0, 3000.0, 3, 0, "probe")
          + replicate_assignments("hybridization", 1e-9, 3500.0, 3, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(tmp_path / "es.json"), active_models=["randles_full"])
    assert "randles_full" in session.groups[0].fit_results             # le fit du groupe existe bien

    labels = {r["label"] for r in _dicts(E.export_reconstruction_csv({1: session}))}
    assert labels == {"bare", "probe"}


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
        "export/fit_randles/parametres_electrode_1.csv",
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
    # Sessions présentes mais sans fit Randles : pas de dossier fit_randles.
    only_drt = _group_session(["drt_bayes"])
    names = _zip_names(E.export_full_zip(None, {1: only_drt}, {}))
    assert not any(n.startswith("export/fit_randles/") for n in names)


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
