"""Tests pour core/config.py — circuit par défaut (fit.circuit) et réglages DRT."""

import math

import pytest
from pydantic import ValidationError

from core.config import AppSettings, load_config, config_to_dict
from core.pipeline import circuit_fit_from_config
from circuit import parse_circuit


def test_circuit_bounds_and_guesses_are_floats():
    """Guess et bornes chargés depuis le YAML sont des float (garde-fou anti-piège YAML
    1.1 : sans typage, « 1.0e9 » resterait une chaîne)."""
    cfg = config_to_dict(load_config())
    for name, p in cfg["fit"]["circuit"]["parameters"].items():
        for key in ("initial", "lower", "upper"):
            v = p[key]
            assert v is None or isinstance(v, float), f"{name}.{key} = {v!r} n'est pas un float"


def test_default_circuit_parameters_match_the_expression_exactly():
    """Les paramètres configurés couvrent EXACTEMENT ceux de l'expression (un nom en
    trop ou manquant ferait refuser l'analyse par compile_circuit_fit)."""
    c = config_to_dict(load_config())["fit"]["circuit"]
    _z, names = parse_circuit(c["expression"])
    assert set(c["parameters"]) == set(names)
    assert c["target_param"] in names


def test_default_circuit_compiles_into_a_circuit_fit():
    cf = circuit_fit_from_config(config_to_dict(load_config()))
    assert cf.target_param == "Rct"
    assert list(cf.param_names) == ["Re", "Re_prime", "Rct", "R_D", "tau_d", "Qdl", "alpha", "Cb"]
    assert cf.specs["alpha"].lower == 0.3 and cf.specs["alpha"].upper == 1.0


def test_yaml_and_pydantic_defaults_agree():
    """Le YAML livré et les défauts Pydantic (sans YAML) décrivent le MÊME circuit."""
    from_yaml = config_to_dict(load_config())["fit"]["circuit"]
    from_code = config_to_dict(AppSettings())["fit"]["circuit"]
    assert from_yaml == from_code


def test_a_null_bound_means_unbounded():
    cfg = config_to_dict(load_config())
    cfg["fit"]["circuit"]["parameters"]["Rct"]["upper"] = None
    cfg["fit"]["circuit"]["parameters"]["Re"]["lower"] = None
    cf = circuit_fit_from_config(cfg)
    assert cf.specs["Rct"].upper == math.inf and cf.specs["Re"].lower == -math.inf


def test_drt_defaults_and_mode_validation():
    drt = config_to_dict(load_config())["fit"]["drt"]
    assert drt["enabled"] is True and drt["mode"] == "optimize"
    with pytest.raises(ValidationError, match="optimize"):
        AppSettings.model_validate({"fit": {"drt": {"mode": "nuts"}}})


def test_obsolete_persistence_keys_are_gone():
    """La structure d'erreur n'est plus persistée (AUDIT.md ERR-2) : seule reste
    min_replicates ; l'ancienne section bounds_randles_full a disparu."""
    fit = config_to_dict(load_config())["fit"]
    assert fit["error_structure"] == {"min_replicates": 3}
    assert "bounds_randles_full" not in fit


def test_drt_engine_settings_match_the_engine_defaults():
    """Les réglages d'inversion de fit.drt (YAML ET Pydantic) valent les défauts du moteur :
    transmis au moteur, ils ne changent aucun résultat."""
    import drt.engine as eng

    expected = {"nonneg": eng.DEFAULT_NONNEG, "init_from_ridge": eng.DEFAULT_INIT_FROM_RIDGE,
                "random_seed": eng.DEFAULT_RANDOM_SEED, "chains": eng.DEFAULT_CHAINS,
                "warmup": eng.DEFAULT_WARMUP, "samples": eng.DEFAULT_SAMPLES,
                "adapt_delta": eng.DEFAULT_ADAPT_DELTA, "max_iter": eng.DEFAULT_MAX_ITER}
    for drt in (config_to_dict(load_config())["fit"]["drt"], config_to_dict(AppSettings())["fit"]["drt"]):
        assert {k: drt[k] for k in expected} == expected


def test_an_unknown_drt_key_is_rejected_but_known_ones_are_accepted():
    cfg = AppSettings.model_validate({"fit": {"drt": {"mode": "sample", "chains": 8, "samples": 500}}})
    assert cfg.fit.drt.chains == 8 and cfg.fit.drt.samples == 500
    with pytest.raises(ValidationError, match="chain"):
        AppSettings.model_validate({"fit": {"drt": {"chain": 8}}})      # faute de frappe
    with pytest.raises(ValidationError, match="tau_min"):
        AppSettings.model_validate({"fit": {"drt": {"tau_min": 1e-6}}})  # non lu par le moteur


def test_configured_drt_settings_reach_fit_drt(monkeypatch):
    """Les réglages configurés sont transmis à fit_drt ; les absents restent au défaut moteur."""
    from core.pipeline import _drt_request

    cfg = config_to_dict(AppSettings.model_validate({"fit": {"drt": {"chains": 2, "warmup": 100}}}))
    kw = _drt_request(cfg, None, None)
    assert kw["chains"] == 2 and kw["warmup"] == 100 and kw["mode"] == "optimize"
    assert _drt_request({"fit": {"drt": {"mode": "optimize"}}}, None, None) == {"mode": "optimize"}
