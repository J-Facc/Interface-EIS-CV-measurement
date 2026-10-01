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
    assert drt == {"enabled": True, "mode": "optimize"}
    with pytest.raises(ValidationError, match="optimize"):
        AppSettings.model_validate({"fit": {"drt": {"mode": "nuts"}}})


def test_obsolete_persistence_keys_are_gone():
    """La structure d'erreur n'est plus persistée (AUDIT.md ERR-2) : seule reste
    min_replicates ; l'ancienne section bounds_randles_full a disparu."""
    fit = config_to_dict(load_config())["fit"]
    assert fit["error_structure"] == {"min_replicates": 3}
    assert "bounds_randles_full" not in fit
