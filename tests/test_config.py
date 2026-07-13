"""Tests pour core/config.py — cohérence YAML ↔ Pydantic des bornes de fit."""

from core.config import load_config, config_to_dict
from fits.randles_full import RandlesFullModel, _PARAM_NAMES


def test_bounds_all_float():
    """Toutes les bornes chargées sont des float (garde-fou anti-piège YAML 1.1).

    Le résolveur float de YAML 1.1 ne reconnaît un exposant que s'il porte un
    signe (1.0e+9, pas 1.0e9). Sans typage list[float], une borne comme
    Rct: [100.0, 1.0e9] resterait la chaîne "1.0e9". Ce test échoue si le piège
    revient (soit dans le YAML, soit par un champ Pydantic re-typé en list nu).
    """
    cfg = config_to_dict(load_config())
    bounds = cfg["fit"]["bounds_randles_full"]
    for name, (lo, hi) in bounds.items():
        assert isinstance(lo, float), f"borne basse {name} = {lo!r} n'est pas un float"
        assert isinstance(hi, float), f"borne haute {name} = {hi!r} n'est pas un float"


def test_bounds_keys_match_param_names():
    """Les clés des bornes couvrent exactement les paramètres du modèle Randles.

    Empêche la régression I2 : des clés Pydantic (ZD0/D_eff) divergentes des
    clés YAML (R_D/tau_d) faisaient ignorer silencieusement les bornes de R_D et
    tau_d au profit du fallback en dur de RandlesFullModel.bounds.
    """
    cfg = config_to_dict(load_config())
    bounds_keys = set(cfg["fit"]["bounds_randles_full"].keys())
    assert bounds_keys == set(_PARAM_NAMES), (
        f"clés bornes {sorted(bounds_keys)} != paramètres {sorted(_PARAM_NAMES)}"
    )


def test_configured_bounds_are_read_by_model():
    """Une borne modifiée dans la config est bien lue par RandlesFullModel.bounds
    (y compris R_D/tau_d, autrefois ignorées)."""
    cfg = config_to_dict(load_config())
    cfg["fit"]["bounds_randles_full"]["R_D"] = [42.0, 4242.0]
    cfg["fit"]["bounds_randles_full"]["Rct"] = [7.0, 7e7]
    lo, hi = RandlesFullModel().bounds(cfg)
    assert (lo["R_D"], hi["R_D"]) == (42.0, 4242.0)
    assert (lo["Rct"], hi["Rct"]) == (7.0, 7e7)
