"""Saisie du circuit équivalent (Randles par défaut, modifiable) : expression, guess/bornes, cible."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from circuit import CircuitError, parse_circuit
from core.analysis import circuit_from_dict
from core.pipeline import InvalidAnalysisInput


def _num_or_none(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def circuit_editor(config: dict):
    """Retourne la description du circuit (``CircuitFit.to_dict()``) ou None si la saisie est invalide."""
    c = config["fit"]["circuit"]
    expr = st.text_area("Circuit équivalent (syntaxe : docs/CIRCUIT_UTILISATEUR.md)", value=c["expression"],
                        key="an_circuit_expr", height=80,
                        help="Défaut : Randles d'Orazem Re + [R'e + (Rct + Z_D) ∥ CPE(Qdl, α)] ∥ Cb. "
                             "`+` = série, parallel(a, b, …) = parallèle.")
    try:
        _z, names = parse_circuit(expr)
    except CircuitError as exc:
        st.error(f"Circuit invalide : {exc}")
        return None
    params_cfg = c.get("parameters") or {}
    table = pd.DataFrame([{"paramètre": n, "initial": (params_cfg.get(n) or {}).get("initial"),
                           "borne basse": (params_cfg.get(n) or {}).get("lower"),
                           "borne haute": (params_cfg.get(n) or {}).get("upper")} for n in names])
    st.caption("Guess de départ et bornes (case vide = non borné). Le fit part de ce guess puis de "
               "perturbations (±0,5 décade).")
    edited = st.data_editor(table, key=f"an_circuit_params_{'_'.join(names)}", disabled=["paramètre"],
                            hide_index=True, width="stretch")
    default_target = c.get("target_param")
    target = st.selectbox("Paramètre cible (Rct du fit)", names,
                          index=names.index(default_target) if default_target in names else 0,
                          key="an_circuit_target")
    params = {}
    for _i, row in edited.iterrows():
        lo, hi = _num_or_none(row["borne basse"]), _num_or_none(row["borne haute"])
        params[row["paramètre"]] = {"initial": _num_or_none(row["initial"]), "lower": lo, "upper": hi}
    desc = {"expression": expr, "target_param": target, "parameters": params}
    try:
        return circuit_from_dict(desc, config).to_dict()
    except InvalidAnalysisInput as exc:
        st.error(str(exc))
        return None
