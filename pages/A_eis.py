"""Page A — Analyse EIS seule.

Charge les spectres d'impédance, effectue les fits Randles/DRT,
valide par Kramers-Kronig, et produit une calibration OLS log(Rct) vs log([c]).
"""

import streamlit as st

from core.config import load_config, config_to_dict
from core.pipeline import run_pipeline
from ui.sidebar import render_sidebar
from ui.tabs import render_eis_tabs

_DEFAULT_CONFIG = config_to_dict(load_config())


def _merge_overrides(base: dict, overrides: dict) -> dict:
    """Apply user physical-parameter overrides to a config copy."""
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    cfg.setdefault("conditions", {})
    cfg.setdefault("geometry", {})
    cfg.setdefault("physics", {})
    key_map = {
        "Fv": "conditions",
        "xe": "geometry",
        "h": "geometry",
        "d": "geometry",
        "T": "physics",
        "C0": "physics",
    }
    for key, value in overrides.items():
        section = key_map.get(key)
        if section:
            cfg[section][key] = value
    return cfg


def main() -> None:
    st.title("📡 Analyse EIS — Spectroscopie d'impédance")
    st.caption("Fit Randles · DRT · Validation Kramers-Kronig · Calibration OLS")

    if "experiment_clean" not in st.session_state or st.session_state["experiment_clean"] is None:
        st.warning("⚠️ Importez et prétraitez vos données avant l'analyse.")
        st.page_link("pages/0_import.py", label="Aller à l'import", icon="📂")
        st.stop()
        return

    file_assignments, active_models, run_clicked, phys_overrides = render_sidebar("eis")
    cfg = _merge_overrides(_DEFAULT_CONFIG, phys_overrides)

    if run_clicked:
        if not file_assignments:
            st.warning("⚠️ Veuillez charger au moins un fichier CSV.")
            return
        if not active_models:
            st.warning("⚠️ Sélectionnez au moins un modèle de fit.")
            return
        with st.spinner("Analyse EIS en cours…"):
            try:
                session, validation_results = run_pipeline(
                    file_assignments=file_assignments,
                    config=cfg,
                    active_models=active_models,
                )
                st.session_state["eis_session"] = session
                st.session_state["eis_config"] = cfg
                st.session_state["eis_validation"] = validation_results
                st.success(f"✅ Analyse terminée — {len(session.groups)} groupe(s).")
            except Exception as exc:
                st.error(f"❌ Erreur : {exc}")
                return

    if st.session_state.get("eis_session") is None:
        st.info(
            "Chargez vos fichiers CSV dans la sidebar, assignez les étapes "
            "(bare / probe / hybridation), puis cliquez sur **▶ Analyser EIS**."
        )
        return

    render_eis_tabs(
        st.session_state["eis_session"],
        st.session_state.get("eis_config", cfg),
        validation_results=st.session_state.get("eis_validation"),
    )


if __name__ == "__main__":
    main()
