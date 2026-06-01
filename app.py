"""EIS Analyzer — Streamlit entry point.

All application state lives in st.session_state['session'] (EISSession).
No global mutable state outside of session_state.
"""

import streamlit as st

from core.config import load_config, config_to_dict
from core.pipeline import run_pipeline
from core.cv_pipeline import run_cv_pipeline
from core.cv_models import CVSession
from ui.sidebar import render_sidebar
from ui.tabs import render_eis_tabs, render_cv_tabs

st.set_page_config(
    page_title="EIS Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    mode = st.radio(
        "Interface",
        ["⚡ EIS", "📈 CV"],
        horizontal=True,
        key="mode",
        label_visibility="collapsed",
    )

    if mode == "⚡ EIS":
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
                    st.session_state["session"] = session
                    st.session_state["config"] = cfg
                    st.session_state["validation_results"] = validation_results
                    st.success(f"✅ Analyse terminée — {len(session.groups)} groupe(s).")
                except Exception as exc:
                    st.error(f"❌ Erreur : {exc}")
                    return

        if "session" not in st.session_state:
            st.markdown(
                """
## Bienvenue dans EIS Analyzer

Analysez vos spectres d'impédance électrochimique (EIS) pour des biosenseurs
microfluidiques ADN/ARN.

**Pour démarrer :**
1. Chargez vos fichiers CSV dans la sidebar (gauche).
2. Assignez chaque fichier à une étape : *bare*, *probe* ou *hybridation*.
3. Saisissez la concentration pour les fichiers d'hybridation.
4. Sélectionnez les modèles de fit souhaités.
5. Cliquez sur **▶ Analyser EIS**.

---
**Modèles disponibles :**
- **Fit circulaire** — lecture géométrique rapide, aucun paramètre physique
- **Randles contraint** — Re fixé, 3 paramètres libres (Rct, Qdl, α)
- **Randles complet** — 8 paramètres libres, pondération Modulus
- **DRT (FFT)** — distribution des temps de relaxation via FFT
"""
            )
            return

        render_eis_tabs(
            st.session_state["session"],
            st.session_state.get("config", cfg),
            validation_results=st.session_state.get("validation_results"),
        )

    else:  # mode CV
        cv_assignments, run_clicked_cv = render_sidebar("cv")

        if run_clicked_cv:
            if not cv_assignments:
                st.warning("⚠️ Veuillez charger au moins un fichier CV.")
                return
            with st.spinner("Analyse CV en cours…"):
                try:
                    cv_session = run_cv_pipeline(cv_assignments)
                    st.session_state["cv_session"] = cv_session
                    st.success("✅ Analyse CV terminée.")
                except Exception as exc:
                    st.error(f"❌ Erreur CV : {exc}")
                    return

        if "cv_session" not in st.session_state:
            st.markdown(
                """
## Voltampérométrie cyclique

**Pour démarrer :**
1. Chargez vos fichiers CV dans la sidebar (Bare, Probe, Hybridations).
2. Saisissez les concentrations.
3. Cliquez sur **▶ Analyser CV**.
"""
            )
            return

        render_cv_tabs(st.session_state["cv_session"])


if __name__ == "__main__":
    main()
