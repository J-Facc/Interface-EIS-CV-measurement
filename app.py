"""EIS Analyzer — Streamlit entry point.

All application state lives in st.session_state['session'] (EISSession).
No global mutable state outside of session_state.
"""

import streamlit as st

from core.config import load_config, config_to_dict
from core.pipeline import run_pipeline
from ui.sidebar import render_sidebar
from ui.tabs import render_tabs

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
    file_assignments, active_models, run_clicked, theme_mode, phys_overrides = render_sidebar()

    cfg = _merge_overrides(_DEFAULT_CONFIG, phys_overrides)

    if run_clicked:
        if not file_assignments:
            st.warning("⚠️ Veuillez d'abord charger au moins un fichier CSV.")
            return
        if not active_models:
            st.warning("⚠️ Sélectionnez au moins un modèle de fit dans la sidebar.")
            return

        with st.spinner("Analyse en cours…"):
            try:
                session = run_pipeline(
                    file_assignments=file_assignments,
                    config=cfg,
                    active_models=active_models,
                )
                st.session_state["session"] = session
                st.session_state["config"] = cfg
                st.success(
                    f"✅ Analyse terminée — "
                    f"{len(session.groups)} groupe(s) de concentration."
                )
            except Exception as exc:
                st.error(f"❌ Erreur pendant l'analyse : {exc}")
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
5. Cliquez sur **▶ Analyser**.

---
**Modèles disponibles :**
- **Fit circulaire** — lecture géométrique rapide, aucun paramètre physique
- **Randles contraint** — Re fixé, 3 paramètres libres (Rct, Qdl, α)
- **Randles complet** — 8 paramètres libres, pondération Modulus
- **DRT Tikhonov** — distribution des temps de relaxation, λ auto (L-curve)
"""
        )
        return

    session = st.session_state["session"]
    saved_cfg = st.session_state.get("config", cfg)

    render_tabs(session, theme_mode, saved_cfg)


if __name__ == "__main__":
    main()
