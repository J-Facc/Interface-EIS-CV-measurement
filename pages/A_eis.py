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
        st.markdown(
            """
## Bienvenue dans EIS Analyzer

Analysez vos spectres d'impédance électrochimique pour des biosenseurs microfluidiques ADN/ARN.

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
        st.session_state["eis_session"],
        st.session_state.get("eis_config", cfg),
        validation_results=st.session_state.get("eis_validation"),
    )


if __name__ == "__main__":
    main()
