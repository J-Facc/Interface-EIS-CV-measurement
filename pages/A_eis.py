"""Page A — Analyse EIS seule.

Charge les spectres d'impédance depuis experiment_clean (session_state),
effectue les fits Randles/DRT, valide par Kramers-Kronig, et produit
une calibration OLS log(Rct) vs log([c]).
"""

import streamlit as st

from core.config import load_config, config_to_dict
from core.pipeline import run_pipeline
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
        "h":  "geometry",
        "d":  "geometry",
        "T":  "physics",
        "C0": "physics",
    }
    for key, value in overrides.items():
        section = key_map.get(key)
        if section:
            cfg[section][key] = value
    return cfg


def _build_file_assignments(experiment: dict) -> list:
    """Convertit experiment_clean en liste de file_assignments pour run_pipeline."""
    mode = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)
    assignments = []

    if mode not in ("eis_only", "both"):
        return assignments

    probe_eis = (experiment.get("probe") or {}).get("eis") or {}
    cal_eis   = (experiment.get("calibration") or {}).get("eis") or {}

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        for ri, bio in enumerate(probe_eis.get(key) or []):
            if bio is None:
                continue
            bio.seek(0)
            content = bio.read()
            bio.seek(0)
            assignments.append({
                "content":       content,
                "filename":      f"probe_e{e}_r{ri + 1}.csv",
                "step":          "probe",
                "concentration": 0.0,
            })
        for ci, rep_list in enumerate(cal_eis.get(key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            for ri, bio in enumerate(rep_list or []):
                if bio is None:
                    continue
                bio.seek(0)
                content = bio.read()
                bio.seek(0)
                assignments.append({
                    "content":       content,
                    "filename":      f"e{e}_c{ci + 1}_r{ri + 1}.csv",
                    "step":          "hybridization",
                    "concentration": conc,
                })

    return assignments


def main() -> None:
    st.title("📡 Analyse EIS — Spectroscopie d'impédance")
    st.caption("Fit Randles · DRT · Validation Kramers-Kronig · Calibration OLS")

    # Vérification que les données sont disponibles
    if not st.session_state.get("preprocessing_done", False):
        st.warning(
            "⚠️ Aucune donnée disponible. "
            "Importez et prétraitez vos données d'abord."
        )
        st.page_link("pages/0_import.py", label="→ Aller à l'import", icon="📂")
        st.stop()
        return

    # Récupérer les données prétraitées
    experiment = st.session_state["experiment_clean"]

    # Récupérer les résultats de validation KK si disponibles
    validation_results = st.session_state.get("validation_results", None)

    # Paramètres de fit
    with st.expander("⚙️ Modèles et paramètres physiques", expanded=False):
        col_models, col_phys = st.columns(2)
        with col_models:
            st.markdown("**Modèles de fit**")
            model_choices = {
                "circular":            "Fit circulaire",
                "randles_constrained": "Randles contraint",
                "randles_full":        "Randles complet",
                "drt_fft":             "DRT (FFT)",
            }
            active_models = [
                m for m, label in model_choices.items()
                if st.checkbox(label, value=(m == "circular"), key=f"eis_model_{m}")
            ]
        with col_phys:
            st.markdown("**Paramètres physiques**")
            phys_overrides = {
                "Fv": st.number_input("Fv (m³/s)", value=5e-10, format="%.2e", key="eis_Fv"),
                "xe": st.number_input("xe (m)",    value=30e-6, format="%.2e", key="eis_xe"),
                "h":  st.number_input("h (m)",     value=60e-6, format="%.2e", key="eis_h"),
                "d":  st.number_input("d (m)",     value=300e-6, format="%.2e", key="eis_d"),
                "T":  st.number_input("T (K)",     value=298.0, format="%.1f",  key="eis_T"),
                "C0": st.number_input("C0 (M)",    value=0.02,  format="%.4f",  key="eis_C0"),
            }

    cfg = _merge_overrides(_DEFAULT_CONFIG, phys_overrides)

    if st.button("↺ Relancer l'analyse", key="eis_rerun_btn"):
        st.session_state["eis_session"]    = None
        st.session_state["eis_validation"] = None
        st.rerun()

    if st.session_state.get("eis_session") is None:
        file_assignments = _build_file_assignments(experiment)
        if not file_assignments:
            st.warning("⚠️ Aucun spectre EIS trouvé dans l'expérience. Vérifiez le prétraitement.")
            return
        if not active_models:
            st.warning("⚠️ Sélectionnez au moins un modèle de fit dans les paramètres ci-dessus.")
            return
        with st.spinner("Analyse EIS en cours…"):
            try:
                session, vr_pipeline = run_pipeline(
                    file_assignments=file_assignments,
                    config=cfg,
                    active_models=active_models,
                )
                st.session_state["eis_session"]    = session
                st.session_state["eis_config"]     = cfg
                st.session_state["eis_validation"] = vr_pipeline or None
                st.success(f"✅ Analyse terminée — {len(session.groups)} groupe(s).")
            except Exception as exc:
                st.error(f"❌ Erreur lors de l'analyse EIS : {exc}")
                return

    if st.session_state.get("eis_session") is None:
        return

    vr = st.session_state.get("eis_validation") or validation_results
    render_eis_tabs(
        st.session_state["eis_session"],
        st.session_state.get("eis_config", cfg),
        validation_results=vr,
    )


if __name__ == "__main__":
    main()
