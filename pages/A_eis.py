"""Page A — Analyse EIS seule.

Charge les spectres d'impédance depuis experiment_clean (session_state),
effectue les fits Randles/DRT, valide par Kramers-Kronig, et produit
une calibration OLS log(Rct) vs log([c]).
"""

import numpy as np
import streamlit as st

from core.config import load_config, config_to_dict
from core.loader import load_spectrum, average_replicates
from core.pipeline import run_pipeline
from plotting.eis_plots import nyquist_figure_electrode
from ui.tabs import render_eis_tabs

_DEFAULT_CONFIG = config_to_dict(load_config())

# Seules ces deux méthodes sont affichées dans les graphes et tableaux EIS.
# Le pipeline peut en calculer d'autres en arrière-plan.
METHODS_TO_DISPLAY = ["randles_full", "drt_fft"]


def _filter_session_display(session):
    """Retourne une copie légère de la session avec seulement les méthodes à afficher."""
    from copy import deepcopy
    s = deepcopy(session)
    for sp in (s.bare, s.probe):
        if sp is not None:
            sp.fit_results = {k: v for k, v in sp.fit_results.items() if k in METHODS_TO_DISPLAY}
    for grp in s.groups:
        grp.fit_results = {k: v for k, v in grp.fit_results.items() if k in METHODS_TO_DISPLAY}
    return s


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


def _load_electrode_spectra(experiment: dict, elec_idx: int) -> list:
    """Charge et moyenne les spectres EIS pour une électrode.

    Retourne une liste de dicts {"label", "Zre", "Zim", "concentration"}
    pour le probe (concentration=0) et chaque concentration de calibration.
    """
    key = f"electrode_{elec_idx}"
    probe_dict = (experiment.get("probe") or {}).get("eis") or {}
    cal_dict   = (experiment.get("calibration") or {}).get("eis") or {}
    concs      = experiment.get("concentrations") or []

    result = []

    # Probe
    probe_files = probe_dict.get(key) or []
    probe_specs = []
    for ri, bio in enumerate(probe_files):
        if bio is None:
            continue
        try:
            bio.seek(0)
            sp = load_spectrum(bio.read(), label=f"probe_e{elec_idx}_r{ri+1}")
            bio.seek(0)
            probe_specs.append(sp)
        except Exception:
            pass
    if probe_specs:
        avg = average_replicates(probe_specs) if len(probe_specs) > 1 else probe_specs[0]
        result.append({"label": "Probe", "Zre": avg.Zre, "Zim": avg.Zim, "concentration": 0.0})

    # Calibration
    cal_by_conc = cal_dict.get(key) or []
    for ci, rep_files in enumerate(cal_by_conc):
        conc = concs[ci] if ci < len(concs) else 0.0
        exp = int(np.floor(np.log10(conc))) if conc > 0 else 0
        mant = conc / 10 ** exp if conc > 0 else 0
        lbl = f"C{ci+1} = {mant:.0f}×10{str(exp).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))} M"
        reps = []
        for ri, bio in enumerate(rep_files or []):
            if bio is None:
                continue
            try:
                bio.seek(0)
                sp = load_spectrum(bio.read(), label=f"e{elec_idx}_c{ci+1}_r{ri+1}")
                bio.seek(0)
                reps.append(sp)
            except Exception:
                pass
        if reps:
            avg = average_replicates(reps) if len(reps) > 1 else reps[0]
            result.append({"label": lbl, "Zre": avg.Zre, "Zim": avg.Zim, "concentration": conc})

    return result


def _render_three_nyquist(experiment: dict) -> None:
    """Affiche 3 graphes Nyquist côte à côte : E1, E2, Moyenne."""
    if experiment.get("mode") not in ("eis_only", "both"):
        return

    specs_e1 = _load_electrode_spectra(experiment, 1)
    specs_e2 = _load_electrode_spectra(experiment, 2) if experiment.get("n_electrodes", 2) >= 2 else []

    # Moyenne : pour chaque entrée, moyenne Zre/Zim entre E1 et E2
    specs_avg = []
    for s1 in specs_e1:
        match = next((s2 for s2 in specs_e2 if abs(s2["concentration"] - s1["concentration"]) < 1e-30), None)
        if match is not None and len(s1["Zre"]) == len(match["Zre"]):
            specs_avg.append({
                "label": s1["label"],
                "Zre": (np.asarray(s1["Zre"]) + np.asarray(match["Zre"])) / 2,
                "Zim": (np.asarray(s1["Zim"]) + np.asarray(match["Zim"])) / 2,
                "concentration": s1["concentration"],
            })
        else:
            specs_avg.append(s1)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if specs_e1:
            st.plotly_chart(nyquist_figure_electrode(specs_e1, title="Électrode 1"),
                            use_container_width=True, key="nyq_e1")
        else:
            st.info("Aucun spectre EIS — Électrode 1")
    with col2:
        if specs_e2:
            st.plotly_chart(nyquist_figure_electrode(specs_e2, title="Électrode 2"),
                            use_container_width=True, key="nyq_e2")
        else:
            st.info("Aucun spectre EIS — Électrode 2")
    with col3:
        if specs_avg:
            st.plotly_chart(nyquist_figure_electrode(specs_avg, title="Moyenne E1 + E2"),
                            use_container_width=True, key="nyq_avg")
        else:
            st.info("Moyenne non disponible")


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
            _displayed = {"randles_full", "drt_fft"}
            active_models = [
                m for m, label in model_choices.items()
                if st.checkbox(label, value=(m in _displayed), key=f"eis_model_{m}")
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

    # --- Diagrammes Nyquist par électrode ---
    st.markdown("### Diagrammes de Nyquist")
    _render_three_nyquist(experiment)

    st.divider()

    vr = st.session_state.get("eis_validation") or validation_results
    display_session = _filter_session_display(st.session_state["eis_session"])
    render_eis_tabs(
        display_session,
        st.session_state.get("eis_config", cfg),
        validation_results=vr,
    )


if __name__ == "__main__":
    main()
