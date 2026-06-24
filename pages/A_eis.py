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
from plotting.eis_plots import nyquist_figure_electrode, nyquist_normalized_figure, _spectrum_label
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
        for rep in grp.replicate_spectra:
            rep.fit_results = {k: v for k, v in rep.fit_results.items() if k in METHODS_TO_DISPLAY}
    for rep in s.bare_replicate_spectra:
        rep.fit_results = {k: v for k, v in rep.fit_results.items() if k in METHODS_TO_DISPLAY}
    for rep in s.probe_replicate_spectra:
        rep.fit_results = {k: v for k, v in rep.fit_results.items() if k in METHODS_TO_DISPLAY}
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


def _build_normalized_session(sessions: dict) -> dict:
    """Combine les sessions par électrode en spectres normalisés par concentration.

    Retourne {label_concentration: {"Zre_norm": array, "Zim_norm": array, "concentration": float}}
    en moyennant les versions normalisées de chaque électrode disponible.
    """
    def _normalize_electrode(session) -> dict:
        """Zre_norm/Zim_norm = |(Z_probe - Z_Ci) / Z_probe|, point à point, par concentration."""
        probe = session.probe if session is not None else None
        if probe is None:
            return {}
        out = {}
        for grp in session.groups:
            sp = grp.spectrum
            if len(sp.f) == len(probe.f):
                Zre_c, Zim_c = np.asarray(sp.Zre), np.asarray(sp.Zim)
            else:
                log_f_probe = np.log10(np.asarray(probe.f, dtype=float))
                log_f_c     = np.log10(np.asarray(sp.f, dtype=float))
                order = np.argsort(log_f_c)
                Zre_c = np.interp(log_f_probe, log_f_c[order], np.asarray(sp.Zre)[order])
                Zim_c = np.interp(log_f_probe, log_f_c[order], np.asarray(sp.Zim)[order])

            Zre_probe = np.asarray(probe.Zre)
            Zim_probe = np.asarray(probe.Zim)
            with np.errstate(invalid="ignore", divide="ignore"):
                Zre_norm = np.abs((Zre_probe - Zre_c) / Zre_probe)
                Zim_norm = np.abs((Zim_probe - Zim_c) / Zim_probe)

            out[grp.concentration] = {
                "label":         _spectrum_label(grp.spectrum),
                "f":             probe.f,
                "Zre_norm":      Zre_norm,
                "Zim_norm":      Zim_norm,
                "concentration": grp.concentration,
            }
        return out

    norm_e1 = _normalize_electrode(sessions.get(1))
    norm_e2 = _normalize_electrode(sessions.get(2))

    if not norm_e1 and not norm_e2:
        return {}
    if not norm_e2:
        return norm_e1
    if not norm_e1:
        return norm_e2

    result = {}
    all_concs = set(norm_e1) | set(norm_e2)
    for conc in all_concs:
        d1 = norm_e1.get(conc)
        d2 = norm_e2.get(conc)
        if d1 is not None and d2 is not None and len(d1["Zre_norm"]) == len(d2["Zre_norm"]):
            result[conc] = {
                "label":         d1["label"],
                "f":             d1["f"],
                "Zre_norm":      (d1["Zre_norm"] + d2["Zre_norm"]) / 2,
                "Zim_norm":      (d1["Zim_norm"] + d2["Zim_norm"]) / 2,
                "concentration": conc,
            }
        else:
            result[conc] = d1 if d1 is not None else d2
    return result


def _render_three_nyquist(experiment: dict, sessions: dict) -> None:
    """Affiche 3 graphes Nyquist côte à côte : E1, E2, Normalisé E1+E2."""
    if experiment.get("mode") not in ("eis_only", "both"):
        return

    specs_e1 = _load_electrode_spectra(experiment, 1)
    specs_e2 = _load_electrode_spectra(experiment, 2) if experiment.get("n_electrodes", 2) >= 2 else []

    normalized = _build_normalized_session(sessions)
    specs_norm = [
        {"label": d["label"], "Zre_norm": d["Zre_norm"], "Zim_norm": d["Zim_norm"], "concentration": d["concentration"]}
        for d in sorted(normalized.values(), key=lambda d: d["concentration"])
    ] if normalized else []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if specs_e1:
            st.plotly_chart(nyquist_figure_electrode(specs_e1, title="Électrode 1"),
                            width='stretch', key="nyq_e1")
        else:
            st.info("Aucun spectre EIS — Électrode 1")
    with col2:
        if specs_e2:
            st.plotly_chart(nyquist_figure_electrode(specs_e2, title="Électrode 2"),
                            width='stretch', key="nyq_e2")
        else:
            st.info("Aucun spectre EIS — Électrode 2")
    with col3:
        if specs_norm:
            st.plotly_chart(nyquist_normalized_figure(specs_norm, title="Normalisé E1 + E2"),
                            width='stretch', key="nyq_norm")
        else:
            st.info("Normalisation non disponible")


def _build_file_assignments_electrode(experiment: dict, elec_idx: int) -> list:
    """Convertit experiment_clean en liste de file_assignments pour une seule électrode."""
    mode = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    assignments = []

    if mode not in ("eis_only", "both"):
        return assignments

    probe_eis = (experiment.get("probe") or {}).get("eis") or {}
    cal_eis   = (experiment.get("calibration") or {}).get("eis") or {}

    key = f"electrode_{elec_idx}"
    for ri, bio in enumerate(probe_eis.get(key) or []):
        if bio is None:
            continue
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        assignments.append({
            "content":       content,
            "filename":      f"probe_e{elec_idx}_r{ri + 1}.csv",
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
                "filename":      f"e{elec_idx}_c{ci + 1}_r{ri + 1}.csv",
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
        st.session_state["eis_sessions"]    = None
        st.session_state["eis_validations"] = None
        st.rerun()

    if not st.session_state.get("eis_sessions"):
        if not active_models:
            st.warning("⚠️ Sélectionnez au moins un modèle de fit dans les paramètres ci-dessus.")
            return
        n_elec = experiment.get("n_electrodes", 2)
        sessions = {}
        validations = {}
        with st.spinner("Analyse EIS en cours…"):
            try:
                for e in range(1, n_elec + 1):
                    file_assignments = _build_file_assignments_electrode(experiment, e)
                    if not file_assignments:
                        continue
                    session, vr_pipeline = run_pipeline(
                        file_assignments=file_assignments,
                        config=cfg,
                        active_models=active_models,
                    )
                    sessions[e] = session
                    validations[e] = vr_pipeline or None
            except Exception as exc:
                st.error(f"❌ Erreur lors de l'analyse EIS : {exc}")
                return

        if not sessions:
            st.warning("⚠️ Aucun spectre EIS trouvé dans l'expérience. Vérifiez le prétraitement.")
            return

        st.session_state["eis_sessions"]    = sessions
        st.session_state["eis_config"]      = cfg
        st.session_state["eis_validations"] = validations
        n_groups = sum(len(s.groups) for s in sessions.values())
        st.success(f"✅ Analyse terminée — {n_groups} groupe(s) au total sur {len(sessions)} électrode(s).")

    sessions = st.session_state.get("eis_sessions")
    if not sessions:
        return

    # --- Diagrammes Nyquist par électrode ---
    st.markdown("### Diagrammes de Nyquist")
    _render_three_nyquist(experiment, sessions)

    st.divider()

    validations = st.session_state.get("eis_validations") or {}
    if not validations and validation_results:
        main_elec = next(iter(sorted(sessions)), None)
        if main_elec is not None:
            validations = {main_elec: validation_results}

    display_sessions = {e: _filter_session_display(s) for e, s in sessions.items()}
    normalized = _build_normalized_session(sessions)
    st.session_state["eis_normalized"] = normalized
    render_eis_tabs(
        display_sessions,
        normalized,
        st.session_state.get("eis_config", cfg),
        validations,
    )


if __name__ == "__main__":
    main()
