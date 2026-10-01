"""Page A — Analyse EIS seule.

Charge les spectres d'impédance depuis experiment_clean (session_state), puis lance
core.pipeline.run_pipeline par électrode : pour chaque groupe de réplicats, measurement
model + verdict Kramers-Kronig AVANT le fit, fit Orazem du circuit défini ici (chaque
réplicat + la moyenne) et DRT (chaque réplicat + la moyenne). Les groupes arrêtés
(structure d'erreur non caractérisable, AUDIT.md ERR-1) sont signalés explicitement.
"""

import math

import numpy as np
import pandas as pd
import streamlit as st

from circuit import CircuitError, parse_circuit
from core.app_state import preprocessing_ready
from core.config import load_config, config_to_dict
from core.loader import load_spectrum, average_replicates
from core.pipeline import InvalidAnalysisInput, build_circuit_fit, run_pipeline
from plotting.eis_plots import nyquist_figure_electrode, nyquist_normalized_figure, _spectrum_label
from ui.tabs import render_eis_tabs

_DEFAULT_CONFIG = config_to_dict(load_config())

#: Clés de st.session_state des résultats EIS (sous-ensemble de core.app_state).
_EIS_RESULT_KEYS = ("eis_sessions", "eis_validations", "eis_normalized")


def _collect_fit_diagnostics(sessions: dict) -> list:
    """Alertes d'ajustement à remonter à l'utilisateur (I7) : alertes de chaque fit du
    spectre MOYEN (circuit et DRT) et alertes de groupe (verdict KK, réplicats écartés,
    hétérogénéité, DRT non calculée…). Les groupes arrêtés sont signalés à part."""
    out = []
    for e, session in sorted(sessions.items()):
        for label, mean_sp, _reps, an in session.iter_groups():
            if an is not None and not an.ok:
                continue
            for model, fr in mean_sp.fit_results.items():
                msgs = list(getattr(fr, "warnings", []) or [])
                if not fr.converged and not any("convergé" in m for m in msgs):
                    msgs.append("ajustement non convergé")
                for m in msgs:
                    out.append(f"E{e} · {label} (moyenne) · {model} : {m}")
            for m in (an.warnings if an is not None else []):
                out.append(f"E{e} · {label} : {m}")
    return out


def _render_analysis_status(sessions: dict) -> None:
    """Statut EXPLICITE de l'analyse : groupes arrêtés, fichiers écartés, moteur DRT.

    Jamais « ✅ Analyse terminée » quand un groupe n'a produit aucun fit (ERR-1).
    """
    n_total = n_failed = 0
    for e, session in sorted(sessions.items()):
        for msg in session.messages:
            st.warning(f"Électrode {e} : {msg}")
        for le in session.load_errors:
            st.warning(f"Électrode {e} : fichier « {le['filename']} » écarté — {le['message']}")
        for _lbl, _sp, _reps, an in session.iter_groups():
            n_total += 1
            if an is not None and not an.ok:
                n_failed += 1
                st.error(f"Électrode {e} — {an.message}")
    if n_total == 0:
        st.warning("⚠️ Aucun spectre EIS exploitable. Vérifiez le prétraitement.")
    elif n_failed:
        st.error(
            f"❌ Analyse terminée avec **{n_failed} groupe(s) ARRÊTÉ(S)** sur {n_total} : "
            "aucun fit (ni circuit ni DRT) pour ces groupes — voir les messages ci-dessus."
        )
    else:
        st.success(f"✅ Analyse terminée — {n_total} groupe(s) analysé(s) sur "
                   f"{len(sessions)} électrode(s).")


def _num_or_none(v):
    """Cellule du tableau → float, ou None si vide / NaN."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _circuit_editor(defaults: dict):
    """Saisie du circuit, des guess/bornes et du paramètre cible.

    Returns:
        (CircuitFit ou None si la saisie est invalide — l'erreur est affichée, description
        sérialisable de la saisie pour détecter une modification après l'analyse).
    """
    c = defaults["fit"]["circuit"]
    expr = st.text_area(
        "Circuit équivalent (syntaxe : docs/CIRCUIT_UTILISATEUR.md)",
        value=c["expression"], key="eis_circuit_expr", height=80,
        help="Ex. Re + parallel(R(Rct), Q(Qdl, alpha)). `+` = série, parallel(a, b, …) = "
             "parallèle ; tout nom autre que w et les éléments est un paramètre ajusté.",
    )
    try:
        _z, names = parse_circuit(expr)
    except CircuitError as exc:
        st.error(f"Circuit invalide : {exc}")
        return None, None
    params_cfg = c.get("parameters") or {}
    table = pd.DataFrame([
        {"paramètre": n,
         "initial": (params_cfg.get(n) or {}).get("initial"),
         "borne basse": (params_cfg.get(n) or {}).get("lower"),
         "borne haute": (params_cfg.get(n) or {}).get("upper")}
        for n in names
    ])
    st.caption(
        "Guess de départ et bornes de chaque paramètre (case vide = non borné). Valeurs "
        "proposées : config/default.yaml (`fit.circuit`) ; un paramètre nouveau doit être "
        "renseigné. Le fit part de ce guess puis de 7 perturbations (±0,5 décade)."
    )
    edited = st.data_editor(table, key=f"eis_circuit_params_{'_'.join(names)}",
                            disabled=["paramètre"], hide_index=True, width='stretch')
    default_target = c.get("target_param")
    target = st.selectbox("Paramètre cible (signal de calibration)", names,
                          index=names.index(default_target) if default_target in names else 0,
                          key="eis_circuit_target")
    params = {}
    for _i, row in edited.iterrows():
        lo, hi = _num_or_none(row["borne basse"]), _num_or_none(row["borne haute"])
        params[row["paramètre"]] = {
            "initial": _num_or_none(row["initial"]),
            "lower": -math.inf if lo is None else lo,
            "upper": math.inf if hi is None else hi,
        }
    fingerprint = {"expression": expr, "parameters": params, "target": target}
    try:
        return build_circuit_fit(expr, params, target), fingerprint
    except InvalidAnalysisInput as exc:
        st.error(str(exc))
        return None, fingerprint


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


def _load_bare_eis(experiment: dict, elec_idx: int):
    """Charge et moyenne les fichiers « électrode nue » EIS d'une électrode.

    Réutilise les loaders EXISTANTS (load_spectrum + average_replicates) — aucun
    parsing maison. Retourne un EISSpectrum (moyenne des réplicats, UNE seule
    trace) ou None. Référence d'AFFICHAGE SEULE : jamais passée au pipeline.
    """
    key = f"electrode_{elec_idx}"
    bare_dict = (experiment.get("bare") or {}).get("eis") or {}
    bare_files = bare_dict.get(key) or []

    specs = []
    for ri, bio in enumerate(bare_files):
        if bio is None:
            continue
        try:
            bio.seek(0)
            sp = load_spectrum(
                bio.read(),
                label="Électrode nue (réf.)",
                concentration=0.0,
                step="bare",
            )
            bio.seek(0)
            specs.append(sp)
        except Exception:
            pass

    if not specs:
        return None
    return average_replicates(specs) if len(specs) > 1 else specs[0]


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

    bare_e1 = sessions.get(1).bare_reference if sessions.get(1) else None
    bare_e2 = sessions.get(2).bare_reference if sessions.get(2) else None

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if specs_e1:
            st.plotly_chart(nyquist_figure_electrode(specs_e1, title="Électrode 1", bare=bare_e1),
                            width='stretch', key="nyq_e1")
        else:
            st.info("Aucun spectre EIS — Électrode 1")
    with col2:
        if specs_e2:
            st.plotly_chart(nyquist_figure_electrode(specs_e2, title="Électrode 2", bare=bare_e2),
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
    st.caption("Measurement model & Kramers-Kronig · Fit Orazem du circuit · DRT · Calibration")

    # Vérification que les données sont disponibles (B-STATE-b : jamais de KeyError)
    if not preprocessing_ready(st.session_state):
        st.warning(
            "⚠️ Aucune donnée disponible. "
            "Importez et prétraitez vos données d'abord."
        )
        st.page_link("pages/0_import.py", label="→ Aller à l'import", icon="📂")
        st.stop()

    experiment = st.session_state["experiment_clean"]

    with st.expander("⚙️ Circuit, DRT et paramètres physiques", expanded=False):
        circuit_fit, circuit_inputs = _circuit_editor(_DEFAULT_CONFIG)

        st.markdown("**DRT** (drt/engine.py — chaque réplicat ET la moyenne de chaque groupe)")
        drt_cfg = _DEFAULT_CONFIG["fit"]["drt"]
        run_drt = st.checkbox("Calculer la DRT", value=bool(drt_cfg.get("enabled", True)),
                              key="eis_drt_enabled")
        drt_mode = st.radio(
            "Mode DRT", ["optimize", "sample"],
            index=0 if drt_cfg.get("mode", "optimize") == "optimize" else 1,
            format_func=lambda m: ("MAP (optimize) — ~1 s/spectre, sans diagnostic de convergence"
                                   if m == "optimize" else
                                   "HMC (sample) — 2 à 5 min/spectre, R̂/divergences/ESS + intervalles"),
            key="eis_drt_mode", disabled=not run_drt,
        )

        st.markdown("**Paramètres physiques**")
        phys_overrides = {
            "Fv": st.number_input("Fv (m³/s)", value=5e-10, format="%.2e", key="eis_Fv"),
            "xe": st.number_input("xe (m)",    value=30e-6, format="%.2e", key="eis_xe"),
            "h":  st.number_input("h (m)",     value=60e-6, format="%.2e", key="eis_h"),
            "d":  st.number_input("d (m)",     value=300e-6, format="%.2e", key="eis_d"),
            "T":  st.number_input("T (K)",     value=298.0, format="%.1f",  key="eis_T"),
            "C0": st.number_input("C0 (M)",    value=0.02,  format="%.4f",  key="eis_C0"),
        }

        st.markdown("**Pondération du fit**")
        st.caption(
            "⚖️ Méthode UNIQUE : structure d'erreur d'Orazem σ = α|Zj| + β|Zr − R_sol| + "
            "γ|Z|² + δ, caractérisée par le measurement model sur les réplicats BRUTS de "
            "CHAQUE groupe (≥ 3), à chaque analyse ; poids = 1/σ² (σ/√n pour la moyenne). "
            "χ²ᵣ ≈ 1 est un vrai test d'adéquation. Sans réplicats en nombre suffisant, le "
            "groupe est **arrêté** avec un message (aucun repli arbitraire)."
        )

    cfg = _merge_overrides(_DEFAULT_CONFIG, phys_overrides)
    inputs = {"circuit": circuit_inputs, "run_drt": run_drt, "drt_mode": drt_mode}

    if st.button("↺ Relancer l'analyse", key="eis_rerun_btn"):
        for key in _EIS_RESULT_KEYS:
            st.session_state[key] = None
        st.rerun()

    if not st.session_state.get("eis_sessions"):
        if circuit_fit is None:
            st.warning("⚠️ Corrigez le circuit ou ses paramètres (section ⚙️ ci-dessus) "
                       "pour lancer l'analyse.")
            return
        n_elec = experiment.get("n_electrodes", 2)
        sessions = {}
        validations = {}
        with st.spinner("Analyse EIS en cours…" + (" (DRT HMC : plusieurs minutes par spectre)"
                                                   if run_drt and drt_mode == "sample" else "")):
            try:
                for e in range(1, n_elec + 1):
                    file_assignments = _build_file_assignments_electrode(experiment, e)
                    if not file_assignments:
                        continue
                    session, vr_pipeline = run_pipeline(
                        file_assignments, cfg, circuit_fit, run_drt=run_drt, drt_mode=drt_mode,
                    )
                    # Référence « électrode nue » — attachée APRÈS l'analyse,
                    # jamais lue par run_pipeline (affichage seul).
                    session.bare_reference = _load_bare_eis(experiment, e)
                    sessions[e] = session
                    validations[e] = vr_pipeline or None
            except InvalidAnalysisInput as exc:          # saisie invalide : message clair
                st.error(f"❌ {exc}")
                return
            except Exception as exc:                     # bug logiciel : jamais avalé
                st.error(
                    "❌ Erreur LOGICIELLE pendant l'analyse EIS — elle ne vient pas de vos "
                    "données. Merci de la signaler avec le détail ci-dessous."
                )
                st.exception(exc)
                return

        st.session_state["eis_sessions"]    = sessions
        st.session_state["eis_config"]      = cfg
        st.session_state["eis_validations"] = validations
        st.session_state["eis_inputs"]      = inputs

    sessions = st.session_state.get("eis_sessions")
    if not sessions:
        st.warning("⚠️ Aucun spectre EIS trouvé dans l'expérience. Vérifiez le prétraitement.")
        return

    if st.session_state.get("eis_inputs") != inputs:
        st.info("ℹ️ Le circuit ou les réglages DRT ont changé depuis cette analyse : "
                "cliquez sur **↺ Relancer l'analyse** pour les appliquer.")

    _render_analysis_status(sessions)

    # I7 : remonter les diagnostics d'ajustement à l'utilisateur, pas dans les logs.
    diags = _collect_fit_diagnostics(sessions)
    if diags:
        with st.expander(f"⚠️ Diagnostics d'ajustement ({len(diags)})", expanded=False):
            for line in diags:
                st.warning(line)

    st.markdown("### Diagrammes de Nyquist")
    _render_three_nyquist(experiment, sessions)

    st.divider()

    normalized = _build_normalized_session(sessions)
    st.session_state["eis_normalized"] = normalized
    render_eis_tabs(
        sessions,
        st.session_state.get("eis_config", cfg),
        st.session_state.get("eis_validations") or {},
    )


if __name__ == "__main__":
    main()
