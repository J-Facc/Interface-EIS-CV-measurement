"""Page 1 — Prétraitement et validation des données.

Visualisation interactive des spectres EIS / courbes CV,
validation Kramers-Kronig, exclusion des réplicats aberrants,
et validation finale avant l'analyse.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.loader import load_spectrum
from core.cv_loader import load_cv_file
from core.models import EISSpectrum
from core.cv_models import CVScan
from core.validator import validate_spectrum, validate_replicate_group, KKResult
from plotting.eis_plots import nyquist_figure as _nyquist_figure


# ─────────────────────────────────────────────
# Guard — import obligatoire
# ─────────────────────────────────────────────

def _check_import() -> bool:
    if "import_validated" not in st.session_state:
        st.warning("⚠️ Importez d'abord vos données dans l'onglet **Import**.")
        st.page_link("pages/0_import.py", label="Aller à l'import", icon="📂")
        st.stop()
        return False
    return True


# ─────────────────────────────────────────────
# Chargement des spectres depuis session_state
# ─────────────────────────────────────────────

def _load_eis_spectra(experiment: dict) -> Dict[str, Any]:
    """
    Charge tous les spectres EIS depuis experiment.
    Retourne un dict structuré par (type, electrode, concentration_idx, replicat_idx).
    """
    calibration = (experiment.get("calibration") or {}).get("eis") or {}
    probe_dict  = (experiment.get("probe") or {}).get("eis") or {}
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    spectra: Dict[str, Any] = {"probe": {}, "calibration": {}}

    # Probe
    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        rep_list = probe_dict.get(key) or []
        loaded = []
        for ri, bio in enumerate(rep_list):
            sp = _bio_to_eis(bio, f"probe_e{e}_r{ri+1}")
            if sp is not None:
                loaded.append(sp)
        spectra["probe"][key] = loaded

    # Calibration
    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        concs_list = calibration.get(key) or []
        spectra["calibration"][key] = []
        for ci, rep_files in enumerate(concs_list):
            loaded = []
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            for ri, bio in enumerate(rep_files or []):
                sp = _bio_to_eis(bio, f"e{e}_c{ci+1}_r{ri+1}")
                if sp is not None:
                    loaded.append(sp)
            spectra["calibration"][key].append(loaded)

    return spectra


def _bio_to_eis(bio, label: str):
    """Charge un BytesIO comme spectre EIS. Retourne None en cas d'erreur."""
    if bio is None:
        return None
    try:
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        sp = load_spectrum(content, label=label)
        return sp
    except Exception as exc:
        return None


def _bio_to_cv(bio, label: str, concentration: float, step: str):
    """Charge un BytesIO comme scan CV. Retourne None en cas d'erreur."""
    if bio is None:
        return None
    try:
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        return load_cv_file(content, label=label,
                            concentration=concentration, step=step)
    except Exception:
        return None


def _load_cv_scans(experiment: dict) -> Dict[str, Any]:
    calibration = (experiment.get("calibration") or {}).get("cv") or {}
    probe_dict  = (experiment.get("probe") or {}).get("cv") or {}
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    scans: Dict[str, Any] = {"probe": {}, "calibration": {}}

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        rep_list = probe_dict.get(key) or []
        loaded = []
        for ri, bio in enumerate(rep_list):
            sc = _bio_to_cv(bio, f"probe_e{e}_r{ri+1}", 0.0, "probe")
            if sc is not None:
                loaded.append(sc)
        scans["probe"][key] = loaded

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        concs_list = calibration.get(key) or []
        scans["calibration"][key] = []
        for ci, rep_files in enumerate(concs_list):
            loaded = []
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            for ri, bio in enumerate(rep_files or []):
                sc = _bio_to_cv(bio, f"e{e}_c{ci+1}_r{ri+1}", conc, "hybridization")
                if sc is not None:
                    loaded.append(sc)
            scans["calibration"][key].append(loaded)

    return scans


# ─────────────────────────────────────────────
# Onglet 1 — Visualisation
# ─────────────────────────────────────────────

def _tab_visualisation(experiment: dict, exclusions: dict) -> None:
    mode = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    has_eis = mode in ("eis_only", "both")
    has_cv  = mode in ("cv_only", "both")

    if has_eis:
        st.markdown("### 📡 Spectres EIS")
        eis_spectra = _load_eis_spectra(experiment)
        _visualize_eis(eis_spectra, concentrations, n_elec, exclusions)

    if has_cv:
        st.markdown("### 📈 Voltammogrammes CV")
        cv_scans = _load_cv_scans(experiment)
        _visualize_cv(cv_scans, concentrations, n_elec, exclusions)


def _visualize_eis(
    spectra: dict,
    concentrations: list,
    n_elec: int,
    exclusions: dict,
) -> None:
    elec_opts  = [f"Électrode {e}" for e in range(1, n_elec + 1)]
    conc_labels = ["Probe"] + [f"{c:.2e} M" for c in concentrations]

    c1, c2 = st.columns(2)
    with c1:
        sel_elec = st.selectbox("Électrode", elec_opts, key="vis_eis_elec")
    with c2:
        sel_conc = st.selectbox("Concentration", conc_labels, key="vis_eis_conc")

    e_idx = elec_opts.index(sel_elec) + 1
    elec_key = f"electrode_{e_idx}"

    if sel_conc == "Probe":
        reps = spectra["probe"].get(elec_key) or []
        group_label = f"probe_{elec_key}"
    else:
        ci = conc_labels.index(sel_conc) - 1
        conc_list = spectra["calibration"].get(elec_key) or []
        reps = conc_list[ci] if ci < len(conc_list) else []
        group_label = f"eis_{elec_key}_c{ci}"

    if not reps:
        st.info("Aucun spectre chargé pour cette sélection.")
        return

    # Exclusion checkboxes (courbes entières)
    _render_exclusion_checkboxes(reps, group_label, exclusions)

    # Nyquist superposé avec mode édition par réplicat
    st.markdown("**Diagramme de Nyquist**")
    for ri, sp in enumerate(reps):
        excluded_curve = _is_excluded(group_label, ri, exclusions)
        spectrum_label = f"{group_label}_r{ri}"

        edit_mode = st.toggle(
            f"✏️ Mode suppression de points — R{ri+1}",
            key=f"edit_{spectrum_label}",
        )

        if edit_mode:
            fig = _nyquist_figure(spectrum=sp, selection_mode=True)
            event = st.plotly_chart(
                fig,
                width='stretch',
                key=f"nyquist_edit_{spectrum_label}",
                on_select="rerun",
            )

            selected_points = []
            if event and event.selection and event.selection.points:
                selected_points = [
                    p['point_index'] for p in event.selection.points
                ]

            if selected_points:
                st.warning(
                    f"{len(selected_points)} point(s) sélectionné(s) : "
                    f"indices {selected_points}"
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(
                        "🗑️ Supprimer ces points",
                        key=f"del_{spectrum_label}",
                    ):
                        key = f"deleted_points_{spectrum_label}"
                        existing = st.session_state.get(key, [])
                        st.session_state[key] = list(
                            set(existing + selected_points)
                        )
                        st.success(f"Points supprimés du spectre R{ri+1}")
                        st.rerun()

                with col_b:
                    key = f"deleted_points_{spectrum_label}"
                    if st.session_state.get(key):
                        if st.button(
                            "↩️ Restaurer tous les points",
                            key=f"restore_{spectrum_label}",
                        ):
                            st.session_state[key] = []
                            st.rerun()

            key = f"deleted_points_{spectrum_label}"
            deleted = st.session_state.get(key, [])
            if deleted:
                st.info(
                    f"Points actuellement exclus de R{ri+1} : "
                    f"indices {sorted(deleted)}"
                )
        else:
            key = f"deleted_points_{spectrum_label}"
            deleted = st.session_state.get(key, [])
            opacity = 0.35 if excluded_curve else 1.0
            fig = _nyquist_figure(
                spectrum=sp,
                excluded_indices=deleted if deleted else None,
                selection_mode=False,
            )
            fig.update_traces(opacity=opacity)
            st.plotly_chart(fig, width='stretch', key=f"nyquist_view_{spectrum_label}")

    # Bode
    fig_bode = go.Figure()
    for ri, sp in enumerate(reps):
        excluded = _is_excluded(group_label, ri, exclusions)
        dash = "dot" if excluded else "solid"
        opacity = 0.35 if excluded else 1.0
        name = f"R{ri+1}" + (" [Exclu]" if excluded else "")
        Z_mod = np.sqrt(np.array(sp.Zre)**2 + np.array(sp.Zim)**2)
        fig_bode.add_trace(go.Scatter(
            x=sp.f, y=Z_mod,
            mode="lines", name=name,
            line=dict(dash=dash),
            opacity=opacity,
        ))
    fig_bode.update_layout(
        title="Bode — Module |Z| vs Fréquence",
        xaxis_title="Fréquence (Hz)",
        xaxis_type="log",
        yaxis_title="|Z| (Ω)",
        yaxis_type="log",
    )
    st.plotly_chart(fig_bode, width='stretch')


def _cv_figure_single(
    sc,
    excluded_indices: list = None,
    selection_mode: bool = False,
) -> go.Figure:
    """Plotly figure for a single CV scan with optional point exclusion highlighting."""
    fig = go.Figure()
    E = np.array(sc.E)
    I = np.array(sc.I) * 1e6

    if excluded_indices:
        excl_set = set(excluded_indices)
        keep_idx = [i for i in range(len(E)) if i not in excl_set]
        excl_idx = sorted(excl_set)
        fig.add_trace(go.Scatter(
            x=E[keep_idx], y=I[keep_idx],
            mode="lines+markers",
            name=sc.label,
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=E[excl_idx], y=I[excl_idx],
            mode="markers",
            name="Points exclus",
            marker=dict(color="lightgray", size=5, symbol="x",
                        line=dict(width=1, color="gray")),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=E, y=I,
            mode="lines+markers",
            name=sc.label,
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=f"CV — {sc.label}",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
    )
    if selection_mode:
        fig.update_layout(dragmode="select", clickmode="event+select")
    return fig


def _visualize_cv(
    scans: dict,
    concentrations: list,
    n_elec: int,
    exclusions: dict,
) -> None:
    elec_opts  = [f"Électrode {e}" for e in range(1, n_elec + 1)]
    conc_labels = ["Probe"] + [f"{c:.2e} M" for c in concentrations]

    c1, c2 = st.columns(2)
    with c1:
        sel_elec = st.selectbox("Électrode", elec_opts, key="vis_cv_elec")
    with c2:
        sel_conc = st.selectbox("Concentration", conc_labels, key="vis_cv_conc")

    e_idx = elec_opts.index(sel_elec) + 1
    elec_key = f"electrode_{e_idx}"

    if sel_conc == "Probe":
        reps = scans["probe"].get(elec_key) or []
        group_label = f"cv_probe_{elec_key}"
    else:
        ci = conc_labels.index(sel_conc) - 1
        conc_list = scans["calibration"].get(elec_key) or []
        reps = conc_list[ci] if ci < len(conc_list) else []
        group_label = f"cv_{elec_key}_c{ci}"

    if not reps:
        st.info("Aucun scan CV chargé pour cette sélection.")
        return

    _render_exclusion_checkboxes(reps, group_label, exclusions)

    st.markdown("**Voltammogrammes**")
    for ri, sc in enumerate(reps):
        excluded_curve = _is_excluded(group_label, ri, exclusions)
        spectrum_label = f"{group_label}_r{ri}"

        edit_mode = st.toggle(
            f"✏️ Mode suppression de points — R{ri+1}",
            key=f"edit_{spectrum_label}",
        )

        if edit_mode:
            fig = _cv_figure_single(sc, selection_mode=True)
            event = st.plotly_chart(
                fig,
                width='stretch',
                key=f"cv_edit_{spectrum_label}",
                on_select="rerun",
            )

            selected_points = []
            if event and event.selection and event.selection.points:
                selected_points = [
                    p['point_index'] for p in event.selection.points
                ]

            if selected_points:
                st.warning(
                    f"{len(selected_points)} point(s) sélectionné(s) : "
                    f"indices {selected_points}"
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(
                        "🗑️ Supprimer ces points",
                        key=f"del_{spectrum_label}",
                    ):
                        key = f"deleted_points_{spectrum_label}"
                        existing = st.session_state.get(key, [])
                        st.session_state[key] = list(
                            set(existing + selected_points)
                        )
                        st.success(f"Points supprimés de R{ri+1}")
                        st.rerun()

                with col_b:
                    key = f"deleted_points_{spectrum_label}"
                    if st.session_state.get(key):
                        if st.button(
                            "↩️ Restaurer tous les points",
                            key=f"restore_{spectrum_label}",
                        ):
                            st.session_state[key] = []
                            st.rerun()

            key = f"deleted_points_{spectrum_label}"
            deleted = st.session_state.get(key, [])
            if deleted:
                st.info(
                    f"Points actuellement exclus de R{ri+1} : "
                    f"indices {sorted(deleted)}"
                )
        else:
            key = f"deleted_points_{spectrum_label}"
            deleted = st.session_state.get(key, [])
            opacity = 0.35 if excluded_curve else 1.0
            fig = _cv_figure_single(sc, excluded_indices=deleted if deleted else None)
            fig.update_traces(opacity=opacity)
            st.plotly_chart(fig, width='stretch', key=f"cv_view_{spectrum_label}")


def _render_exclusion_checkboxes(reps, group_label: str, exclusions: dict) -> None:
    """Affiche des checkboxes d'exclusion pour chaque réplicat."""
    if not reps:
        return
    cols = st.columns(len(reps))
    for ri, (col, _) in enumerate(zip(cols, reps)):
        with col:
            excluded = _is_excluded(group_label, ri, exclusions)
            new_val = st.checkbox(
                f"Exclure R{ri+1}",
                value=excluded,
                key=f"excl_{group_label}_{ri}",
            )
            if new_val != excluded:
                if group_label not in exclusions:
                    exclusions[group_label] = set()
                if new_val:
                    exclusions[group_label].add(ri)
                else:
                    exclusions[group_label].discard(ri)
                st.session_state["exclusions"] = exclusions


def _is_excluded(group_label: str, ri: int, exclusions: dict) -> bool:
    return ri in exclusions.get(group_label, set())


# ─────────────────────────────────────────────
# Onglet 2 — Validation KK
# ─────────────────────────────────────────────

def _tab_validation_kk(experiment: dict, exclusions: dict) -> None:
    mode = experiment.get("mode", "both")
    has_eis = mode in ("eis_only", "both")
    has_cv  = mode in ("cv_only", "both")

    if has_eis:
        st.markdown("### 🔍 Validation Kramers-Kronig (EIS)")
        eis_spectra = _load_eis_spectra(experiment)
        _render_kk_section(eis_spectra, experiment, exclusions)

    if has_cv:
        st.markdown("### 📊 Validation CV")
        cv_scans = _load_cv_scans(experiment)
        _render_cv_validation(cv_scans, experiment, exclusions)


def _render_kk_section(eis_spectra: dict, experiment: dict, exclusions: dict) -> None:
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    # Construire toutes les lignes du tableau
    rows = []
    kk_cache: Dict[str, KKResult] = {}

    for e in range(1, n_elec + 1):
        elec_key = f"electrode_{e}"
        # Probe
        for ri, sp in enumerate(eis_spectra["probe"].get(elec_key) or []):
            label = f"probe_e{e}_r{ri+1}"
            kk = validate_spectrum(np.array(sp.f), np.array(sp.Zre), np.array(sp.Zim), label)
            kk_cache[label] = kk
            group_label = f"probe_{elec_key}"
            excluded = _is_excluded(group_label, ri, exclusions)
            rows.append(_kk_row(label, kk, group_label, ri, excluded))
        # Calibration
        for ci, rep_list in enumerate(eis_spectra["calibration"].get(elec_key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            group_label = f"eis_{elec_key}_c{ci}"
            for ri, sp in enumerate(rep_list):
                label = f"e{e}_c{ci+1}_r{ri+1} ({conc:.1e} M)"
                kk = validate_spectrum(np.array(sp.f), np.array(sp.Zre), np.array(sp.Zim), label)
                kk_cache[label] = kk
                excluded = _is_excluded(group_label, ri, exclusions)
                rows.append(_kk_row(label, kk, group_label, ri, excluded))

    if not rows:
        st.info("Aucun spectre EIS chargé pour la validation KK.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df[["Spectre", "µ", "χ²", "Plage valide", "Verdict"]],
        use_container_width=True,
        hide_index=True,
    )

    # Boutons d'action globaux
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚫 Exclure tous les invalides", key="kk_excl_invalid"):
            for row in rows:
                if row["_is_valid"] is False:
                    gl = row["_group_label"]
                    ri = row["_rep_idx"]
                    if gl not in exclusions:
                        exclusions[gl] = set()
                    exclusions[gl].add(ri)
            st.session_state["exclusions"] = exclusions
            st.rerun()
    with col3:
        if st.button("🔄 Réinitialiser les exclusions", key="kk_reset"):
            st.session_state["exclusions"] = {}
            st.rerun()

    # Résidus pour le spectre sélectionné
    st.markdown("**Résidus KK — spectre sélectionné**")
    label_opts = [r["Spectre"] for r in rows]
    sel_label = st.selectbox("Spectre", label_opts, key="kk_sel_spectre")
    if sel_label and sel_label in kk_cache:
        kk = kk_cache[sel_label]
        _plot_kk_residuals(kk)

    # Expander d'interprétation
    with st.expander("ℹ️ Interpréter les résultats KK"):
        st.markdown("""
**µ < 0.85** : proportion de capacités RC négatives acceptable.
Au-delà, le système dévie des hypothèses KK (non-stationnarité, non-linéarité).

**Résidus < 2%** : cohérence Re/Im satisfaisante sur la plage considérée.

**Drift inter-réplicats** : si les résidus augmentent systématiquement
du 1er au 3e réplicat, la surface évolue pendant la mesure.
        """)


def _kk_row(label: str, kk: KKResult, group_label: str, ri: int, excluded: bool) -> dict:
    verdict = "✅ Valide" if kk.is_valid else "❌ Invalide"
    if excluded:
        verdict += " [Exclu]"
    plage = f"{kk.f_min_valid:.1f}–{kk.f_max_valid:.0f} Hz"
    return {
        "Spectre":      label,
        "µ":            f"{kk.mu:.3f}",
        "χ²":           f"{kk.chi2_pseudo:.4f}",
        "Plage valide": plage,
        "Verdict":      verdict,
        "_is_valid":    kk.is_valid,
        "_group_label": group_label,
        "_rep_idx":     ri,
    }


def _plot_kk_residuals(kk: KKResult) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kk.frequencies, y=kk.residuals_re,
        mode="lines+markers", name="Résidus Re(Z) (%)",
        line=dict(color="steelblue"),
    ))
    fig.add_trace(go.Scatter(
        x=kk.frequencies, y=kk.residuals_im,
        mode="lines+markers", name="Résidus Im(Z) (%)",
        line=dict(color="firebrick"),
    ))
    # Seuil ±2 %
    for val, color in [(2, "red"), (-2, "red")]:
        fig.add_hline(y=val, line_dash="dot", line_color=color, opacity=0.6)
    # Plage valide
    fig.add_vrect(
        x0=kk.f_min_valid, x1=kk.f_max_valid,
        fillcolor="lightgreen", opacity=0.15,
        layer="below", line_width=0,
        annotation_text="Plage valide",
    )
    fig.update_layout(
        title=f"Résidus KK — {kk.label}",
        xaxis_title="Fréquence (Hz)",
        xaxis_type="log",
        yaxis_title="Résidu normalisé (%)",
    )
    st.plotly_chart(fig, width='stretch')


def _render_cv_validation(cv_scans: dict, experiment: dict, exclusions: dict) -> None:
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    rows = []
    for e in range(1, n_elec + 1):
        elec_key = f"electrode_{e}"
        for ci, rep_list in enumerate(cv_scans["calibration"].get(elec_key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            group_label = f"cv_{elec_key}_c{ci}"
            for ri, sc in enumerate(rep_list):
                excluded = _is_excluded(group_label, ri, exclusions)
                row = _cv_validation_row(sc, e, ci, ri, conc, group_label, excluded)
                rows.append(row)

    if not rows:
        st.info("Aucun scan CV chargé pour la validation.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df[["Spectre", "I_pic_ox (µA)", "I_pic_red (µA)", "Symétrie", "E_pic_ox (V)", "Verdict"]],
        use_container_width=True,
        hide_index=True,
    )


def _cv_validation_row(sc, e, ci, ri, conc, group_label, excluded):
    I = np.array(sc.I)
    E = np.array(sc.E)
    idx_ox  = int(np.argmax(I))
    idx_red = int(np.argmin(I))
    I_ox  = float(I[idx_ox])
    I_red = float(I[idx_red])
    sym   = abs(I_ox) / max(abs(I_red), 1e-20)
    sym_ok = 0.8 <= sym <= 1.2

    verdict = "✅" if sym_ok else "⚠️ Asymétrique"
    if excluded:
        verdict += " [Exclu]"

    return {
        "Spectre":       f"e{e}_c{ci+1}_r{ri+1} ({conc:.1e} M)",
        "I_pic_ox (µA)": f"{I_ox*1e6:.3f}",
        "I_pic_red (µA)": f"{I_red*1e6:.3f}",
        "Symétrie":      f"{sym:.2f}",
        "E_pic_ox (V)":  f"{float(E[idx_ox]):.4f}",
        "Verdict":       verdict,
        "_group_label":  group_label,
        "_rep_idx":      ri,
    }


# ─────────────────────────────────────────────
# Résumé et validation finale
# ─────────────────────────────────────────────

def _collect_all_spectrum_labels(experiment: dict) -> list:
    """Return all spectrum_label strings (used as deleted_points keys) in the experiment."""
    labels = []
    mode = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)
    concentrations = experiment.get("concentrations") or []

    if mode in ("eis_only", "both"):
        for e in range(1, n_elec + 1):
            elec_key = f"electrode_{e}"
            probe_reps = ((experiment.get("probe") or {}).get("eis") or {}).get(elec_key) or []
            for ri in range(len(probe_reps)):
                group_label = f"probe_{elec_key}"
                labels.append(f"{group_label}_r{ri}")
            cal = ((experiment.get("calibration") or {}).get("eis") or {}).get(elec_key) or []
            for ci, rep_list in enumerate(cal):
                group_label = f"eis_{elec_key}_c{ci}"
                for ri in range(len(rep_list or [])):
                    labels.append(f"{group_label}_r{ri}")

    if mode in ("cv_only", "both"):
        for e in range(1, n_elec + 1):
            elec_key = f"electrode_{e}"
            probe_reps = ((experiment.get("probe") or {}).get("cv") or {}).get(elec_key) or []
            for ri in range(len(probe_reps)):
                group_label = f"cv_probe_{elec_key}"
                labels.append(f"{group_label}_r{ri}")
            cal = ((experiment.get("calibration") or {}).get("cv") or {}).get(elec_key) or []
            for ci, rep_list in enumerate(cal):
                group_label = f"cv_{elec_key}_c{ci}"
                for ri in range(len(rep_list or [])):
                    labels.append(f"{group_label}_r{ri}")

    return labels


def _section_final_validation(experiment: dict, exclusions: dict) -> None:
    st.divider()
    st.subheader("📋 Résumé et validation finale")

    n_total  = _count_spectra(experiment)
    n_curves_excluded = sum(len(v) for v in exclusions.values())
    n_retenu = max(n_total - n_curves_excluded, 0)

    all_spectrum_labels = _collect_all_spectrum_labels(experiment)
    total_points_excluded = sum(
        len(st.session_state.get(f"deleted_points_{lbl}", []))
        for lbl in all_spectrum_labels
    )

    st.markdown("#### Résumé des modifications")
    st.write(f"- {n_curves_excluded} courbe(s) complète(s) exclue(s)")
    st.write(f"- {total_points_excluded} point(s) individuel(s) exclu(s)")
    st.write(f"- {n_retenu} spectre(s) retenus pour l'analyse")

    if st.button(
        "✅ Valider le prétraitement et passer à l'analyse",
        key="preproc_validate_btn",
        type="primary",
    ):
        exp_clean = _apply_exclusions(experiment, exclusions)
        st.session_state["experiment_clean"] = exp_clean
        # Store point exclusions separately so downstream can apply them
        point_exclusions = {
            lbl: st.session_state[f"deleted_points_{lbl}"]
            for lbl in all_spectrum_labels
            if st.session_state.get(f"deleted_points_{lbl}")
        }
        st.session_state["point_exclusions"] = point_exclusions
        st.success(
            "✅ Prétraitement validé. Rendez-vous dans les pages "
            "**EIS seule**, **CV seule**, **Comparatif** ou **Prédiction**."
        )


def _count_spectra(experiment: dict) -> int:
    n = 0
    mode = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)
    for sig in (["eis"] if mode == "eis_only" else ["cv"] if mode == "cv_only" else ["eis", "cv"]):
        probe  = (experiment.get("probe") or {}).get(sig) or {}
        cal    = (experiment.get("calibration") or {}).get(sig) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            n += len(probe.get(key) or [])
            for rep_list in (cal.get(key) or []):
                n += len(rep_list or [])
    return n


def _apply_exclusions(experiment: dict, exclusions: dict) -> dict:
    """
    Retourne une copie de experiment sans les BytesIO exclus.
    Les listes de réplicats sont filtrées en place.
    """
    import copy
    exp_clean = copy.deepcopy(experiment)
    mode   = exp_clean.get("mode", "both")
    n_elec = exp_clean.get("n_electrodes", 2)

    sig_types = (
        ["eis"] if mode == "eis_only"
        else ["cv"] if mode == "cv_only"
        else ["eis", "cv"]
    )

    for sig in sig_types:
        # Probe
        probe_sig = (exp_clean.get("probe") or {}).get(sig) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            group_label = f"probe_{key}" if sig == "eis" else f"cv_probe_{key}"
            excl_set = exclusions.get(group_label, set())
            reps = probe_sig.get(key) or []
            probe_sig[key] = [r for i, r in enumerate(reps) if i not in excl_set]

        # Calibration
        cal_sig = (exp_clean.get("calibration") or {}).get(sig) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            concs_list = cal_sig.get(key) or []
            for ci, rep_list in enumerate(concs_list):
                prefix = "eis" if sig == "eis" else "cv"
                group_label = f"{prefix}_{key}_c{ci}"
                excl_set = exclusions.get(group_label, set())
                concs_list[ci] = [r for i, r in enumerate(rep_list or []) if i not in excl_set]

    return exp_clean


# ─────────────────────────────────────────────
# Page principale
# ─────────────────────────────────────────────

def main() -> None:
    st.title("🔬 Prétraitement des données")
    st.caption("Visualisation, validation KK, exclusion des outliers.")

    _check_import()

    experiment = st.session_state["experiment"]

    # Initialiser les exclusions
    if "exclusions" not in st.session_state:
        st.session_state["exclusions"] = {}
    exclusions: dict = st.session_state["exclusions"]

    tab_visu, tab_kk = st.tabs(["📊 Visualisation", "🔍 Validation & Nettoyage"])

    with tab_visu:
        _tab_visualisation(experiment, exclusions)

    with tab_kk:
        _tab_validation_kk(experiment, exclusions)

    _section_final_validation(experiment, exclusions)


if __name__ == "__main__":
    main()
