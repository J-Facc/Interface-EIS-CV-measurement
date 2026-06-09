"""Page 1 — Prétraitement et validation des données.

Nouvelle interface (refonte complète) :
  Niveau 1 — Vue d'ensemble par concentration (spectres superposés, contrôles de réplicats)
  Niveau 2 — Dialog d'édition de points (st.dialog, EIS uniquement)
  Panneau droit   — Graphes moyens en direct, mis à jour selon les exclusions actives
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.loader import load_spectrum, average_replicates
from core.cv_loader import load_cv_file, average_cv_replicates
from core.models import EISSpectrum
from core.cv_models import CVScan
from plotting.eis_plots import nyquist_figure as _nyquist_figure

# ── Palette réplicats ──────────────────────────────────────────────────────────
# Rep 1 → bleu, Rep 2 → orange, Rep 3 → vert (palette Plotly standard)
REP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
EXCL_COLOR = "lightgray"

_SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _format_conc(conc: float) -> str:
    if conc <= 0:
        return f"{conc}"
    exp = int(np.floor(np.log10(conc)))
    mant = conc / 10**exp
    exp_str = str(exp).translate(_SUP_MAP)
    return f"{mant:.0f}×10{exp_str} M"


# ─────────────────────────────────────────────
# Guard — import obligatoire
# ─────────────────────────────────────────────

def _check_import() -> None:
    if "import_validated" not in st.session_state:
        st.warning("⚠️ Importez d'abord vos données dans l'onglet **Import**.")
        st.page_link("pages/0_import.py", label="Aller à l'import", icon="📂")
        st.stop()


# ─────────────────────────────────────────────
# Chargement des spectres / scans
# ─────────────────────────────────────────────

def _bio_to_eis(bio, label: str) -> Optional[EISSpectrum]:
    if bio is None:
        return None
    try:
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        return load_spectrum(content, label=label)
    except Exception:
        return None


def _bio_to_cv(bio, label: str, concentration: float, step: str) -> Optional[CVScan]:
    if bio is None:
        return None
    try:
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        return load_cv_file(content, label=label, concentration=concentration, step=step)
    except Exception:
        return None


def _load_eis_spectra(experiment: dict) -> Dict[str, Any]:
    calibration = (experiment.get("calibration") or {}).get("eis") or {}
    probe_dict  = (experiment.get("probe") or {}).get("eis") or {}
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    spectra: Dict[str, Any] = {"probe": {}, "calibration": {}}

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        spectra["probe"][key] = [
            sp for ri, bio in enumerate(probe_dict.get(key) or [])
            if (sp := _bio_to_eis(bio, f"probe_e{e}_r{ri+1}")) is not None
        ]
        spectra["calibration"][key] = []
        for ci, rep_files in enumerate(calibration.get(key) or []):
            reps = [
                sp for ri, bio in enumerate(rep_files or [])
                if (sp := _bio_to_eis(bio, f"e{e}_c{ci+1}_r{ri+1}")) is not None
            ]
            spectra["calibration"][key].append(reps)

    return spectra


def _load_cv_scans(experiment: dict) -> Dict[str, Any]:
    calibration = (experiment.get("calibration") or {}).get("cv") or {}
    probe_dict  = (experiment.get("probe") or {}).get("cv") or {}
    concentrations = experiment.get("concentrations") or []
    n_elec = experiment.get("n_electrodes", 2)

    scans: Dict[str, Any] = {"probe": {}, "calibration": {}}

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        scans["probe"][key] = [
            sc for ri, bio in enumerate(probe_dict.get(key) or [])
            if (sc := _bio_to_cv(bio, f"probe_e{e}_r{ri+1}", 0.0, "probe")) is not None
        ]
        scans["calibration"][key] = []
        for ci, rep_files in enumerate(calibration.get(key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            reps = [
                sc for ri, bio in enumerate(rep_files or [])
                if (sc := _bio_to_cv(bio, f"e{e}_c{ci+1}_r{ri+1}", conc, "hybridization")) is not None
            ]
            scans["calibration"][key].append(reps)

    return scans


# ─────────────────────────────────────────────
# Helpers exclusions / points supprimés
# ─────────────────────────────────────────────

def _is_excluded(group_label: str, ri: int, exclusions: dict) -> bool:
    return ri in exclusions.get(group_label, set())


def _toggle_exclusion(group_label: str, ri: int, exclusions: dict) -> None:
    if group_label not in exclusions:
        exclusions[group_label] = set()
    if ri in exclusions[group_label]:
        exclusions[group_label].discard(ri)
    else:
        exclusions[group_label].add(ri)
    st.session_state["exclusions"] = exclusions


def _deleted_key(group_label: str, ri: int) -> str:
    return f"deleted_points_{group_label}_r{ri}"


def _get_deleted(group_label: str, ri: int) -> list:
    return list(st.session_state.get(_deleted_key(group_label, ri), []))


# ─────────────────────────────────────────────
# Figures superposées
# ─────────────────────────────────────────────

def _superposed_nyquist(reps: list, group_label: str, exclusions: dict) -> go.Figure:
    """Nyquist avec tous les réplicats superposés ; exclus grisés/pointillés."""
    fig = go.Figure()
    for ri, sp in enumerate(reps):
        excluded = _is_excluded(group_label, ri, exclusions)
        deleted  = _get_deleted(group_label, ri)

        color   = EXCL_COLOR if excluded else REP_COLORS[ri % len(REP_COLORS)]
        opacity = 0.3 if excluded else 1.0
        name    = f"Rép {ri+1}" + (" [Exclu]" if excluded else "")

        keep = np.ones(len(sp.f), dtype=bool)
        for idx in deleted:
            if 0 <= idx < len(sp.f):
                keep[idx] = False

        Zre = np.asarray(sp.Zre)[keep]
        Zim = np.asarray(sp.Zim)[keep]

        fig.add_trace(go.Scatter(
            x=Zre, y=Zim,
            mode="markers",
            name=name,
            marker=dict(color=color, size=6),
            opacity=opacity,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Re(Z) = %{x:.1f} Ω<br>−Im(Z) = %{y:.1f} Ω<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        height=300,
        margin=dict(t=30, b=40, l=50, r=20),
    )
    return fig


def _superposed_cv(reps: list, group_label: str, exclusions: dict) -> go.Figure:
    """CV avec tous les réplicats superposés ; exclus grisés/pointillés."""
    fig = go.Figure()
    for ri, sc in enumerate(reps):
        excluded = _is_excluded(group_label, ri, exclusions)
        color   = EXCL_COLOR if excluded else REP_COLORS[ri % len(REP_COLORS)]
        opacity = 0.3 if excluded else 1.0
        dash    = "dot" if excluded else "solid"
        name    = f"Rép {ri+1}" + (" [Exclu]" if excluded else "")

        fig.add_trace(go.Scatter(
            x=np.asarray(sc.E),
            y=np.asarray(sc.I) * 1e6,
            mode="lines",
            name=name,
            line=dict(color=color, dash=dash),
            opacity=opacity,
        ))

    fig.update_layout(
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        height=240,
        margin=dict(t=30, b=40, l=50, r=20),
    )
    return fig


# ─────────────────────────────────────────────
# Dialog d'édition de points (EIS)
# ─────────────────────────────────────────────

@st.dialog("Édition du réplicat", width="large")
def _edit_replicate_dialog(
    sp: EISSpectrum,
    group_label: str,
    ri: int,
    electrode_label: str,
    conc_label: str,
) -> None:
    dkey = _deleted_key(group_label, ri)
    deleted: list = list(st.session_state.get(dkey, []))

    st.markdown(f"**{electrode_label} — {conc_label} — Réplicat {ri + 1}**")

    col_graph, col_table = st.columns([0.6, 0.4])

    with col_graph:
        fig = _nyquist_figure(
            spectrum=sp,
            excluded_indices=deleted if deleted else None,
            selection_mode=True,
        )
        fig.update_layout(height=400, title=None, margin=dict(t=10, b=40, l=50, r=20))
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"dialog_nyquist_{group_label}_r{ri}",
            on_select="rerun",
        )

    selected: list = []
    if event and hasattr(event, "selection") and event.selection and event.selection.points:
        selected = [p["point_index"] for p in event.selection.points]

    with col_table:
        st.markdown("**Points du spectre**")
        excl_set = set(deleted)
        rows = [
            {
                "Idx": i,
                "f (Hz)": f"{sp.f[i]:.3e}",
                "Re(Z) Ω": f"{sp.Zre[i]:.2f}",
                "−Im(Z) Ω": f"{sp.Zim[i]:.2f}",
                "État": "✗" if i in excl_set else "",
            }
            for i in range(len(sp.f))
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=320)

    st.divider()
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if selected:
            if st.button(f"🗑️ Exclure {len(selected)} point(s)", type="primary",
                         key=f"dlg_excl_{group_label}_r{ri}"):
                existing = st.session_state.get(dkey, [])
                st.session_state[dkey] = list(set(list(existing) + selected))
                st.rerun()
        else:
            st.button("🗑️ Exclure sélection", disabled=True,
                      key=f"dlg_excl_dis_{group_label}_r{ri}")

    with col_b:
        if deleted:
            if st.button("↩️ Restaurer tous", key=f"dlg_restore_{group_label}_r{ri}"):
                st.session_state[dkey] = []
                st.rerun()

    with col_c:
        if deleted:
            st.info(f"{len(deleted)} point(s) exclu(s)")
        elif selected:
            st.caption(f"{len(selected)} point(s) sélectionné(s)")


# ─────────────────────────────────────────────
# Ligne de contrôle d'un réplicat
# ─────────────────────────────────────────────

def _replicate_control_row(
    ri: int,
    reps_eis: list,
    reps_cv: list,
    group_label_eis: str,
    group_label_cv: str,
    electrode_label: str,
    conc_label: str,
    mode: str,
    exclusions: dict,
) -> None:
    primary = group_label_eis if mode != "cv_only" else group_label_cv
    excluded = _is_excluded(primary, ri, exclusions)

    cols = st.columns([0.18, 0.22, 0.20, 0.20, 0.20])

    with cols[0]:
        dot = "🔴" if excluded else "🟢"
        st.markdown(f"**{dot} Rép {ri+1}**")

    with cols[1]:
        if excluded:
            st.markdown(":orange[⚠️ Exclu]")
        else:
            st.markdown(":green[✓ Inclus]")

    with cols[2]:
        btn = "Restaurer" if excluded else "Exclure"
        if st.button(btn, key=f"toggle_{primary}_{ri}", use_container_width=True):
            for gl in [group_label_eis, group_label_cv]:
                if gl:
                    _toggle_exclusion(gl, ri, exclusions)
            st.rerun()

    with cols[3]:
        # Bouton Éditer — EIS uniquement
        if mode != "cv_only" and ri < len(reps_eis):
            dkey = _deleted_key(group_label_eis, ri)
            n_del = len(st.session_state.get(dkey, []))
            label = f"✏️ Éditer" + (f" ({n_del}✗)" if n_del else "")
            if st.button(label, key=f"edit_{group_label_eis}_r{ri}", use_container_width=True):
                _edit_replicate_dialog(
                    sp=reps_eis[ri],
                    group_label=group_label_eis,
                    ri=ri,
                    electrode_label=electrode_label,
                    conc_label=conc_label,
                )
        else:
            st.empty()

    with cols[4]:
        st.empty()


# ─────────────────────────────────────────────
# Bloc par concentration (expander)
# ─────────────────────────────────────────────

def _concentration_block(
    conc_label: str,
    reps_eis: list,
    reps_cv: list,
    group_label_eis: str,
    group_label_cv: str,
    electrode_label: str,
    mode: str,
    exclusions: dict,
) -> None:
    with st.expander(f"**{conc_label}**", expanded=True):
        # Graphe Nyquist superposé
        if mode in ("eis_only", "both") and reps_eis:
            st.plotly_chart(
                _superposed_nyquist(reps_eis, group_label_eis, exclusions),
                use_container_width=True,
                key=f"nyq_{group_label_eis}",
            )

        # Graphe CV superposé
        if mode in ("cv_only", "both") and reps_cv:
            st.plotly_chart(
                _superposed_cv(reps_cv, group_label_cv, exclusions),
                use_container_width=True,
                key=f"cv_{group_label_cv}",
            )

        # Lignes de contrôle par réplicat
        n_reps = max(len(reps_eis) if reps_eis else 0, len(reps_cv) if reps_cv else 0)
        if n_reps:
            st.divider()
            for ri in range(n_reps):
                _replicate_control_row(
                    ri=ri,
                    reps_eis=reps_eis,
                    reps_cv=reps_cv,
                    group_label_eis=group_label_eis,
                    group_label_cv=group_label_cv,
                    electrode_label=electrode_label,
                    conc_label=conc_label,
                    mode=mode,
                    exclusions=exclusions,
                )


# ─────────────────────────────────────────────
# Panneau droit — Graphes moyens en direct
# ─────────────────────────────────────────────

def _average_nyquist(all_groups: list, exclusions: dict) -> go.Figure:
    """Un graphe Nyquist avec la moyenne des réplicats actifs par groupe."""
    fig = go.Figure()
    for ci, (label, reps, gl) in enumerate(all_groups):
        active = [sp for ri, sp in enumerate(reps) if not _is_excluded(gl, ri, exclusions)]
        if not active:
            continue
        try:
            avg = average_replicates(active)
            color = REP_COLORS[ci % len(REP_COLORS)]
            fig.add_trace(go.Scatter(
                x=avg.Zre, y=avg.Zim,
                mode="markers+lines",
                name=label,
                marker=dict(color=color, size=5),
                line=dict(color=color, width=1),
            ))
        except Exception:
            pass

    fig.update_layout(
        title="Nyquist moyen",
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        hovermode="closest",
        height=290,
        margin=dict(t=40, b=40, l=50, r=10),
        legend=dict(font=dict(size=10)),
    )
    return fig


def _average_cv(all_groups: list, exclusions: dict) -> go.Figure:
    """Un graphe CV avec la moyenne des réplicats actifs par groupe."""
    fig = go.Figure()
    for ci, (label, reps, gl) in enumerate(all_groups):
        active = [sc for ri, sc in enumerate(reps) if not _is_excluded(gl, ri, exclusions)]
        if not active:
            continue
        try:
            avg = average_cv_replicates(active)
            color = REP_COLORS[ci % len(REP_COLORS)]
            fig.add_trace(go.Scatter(
                x=avg.E,
                y=np.asarray(avg.I) * 1e6,
                mode="lines",
                name=label,
                line=dict(color=color),
            ))
        except Exception:
            pass

    fig.update_layout(
        title="CV moyen",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
        hovermode="closest",
        height=240,
        margin=dict(t=40, b=40, l=50, r=10),
        legend=dict(font=dict(size=10)),
    )
    return fig


def _render_average_panel(
    eis_spectra: dict,
    cv_scans: dict,
    elec_key: str,
    concentrations: list,
    mode: str,
    exclusions: dict,
) -> None:
    st.markdown("#### Moyennes en direct")
    st.caption("Mises à jour selon les exclusions actives.")

    if mode in ("eis_only", "both"):
        groups_eis: list = []
        probe_reps = eis_spectra["probe"].get(elec_key) or []
        if probe_reps:
            groups_eis.append(("Probe", probe_reps, f"probe_{elec_key}"))
        for ci, reps in enumerate(eis_spectra["calibration"].get(elec_key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            groups_eis.append((_format_conc(conc), reps, f"eis_{elec_key}_c{ci}"))

        if groups_eis:
            st.plotly_chart(_average_nyquist(groups_eis, exclusions),
                            use_container_width=True, key=f"avg_nyq_{elec_key}")

    if mode in ("cv_only", "both"):
        groups_cv: list = []
        probe_reps = cv_scans["probe"].get(elec_key) or []
        if probe_reps:
            groups_cv.append(("Probe", probe_reps, f"cv_probe_{elec_key}"))
        for ci, reps in enumerate(cv_scans["calibration"].get(elec_key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            groups_cv.append((_format_conc(conc), reps, f"cv_{elec_key}_c{ci}"))

        if groups_cv:
            st.plotly_chart(_average_cv(groups_cv, exclusions),
                            use_container_width=True, key=f"avg_cv_{elec_key}")


# ─────────────────────────────────────────────
# Résumé et validation finale
# ─────────────────────────────────────────────

def _collect_all_spectrum_labels(experiment: dict) -> list:
    labels = []
    mode = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)

    if mode in ("eis_only", "both"):
        for e in range(1, n_elec + 1):
            elec_key = f"electrode_{e}"
            probe_reps = ((experiment.get("probe") or {}).get("eis") or {}).get(elec_key) or []
            for ri in range(len(probe_reps)):
                labels.append(f"probe_{elec_key}_r{ri}")
            cal = ((experiment.get("calibration") or {}).get("eis") or {}).get(elec_key) or []
            for ci, rep_list in enumerate(cal):
                for ri in range(len(rep_list or [])):
                    labels.append(f"eis_{elec_key}_c{ci}_r{ri}")

    if mode in ("cv_only", "both"):
        for e in range(1, n_elec + 1):
            elec_key = f"electrode_{e}"
            probe_reps = ((experiment.get("probe") or {}).get("cv") or {}).get(elec_key) or []
            for ri in range(len(probe_reps)):
                labels.append(f"cv_probe_{elec_key}_r{ri}")
            cal = ((experiment.get("calibration") or {}).get("cv") or {}).get(elec_key) or []
            for ci, rep_list in enumerate(cal):
                for ri in range(len(rep_list or [])):
                    labels.append(f"cv_{elec_key}_c{ci}_r{ri}")

    return labels


def _count_spectra(experiment: dict) -> int:
    n = 0
    mode = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)
    sigs = ["eis"] if mode == "eis_only" else ["cv"] if mode == "cv_only" else ["eis", "cv"]
    for sig in sigs:
        probe = (experiment.get("probe") or {}).get(sig) or {}
        cal   = (experiment.get("calibration") or {}).get(sig) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            n += len(probe.get(key) or [])
            for rep_list in (cal.get(key) or []):
                n += len(rep_list or [])
    return n


def _apply_exclusions(experiment: dict, exclusions: dict) -> dict:
    exp_clean = copy.deepcopy(experiment)
    mode   = exp_clean.get("mode", "both")
    n_elec = exp_clean.get("n_electrodes", 2)
    sigs = ["eis"] if mode == "eis_only" else ["cv"] if mode == "cv_only" else ["eis", "cv"]

    for sig in sigs:
        probe_sig = (exp_clean.get("probe") or {}).get(sig) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            group_label = f"probe_{key}" if sig == "eis" else f"cv_probe_{key}"
            excl_set = exclusions.get(group_label, set())
            reps = probe_sig.get(key) or []
            probe_sig[key] = [r for i, r in enumerate(reps) if i not in excl_set]

        cal_sig = (exp_clean.get("calibration") or {}).get(sig) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            for ci, rep_list in enumerate(cal_sig.get(key) or []):
                prefix = "eis" if sig == "eis" else "cv"
                group_label = f"{prefix}_{key}_c{ci}"
                excl_set = exclusions.get(group_label, set())
                cal_sig[key][ci] = [r for i, r in enumerate(rep_list or []) if i not in excl_set]

    return exp_clean


def _section_final_validation(experiment: dict, exclusions: dict) -> None:
    st.divider()
    st.subheader("📋 Résumé et validation finale")

    n_total = _count_spectra(experiment)
    n_excl  = sum(len(v) for v in exclusions.values())
    n_kept  = max(n_total - n_excl, 0)

    all_labels = _collect_all_spectrum_labels(experiment)
    n_pts_excl = sum(
        len(st.session_state.get(f"deleted_points_{lbl}", []))
        for lbl in all_labels
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Courbes exclues", n_excl)
    col_b.metric("Points exclus", n_pts_excl)
    col_c.metric("Spectres retenus", n_kept)

    if st.button(
        "✅ Valider le prétraitement et passer à l'analyse",
        key="preproc_validate_btn",
        type="primary",
    ):
        exp_clean = _apply_exclusions(experiment, exclusions)
        st.session_state["experiment_clean"] = exp_clean
        point_exclusions = {
            lbl: st.session_state[f"deleted_points_{lbl}"]
            for lbl in all_labels
            if st.session_state.get(f"deleted_points_{lbl}")
        }
        st.session_state["point_exclusions"] = point_exclusions
        st.success(
            "✅ Prétraitement validé. Rendez-vous dans les pages "
            "**EIS seule**, **CV seule**, **Comparatif** ou **Prédiction**."
        )


# ─────────────────────────────────────────────
# Page principale
# ─────────────────────────────────────────────

def main() -> None:
    st.title("🔬 Prétraitement des données")
    st.caption("Vue d'ensemble des réplicats, exclusion de courbes, édition de points.")

    _check_import()

    experiment    = st.session_state["experiment"]
    mode          = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    n_elec        = experiment.get("n_electrodes", 2)

    if "exclusions" not in st.session_state:
        st.session_state["exclusions"] = {}
    exclusions: dict = st.session_state["exclusions"]

    # ── Sélecteur d'électrode ────────────────────────────────────────────────
    elec_opts = [f"Électrode {e}" for e in range(1, n_elec + 1)]
    sel_elec  = st.radio("Électrode", elec_opts, horizontal=True, key="preproc_elec_radio")
    e_idx     = elec_opts.index(sel_elec) + 1
    elec_key  = f"electrode_{e_idx}"

    st.divider()

    # ── Chargement ───────────────────────────────────────────────────────────
    has_eis = mode in ("eis_only", "both")
    has_cv  = mode in ("cv_only", "both")

    eis_spectra = _load_eis_spectra(experiment) if has_eis else {"probe": {}, "calibration": {}}
    cv_scans    = _load_cv_scans(experiment)    if has_cv  else {"probe": {}, "calibration": {}}

    # ── Deux colonnes ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([0.65, 0.35])

    with col_left:
        st.markdown("### Vue d'ensemble")

        # ── Probe ──────────────────────────────────────────────────────────
        reps_eis_probe = eis_spectra["probe"].get(elec_key) or []
        reps_cv_probe  = cv_scans["probe"].get(elec_key)   or []
        if reps_eis_probe or reps_cv_probe:
            _concentration_block(
                conc_label="Probe",
                reps_eis=reps_eis_probe,
                reps_cv=reps_cv_probe,
                group_label_eis=f"probe_{elec_key}",
                group_label_cv=f"cv_probe_{elec_key}",
                electrode_label=sel_elec,
                mode=mode,
                exclusions=exclusions,
            )

        # ── Calibration par concentration ───────────────────────────────────
        concs_eis = eis_spectra["calibration"].get(elec_key) or []
        concs_cv  = cv_scans["calibration"].get(elec_key)   or []

        for ci, conc in enumerate(concentrations):
            reps_eis = concs_eis[ci] if ci < len(concs_eis) else []
            reps_cv  = concs_cv[ci]  if ci < len(concs_cv)  else []
            if not reps_eis and not reps_cv:
                continue

            conc_label = f"Concentration {ci+1} — {_format_conc(conc)}"
            _concentration_block(
                conc_label=conc_label,
                reps_eis=reps_eis,
                reps_cv=reps_cv,
                group_label_eis=f"eis_{elec_key}_c{ci}",
                group_label_cv=f"cv_{elec_key}_c{ci}",
                electrode_label=sel_elec,
                mode=mode,
                exclusions=exclusions,
            )

    with col_right:
        _render_average_panel(
            eis_spectra=eis_spectra,
            cv_scans=cv_scans,
            elec_key=elec_key,
            concentrations=concentrations,
            mode=mode,
            exclusions=exclusions,
        )

    _section_final_validation(experiment, exclusions)


if __name__ == "__main__":
    main()
