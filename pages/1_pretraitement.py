"""Page 1 — Prétraitement et validation des données.

Structure des exclusions (nouvelle, indépendante par modalité) :
  st.session_state['exclusions'] = {
      'e1': {
          'eis': {'probe': [False, True], 0: [False, False, True], ...},
          'cv':  {'probe': [False, False], 0: [True, False, False], ...},
      },
      'e2': {...}
  }

Clés deleted_points : f"deleted_points_{e_str}_{modality}_c{ci_str}_r{ri}"
  ex : "deleted_points_e1_eis_cprobe_r0", "deleted_points_e1_eis_c0_r2"
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.loader import load_spectrum, average_replicates
from core.cv_loader import load_cv_file, average_cv_replicates
from core.experiment_io import apply_exclusions as _apply_exclusions
from core.models import EISSpectrum
from core.cv_models import CVScan
from plotting.eis_plots import nyquist_figure as _nyquist_figure

# ── Palette réplicats ──────────────────────────────────────────────────────────
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


def _init_exclusions(experiment: dict) -> None:
    """
    Initialise st.session_state['exclusions'] si absent ou incomplet.
    Structure : {e_str: {modality: {ci: [False, ...]}}}
    """
    if "exclusions" not in st.session_state:
        st.session_state["exclusions"] = {}

    excl = st.session_state["exclusions"]
    mode           = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations", [])
    n_electrodes   = experiment.get("n_electrodes", 2)

    modalities: list = []
    if mode in ("eis_only", "both"):
        modalities.append("eis")
    if mode in ("cv_only", "both"):
        modalities.append("cv")

    for elec_idx in range(1, n_electrodes + 1):
        e_str    = f"e{elec_idx}"
        elec_key = f"electrode_{elec_idx}"
        e_dict   = excl.setdefault(e_str, {})

        for mod in modalities:
            mod_dict = e_dict.setdefault(mod, {})

            # Probe
            if "probe" not in mod_dict:
                probe_reps = (
                    (experiment.get("probe") or {})
                    .get(mod, {})
                    .get(elec_key) or []
                )
                mod_dict["probe"] = [False] * len(probe_reps)

            # Calibration
            cal_concs = (
                (experiment.get("calibration") or {})
                .get(mod, {})
                .get(elec_key) or []
            )
            for c_idx in range(len(concentrations)):
                if c_idx not in mod_dict:
                    reps   = cal_concs[c_idx] if c_idx < len(cal_concs) else []
                    n_reps = len(reps)
                    mod_dict[c_idx] = [False] * n_reps

    if "deleted_points" not in st.session_state:
        st.session_state["deleted_points"] = {}


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
# Helpers exclusions 2D
# ─────────────────────────────────────────────

def _excl_get(exclusions: dict, e_str: str, modality: str, ci) -> list:
    """Retourne la liste de booléens pour (e_str, modality, ci). Jamais None."""
    return exclusions.get(e_str, {}).get(modality, {}).get(ci, [])


def _excl_toggle(exclusions: dict, e_str: str, modality: str, ci, ri: int) -> None:
    """Bascule l'exclusion du réplicat ri pour (e_str, modality, ci)."""
    e_dict   = exclusions.setdefault(e_str, {})
    mod_dict = e_dict.setdefault(modality, {})
    excl_list = mod_dict.setdefault(ci, [])
    while len(excl_list) <= ri:
        excl_list.append(False)
    excl_list[ri] = not excl_list[ri]
    st.session_state["exclusions"] = exclusions


def _count_excluded(exclusions: dict) -> int:
    n = 0
    for e_dict in exclusions.values():
        for mod_dict in e_dict.values():
            for excl_list in mod_dict.values():
                n += sum(1 for x in excl_list if x)
    return n


# ─────────────────────────────────────────────
# Helpers points supprimés (EIS uniquement)
# ─────────────────────────────────────────────

def _dp_key(e_str: str, modality: str, ci, ri: int) -> str:
    ci_str = "probe" if ci == "probe" else str(ci)
    return f"deleted_points_{e_str}_{modality}_c{ci_str}_r{ri}"


def _get_deleted(e_str: str, modality: str, ci, ri: int) -> list:
    return list(st.session_state.get(_dp_key(e_str, modality, ci, ri), []))


def _collect_dp_labels(experiment: dict) -> list:
    """Labels internes (sans préfixe 'deleted_points_') pour tous les réplicats EIS."""
    labels = []
    mode = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)

    if mode not in ("eis_only", "both"):
        return labels

    for e_idx in range(1, n_elec + 1):
        e_str    = f"e{e_idx}"
        elec_key = f"electrode_{e_idx}"
        probe_reps = ((experiment.get("probe") or {}).get("eis") or {}).get(elec_key) or []
        for ri in range(len(probe_reps)):
            labels.append(f"{e_str}_eis_cprobe_r{ri}")
        cal = ((experiment.get("calibration") or {}).get("eis") or {}).get(elec_key) or []
        for ci, rep_list in enumerate(cal):
            for ri in range(len(rep_list or [])):
                labels.append(f"{e_str}_eis_c{ci}_r{ri}")

    return labels


# ─────────────────────────────────────────────
# Figures superposées
# ─────────────────────────────────────────────

def _superposed_nyquist(reps: list, e_str: str, ci, exclusions: dict) -> go.Figure:
    """Nyquist avec tous les réplicats superposés ; exclus grisés/pointillés."""
    fig = go.Figure()
    excl_list = _excl_get(exclusions, e_str, "eis", ci)
    for ri, sp in enumerate(reps):
        excluded = excl_list[ri] if ri < len(excl_list) else False
        deleted  = _get_deleted(e_str, "eis", ci, ri)

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


def _superposed_cv(reps: list, e_str: str, ci, exclusions: dict) -> go.Figure:
    """CV avec tous les réplicats superposés ; exclus grisés/pointillés."""
    fig = go.Figure()
    excl_list = _excl_get(exclusions, e_str, "cv", ci)
    for ri, sc in enumerate(reps):
        excluded = excl_list[ri] if ri < len(excl_list) else False
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
    e_str: str,
    modality: str,
    ci,
    ri: int,
    electrode_label: str,
    conc_label: str,
) -> None:
    dkey = _dp_key(e_str, modality, ci, ri)
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
            key=f"dialog_nyquist_{e_str}_{modality}_c{ci}_r{ri}",
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
                         key=f"dlg_excl_{e_str}_{modality}_c{ci}_r{ri}"):
                existing = st.session_state.get(dkey, [])
                st.session_state[dkey] = list(set(list(existing) + selected))
                st.rerun()
        else:
            st.button("🗑️ Exclure sélection", disabled=True,
                      key=f"dlg_excl_dis_{e_str}_{modality}_c{ci}_r{ri}")

    with col_b:
        if deleted:
            if st.button("↩️ Restaurer tous", key=f"dlg_restore_{e_str}_{modality}_c{ci}_r{ri}"):
                st.session_state[dkey] = []
                st.rerun()

    with col_c:
        if deleted:
            st.info(f"{len(deleted)} point(s) exclu(s)")
        elif selected:
            st.caption(f"{len(selected)} point(s) sélectionné(s)")


# ─────────────────────────────────────────────
# Ligne de contrôle d'un réplicat (par modalité)
# ─────────────────────────────────────────────

def _replicate_modal_row(
    ri: int,
    rep,
    e_str: str,
    modality: str,
    ci,
    electrode_label: str,
    conc_label: str,
    exclusions: dict,
) -> None:
    excl_list = _excl_get(exclusions, e_str, modality, ci)
    excluded  = excl_list[ri] if ri < len(excl_list) else False

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
        if st.button(btn, key=f"toggle_{e_str}_{modality}_c{ci}_r{ri}", use_container_width=True):
            _excl_toggle(exclusions, e_str, modality, ci, ri)
            st.rerun()

    with cols[3]:
        if modality == "eis" and rep is not None:
            dkey  = _dp_key(e_str, modality, ci, ri)
            n_del = len(st.session_state.get(dkey, []))
            label = "✏️ Éditer" + (f" ({n_del}✗)" if n_del else "")
            if st.button(label, key=f"edit_{e_str}_{modality}_c{ci}_r{ri}", use_container_width=True):
                _edit_replicate_dialog(
                    sp=rep,
                    e_str=e_str,
                    modality=modality,
                    ci=ci,
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
    e_str: str,
    ci,
    electrode_label: str,
    mode: str,
    exclusions: dict,
) -> None:
    with st.expander(f"**{conc_label}**", expanded=True):

        # ── Bloc EIS ──────────────────────────────────────────────────────
        if mode in ("eis_only", "both") and reps_eis:
            st.markdown("**EIS — Nyquist**")
            st.plotly_chart(
                _superposed_nyquist(reps_eis, e_str, ci, exclusions),
                use_container_width=True,
                key=f"nyq_{e_str}_c{ci}",
            )
            st.divider()
            for ri, sp in enumerate(reps_eis):
                _replicate_modal_row(
                    ri=ri,
                    rep=sp,
                    e_str=e_str,
                    modality="eis",
                    ci=ci,
                    electrode_label=electrode_label,
                    conc_label=conc_label,
                    exclusions=exclusions,
                )

        # ── Bloc CV ───────────────────────────────────────────────────────
        if mode in ("cv_only", "both") and reps_cv:
            if mode == "both" and reps_eis:
                st.markdown("---")
            st.markdown("**CV — I(E)**")
            st.plotly_chart(
                _superposed_cv(reps_cv, e_str, ci, exclusions),
                use_container_width=True,
                key=f"cv_{e_str}_c{ci}",
            )
            st.divider()
            for ri, sc in enumerate(reps_cv):
                _replicate_modal_row(
                    ri=ri,
                    rep=sc,
                    e_str=e_str,
                    modality="cv",
                    ci=ci,
                    electrode_label=electrode_label,
                    conc_label=conc_label,
                    exclusions=exclusions,
                )


# ─────────────────────────────────────────────
# Panneau droit — Graphes moyens en direct
# ─────────────────────────────────────────────

def _average_nyquist(groups: list) -> go.Figure:
    """groups : liste de (label, [spectra_actifs])"""
    fig = go.Figure()
    for ci_idx, (label, active) in enumerate(groups):
        if not active:
            continue
        try:
            avg   = average_replicates(active)
            color = REP_COLORS[ci_idx % len(REP_COLORS)]
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


def _average_cv(groups: list) -> go.Figure:
    """groups : liste de (label, [scans_actifs])"""
    fig = go.Figure()
    for ci_idx, (label, active) in enumerate(groups):
        if not active:
            continue
        try:
            avg   = average_cv_replicates(active)
            color = REP_COLORS[ci_idx % len(REP_COLORS)]
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
    e_str: str,
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
            excl  = _excl_get(exclusions, e_str, "eis", "probe")
            active = [sp for ri, sp in enumerate(probe_reps) if not (ri < len(excl) and excl[ri])]
            groups_eis.append(("Probe", active))
        for ci, reps in enumerate(eis_spectra["calibration"].get(elec_key) or []):
            excl   = _excl_get(exclusions, e_str, "eis", ci)
            active = [sp for ri, sp in enumerate(reps) if not (ri < len(excl) and excl[ri])]
            conc   = concentrations[ci] if ci < len(concentrations) else 0.0
            groups_eis.append((_format_conc(conc), active))

        if groups_eis:
            st.plotly_chart(_average_nyquist(groups_eis),
                            use_container_width=True, key=f"avg_nyq_{e_str}")

    if mode in ("cv_only", "both"):
        groups_cv: list = []
        probe_reps = cv_scans["probe"].get(elec_key) or []
        if probe_reps:
            excl   = _excl_get(exclusions, e_str, "cv", "probe")
            active = [sc for ri, sc in enumerate(probe_reps) if not (ri < len(excl) and excl[ri])]
            groups_cv.append(("Probe", active))
        for ci, reps in enumerate(cv_scans["calibration"].get(elec_key) or []):
            excl   = _excl_get(exclusions, e_str, "cv", ci)
            active = [sc for ri, sc in enumerate(reps) if not (ri < len(excl) and excl[ri])]
            conc   = concentrations[ci] if ci < len(concentrations) else 0.0
            groups_cv.append((_format_conc(conc), active))

        if groups_cv:
            st.plotly_chart(_average_cv(groups_cv),
                            use_container_width=True, key=f"avg_cv_{e_str}")


# ─────────────────────────────────────────────
# Résumé et validation finale
# ─────────────────────────────────────────────

def _count_spectra(experiment: dict) -> dict:
    result = {"eis": 0, "cv": 0}
    mode   = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)
    mods   = []
    if mode in ("eis_only", "both"):
        mods.append("eis")
    if mode in ("cv_only", "both"):
        mods.append("cv")
    for mod in mods:
        probe = (experiment.get("probe") or {}).get(mod) or {}
        cal   = (experiment.get("calibration") or {}).get(mod) or {}
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            result[mod] += len(probe.get(key) or [])
            for rep_list in (cal.get(key) or []):
                result[mod] += len(rep_list or [])
    return result



def _run_kk_validation(exp_clean: dict) -> None:
    """Lance la validation KK sur tous les groupes EIS de l'expérience nettoyée."""
    from core.validator import validate_replicate_group

    mode = exp_clean.get("mode", "both")
    if mode not in ("eis_only", "both"):
        return

    n_elec         = exp_clean.get("n_electrodes", 2)
    concentrations = exp_clean.get("concentrations") or []
    validation_results: dict = {}

    eis_spectra = _load_eis_spectra(exp_clean)

    for e_idx in range(1, n_elec + 1):
        elec_key = f"electrode_{e_idx}"

        probe_reps = eis_spectra["probe"].get(elec_key) or []
        if probe_reps:
            vr = validate_replicate_group(
                [np.array(sp.f) for sp in probe_reps],
                [np.array(sp.Zre) for sp in probe_reps],
                [np.array(sp.Zim) for sp in probe_reps],
                label=f"probe_electrode_{e_idx}",
            )
            validation_results[f"probe_electrode_{e_idx}"] = vr

        cal_reps_list = eis_spectra["calibration"].get(elec_key) or []
        for ci, reps in enumerate(cal_reps_list):
            if reps:
                conc  = concentrations[ci] if ci < len(concentrations) else 0.0
                label = f"e{e_idx}_{_format_conc(conc)}"
                vr    = validate_replicate_group(
                    [np.array(sp.f) for sp in reps],
                    [np.array(sp.Zre) for sp in reps],
                    [np.array(sp.Zim) for sp in reps],
                    label=label,
                )
                validation_results[f"eis_electrode_{e_idx}_c{ci}"] = vr

    st.session_state["validation_results"] = validation_results


def _section_save(experiment: dict, exclusions: dict, dp_labels: list) -> None:
    """Section de sauvegarde de l'expérience prétraitée (ZIP téléchargeable)."""
    from core.experiment_io import save_experiment

    st.divider()
    st.markdown("### 💾 Sauvegarder l'expérience prétraitée")
    st.caption(
        "Inclut les données brutes + toutes les exclusions. "
        "Au rechargement, vous pourrez aller directement à l'analyse."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        save_name = st.text_input(
            "Nom de la sauvegarde",
            value=experiment.get("name", "experience") or "experience",
            key="save_name_pretraitement",
        )
    with col2:
        st.write("")
        st.write("")
        save_clicked = st.button("💾 Sauvegarder", key="save_pretraitement")

    if save_clicked and save_name:
        deleted_points = {
            lbl: st.session_state[f"deleted_points_{lbl}"]
            for lbl in dp_labels
            if st.session_state.get(f"deleted_points_{lbl}")
        }
        zip_bytes = save_experiment(
            experiment,
            exclusions=exclusions,
            deleted_points=deleted_points,
            preprocessing_done=True,
        )
        st.download_button(
            label=f"📥 Télécharger {save_name}.zip",
            data=zip_bytes,
            file_name=f"{save_name}.zip",
            mime="application/zip",
            key="download_pretraitement",
        )
        st.success(f"Prêt à télécharger : {save_name}.zip")

    st.divider()


def _section_final_validation(experiment: dict, exclusions: dict) -> None:
    st.subheader("📋 Résumé et validation finale")

    n_counts  = _count_spectra(experiment)
    n_excl    = _count_excluded(exclusions)
    dp_labels = _collect_dp_labels(experiment)
    n_pts_excl = sum(
        len(st.session_state.get(f"deleted_points_{lbl}", []))
        for lbl in dp_labels
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("EIS total", n_counts.get("eis", 0))
    col_b.metric("CV total",  n_counts.get("cv",  0))
    col_c.metric("Réplicats exclus", n_excl)
    col_d.metric("Points exclus (EIS)", n_pts_excl)

    _section_save(experiment, exclusions, dp_labels)

    if st.button(
        "✅ Valider le prétraitement et passer à l'analyse",
        key="preproc_validate_btn",
        type="primary",
    ):
        exp_clean = _apply_exclusions(experiment, exclusions)
        st.session_state["experiment_clean"] = exp_clean
        st.session_state["preprocessing_done"] = True
        for key in ("eis_session", "eis_validation", "comparison_report", "comparison_session_data"):
            st.session_state[key] = None

        deleted_points = {
            lbl: st.session_state[f"deleted_points_{lbl}"]
            for lbl in dp_labels
            if st.session_state.get(f"deleted_points_{lbl}")
        }
        st.session_state["point_exclusions"] = deleted_points

        if experiment.get("mode") in ("eis_only", "both"):
            _run_kk_validation(exp_clean)

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

    experiment     = st.session_state["experiment"]
    mode           = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    n_elec         = experiment.get("n_electrodes", 2)

    _init_exclusions(experiment)
    exclusions: dict = st.session_state["exclusions"]

    # ── Sélecteur d'électrode ────────────────────────────────────────────────
    elec_opts = [f"Électrode {e}" for e in range(1, n_elec + 1)]
    sel_elec  = st.radio("Électrode", elec_opts, horizontal=True, key="preproc_elec_radio")
    e_idx     = elec_opts.index(sel_elec) + 1
    e_str     = f"e{e_idx}"
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
                e_str=e_str,
                ci="probe",
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
                e_str=e_str,
                ci=ci,
                electrode_label=sel_elec,
                mode=mode,
                exclusions=exclusions,
            )

    with col_right:
        _render_average_panel(
            eis_spectra=eis_spectra,
            cv_scans=cv_scans,
            e_str=e_str,
            elec_key=elec_key,
            concentrations=concentrations,
            mode=mode,
            exclusions=exclusions,
        )

    _section_final_validation(experiment, exclusions)


if __name__ == "__main__":
    main()
