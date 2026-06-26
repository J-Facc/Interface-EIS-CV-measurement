"""Streamlit tab rendering — EIS (KK / DRT / Reconstructions / Calibration) and CV
(Visualisation / Pic redox / Calibration).
"""

import numpy as np
import streamlit as st

from core.models import EISSession
from core.cv_models import CVSession
from plotting.eis_plots import (
    drt_figure,
    drt_figure_multi,
    drt_replicates_figure,
    open_drt_matplotlib_window,
    reconstruction_comparison_figure,
    drt_reconstruction_figure,
    drt_reconstruction_figure_dual,
    open_reconstruction_matplotlib_window,
    calibration_figure,
    open_calibration_matplotlib_window,
)
from core.cv_peaks import detect_redox_peaks
from plotting.cv_plots import (
    cv_current_figure,
    cv_calibration_figure,
    cv_calibration_figure_multi,
    redox_peaks_figure,
    cv_params_table_multi,
    open_cv_calibration_matplotlib_window,
    open_cv_calibration_matplotlib_window_multi,
)
from plotting.kk_plots import residuals_figure, validation_summary_table
from exports.exporter import (
    export_calibration_csv,
    export_cv_calibration_csv,
    export_cv_calibration_csv_multi,
)


# ─────────────────────────────────────────────────────────────────────────────
# EIS
# ─────────────────────────────────────────────────────────────────────────────

def _spectra_labels_for_session(session: EISSession) -> dict:
    """label -> (replicate_spectra, average_spectrum_or_group) pour une session."""
    out = {}
    if session.bare is not None:
        out["bare"] = (session.bare_replicate_spectra, session.bare)
    if session.probe is not None:
        out["probe"] = (session.probe_replicate_spectra, session.probe)
    for grp in session.groups:
        out[f"{grp.concentration:.2e}"] = (grp.replicate_spectra, grp.spectrum)
    return out


def _render_kk_tab(validations: dict) -> None:
    """Onglet 1 — Validation KK, dupliqué par électrode."""
    if not validations:
        st.info("Lancez une analyse pour voir les résultats de validation.")
        return

    electrodes = sorted(validations.keys())
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])

    for e, elec_tab in zip(electrodes, elec_tabs):
        with elec_tab:
            validation_results = validations.get(e)
            if not validation_results:
                st.info("Aucun résultat de validation pour cette électrode.")
                continue

            st.markdown("#### Récapitulatif")
            fig_table = validation_summary_table(validation_results, theme_mode="light")
            st.plotly_chart(fig_table, width='stretch', key=f"kk_table_e{e}")

            st.markdown("#### Résidus par spectre")
            labels_kk = list(validation_results.keys())
            selected_kk = st.selectbox("Spectre", labels_kk, key=f"kk_select_e{e}")
            vr = validation_results[selected_kk]

            fig_res = residuals_figure(vr, theme_mode="light")
            st.plotly_chart(fig_res, width='stretch', key=f"kk_res_e{e}")

            for kk in vr.replicates:
                if kk.warning:
                    st.warning(f"**{kk.label}** : {kk.warning}")
            if vr.drift_warning:
                st.error(f"**Drift inter-réplicats** : {vr.drift_warning}")
            if vr.f_min_common and vr.f_max_common < float("inf"):
                st.success(
                    f"Plage KK-valide commune : "
                    f"**{vr.f_min_common:.2f} Hz** → **{vr.f_max_common:.2f} Hz**"
                )


def _render_drt_tab(sessions: dict) -> None:
    """Onglet 2 — Courbes DRT."""
    st.subheader("Distribution des temps de relaxation (DRT)")

    # (a) graphe global toutes électrodes
    st.markdown("#### Moyenne — toutes électrodes")
    st.plotly_chart(drt_figure_multi(sessions), width='stretch', key="drt_multi")

    # (b) deux colonnes par électrode
    st.markdown("#### Moyenne — par électrode")
    electrodes = sorted(sessions.keys())
    cols = st.columns(len(electrodes)) if electrodes else []
    for e, col in zip(electrodes, cols):
        with col:
            st.plotly_chart(drt_figure(sessions[e]), width='stretch', key=f"drt_avg_e{e}")

    # (c) onglets par électrode — réplicats + exclusions
    st.markdown("#### Réplicats — sélection par spectre")
    if "drt_exclusions" not in st.session_state:
        st.session_state["drt_exclusions"] = {}

    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
    for e, elec_tab in zip(electrodes, elec_tabs):
        with elec_tab:
            session = sessions[e]
            spectra_map = _spectra_labels_for_session(session)
            if not spectra_map:
                st.info("Aucun spectre disponible pour cette électrode.")
                continue

            sel_label = st.selectbox(
                "Probe / concentration",
                list(spectra_map.keys()),
                key=f"drt_rep_select_e{e}",
            )
            reps, _avg_spectrum = spectra_map[sel_label]

            if not reps:
                st.info("Aucun réplicat individuel disponible pour ce spectre.")
                continue

            elec_excl = st.session_state["drt_exclusions"].setdefault(e, {})
            excluded = elec_excl.setdefault(sel_label, [False] * len(reps))
            if len(excluded) != len(reps):
                excluded = [False] * len(reps)
                elec_excl[sel_label] = excluded

            rep_frs = [sp.fit_results.get("drt_fft") for sp in reps]
            rep_frs_valid = [fr for fr in rep_frs if fr is not None]

            st.plotly_chart(
                drt_replicates_figure(rep_frs_valid, excluded, label=sel_label),
                width='stretch', key=f"drt_rep_fig_e{e}",
            )

            changed = False
            cbox_cols = st.columns(len(reps))
            for i, col in enumerate(cbox_cols):
                with col:
                    new_val = st.checkbox(
                        f"Inclure réplicat {i+1}",
                        value=not excluded[i],
                        key=f"drt_rep_incl_e{e}_{sel_label}_{i}",
                    )
                    if new_val == excluded[i]:
                        excluded[i] = not new_val
                        changed = True

            if changed:
                st.rerun()

    # (d) bouton sauvegarde matplotlib
    if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="drt_matplotlib_btn"):
        open_drt_matplotlib_window(sessions, st.session_state.get("drt_exclusions", {}))


def _render_reconstruction_tab(sessions: dict) -> None:
    """Onglet 3 — Reconstructions Nyquist (Randles vs DRT)."""
    st.subheader("Reconstructions Nyquist — Randles vs DRT")

    # (a) comparaison moyenne probe, toutes électrodes
    st.plotly_chart(reconstruction_comparison_figure(sessions), width='stretch', key="recon_multi")

    # (b) onglets par électrode
    electrodes = sorted(sessions.keys())
    elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
    for e, elec_tab in zip(electrodes, elec_tabs):
        with elec_tab:
            session = sessions[e]
            if session.probe is not None:
                fr_r = session.probe.fit_results.get("randles_full")
                fr_d = session.probe.fit_results.get("drt_fft")
                st.plotly_chart(
                    drt_reconstruction_figure_dual(
                        session.probe, fr_drt=fr_d, fr_randles=fr_r, label="Probe (moyenne)",
                    ),
                    width='stretch', key=f"recon_avg_e{e}",
                )
            else:
                st.info("Aucun spectre probe pour cette électrode.")

            spectra_map = _spectra_labels_for_session(session)
            if not spectra_map:
                continue

            sel_label = st.selectbox(
                "Probe / concentration", list(spectra_map.keys()), key=f"recon_select_e{e}",
            )
            reps, avg_spectrum = spectra_map[sel_label]

            if reps:
                rep_idx = st.selectbox(
                    "Réplicat",
                    list(range(1, len(reps) + 1)),
                    key=f"recon_rep_select_e{e}",
                )
                rep_sp = reps[rep_idx - 1]
                fr_r = rep_sp.fit_results.get("randles_full")
                fr_d = rep_sp.fit_results.get("drt_fft")
                st.plotly_chart(
                    drt_reconstruction_figure_dual(
                        rep_sp, fr_drt=fr_d, fr_randles=fr_r,
                        label=f"{sel_label} — réplicat {rep_idx}",
                    ),
                    width='stretch', key=f"recon_rep_fig_e{e}",
                )
            else:
                st.info("Aucun réplicat individuel disponible pour ce spectre.")

    if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="recon_matplotlib_btn"):
        open_reconstruction_matplotlib_window(sessions)


def _render_calibration_tab(sessions: dict) -> None:
    """Onglet 4 — Calibration EIS."""
    st.subheader("Courbe de calibration — Signal normalisé vs log([c])")

    electrodes = [
        e for e in sorted(sessions.keys())
        if sum(1 for g in sessions[e].groups if g.concentration > 0) >= 2
    ]
    if not electrodes:
        st.info(
            "Chargez au moins **2 spectres d'hybridation** avec "
            "des concentrations positives pour tracer la calibration."
        )
        return

    cols = st.columns(len(electrodes)) if len(electrodes) > 1 else [st.container()]
    for e, col in zip(electrodes, cols):
        with col:
            st.markdown(f"**Électrode {e}**")
            st.plotly_chart(calibration_figure(sessions[e]), width='stretch', key=f"calib_fig_e{e}")

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="calib_matplotlib_btn"):
            open_calibration_matplotlib_window({e: sessions[e] for e in electrodes})
    with btn_cols[1]:
        st.download_button(
            "📥 Télécharger les valeurs (CSV)",
            data=export_calibration_csv({e: sessions[e] for e in electrodes}),
            file_name="eis_calibration.csv",
            mime="text/csv",
            key="calib_csv_btn",
        )


def render_eis_tabs(sessions: dict, normalized: dict, config: dict, validations: dict) -> None:
    """Render EIS analysis tabs: Validation KK / Courbes DRT / Reconstructions / Calibration.

    Args:
        sessions: {electrode_index: EISSession}, one entry per electrode present.
        normalized: output of pages.A_eis._build_normalized_session (Zre_norm/Zim_norm view).
                    Not directly rendered here — the normalized Nyquist view is handled by
                    pages/A_eis.py::_render_three_nyquist, kept out of this function to
                    avoid duplication.
        config: app config dict.
        validations: {electrode_index: {label: ValidationResult}}.
    """
    del normalized, config  # non utilisés directement ici (cf. docstring)

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Validation KK", "2️⃣ Courbes DRT",
        "3️⃣ Reconstructions Nyquist", "4️⃣ Calibration",
    ])

    with tab1:
        _render_kk_tab(validations or {})

    with tab2:
        _render_drt_tab(sessions)

    with tab3:
        _render_reconstruction_tab(sessions)

    with tab4:
        _render_calibration_tab(sessions)


# ─────────────────────────────────────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────────────────────────────────────

def render_cv_tabs(cv_sessions: dict) -> None:
    """Render CV analysis tabs: Visualisation I/U, Pic redox, Calibration.

    Args:
        cv_sessions: {electrode_index: CVSession}, une entrée par électrode disponible.
    """
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Visualisation I/U", "2️⃣ Pic redox", "3️⃣ Calibration", "4️⃣ Paramètres",
    ])

    electrodes = sorted(cv_sessions.keys())

    with tab1:
        st.subheader("Voltampérométrie cyclique — Courant vs Potentiel")
        if not electrodes:
            st.info("Aucune donnée CV chargée.")
        else:
            elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
            for e, elec_tab in zip(electrodes, elec_tabs):
                with elec_tab:
                    session = cv_sessions[e]
                    if session.probe is not None or session.groups:
                        st.plotly_chart(cv_current_figure(session), width='stretch', key=f"cv_current_e{e}")
                    else:
                        st.info("Aucune donnée CV chargée pour cette électrode.")

    with tab2:
        st.subheader("Pics redox — anodique / cathodique")
        if not electrodes:
            st.info("Aucune donnée CV chargée.")
        else:
            elec_tabs = st.tabs([f"Électrode {e}" for e in electrodes])
            for e, elec_tab in zip(electrodes, elec_tabs):
                with elec_tab:
                    session = cv_sessions[e]
                    if session.probe is not None or session.groups:
                        if session.probe is not None:
                            pp = detect_redox_peaks(session.probe)
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Ipa probe", f"{pp['Ipa']*1e6:.3f} µA")
                            col2.metric("Ipc probe", f"{pp['Ipc']*1e6:.3f} µA")
                            col3.metric("ΔEp probe", f"{pp['delta_Ep']*1e3:.1f} mV")
                        st.plotly_chart(redox_peaks_figure(session), width='stretch', key=f"cv_redox_e{e}")
                    else:
                        st.info("Aucune donnée CV chargée pour cette électrode.")

    with tab3:
        st.subheader("Calibration CV — Signal normalisé")
        calibratable = {
            e: s for e, s in cv_sessions.items()
            if sum(1 for g in s.groups if g.concentration > 0) >= 2
        }
        if not calibratable:
            st.info("Ajoutez au moins 2 concentrations pour la calibration.")
        else:
            st.plotly_chart(cv_calibration_figure_multi(calibratable), width='stretch')

            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("🖼 Ouvrir fenêtre de sauvegarde", key="cv_calib_matplotlib_btn"):
                    open_cv_calibration_matplotlib_window_multi(calibratable)
            with btn_cols[1]:
                st.download_button(
                    "📥 Télécharger les valeurs (CSV)",
                    data=export_cv_calibration_csv_multi(calibratable),
                    file_name="cv_calibration.csv",
                    mime="text/csv",
                    key="cv_calib_csv_btn",
                )

    with tab4:
        st.subheader("Paramètres extraits par concentration et électrode")
        if not electrodes:
            st.info("Aucune donnée CV chargée.")
        else:
            st.plotly_chart(cv_params_table_multi(cv_sessions), width='stretch', key="cv_params_table")
