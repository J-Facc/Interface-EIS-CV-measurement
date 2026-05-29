"""Streamlit tab rendering: Nyquist, Bode, DRT, Paramètres, Calibration, Export, CV."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from core.models import EISSession
from plotting.eis_plots import (
    nyquist_figure,
    bode_figure,
    drt_figure,
    params_table_figure,
    calibration_figure,
)
from exports.exporter import (
    export_params_csv,
    export_spectra_csv,
    export_figure_html,
    export_figure_png,
    export_session_yaml,
)


def render_tabs(
    session: EISSession,
    config: dict,
    cv_session=None,
) -> None:
    """Render all seven analysis tabs.

    Args:
        session: EISSession with loaded spectra and fit results.
        config: App config dict (used by export functions).
        cv_session: Optional CVSession for the CV tab.
    """
    tab_nyq, tab_bode, tab_drt, tab_params, tab_calib, tab_export, tab_cv = st.tabs([
        "Nyquist", "Bode", "DRT", "Paramètres", "Calibration", "Export", "CV",
    ])

    # ── Nyquist ─────────────────────────────────────────────────────────────
    with tab_nyq:
        st.subheader("Diagramme de Nyquist")
        fig = nyquist_figure(session)
        st.plotly_chart(fig, use_container_width=True)

    # ── Bode ─────────────────────────────────────────────────────────────────
    with tab_bode:
        st.subheader("Diagramme de Bode")
        fig = bode_figure(session)
        st.plotly_chart(fig, use_container_width=True)

    # ── DRT ──────────────────────────────────────────────────────────────────
    with tab_drt:
        st.subheader("Distribution des temps de relaxation (DRT)")
        has_drt = any(
            "drt_tikhonov" in grp.fit_results
            for grp in session.groups
        )
        if has_drt:
            fig = drt_figure(session)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Activez **DRT Tikhonov** dans la sidebar pour afficher "
                "la distribution des temps de relaxation."
            )

    # ── Parameters ───────────────────────────────────────────────────────────
    with tab_params:
        st.subheader("Paramètres extraits")
        fig = params_table_figure(session)
        st.plotly_chart(fig, use_container_width=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    with tab_calib:
        st.subheader("Courbe de calibration log(Rct) vs log([c])")
        has_hyb = sum(1 for g in session.groups if g.concentration > 0) >= 2
        if has_hyb:
            fig = calibration_figure(session)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Chargez au moins **2 spectres d'hybridation** avec "
                "des concentrations positives pour tracer la calibration."
            )

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Télécharger les résultats")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📄 Paramètres CSV",
                data=export_params_csv(session),
                file_name="eis_params.csv",
                mime="text/csv",
            )
            st.download_button(
                label="📊 Spectres CSV",
                data=export_spectra_csv(session),
                file_name="eis_spectra.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                label="🗂 Session YAML",
                data=export_session_yaml(session),
                file_name="eis_session.yaml",
                mime="text/yaml",
            )

            nyq_fig = nyquist_figure(session)
            st.download_button(
                label="🌐 Nyquist HTML",
                data=export_figure_html(nyq_fig),
                file_name="nyquist.html",
                mime="text/html",
            )

            try:
                png_data = export_figure_png(nyq_fig, config)
                st.download_button(
                    label="🖼 Nyquist PNG",
                    data=png_data,
                    file_name="nyquist.png",
                    mime="image/png",
                )
            except RuntimeError as exc:
                st.caption(str(exc))

    # ── CV ────────────────────────────────────────────────────────────────────
    with tab_cv:
        st.subheader("Voltampérométrie cyclique (CV)")
        if cv_session is None or (cv_session.probe is None and not cv_session.groups):
            st.info("Chargez des fichiers CV dans la sidebar (section CV) puis cliquez sur ▶ Analyser.")
        else:
            from plotting.cv_plots import cv_current_figure, cv_calibration_figure
            theme_mode = "light"

            st.markdown("### Courbes I vs E")
            fig_ie = cv_current_figure(cv_session, theme_mode)
            st.plotly_chart(fig_ie, use_container_width=True)

            has_calib = any(grp.concentration > 0 for grp in cv_session.groups)
            if has_calib:
                st.markdown("### Courbe de calibration")
                fig_cal = cv_calibration_figure(cv_session, theme_mode)
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.info("Ajoutez des fichiers CV d'hybridation pour afficher la courbe de calibration.")
