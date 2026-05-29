"""Streamlit tab rendering: Nyquist, Bode, DRT, Paramètres, Calibration, Export, CV."""

import streamlit as st

from core.models import EISSession
from core.cv_models import CVSession
from plotting.eis_plots import (
    nyquist_figure,
    bode_figure,
    drt_figure,
    params_table_figure,
    calibration_figure,
)
from plotting.cv_plots import cv_current_figure, cv_calibration_figure
from exports.exporter import (
    export_params_csv,
    export_spectra_csv,
    export_figure_html,
    export_figure_png,
    export_session_yaml,
)


def render_eis_tabs(session: EISSession, config: dict) -> None:
    """Render EIS analysis tabs: Nyquist, Bode, DRT, Paramètres, Calibration, Export."""
    tab_nyq, tab_bode, tab_drt, tab_params, tab_calib, tab_export = st.tabs([
        "Nyquist", "Bode", "DRT", "Paramètres", "Calibration", "Export",
    ])

    with tab_nyq:
        st.subheader("Diagramme de Nyquist")
        st.plotly_chart(nyquist_figure(session), width='stretch')

    with tab_bode:
        st.subheader("Diagramme de Bode")
        st.plotly_chart(bode_figure(session), width='stretch')

    with tab_drt:
        st.subheader("Distribution des temps de relaxation (DRT)")
        has_drt = any(
            "drt_tikhonov" in grp.fit_results
            for grp in session.groups
        )
        if has_drt:
            st.plotly_chart(drt_figure(session), width='stretch')
        else:
            st.info(
                "Activez **DRT Tikhonov** dans la sidebar pour afficher "
                "la distribution des temps de relaxation."
            )

    with tab_params:
        st.subheader("Paramètres extraits")
        st.plotly_chart(params_table_figure(session), width='stretch')

    with tab_calib:
        st.subheader("Courbe de calibration log(Rct) vs log([c])")
        has_hyb = sum(1 for g in session.groups if g.concentration > 0) >= 2
        if has_hyb:
            st.plotly_chart(calibration_figure(session), width='stretch')
        else:
            st.info(
                "Chargez au moins **2 spectres d'hybridation** avec "
                "des concentrations positives pour tracer la calibration."
            )

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


def render_cv_tabs(cv_session: CVSession) -> None:
    """Render CV analysis tabs: Courbes I/E, Calibration."""
    tab_ie, tab_calib = st.tabs(["📉 Courbes I/E", "📊 Calibration"])

    with tab_ie:
        st.subheader("Voltampérométrie cyclique — Courant vs Potentiel")
        if cv_session.probe is not None or cv_session.groups:
            st.plotly_chart(cv_current_figure(cv_session), width='stretch')
        else:
            st.info("Aucune donnée CV chargée.")

    with tab_calib:
        st.subheader("Calibration CV — Signal normalisé")
        if len(cv_session.groups) >= 2:
            st.plotly_chart(cv_calibration_figure(cv_session), width='stretch')
        else:
            st.info("Ajoutez au moins 2 concentrations pour la calibration.")


def render_tabs(session: EISSession, config: dict, cv_session: CVSession = None) -> None:
    """Backward-compatible alias — delegates to render_eis_tabs."""
    render_eis_tabs(session, config)
