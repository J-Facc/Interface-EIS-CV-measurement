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
from plotting.kk_plots import residuals_figure, validation_summary_table
from exports.exporter import (
    export_params_csv,
    export_spectra_csv,
    export_figure_html,
    export_figure_png,
    export_session_yaml,
)


def render_eis_tabs(session: EISSession, config: dict, validation_results=None) -> None:
    """Render EIS analysis tabs: Nyquist, Bode, DRT, Paramètres, Calibration, Export, Validation KK."""
    tab_nyq, tab_bode, tab_drt, tab_params, tab_calib, tab_export, tab7 = st.tabs([
        "Nyquist", "Bode", "DRT", "Paramètres", "Calibration", "Export", "Validation KK",
    ])

    with tab_nyq:
        # Badges de validité KK
        if validation_results:
            kk_cols = st.columns(min(len(validation_results), 6))
            for kk_col, (kk_label, kk_vr) in zip(kk_cols, validation_results.items()):
                if not kk_vr.all_valid:
                    kk_col.error(f"❌ {kk_label}")
                elif kk_vr.drift_detected:
                    kk_col.warning(f"⚠ {kk_label}")
                else:
                    kk_col.success(f"✅ {kk_label}")
        st.subheader("Diagramme de Nyquist")
        st.plotly_chart(nyquist_figure(session), width='stretch')

    with tab_bode:
        st.subheader("Diagramme de Bode")
        st.plotly_chart(bode_figure(session), width='stretch')

    with tab_drt:
        st.subheader("Distribution des temps de relaxation (DRT)")
        has_drt = (
            any("drt_fft" in grp.fit_results for grp in session.groups)
            or (session.bare is not None and "drt_fft" in session.bare.fit_results)
            or (session.probe is not None and "drt_fft" in session.probe.fit_results)
        )
        if has_drt:
            st.plotly_chart(drt_figure(session), width='stretch')
        else:
            st.info(
                "Activez **DRT (FFT)** dans la sidebar pour afficher "
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

    with tab7:
        st.subheader("Validation Kramers-Kronig")

        if not validation_results:
            st.info("Lancez une analyse pour voir les résultats de validation.")
        else:
            st.markdown("#### Récapitulatif")
            fig_table = validation_summary_table(validation_results, theme_mode="light")
            st.plotly_chart(fig_table, width='stretch')

            st.markdown("#### Résidus par spectre")
            labels_kk = list(validation_results.keys())
            selected_kk = st.selectbox("Spectre", labels_kk, key="kk_select")
            vr = validation_results[selected_kk]

            fig_res = residuals_figure(
                vr,
                theme_mode="light",
                residual_threshold_pct=getattr(config, "kk_residual_pct", 2.0),
            )
            st.plotly_chart(fig_res, width='stretch')

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


def render_tabs(session: EISSession, config: dict, cv_session: CVSession = None, validation_results=None) -> None:
    """Backward-compatible alias — delegates to render_eis_tabs."""
    render_eis_tabs(session, config, validation_results=validation_results)
