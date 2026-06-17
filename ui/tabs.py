"""Streamlit tab rendering: Nyquist, Bode, DRT, Paramètres, Calibration, Export, CV."""

import numpy as np
import streamlit as st

from core.models import EISSession
from core.cv_models import CVSession
from plotting.eis_plots import (
    nyquist_figure,
    bode_figure,
    drt_figure,
    params_table_figure,
    calibration_figure,
    kk_figure,
    drt_tikhonov_figure,
    drt_fft_figure,
    drt_reconstruction_figure,
    calibration_drt_figure,
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

        spectra_by_label = {}
        if session.bare is not None:
            spectra_by_label["Bare"] = session.bare
        if session.probe is not None:
            spectra_by_label["Probe"] = session.probe
        for grp in session.groups:
            spectra_by_label[f"{grp.concentration:.2e} M"] = grp.spectrum

        has_drt = any(
            "drt_fft" in sp.fit_results or "drt_tikhonov" in sp.fit_results
            for sp in spectra_by_label.values()
        )

        if not has_drt:
            st.info(
                "Activez **DRT Tikhonov + NNLS** et/ou **DRT FFT Wiener** dans la "
                "sidebar pour afficher la distribution des temps de relaxation."
            )
        else:
            st.plotly_chart(drt_figure(session), width='stretch')

            sel_label = st.selectbox(
                "Spectre", list(spectra_by_label.keys()), key="drt_spectrum_select",
            )
            sel_sp = spectra_by_label[sel_label]

            with st.expander("1️⃣ Validation Kramers-Kronig", expanded=False):
                kk_result = None
                for fr in sel_sp.fit_results.values():
                    if fr.kk_residuals is not None:
                        kk_result = fr.kk_residuals
                        break
                if kk_result is None:
                    st.info("Validation KK non disponible pour ce spectre.")
                else:
                    st.plotly_chart(
                        kk_figure(sel_sp, kk_result, label=sel_label), width='stretch',
                    )

            with st.expander("2️⃣ DRT Tikhonov + NNLS", expanded=False):
                fr_tik = sel_sp.fit_results.get("drt_tikhonov")
                if fr_tik is None:
                    st.info("Activez **DRT Tikhonov + NNLS** dans la sidebar.")
                else:
                    st.plotly_chart(
                        drt_tikhonov_figure(fr_tik, label=sel_label), width='stretch',
                    )

            with st.expander("3️⃣ DRT FFT Wiener", expanded=False):
                fr_fft = sel_sp.fit_results.get("drt_fft")
                if fr_fft is None:
                    st.info("Activez **DRT FFT Wiener** dans la sidebar.")
                else:
                    w_log10 = st.slider(
                        "Filtre Wiener W (log₁₀) — re-calcul en direct",
                        min_value=-10.0, max_value=-5.0,
                        value=float(np.log10(fr_fft.params.get("W", 1e-8))),
                        step=0.5, key="drt_fft_w_log10_live",
                    )
                    live_W = 10 ** w_log10
                    if abs(live_W - fr_fft.params.get("W", 1e-8)) / fr_fft.params.get("W", 1e-8) > 1e-9:
                        from fits.registry import get_model
                        live_config = dict(config)
                        live_fit_cfg = dict(live_config.get("fit", {}))
                        live_fit_cfg["drt_wiener_W"] = live_W
                        live_config["fit"] = live_fit_cfg
                        fr_fft = get_model("drt_fft").fit(sel_sp, live_config)
                    st.plotly_chart(
                        drt_fft_figure(fr_fft, label=sel_label), width='stretch',
                    )

            with st.expander("4️⃣ Comparaison reconstruction", expanded=False):
                cols_rec = st.columns(2)
                if fr_tik is not None:
                    cols_rec[0].plotly_chart(
                        drt_reconstruction_figure(sel_sp, fr_tik, label=f"{sel_label} — Tikhonov"),
                        width='stretch',
                    )
                if fr_fft is not None:
                    cols_rec[1].plotly_chart(
                        drt_reconstruction_figure(sel_sp, fr_fft, label=f"{sel_label} — FFT Wiener"),
                        width='stretch',
                    )
                if fr_tik is None and fr_fft is None:
                    st.info("Aucun modèle DRT actif pour ce spectre.")

            with st.expander("5️⃣ Calibration DRT", expanded=False):
                drt_model_choice = st.selectbox(
                    "Modèle DRT", ["drt_fft", "drt_tikhonov"], key="drt_calib_model",
                )
                has_hyb_drt = sum(
                    1 for g in session.groups
                    if g.concentration > 0 and drt_model_choice in g.fit_results
                ) >= 2
                if has_hyb_drt:
                    st.plotly_chart(
                        calibration_drt_figure(session, model_name=drt_model_choice),
                        width='stretch',
                    )
                else:
                    st.info(
                        "Chargez au moins **2 spectres d'hybridation** avec un fit "
                        f"`{drt_model_choice}` pour tracer la calibration DRT."
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
