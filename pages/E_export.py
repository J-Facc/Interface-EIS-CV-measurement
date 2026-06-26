"""Page Export — téléchargement des tableaux calculés et de l'archive complète.

Lit les sessions EIS (st.session_state['eis_sessions']), la vue normalisée
(st.session_state['eis_normalized']) et les sessions CV par électrode
(st.session_state['cv_sessions']) tels que réellement produits par
pages/A_eis.py et pages/B_cv.py.
"""

import streamlit as st

from exports.exporter import (
    export_drt_csv,
    export_normalization_csv,
    export_params_csv,
    export_reconstruction_csv,
    export_calibration_csv,
    export_full_zip,
)


def main() -> None:
    st.title("💾 Export des données")
    st.markdown("Téléchargez les tableaux calculés ou une archive complète.")

    sessions   = st.session_state.get("eis_sessions")
    normalized = st.session_state.get("eis_normalized") or {}
    cv_sessions = st.session_state.get("cv_sessions")
    experiment_clean = st.session_state.get("experiment_clean")
    config     = st.session_state.get("eis_config")

    if not sessions:
        st.info("Lancez d'abord une analyse EIS depuis la page 'EIS seule'.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tableaux individuels")
        st.download_button(
            "📐 Valeurs DRT (CSV)",
            export_drt_csv(sessions),
            file_name="drt_values.csv",
            mime="text/csv",
        )
        st.download_button(
            "📏 Normalisation Nyquist (CSV)",
            export_normalization_csv(normalized),
            file_name="nyquist_normalise.csv",
            mime="text/csv",
        )
        st.download_button(
            "🔧 Paramètres fit Randles (CSV)",
            export_params_csv(sessions),
            file_name="parametres_randles.csv",
            mime="text/csv",
        )
        st.download_button(
            "🔁 Reconstructions (CSV)",
            export_reconstruction_csv(sessions),
            file_name="reconstructions.csv",
            mime="text/csv",
        )
        st.download_button(
            "📊 Calibration EIS (CSV)",
            export_calibration_csv(sessions),
            file_name="eis_calibration.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("Archive complète")
        st.download_button(
            "🗂 Télécharger tout (ZIP)",
            export_full_zip(experiment_clean, sessions, normalized, cv_sessions, config),
            file_name="eis_cv_export.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
