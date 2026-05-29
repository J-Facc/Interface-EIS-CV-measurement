"""Streamlit sidebar: file upload, step assignment, concentration input, physics editor."""

import streamlit as st


def render_sidebar() -> tuple:
    """Render the full sidebar and collect user inputs.

    Returns:
        Tuple of:
        - file_assignments (list[dict]): Each dict has keys
          content (bytes), filename (str), step (str), concentration (float).
        - active_models (list[str]): Checked model names.
        - run_clicked (bool): True when the user clicks "Analyser".
        - phys_overrides (dict): User-edited physical parameter overrides.
    """
    with st.sidebar:
        st.title("⚡ EIS Analyzer")

        # ── File upload ────────────────────────────────────────────────────────
        st.subheader("1 · Importer les spectres")

        uploaded_files = st.file_uploader(
            "Fichiers CSV / TXT",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="uploaded_files",
        )

        file_assignments: list = []

        if uploaded_files:
            st.markdown("**Assigner chaque fichier :**")
            for uf in uploaded_files:
                short_name = uf.name[:24] + ("…" if len(uf.name) > 24 else "")
                st.markdown(f"**`{short_name}`**")
                cols = st.columns([2, 3])

                with cols[0]:
                    step = st.selectbox(
                        "Étape",
                        options=["bare", "probe", "hybridization"],
                        key=f"step_{uf.name}",
                        label_visibility="collapsed",
                    )

                concentration = 0.0
                if step == "hybridization":
                    with cols[1]:
                        mant = st.number_input(
                            "Mantisse",
                            value=1.0, min_value=0.1, max_value=9.9, step=0.1,
                            key=f"mant_{uf.name}",
                            label_visibility="collapsed",
                            help="Mantisse (0.1 – 9.9)",
                        )
                    exp_val = st.number_input(
                        "Exposant (×10ˣ M)",
                        value=-13, min_value=-20, max_value=0, step=1,
                        key=f"exp_{uf.name}",
                    )
                    concentration = float(mant) * (10 ** int(exp_val))

                content = uf.read()
                uf.seek(0)

                file_assignments.append({
                    "content": content,
                    "filename": uf.name,
                    "step": step,
                    "concentration": concentration,
                })

        st.markdown("---")

        # ── Model selection ────────────────────────────────────────────────────
        st.subheader("2 · Modèles de fit")

        _model_choices = {
            "circular": "Fit circulaire",
            "randles_constrained": "Randles contraint",
            "randles_full": "Randles complet",
            "drt_tikhonov": "DRT Tikhonov",
        }

        active_models: list = []
        for mname, mlabel in _model_choices.items():
            if st.checkbox(mlabel, value=(mname == "circular"), key=f"model_{mname}"):
                active_models.append(mname)

        st.markdown("---")

        # ── Physical parameters ────────────────────────────────────────────────
        st.subheader("3 · Paramètres physiques")

        phys_overrides: dict = {}
        with st.expander("Éditer les paramètres", expanded=False):
            phys_overrides["Fv"] = st.number_input(
                "Fv — débit volumique (m³/s)",
                value=5e-10, format="%.2e", key="Fv",
            )
            phys_overrides["xe"] = st.number_input(
                "xe — largeur WE (m)",
                value=30e-6, format="%.2e", key="xe",
            )
            phys_overrides["h"] = st.number_input(
                "h — hauteur canal (m)",
                value=60e-6, format="%.2e", key="h",
            )
            phys_overrides["d"] = st.number_input(
                "d — largeur canal (m)",
                value=300e-6, format="%.2e", key="d",
            )
            phys_overrides["T"] = st.number_input(
                "T — température (K)",
                value=298.0, format="%.1f", key="T",
            )
            phys_overrides["C0"] = st.number_input(
                "C0 — concentration médiateur (M)",
                value=0.02, format="%.4f", key="C0",
            )

        st.markdown("---")

        run_clicked = st.button(
            "▶  Analyser",
            type="primary",
            use_container_width=True,
        )

    return file_assignments, active_models, run_clicked, phys_overrides
