"""Streamlit sidebar: step-by-step file upload with dynamic concentration list."""

import uuid
import streamlit as st


def _init_session_state() -> None:
    if "cv_concentrations" not in st.session_state:
        st.session_state["cv_concentrations"] = []
    if "_conc_counter" not in st.session_state:
        st.session_state["_conc_counter"] = 0
    if "cv_eis_concentrations" not in st.session_state:
        st.session_state["cv_eis_concentrations"] = []


def _add_concentration(key: str) -> None:
    st.session_state[key].append({
        "id": str(uuid.uuid4()),
        "mantisse": 1.0,
        "exposant": -13,
        "files": None,
    })


def _remove_concentration(key: str, item_id: str) -> None:
    st.session_state[key] = [
        c for c in st.session_state[key] if c["id"] != item_id
    ]


def _render_concentration_list(session_key: str, file_type: list, uploader_prefix: str) -> None:
    """Render a dynamic list of concentration rows with file uploaders."""
    conc_list = st.session_state[session_key]

    for item in conc_list:
        item_id = item["id"]
        cols = st.columns([3, 2, 1])

        with cols[0]:
            item["mantisse"] = st.number_input(
                "Mantisse",
                value=float(item["mantisse"]),
                min_value=0.1,
                max_value=9.9,
                step=0.1,
                format="%.1f",
                key=f"{uploader_prefix}_mant_{item_id}",
                label_visibility="collapsed",
                help="Mantisse (0.1 – 9.9)",
            )

        with cols[1]:
            item["exposant"] = st.number_input(
                "×10ˣ M",
                value=int(item["exposant"]),
                min_value=-20,
                max_value=0,
                step=1,
                key=f"{uploader_prefix}_exp_{item_id}",
                label_visibility="collapsed",
                help="Exposant entier (×10ˣ M)",
            )

        with cols[2]:
            if st.button("✕", key=f"{uploader_prefix}_del_{item_id}", help="Supprimer cette concentration"):
                _remove_concentration(session_key, item_id)
                st.rerun()

        item["files"] = st.file_uploader(
            f"Fichiers pour {item['mantisse']:.1f}×10^{item['exposant']} M",
            type=file_type,
            accept_multiple_files=True,
            key=f"{uploader_prefix}_files_{item_id}",
        )

    if st.button("➕ Ajouter une concentration", use_container_width=True, key=f"{uploader_prefix}_add"):
        _add_concentration(session_key)
        st.rerun()


def render_sidebar() -> tuple:
    """Render the full sidebar and collect user inputs.

    Returns:
        Tuple of:
        - file_assignments (list[dict]): EIS files, each dict has keys
          content (bytes), filename (str), step (str), concentration (float).
        - cv_assignments (list[dict]): CV files, same structure.
        - active_models (list[str]): Checked model names.
        - run_clicked (bool): True when the user clicks "Analyser".
        - theme_mode (str): "light" or "dark" (placeholder, always "light").
        - phys_overrides (dict): User-edited physical parameter overrides.
    """
    _init_session_state()

    with st.sidebar:
        st.title("⚡ EIS Analyzer")

        # ── EIS section ───────────────────────────────────────────────────────
        st.subheader("EIS — Impédance électrochimique")

        # ── 1. Électrode nue ──────────────────────────────────────────────────
        st.subheader("1 · Électrode nue (Bare)")
        bare_files = st.file_uploader(
            "Fichiers CSV / TXT (réplicats → moyennage auto)",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="bare_files",
        )

        st.markdown("---")

        # ── 2. Sonde ──────────────────────────────────────────────────────────
        st.subheader("2 · Sonde (Probe)")
        probe_files = st.file_uploader(
            "Fichiers CSV / TXT (réplicats → moyennage auto)",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="probe_files",
        )

        st.markdown("---")

        # ── 3. Hybridations EIS ───────────────────────────────────────────────
        st.subheader("3 · Hybridations")
        _render_concentration_list("cv_concentrations", ["csv", "txt"], "eis")

        st.markdown("---")

        # ── Modèles de fit ────────────────────────────────────────────────────
        st.subheader("4 · Modèles de fit")

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

        # ── Paramètres physiques ──────────────────────────────────────────────
        st.subheader("5 · Paramètres physiques")

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

        # ── CV section ────────────────────────────────────────────────────────
        st.subheader("📈 CV — Voltampérométrie cyclique")

        cv_probe_files = st.file_uploader(
            "Probe CV — fichiers CSV / TXT (réplicats → moyennage auto)",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="cv_probe_files",
        )

        st.markdown("Concentrations CV :")
        _render_concentration_list("cv_eis_concentrations", ["csv", "txt"], "cv")

        st.markdown("---")

        run_clicked = st.button(
            "▶  Analyser",
            type="primary",
            use_container_width=True,
        )

    # ── Construction de file_assignments (EIS) ────────────────────────────────
    file_assignments: list = []

    for uf in (bare_files or []):
        content = uf.read()
        uf.seek(0)
        file_assignments.append({
            "content": content,
            "filename": uf.name,
            "step": "bare",
            "concentration": 0.0,
        })

    for uf in (probe_files or []):
        content = uf.read()
        uf.seek(0)
        file_assignments.append({
            "content": content,
            "filename": uf.name,
            "step": "probe",
            "concentration": 0.0,
        })

    for item in st.session_state["cv_concentrations"]:
        concentration = float(item["mantisse"]) * (10 ** int(item["exposant"]))
        for uf in (item["files"] or []):
            content = uf.read()
            uf.seek(0)
            file_assignments.append({
                "content": content,
                "filename": uf.name,
                "step": "hybridization",
                "concentration": concentration,
            })

    # ── Construction de cv_assignments ────────────────────────────────────────
    cv_assignments: list = []

    for uf in (cv_probe_files or []):
        content = uf.read()
        uf.seek(0)
        cv_assignments.append({
            "content": content,
            "filename": uf.name,
            "step": "probe",
            "concentration": 0.0,
        })

    for item in st.session_state["cv_eis_concentrations"]:
        concentration = float(item["mantisse"]) * (10 ** int(item["exposant"]))
        for uf in (item["files"] or []):
            content = uf.read()
            uf.seek(0)
            cv_assignments.append({
                "content": content,
                "filename": uf.name,
                "step": "hybridization",
                "concentration": concentration,
            })

    theme_mode = "light"

    return file_assignments, cv_assignments, active_models, run_clicked, theme_mode, phys_overrides
