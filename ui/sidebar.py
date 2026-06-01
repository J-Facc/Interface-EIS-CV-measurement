"""Streamlit sidebar: step-by-step file upload with dynamic concentration list."""

import uuid
import streamlit as st


def _init_session_state() -> None:
    if "eis_concentrations" not in st.session_state:
        st.session_state["eis_concentrations"] = []
    if "cv_concentrations" not in st.session_state:
        st.session_state["cv_concentrations"] = []
    if "_conc_counter" not in st.session_state:
        st.session_state["_conc_counter"] = 0


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


def render_sidebar(mode: str = "eis") -> tuple:
    """Render the sidebar for the given mode.

    Args:
        mode: "eis" or "cv".

    Returns (EIS mode):
        file_assignments, active_models, run_clicked, phys_overrides

    Returns (CV mode):
        cv_assignments, run_clicked_cv
    """
    _init_session_state()

    with st.sidebar:
        st.title("⚡ EIS Analyzer")

        if mode == "eis":
            return _render_eis_sidebar()
        else:
            return _render_cv_sidebar()


def _render_eis_sidebar() -> tuple:
    st.subheader("⚡ Électrode nue (Bare EIS)")
    bare_files = st.file_uploader(
        "Fichiers CSV / TXT (réplicats → moyennage auto)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="eis_bare_files",
    )

    st.markdown("---")

    st.subheader("⚡ Sonde (Probe EIS)")
    probe_files = st.file_uploader(
        "Fichiers CSV / TXT (réplicats → moyennage auto)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="eis_probe_files",
    )

    st.markdown("---")

    st.subheader("⚡ Hybridations EIS")
    _render_concentration_list("eis_concentrations", ["csv", "txt"], "eis")

    st.markdown("---")

    st.subheader("Modèles de fit")
    _model_choices = {
        "circular": "Fit circulaire",
        "randles_constrained": "Randles contraint",
        "randles_full": "Randles complet",
        "drt_fft": "DRT (FFT)",
    }
    active_models: list = []
    for mname, mlabel in _model_choices.items():
        if st.checkbox(mlabel, value=(mname == "circular"), key=f"model_{mname}"):
            active_models.append(mname)

    st.markdown("---")

    st.subheader("Paramètres physiques")
    phys_overrides: dict = {}
    with st.expander("Éditer les paramètres", expanded=False):
        phys_overrides["Fv"] = st.number_input(
            "Fv — débit volumique (m³/s)", value=5e-10, format="%.2e", key="Fv",
        )
        phys_overrides["xe"] = st.number_input(
            "xe — largeur WE (m)", value=30e-6, format="%.2e", key="xe",
        )
        phys_overrides["h"] = st.number_input(
            "h — hauteur canal (m)", value=60e-6, format="%.2e", key="h",
        )
        phys_overrides["d"] = st.number_input(
            "d — largeur canal (m)", value=300e-6, format="%.2e", key="d",
        )
        phys_overrides["T"] = st.number_input(
            "T — température (K)", value=298.0, format="%.1f", key="T",
        )
        phys_overrides["C0"] = st.number_input(
            "C0 — concentration médiateur (M)", value=0.02, format="%.4f", key="C0",
        )

    st.markdown("---")

    run_clicked = st.button("▶ Analyser EIS", type="primary", use_container_width=True)

    # Build file_assignments
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

    for item in st.session_state["eis_concentrations"]:
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

    return file_assignments, active_models, run_clicked, phys_overrides


def _render_cv_sidebar() -> tuple:
    st.subheader("📈 Électrode nue (Bare CV)")
    bare_cv_files = st.file_uploader(
        "Fichiers CSV / TXT (réplicats → moyennage auto)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="cv_bare_files",
    )

    st.markdown("---")

    st.subheader("📈 Sonde (Probe CV)")
    probe_cv_files = st.file_uploader(
        "Fichiers CSV / TXT (réplicats → moyennage auto)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="cv_probe_files",
    )

    st.markdown("---")

    st.subheader("📈 Hybridations CV")
    _render_concentration_list("cv_concentrations", ["csv", "txt"], "cv")

    st.markdown("---")

    run_clicked_cv = st.button("▶ Analyser CV", type="primary", use_container_width=True)

    # Build cv_assignments
    cv_assignments: list = []

    for uf in (bare_cv_files or []):
        content = uf.read()
        uf.seek(0)
        cv_assignments.append({
            "content": content,
            "filename": uf.name,
            "step": "bare",
            "concentration": 0.0,
        })

    for uf in (probe_cv_files or []):
        content = uf.read()
        uf.seek(0)
        cv_assignments.append({
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
            cv_assignments.append({
                "content": content,
                "filename": uf.name,
                "step": "hybridization",
                "concentration": concentration,
            })

    return cv_assignments, run_clicked_cv
