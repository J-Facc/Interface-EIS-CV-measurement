"""Composant Streamlit réutilisable pour l'entrée des données expérimentales.

Utilisé par les pages A (EIS seul), B (CV seul), C (comparatif) et D (inférence).
Aucun import Streamlit en dehors de ce module — toute la logique UI est ici.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Clés de session_state
# ---------------------------------------------------------------------------

def _sk(prefix: str, *parts) -> str:
    """Construit une clé session_state préfixée pour éviter les collisions entre pages."""
    return "__di_" + prefix + "_" + "_".join(str(p) for p in parts)


def _init_state(prefix: str, n_conc: int, n_elec: int) -> None:
    key = _sk(prefix, "conc_params")
    if key not in st.session_state:
        st.session_state[key] = [
            {"mantisse": 1.0, "exposant": -8 - i} for i in range(n_conc)
        ]


def _get_conc_params(prefix: str) -> list[dict]:
    return st.session_state[_sk(prefix, "conc_params")]


def _resize_conc_list(prefix: str, new_n: int) -> None:
    key = _sk(prefix, "conc_params")
    current = st.session_state.get(key, [])
    if len(current) < new_n:
        last_exp = current[-1]["exposant"] if current else -8
        while len(current) < new_n:
            current.append({"mantisse": 1.0, "exposant": last_exp - 1})
            last_exp -= 1
    elif len(current) > new_n:
        current = current[:new_n]
    st.session_state[key] = current


# ---------------------------------------------------------------------------
# Saisie des concentrations
# ---------------------------------------------------------------------------

def _render_concentration_inputs(prefix: str, n_conc: int) -> list[float]:
    """
    Affiche n_conc lignes mantisse × 10^ exposant.
    Retourne la liste des concentrations en molaire.
    """
    conc_params = _get_conc_params(prefix)
    concentrations = []

    st.markdown("**Concentrations de calibration**")
    for i, cp in enumerate(conc_params):
        col_m, col_sep, col_e, col_unit = st.columns([2, 0.3, 1.5, 1])
        with col_m:
            cp["mantisse"] = st.number_input(
                f"Mantisse {i + 1}",
                value=float(cp["mantisse"]),
                min_value=0.1,
                max_value=9.9,
                step=0.1,
                format="%.1f",
                key=_sk(prefix, f"mant_{i}"),
                label_visibility="collapsed",
            )
        with col_sep:
            st.markdown("<div style='padding-top:32px'>×10^</div>", unsafe_allow_html=True)
        with col_e:
            cp["exposant"] = st.number_input(
                f"Exposant {i + 1}",
                value=int(cp["exposant"]),
                min_value=-20,
                max_value=0,
                step=1,
                key=_sk(prefix, f"exp_{i}"),
                label_visibility="collapsed",
            )
        with col_unit:
            st.markdown("<div style='padding-top:32px'>M</div>", unsafe_allow_html=True)

        concentrations.append(float(cp["mantisse"]) * (10 ** int(cp["exposant"])))

    return concentrations


# ---------------------------------------------------------------------------
# Upload d'un bloc de fichiers (réplicats + probe)
# ---------------------------------------------------------------------------

def _render_file_block(
    label: str,
    prefix: str,
    conc_idx: int,
    elec_idx: int,
    n_rep: int,
    signal_type: str,
) -> tuple[list, object]:
    """
    Affiche les uploaders pour n_rep réplicats et 1 probe d'une électrode.
    Retourne ([file_rep1, ...], file_probe).
    """
    st.markdown(f"**{label}**")
    rep_files = []
    for r in range(n_rep):
        f = st.file_uploader(
            f"Réplicat {r + 1}",
            type=["csv", "txt"],
            key=_sk(prefix, signal_type, f"c{conc_idx}_e{elec_idx}_r{r}"),
            label_visibility="visible",
        )
        rep_files.append(f)

    probe_file = st.file_uploader(
        "Probe",
        type=["csv", "txt"],
        key=_sk(prefix, signal_type, f"c{conc_idx}_e{elec_idx}_probe"),
        label_visibility="visible",
    )
    return rep_files, probe_file


# ---------------------------------------------------------------------------
# Assemblage d'un bloc signal complet (toutes concentrations, toutes électrodes)
# ---------------------------------------------------------------------------

def _render_signal_block(
    prefix: str,
    signal_type: str,          # 'eis' | 'cv'
    n_conc: int,
    n_elec: int,
    n_rep: int,
    concentrations: list[float],
    val_prefix: str | None = None,  # préfixe alternatif pour la validation
) -> dict:
    """
    Construit le dict signal complet :
    {
      'electrode_1': [[f, ...], ...],   # [c][rep]
      'electrode_2': [[f, ...], ...],
      'probe_1': [f, ...],              # [c]
      'probe_2': [f, ...],
    }
    """
    pfx = val_prefix if val_prefix else prefix
    result: dict = {}

    for e in range(1, n_elec + 1):
        result[f"electrode_{e}"] = [None] * n_conc
        result[f"probe_{e}"] = [None] * n_conc

    icon = "⚡" if signal_type == "eis" else "📈"
    type_label = "EIS" if signal_type == "eis" else "CV"

    for ci in range(n_conc):
        c_label = f"{concentrations[ci]:.2e} M"
        with st.expander(f"{icon} {type_label} — Concentration {ci + 1} : {c_label}", expanded=False):
            elec_cols = st.columns(n_elec)
            for e in range(1, n_elec + 1):
                with elec_cols[e - 1]:
                    reps, probe = _render_file_block(
                        label=f"Électrode {e}",
                        prefix=pfx,
                        conc_idx=ci,
                        elec_idx=e,
                        n_rep=n_rep,
                        signal_type=signal_type,
                    )
                    result[f"electrode_{e}"][ci] = reps
                    result[f"probe_{e}"][ci] = probe

    return result


# ---------------------------------------------------------------------------
# Comptage des fichiers pour l'indicateur de progression
# ---------------------------------------------------------------------------

def _count_files(signal_dict: dict | None, n_conc: int, n_elec: int, n_rep: int) -> tuple[int, int]:
    """Retourne (fichiers_uploadés, total_attendu) pour un bloc signal."""
    if signal_dict is None:
        return 0, 0

    # total : n_conc × n_elec × (n_rep + 1 probe)
    total = n_conc * n_elec * (n_rep + 1)
    uploaded = 0

    for e in range(1, n_elec + 1):
        reps_list = signal_dict.get(f"electrode_{e}", [])
        probe_list = signal_dict.get(f"probe_{e}", [])
        for ci in range(n_conc):
            reps = reps_list[ci] if ci < len(reps_list) else []
            if reps:
                uploaded += sum(1 for f in reps if f is not None)
            probe = probe_list[ci] if ci < len(probe_list) else None
            if probe is not None:
                uploaded += 1

    return uploaded, total


def _all_required_uploaded(signal_dict: dict | None, n_conc: int, n_elec: int, n_rep: int) -> bool:
    if signal_dict is None:
        return True  # non requis dans ce mode
    up, total = _count_files(signal_dict, n_conc, n_elec, n_rep)
    return up >= total


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def render_data_input(mode: str, prefix: str = "main") -> dict:
    """
    Composant Streamlit d'entrée des données expérimentales.

    Affiche les trois étapes décrites dans PROJECT.md §5 et retourne
    un dict structuré prêt à être consommé par les pages A, B, C et D.

    Parameters
    ----------
    mode : str
        'eis_only' — upload EIS uniquement
        'cv_only'  — upload CV uniquement
        'both'     — upload EIS + CV (page C comparatif)
    prefix : str
        Préfixe pour les clés session_state. Doit être unique par page
        si plusieurs instances coexistent. Défaut : 'main'.

    Returns
    -------
    dict avec les clés :
        concentrations   : list[float]   — en molaire
        n_electrodes     : int
        n_replicats      : int
        calibration      : dict
            eis          : dict | None
            cv           : dict | None
        validation       : dict | None
            concentrations : list[float]
            eis          : dict | None
            cv           : dict | None
        ready            : bool   — True si tous les fichiers obligatoires sont uploadés
        run_clicked      : bool   — True si l'utilisateur a cliqué sur "Lancer l'analyse"
    """
    if mode not in ("eis_only", "cv_only", "both"):
        raise ValueError(f"mode doit être 'eis_only', 'cv_only' ou 'both', reçu '{mode}'")

    use_eis = mode in ("eis_only", "both")
    use_cv  = mode in ("cv_only",  "both")

    # -------------------------------------------------------------------
    # Étape 1 — Déclaration de la session
    # -------------------------------------------------------------------
    st.subheader("Étape 1 — Paramètres de la session")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_conc = st.number_input(
            "Nombre de concentrations",
            min_value=3, max_value=20, value=7, step=1,
            key=_sk(prefix, "n_conc"),
        )
    with col2:
        n_elec = st.number_input(
            "Électrodes par concentration",
            min_value=1, max_value=4, value=2, step=1,
            key=_sk(prefix, "n_elec"),
        )
    with col3:
        n_rep = st.number_input(
            "Réplicats par électrode",
            min_value=1, max_value=6, value=3, step=1,
            key=_sk(prefix, "n_rep"),
        )

    n_conc = int(n_conc)
    n_elec = int(n_elec)
    n_rep  = int(n_rep)

    _init_state(prefix, n_conc, n_elec)
    _resize_conc_list(prefix, n_conc)

    concentrations = _render_concentration_inputs(prefix, n_conc)

    st.markdown("---")

    # -------------------------------------------------------------------
    # Étape 2 — Upload des fichiers de calibration
    # -------------------------------------------------------------------
    st.subheader("Étape 2 — Fichiers de calibration")

    eis_calib: dict | None = None
    cv_calib:  dict | None = None

    if use_eis:
        if mode == "both":
            st.markdown("#### ⚡ Spectres EIS")
        eis_calib = _render_signal_block(
            prefix=prefix,
            signal_type="eis",
            n_conc=n_conc,
            n_elec=n_elec,
            n_rep=n_rep,
            concentrations=concentrations,
        )

    if use_cv:
        if mode == "both":
            st.markdown("#### 📈 Courbes CV")
        cv_calib = _render_signal_block(
            prefix=prefix,
            signal_type="cv",
            n_conc=n_conc,
            n_elec=n_elec,
            n_rep=n_rep,
            concentrations=concentrations,
        )

    # -------------------------------------------------------------------
    # Étape 3 — Données de validation (mode 'both' uniquement)
    # -------------------------------------------------------------------
    validation: dict | None = None

    if mode == "both":
        st.markdown("---")
        st.subheader("Étape 3 — Données de validation (optionnel)")
        add_val = st.checkbox(
            "Ajouter des données de validation (autre jour, électrodes indépendantes)",
            key=_sk(prefix, "add_val"),
        )

        if add_val:
            n_val = int(st.number_input(
                "Nombre de concentrations de validation",
                min_value=1, max_value=20, value=6, step=1,
                key=_sk(prefix, "n_val"),
            ))

            val_prefix = prefix + "_val"
            _init_state(val_prefix, n_val, n_elec)
            _resize_conc_list(val_prefix, n_val)

            st.markdown("**Concentrations de validation**")
            val_concentrations = _render_concentration_inputs(val_prefix, n_val)

            val_eis = _render_signal_block(
                prefix=prefix,
                signal_type="eis",
                n_conc=n_val,
                n_elec=n_elec,
                n_rep=n_rep,
                concentrations=val_concentrations,
                val_prefix=val_prefix + "_eis",
            )
            val_cv = _render_signal_block(
                prefix=prefix,
                signal_type="cv",
                n_conc=n_val,
                n_elec=n_elec,
                n_rep=n_rep,
                concentrations=val_concentrations,
                val_prefix=val_prefix + "_cv",
            )

            validation = {
                "concentrations": val_concentrations,
                "eis": val_eis,
                "cv": val_cv,
            }

    # -------------------------------------------------------------------
    # Indicateur de progression + bouton
    # -------------------------------------------------------------------
    st.markdown("---")

    up_eis, tot_eis = _count_files(eis_calib, n_conc, n_elec, n_rep)
    up_cv,  tot_cv  = _count_files(cv_calib,  n_conc, n_elec, n_rep)
    uploaded = up_eis + up_cv
    total    = tot_eis + tot_cv

    if total > 0:
        progress = uploaded / total
        st.progress(progress, text=f"Fichiers chargés : {uploaded} / {total}")

        if uploaded < total:
            missing = total - uploaded
            st.caption(f"⚠️  {missing} fichier(s) obligatoire(s) manquant(s)")

    ready = (
        _all_required_uploaded(eis_calib, n_conc, n_elec, n_rep)
        and _all_required_uploaded(cv_calib, n_conc, n_elec, n_rep)
    )

    run_clicked = st.button(
        "▶ Lancer l'analyse",
        type="primary",
        disabled=not ready,
        use_container_width=True,
        key=_sk(prefix, "run_btn"),
    )

    # -------------------------------------------------------------------
    # Résultat structuré
    # -------------------------------------------------------------------
    return {
        "concentrations": concentrations,
        "n_electrodes": n_elec,
        "n_replicats": n_rep,
        "calibration": {
            "eis": eis_calib,
            "cv":  cv_calib,
        },
        "validation": validation,
        "ready": ready,
        "run_clicked": run_clicked,
    }
