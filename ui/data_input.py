"""Composant Streamlit réutilisable pour l'entrée des données expérimentales.

Utilisé par les pages A (EIS seul), B (CV seul), C (comparatif) et D (inférence).
Aucun import Streamlit en dehors de ce module — toute la logique UI est ici.

Structure retournée :
    probe        : {"eis": {"electrode_1": file, ...}, "cv": {...}}
    calibration  : {"eis": {"electrode_1": [[f,f,...], ...], ...}, "cv": {...}}
    Le probe est mesuré une fois par électrode par session.
    Les réplicats sont uploadés en multi-fichier (accept_multiple_files=True).
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
    """Affiche n_conc lignes mantisse × 10^ exposant. Retourne la liste des concentrations en M."""
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
# Section probe — un seul fichier par électrode, commun à toutes les concentrations
# ---------------------------------------------------------------------------

def _render_probe_section(
    prefix: str,
    signal_type: str,   # 'eis' | 'cv'
    n_elec: int,
    key_prefix: str | None = None,
) -> dict:
    """
    Affiche un file_uploader par électrode pour le probe.
    Retourne {"electrode_1": file|None, "electrode_2": file|None, ...}.
    """
    pfx = key_prefix if key_prefix else prefix
    icon  = "⚡" if signal_type == "eis" else "📈"
    label = "EIS" if signal_type == "eis" else "CV"

    result = {}
    cols = st.columns(n_elec)
    for e in range(1, n_elec + 1):
        with cols[e - 1]:
            f = st.file_uploader(
                f"{icon} {label} — Électrode {e}",
                type=["csv", "txt"],
                accept_multiple_files=False,
                key=_sk(pfx, signal_type, f"probe_e{e}"),
            )
            if f is not None:
                st.caption(f":green[✓ {f.name}]")
            else:
                st.caption("Aucun fichier")
            result[f"electrode_{e}"] = f

    return result


# ---------------------------------------------------------------------------
# Section calibration — réplicats par concentration (multi-fichier)
# ---------------------------------------------------------------------------

def _render_signal_block(
    prefix: str,
    signal_type: str,          # 'eis' | 'cv'
    n_conc: int,
    n_elec: int,
    concentrations: list[float],
    key_prefix: str | None = None,
) -> dict:
    """
    Affiche des uploaders multi-fichiers pour les réplicats de chaque concentration.

    Retourne :
    {
      "electrode_1": [[f, f, ...], [f], ...],  # [ci][replicats]
      "electrode_2": [...],
    }
    """
    pfx  = key_prefix if key_prefix else prefix
    icon = "⚡" if signal_type == "eis" else "📈"
    type_label = "EIS" if signal_type == "eis" else "CV"

    result: dict = {f"electrode_{e}": [None] * n_conc for e in range(1, n_elec + 1)}

    for ci in range(n_conc):
        c_label = f"{concentrations[ci]:.2e} M"
        with st.expander(
            f"{icon} {type_label} — Concentration {ci + 1} : {c_label}",
            expanded=False,
        ):
            elec_cols = st.columns(n_elec)
            for e in range(1, n_elec + 1):
                with elec_cols[e - 1]:
                    files = st.file_uploader(
                        f"Électrode {e} — réplicats",
                        type=["csv", "txt"],
                        accept_multiple_files=True,
                        key=_sk(pfx, signal_type, f"c{ci}_e{e}"),
                    )
                    if files:
                        st.markdown(
                            f":green[{len(files)} fichier(s) chargé(s)]",
                            help="\n".join(f.name for f in files),
                        )
                    else:
                        st.caption("Glisser les réplicats ici")
                    result[f"electrode_{e}"][ci] = files if files else []

    return result


# ---------------------------------------------------------------------------
# Comptage des fichiers pour l'indicateur de progression
# ---------------------------------------------------------------------------

def _count_files(
    calib_dict: dict | None,
    probe_dict: dict | None,
    n_conc: int,
    n_elec: int,
) -> tuple[int, int]:
    """
    Retourne (fichiers_uploadés, total_attendu).
    Total = n_elec probes + n_conc × n_elec créneaux calibration (≥ 1 fichier chacun).
    """
    if calib_dict is None:
        return 0, 0

    # Probes : un par électrode
    total_probes = n_elec
    uploaded_probes = 0
    if probe_dict:
        for e in range(1, n_elec + 1):
            if probe_dict.get(f"electrode_{e}") is not None:
                uploaded_probes += 1

    # Calibration : au moins un fichier par créneau (conc × élec)
    total_calib = n_conc * n_elec
    uploaded_calib = 0
    for e in range(1, n_elec + 1):
        reps_list = calib_dict.get(f"electrode_{e}", [])
        for ci in range(n_conc):
            files = reps_list[ci] if ci < len(reps_list) else []
            if files:  # liste non-vide
                uploaded_calib += 1

    total    = total_probes + total_calib
    uploaded = uploaded_probes + uploaded_calib
    return uploaded, total


def _all_required_uploaded(
    calib_dict: dict | None,
    probe_dict: dict | None,
    n_conc: int,
    n_elec: int,
) -> bool:
    if calib_dict is None:
        return True  # non requis dans ce mode
    up, total = _count_files(calib_dict, probe_dict, n_conc, n_elec)
    return up >= total


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def render_data_input(mode: str, prefix: str = "main") -> dict:
    """
    Composant Streamlit d'entrée des données expérimentales.

    Parameters
    ----------
    mode : str
        'eis_only' | 'cv_only' | 'both'
    prefix : str
        Préfixe unique pour les clés session_state (évite les collisions).

    Returns
    -------
    dict :
        concentrations   : list[float]
        n_electrodes     : int
        n_replicats      : int   (valeur saisie, maintenant indicative)
        probe            : {
            "eis": {"electrode_1": file|None, ...} | None,
            "cv":  {"electrode_1": file|None, ...} | None,
        }
        calibration      : {
            "eis": {"electrode_1": [[files], ...], ...} | None,
            "cv":  {"electrode_1": [[files], ...], ...} | None,
        }
        validation       : dict | None
        ready            : bool
        run_clicked      : bool
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
            "Réplicats par électrode (indicatif)",
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
    # Étape 2a — Références probe (une mesure par électrode)
    # -------------------------------------------------------------------
    st.subheader("Étape 2 — Fichiers de calibration")
    st.markdown("#### 🔬 Références probe (une mesure par électrode, commune à toutes les concentrations)")

    probe_eis: dict | None = None
    probe_cv:  dict | None = None

    if use_eis:
        if mode == "both":
            st.markdown("**⚡ Probe EIS**")
        probe_eis = _render_probe_section(prefix, "eis", n_elec)

    if use_cv:
        if mode == "both":
            st.markdown("**📈 Probe CV**")
        probe_cv = _render_probe_section(prefix, "cv", n_elec)

    st.markdown("#### 📂 Mesures d'hybridation (glisser les réplicats par concentration)")

    # -------------------------------------------------------------------
    # Étape 2b — Calibration par concentration
    # -------------------------------------------------------------------
    eis_calib: dict | None = None
    cv_calib:  dict | None = None

    if use_eis:
        if mode == "both":
            st.markdown("##### ⚡ Spectres EIS")
        eis_calib = _render_signal_block(
            prefix=prefix,
            signal_type="eis",
            n_conc=n_conc,
            n_elec=n_elec,
            concentrations=concentrations,
        )

    if use_cv:
        if mode == "both":
            st.markdown("##### 📈 Courbes CV")
        cv_calib = _render_signal_block(
            prefix=prefix,
            signal_type="cv",
            n_conc=n_conc,
            n_elec=n_elec,
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

            st.markdown("**🔬 Probe validation**")
            val_probe_eis = _render_probe_section(
                prefix, "eis", n_elec, key_prefix=val_prefix + "_probe_eis"
            )
            val_probe_cv = _render_probe_section(
                prefix, "cv", n_elec, key_prefix=val_prefix + "_probe_cv"
            )

            val_eis = _render_signal_block(
                prefix=prefix,
                signal_type="eis",
                n_conc=n_val,
                n_elec=n_elec,
                concentrations=val_concentrations,
                key_prefix=val_prefix + "_eis",
            )
            val_cv = _render_signal_block(
                prefix=prefix,
                signal_type="cv",
                n_conc=n_val,
                n_elec=n_elec,
                concentrations=val_concentrations,
                key_prefix=val_prefix + "_cv",
            )

            validation = {
                "concentrations": val_concentrations,
                "probe": {"eis": val_probe_eis, "cv": val_probe_cv},
                "eis": val_eis,
                "cv":  val_cv,
            }

    # -------------------------------------------------------------------
    # Indicateur de progression + bouton
    # -------------------------------------------------------------------
    st.markdown("---")

    up_eis, tot_eis = _count_files(eis_calib, probe_eis, n_conc, n_elec)
    up_cv,  tot_cv  = _count_files(cv_calib,  probe_cv,  n_conc, n_elec)
    uploaded = up_eis + up_cv
    total    = tot_eis + tot_cv

    if total > 0:
        progress = uploaded / total
        st.progress(progress, text=f"Fichiers chargés : {uploaded} / {total}")
        if uploaded < total:
            missing = total - uploaded
            st.caption(f"⚠️  {missing} fichier(s) obligatoire(s) manquant(s)")

    ready = (
        _all_required_uploaded(eis_calib, probe_eis, n_conc, n_elec)
        and _all_required_uploaded(cv_calib, probe_cv, n_conc, n_elec)
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
        "probe": {
            "eis": probe_eis,
            "cv":  probe_cv,
        },
        "calibration": {
            "eis": eis_calib,
            "cv":  cv_calib,
        },
        "validation": validation,
        "ready": ready,
        "run_clicked": run_clicked,
    }
