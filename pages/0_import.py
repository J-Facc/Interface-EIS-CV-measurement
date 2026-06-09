"""Page 0 — Import des données expérimentales.

Point d'entrée unique pour toutes les données de l'application.
Les autres pages lisent exclusivement depuis st.session_state['experiment'].
"""

from __future__ import annotations

import io
from datetime import datetime

import streamlit as st

from core.experiment_io import load_experiment, save_experiment


# ─────────────────────────────────────────────
# Helpers session_state
# ─────────────────────────────────────────────

def _sk(*parts) -> str:
    return "__imp_" + "_".join(str(p) for p in parts)


def _init_state() -> None:
    defaults = {
        _sk("mode"):        "EIS + CV",
        _sk("n_conc"):      7,
        _sk("n_elec"):      2,
        _sk("n_rep"):       3,
        _sk("conc_params"): None,
        _sk("exp_name"):    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state[_sk("conc_params")] is None:
        n = st.session_state[_sk("n_conc")]
        st.session_state[_sk("conc_params")] = [
            {"mantisse": 1.0, "exposant": -8 - i} for i in range(n)
        ]


def _resize_conc_params(new_n: int) -> None:
    key = _sk("conc_params")
    current: list = st.session_state.get(key) or []
    if len(current) < new_n:
        last_exp = current[-1]["exposant"] if current else -8
        while len(current) < new_n:
            current.append({"mantisse": 1.0, "exposant": last_exp - 1})
            last_exp -= 1
    elif len(current) > new_n:
        current = current[:new_n]
    st.session_state[key] = current


def _conc_value(cp: dict) -> float:
    return float(cp["mantisse"]) * (10 ** int(cp["exposant"]))


# ─────────────────────────────────────────────
# Sections de la page
# ─────────────────────────────────────────────

def _section_load_existing() -> None:
    """Charger une expérience existante depuis un ZIP."""
    st.subheader("📂 Charger une expérience existante (optionnel)")
    zip_file = st.file_uploader(
        "Charger un dossier expérience (ZIP)",
        type=["zip"],
        accept_multiple_files=False,
        key=_sk("zip_upload"),
    )
    if zip_file is not None:
        try:
            exp = load_experiment(zip_file.read())
            # Pré-remplir le session_state depuis le ZIP chargé
            st.session_state[_sk("mode")] = _mode_from_key(exp["mode"])
            st.session_state[_sk("n_conc")] = len(exp.get("concentrations") or [])
            st.session_state[_sk("n_elec")] = exp.get("n_electrodes", 2)
            st.session_state[_sk("n_rep")]  = exp.get("n_replicats", 3)
            st.session_state[_sk("exp_name")] = exp.get("name", "")
            # Reconstituer conc_params
            concs = exp.get("concentrations") or []
            params = []
            for c in concs:
                import math
                if c > 0:
                    exp10 = int(math.floor(math.log10(c)))
                    mant  = round(c / (10 ** exp10), 2)
                else:
                    exp10, mant = -9, 1.0
                params.append({"mantisse": mant, "exposant": exp10})
            st.session_state[_sk("conc_params")] = params
            # Stocker l'expérience chargée pour pré-remplir les uploaders
            st.session_state[_sk("loaded_exp")] = exp
            # Restaurer les exclusions et points supprimés si présents dans le ZIP
            if exp.get("exclusions"):
                st.session_state["exclusions"] = exp["exclusions"]
            if exp.get("deleted_points"):
                for lbl, pts in exp["deleted_points"].items():
                    st.session_state[f"deleted_points_{lbl}"] = pts
            st.success(
                f"✅ Expérience chargée : **{exp.get('name', '—')}** ({exp.get('date', '—')})"
            )
        except Exception as exc:
            st.error(f"❌ Impossible de charger le ZIP : {exc}")


def _section_session_declaration() -> tuple[str, list[float], int, int]:
    """Mode, concentrations, électrodes, réplicats. Retourne (mode_key, concs, n_elec, n_rep)."""
    st.subheader("📋 Déclaration de la session")

    mode_label = st.radio(
        "Mode de mesure",
        ["EIS seule", "CV seule", "EIS + CV"],
        index=["EIS seule", "CV seule", "EIS + CV"].index(
            st.session_state[_sk("mode")]
        ),
        horizontal=True,
        key=_sk("mode_radio"),
    )
    st.session_state[_sk("mode")] = mode_label

    col1, col2, col3 = st.columns(3)
    with col1:
        n_conc = st.number_input(
            "Nombre de concentrations",
            min_value=3, max_value=20,
            value=st.session_state[_sk("n_conc")],
            step=1, key=_sk("n_conc_input"),
        )
        if n_conc != st.session_state[_sk("n_conc")]:
            st.session_state[_sk("n_conc")] = n_conc
            _resize_conc_params(n_conc)
    with col2:
        n_elec = st.number_input(
            "Nombre d'électrodes",
            min_value=1, max_value=4,
            value=st.session_state[_sk("n_elec")],
            step=1, key=_sk("n_elec_input"),
        )
        st.session_state[_sk("n_elec")] = n_elec
    with col3:
        n_rep = st.number_input(
            "Réplicats par électrode",
            min_value=1, max_value=6,
            value=st.session_state[_sk("n_rep")],
            step=1, key=_sk("n_rep_input"),
        )
        st.session_state[_sk("n_rep")] = n_rep

    # Saisie des concentrations
    st.markdown("**Concentrations (mantisse × 10^ exposant)**")
    _resize_conc_params(n_conc)
    conc_params = st.session_state[_sk("conc_params")]
    concentrations: list[float] = []
    for i, cp in enumerate(conc_params):
        c1, c2, c3, c4 = st.columns([2, 0.5, 1.5, 0.7])
        with c1:
            cp["mantisse"] = st.number_input(
                f"M{i+1}", value=float(cp["mantisse"]),
                min_value=0.1, max_value=9.9, step=0.1, format="%.1f",
                key=_sk(f"mant_{i}"), label_visibility="collapsed",
            )
        with c2:
            st.markdown("<div style='padding-top:30px'>×10^</div>", unsafe_allow_html=True)
        with c3:
            cp["exposant"] = st.number_input(
                f"E{i+1}", value=int(cp["exposant"]),
                min_value=-20, max_value=0, step=1,
                key=_sk(f"exp_{i}"), label_visibility="collapsed",
            )
        with c4:
            st.markdown("<div style='padding-top:30px'>M</div>", unsafe_allow_html=True)
        concentrations.append(_conc_value(cp))

    return mode_label, concentrations, int(n_elec), int(n_rep)


def _section_probe_uploads(n_elec: int, mode_key: str) -> dict:
    """Section 'Références probe'. Retourne dict {electrode_k: [BytesIO, ...]}."""
    st.subheader("🔬 Références probe")
    st.caption("Mesures de référence avant hybridation — une fois pour toute la session.")

    sig_types = _sig_types_for_mode(mode_key)
    probe: dict = {}

    if len(sig_types) == 1:
        sig = sig_types[0]
        probe[sig] = _upload_probe_elec(sig, n_elec)
    else:
        cols = st.columns(len(sig_types))
        for col, sig in zip(cols, sig_types):
            with col:
                st.markdown(f"**{sig.upper()}**")
                probe[sig] = _upload_probe_elec(sig, n_elec)

    return probe


def _upload_probe_elec(sig: str, n_elec: int) -> dict:
    result = {}
    for e in range(1, n_elec + 1):
        files = st.file_uploader(
            f"Électrode {e}",
            accept_multiple_files=True,
            key=_sk(f"probe_{sig}_e{e}"),
        )
        bios = _files_to_bytesio(files)
        result[f"electrode_{e}"] = bios
        n = len(bios)
        if n > 0:
            st.caption(f"✔ {n} réplicat(s) probe chargé(s)")
        else:
            st.caption("Aucun fichier")
    return result


def _section_calibration_uploads(
    concentrations: list[float],
    n_elec: int,
    n_rep: int,
    mode_key: str,
) -> dict:
    """
    Uploads de calibration par concentration, électrode et réplicat.
    Retourne calibration dict.
    """
    st.subheader("📁 Fichiers de calibration")

    sig_types = _sig_types_for_mode(mode_key)
    calibration: dict = {s: {f"electrode_{e}": [] for e in range(1, n_elec + 1)}
                         for s in sig_types}

    for ci, conc in enumerate(concentrations):
        label = f"{conc:.2e} M"
        with st.expander(f"Concentration {ci+1} — {label}", expanded=False):
            if len(sig_types) == 1:
                sig = sig_types[0]
                _upload_conc_block(calibration[sig], sig, ci, n_elec)
            else:
                cols = st.columns(len(sig_types))
                for col, sig in zip(cols, sig_types):
                    with col:
                        st.markdown(f"**{sig.upper()}**")
                        _upload_conc_block(calibration[sig], sig, ci, n_elec)

    return calibration


def _upload_conc_block(elec_dict: dict, sig: str, ci: int, n_elec: int) -> None:
    """Remplit elec_dict[electrode_k][ci] avec la liste des BytesIO uploadés."""
    for e in range(1, n_elec + 1):
        elec_key = f"electrode_{e}"
        files = st.file_uploader(
            f"Élec. {e}",
            accept_multiple_files=True,
            key=_sk(f"cal_{sig}_e{e}_c{ci}"),
        )
        bios = _files_to_bytesio(files)
        # S'assurer que la liste est assez longue
        while len(elec_dict[elec_key]) <= ci:
            elec_dict[elec_key].append([])
        elec_dict[elec_key][ci] = bios
        n = len(bios)
        st.caption(f"{'✔' if n else '○'} {n} fichier(s)" if True else "")


def _section_validation_uploads(
    concentrations: list[float],
    n_elec: int,
    mode_key: str,
) -> dict | None:
    """Uploads optionnels de validation. Retourne dict ou None."""
    add_val = st.checkbox(
        "Ajouter des données de validation (autre jour)",
        key=_sk("add_validation"),
    )
    if not add_val:
        return None

    st.subheader("🧪 Données de validation")

    sig_types = _sig_types_for_mode(mode_key)
    validation: dict = {s: {f"electrode_{e}": [] for e in range(1, n_elec + 1)}
                        for s in sig_types}

    for ci, conc in enumerate(concentrations):
        label = f"{conc:.2e} M"
        with st.expander(f"Validation — {label}", expanded=False):
            if len(sig_types) == 1:
                sig = sig_types[0]
                _upload_conc_block(validation[sig], sig, ci, n_elec)
            else:
                cols = st.columns(len(sig_types))
                for col, sig in zip(cols, sig_types):
                    with col:
                        st.markdown(f"**{sig.upper()}**")
                        _upload_conc_block(validation[sig], sig, ci, n_elec)

    return validation


def _section_progress(probe: dict, calibration: dict, mode_key: str, n_elec: int,
                      n_conc: int) -> tuple[int, int]:
    """Indicateur de progression — retourne (n_loaded, n_total)."""
    sig_types = _sig_types_for_mode(mode_key)

    n_loaded = 0
    n_total  = 0

    # Probe
    for sig in sig_types:
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            n_total  += 1
            if probe.get(sig, {}).get(key):
                n_loaded += 1

    # Calibration
    for sig in sig_types:
        for e in range(1, n_elec + 1):
            key = f"electrode_{e}"
            concs_list = calibration.get(sig, {}).get(key, [])
            for ci in range(n_conc):
                n_total += 1
                if ci < len(concs_list) and concs_list[ci]:
                    n_loaded += 1

    return n_loaded, n_total


def _section_save(experiment: dict) -> None:
    st.subheader("💾 Sauvegarder l'expérience")
    name = st.text_input(
        "Nom de l'expérience",
        value=st.session_state[_sk("exp_name")],
        key=_sk("exp_name_input"),
    )
    st.session_state[_sk("exp_name")] = name

    if st.button("💾 Sauvegarder (ZIP)", key=_sk("save_btn")):
        exp_to_save = dict(experiment)
        exp_to_save["name"] = name
        try:
            zip_bytes = save_experiment(exp_to_save)
            safe_name = (name or "experience").replace(" ", "_")
            st.download_button(
                label="⬇️ Télécharger le ZIP",
                data=zip_bytes,
                file_name=f"{safe_name}.zip",
                mime="application/zip",
                key=_sk("dl_zip"),
            )
        except Exception as exc:
            st.error(f"Erreur lors de la sauvegarde : {exc}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _files_to_bytesio(files) -> list[io.BytesIO]:
    if not files:
        return []
    result = []
    for f in files:
        data = f.read()
        f.seek(0)
        bio = io.BytesIO(data)
        bio.name = getattr(f, "name", "file.bin")
        result.append(bio)
    return result


def _sig_types_for_mode(mode_label: str) -> list[str]:
    if mode_label == "EIS seule":
        return ["eis"]
    if mode_label == "CV seule":
        return ["cv"]
    return ["eis", "cv"]


def _mode_from_key(mode_key: str) -> str:
    return {"eis_only": "EIS seule", "cv_only": "CV seule", "both": "EIS + CV"}.get(
        mode_key, "EIS + CV"
    )


def _mode_to_key(mode_label: str) -> str:
    return {"EIS seule": "eis_only", "CV seule": "cv_only", "EIS + CV": "both"}.get(
        mode_label, "both"
    )


def _has_probe(probe: dict) -> bool:
    for sig_dict in probe.values():
        for rep_list in sig_dict.values():
            if rep_list:
                return True
    return False


def _has_calibration(calibration: dict) -> bool:
    for sig_dict in calibration.values():
        if not sig_dict:
            continue
        for concs in sig_dict.values():
            for rep_list in concs:
                if rep_list:
                    return True
    return False


# ─────────────────────────────────────────────
# Page principale
# ─────────────────────────────────────────────

def main() -> None:
    st.title("📂 Import des données")
    st.caption("Chargez vos fichiers, déclarez la session, puis validez pour passer à l'analyse.")

    _init_state()

    # ── 1. Charger expérience existante ─────────────────────────────────
    _section_load_existing()
    st.divider()

    # ── 2. Déclaration de la session ────────────────────────────────────
    mode_label, concentrations, n_elec, n_rep = _section_session_declaration()
    mode_key = _mode_to_key(mode_label)
    n_conc = len(concentrations)
    st.divider()

    # ── 3. Probe ────────────────────────────────────────────────────────
    probe = _section_probe_uploads(n_elec, mode_label)
    st.divider()

    # ── 4. Calibration ──────────────────────────────────────────────────
    calibration = _section_calibration_uploads(concentrations, n_elec, n_rep, mode_label)
    st.divider()

    # ── 5. Validation (optionnel) ────────────────────────────────────────
    validation = _section_validation_uploads(concentrations, n_elec, mode_label)
    st.divider()

    # ── 6. Progression ──────────────────────────────────────────────────
    n_loaded, n_total = _section_progress(probe, calibration, mode_label, n_elec, n_conc)
    st.markdown(f"**Progression :** {n_loaded} / {n_total} fichiers chargés")
    st.progress(n_loaded / max(n_total, 1))

    # ── 7. Bouton Valider ───────────────────────────────────────────────
    can_validate = _has_probe(probe) and _has_calibration(calibration)
    if st.button(
        "✅ Valider l'import et passer au prétraitement",
        disabled=not can_validate,
        key=_sk("validate_btn"),
        type="primary",
    ):
        experiment = {
            "name":         st.session_state[_sk("exp_name")],
            "date":         datetime.now().strftime("%Y-%m-%d"),
            "mode":         mode_key,
            "concentrations": concentrations,
            "n_electrodes": n_elec,
            "n_replicats":  n_rep,
            "probe":        probe,
            "calibration":  calibration,
            "validation":   validation,
        }
        st.session_state["experiment"] = experiment
        st.session_state["import_validated"] = True
        # Réinitialiser les données nettoyées si elles existent déjà
        st.session_state.pop("experiment_clean", None)
        st.session_state.pop("exclusions", None)
        st.success("✅ Import validé ! Rendez-vous dans l'onglet **Prétraitement**.")
        st.balloons()

    if not can_validate:
        st.caption("⚠️ Chargez au minimum les fichiers probe et une concentration pour valider.")

    st.divider()

    # ── 8. Sauvegarder ───────────────────────────────────────────────────
    current_exp = st.session_state.get("experiment") or {
        "name": st.session_state[_sk("exp_name")],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mode": mode_key,
        "concentrations": concentrations,
        "n_electrodes": n_elec,
        "n_replicats": n_rep,
        "probe": probe,
        "calibration": calibration,
        "validation": validation,
    }
    _section_save(current_exp)


if __name__ == "__main__":
    main()
