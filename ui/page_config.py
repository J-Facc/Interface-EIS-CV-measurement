"""Page 1 — Configuration de l'expérience : biomarqueurs, canaux, concentrations.

Aucun calcul. Produit ``state.EXPERIMENT`` (``core.experiment.Experiment``). Les noms de
biomarqueurs saisis sont enregistrés dans ``user_data/biomarkers.json`` (hors du dossier de
l'application) et proposés dans tous les menus déroulants.
"""

from __future__ import annotations

import streamlit as st

from core.experiment import (
    MODE_LABELS,
    Channel,
    Experiment,
    concentration_from_parts,
    concentration_parts,
    default_biomarker_for_channel,
    experiment_summary_rows,
    validate_experiment,
)
from core.user_data import add_biomarkers, load_biomarkers
from ui import nav
from ui.archive_loader import render_archive_loader
from ui import state as S

ADD_NEW = "+ Ajouter un nouveau biomarqueur…"
_P = "cfg_"


def _k(*parts) -> str:
    return _P + "_".join(str(p) for p in parts)


def _val(key: str, default) -> dict:
    """``value=`` seulement si le widget n'a pas encore d'état (sinon Streamlit avertit)."""
    return {} if key in st.session_state else {"value": default}


def _same_config(a: Experiment, b: Experiment) -> bool:
    strip = {"created_at": None, "app_version": None}
    return a is not None and b is not None and (a.to_dict() | strip) == (b.to_dict() | strip)


def _seed_from(exp: Experiment) -> None:
    """Pré-remplit les widgets depuis une expérience (archive chargée, retour sur la page)."""
    for key in [k for k in st.session_state if str(k).startswith(_P) and str(k) != _k("seeded")]:
        del st.session_state[key]
    st.session_state[_k("name")] = exp.name
    st.session_state[_k("n_bio")] = max(1, len(exp.biomarkers))
    for i, b in enumerate(exp.biomarkers):
        st.session_state[_k("bio_sel", i)] = b
    st.session_state[_k("n_ch")] = max(1, len(exp.channels))
    for k, ch in enumerate(exp.channels):
        st.session_state[_k("ch_bio_sel", k)] = ch.biomarker
        st.session_state[_k("ch_nc", k)] = max(1, len(ch.concentrations))
        for j, c in enumerate(ch.concentrations):
            m, e = concentration_parts(c)
            st.session_state[_k("m", k, j)] = float(m)
            st.session_state[_k("e", k, j)] = int(e)


def _commit_new_biomarker(i: int) -> None:
    """Rappel du champ « Nom du biomarqueur » : persiste le nom et le sélectionne aussitôt."""
    name = (st.session_state.get(_k("bio_new", i)) or "").strip()
    if name:
        add_biomarkers([name])
        st.session_state[_k("bio_sel", i)] = name
        st.session_state[_k("bio_new", i)] = ""


def _biomarker_inputs(saved: list) -> list:
    st.subheader("1. Biomarqueurs mesurés")
    n_bio = st.number_input("Nombre de biomarqueurs", min_value=1, max_value=20, step=1,
                            key=_k("n_bio"), **_val(_k("n_bio"), 1))
    names = []
    cols = st.columns(min(int(n_bio), 3))
    for i in range(int(n_bio)):
        with cols[i % len(cols)]:
            sel_key = _k("bio_sel", i)
            current = st.session_state.get(sel_key)
            options = list(saved)
            if current and current != ADD_NEW and current not in options:
                options.append(current)
            options.append(ADD_NEW)
            if current not in options:    # premier affichage : i-ème nom enregistré, sinon « + Ajouter »
                st.session_state[sel_key] = saved[i] if i < len(saved) else ADD_NEW
            choice = st.selectbox(f"Biomarqueur {i + 1}", options, key=sel_key)
            if choice == ADD_NEW:
                # Persisté dès la saisie (Entrée) : le nom alimente aussitôt tous les menus.
                st.text_input(f"Nom du biomarqueur {i + 1}", key=_k("bio_new", i), placeholder="ex. miR-21",
                              on_change=_commit_new_biomarker, args=(i,))
                names.append("")
            else:
                names.append(choice)
    return names


def _concentration_inputs(k: int) -> list:
    n_c = st.number_input("Nombre de concentrations mesurées", min_value=1, max_value=20, step=1,
                          key=_k("ch_nc", k), **_val(_k("ch_nc", k), 1))
    st.caption("Valeurs en mol/L, saisies mantisse × 10^exposant, dans l'ordre croissant d'injection.")
    concs = []
    for j in range(int(n_c)):
        c1, c2, c3 = st.columns([2, 2, 3])
        m = c1.number_input(f"Mantisse c{j + 1}", min_value=0.0, max_value=9.99, step=0.1, format="%.2f",
                            key=_k("m", k, j), **_val(_k("m", k, j), 1.0))
        e = c2.number_input(f"Exposant c{j + 1}", min_value=-18, max_value=0, step=1,
                            key=_k("e", k, j), **_val(_k("e", k, j), -12 + j))
        c = concentration_from_parts(m, e)
        c3.markdown(f"<div style='padding-top:2rem'>= <b>{c:.2e}</b> mol/L</div>", unsafe_allow_html=True)
        concs.append(c)
    return concs


def _channel_inputs(biomarkers: list) -> list:
    st.subheader("2. Canaux")
    st.caption("Un canal = deux électrodes (électrode 1 et électrode 2), mesurées chacune séparément.")
    n_ch = st.number_input("Nombre de canaux", min_value=1, max_value=16, step=1,
                           key=_k("n_ch"), **_val(_k("n_ch"), 1))
    valid_bio = [b for b in biomarkers if b]
    channels = []
    for k in range(int(n_ch)):
        with st.container(border=True):
            st.markdown(f"**Canal {k + 1}**")
            sel_key = _k("ch_bio_sel", k)
            if valid_bio and st.session_state.get(sel_key) not in valid_bio:
                st.session_state[sel_key] = default_biomarker_for_channel(k, valid_bio, int(n_ch))
            if valid_bio:
                bio = st.selectbox("Biomarqueur", valid_bio, key=sel_key)
            else:
                st.info("Déclarez d'abord les biomarqueurs.")
                bio = ""
            concs = _concentration_inputs(k)
            ch = Channel(id=k + 1, biomarker=bio, concentrations=concs)
            st.caption(f"Mode déduit : **{MODE_LABELS[ch.mode]}**")
            channels.append(ch)
    return channels


def _render_archive_loader() -> None:
    with st.expander("📦 Charger une archive (.zip) — affichage sans recalcul", expanded=False):
        render_archive_loader()


def render() -> None:
    st.title("⚙️ Configuration de l'expérience")
    _render_archive_loader()

    exp_in_state = st.session_state.get(S.EXPERIMENT)
    if exp_in_state is not None and st.session_state.get(_k("seeded")) != id(exp_in_state):
        _seed_from(exp_in_state)
        st.session_state[_k("seeded")] = id(exp_in_state)

    st.text_input("Nom de l'expérience (facultatif)", key=_k("name"))
    saved = load_biomarkers()
    biomarkers = _biomarker_inputs(saved)
    channels = _channel_inputs(biomarkers)

    exp = Experiment(biomarkers=biomarkers, channels=channels,
                     name=(st.session_state.get(_k("name")) or "").strip())
    st.subheader("3. Récapitulatif")
    errors = validate_experiment(exp)
    if errors:
        for e in errors:
            st.error(e)
        return
    st.dataframe(experiment_summary_rows(exp), hide_index=True, width="stretch")

    if _same_config(exp_in_state, exp):
        st.success("✅ Configuration validée.")
        nav.link("import", "→ Import & nettoyage")
        return
    if exp_in_state is not None:
        st.warning("La configuration affichée diffère de celle validée : la valider invalide le "
                   "nettoyage et les résultats d'analyse.")
    if st.button("✅ Valider la configuration", type="primary"):
        add_biomarkers(biomarkers)
        S.new_experiment(st.session_state, exp)
        st.session_state[_k("seeded")] = id(exp)
        st.rerun()
