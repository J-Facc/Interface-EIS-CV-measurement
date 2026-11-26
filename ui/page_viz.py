"""Page 3 — Visualisation des données NETTOYÉES : CV, Nyquist, Bode. Aucun calcul.

N'importe ni fit ni DRT (garde-fou : ``tests/test_ui_pages.py``). Fonctionne aussi après le
chargement d'une archive (les mesures nettoyées y sont).
"""

from __future__ import annotations

import streamlit as st

from core.display import cv_display_groups, eis_display_groups, selection_options, step_display_label
from core.experiment import KIND_CV, KIND_EIS, step_sort_key
from plotting.cv_plots import cv_figure
from plotting.eis_plots import bode_figure, nyquist_replicates_figure
from ui import nav
from ui import state as S


def _selectors(opts: dict, exp, prefix: str):
    """Biomarqueur → canal → électrode → étapes → réplicats. Retourne (canal, électrode, étapes, réplicats)."""
    c1, c2, c3 = st.columns(3)
    bio = c1.selectbox("Biomarqueur", sorted(opts), key=f"{prefix}_bio")
    ch = c2.selectbox("Canal", sorted(opts[bio]), key=f"{prefix}_ch")
    el = c3.selectbox("Électrode", sorted(opts[bio][ch]), key=f"{prefix}_el")
    steps_avail = sorted(opts[bio][ch][el], key=step_sort_key)
    channel = exp.channel(ch)
    c4, c5 = st.columns([3, 2])
    steps = c4.multiselect("Étapes superposées", steps_avail, default=steps_avail, key=f"{prefix}_steps_{ch}_{el}",
                           format_func=lambda s: step_display_label(s, channel.concentration_of(s)))
    reps_avail = sorted({r for s in steps for r in opts[bio][ch][el][s]})
    reps = c5.multiselect("Réplicats", reps_avail, default=reps_avail, key=f"{prefix}_reps_{ch}_{el}",
                          format_func=lambda r: f"r{r}")
    return ch, el, steps, reps


def render() -> None:
    st.title("👁️ Visualisation")
    exp = st.session_state.get(S.EXPERIMENT)
    ms = st.session_state.get(S.MEASUREMENTS)
    if exp is None or not ms:
        st.warning("Aucune donnée nettoyée : configurez l'expérience puis importez et nettoyez les fichiers "
                   "(ou chargez une archive).")
        nav.link("import", "→ Import & nettoyage")
        st.stop()
    if st.session_state.get(S.RESULTS_SOURCE) == "archive":
        st.caption("📦 Données issues d'une archive rechargée.")
    theme = st.session_state[S.THEME]
    tab_cv, tab_eis = st.tabs(["CV", "EIS — Nyquist et Bode"])
    with tab_cv:
        opts = selection_options(ms, exp, KIND_CV)
        if not opts:
            st.info("Aucune mesure CV.")
        else:
            ch, el, steps, reps = _selectors(opts, exp, "viz_cv")
            groups = cv_display_groups(ms, exp, ch, el, steps, reps)
            if groups:
                st.plotly_chart(cv_figure(groups, title=f"CV — canal {ch}, électrode {el}", theme_mode=theme),
                                width="stretch", key="viz_cv_fig")
            else:
                st.info("Sélectionnez au moins une étape et un réplicat.")
    with tab_eis:
        opts = selection_options(ms, exp, KIND_EIS)
        if not opts:
            st.info("Aucune mesure EIS.")
        else:
            ch, el, steps, reps = _selectors(opts, exp, "viz_eis")
            groups = eis_display_groups(ms, exp, ch, el, steps, reps)
            if groups:
                st.plotly_chart(nyquist_replicates_figure(groups, title=f"Nyquist — canal {ch}, électrode {el}",
                                                          theme_mode=theme), width="stretch", key="viz_nyq")
                st.plotly_chart(bode_figure(groups, title=f"Bode — canal {ch}, électrode {el}", theme_mode=theme),
                                width="stretch", key="viz_bode")
            else:
                st.info("Sélectionnez au moins une étape et un réplicat.")
