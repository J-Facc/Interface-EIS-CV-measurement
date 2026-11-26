"""Page 3 — Visualisation des données NETTOYÉES : CV, Nyquist, Bode. Aucun fit, aucune DRT.

Menu « Canal » :

* **Vue synthétique** (défaut) : toutes les courbes MOYENNES (réplicats des deux électrodes) de
  chaque canal et chaque étape, avec leur zone d'erreur (± écart-type) ;
* **un canal** : trois graphes — électrode 1 (réplicats), électrode 2 (réplicats), et les moyennes
  (électrode 1, électrode 2, deux électrodes) par étape.

Les moyennes viennent de ``core/display.py`` (moyennes point à point pour l'affichage). N'importe ni
fit ni DRT (garde-fou : ``tests/test_ui_pages.py``). Fonctionne aussi après le chargement d'une archive.
"""

from __future__ import annotations

import streamlit as st

from core.display import (
    VIEW_SYNTHESIS,
    channel_mean_view,
    cv_display_groups,
    eis_display_groups,
    step_display_label,
    synthesis_view,
)
from core.experiment import KIND_CV, KIND_EIS, step_sort_key
from plotting.cv_plots import cv_figure, cv_means_figure
from plotting.eis_plots import bode_figure, bode_means_figure, nyquist_means_figure, nyquist_replicates_figure
from ui import nav
from core import app_state as S


def _channel_filters(ms, exp, ch_id: int):
    """Étapes et réplicats affichés pour un canal (tous par défaut)."""
    avail = {}
    for m in ms:
        if m.channel_id == ch_id:
            avail.setdefault(m.step, set()).add(m.replicate)
    channel = exp.channel(ch_id)
    steps_avail = sorted(avail, key=step_sort_key)
    c1, c2 = st.columns([3, 2])
    steps = c1.multiselect("Étapes superposées", steps_avail, default=steps_avail, key=f"viz_steps_{ch_id}",
                           format_func=lambda s: step_display_label(s, channel.concentration_of(s)))
    reps_avail = sorted({r for s in steps for r in avail[s]})
    reps = c2.multiselect("Réplicats", reps_avail, default=reps_avail, key=f"viz_reps_{ch_id}",
                          format_func=lambda r: f"r{r}")
    return steps, reps


def _synthesis(ms, exp, theme: str, eis_mode: str) -> None:
    bios = sorted({c.biomarker for c in exp.channels})
    pick = st.multiselect("Biomarqueurs", bios, default=bios, key="viz_syn_bios")
    st.caption("Moyenne des réplicats des deux électrodes, par canal et par étape ; zone = ± écart-type "
               "(ddof = 1). Couleur = étape, style de trait = canal.")
    tab_cv, tab_eis = st.tabs(["CV", "EIS"])
    with tab_cv:
        items = synthesis_view(ms, exp, KIND_CV, pick)
        if items:
            st.plotly_chart(cv_means_figure(items, title="CV — vue synthétique", theme_mode=theme),
                            width="stretch", key="viz_syn_cv")
        else:
            st.info("Aucune mesure CV.")
    with tab_eis:
        items = synthesis_view(ms, exp, KIND_EIS, pick)
        if not items:
            st.info("Aucune mesure EIS.")
        elif eis_mode == "Nyquist":
            st.plotly_chart(nyquist_means_figure(items, title="Nyquist — vue synthétique", theme_mode=theme),
                            width="stretch", key="viz_syn_nyq")
        else:
            st.plotly_chart(bode_means_figure(items, title="|Z| — vue synthétique", theme_mode=theme),
                            width="stretch", key="viz_syn_bode")


def _channel(ms, exp, ch_id: int, theme: str, eis_mode: str) -> None:
    steps, reps = _channel_filters(ms, exp, ch_id)
    if not steps or not reps:
        st.info("Sélectionnez au moins une étape et un réplicat.")
        return
    tab_cv, tab_eis = st.tabs(["CV", "EIS"])
    with tab_cv:
        if not any(m.kind == KIND_CV and m.channel_id == ch_id for m in ms):
            st.info("Aucune mesure CV pour ce canal.")
        else:
            for el in (1, 2):
                g = cv_display_groups(ms, exp, ch_id, el, steps, reps)
                if g:
                    st.plotly_chart(cv_figure(g, title=f"CV — canal {ch_id}, électrode {el}", theme_mode=theme),
                                    width="stretch", key=f"viz_cv_{ch_id}_{el}")
            st.plotly_chart(cv_means_figure(channel_mean_view(ms, exp, KIND_CV, ch_id, steps, reps),
                                            title=f"CV — canal {ch_id} : moyennes (- - E1 · ··· E2 · — E1+E2)",
                                            theme_mode=theme), width="stretch", key=f"viz_cv_{ch_id}_mean")
    with tab_eis:
        if not any(m.kind == KIND_EIS and m.channel_id == ch_id for m in ms):
            st.info("Aucune mesure EIS pour ce canal.")
            return
        for el in (1, 2):
            g = eis_display_groups(ms, exp, ch_id, el, steps, reps)
            if not g:
                continue
            fig = (nyquist_replicates_figure(g, title=f"Nyquist — canal {ch_id}, électrode {el}", theme_mode=theme)
                   if eis_mode == "Nyquist" else
                   bode_figure(g, title=f"Bode — canal {ch_id}, électrode {el}", theme_mode=theme))
            st.plotly_chart(fig, width="stretch", key=f"viz_eis_{ch_id}_{el}")
        items = channel_mean_view(ms, exp, KIND_EIS, ch_id, steps, reps)
        title = f"canal {ch_id} : moyennes (- - E1 · ··· E2 · — E1+E2)"
        fig = (nyquist_means_figure(items, title=f"Nyquist — {title}", theme_mode=theme) if eis_mode == "Nyquist"
               else bode_means_figure(items, title=f"|Z| — {title}", theme_mode=theme))
        st.plotly_chart(fig, width="stretch", key=f"viz_eis_{ch_id}_mean")


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
    channels = sorted({m.channel_id for m in ms})
    labels = {VIEW_SYNTHESIS: "Vue synthétique (toutes les moyennes)"}
    labels.update({c: f"Canal {c} — {exp.channel(c).biomarker}" for c in channels if exp.channel(c)})
    c1, c2 = st.columns([3, 2])
    choice = c1.selectbox("Canal", list(labels), format_func=labels.get, key="viz_channel")
    eis_mode = c2.radio("Représentation EIS", ["Nyquist", "Bode"], horizontal=True, key="viz_eis_mode")
    if choice == VIEW_SYNTHESIS:
        _synthesis(ms, exp, theme, eis_mode)
    else:
        _channel(ms, exp, choice, theme, eis_mode)

