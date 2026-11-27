"""Page 3 — Visualisation des données NETTOYÉES, PAR CANAL : CV, Nyquist, Bode.

Aucun fit, aucune DRT, aucune normalisation (la normalisation par le probe ne sert qu'à
l'extraction des concentrations, page Analyse). Menu « Vue » :

* **Vue synthétique** (défaut) : courbes MOYENNES de chaque électrode et de chaque étape, avec leur
  zone d'erreur (± écart-type) ; filtre Tout / Bare / Probe / Concentrations ;
* **un canal** : moyennes des réplicats par étape (± écart-type), en trois graphes — électrode 1,
  électrode 2, puis les deux superposées (style de trait = électrode).

Moyennes : ``core/display.py``. Garde-fou : la page n'importe ni fit ni DRT
(``tests/test_ui_pages.py``). Fonctionne aussi après le chargement d'une archive.
"""

from __future__ import annotations

import streamlit as st

from core import app_state as S
from core.display import (
    VIEW_SYNTHESIS,
    channel_electrode_views,
    channels_of,
    step_display_label,
    synthesis_view,
)
from core.experiment import KIND_CV, KIND_EIS, step_sort_key
from plotting.cv_plots import cv_means_figure
from plotting.eis_plots import bode_means_figure, nyquist_means_figure
from ui import nav

_SHOW = {"all": "Tout", "bare": "Bare", "probe": "Probe", "conc": "Concentrations"}


def _eis_means(items, title, theme, mode, key):
    fig = (nyquist_means_figure(items, title=f"Nyquist — {title}", theme_mode=theme) if mode == "Nyquist"
           else bode_means_figure(items, title=f"|Z| — {title}", theme_mode=theme))
    st.plotly_chart(fig, width="stretch", key=key)


def _synthesis(ms, exp, theme: str, eis_mode: str) -> None:
    bios = sorted({c.biomarker for c in exp.channels})
    c1, c2 = st.columns([3, 3])
    pick = c1.multiselect("Biomarqueurs", bios, default=bios, key="viz_syn_bios")
    show = c2.segmented_control("Afficher", list(_SHOW), format_func=_SHOW.get, default="all",
                                key="viz_syn_show") or "all"
    st.caption("Moyenne des réplicats de chaque électrode, par étape ; zone = ± écart-type (ddof = 1). "
               "Couleur = étape, style de trait = électrode. Données brutes, non normalisées.")
    tab_cv, tab_eis = st.tabs(["CV", "EIS"])
    with tab_cv:
        items = synthesis_view(ms, exp, KIND_CV, pick, show)
        if items:
            st.plotly_chart(cv_means_figure(items, title="CV — vue synthétique", theme_mode=theme),
                            width="stretch", key="viz_syn_cv")
        else:
            st.info("Aucune courbe CV pour cette sélection.")
    with tab_eis:
        items = synthesis_view(ms, exp, KIND_EIS, pick, show)
        if items:
            _eis_means(items, "vue synthétique", theme, eis_mode, "viz_syn_eis")
        else:
            st.info("Aucun spectre EIS pour cette sélection.")


def _channel(ms, exp, ch_id: int, theme: str, eis_mode: str) -> None:
    """Un canal : moyennes des réplicats, 3 graphes — électrode 1, électrode 2, les deux superposées."""
    channel = exp.channel(ch_id)
    steps_avail = sorted({m.step for m in ms if m.channel_id == ch_id}, key=step_sort_key)
    steps = st.multiselect("Étapes", steps_avail, default=steps_avail, key=f"viz_steps_{ch_id}",
                           format_func=lambda s: step_display_label(s, channel.concentration_of(s)))
    if not steps:
        st.info("Sélectionnez au moins une étape.")
        return
    st.caption("Moyenne des réplicats par étape ; zone / barres = ± écart-type (ddof = 1). Données brutes.")
    tab_cv, tab_eis = st.tabs(["CV", "EIS"])
    for tab, kind in ((tab_cv, KIND_CV), (tab_eis, KIND_EIS)):
        with tab:
            views = channel_electrode_views(ms, exp, kind, ch_id, steps)
            if not views:
                st.info(f"Aucune mesure {kind} pour ce canal.")
                continue
            for key, items in views.items():
                name = "les deux électrodes" if key == "both" else f"électrode {key}"
                title = f"{kind} — canal {ch_id}, {name}"
                if kind == KIND_CV:
                    st.plotly_chart(cv_means_figure(items, title=title, theme_mode=theme), width="stretch",
                                    key=f"viz_cv_{ch_id}_{key}")
                else:
                    _eis_means(items, f"canal {ch_id}, {name}", theme, eis_mode, f"viz_eis_{ch_id}_{key}")


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
    labels = {VIEW_SYNTHESIS: "Vue synthétique (toutes les moyennes)"}
    for ch in channels_of(ms):
        if exp.channel(ch):
            labels[str(ch)] = f"Canal {ch} — {exp.channel(ch).biomarker}"
    c1, c2 = st.columns([3, 2])
    choice = c1.selectbox("Vue", list(labels), format_func=labels.get, key="viz_electrode")
    eis_mode = c2.radio("Représentation EIS", ["Nyquist", "Bode"], horizontal=True, key="viz_eis_mode")
    if choice == VIEW_SYNTHESIS:
        _synthesis(ms, exp, theme, eis_mode)
    else:
        _channel(ms, exp, int(choice), theme, eis_mode)
