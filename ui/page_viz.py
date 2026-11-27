"""Page 3 — Visualisation des données NETTOYÉES, PAR ÉLECTRODE : CV, Nyquist, Bode.

Aucun fit, aucune DRT, aucune normalisation (la normalisation par le probe ne sert qu'à
l'extraction des concentrations, page Analyse). Menu « Électrode » :

* **Vue synthétique** (défaut) : courbes MOYENNES de chaque électrode et de chaque étape, avec leur
  zone d'erreur (± écart-type) ; filtre Tout / Bare / Probe / Concentrations ;
* **une électrode** : ses réplicats superposés, puis ses moyennes par étape (± écart-type).

Moyennes : ``core/display.py``. Garde-fou : la page n'importe ni fit ni DRT
(``tests/test_ui_pages.py``). Fonctionne aussi après le chargement d'une archive.
"""

from __future__ import annotations

import streamlit as st

from core import app_state as S
from core.display import (
    VIEW_SYNTHESIS,
    cv_display_groups,
    eis_display_groups,
    electrode_means,
    electrodes_of,
    step_display_label,
    synthesis_view,
)
from core.experiment import KIND_CV, KIND_EIS, step_sort_key
from plotting.cv_plots import cv_figure, cv_means_figure
from plotting.eis_plots import bode_figure, bode_means_figure, nyquist_means_figure, nyquist_replicates_figure
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


def _electrode(ms, exp, ch_id: int, el: int, theme: str, eis_mode: str) -> None:
    avail = {}
    for m in ms:
        if (m.channel_id, m.electrode) == (ch_id, el):
            avail.setdefault(m.step, set()).add(m.replicate)
    channel = exp.channel(ch_id)
    steps_avail = sorted(avail, key=step_sort_key)
    c1, c2 = st.columns([3, 2])
    steps = c1.multiselect("Étapes superposées", steps_avail, default=steps_avail, key=f"viz_steps_{ch_id}_{el}",
                           format_func=lambda s: step_display_label(s, channel.concentration_of(s)))
    reps_avail = sorted({r for s in steps for r in avail[s]})
    reps = c2.multiselect("Réplicats", reps_avail, default=reps_avail, key=f"viz_reps_{ch_id}_{el}",
                          format_func=lambda r: f"r{r}")
    if not steps or not reps:
        st.info("Sélectionnez au moins une étape et un réplicat.")
        return
    name = f"canal {ch_id}, électrode {el}"
    tab_cv, tab_eis = st.tabs(["CV", "EIS"])
    with tab_cv:
        g = cv_display_groups(ms, exp, ch_id, el, steps, reps)
        if not g:
            st.info("Aucune mesure CV pour cette électrode.")
        else:
            st.plotly_chart(cv_figure(g, title=f"CV — {name} : réplicats", theme_mode=theme),
                            width="stretch", key=f"viz_cv_{ch_id}_{el}")
            st.plotly_chart(cv_means_figure(electrode_means(ms, exp, KIND_CV, ch_id, el, steps, reps),
                                            title=f"CV — {name} : moyennes ± écart-type", theme_mode=theme),
                            width="stretch", key=f"viz_cv_{ch_id}_{el}_mean")
    with tab_eis:
        g = eis_display_groups(ms, exp, ch_id, el, steps, reps)
        if not g:
            st.info("Aucune mesure EIS pour cette électrode.")
            return
        fig = (nyquist_replicates_figure(g, title=f"Nyquist — {name} : réplicats", theme_mode=theme)
               if eis_mode == "Nyquist" else bode_figure(g, title=f"Bode — {name} : réplicats", theme_mode=theme))
        st.plotly_chart(fig, width="stretch", key=f"viz_eis_{ch_id}_{el}")
        _eis_means(electrode_means(ms, exp, KIND_EIS, ch_id, el, steps, reps), f"{name} : moyennes ± écart-type",
                   theme, eis_mode, f"viz_eis_{ch_id}_{el}_mean")


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
    for ch, el in electrodes_of(ms):
        if exp.channel(ch):
            labels[f"{ch}:{el}"] = f"Canal {ch} · électrode {el} — {exp.channel(ch).biomarker}"
    c1, c2 = st.columns([3, 2])
    choice = c1.selectbox("Électrode", list(labels), format_func=labels.get, key="viz_electrode")
    eis_mode = c2.radio("Représentation EIS", ["Nyquist", "Bode"], horizontal=True, key="viz_eis_mode")
    if choice == VIEW_SYNTHESIS:
        _synthesis(ms, exp, theme, eis_mode)
    else:
        ch, el = (int(x) for x in choice.split(":"))
        _electrode(ms, exp, ch, el, theme, eis_mode)
