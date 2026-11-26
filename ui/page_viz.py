"""Page 3 — Visualisation des données NETTOYÉES : CV, Nyquist, Bode. Aucun fit, aucune DRT.

Menu « Canal » (les graphes de MOYENNES sont normalisés, par défaut, par le probe moyen de la
même électrode — voir ``core/display.py``) :

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


_SHOW = {"all": "Tout", "bare": "Bare", "probe": "Probe", "conc": "Concentrations"}
_NORM_HELP = ("Chaque réplicat est normalisé par le probe MOYEN de sa propre électrode, puis moyenné. "
              "EIS : |Re_probe − Re| / |Re_probe| et |Im_probe − Im| / |Im_probe| point à point "
              "(Bode : écart relatif de |Z|). CV : I / I_pic,probe (pic anodique du probe, fenêtre "
              "cv.peak_window_V).")


def _cv_window() -> tuple:
    from core.config import config_to_dict, load_config
    return tuple(config_to_dict(load_config())["cv"]["peak_window_V"])


def _skipped_note(items: list) -> None:
    sk = sorted({(i["channel_id"], e) for i in items for e in i.get("skipped") or []})
    if sk:
        st.caption("⚠️ Sans probe, non normalisables (écartées) : "
                   + ", ".join(f"canal {c} électrode {e}" for c, e in sk) + ".")


def _synthesis(ms, exp, theme: str, eis_mode: str) -> None:
    bios = sorted({c.biomarker for c in exp.channels})
    c1, c2, c3 = st.columns([3, 3, 2])
    pick = c1.multiselect("Biomarqueurs", bios, default=bios, key="viz_syn_bios")
    show = c2.segmented_control("Afficher", list(_SHOW), format_func=_SHOW.get, default="all",
                                key="viz_syn_show") or "all"
    norm = c3.toggle("Normaliser par le probe", value=True, key="viz_syn_norm", help=_NORM_HELP)
    st.caption("Moyenne des réplicats des deux électrodes, par canal et par étape ; zone = ± écart-type "
               "(ddof = 1). Couleur = étape, style de trait = canal."
               + (" Courbes normalisées par le probe de chaque électrode." if norm else ""))
    win = _cv_window()
    tab_cv, tab_eis = st.tabs(["CV", "EIS"])
    with tab_cv:
        items = synthesis_view(ms, exp, KIND_CV, pick, norm, show, win)
        if items:
            st.plotly_chart(cv_means_figure(items, title="CV — vue synthétique", theme_mode=theme, normalized=norm),
                            width="stretch", key="viz_syn_cv")
            _skipped_note(items)
        else:
            st.info("Aucune courbe CV pour cette sélection.")
    with tab_eis:
        items = synthesis_view(ms, exp, KIND_EIS, pick, norm, show, win)
        if not items:
            st.info("Aucun spectre EIS pour cette sélection.")
            return
        if eis_mode == "Nyquist":
            st.plotly_chart(nyquist_means_figure(items, title="Nyquist — vue synthétique", theme_mode=theme,
                                                 normalized=norm), width="stretch", key="viz_syn_nyq")
        else:
            st.plotly_chart(bode_means_figure(items, title="|Z| — vue synthétique", theme_mode=theme,
                                              normalized=norm), width="stretch", key="viz_syn_bode")
        _skipped_note(items)


def _channel(ms, exp, ch_id: int, theme: str, eis_mode: str) -> None:
    steps, reps = _channel_filters(ms, exp, ch_id)
    norm = st.toggle("Normaliser les moyennes par le probe de chaque électrode", value=True,
                     key=f"viz_norm_{ch_id}", help=_NORM_HELP)
    win = _cv_window()
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
            items = channel_mean_view(ms, exp, KIND_CV, ch_id, steps, reps, norm, win)
            st.plotly_chart(cv_means_figure(items, title=f"CV — canal {ch_id} : moyennes"
                                            f"{' normalisées' if norm else ''} (- - E1 · ··· E2 · — E1+E2)",
                                            theme_mode=theme, normalized=norm),
                            width="stretch", key=f"viz_cv_{ch_id}_mean")
            _skipped_note(items)
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
        items = channel_mean_view(ms, exp, KIND_EIS, ch_id, steps, reps, norm, win)
        title = f"canal {ch_id} : moyennes{' normalisées' if norm else ''} (- - E1 · ··· E2 · — E1+E2)"
        fig = (nyquist_means_figure(items, title=f"Nyquist — {title}", theme_mode=theme, normalized=norm)
               if eis_mode == "Nyquist" else
               bode_means_figure(items, title=f"|Z| — {title}", theme_mode=theme, normalized=norm))
        st.plotly_chart(fig, width="stretch", key=f"viz_eis_{ch_id}_mean")
        _skipped_note(items)


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

