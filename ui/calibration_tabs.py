"""Onglets « Étalonnage EIS », « CV » et « Étalonnage CV » de la page Analyse.

Aucun calcul : courbes de ``core.calibration``, tableaux de ``core.report_tables``, figures de
``plotting``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.calibration import (
    LEGACY_LOGLOG,
    LEGACY_SIGNAL,
    SOURCE_DRT,
    SOURCE_FIT,
    SOURCE_LABELS,
    VIEW_BIOMARKER,
    VIEW_ELECTRODE,
    cv_calibration,
    eis_calibration,
    legacy_curves,
)
from core.display import cv_display_groups
from core.report_tables import (
    calibration_level_rows,
    calibration_point_rows,
    calibration_regression_rows,
    cv_rows,
)
from plotting.calibration_plots import calibration_figure_v3
from plotting.cv_plots import cv_figure

_VIEWS = {VIEW_BIOMARKER: "regroupé par biomarqueur", VIEW_ELECTRODE: "par électrode"}


def _view(key: str) -> str:
    return st.radio("Vue", list(_VIEWS), format_func=_VIEWS.get, horizontal=True, key=key)


def _curves_block(curves: list, title: str, y_title: str, theme: str, key: str) -> None:
    if not curves:
        st.info("Aucun point d'étalonnage : il faut des mesures « probe » et des concentrations.")
        return
    bios = sorted({c.biomarker for c in curves})
    pick = st.multiselect("Biomarqueurs", bios, default=bios, key=f"{key}_bios")
    curves = [c for c in curves if c.biomarker in pick]
    for bio in pick:                     # une figure d'étalonnage par biomarqueur
        mine = [c for c in curves if c.biomarker == bio]
        st.plotly_chart(calibration_figure_v3(mine, title=f"{title} — {bio}", y_title=y_title, theme_mode=theme),
                        width="stretch", key=f"{key}_fig_{bio}")
        for c in mine:
            if c.message:
                st.caption(f"• {c.label} : {c.message}")
    st.markdown("**Régressions** (y = a·log10([c]) + b, sur les moyennes par concentration)")
    st.dataframe(pd.DataFrame(calibration_regression_rows(curves)), hide_index=True, width="stretch")
    st.markdown("**Points moyens** (moyenne ± écart-type entre réplicats, ddof = 1)")
    st.dataframe(pd.DataFrame(calibration_level_rows(curves)), hide_index=True, width="stretch")
    with st.expander("Tous les réplicats (inclus et exclus, avec la raison)"):
        st.dataframe(pd.DataFrame(calibration_point_rows(curves)), hide_index=True, width="stretch")


def render_eis_calibration_tab(results, experiment, theme: str) -> None:
    st.caption("Rct_norm = Rct / Rct_probe, Rct_probe = moyenne des réplicats « probe » de la MÊME électrode ; "
               "log10(Rct_norm) vs log10([c]). Chaque réplicat est normalisé puis les réplicats sont moyennés "
               "par concentration. Un verdict KK non conforme est signalé (◆), jamais exclu.")
    c1, c2 = st.columns(2)
    source = c1.radio("Source de Rct", [SOURCE_DRT, SOURCE_FIT], format_func=SOURCE_LABELS.get, horizontal=True,
                      key="cal_eis_source")
    with c2:
        view = _view("cal_eis_view")
    curves = eis_calibration(results, experiment, source, view)
    _curves_block(curves, f"Étalonnage EIS ({SOURCE_LABELS[source]})", "log10(Rct / Rct_probe)", theme, "cal_eis")
    with st.expander("Anciennes courbes (v2) : spectre moyen, |Rct_probe − Rct_c| / Rct_probe et log-log"):
        c3, c4, c5 = st.columns(3)
        lsrc = c3.radio("Source", [SOURCE_FIT, SOURCE_DRT], horizontal=True, key="cal_legacy_src",
                        format_func=lambda s: "fit Randles" if s == SOURCE_FIT else "DRT (moteur, pic pénultième)")
        kind = c4.radio("Courbe", [LEGACY_SIGNAL, LEGACY_LOGLOG], horizontal=True, key="cal_legacy_kind",
                        format_func=lambda k: "signal normalisé" if k == LEGACY_SIGNAL else "log10(Rct)")
        strict = c5.checkbox("Régression stricte (exclure KK non conforme)", key="cal_legacy_strict")
        lc = legacy_curves(results, experiment, lsrc, kind, strict)
        if not lc:
            st.info("Aucun spectre moyen exploitable.")
        else:
            st.plotly_chart(calibration_figure_v3(lc, title="Étalonnage v2 (spectre moyen, par électrode)",
                                                  y_title=lc[0].quantity, theme_mode=theme, show_replicates=False),
                            width="stretch", key="cal_legacy_fig")
            st.dataframe(pd.DataFrame(calibration_regression_rows(lc)), hide_index=True, width="stretch")


def render_cv_tab(results, experiment, measurements, theme: str) -> None:
    st.caption("Courant de pic ANODIQUE : maximum de I sur la branche E croissante du cycle retenu, dans la fenêtre "
               f"cv.peak_window_V = {(results.config.get('cv') or {}).get('peak_window_V')} V (pas de correction "
               "de ligne de base). ΔI/I = (I_probe − I) / I_probe, I_probe = moyenne des pics « probe » de "
               "l'électrode.")
    if not results.cv:
        st.info("Aucune mesure CV.")
        return
    elecs = sorted({(f.channel_id, f.electrode) for f in results.cv.values()})
    ce = st.selectbox("Électrode", elecs, format_func=lambda x: f"Canal {x[0]} · électrode {x[1]}", key="cv_elec")
    if measurements:
        groups = cv_display_groups(measurements, experiment, ce[0], ce[1])
        peaks = [(f"{f.step} r{f.replicate}", f.e_peak, f.i_peak) for f in results.cv.values()
                 if (f.channel_id, f.electrode) == tuple(ce) and f.ok]
        st.plotly_chart(cv_figure(groups, title=f"CV — canal {ce[0]}, électrode {ce[1]} (× = pic anodique)",
                                  theme_mode=theme, peaks=peaks), width="stretch", key="cv_peaks_fig")
    st.dataframe(pd.DataFrame(cv_rows(results)), hide_index=True, width="stretch")


def render_cv_calibration_tab(results, experiment, theme: str) -> None:
    st.caption("ΔI/I vs log10([c]) : chaque réplicat normalisé par le probe de son électrode, puis moyenne ± "
               "écart-type (ddof = 1) par concentration.")
    view = _view("cal_cv_view")
    _curves_block(cv_calibration(results, experiment, view), "Étalonnage CV", "ΔI/I", theme, "cal_cv")
