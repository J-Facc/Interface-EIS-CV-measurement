"""Rapport HTML interactif autonome (plotly.js embarqué, sommaire) — aucun import Streamlit.

Mêmes figures que l'application (fonctions de ``plotting``, version Plotly) et mêmes tableaux
(``core.report_tables``). plotly.js est inclus UNE fois, dans la première figure : le fichier
s'ouvre hors ligne, sans dépendance.
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from core.calibration import SOURCE_DRT, SOURCE_FIT, SOURCE_LABELS, VIEW_BIOMARKER, cv_calibration, eis_calibration
from core.display import cv_display_groups, eis_display_groups
from core.experiment import KIND_CV, KIND_EIS, MODE_LABELS, step_sort_key
from core.report_tables import (
    calibration_level_rows,
    calibration_regression_rows,
    cv_rows,
    failure_rows,
    group_rows,
    rct_rows,
)
from core.version import APP_VERSION
from plotting.calibration_plots import calibration_figure_v3
from plotting.cv_plots import cv_figure
from plotting.eis_plots import bode_figure, drt_peak_figure, nyquist_replicates_figure

_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1f2937;background:#fff}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:1.6rem;margin:.5rem 0}h2{border-bottom:2px solid #e5e7eb;padding-bottom:4px;margin-top:2rem}
nav{background:#f3f4f6;padding:12px 16px;border-radius:8px}nav a{display:inline-block;margin:2px 10px 2px 0}
table.dataframe{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto}
table.dataframe th,table.dataframe td{border:1px solid #d1d5db;padding:3px 6px;text-align:right}
table.dataframe th{background:#eef2f7}.meta{color:#4b5563}
"""


class _Doc:
    def __init__(self):
        self.parts: list = []
        self.toc: list = []
        self.n_figures = 0

    def section(self, sid: str, title: str) -> None:
        self.toc.append((sid, title))
        self.parts.append(f'<h2 id="{sid}">{html.escape(title)}</h2>')

    def fig(self, fig) -> None:
        self.parts.append(fig.to_html(full_html=False, include_plotlyjs=(self.n_figures == 0),
                                      config={"displaylogo": False}))
        self.n_figures += 1

    def table(self, rows: list) -> None:
        if rows:
            self.parts.append(pd.DataFrame(rows).to_html(index=False, na_rep="—", float_format=lambda x: f"{x:.5g}",
                                                         border=0))
        else:
            self.parts.append("<p class='meta'>Aucune ligne.</p>")

    def p(self, text: str) -> None:
        self.parts.append(f"<p>{html.escape(text)}</p>")


def build_html(experiment, measurements, results=None, theme_mode: str = "light") -> str:
    """Rapport HTML complet (str, autonome)."""
    d = _Doc()
    d.section("experience", "Expérience")
    d.p(f"{experiment.name or 'Expérience'} — biomarqueurs : {', '.join(experiment.biomarkers)} — "
        f"{len(experiment.channels)} canal(aux) de 2 électrodes.")
    d.table([{"Canal": ch.id, "Biomarqueur": ch.biomarker, "Mode": MODE_LABELS[ch.mode],
              "Concentrations (mol/L)": ", ".join(f"{c:.2e}" for c in ch.concentrations)} for ch in experiment.channels])
    if results is not None:
        c = results.eis.circuit or {}
        d.p(f"Analyse du {results.computed_at} (v{results.app_version}) — circuit {c.get('expression', '—')}, "
            f"paramètre cible {c.get('target_param', '—')}, DRT : {results.eis.drt_mode or 'non calculée'}.")

    for ch in sorted({m.channel_id for m in measurements or [] if m.kind == KIND_CV}):
        d.section(f"cv-c{ch}", f"CV — canal {ch}")
        for el in (1, 2):
            groups = cv_display_groups(measurements, experiment, ch, el)
            if groups:
                peaks = ([(f.step, f.e_peak, f.i_peak) for f in results.cv.values()
                          if (f.channel_id, f.electrode) == (ch, el) and f.ok] if results is not None else None)
                d.fig(cv_figure(groups, title=f"CV — canal {ch}, électrode {el}", theme_mode=theme_mode, peaks=peaks))
    for ch in sorted({m.channel_id for m in measurements or [] if m.kind == KIND_EIS}):
        d.section(f"eis-c{ch}", f"EIS — canal {ch} (Nyquist et Bode)")
        for el in (1, 2):
            groups = eis_display_groups(measurements, experiment, ch, el)
            if groups:
                d.fig(nyquist_replicates_figure(groups, title=f"Nyquist — canal {ch}, électrode {el}",
                                                theme_mode=theme_mode))
                d.fig(bode_figure(groups, title=f"Bode — canal {ch}, électrode {el}", theme_mode=theme_mode))
    if results is not None:
        drt_ch = sorted({s.channel_id for s in results.eis.spectra.values() if s.drt is not None})
        for ch in drt_ch:
            d.section(f"drt-c{ch}", f"DRT — canal {ch}")
            for el in (1, 2):
                sps = sorted([s for s in results.eis.spectra.values() if s.channel_id == ch and s.electrode == el
                              and s.is_mean and s.drt is not None], key=lambda s: step_sort_key(s.step))
                if sps:
                    d.fig(drt_peak_figure([(s.step, s.drt) for s in sps],
                                          title=f"DRT — canal {ch}, électrode {el} (spectres moyens)",
                                          theme_mode=theme_mode))
        d.section("rct", "Rct — fit Randles et DRT")
        d.table(group_rows(results))
        d.table(rct_rows(results))
        d.section("cal-eis", "Étalonnage EIS")
        for src in (SOURCE_DRT, SOURCE_FIT):
            curves = eis_calibration(results, experiment, src, VIEW_BIOMARKER)
            for bio in sorted({c.biomarker for c in curves}):
                d.fig(calibration_figure_v3([c for c in curves if c.biomarker == bio],
                                            title=f"Étalonnage EIS — {bio} — Rct {SOURCE_LABELS[src]}",
                                            y_title="log10(Rct / Rct_probe)", theme_mode=theme_mode))
            d.table(calibration_regression_rows(curves))
            d.table(calibration_level_rows(curves))
        d.section("cv-features", "CV — courants de pic et ΔI/I")
        d.table(cv_rows(results))
        d.section("cal-cv", "Étalonnage CV")
        curves = cv_calibration(results, experiment, VIEW_BIOMARKER)
        if curves:
            d.fig(calibration_figure_v3(curves, title="Étalonnage CV", y_title="ΔI/I", theme_mode=theme_mode))
        d.table(calibration_regression_rows(curves))
        d.table(calibration_level_rows(curves))
        if results.failures:
            d.section("echecs", "Spectres / courbes en échec")
            d.table(failure_rows(results))

    toc = "".join(f'<a href="#{sid}">{html.escape(t)}</a>' for sid, t in d.toc)
    title = html.escape(f"Rapport EIS — {experiment.name or 'expérience'}")
    return (f"<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{title}</title>"
            f"<style>{_CSS}</style></head><body><main><h1>{title}</h1>"
            f"<p class='meta'>Généré le {datetime.now():%d/%m/%Y %H:%M} — EIS Analyzer v{APP_VERSION}</p>"
            f"<nav><strong>Sommaire :</strong> {toc}</nav>{''.join(d.parts)}</main></body></html>")
