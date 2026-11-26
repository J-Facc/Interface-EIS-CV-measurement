"""Rapport PDF multi-pages (matplotlib, ``PdfPages``, backend Agg) — aucun import Streamlit.

Pages, dans l'ordre (:func:`pdf_plan`) :

1. page de garde (expérience, biomarqueurs, canaux, concentrations, date, version) ;
2. CV par canal (électrodes 1 et 2) ;
3. Nyquist + Bode par canal ;
4. DRT par canal (spectres moyens, pic principal et zone intégrée) ;
5. tableau des Rct (paysage, 28 lignes par page) ;
6. étalonnage EIS (une page par biomarqueur : sources DRT et fit) ;
7. étalonnage CV ;
8. spectres / courbes en échec, s'il y en a.

Aucun recalcul : données nettoyées, ``AnalysisResults`` (session fraîche ou archive) et
fonctions pures de ``core`` (affichage, étalonnage, tableaux).
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from core.calibration import SOURCE_DRT, SOURCE_FIT, SOURCE_LABELS, VIEW_BIOMARKER, cv_calibration, eis_calibration  # noqa: E402
from core.display import cv_display_groups, eis_display_groups  # noqa: E402
from core.experiment import KIND_CV, KIND_EIS, MODE_LABELS, step_sort_key  # noqa: E402
from core.report_tables import failure_rows, param_names, rct_rows  # noqa: E402
from core.version import APP_VERSION  # noqa: E402
from plotting import mpl_plots as mp  # noqa: E402

A4 = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)
RCT_ROWS_PER_PAGE = 28


def _channels_with(measurements, kind) -> list:
    return sorted({m.channel_id for m in measurements or [] if m.kind == kind})


def _channels_with_drt(results) -> list:
    if results is None:
        return []
    return sorted({s.channel_id for s in results.eis.spectra.values() if s.drt is not None})


def pdf_plan(experiment, measurements, results) -> list:
    """Liste ordonnée des pages ``(type, argument)`` — le PDF a exactement ``len(pdf_plan(...))`` pages."""
    plan = [("cover", None)]
    plan += [("cv", ch) for ch in _channels_with(measurements, KIND_CV)]
    plan += [("eis", ch) for ch in _channels_with(measurements, KIND_EIS)]
    plan += [("drt", ch) for ch in _channels_with_drt(results)]
    if results is not None:
        rows = rct_rows(results)
        plan += [("rct", i) for i in range(0, len(rows), RCT_ROWS_PER_PAGE)] or []
        bios = sorted({c.biomarker for c in eis_calibration(results, experiment, SOURCE_FIT, VIEW_BIOMARKER)})
        plan += [("cal_eis", b) for b in bios]
        if cv_calibration(results, experiment, VIEW_BIOMARKER):
            plan.append(("cal_cv", None))
        if results.failures:
            plan.append(("failures", None))
    return plan


def _cover(pdf, experiment, measurements, results) -> None:
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.9])
    lines = [f"Expérience : {experiment.name or '—'}",
             f"Rapport généré le {datetime.now():%d/%m/%Y %H:%M} — EIS Analyzer v{APP_VERSION}",
             f"Configuration créée le {experiment.created_at}",
             "",
             "Biomarqueurs : " + ", ".join(experiment.biomarkers),
             f"Canaux : {len(experiment.channels)} (2 électrodes chacun)", ""]
    for ch in experiment.channels:
        lines.append(f"  • Canal {ch.id} — {ch.biomarker} — {MODE_LABELS[ch.mode]} — "
                     + ", ".join(f"{c:.2e}" for c in ch.concentrations) + " mol/L")
    n_eis = sum(1 for m in measurements or [] if m.kind == KIND_EIS)
    lines += ["", f"Mesures nettoyées : {n_eis} EIS, {len(measurements or []) - n_eis} CV"]
    if results is not None:
        c = results.eis.circuit or {}
        lines += [f"Analyse du {results.computed_at} (v{results.app_version})",
                  f"Circuit : {c.get('expression', '—')}",
                  f"Paramètre cible : {c.get('target_param', '—')} — DRT : {results.eis.drt_mode or 'non calculée'}",
                  f"Fenêtre τ du pic principal : {(results.config.get('fit') or {}).get('drt', {}).get('tau_window')} s",
                  f"Fenêtre du pic CV : {(results.config.get('cv') or {}).get('peak_window_V')} V",
                  f"Éléments en échec : {len(results.failures)}"]
    else:
        lines += ["Aucune analyse : seules les données nettoyées sont présentées."]
    mp.text_axes(ax, lines, title="Rapport d'analyse EIS / CV")
    pdf.savefig(fig)
    plt.close(fig)


def _per_electrode(pdf, title, draw, n_rows=1) -> None:
    fig, axes = plt.subplots(n_rows, 2, figsize=A4, squeeze=False)
    fig.suptitle(title, fontsize=12)
    for el in (1, 2):
        draw(el, [axes[r][el - 1] for r in range(n_rows)])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _page(pdf, kind, arg, experiment, measurements, results) -> None:
    if kind == "cover":
        _cover(pdf, experiment, measurements, results)
    elif kind == "cv":
        def draw(el, axs):
            peaks = ([(f.step, f.e_peak, f.i_peak) for f in results.cv.values()
                      if (f.channel_id, f.electrode) == (arg, el) and f.ok] if results is not None else None)
            mp.cv_axes(axs[0], cv_display_groups(measurements, experiment, arg, el), f"Électrode {el}", peaks)
        _per_electrode(pdf, f"CV — canal {arg}", draw)
    elif kind == "eis":
        def draw(el, axs):
            groups = eis_display_groups(measurements, experiment, arg, el)
            mp.nyquist_axes(axs[0], groups, f"Nyquist — électrode {el}")
            mp.bode_axes(axs[1], axs[2], groups, f"Bode — électrode {el}")
        _per_electrode(pdf, f"EIS — canal {arg}", draw, n_rows=3)
    elif kind == "drt":
        def draw(el, axs):
            sps = sorted([s for s in results.eis.spectra.values() if s.channel_id == arg and s.electrode == el
                          and s.is_mean], key=lambda s: step_sort_key(s.step))
            mp.drt_axes(axs[0], [(s.step, s.drt) for s in sps], f"Électrode {el} — spectres moyens")
        _per_electrode(pdf, f"DRT — canal {arg}", draw)
    elif kind == "rct":
        rows = rct_rows(results)[arg:arg + RCT_ROWS_PER_PAGE]
        cols = (["Canal", "Électrode", "Étape", "Réplicat"] + param_names(results)
                + ["χ²ᵣ", "KK", "Rct_DRT (pic principal, Ω)", "Rct DRT moteur (contrôle, Ω)"])
        fig = plt.figure(figsize=A4_LANDSCAPE)
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.9])
        short = [c.replace(" (pic principal, Ω)", "").replace(" moteur (contrôle, Ω)", " moteur") for c in cols]
        mp.table_axes(ax, [{s: r.get(c) for s, c in zip(short, cols)} for r in rows], short,
                      title=f"Paramètres du fit Randles et Rct_DRT (Ω) — lignes {arg + 1} à {arg + len(rows)}")
        pdf.savefig(fig)
        plt.close(fig)
    elif kind == "cal_eis":
        fig, axes = plt.subplots(2, 1, figsize=A4)
        for ax, src in zip(axes, (SOURCE_DRT, SOURCE_FIT)):
            curves = [c for c in eis_calibration(results, experiment, src, VIEW_BIOMARKER) if c.biomarker == arg]
            mp.calibration_axes(ax, curves, f"Étalonnage EIS — {arg} — Rct {SOURCE_LABELS[src]}",
                                "log10(Rct / Rct_probe)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    elif kind == "cal_cv":
        fig, ax = plt.subplots(1, 1, figsize=A4)
        mp.calibration_axes(ax, cv_calibration(results, experiment, VIEW_BIOMARKER), "Étalonnage CV",
                            "ΔI/I = (I_probe − I) / I_probe")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    elif kind == "failures":
        fig = plt.figure(figsize=A4_LANDSCAPE)
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.9])
        rows = failure_rows(results)
        mp.table_axes(ax, rows[:40], list(rows[0]) if rows else [], title="Spectres / courbes en échec",
                      max_chars=110)
        pdf.savefig(fig)
        plt.close(fig)


def build_pdf(experiment, measurements, results=None) -> bytes:
    """Rapport PDF complet (bytes)."""
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"Rapport EIS — {experiment.name or 'expérience'}",
                                 "Creator": f"EIS Analyzer v{APP_VERSION}"}) as pdf:
        for kind, arg in pdf_plan(experiment, measurements, results):
            _page(pdf, kind, arg, experiment, measurements, results)
    return buf.getvalue()
