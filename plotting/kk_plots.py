"""
plotting/kk_plots.py
====================
Figures Plotly pour la visualisation des résultats de validation KK.

Une figure : residuals_figure() — résidus Re et Im par fréquence (tous réplicats), lue par
l'onglet « Measurement model & fit Orazem » (ui/tabs.py). Le récapitulatif par spectre est
un tableau natif (core/results_table.kk_rows), plus une figure Plotly.

Règle architecture : reçoit des données, ne les calcule pas.
Aucun import Streamlit.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# Palettes (reprend la logique de plotting/theme.py)
# ─────────────────────────────────────────────

def _colors(theme_mode: str) -> dict:
    if theme_mode == "dark":
        return {
            "bg": "#0f1117",
            "paper": "#1a1d27",
            "text": "#e5e7eb",
            "grid": "#2d3148",
            "ok": "#22c55e",
            "warn": "#f59e0b",
            "error": "#ef4444",
            "re": "#60a5fa",
            "im": "#f472b6",
            "threshold": "#6b7280",
        }
    return {
        "bg": "#f7f4ef",
        "paper": "#ffffff",
        "text": "#0f1117",
        "grid": "#e2ddd6",
        "ok": "#15803d",
        "warn": "#b45309",
        "error": "#b91c1c",
        "re": "#1a56db",
        "im": "#db2777",
        "threshold": "#9ca3af",
    }


# ─────────────────────────────────────────────
# Figure 1 : résidus KK
# ─────────────────────────────────────────────

def residuals_figure(
    validation_result,          # ValidationResult de core/validator.py
    theme_mode: str = "light",
) -> go.Figure:
    """
    Retourne une figure Plotly avec les résidus KK normalisés (% de |Z|)
    en fonction de la fréquence, pour chaque réplicat du groupe.

    Measurement model : en haut, écart entre Re(Z) et la partie réelle PRÉDITE
    depuis l'ajustement de Im seule (statistique du test) ; en bas, résidu de cet
    ajustement de Im. Pointillés : bande ±2σ de chaque réplicat (bruit caractérisé
    + incertitude de prédiction). Sans structure d'erreur : résidus Lin-KK, sans
    bande (aucun verdict possible).
    Zone grisée hors de la plage KK-valide commune.
    """
    c = _colors(theme_mode)
    mm = any(getattr(kk, "method", "") == "measurement_model" for kk in validation_result.replicates)
    titles = (["Re(Z) − Re prédite depuis Im (%)", "Im(Z) − ajustement de Im (%)"] if mm
              else ["Résidus Lin-KK Re(Z) (%) — indicatifs", "Résidus Lin-KK Im(Z) (%) — indicatifs"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.10,
    )

    rep_colors_re = ["#1a56db", "#2563eb", "#3b82f6"]
    rep_colors_im = ["#db2777", "#ec4899", "#f472b6"]

    for i, kk in enumerate(validation_result.replicates):
        color_re = rep_colors_re[i % len(rep_colors_re)]
        color_im = rep_colors_im[i % len(rep_colors_im)]
        name = kk.label or f"Réplicat {i+1}"
        ok = kk.is_valid is not False
        opacity = 0.9 if ok else 0.4

        # Résidus Re
        fig.add_trace(go.Scatter(
            x=kk.frequencies,
            y=kk.residuals_re,
            mode="markers+lines",
            name=name,
            marker=dict(size=5, color=color_re, opacity=opacity),
            line=dict(width=1, color=color_re, dash="solid" if ok else "dot"),
            legendgroup=f"rep{i}",
            showlegend=True,
        ), row=1, col=1)

        # Résidus Im
        fig.add_trace(go.Scatter(
            x=kk.frequencies,
            y=kk.residuals_im,
            mode="markers+lines",
            name=name,
            marker=dict(size=5, color=color_im, opacity=opacity),
            line=dict(width=1, color=color_im, dash="solid" if ok else "dot"),
            legendgroup=f"rep{i}",
            showlegend=False,
        ), row=2, col=1)

        # Bandes ±2σ (critère unique, fits/kk_validation.kk_verdict)
        for row, band in ((1, getattr(kk, "band_re", None)), (2, getattr(kk, "band_im", None))):
            if band is None:
                continue
            for sign in (+1, -1):
                fig.add_trace(go.Scatter(
                    x=kk.frequencies, y=sign * band, mode="lines",
                    line=dict(width=1, color=c["threshold"], dash="dash"),
                    name="±2σ", legendgroup="band", showlegend=(i == 0 and row == 1 and sign > 0),
                    hoverinfo="skip",
                ), row=row, col=1)

    # Ligne zéro
    for row in [1, 2]:
        fig.add_hline(y=0, line=dict(color=c["text"], width=0.5), row=row, col=1)

    # Zone hors plage valide commune (grisée)
    if (validation_result.f_min_common > 0 and
            validation_result.f_max_common < np.inf and
            validation_result.replicates):
        all_freqs = validation_result.replicates[0].frequencies
        f_lo = all_freqs[0]
        f_hi = all_freqs[-1]
        for row in [1, 2]:
            # Zone BF invalide
            if validation_result.f_min_common > f_lo:
                fig.add_vrect(
                    x0=f_lo, x1=validation_result.f_min_common,
                    fillcolor="gray", opacity=0.12, line_width=0,
                    row=row, col=1,
                )
            # Zone HF invalide
            if validation_result.f_max_common < f_hi:
                fig.add_vrect(
                    x0=validation_result.f_max_common, x1=f_hi,
                    fillcolor="gray", opacity=0.12, line_width=0,
                    row=row, col=1,
                )

    # Mise en forme
    fig.update_xaxes(
        type="log",
        title_text="Fréquence (Hz)",
        gridcolor=c["grid"],
        row=2, col=1,
    )
    fig.update_xaxes(type="log", gridcolor=c["grid"], row=1, col=1)
    fig.update_yaxes(gridcolor=c["grid"])

    # Titre dynamique selon la validité globale
    status = {True: "✓ Conforme KK", False: "⚠ Non conforme KK",
              None: "? Verdict indéterminé (bruit non caractérisé)"}[validation_result.all_valid]
    drift_tag = " — drift inter-réplicats" if validation_result.drift_detected else ""

    fig.update_layout(
        title=dict(
            text=f"Validation KK — {validation_result.label} — {status}{drift_tag}",
            font=dict(size=13),
        ),
        paper_bgcolor=c["paper"],
        plot_bgcolor=c["bg"],
        font=dict(color=c["text"], size=11),
        legend=dict(
            bgcolor=c["paper"],
            bordercolor=c["grid"],
            borderwidth=1,
            font=dict(size=10),
        ),
        height=480,
        margin=dict(l=60, r=20, t=60, b=50),
    )

    return fig
