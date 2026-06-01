"""
plotting/kk_plots.py
====================
Figures Plotly pour la visualisation des résultats de validation KK.

Deux figures :
  1. residuals_figure()  — résidus Re et Im par fréquence (tous réplicats)
  2. validation_badge()  — indicateur de validité compact pour l'onglet Nyquist

Règle architecture : reçoit des données, ne les calcule pas.
Aucun import Streamlit.
"""

from __future__ import annotations

from typing import List, Optional

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
    residual_threshold_pct: float = 2.0,
) -> go.Figure:
    """
    Retourne une figure Plotly avec les résidus KK normalisés (%)
    en fonction de la fréquence, pour chaque réplicat du groupe.

    Deux sous-graphes : résidus Re (haut) et résidus Im (bas).
    Ligne en pointillés aux seuils ±threshold_pct.
    Zone grisée hors de la plage KK-valide commune.
    """
    c = _colors(theme_mode)
    n_rep = len(validation_result.replicates)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Résidus Re(Z) (%)", "Résidus Im(Z) (%)"],
        vertical_spacing=0.10,
    )

    rep_colors_re = ["#1a56db", "#2563eb", "#3b82f6"]
    rep_colors_im = ["#db2777", "#ec4899", "#f472b6"]

    for i, kk in enumerate(validation_result.replicates):
        color_re = rep_colors_re[i % len(rep_colors_re)]
        color_im = rep_colors_im[i % len(rep_colors_im)]
        name = kk.label or f"Réplicat {i+1}"
        opacity = 0.9 if kk.is_valid else 0.4

        # Résidus Re
        fig.add_trace(go.Scatter(
            x=kk.frequencies,
            y=kk.residuals_re,
            mode="markers+lines",
            name=name,
            marker=dict(size=5, color=color_re, opacity=opacity),
            line=dict(width=1, color=color_re, dash="solid" if kk.is_valid else "dot"),
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
            line=dict(width=1, color=color_im, dash="solid" if kk.is_valid else "dot"),
            legendgroup=f"rep{i}",
            showlegend=False,
        ), row=2, col=1)

    # Lignes seuil ±threshold
    for row in [1, 2]:
        for sign in [+1, -1]:
            fig.add_hline(
                y=sign * residual_threshold_pct,
                line=dict(color=c["threshold"], width=1, dash="dash"),
                row=row, col=1,
            )

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
    status = "✓ Valide" if validation_result.all_valid else "⚠ Problème détecté"
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


# ─────────────────────────────────────────────
# Figure 2 : badge compact (inline dans onglet Nyquist)
# ─────────────────────────────────────────────

def validation_summary_table(
    validation_results: dict,   # { label: ValidationResult }
    theme_mode: str = "light",
) -> go.Figure:
    """
    Tableau Plotly compact résumant la validité KK de tous les spectres.
    Une ligne par spectre, colonnes : label, µ moyen, χ² moyen, drift, verdict.
    Destiné à être affiché dans un expander de l'onglet Paramètres ou Nyquist.
    """
    c = _colors(theme_mode)

    labels, mus, chi2s, drifts, verdicts, colors_cell = [], [], [], [], [], []

    for label, vr in validation_results.items():
        if not vr.replicates:
            continue
        mu_mean = np.mean([kk.mu for kk in vr.replicates])
        chi2_mean = np.mean([kk.chi2_pseudo for kk in vr.replicates])

        labels.append(label)
        mus.append(f"{mu_mean:.3f}")
        chi2s.append(f"{chi2_mean:.4f}")
        drifts.append("Oui ⚠" if vr.drift_detected else "Non ✓")

        if not vr.all_valid:
            verdicts.append("❌ Invalide")
            colors_cell.append("#fee2e2")
        elif vr.drift_detected:
            verdicts.append("⚠ Drift")
            colors_cell.append("#fef9c3")
        else:
            verdicts.append("✅ Valide")
            colors_cell.append("#dcfce7")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Spectre", "µ moyen", "χ² pseudo", "Drift", "Verdict"],
            fill_color=c["paper"],
            font=dict(color=c["text"], size=11),
            align="left",
            line_color=c["grid"],
        ),
        cells=dict(
            values=[labels, mus, chi2s, drifts, verdicts],
            fill_color=[
                [c["paper"]] * len(labels),
                [c["paper"]] * len(labels),
                [c["paper"]] * len(labels),
                [c["paper"]] * len(labels),
                colors_cell,
            ],
            font=dict(color=c["text"], size=11),
            align="left",
            line_color=c["grid"],
        ),
    )])

    fig.update_layout(
        paper_bgcolor=c["paper"],
        font=dict(color=c["text"]),
        margin=dict(l=0, r=0, t=10, b=0),
        height=max(120, 40 * len(labels) + 50),
    )

    return fig
