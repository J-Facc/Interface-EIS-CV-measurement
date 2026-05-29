"""Plotly figures for CV analysis: I/E curves and normalized calibration plot."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from core.cv_models import CVSession
from plotting.theme import apply_theme_to_figure, get_theme


def cv_current_figure(cv_session: CVSession, theme_mode: str = "light") -> go.Figure:
    """Plot I vs E for probe and each concentration.

    Args:
        cv_session: CVSession with probe and groups.
        theme_mode: 'light' or 'dark'.

    Returns:
        Plotly Figure.
    """
    theme = get_theme(theme_mode)
    colors = theme["colors"]
    fig = go.Figure()

    color_idx = 0

    if cv_session.probe is not None:
        probe = cv_session.probe
        fig.add_trace(go.Scatter(
            x=probe.E,
            y=probe.I,
            mode="lines",
            name="Probe",
            line=dict(color=colors[color_idx % len(colors)], width=2),
        ))
        color_idx += 1

    for grp in cv_session.groups:
        sc = grp.scan
        conc = grp.concentration
        if conc > 0:
            exp = int(np.floor(np.log10(conc)))
            mant = conc / 10 ** exp
            label = f"{mant:.1f}×10<sup>{exp}</sup> M"
        else:
            label = "0 M"
        fig.add_trace(go.Scatter(
            x=sc.E,
            y=sc.I,
            mode="lines",
            name=label,
            line=dict(color=colors[color_idx % len(colors)], width=2),
        ))
        color_idx += 1

    fig.update_layout(
        title="Voltampérométrie cyclique — Courant vs Potentiel",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (A)",
        legend=dict(title="Étape / Concentration"),
    )

    apply_theme_to_figure(fig, theme_mode)
    return fig


def cv_calibration_figure(cv_session: CVSession, theme_mode: str = "light") -> go.Figure:
    """Plot mean normalized delta_signal vs log10(concentration) with linear fit.

    Args:
        cv_session: CVSession with groups containing delta_signal arrays.
        theme_mode: 'light' or 'dark'.

    Returns:
        Plotly Figure.
    """
    theme = get_theme(theme_mode)
    colors = theme["colors"]

    valid_groups = [
        grp for grp in cv_session.groups
        if grp.concentration > 0 and not np.all(np.isnan(grp.delta_signal))
    ]

    fig = go.Figure()

    if not valid_groups:
        fig.add_annotation(
            text="Pas de données de calibration disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        apply_theme_to_figure(fig, theme_mode)
        return fig

    log_concs = np.array([np.log10(grp.concentration) for grp in valid_groups])
    mean_signals = np.array([np.nanmean(grp.delta_signal) for grp in valid_groups])

    fig.add_trace(go.Scatter(
        x=log_concs,
        y=mean_signals,
        mode="markers",
        name="Signal normalisé",
        marker=dict(color=colors[0], size=10),
    ))

    if len(valid_groups) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_concs, mean_signals)
        r2 = r_value ** 2

        x_fit = np.linspace(log_concs.min(), log_concs.max(), 100)
        y_fit = slope * x_fit + intercept

        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name="Fit linéaire",
            line=dict(color=colors[1], dash="dash", width=2),
        ))

        sign = "+" if intercept >= 0 else "-"
        annotation_text = (
            f"y = {slope:.4f}·x {sign} {abs(intercept):.4f}<br>"
            f"R² = {r2:.4f}"
        )
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        title=(
            "Calibration CV — Signal normalisé vs log₁₀([c])<br>"
            "<sup>Méthode : régression linéaire (scipy.stats.linregress)</sup>"
        ),
        xaxis_title="log₁₀(concentration [M])",
        yaxis_title="Signal normalisé moyen |ΔI| / |I_probe|",
        legend=dict(title=""),
    )

    apply_theme_to_figure(fig, theme_mode)
    return fig
