"""Plotly figures for CV analysis."""

import numpy as np
from scipy import stats
import plotly.graph_objects as go

from core.cv_models import CVSession


def cv_current_figure(cv_session: CVSession) -> go.Figure:
    """I (A) vs E (V) for probe + each concentration group."""
    fig = go.Figure()

    if cv_session.probe is not None:
        p = cv_session.probe
        fig.add_trace(go.Scatter(
            x=p.E, y=p.I,
            mode="lines",
            name=f"Probe ({', '.join(p.source_files)})",
        ))

    for grp in cv_session.groups:
        s = grp.scan
        fig.add_trace(go.Scatter(
            x=s.E, y=s.I,
            mode="lines",
            name=f"{grp.concentration:.2e} M",
        ))

    fig.update_layout(
        title="Voltampérométrie cyclique — Courant vs Potentiel",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (A)",
        legend_title="Scan",
    )
    return fig


def cv_calibration_figure(cv_session: CVSession) -> go.Figure:
    """Normalized signal vs log([c]) with linear regression."""
    if len(cv_session.groups) < 2:
        return go.Figure()

    concentrations = []
    signals = []
    for grp in cv_session.groups:
        if grp.concentration <= 0:
            continue
        mean_sig = np.nanmean(grp.delta_signal)
        if np.isfinite(mean_sig):
            concentrations.append(grp.concentration)
            signals.append(mean_sig)

    if len(concentrations) < 2:
        return go.Figure()

    log_c = np.log10(concentrations)
    sig = np.array(signals)

    slope, intercept, r_value, _, _ = stats.linregress(log_c, sig)
    r2 = r_value ** 2

    x_fit = np.linspace(log_c.min(), log_c.max(), 200)
    y_fit = slope * x_fit + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_c, y=sig,
        mode="markers",
        name="Signal mesuré",
        marker=dict(size=10),
    ))
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines",
        name="Régression linéaire",
        line=dict(dash="dash"),
    ))

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=f"R² = {r2:.2f} — régression linéaire (scipy.stats.linregress)",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="gray",
        borderwidth=1,
    )

    fig.update_layout(
        title="Calibration CV — Signal normalisé vs log([c])",
        xaxis_title="log([c] / M)",
        yaxis_title="|ΔI| / |I_probe|",
        legend_title="Courbe",
    )
    return fig
