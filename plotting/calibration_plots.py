"""Figures d'étalonnage v3 (Plotly) : moyennes ± écart-type, régression, R².

Aucun calcul : les courbes (niveaux, régression) viennent de ``core.calibration``.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from plotting.theme import apply_theme_to_figure, get_theme


def _fmt(x, spec=".4g") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, spec)


def calibration_figure_v3(curves: list, title: str = "Étalonnage", y_title: str = "",
                          theme_mode: str = "light", show_replicates: bool = True) -> go.Figure:
    """Une trace par courbe : moyennes par concentration avec barres d'erreur (± écart-type,
    ddof = 1 ; aucune barre pour un seul réplicat), réplicats inclus en points pâles, droite de
    régression et R² dans la légende ; points exclus en croix grises."""
    theme = get_theme(theme_mode)
    fig = go.Figure()
    for k, c in enumerate(curves):
        color = theme["colors"][k % len(theme["colors"])]
        if show_replicates:
            inc = [p for p in c.points if p.included and np.isfinite(p.y)]
            if inc:
                fig.add_trace(go.Scatter(
                    x=[np.log10(p.concentration) for p in inc], y=[p.y for p in inc], mode="markers",
                    marker=dict(color=color, size=6, opacity=0.35,
                                symbol=["diamond" if p.kk_nonconform else "circle" for p in inc]),
                    legendgroup=c.label, showlegend=False,
                    text=[f"canal {p.channel_id} · E{p.electrode} · {p.step} r{p.replicate}"
                          + (" — KK non conforme" if p.kk_nonconform else "") for p in inc],
                    hovertemplate="%{text}<br>y = %{y:.4g}<extra></extra>"))
            exc = [p for p in c.points if not p.included]
            if exc:
                fig.add_trace(go.Scatter(
                    x=[np.log10(p.concentration) for p in exc], y=[0.0] * len(exc), mode="markers",
                    marker=dict(color="gray", symbol="x", size=8), legendgroup=c.label, showlegend=False,
                    text=[f"exclu : canal {p.channel_id} · E{p.electrode} · {p.step} r{p.replicate} — {p.reason}"
                          for p in exc], hovertemplate="%{text}<extra></extra>"))
        if not c.levels:
            continue
        x = np.array([lv.log_c for lv in c.levels])
        y = np.array([lv.mean for lv in c.levels])
        err = np.array([lv.std if np.isfinite(lv.std) else 0.0 for lv in c.levels])
        name = c.label + (f" — R² = {c.r2:.4f}" if c.has_regression else "")
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=name, legendgroup=c.label,
            marker=dict(color=color, size=11, line=dict(color=theme["text"], width=1)),
            error_y=dict(type="data", array=err, visible=True, color=color, thickness=1.5, width=6),
            customdata=np.array([[lv.n, lv.std, lv.concentration] for lv in c.levels]),
            hovertemplate=("[c] = %{customdata[2]:.2e} M<br>moyenne = %{y:.4g}<br>"
                           "écart-type = %{customdata[1]:.3g}<br>n = %{customdata[0]}<extra></extra>")))
        if c.has_regression:
            xx = np.linspace(x.min() - 0.2, x.max() + 0.2, 50)
            fig.add_trace(go.Scatter(
                x=xx, y=c.slope * xx + c.intercept, mode="lines", legendgroup=c.label, showlegend=False,
                line=dict(color=color, dash="dash", width=1.5),
                hovertemplate=(f"y = {_fmt(c.slope)}·log10([c]) + {_fmt(c.intercept)}<br>"
                               f"R² = {c.r2:.4f}<extra></extra>")))
    fig.update_layout(title=title, xaxis_title="log10([c] / (mol/L))", yaxis_title=y_title, height=500,
                      hovermode="closest", legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
                      margin=dict(r=200))
    apply_theme_to_figure(fig, theme_mode)
    return fig
