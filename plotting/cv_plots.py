"""Figures Plotly CV : superposition des courbes I(E) et des pics anodiques. Aucun calcul.

Les régressions d'étalonnage viennent de ``core/calibration.py`` et sont tracées par
``plotting/calibration_plots.py``.
"""

import numpy as np
import plotly.graph_objects as go


def _step_color(step: str, concentration, c_min: float, c_max: float) -> str:
    import plotly.colors as pc
    if concentration and concentration > 0:
        t = 0.5 if c_min >= c_max else (np.log10(concentration) - np.log10(c_min)) / (np.log10(c_max) - np.log10(c_min))
        return pc.sample_colorscale("plasma", [0.85 * float(np.clip(t, 0, 1))])[0]
    return "#6b7280" if step == "bare" else "black"


def cv_figure(groups: list, title: str = "Voltammétrie cyclique", theme_mode: str = "light",
              peaks: list = None) -> go.Figure:
    """Courbes I(E) superposées : une couleur par étape, réplicats en traits fins.

    Args:
        groups: ``[{"label", "step", "concentration", "curves": [(nom, E, I)]}]`` — une entrée
            par étape (bare, probe, conc_i) ; I en ampères (affiché en µA).
        peaks: points de pic à marquer ``[(nom, E_pic, I_pic)]`` (onglet CV de l'analyse).
    """
    from plotting.theme import apply_theme_to_figure

    fig = go.Figure()
    concs = [g["concentration"] for g in groups if g.get("concentration")]
    c_min, c_max = (min(concs), max(concs)) if concs else (1.0, 1.0)
    for g in groups:
        color = _step_color(g["step"], g.get("concentration"), c_min, c_max)
        for i, (name, E, I) in enumerate(g["curves"]):
            fig.add_trace(go.Scatter(
                x=np.asarray(E), y=np.asarray(I) * 1e6, mode="lines", legendgroup=g["label"],
                name=g["label"] if i == 0 else name, showlegend=(i == 0),
                line=dict(color=color, width=2 if len(g["curves"]) == 1 else 1.2),
                opacity=1.0 if len(g["curves"]) == 1 else 0.75,
                hovertemplate=f"<b>{name}</b><br>E = %{{x:.4f}} V<br>I = %{{y:.4g}} µA<extra></extra>",
            ))
    for name, e, i in peaks or []:
        fig.add_trace(go.Scatter(x=[e], y=[i * 1e6], mode="markers", name=f"pic — {name}", showlegend=False,
                                 marker=dict(symbol="x", size=10, color="red"),
                                 hovertemplate=f"<b>pic {name}</b><br>E = %{{x:.4f}} V<br>"
                                               "I = %{y:.4g} µA<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="E (V)", yaxis_title="I (µA)", hovermode="closest",
                      height=480, legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
                      margin=dict(r=120))
    apply_theme_to_figure(fig, theme_mode)
    return fig
