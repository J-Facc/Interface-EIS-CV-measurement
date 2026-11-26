"""Plotly figures for CV analysis: curve overlay and calibration.

Visualisation et calibration seulement. Toute régression vient de
core.calibration.compute_cv_calibration (source unique, partagée avec les
exports CSV) : ce module ne recalcule jamais de régression lui-même.
"""

import numpy as np
import plotly.graph_objects as go

from core.calibration import compute_cv_calibration
from core.cv_models import CVSession


# Gris atténué par thème pour la trace de référence bare (cohérent avec
# plotting/eis_plots.py). Aucun calcul n'en dépend.
_BARE_REF_COLOR = {"light": "#8a8f98", "dark": "#c7ccd4"}


def cv_current_figure(cv_session: CVSession, theme_mode: str = "light") -> go.Figure:
    """I (A) vs E (V) for probe + each concentration group.

    Si `cv_session.bare_reference` est renseigné, la courbe « électrode nue » est
    superposée en style référence (pointillés gris, opacité réduite). AFFICHAGE
    SEUL — n'entre dans aucun calcul.
    """
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

    bare = getattr(cv_session, "bare_reference", None)
    if bare is not None:
        color = _BARE_REF_COLOR.get(theme_mode, _BARE_REF_COLOR["light"])
        fig.add_trace(go.Scatter(
            x=bare.E, y=bare.I,
            mode="lines",
            name="Électrode nue (réf.)",
            line=dict(color=color, dash="dot", width=1.5),
            opacity=0.6,
        ))

    fig.update_layout(
        title="Voltampérométrie cyclique — Courant vs Potentiel",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (A)",
        legend_title="Scan",
    )
    return fig


_ELECTRODE_STYLES = [
    dict(marker_color="steelblue", line_color="steelblue", line_dash="solid"),
    dict(marker_color="darkorange", line_color="darkorange", line_dash="dash"),
    dict(marker_color="seagreen", line_color="seagreen", line_dash="dot"),
    dict(marker_color="firebrick", line_color="firebrick", line_dash="dashdot"),
]


def _fit_line(cal, n: int = 200):
    """Points (x, y) de la droite de régression d'un CVCalibrationResult."""
    x = np.linspace(cal.log_c.min(), cal.log_c.max(), n)
    return x, cal.slope * x + cal.intercept


def cv_calibration_figure_multi(cv_sessions: dict) -> go.Figure:
    """Une droite de régression par électrode, sur le même graphe.

    Les régressions viennent de compute_cv_calibration. Une électrode à moins
    de 2 concentrations exploitables est omise.
    """
    fig = go.Figure()
    annotation_lines = []

    for i, (elec, session) in enumerate(sorted(cv_sessions.items())):
        cal = compute_cv_calibration(session)
        if cal is None:
            continue

        style = _ELECTRODE_STYLES[i % len(_ELECTRODE_STYLES)]
        x_fit, y_fit = _fit_line(cal)

        fig.add_trace(go.Scatter(
            x=cal.log_c, y=cal.signals,
            mode="markers",
            name=f"Électrode {elec} — mesuré",
            marker=dict(size=10, color=style["marker_color"]),
            legendgroup=f"e{elec}",
        ))
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines",
            name=f"Électrode {elec} — régression R²={cal.r2:.3f}",
            line=dict(color=style["line_color"], dash=style["line_dash"]),
            legendgroup=f"e{elec}",
        ))

        annotation_lines.append(
            f"Électrode {elec} : R² = {cal.r2:.4f}, pente = {cal.slope:.3f}, "
            f"ordonnée = {cal.intercept:.3f}"
        )

    if annotation_lines:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text="<br>".join(annotation_lines),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        title="Calibration CV — Signal normalisé vs log([c]) par électrode",
        xaxis_title="log([c] / M)",
        yaxis_title="|ΔI| / |I_probe|",
        legend_title="Électrode",
    )
    return fig


def open_cv_calibration_matplotlib_window_multi(cv_sessions: dict) -> None:
    """Ouvre une fenêtre matplotlib (bloquante) reproduisant
    cv_calibration_figure_multi() : une droite par électrode."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "d"]
    linestyles = ["--", "-.", ":", "-"]

    any_plotted = False
    for i, (elec, session) in enumerate(sorted(cv_sessions.items())):
        cal = compute_cv_calibration(session)
        if cal is None:
            continue
        ax.plot(cal.log_c, cal.signals, markers[i % len(markers)],
                label=f"Électrode {elec} — mesuré")
        x_fit, y_fit = _fit_line(cal)
        ax.plot(x_fit, y_fit, linestyles[i % len(linestyles)],
                label=f"Électrode {elec} — régression (R²={cal.r2:.3f})")
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "Pas assez de points", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title("Calibration CV — Signal normalisé vs log([c]) par électrode")
    ax.set_xlabel("log([c] / M)")
    ax.set_ylabel("|ΔI| / |I_probe|")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# v3 — Visualisation des CV nettoyées (aucun calcul)
# ═════════════════════════════════════════════════════════════════════════════

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
