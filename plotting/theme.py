"""Colour palettes and Plotly layout styles for light and dark modes."""

import plotly.graph_objects as go

THEMES: dict = {
    "light": {
        "bg": "#FFFFFF",
        "paper_bg": "#F8F9FA",
        "text": "#212529",
        "grid": "#DEE2E6",
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "fit_dash": ["solid", "dash", "dot", "dashdot"],
    },
    "dark": {
        "bg": "#1a1a2e",
        "paper_bg": "#16213e",
        "text": "#E0E0E0",
        "grid": "#2d2d2d",
        "colors": ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#ff8a65"],
        "fit_dash": ["solid", "dash", "dot", "dashdot"],
    },
}


def get_theme(mode: str) -> dict:
    """Return theme dict for the requested mode.

    Args:
        mode: 'light' or 'dark'.

    Returns:
        Theme dict with keys bg, paper_bg, text, grid, colors, fit_dash.
    """
    return THEMES.get(mode, THEMES["light"])


def apply_theme_to_figure(fig: go.Figure, mode: str) -> go.Figure:
    """Apply colour theme to a Plotly figure.

    Args:
        fig: Plotly Figure to update in-place.
        mode: 'light' or 'dark'.

    Returns:
        The updated figure (same object).
    """
    theme = get_theme(mode)
    fig.update_layout(
        plot_bgcolor=theme["bg"],
        paper_bgcolor=theme["paper_bg"],
        font=dict(color=theme["text"]),
        xaxis=dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
        yaxis=dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"]),
    )
    return fig
