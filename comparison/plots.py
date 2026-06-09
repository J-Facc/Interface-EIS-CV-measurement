"""Figures Plotly pour le rapport comparatif — page C.

Aucun import Streamlit. Toutes les fonctions reçoivent des données et
retournent un go.Figure. Les calculs statistiques (OLS, IC) sont faits ici
car ils sont inséparables de la construction visuelle.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from plotting.theme import get_theme, apply_theme_to_figure


# Palette fixe pour les 6 méthodes — identique sur toutes les figures
_METHOD_COLORS = {
    "A1": "#1f77b4",   # bleu
    "A2": "#ff7f0e",   # orange
    "A3": "#2ca02c",   # vert
    "B1": "#d62728",   # rouge
    "B2": "#9467bd",   # violet
    "B3": "#8c564b",   # brun
}
_METHOD_SYMBOLS = {
    "A1": "circle",
    "A2": "square",
    "A3": "diamond",
    "B1": "cross",
    "B2": "x",
    "B3": "triangle-up",
}

_SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup(n: int) -> str:
    return str(n).translate(_SUP_MAP)


def _ols_with_ci(x: np.ndarray, y: np.ndarray, n_fit: int = 200) -> dict:
    """
    Régression OLS y ~ x avec IC 95 % analytique.

    Returns dict : slope, intercept, r2, x_fit, y_fit, ci_low, ci_high.
    """
    res = stats.linregress(x, y)
    n = len(x)
    t95 = stats.t.ppf(0.975, df=max(n - 2, 1))
    x_fit = np.linspace(x.min(), x.max(), n_fit)
    y_fit = res.slope * x_fit + res.intercept
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    se_fit = res.stderr * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / (denom if denom > 0 else 1))
    return {
        "slope":     res.slope,
        "intercept": res.intercept,
        "r2":        res.rvalue ** 2,
        "stderr":    res.stderr,
        "x_fit":     x_fit,
        "y_fit":     y_fit,
        "ci_low":    y_fit - t95 * se_fit,
        "ci_high":   y_fit + t95 * se_fit,
    }


def plot_calibration_scalar(
    signal: np.ndarray,
    concentrations: np.ndarray,
    method_name: str,
    theme_mode: str = "light",
) -> go.Figure:
    """
    Calibration scalaire : log₁₀(signal) vs log₁₀([c]) + OLS + IC 95 %.

    Utilisé pour les méthodes A1 (ΔI_norm), A2 (Rct fit), A3 (Rct DRT).

    Parameters
    ----------
    signal : np.ndarray
        Valeurs du signal normalisé (ΔI_norm ou Rct_norm) pour chaque concentration.
        Doit être strictement positif pour l'axe log.
    concentrations : np.ndarray
        Concentrations en molaire, même ordre que signal.
    method_name : str
        Nom de la méthode affiché dans le titre et la légende (ex. 'A2').
    theme_mode : str
        'light' ou 'dark'.

    Returns
    -------
    go.Figure
        Figure Plotly interactive.
    """
    signal = np.asarray(signal, dtype=float)
    concentrations = np.asarray(concentrations, dtype=float)

    color = _METHOD_COLORS.get(method_name, "#333333")
    theme = get_theme(theme_mode)

    # Filtrer les valeurs non-positives pour l'axe log
    mask = (signal > 0) & (concentrations > 0) & np.isfinite(signal) & np.isfinite(concentrations)
    x = np.log10(concentrations[mask])
    y = np.log10(signal[mask])

    fig = go.Figure()

    if len(x) >= 2:
        ols = _ols_with_ci(x, y)

        # IC 95 % — bande remplie
        fig.add_trace(go.Scatter(
            x=np.concatenate([ols["x_fit"], ols["x_fit"][::-1]]),
            y=np.concatenate([ols["ci_high"], ols["ci_low"][::-1]]),
            fill="toself",
            fillcolor=f"rgba{tuple(list(_hex_to_rgb(color)) + [0.12])}",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 95 %",
            hoverinfo="skip",
        ))
        # Droite OLS
        fig.add_trace(go.Scatter(
            x=ols["x_fit"], y=ols["y_fit"],
            mode="lines",
            name=f"OLS — R²={ols['r2']:.4f}",
            line=dict(color=color, width=2, dash="dash"),
        ))
        fig.add_annotation(
            xref="paper", yref="paper", x=0.04, y=0.96,
            text=(
                f"<b>{method_name}</b>  y = {ols['slope']:.3f}·x "
                f"+ {ols['intercept']:.3f}<br>R² = {ols['r2']:.4f}"
            ),
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.85)" if theme_mode == "light" else "rgba(30,30,50,0.85)",
            bordercolor=theme["grid"], borderwidth=1,
            font=dict(color=theme["text"]),
        )

    # Points expérimentaux
    hover = [
        f"{c:.2e} M → signal={s:.4f}" for c, s in zip(concentrations[mask], signal[mask])
    ]
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name=f"{method_name} — données",
        marker=dict(
            color=color,
            size=10,
            symbol=_METHOD_SYMBOLS.get(method_name, "circle"),
            line=dict(color="white", width=1),
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Calibration {method_name} — log₁₀(signal) vs log₁₀([c])",
        xaxis_title="log₁₀([c] / M)",
        yaxis_title="log₁₀(signal normalisé)",
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)"),
    )
    return apply_theme_to_figure(fig, theme_mode)


def plot_pls_scree(
    explained_variance: np.ndarray,
    method_name: str,
    theme_mode: str = "light",
) -> go.Figure:
    """
    Scree plot : variance expliquée par composante latente PLS.

    Parameters
    ----------
    explained_variance : np.ndarray, shape (n_components,)
        Fraction de variance expliquée par chaque composante (entre 0 et 1).
    method_name : str
        Nom de la méthode (ex. 'B2').
    theme_mode : str
        'light' ou 'dark'.

    Returns
    -------
    go.Figure
        Barres + courbe cumulée.
    """
    explained_variance = np.asarray(explained_variance, dtype=float)
    n = len(explained_variance)
    color = _METHOD_COLORS.get(method_name, "#333333")
    theme = get_theme(theme_mode)

    components = list(range(1, n + 1))
    cumulative = np.cumsum(explained_variance) * 100
    individual = explained_variance * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=components,
        y=individual,
        name="Variance individuelle (%)",
        marker_color=color,
        opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=components,
        y=cumulative,
        mode="lines+markers",
        name="Variance cumulée (%)",
        line=dict(color=theme["text"], width=2, dash="dot"),
        marker=dict(size=8),
    ), secondary_y=True)

    fig.add_hline(y=95, line=dict(color="red", width=1, dash="dash"),
                  annotation_text="95 %", annotation_position="right",
                  secondary_y=True)

    fig.update_layout(
        title=f"Scree plot PLS — {method_name}",
        xaxis=dict(title="Composante latente", tickvals=components, ticktext=[f"C{i}" for i in components]),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(title_text="Variance par composante (%)", secondary_y=False)
    fig.update_yaxes(title_text="Variance cumulée (%)", range=[0, 105], secondary_y=True)

    return apply_theme_to_figure(fig, theme_mode)


def plot_pls_loadings(
    loadings: np.ndarray,
    axis: np.ndarray,
    axis_label: str,
    component: int = 1,
    theme_mode: str = "light",
) -> go.Figure:
    """
    Loadings PLS d'une composante sur l'axe fréquence (EIS) ou potentiel (CV).

    Parameters
    ----------
    loadings : np.ndarray, shape (n_features,)
        Vecteur de loadings de la composante sélectionnée.
    axis : np.ndarray, shape (n_features,)
        Axe physique : fréquences en Hz (EIS) ou potentiels en V (CV).
    axis_label : str
        Label de l'axe x affiché (ex. 'Fréquence (Hz)' ou 'Potentiel (V)').
    component : int
        Numéro de la composante latente (1-indexé, affiché dans le titre).
    theme_mode : str
        'light' ou 'dark'.

    Returns
    -------
    go.Figure
        Courbe des loadings avec zone positive/négative colorée.
    """
    loadings = np.asarray(loadings, dtype=float)
    axis    = np.asarray(axis, dtype=float)
    theme = get_theme(theme_mode)

    # Détecte si l'axe est log (fréquences : décades régulièrement espacées)
    use_log = axis.min() > 0 and (np.log10(axis.max()) - np.log10(axis.min())) > 1

    fig = go.Figure()

    # Zone positive
    fig.add_trace(go.Scatter(
        x=axis,
        y=np.where(loadings >= 0, loadings, 0),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Loadings > 0",
        hoverinfo="skip",
    ))
    # Zone négative
    fig.add_trace(go.Scatter(
        x=axis,
        y=np.where(loadings < 0, loadings, 0),
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Loadings < 0",
        hoverinfo="skip",
    ))
    # Courbe principale
    fig.add_trace(go.Scatter(
        x=axis,
        y=loadings,
        mode="lines",
        name=f"Composante {component}",
        line=dict(color=theme["text"], width=1.5),
    ))
    fig.add_hline(y=0, line=dict(color=theme["grid"], width=1))

    fig.update_layout(
        title=f"Loadings PLS — Composante {component}",
        xaxis=dict(
            title=axis_label,
            type="log" if use_log else "linear",
        ),
        yaxis_title="Loading",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    return apply_theme_to_figure(fig, theme_mode)


def plot_predicted_vs_true(
    results_by_method: dict,
    theme_mode: str = "light",
) -> go.Figure:
    """
    Figure principale de performance prédictive : log(c_pred) vs log(c_true).

    Une couleur et un symbole par méthode, barres d'erreur ± std réplicats,
    droite y = x en pointillés noirs.

    Parameters
    ----------
    results_by_method : dict
        {method_name: {"y_pred": np.ndarray, "y_true": np.ndarray, "y_std": np.ndarray}}
        y_pred, y_true, y_std sont en log10([c]).
        y_std peut être un tableau de zéros si pas d'écart-type disponible.
    theme_mode : str
        'light' ou 'dark'.

    Returns
    -------
    go.Figure
        Figure Plotly interactive.
    """
    theme = get_theme(theme_mode)
    fig = go.Figure()

    # Récupère la plage commune pour la droite y = x
    all_true = np.concatenate([
        np.asarray(v["y_true"]) for v in results_by_method.values()
        if v.get("y_true") is not None and len(v["y_true"]) > 0
    ]) if results_by_method else np.array([-18, -8])

    x_range = [all_true.min() - 0.5, all_true.max() + 0.5] if len(all_true) > 0 else [-18, -8]
    diag = np.array(x_range)

    fig.add_trace(go.Scatter(
        x=diag, y=diag,
        mode="lines",
        name="y = x (parfait)",
        line=dict(color=theme["grid"], width=2, dash="dot"),
        hoverinfo="skip",
    ))

    for method, data in results_by_method.items():
        y_pred = np.asarray(data["y_pred"], dtype=float)
        y_true = np.asarray(data["y_true"], dtype=float)
        y_std  = np.asarray(data.get("y_std", np.zeros_like(y_pred)), dtype=float)

        color  = _METHOD_COLORS.get(method, "#333333")
        symbol = _METHOD_SYMBOLS.get(method, "circle")

        # Labels hover
        hover = [
            f"<b>{method}</b><br>"
            f"c_vraie = 10{_sup(int(round(yt)))} M<br>"
            f"c_pred  = {cp:.2f} (log₁₀)<br>"
            f"σ = {s:.3f} dec"
            for yt, cp, s in zip(y_true, y_pred, y_std)
        ]

        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name=method,
            marker=dict(color=color, size=10, symbol=symbol, line=dict(color="white", width=1)),
            error_y=dict(
                type="data",
                array=y_std,
                visible=True,
                color=color,
                thickness=1.5,
                width=6,
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Performance prédictive — log₁₀(c_pred) vs log₁₀(c_true)",
        xaxis_title="log₁₀([c] vraie / M)",
        yaxis_title="log₁₀([c] prédite / M)",
        legend_title="Méthode",
    )
    return apply_theme_to_figure(fig, theme_mode)


def plot_variance_decomposition(
    metrics_by_method: dict,
    theme_mode: str = "light",
) -> go.Figure:
    """
    Barres groupées : σ_intra / σ_inter_brut / σ_inter_norm par méthode.

    Parameters
    ----------
    metrics_by_method : dict
        {method_name: {"sigma_intra": float, "sigma_inter_raw": float, "sigma_inter_norm": float}}
    theme_mode : str
        'light' ou 'dark'.

    Returns
    -------
    go.Figure
        Barres groupées, une série par type de variabilité.
    """
    theme = get_theme(theme_mode)
    methods = list(metrics_by_method.keys())

    sigma_intra = [metrics_by_method[m].get("sigma_intra", 0) or 0 for m in methods]
    sigma_raw   = [metrics_by_method[m].get("sigma_inter_raw", 0) or 0 for m in methods]
    sigma_norm  = [metrics_by_method[m].get("sigma_inter_norm", 0) or 0 for m in methods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="σ_intra (bruit instrumental)",
        x=methods,
        y=sigma_intra,
        marker_color="#4878d0",
        hovertemplate="<b>σ_intra</b> = %{y:.4f} décades<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="σ_inter brut (avant normalisation)",
        x=methods,
        y=sigma_raw,
        marker_color="#ee854a",
        hovertemplate="<b>σ_inter brut</b> = %{y:.4f} décades<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="σ_inter normalisé (après normalisation probe)",
        x=methods,
        y=sigma_norm,
        marker_color="#6acc65",
        hovertemplate="<b>σ_inter norm</b> = %{y:.4f} décades<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        title="Décomposition de la variance par méthode",
        xaxis_title="Méthode",
        yaxis_title="Variabilité (décades de log₁₀([c]))",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )
    return apply_theme_to_figure(fig, theme_mode)


# ---------------------------------------------------------------------------
# Utilitaire interne
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convertit '#rrggbb' en (r, g, b) entiers 0-255."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
