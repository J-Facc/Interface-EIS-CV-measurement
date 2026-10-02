"""Figures Plotly EIS : Nyquist, Bode, fit du circuit, DRT, calibration.

Aucun import Streamlit, aucun calcul métier : les figures reçoivent des données déjà produites
(``core/pipeline.py``, ``core/results_table.py``).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import EISSession, EISSpectrum
from core.calibration import compute_calibration_all, compute_calibration_loglog_all
from plotting.theme import get_theme, apply_theme_to_figure


import plotly.colors as _pc

_SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup(n: int) -> str:
    return str(n).translate(_SUP_MAP)


def _spectrum_label(sp: EISSpectrum) -> str:
    if sp.step in ("bare", "probe"):
        return sp.step.capitalize()
    c = sp.concentration
    if c <= 0:
        return sp.label
    exp = int(np.floor(np.log10(c)))
    mant = c / 10 ** exp
    return f"[{mant:.1f}×10{_sup(exp)} M]"


# ── Référence « électrode nue » (affichage seul) ──────────────────────────────────────

# Gris atténué par thème pour la trace de référence bare (contraste doux sur
# fond clair comme sombre). Aucun calcul n'en dépend.
_BARE_REF_COLOR = {"light": "#8a8f98", "dark": "#c7ccd4"}


def _add_bare_reference_trace(fig, x, y, theme_mode: str, hovertemplate: str) -> None:
    """Superpose une courbe de référence « électrode nue » (pointillés gris,
    opacité réduite, légende dédiée). AFFICHAGE SEUL — ne calcule rien."""
    color = _BARE_REF_COLOR.get(theme_mode, _BARE_REF_COLOR["light"])
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name="Électrode nue (réf.)",
        line=dict(color=color, dash="dot", width=1.5),
        opacity=0.6,
        hovertemplate=hovertemplate,
    ))


# ── Nyquist ───────────────────────────────────────────────────────────────────────────

def nyquist_figure(
    session: EISSession = None,
    spectrum=None,
    excluded_indices: list = None,
    selection_mode: bool = False,
) -> go.Figure:
    """Nyquist plot. X = Re(Z), Y = -Im(Z) > 0 (upper-right quadrant).

    Can be called with a full EISSession (session=) or a single EISSpectrum
    (spectrum=) for the pretraitement point-editing mode.
    """
    theme = get_theme("light")
    colors = theme["colors"]
    dashes = theme["fit_dash"]
    fig = go.Figure()

    # ── Single-spectrum mode (pretraitement editing) ────────────────────────
    if spectrum is not None:
        sp = spectrum
        color = colors[0]
        lbl = sp.label

        if excluded_indices:
            excl_set = set(excluded_indices)
            keep_idx = [i for i in range(len(sp.f)) if i not in excl_set]
            excl_idx = sorted(excl_set)

            fig.add_trace(go.Scatter(
                x=sp.Zre[keep_idx], y=sp.Zim[keep_idx],
                mode="markers",
                name=f"{lbl} — exp",
                marker=dict(color=color, size=6, symbol="circle"),
                customdata=[[sp.f[i], i] for i in keep_idx],
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    "Re(Z) = %{x:.1f} Ω<br>"
                    "−Im(Z) = %{y:.1f} Ω<br>"
                    "f = %{customdata[0]:.3e} Hz<br>"
                    "index = %{customdata[1]}<extra></extra>"
                ),
            ))
            fig.add_trace(go.Scatter(
                x=sp.Zre[excl_idx], y=sp.Zim[excl_idx],
                mode="markers",
                name="Points exclus",
                marker=dict(color="lightgray", size=6, symbol="x", line=dict(width=1, color="gray")),
                customdata=[[sp.f[i], i] for i in excl_idx],
                hovertemplate=(
                    "<b>Exclu</b><br>"
                    "Re(Z) = %{x:.1f} Ω<br>"
                    "−Im(Z) = %{y:.1f} Ω<br>"
                    "f = %{customdata[0]:.3e} Hz<br>"
                    "index = %{customdata[1]}<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sp.Zre, y=sp.Zim,
                mode="markers",
                name=f"{lbl} — exp",
                marker=dict(color=color, size=6, symbol="circle"),
                customdata=[[f, i] for i, f in enumerate(sp.f)],
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    "Re(Z) = %{x:.1f} Ω<br>"
                    "−Im(Z) = %{y:.1f} Ω<br>"
                    "f = %{customdata[0]:.3e} Hz<br>"
                    "index = %{customdata[1]}<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=f"Nyquist — {lbl}",
            xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
            yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
            hovermode="closest",
        )
        if selection_mode:
            fig.update_layout(
                dragmode="select",
                clickmode="event+select",
            )
        apply_theme_to_figure(fig, "light")
        return fig

    # ── Full session mode ───────────────────────────────────────────────
    all_spectra: list = []
    if session.bare:
        all_spectra.append(session.bare)
    if session.probe:
        all_spectra.append(session.probe)
    for grp in session.groups:
        all_spectra.append(grp.spectrum)

    for ci, sp in enumerate(all_spectra):
        color = colors[ci % len(colors)]
        lbl = _spectrum_label(sp)
        fig.add_trace(go.Scatter(
            x=sp.Zre, y=sp.Zim, mode="markers",
            name=f"{lbl} — exp",
            marker=dict(color=color, size=6, symbol="circle"),
            customdata=sp.f,
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "Re(Z) = %{x:.1f} Ω<br>"
                "−Im(Z) = %{y:.1f} Ω<br>"
                "f = %{customdata:.3e} Hz<extra></extra>"
            ),
        ))

    base_offset = len(all_spectra) - len(session.groups)
    for gi, grp in enumerate(session.groups):
        color = colors[(base_offset + gi) % len(colors)]
        lbl = _spectrum_label(grp.spectrum)
        for di, (model_name, fr) in enumerate(grp.fit_results.items()):
            dash = dashes[di % len(dashes)]
            fig.add_trace(go.Scatter(
                x=fr.Zfit_re, y=fr.Zfit_im, mode="lines",
                name=f"{lbl} — {model_name}",
                line=dict(color=color, dash=dash, width=2),
                hovertemplate=(
                    f"<b>{lbl} — {model_name}</b><br>"
                    "Re(Z) = %{x:.1f} Ω<br>"
                    "−Im(Z) = %{y:.1f} Ω<extra></extra>"
                ),
            ))

    fig.update_layout(
        title="Diagramme de Nyquist",
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top"),
        hovermode="closest",
    )
    apply_theme_to_figure(fig, "light")
    return fig


# ── Nyquist par électrode ─────────────────────────────────────────────────────────

def _conc_color(concentration: float, c_min: float, c_max: float) -> str:
    """Couleur log-interpolée violet→rouge pour une concentration donnée."""
    if c_min > 0 and c_max > 0 and c_min < c_max:
        t = (np.log10(concentration) - np.log10(c_min)) / (np.log10(c_max) - np.log10(c_min))
    else:
        t = 0.5
    t = float(np.clip(t, 0, 1))
    return _pc.sample_colorscale("plasma", [t])[0]


def nyquist_figure_electrode(
    spectra: list,
    title: str = "",
    bare=None,
    theme_mode: str = "light",
) -> go.Figure:
    """Nyquist pour une électrode : une trace par concentration, couleur log-scale.

    Parameters
    ----------
    spectra : list de dicts avec clés "label", "Zre", "Zim", "concentration".
              Passer le probe avec concentration=0 pour l'afficher en noir.
    title   : titre du graphique.
    bare    : EISSpectrum optionnel (référence « électrode nue »). AFFICHAGE SEUL
              — simplement superposé, aucun calcul. None = pas de superposition.
    theme_mode : 'light' ou 'dark' (thème jour/nuit).
    """
    fig = go.Figure()

    pos = [s for s in spectra if s["concentration"] > 0]
    c_vals = [s["concentration"] for s in pos]
    c_min  = min(c_vals) if c_vals else 1e-12
    c_max  = max(c_vals) if c_vals else 1e-8

    for s in spectra:
        conc = s["concentration"]
        Zre  = np.asarray(s["Zre"])
        Zim  = np.asarray(s["Zim"])
        lbl  = s["label"]

        if conc <= 0:
            color = "black"
        else:
            color = _conc_color(conc, c_min, c_max)

        fig.add_trace(go.Scatter(
            x=Zre, y=Zim,
            mode="markers",
            name=lbl,
            marker=dict(color=color, size=5, symbol="circle"),
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "Re(Z) = %{x:.1f} Ω<br>"
                "−Im(Z) = %{y:.1f} Ω<extra></extra>"
            ),
        ))

    # Référence « électrode nue » — superposition d'affichage seule (pointillés,
    # gris atténué, opacité réduite). Ne dérive aucun calcul.
    if bare is not None:
        _add_bare_reference_trace(
            fig, np.asarray(bare.Zre), np.asarray(bare.Zim), theme_mode,
            hovertemplate=(
                "<b>Électrode nue (réf.)</b><br>"
                "Re(Z) = %{x:.1f} Ω<br>"
                "−Im(Z) = %{y:.1f} Ω<extra></extra>"
            ),
        )

    fig.update_layout(
        title=title or "Diagramme de Nyquist",
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        hovermode="closest",
        margin=dict(r=120),
    )
    apply_theme_to_figure(fig, theme_mode)
    return fig


def nyquist_normalized_figure(
    spectra: list,
    title: str = "",
) -> go.Figure:
    """Nyquist normalisé : trace Zre_norm vs Zim_norm pour chaque concentration.

    Parameters
    ----------
    spectra : list de dicts avec clés "label", "Zre_norm", "Zim_norm", "concentration".
    title   : titre du graphique.
    """
    fig = go.Figure()

    c_vals = [s["concentration"] for s in spectra if s["concentration"] > 0]
    c_min  = min(c_vals) if c_vals else 1e-12
    c_max  = max(c_vals) if c_vals else 1e-8

    for s in spectra:
        conc     = s["concentration"]
        Zre_norm = np.asarray(s["Zre_norm"])
        Zim_norm = np.asarray(s["Zim_norm"])
        lbl      = s["label"]
        color    = _conc_color(conc, c_min, c_max)

        fig.add_trace(go.Scatter(
            x=Zre_norm, y=Zim_norm,
            mode="markers",
            name=lbl,
            marker=dict(color=color, size=5, symbol="circle"),
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "|ΔRe(Z)/Re(Z)_probe| = %{x:.3f}<br>"
                "|ΔIm(Z)/Im(Z)_probe| = %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title or "Diagramme de Nyquist normalisé",
        xaxis=dict(title="|ΔRe(Z)/Re(Z)_probe|", rangemode="tozero"),
        yaxis=dict(title="|ΔIm(Z)/Im(Z)_probe|", rangemode="tozero"),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        hovermode="closest",
        margin=dict(r=120),
    )
    apply_theme_to_figure(fig, "light")
    return fig


# ── DRT ───────────────────────────────────────────────────────────────────────────

def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convertit une couleur hex (#rrggbb) en rgba() pour les bandes semi-opaques."""
    c = color.lstrip("#")
    if len(c) == 6:
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return color


# Normalisation dimensionnelle du tracé DRT (convention impérative) : on trace
# ln(γ/γ₀) en fonction de ln(τ/τ₀). γ₀ et τ₀ valent 1 (ils ne changent donc pas la
# valeur numérique) mais restent EXPLICITES pour adimensionnaliser l'argument du
# logarithme et rendre la normalisation lisible dans le code et sur les axes.
DRT_GAMMA0_OHM = 1.0   # γ₀ = 1 Ω
DRT_TAU0_S = 1.0       # τ₀ = 1 s

# Libellés lisibles des deux modes DRT (badge + légende).
_DRT_MODE_LABEL = {
    "optimize": "DRT MAP (optimize)",
    "sample": "DRT bayésienne (sample)",
}


def _drt_mode_of(res) -> str:
    """Mode DRT d'un FitResult ('optimize'/'sample'), défaut 'optimize'."""
    mode = getattr(res, "drt_mode", None)
    return mode if mode in ("optimize", "sample") else "optimize"


def drt_figure(
    results,
    title: str = "Distribution des temps de relaxation (DRT)",
    theme_mode: str = "light",
) -> go.Figure:
    """Trace ln(γ(τ)/γ₀) vs ln(τ/τ₀) pour un ou plusieurs FitResult DRT.

    Convention de tracé (impérative) :

    * Logarithme **népérien** (base e, ``np.log``) — jamais ``log10``.
    * Abscisse ``ln(τ/τ₀)`` avec ``τ₀ = 1 s`` ; ordonnée ``ln(γ/γ₀)`` avec
      ``γ₀ = 1 Ω``. γ₀/τ₀ sont explicites (adimensionnalisation de l'argument du
      log) même s'ils valent 1.
    * En mode 'sample', la bande d'incertitude (γ_lo/γ_hi, 2.5/97.5 %) est
      transformée de la même façon et **identifiée explicitement** dans la légende.

    Un **badge de mode** (« DRT MAP (optimize) » vs « DRT bayésienne (sample) »)
    est affiché pour que l'utilisateur ne compare jamais sans le savoir des DRT
    calculées par deux modes différents.

    Args:
        results: itérable de tuples ``(label, FitResult)``. Un FitResult seul ou un
            unique tuple sont aussi acceptés par commodité.
        title: titre de la figure.
        theme_mode: 'light' ou 'dark' (thème jour/nuit, cf. plotting/theme.py).
    """
    items = _coerce_drt_items(results)

    theme = get_theme(theme_mode)
    colors = theme["colors"]
    fig = go.Figure()

    plotted = 0
    modes_seen = set()
    for idx, (lbl, res) in enumerate(items):
        if res is None:
            continue
        tau = getattr(res, "drt_tau", None)
        gamma = getattr(res, "drt_gamma", None)
        if tau is None or gamma is None or len(tau) == 0:
            continue

        color = colors[idx % len(colors)]
        tau_arr = np.asarray(tau, dtype=float)
        gamma_arr = np.asarray(gamma, dtype=float)
        mode = _drt_mode_of(res)
        modes_seen.add(mode)

        # ln(0) et ln(<0) sont indéfinis ; on masque γ <= 0 / τ <= 0 avant le ln.
        valid = (gamma_arr > 0) & (tau_arr > 0)
        if not np.any(valid):
            continue
        x = np.log(tau_arr[valid] / DRT_TAU0_S)        # ln(τ/τ₀), τ₀ = 1 s
        y = np.log(gamma_arr[valid] / DRT_GAMMA0_OHM)  # ln(γ/γ₀), γ₀ = 1 Ω

        # Bande d'incertitude bayésienne (mode 'sample' uniquement), transformée
        # de la même façon (bornes ln(γ_lo/γ₀), ln(γ_hi/γ₀)).
        gamma_lo = getattr(res, "drt_gamma_lo", None)
        gamma_hi = getattr(res, "drt_gamma_hi", None)
        if gamma_lo is not None and gamma_hi is not None:
            lo = np.asarray(gamma_lo, dtype=float)
            hi = np.asarray(gamma_hi, dtype=float)
            band = valid & (lo > 0) & (hi > 0)
            if np.any(band):
                xb = np.log(tau_arr[band] / DRT_TAU0_S)
                lo_b = np.log(lo[band] / DRT_GAMMA0_OHM)
                hi_b = np.log(hi[band] / DRT_GAMMA0_OHM)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xb, xb[::-1]]),
                    y=np.concatenate([hi_b, lo_b[::-1]]),
                    fill="toself", fillcolor=_hex_to_rgba(color, 0.18),
                    line=dict(width=0), hoverinfo="skip",
                    name=f"{lbl} — IC 95 % (sample, bayésien)", showlegend=True,
                ))

        suffix = " — MAP (optimize)" if mode == "optimize" else " — sample (bayésien)"
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=f"{lbl}{suffix}",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{lbl}</b> [{mode}]<br>"
                "ln(τ/τ₀) = %{x:.3f}<br>"
                "ln(γ/γ₀) = %{y:.4g}<extra></extra>"
            ),
        ))
        plotted += 1

    if plotted == 0:
        fig.add_annotation(
            text="Aucune DRT disponible — lancez l'analyse.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, theme_mode)
        return fig

    # Badge de mode : un seul mode → libellé plein ; modes mixtes → avertissement.
    if len(modes_seen) == 1:
        badge = _DRT_MODE_LABEL[next(iter(modes_seen))]
    else:
        badge = "⚠ Modes DRT mixtes (voir légende)"
    fig.add_annotation(
        text=badge, xref="paper", yref="paper", x=0.0, y=1.08,
        xanchor="left", showarrow=False,
        font=dict(size=12, color=theme["text"]),
        bgcolor=_hex_to_rgba(theme["paper_bg"], 0.7),
        bordercolor=theme["grid"], borderwidth=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title=r"$\ln(\tau/\tau_0)\;\;[\tau_0 = 1\,\mathrm{s}]$",
        yaxis_title=r"$\ln(\gamma/\gamma_0)\;\;[\gamma_0 = 1\,\Omega]$",
        showlegend=True,
        legend=dict(
            orientation="v", x=1.02, xanchor="left", y=1.0,
            font=dict(color=theme["text"]),
            bgcolor=_hex_to_rgba(theme["paper_bg"], 0.6),
            bordercolor=theme["grid"], borderwidth=1,
        ),
        hovermode="closest",
    )
    apply_theme_to_figure(fig, theme_mode)
    return fig


def _coerce_drt_items(results):
    """Normalise l'entrée de drt_figure en liste de (label, FitResult)."""
    if results is None:
        return []
    # FitResult seul (porte drt_tau/drt_gamma, pas une itération de tuples).
    if hasattr(results, "drt_tau") and hasattr(results, "drt_gamma"):
        return [("DRT", results)]
    # dict {label: result}
    if isinstance(results, dict):
        return list(results.items())
    # tuple unique (label, result)
    if isinstance(results, tuple) and len(results) == 2 and isinstance(results[0], str):
        return [results]
    # itérable de (label, result)
    return list(results)


# ── Calibration ─────────────────────────────────────────────────────────────

def calibration_figure(session: EISSession) -> go.Figure:
    """Calibration: one regression curve per model + metrics table below.

    Returns a single Figure with two vertically stacked subplots:
      row 1 (scatter): signal_norm vs log([c]), one series per model
      row 2 (table):   R², slope, intercept, p-value, std_err per model
    """
    theme = get_theme("light")
    colors = theme["colors"]

    probe_fr = getattr(session.probe, "fit_results", {}) if session.probe else {}
    if not probe_fr:
        fig = go.Figure()
        fig.add_annotation(
            text="Spectre probe requis (avec fit) pour la calibration normalisée.",
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    dashes = ["solid", "dash", "dot", "dashdot"]

    # Calcul délégué à core/calibration.py : source unique partagée avec
    # export_calibration_csv (pente/ordonnée/R² identiques figure ↔ export).
    model_data = []
    for mi, cal in enumerate(compute_calibration_all(session)):
        model_data.append({
            "model": cal.model,
            "log_c": cal.log_c,
            "signals": cal.y,
            "slope": cal.slope,
            "intercept": cal.intercept,
            "r2": cal.r2,
            "pvalue": cal.pvalue,
            "stderr": cal.stderr,
            "color": colors[mi % len(colors)],
            "dash": dashes[mi % len(dashes)],
        })

    if not model_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Pas assez de points (min. 2 concentrations positives avec fit probe).",
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    # Build figure: scatter subplot + table subplot
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.35, 0.65],
        horizontal_spacing=0.06,
        specs=[[{"type": "table"}, {"type": "xy"}]],
    )

    for d in model_data:
        log_c = d["log_c"]
        log_c_line = np.linspace(log_c.min(), log_c.max(), 200)
        y_line = d["slope"] * log_c_line + d["intercept"]
        sign = "+" if d["intercept"] >= 0 else "-"
        legend_label = (
            f"{d['model']}  "
            f"R²={d['r2']:.3f}  "
            f"y={d['slope']:.3f}x {sign} {abs(d['intercept']):.3f}"
        )

        # Data points
        fig.add_trace(go.Scatter(
            x=d["log_c"], y=d["signals"],
            mode="markers",
            name=d["model"],
            legendgroup=d["model"],
            marker=dict(color=d["color"], size=9, symbol="circle"),
            hovertemplate="log([c]) = %{x:.2f}<br>Signal = %{y:.4f}<extra></extra>",
            showlegend=True,
        ), row=1, col=2)

        # Regression line
        fig.add_trace(go.Scatter(
            x=log_c_line, y=y_line,
            mode="lines",
            name=legend_label,
            legendgroup=d["model"],
            line=dict(color=d["color"], dash=d["dash"], width=2),
            showlegend=True,
        ), row=1, col=2)

    # Metrics table
    n = len(model_data)
    row_colors = ["#EEF0F8" if i % 2 == 0 else "#FFFFFF" for i in range(n)]
    fig.add_trace(go.Table(
        header=dict(
            values=["Modèle", "R²", "Pente", "Intercept", "p-value", "Std err"],
            fill_color="#4472C4",
            font=dict(color="white", size=11),
            align="center",
        ),
        cells=dict(
            values=[
                [d["model"] for d in model_data],
                [f"{d['r2']:.4f}" for d in model_data],
                [f"{d['slope']:.4f}" for d in model_data],
                [f"{d['intercept']:.4f}" for d in model_data],
                [f"{d['pvalue']:.2e}" for d in model_data],
                [f"{d['stderr']:.4f}" for d in model_data],
            ],
            fill_color=[row_colors] * 6,
            align="center",
            font=dict(size=10),
        ),
    ), row=1, col=1)

    fig.update_xaxes(title_text="log([c] / M)", row=1, col=2)
    fig.update_yaxes(title_text="|Rct_probe − Rct_c| / |Rct_probe|", row=1, col=2)
    fig.update_layout(
        title="Calibration EIS — Signal normalisé vs log([c])",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        hovermode="closest",
    )
    apply_theme_to_figure(fig, "light")
    return fig


# ── Visualisation : tous les réplicats + moyenne (Nyquist, Bode) ─────────────────────

def _group_color(group, c_min: float, c_max: float) -> str:
    """Probe en noir ; bare en gris ; concentrations en dégradé log (comme Nyquist/électrode)."""
    if group.concentration and group.concentration > 0:
        if c_min < c_max:
            t = (np.log10(group.concentration) - np.log10(c_min)) / (np.log10(c_max) - np.log10(c_min))
        else:
            t = 0.5
        # Plasma tronqué à 85 % : son extrémité jaune est illisible sur fond clair.
        return _pc.sample_colorscale("plasma", [0.85 * float(np.clip(t, 0, 1))])[0]
    return "#6b7280" if group.step == "bare" else "black"


def _concentration_range(groups: list) -> tuple:
    c = [g.concentration for g in groups if g.concentration and g.concentration > 0]
    return (min(c), max(c)) if c else (1e-12, 1e-8)


def _phase_deg(sp) -> np.ndarray:
    """−φ en degrés, φ = arg Z, Z = Zre − j·Zim : positif pour un comportement capacitif."""
    return np.degrees(np.arctan2(np.asarray(sp.Zim, dtype=float), np.asarray(sp.Zre, dtype=float)))


def _modulus(sp) -> np.ndarray:
    return np.hypot(np.asarray(sp.Zre, dtype=float), np.asarray(sp.Zim, dtype=float))


def nyquist_replicates_figure(
    groups: list,
    bare=None,
    show_replicates: bool = True,
    title: str = "Diagramme de Nyquist",
    theme_mode: str = "light",
) -> go.Figure:
    """Nyquist d'une électrode : pour chaque groupe, ses réplicats (traits fins) et sa moyenne.

    Args:
        groups: ``core.models.DisplayGroup``. Une trace de légende par groupe : la
            cliquer masque aussi les réplicats de ce groupe (``legendgroup``).
        bare: EISSpectrum de référence « électrode nue » (affichage seul), ou None.
        show_replicates: False = moyennes seules.
    """
    fig = go.Figure()
    c_min, c_max = _concentration_range(groups)
    for g in groups:
        color = _group_color(g, c_min, c_max)
        has_mean = g.mean is not None
        if show_replicates or not has_mean:
            for i, sp in enumerate(g.replicates):
                fig.add_trace(go.Scatter(
                    x=sp.Zre, y=sp.Zim, mode="lines+markers", legendgroup=g.label,
                    name=f"{g.label} — réplicat {i + 1}",
                    showlegend=(not has_mean and i == 0),
                    marker=dict(color=color, size=4), line=dict(color=color, width=1),
                    opacity=0.45, customdata=sp.f,
                    hovertemplate=(f"<b>{g.label} — réplicat {i + 1}</b><br>Re(Z) = %{{x:.1f}} Ω<br>"
                                   "−Im(Z) = %{y:.1f} Ω<br>f = %{customdata:.3e} Hz<extra></extra>"),
                ))
        if has_mean:
            sp = g.mean
            n = len(g.replicates)
            suffix = f" (moyenne de {n})" if show_replicates and n > 1 else ""
            fig.add_trace(go.Scatter(
                x=sp.Zre, y=sp.Zim, mode="lines+markers", legendgroup=g.label,
                name=g.label + suffix, marker=dict(color=color, size=7),
                line=dict(color=color, width=2), customdata=sp.f,
                hovertemplate=(f"<b>{g.label}</b> (moyenne)<br>Re(Z) = %{{x:.1f}} Ω<br>"
                               "−Im(Z) = %{y:.1f} Ω<br>f = %{customdata:.3e} Hz<extra></extra>"),
            ))
    if bare is not None:
        _add_bare_reference_trace(
            fig, np.asarray(bare.Zre), np.asarray(bare.Zim), theme_mode,
            hovertemplate=("<b>Électrode nue (réf.)</b><br>Re(Z) = %{x:.1f} Ω<br>"
                           "−Im(Z) = %{y:.1f} Ω<extra></extra>"))
    fig.update_layout(
        title=title, xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        hovermode="closest", margin=dict(r=120), height=480,
    )
    apply_theme_to_figure(fig, theme_mode)
    return fig


def bode_figure(
    groups: list,
    bare=None,
    show_replicates: bool = True,
    title: str = "Diagramme de Bode",
    theme_mode: str = "light",
) -> go.Figure:
    """Bode d'une électrode : |Z| (haut) et −phase (bas) vs fréquence, mêmes conventions
    (couleurs, légende par groupe, réplicats + moyenne) que :func:`nyquist_replicates_figure`.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["|Z| (Ω)", "−phase (°)"])
    c_min, c_max = _concentration_range(groups)

    def _add(sp, g, color, name, mean: bool, show_leg: bool):
        for row, y, unit in ((1, _modulus(sp), "|Z| = %{y:.4g} Ω"),
                             (2, _phase_deg(sp), "−φ = %{y:.2f}°")):
            fig.add_trace(go.Scatter(
                x=sp.f, y=y, mode="lines+markers", legendgroup=g.label, name=name,
                showlegend=(show_leg and row == 1),
                marker=dict(color=color, size=7 if mean else 4),
                line=dict(color=color, width=2 if mean else 1),
                opacity=1.0 if mean else 0.45,
                hovertemplate=f"<b>{name}</b><br>f = %{{x:.3e}} Hz<br>{unit}<extra></extra>",
            ), row=row, col=1)

    for g in groups:
        color = _group_color(g, c_min, c_max)
        has_mean = g.mean is not None
        if show_replicates or not has_mean:
            for i, sp in enumerate(g.replicates):
                _add(sp, g, color, f"{g.label} — réplicat {i + 1}", False, not has_mean and i == 0)
        if has_mean:
            n = len(g.replicates)
            suffix = f" (moyenne de {n})" if show_replicates and n > 1 else ""
            _add(g.mean, g, color, g.label + suffix, True, True)
    if bare is not None:
        for row, y in ((1, _modulus(bare)), (2, _phase_deg(bare))):
            color = _BARE_REF_COLOR.get(theme_mode, _BARE_REF_COLOR["light"])
            fig.add_trace(go.Scatter(
                x=bare.f, y=y, mode="lines", name="Électrode nue (réf.)", opacity=0.6,
                line=dict(color=color, dash="dot", width=1.5), showlegend=(row == 1),
                hovertemplate="<b>Électrode nue (réf.)</b><br>f = %{x:.3e} Hz<extra></extra>",
            ), row=row, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Fréquence (Hz)", row=2, col=1)
    fig.update_layout(title=title, legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
                      hovermode="closest", margin=dict(r=120), height=480)
    apply_theme_to_figure(fig, theme_mode)
    # apply_theme_to_figure réécrit xaxis/yaxis de la 1re ligne : on rétablit les échelles log.
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", tickformat=".4~g", row=1, col=1)
    fig.update_xaxes(type="log", row=2, col=1)
    return fig


# ── Fit du circuit : Nyquist expérimental + courbe ajustée, résidus ──────────────────

def fit_nyquist_figure(sp, fr, title: str = "", theme_mode: str = "light") -> go.Figure:
    """Nyquist expérimental (points) et courbe du circuit ajusté (trait), pour UN spectre.

    ``sp`` et ``fr`` doivent porter les MÊMES points (``fr.Zfit_re`` aligné sur ``sp.f``).
    """
    theme = get_theme(theme_mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sp.Zre, y=sp.Zim, mode="markers", name="Expérience", customdata=sp.f,
        marker=dict(color=theme["colors"][0], size=7),
        hovertemplate=("<b>Expérience</b><br>Re(Z) = %{x:.1f} Ω<br>−Im(Z) = %{y:.1f} Ω<br>"
                       "f = %{customdata:.3e} Hz<extra></extra>")))
    fig.add_trace(go.Scatter(
        x=fr.Zfit_re, y=fr.Zfit_im, mode="lines", name="Circuit ajusté", customdata=sp.f,
        line=dict(color=theme["colors"][3], width=2.5),
        hovertemplate=("<b>Circuit ajusté</b><br>Re(Z) = %{x:.1f} Ω<br>−Im(Z) = %{y:.1f} Ω<br>"
                       "f = %{customdata:.3e} Hz<extra></extra>")))
    fig.update_layout(
        title=title or "Nyquist — expérience vs circuit ajusté",
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="h", x=0.0, y=-0.2), hovermode="closest", height=420)
    apply_theme_to_figure(fig, theme_mode)
    return fig


def fit_residuals_figure(sp, fr, title: str = "", theme_mode: str = "light") -> go.Figure:
    """Résidus du fit (données − modèle) en % de |Z|, parties réelle et imaginaire, vs f."""
    theme = get_theme(theme_mode)
    z = np.maximum(_modulus(sp), 1e-300)
    fig = go.Figure()
    for name, res, color in (("Re(Z)", fr.residuals_re, theme["colors"][0]),
                             ("−Im(Z)", fr.residuals_im, theme["colors"][1])):
        fig.add_trace(go.Scatter(
            x=sp.f, y=100.0 * np.asarray(res, dtype=float) / z, mode="markers+lines", name=name,
            marker=dict(color=color, size=5), line=dict(color=color, width=1),
            hovertemplate=f"<b>{name}</b><br>f = %{{x:.3e}} Hz<br>résidu = %{{y:.2f}} %<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=theme["text"], width=0.5))
    fig.update_layout(
        title=title or "Résidus du fit (% de |Z|)", xaxis=dict(title="Fréquence (Hz)", type="log"),
        yaxis=dict(title="(données − modèle) / |Z| (%)"),
        legend=dict(orientation="h", x=0.0, y=-0.2), hovermode="closest", height=420)
    apply_theme_to_figure(fig, theme_mode)
    fig.update_xaxes(type="log")
    return fig


# ── DRT : vue agrégée (variabilité inter-réplicats) ──────────────────────────────────

def drt_aggregate_figure(
    envelope: dict,
    replicate_items: list,
    mean_item=None,
    title: str = "DRT agrégée",
    theme_mode: str = "light",
) -> go.Figure:
    """γ(τ) agrégée sur les réplicats : courbe moyenne et bande de variabilité inter-réplicats.

    Même convention de tracé que :func:`drt_figure` (ln γ/γ₀ vs ln τ/τ₀). Deux notions
    d'incertitude NE SONT PAS confondues : la bande LARGE (ici) est l'étendue min–max des
    réplicats, la variabilité expérimentale ; la bande fine de crédibilité HMC d'UN
    réplicat est tracée par :func:`drt_figure` sur la vue du réplicat.

    Args:
        envelope: ``core.results_table.drt_replicate_envelope`` (tau, mean, lo, hi, n).
        replicate_items: [(label, FitResult DRT)] des réplicats, tracés en traits fins.
        mean_item: (label, FitResult DRT) du spectre moyen, tracé en pointillés (légende
            seule : cliquer pour l'afficher), ou None.
    """
    theme = get_theme(theme_mode)
    color = theme["colors"][0]
    fig = go.Figure()
    tau = np.asarray(envelope["tau"], dtype=float)
    lo, hi, mean = (np.asarray(envelope[k], dtype=float) for k in ("lo", "hi", "mean"))
    band = (tau > 0) & (lo > 0) & (hi > 0)
    if np.any(band):
        xb = np.log(tau[band] / DRT_TAU0_S)
        fig.add_trace(go.Scatter(
            x=np.concatenate([xb, xb[::-1]]),
            y=np.concatenate([np.log(hi[band] / DRT_GAMMA0_OHM), np.log(lo[band][::-1] / DRT_GAMMA0_OHM)]),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.18), line=dict(width=0), hoverinfo="skip",
            name=f"Variabilité inter-réplicats (min–max, n = {envelope['n']})"))
    for i, (lbl, fr) in enumerate(replicate_items):
        t = np.asarray(fr.drt_tau, dtype=float)
        g = np.asarray(fr.drt_gamma, dtype=float)
        ok = (t > 0) & (g > 0)
        fig.add_trace(go.Scatter(
            x=np.log(t[ok] / DRT_TAU0_S), y=np.log(g[ok] / DRT_GAMMA0_OHM), mode="lines", name=lbl,
            line=dict(color=theme["colors"][(i + 1) % len(theme["colors"])], width=1), opacity=0.7,
            hovertemplate=f"<b>{lbl}</b><br>ln(τ/τ₀) = %{{x:.3f}}<br>ln(γ/γ₀) = %{{y:.4g}}<extra></extra>"))
    ok = (tau > 0) & (mean > 0)
    fig.add_trace(go.Scatter(
        x=np.log(tau[ok] / DRT_TAU0_S), y=np.log(mean[ok] / DRT_GAMMA0_OHM), mode="lines",
        name="Moyenne des réplicats", line=dict(color=color, width=3),
        hovertemplate="<b>Moyenne des réplicats</b><br>ln(τ/τ₀) = %{x:.3f}<br>ln(γ/γ₀) = %{y:.4g}<extra></extra>"))
    if mean_item is not None and mean_item[1] is not None and mean_item[1].drt_tau is not None:
        lbl, fr = mean_item
        t = np.asarray(fr.drt_tau, dtype=float)
        g = np.asarray(fr.drt_gamma, dtype=float)
        ok = (t > 0) & (g > 0)
        fig.add_trace(go.Scatter(
            x=np.log(t[ok] / DRT_TAU0_S), y=np.log(g[ok] / DRT_GAMMA0_OHM), mode="lines",
            name="DRT du spectre moyen", visible="legendonly",
            line=dict(color=theme["colors"][3], width=2, dash="dash")))
    fig.update_layout(
        title=title, xaxis_title=r"$\ln(\tau/\tau_0)\;\;[\tau_0 = 1\,\mathrm{s}]$",
        yaxis_title=r"$\ln(\gamma/\gamma_0)\;\;[\gamma_0 = 1\,\Omega]$",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0,
                    font=dict(color=theme["text"]), bgcolor=_hex_to_rgba(theme["paper_bg"], 0.6),
                    bordercolor=theme["grid"], borderwidth=1),
        hovermode="closest", margin=dict(r=140))
    apply_theme_to_figure(fig, theme_mode)
    return fig


# ── Calibration : log10(Rct) vs log10([c]), un jeu de points par méthode ─────────────

def calibration_loglog_figure(session: EISSession, theme_mode: str = "light") -> go.Figure:
    """log10(valeur cible) vs log10([c]) pour chaque méthode (circuit Orazem, DRT…), avec
    régression et résidus. Régressions de ``core.calibration.compute_calibration_loglog_all``
    (source unique, points retenus seulement) : cette figure ne calcule aucune régression.
    """
    theme = get_theme(theme_mode)
    cals = compute_calibration_loglog_all(session)
    if not cals:
        fig = go.Figure()
        fig.add_annotation(text="Pas assez de points retenus (min. 2 concentrations positives).",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13))
        apply_theme_to_figure(fig, theme_mode)
        return fig
    fig = make_subplots(rows=1, cols=2, subplot_titles=["log(Rct) vs log([c])", "Résidus de régression"])
    for i, cal in enumerate(cals):
        color = theme["colors"][i % len(theme["colors"])]
        x_line = np.linspace(cal.log_c.min(), cal.log_c.max(), 200)
        # d(log10 R) = (dR/R)/ln10 : l'erreur de reconstruction relative donne la barre d'erreur.
        fig.add_trace(go.Scatter(
            x=cal.log_c, y=cal.y, mode="markers", name=cal.model, legendgroup=cal.model,
            marker=dict(color=color, size=9),
            error_y=dict(type="data", array=np.asarray(cal.errs) / np.log(10.0), visible=True),
            hovertemplate="log([c]) = %{x:.2f}<br>log(Rct) = %{y:.4f}<extra></extra>"), row=1, col=1)
        sign = "+" if cal.intercept >= 0 else "−"
        fig.add_trace(go.Scatter(
            x=x_line, y=cal.slope * x_line + cal.intercept, mode="lines", legendgroup=cal.model,
            name=f"{cal.model} — R²={cal.r2:.3f}, y={cal.slope:.3f}x {sign} {abs(cal.intercept):.3f}",
            line=dict(color=color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=cal.log_c, y=cal.y - (cal.slope * cal.log_c + cal.intercept), mode="markers",
            name=f"{cal.model} — résidus", legendgroup=cal.model, showlegend=False,
            marker=dict(color=color, size=9)), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), row=1, col=2)
    fig.update_xaxes(title_text="log([c] / M)")
    fig.update_yaxes(title_text="log(Rct / Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Résidu", row=1, col=2)
    fig.update_layout(title="Calibration log-log — log(Rct) vs log([c])",
                      legend=dict(orientation="h", y=-0.25), hovermode="closest")
    apply_theme_to_figure(fig, theme_mode)
    return fig
