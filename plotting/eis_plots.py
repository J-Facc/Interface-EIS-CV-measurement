"""All EIS Plotly figures: Nyquist, Bode, DRT, parameter table, calibration."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import EISSession, EISSpectrum
from plotting.theme import get_theme, apply_theme_to_figure


# ── Label helpers ──────────────────────────────────────────────────────────────

_SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup(n: int) -> str:
    """Convert integer to unicode superscript string."""
    return str(n).translate(_SUP_MAP)


def _spectrum_label(sp: EISSpectrum) -> str:
    """Human-readable label for a spectrum used in legends and titles."""
    if sp.step in ("bare", "probe"):
        return sp.step.capitalize()
    c = sp.concentration
    if c <= 0:
        return sp.label
    exp = int(np.floor(np.log10(c)))
    mant = c / 10 ** exp
    return f"[{mant:.1f}×10{_sup(exp)} M]"


# ── Nyquist ────────────────────────────────────────────────────────────────────

def nyquist_figure(session: EISSession) -> go.Figure:
    """Build interactive Nyquist plot with experimental data and fit overlays.

    Legend:
    - One entry per experimental spectrum: "<label> — exp"
    - One entry per fit curve: "<label> — <model>"
    Same colour per spectrum; line style varies per model.

    Args:
        session: EISSession with spectra and fit_results.

    Returns:
        Plotly Figure.
    """
    theme = get_theme("light")
    colors = theme["colors"]
    dashes = theme["fit_dash"]

    fig = go.Figure()

    # Collect all spectra in display order
    all_spectra: list = []
    if session.bare:
        all_spectra.append(session.bare)
    if session.probe:
        all_spectra.append(session.probe)
    for grp in session.groups:
        all_spectra.append(grp.spectrum)

    # Experimental scatter traces
    for ci, sp in enumerate(all_spectra):
        color = colors[ci % len(colors)]
        lbl = _spectrum_label(sp)

        fig.add_trace(go.Scatter(
            x=sp.Zre,
            y=-sp.Zim,
            mode="markers",
            name=f"{lbl} — exp",
            marker=dict(color=color, size=6, symbol="circle"),
            customdata=sp.f,
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "Zre = %{x:.1f} Ω<br>"
                "-Zim = %{y:.1f} Ω<br>"
                "f = %{customdata:.3e} Hz<extra></extra>"
            ),
        ))

    # Fit overlay traces
    base_offset = len(all_spectra) - len(session.groups)
    for gi, grp in enumerate(session.groups):
        color = colors[(base_offset + gi) % len(colors)]
        lbl = _spectrum_label(grp.spectrum)

        for di, (model_name, fr) in enumerate(grp.fit_results.items()):
            dash = dashes[di % len(dashes)]
            fig.add_trace(go.Scatter(
                x=fr.Zfit_re,
                y=-fr.Zfit_im,
                mode="lines",
                name=f"{lbl} — {model_name}",
                line=dict(color=color, dash=dash, width=2),
                hovertemplate=(
                    f"<b>{lbl} — {model_name}</b><br>"
                    "Zre = %{x:.1f} Ω<br>"
                    "-Zim = %{y:.1f} Ω<extra></extra>"
                ),
            ))

    fig.update_layout(
        title="Diagramme de Nyquist",
        xaxis_title="Z' (Ω)",
        yaxis_title="−Z'' (Ω)",
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0, yanchor="top",
        ),
        hovermode="closest",
    )
    apply_theme_to_figure(fig, "light")
    return fig


# ── Bode ───────────────────────────────────────────────────────────────────────

def bode_figure(session: EISSession) -> go.Figure:
    """Build Bode plot: |Z| and phase vs frequency.

    Args:
        session: EISSession.

    Returns:
        Plotly Figure with two vertically stacked subplots.
    """
    theme = get_theme("light")
    colors = theme["colors"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Module |Z| (Ω)", "Phase (°)"],
        vertical_spacing=0.12,
    )

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
        Zmod = np.sqrt(sp.Zre ** 2 + sp.Zim ** 2)
        phase_deg = np.degrees(np.arctan2(-sp.Zim, sp.Zre))

        shared_kw = dict(
            x=sp.f, mode="markers+lines",
            marker=dict(color=color, size=5),
            line=dict(color=color),
        )

        fig.add_trace(go.Scatter(
            **shared_kw, y=Zmod, name=lbl, showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            **shared_kw, y=phase_deg, name=lbl, showlegend=False,
        ), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Fréquence (Hz)", row=2, col=1)
    fig.update_yaxes(type="log", title_text="|Z| (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)", row=2, col=1)

    fig.update_layout(title="Diagramme de Bode")
    apply_theme_to_figure(fig, "light")
    return fig


# ── DRT ────────────────────────────────────────────────────────────────────────

def drt_figure(session: EISSession) -> go.Figure:
    """Plot DRT gamma(tau) spectra for all concentration groups.

    Args:
        session: EISSession.

    Returns:
        Plotly Figure.
    """
    theme = get_theme("light")
    colors = theme["colors"]

    fig = go.Figure()

    for ci, grp in enumerate(session.groups):
        fr = grp.fit_results.get("drt_tikhonov")
        if fr is None:
            continue

        tau = np.array(fr.params.get("tau", []))
        gamma = np.array(fr.params.get("gamma", []))
        if len(tau) == 0:
            continue

        lbl = _spectrum_label(grp.spectrum)
        color = colors[ci % len(colors)]

        fig.add_trace(go.Scatter(
            x=tau, y=gamma,
            mode="lines",
            name=lbl,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Distribution des temps de relaxation (DRT)",
        xaxis=dict(type="log", title_text="τ (s)"),
        yaxis_title="γ(τ) (Ω)",
    )
    apply_theme_to_figure(fig, "light")
    return fig


# ── Parameters table ───────────────────────────────────────────────────────────

def params_table_figure(session: EISSession) -> go.Figure:
    """Build a Plotly table of all extracted fit parameters.

    Args:
        session: EISSession.

    Returns:
        Plotly Figure with a single Table trace.
    """
    rows = []
    for grp in session.groups:
        lbl = _spectrum_label(grp.spectrum)
        for model_name, fr in grp.fit_results.items():
            row: dict = {
                "Concentration": lbl,
                "Modèle": model_name,
                "Rct (Ω)": f"{fr.Rct:.3e}",
                "χ²": f"{fr.chi2:.3e}",
                "Conv.": "✓" if fr.converged else "✗",
            }
            for k, v in fr.params.items():
                if k not in ("tau", "gamma") and isinstance(v, (int, float)):
                    row[k] = f"{v:.3e}"
            rows.append(row)

    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucun résultat de fit disponible.",
            showarrow=False, font=dict(size=14),
        )
        return fig

    all_keys = list(rows[0].keys())
    cells = [[r.get(k, "—") for r in rows] for k in all_keys]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=all_keys,
            fill_color="#4472C4",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cells,
            fill_color=[["#EEF0F8", "#FFFFFF"] * (len(rows) // 2 + 1)],
            align="left",
            font=dict(size=11),
        ),
    )])
    fig.update_layout(title="Paramètres extraits par modèle")
    return fig


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_figure(session: EISSession) -> go.Figure:
    """Build calibration curve: log(Rct) vs log([concentration]).

    Uses the first available fit model for each concentration group.
    Overlays a linear regression and displays R².

    Args:
        session: EISSession.

    Returns:
        Plotly Figure.
    """
    theme = get_theme("light")
    colors = theme["colors"]

    fig = go.Figure()

    concs: list = []
    rcts: list = []
    model_used = ""

    for grp in session.groups:
        if grp.concentration <= 0:
            continue
        for m_name, fr in grp.fit_results.items():
            if fr.Rct > 0:
                concs.append(grp.concentration)
                rcts.append(fr.Rct)
                model_used = m_name
                break

    if len(concs) < 2:
        fig.add_annotation(
            text="Pas assez de points (min. 2 concentrations positives).",
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    log_c = np.log10(concs)
    log_rct = np.log10(rcts)

    # Linear regression
    coeffs = np.polyfit(log_c, log_rct, 1)
    p = np.poly1d(coeffs)
    log_c_line = np.linspace(log_c.min(), log_c.max(), 200)

    rct_pred = p(log_c)
    ss_res = float(np.sum((log_rct - rct_pred) ** 2))
    ss_tot = float(np.sum((log_rct - np.mean(log_rct)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig.add_trace(go.Scatter(
        x=log_c, y=log_rct,
        mode="markers",
        name="Données",
        marker=dict(color=colors[0], size=10, symbol="circle"),
        hovertemplate=(
            "log([c]) = %{x:.2f}<br>"
            "log(Rct) = %{y:.2f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=log_c_line, y=p(log_c_line),
        mode="lines",
        name=f"Régression linéaire (R²={r2:.4f}, pente={coeffs[0]:.2f})",
        line=dict(color=colors[1], dash="dash", width=2),
    ))

    fig.update_layout(
        title=f"Courbe de calibration — {model_used}",
        xaxis_title="log([c] / M)",
        yaxis_title="log(Rct / Ω)",
    )
    apply_theme_to_figure(fig, "light")
    return fig
