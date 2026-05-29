"""All EIS Plotly figures: Nyquist, Bode, DRT, parameter table, calibration."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

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
    """Build a Plotly table showing Rct for each step and fit model.

    Args:
        session: EISSession.

    Returns:
        Plotly Figure with a single Table trace.
    """
    # Collect all model names present in the session
    model_names: list[str] = []
    for grp in session.groups:
        for m in grp.fit_results:
            if m not in model_names:
                model_names.append(m)
    # Also check bare/probe if they have fit_results
    for sp in (session.bare, session.probe):
        if sp is not None and hasattr(sp, "fit_results"):
            for m in sp.fit_results:
                if m not in model_names:
                    model_names.append(m)

    if not model_names:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucun résultat de fit disponible.",
            showarrow=False, font=dict(size=14),
        )
        return fig

    def _rct_str(fit_results: dict, model: str) -> str:
        fr = fit_results.get(model)
        if fr is None or fr.Rct <= 0:
            return "—"
        return f"{fr.Rct:.1f} Ω"

    # Build rows: [Étape, model1, model2, ...]
    header_values = ["Étape"] + model_names
    step_col: list[str] = []
    model_cols: list[list[str]] = [[] for _ in model_names]

    # Bare row
    if session.bare is not None:
        step_col.append("Bare")
        bare_fr = getattr(session.bare, "fit_results", {})
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(bare_fr, m))

    # Probe row
    if session.probe is not None:
        step_col.append("Probe")
        probe_fr = getattr(session.probe, "fit_results", {})
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(probe_fr, m))

    # Concentration rows
    for grp in session.groups:
        step_col.append(f"{grp.concentration:.2e} M")
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(grp.fit_results, m))

    n_rows = len(step_col)
    row_colors = ["#EEF0F8" if i % 2 == 0 else "#FFFFFF" for i in range(n_rows)]

    cell_values = [step_col] + model_cols

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_values,
            fill_color="#4472C4",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cell_values,
            fill_color=[row_colors] * len(header_values),
            align="left",
            font=dict(size=11),
        ),
    )])
    fig.update_layout(title="Rct par étape et modèle")
    return fig


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_figure(session: EISSession) -> go.Figure:
    """Build calibration curve: normalized Rct signal vs log([concentration]).

    signal_norm = |Rct_probe - Rct_conc| / |Rct_probe|

    Selects the fit model with the best R² on the log-linear regression.

    Args:
        session: EISSession.

    Returns:
        Plotly Figure.
    """
    theme = get_theme("light")
    colors = theme["colors"]

    fig = go.Figure()

    # Require probe with fit_results
    probe_fr = getattr(session.probe, "fit_results", None) if session.probe else None
    if not probe_fr:
        fig.add_annotation(
            text="Spectre probe requis pour la calibration normalisée.",
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    # Collect all model names available in both probe and concentration groups
    candidate_models = list(probe_fr.keys())

    best_r2 = -np.inf
    best_model = ""
    best_concs: list = []
    best_signals: list = []
    best_slope = 0.0
    best_intercept = 0.0

    for model in candidate_models:
        probe_rct = probe_fr[model].Rct if probe_fr.get(model) else None
        if probe_rct is None or probe_rct <= 0:
            continue

        concs: list = []
        signals: list = []
        for grp in session.groups:
            if grp.concentration <= 0:
                continue
            fr = grp.fit_results.get(model)
            if fr is None or fr.Rct <= 0:
                continue
            signal_norm = abs(probe_rct - fr.Rct) / abs(probe_rct)
            concs.append(grp.concentration)
            signals.append(signal_norm)

        if len(concs) < 2:
            continue

        log_c = np.log10(concs)
        result = stats.linregress(log_c, signals)
        r2 = result.rvalue ** 2
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_concs = concs
            best_signals = signals
            best_slope = result.slope
            best_intercept = result.intercept

    if not best_concs:
        fig.add_annotation(
            text="Pas assez de points (min. 2 concentrations positives avec fit probe).",
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    log_c_arr = np.log10(best_concs)
    log_c_line = np.linspace(log_c_arr.min(), log_c_arr.max(), 200)
    y_line = best_slope * log_c_line + best_intercept

    sign_str = "+" if best_intercept >= 0 else "-"
    eq_str = f"y = {best_slope:.3f}·x {sign_str} {abs(best_intercept):.3f}"

    fig.add_trace(go.Scatter(
        x=log_c_arr, y=best_signals,
        mode="markers",
        name="Données",
        marker=dict(color=colors[0], size=10, symbol="circle"),
        hovertemplate=(
            "log([c]) = %{x:.2f}<br>"
            "Signal norm. = %{y:.4f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=log_c_line, y=y_line,
        mode="lines",
        name=f"R² = {best_r2:.4f}",
        line=dict(color=colors[1], dash="dash", width=2),
    ))

    fig.add_annotation(
        text=f"{eq_str}<br>R² = {best_r2:.4f}",
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#4472C4",
        borderwidth=1,
        font=dict(size=11),
    )

    fig.update_layout(
        title=f"Calibration — {best_model}",
        xaxis_title="log([c] / M)",
        yaxis_title="|Rct_probe − Rct_c| / |Rct_probe|",
    )
    apply_theme_to_figure(fig, "light")
    return fig
