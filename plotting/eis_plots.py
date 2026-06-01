"""All EIS Plotly figures: Nyquist, Bode, DRT, parameter table, calibration."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from core.models import EISSession, EISSpectrum
from plotting.theme import get_theme, apply_theme_to_figure


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


# ── Nyquist ────────────────────────────────────────────────────────────────────

def nyquist_figure(session: EISSession) -> go.Figure:
    """Nyquist plot. X = Re(Z), Y = -Im(Z) > 0 (upper-right quadrant)."""
    theme = get_theme("light")
    colors = theme["colors"]
    dashes = theme["fit_dash"]
    fig = go.Figure()

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


# ── Bode ───────────────────────────────────────────────────────────────────────

def bode_figure(session: EISSession) -> go.Figure:
    theme = get_theme("light")
    colors = theme["colors"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
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
        kw = dict(x=sp.f, mode="markers+lines",
                  marker=dict(color=color, size=5), line=dict(color=color))
        fig.add_trace(go.Scatter(**kw, y=Zmod, name=lbl, showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(**kw, y=phase_deg, name=lbl, showlegend=False), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Fréquence (Hz)", row=2, col=1)
    fig.update_yaxes(type="log", title_text="|Z| (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)", row=2, col=1)
    fig.update_layout(title="Diagramme de Bode")
    apply_theme_to_figure(fig, "light")
    return fig


# ── DRT ────────────────────────────────────────────────────────────────────────

def drt_figure(session: EISSession, log_y: bool = True) -> go.Figure:
    """Distribution des temps de relaxation — axe ln(τ), convention Bissessur (2026).

    Axe X : ln(τ)  où τ = 1/(2πf).
    Axe Y : ln(γ) si log_y=True (défaut), γ linéaire sinon.
    Inclut bare, probe et tous les groupes de concentration.
    """
    theme  = get_theme("light")
    colors = theme["colors"]
    fig    = go.Figure()

    # Collecte : bare → probe → groupes
    all_items = []
    ci = 0
    for sp in (session.bare, session.probe):
        if sp is not None:
            fr = sp.fit_results.get("drt_fft")
            if fr is not None:
                all_items.append((_spectrum_label(sp), fr, ci))
            ci += 1
    for grp in session.groups:
        fr = grp.fit_results.get("drt_fft")
        if fr is not None:
            all_items.append((_spectrum_label(grp.spectrum), fr, ci))
        ci += 1

    if not all_items:
        fig.add_annotation(
            text="Aucune DRT disponible — lancez l'analyse.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    for lbl, fr, color_idx in all_items:
        color = colors[color_idx % len(colors)]

        # Axe X : ln(τ) si disponible (v3), sinon log(τ) converti
        if "ln_tau" in fr.params and len(fr.params["ln_tau"]) > 0:
            x_vals  = np.array(fr.params["ln_tau"])
            x_title = "ln(τ)   [τ = 1/(2πf), s]"
        else:
            tau_raw = np.array(fr.params.get("tau", []))
            x_vals  = np.log(tau_raw + 1e-30)
            x_title = "ln(τ)  (s)"

        gamma = np.array(fr.params.get("gamma", []))
        if len(x_vals) == 0 or len(gamma) == 0:
            continue

        mask       = gamma > np.max(gamma) / np.exp(10)
        x_plot     = x_vals[mask]
        gamma_plot = gamma[mask]

        if log_y:
            y_vals  = np.log(np.clip(gamma_plot, 1e-30, None))
            y_title = "ln γ(τ)  [Ω]"
        else:
            y_vals  = gamma_plot
            y_title = "γ(τ)  [Ω]"

        fig.add_trace(go.Scatter(
            x=x_plot, y=y_vals, mode="lines", name=lbl,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "ln(τ) = %{x:.3f}<br>"
                f"{y_title} = %{{y:.4f}}<extra></extra>"
            ),
        ))

        # Marqueurs de pics
        tau_peaks = fr.params.get("tau_peaks", [])
        if tau_peaks:
            ln_pk = np.log(np.array(tau_peaks))
            g_pk  = np.interp(ln_pk, x_vals, gamma)
            y_pk  = np.log(np.clip(g_pk, 1e-30, None)) if log_y else g_pk
            fig.add_trace(go.Scatter(
                x=ln_pk, y=y_pk, mode="markers",
                name=f"{lbl} pics",
                marker=dict(symbol="diamond", size=9, color=color,
                            line=dict(width=1.5, color="white")),
                customdata=tau_peaks,
                hovertemplate=(
                    f"<b>Pic · {lbl}</b><br>"
                    "τ = %{customdata:.3e} s<br>"
                    f"{y_title} = %{{y:.4f}}<extra></extra>"
                ),
            ))

    # Titre avec λ du dernier spectre
    lam_info = ""
    if all_items:
        fr_last  = all_items[-1][1]
        lam_val  = fr_last.params.get("lambda")
        lam_meth = fr_last.params.get("_str_lambda_method", "")
        if lam_val is not None:
            lam_info = f" — λ={lam_val:.2e} ({lam_meth})"

    fig.update_layout(
        title=f"Distribution des temps de relaxation (DRT){lam_info}",
        xaxis_title=x_title if all_items else "ln(τ)",
        yaxis_title=y_title if all_items else "γ(τ)",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        hovermode="closest",
    )
    apply_theme_to_figure(fig, "light")
    return fig


def drt_lambda_diag_figure(fit_result, label: str = "") -> go.Figure:
    """Panneau de diagnostic λ : L-curve (gauche) + GCV (droite).

    Appeler depuis ui/tabs.py dans l'onglet DRT.
    """
    params   = fit_result.params
    lc_lams  = np.array(params.get("_lc_lambdas",  []))
    lc_rho   = np.array(params.get("_lc_rho",       []))
    lc_eta   = np.array(params.get("_lc_eta",       []))
    gcv_lams = np.array(params.get("_gcv_lambdas",  []))
    gcv_sc   = np.array(params.get("_gcv_scores",   []))
    lam_lc   = params.get("lambda_lcurve", params.get("lambda"))
    lam_gcv  = params.get("lambda_gcv",    params.get("lambda"))

    if len(lc_lams) == 0 and len(gcv_lams) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Diagnostic λ non disponible (mode fixe).",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["L-curve", "Score GCV vs λ"],
                        horizontal_spacing=0.12)

    if len(lc_lams) > 0:
        fig.add_trace(go.Scatter(
            x=lc_rho, y=lc_eta, mode="lines+markers", name="L-curve",
            line=dict(color="#c2410c", width=2), marker=dict(size=4),
            customdata=lc_lams,
            hovertemplate="λ=%{customdata:.2e}<br>‖r‖=%{x:.3e}<br>‖s‖=%{y:.3e}<extra></extra>",
        ), row=1, col=1)
        if lam_lc is not None:
            r_lc = float(np.interp(lam_lc, lc_lams, lc_rho))
            e_lc = float(np.interp(lam_lc, lc_lams, lc_eta))
            fig.add_trace(go.Scatter(
                x=[r_lc], y=[e_lc], mode="markers",
                name=f"λ={lam_lc:.2e}",
                marker=dict(symbol="star", size=14, color="#dc2626"),
            ), row=1, col=1)
        fig.update_xaxes(title_text="‖Aγ−b‖", type="log", row=1, col=1)
        fig.update_yaxes(title_text="‖Lγ‖",    type="log", row=1, col=1)

    if len(gcv_lams) > 0:
        fig.add_trace(go.Scatter(
            x=gcv_lams, y=gcv_sc, mode="lines", name="GCV",
            line=dict(color="#1a56db", width=2),
            hovertemplate="λ=%{x:.2e}<br>GCV=%{y:.3e}<extra></extra>",
        ), row=1, col=2)
        if lam_gcv is not None:
            g_best = float(np.interp(lam_gcv, gcv_lams, gcv_sc))
            fig.add_trace(go.Scatter(
                x=[lam_gcv], y=[g_best], mode="markers",
                name=f"λ_GCV={lam_gcv:.2e}",
                marker=dict(symbol="star", size=14, color="#7c3aed"),
            ), row=1, col=2)
        fig.update_xaxes(title_text="λ", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Score GCV", type="log", row=1, col=2)

    title = f"Diagnostic λ — {label}" if label else "Diagnostic sélection λ"
    fig.update_layout(title=title, legend=dict(orientation="h", y=-0.2))
    apply_theme_to_figure(fig, "light")
    return fig


# ── Parameters table ───────────────────────────────────────────────────────────

def params_table_figure(session: EISSession) -> go.Figure:
    """Table of Rct for bare, probe and each concentration, by model."""
    model_names: list = []
    for sp in (session.bare, session.probe):
        if sp is not None:
            for m in sp.fit_results:
                if m not in model_names:
                    model_names.append(m)
    for grp in session.groups:
        for m in grp.fit_results:
            if m not in model_names:
                model_names.append(m)

    if not model_names:
        fig = go.Figure()
        fig.add_annotation(text="Aucun résultat de fit disponible.", showarrow=False, font=dict(size=14))
        return fig

    def _rct_str(fit_results: dict, model: str) -> str:
        fr = fit_results.get(model)
        if fr is None or fr.Rct <= 0:
            return "—"
        return f"{fr.Rct:.1f} Ω"

    step_col: list = []
    model_cols: list = [[] for _ in model_names]

    if session.bare is not None:
        step_col.append("Bare")
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(session.bare.fit_results, m))

    if session.probe is not None:
        step_col.append("Probe")
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(session.probe.fit_results, m))

    for grp in session.groups:
        step_col.append(f"{grp.concentration:.2e} M")
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(grp.fit_results, m))

    n_rows = len(step_col)
    row_colors = ["#EEF0F8" if i % 2 == 0 else "#FFFFFF" for i in range(n_rows)]
    header_values = ["Étape"] + model_names
    cell_values = [step_col] + model_cols

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, fill_color="#4472C4",
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=cell_values,
                   fill_color=[row_colors] * len(header_values),
                   align="left", font=dict(size=11)),
    )])
    fig.update_layout(title="Rct par étape et modèle")
    return fig


# ── Calibration ────────────────────────────────────────────────────────────────

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

    model_names = list(probe_fr.keys())
    dashes = ["solid", "dash", "dot", "dashdot"]

    # Collect per-model regression results first
    model_data = []
    for mi, model in enumerate(model_names):
        probe_fit = probe_fr.get(model)
        if probe_fit is None or probe_fit.Rct <= 0:
            continue
        probe_rct = probe_fit.Rct

        concs: list = []
        signals: list = []
        for grp in session.groups:
            if grp.concentration <= 0:
                continue
            fr = grp.fit_results.get(model)
            if fr is None or fr.Rct <= 0:
                continue
            concs.append(grp.concentration)
            signals.append(abs(probe_rct - fr.Rct) / abs(probe_rct))

        if len(concs) < 2:
            continue

        log_c = np.log10(concs)
        reg = stats.linregress(log_c, signals)
        model_data.append({
            "model": model,
            "log_c": log_c,
            "signals": signals,
            "reg": reg,
            "r2": reg.rvalue ** 2,
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
        reg = d["reg"]
        log_c = d["log_c"]
        log_c_line = np.linspace(log_c.min(), log_c.max(), 200)
        y_line = reg.slope * log_c_line + reg.intercept
        sign = "+" if reg.intercept >= 0 else "-"
        legend_label = (
            f"{d['model']}  "
            f"R²={d['r2']:.3f}  "
            f"y={reg.slope:.3f}x {sign} {abs(reg.intercept):.3f}"
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
                [f"{d['reg'].slope:.4f}" for d in model_data],
                [f"{d['reg'].intercept:.4f}" for d in model_data],
                [f"{d['reg'].pvalue:.2e}" for d in model_data],
                [f"{d['reg'].stderr:.4f}" for d in model_data],
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
