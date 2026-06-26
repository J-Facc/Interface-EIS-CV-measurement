"""Plotly figures for CV analysis."""

import numpy as np
import plotly.colors as _pc
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.cv_models import CVSession
from core.cv_peaks import detect_redox_peaks


def _cv_conc_color(concentration: float, c_min: float, c_max: float) -> str:
    """Couleur log-interpolée violet→rouge pour une concentration CV."""
    if c_min > 0 and c_max > 0 and c_min < c_max:
        t = (np.log10(concentration) - np.log10(c_min)) / (np.log10(c_max) - np.log10(c_min))
    else:
        t = 0.5
    return _pc.sample_colorscale("plasma", [float(np.clip(t, 0, 1))])[0]


def cv_figure_electrode(
    scans: list,
    title: str = "",
) -> go.Figure:
    """Voltammogramme pour une électrode : une trace par concentration + probe.

    Parameters
    ----------
    scans : list de dicts avec clés "label", "E", "I", "concentration".
            Probe avec concentration=0, autres avec concentration > 0.
    title : titre du graphique.
    """
    fig = go.Figure()

    pos   = [s for s in scans if s["concentration"] > 0]
    c_min = min(s["concentration"] for s in pos) if pos else 1e-12
    c_max = max(s["concentration"] for s in pos) if pos else 1e-8

    for s in scans:
        E    = np.asarray(s["E"])
        I    = np.asarray(s["I"])
        lbl  = s["label"]
        conc = s["concentration"]

        if conc <= 0:
            color = "black"
            dash  = "dash"
        else:
            color = _cv_conc_color(conc, c_min, c_max)
            dash  = "solid"

        fig.add_trace(go.Scatter(
            x=E, y=I * 1e6,
            mode="lines",
            name=lbl,
            line=dict(color=color, dash=dash, width=2),
        ))

    fig.update_layout(
        title=title or "Voltammogramme",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        margin=dict(r=120),
    )
    return fig


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


_ELECTRODE_STYLES = [
    dict(marker_color="steelblue", line_color="steelblue", line_dash="solid"),
    dict(marker_color="darkorange", line_color="darkorange", line_dash="dash"),
    dict(marker_color="seagreen", line_color="seagreen", line_dash="dot"),
    dict(marker_color="firebrick", line_color="firebrick", line_dash="dashdot"),
]


def cv_calibration_figure_multi(cv_sessions: dict) -> go.Figure:
    """Une droite de régression par électrode, sur le même graphe.

    Reprend la logique de cv_calibration_figure mais bouclée sur
    cv_sessions.items() (un CVSession par électrode), avec un style distinct
    par électrode et une annotation empilée (une ligne par électrode).
    Si une seule électrode est présente, se comporte comme
    cv_calibration_figure.
    """
    fig = go.Figure()
    annotation_lines = []

    for i, (elec, session) in enumerate(sorted(cv_sessions.items())):
        concentrations, signals = [], []
        for grp in session.groups:
            if grp.concentration <= 0:
                continue
            mean_sig = np.nanmean(grp.delta_signal)
            if np.isfinite(mean_sig):
                concentrations.append(grp.concentration)
                signals.append(mean_sig)

        if len(concentrations) < 2:
            continue

        style = _ELECTRODE_STYLES[i % len(_ELECTRODE_STYLES)]
        log_c = np.log10(concentrations)
        sig = np.array(signals)

        slope, intercept, r_value, _, _ = stats.linregress(log_c, sig)
        r2 = r_value ** 2

        x_fit = np.linspace(log_c.min(), log_c.max(), 200)
        y_fit = slope * x_fit + intercept

        fig.add_trace(go.Scatter(
            x=log_c, y=sig,
            mode="markers",
            name=f"Électrode {elec} — mesuré",
            marker=dict(size=10, color=style["marker_color"]),
            legendgroup=f"e{elec}",
        ))
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines",
            name=f"Électrode {elec} — régression R²={r2:.3f}",
            line=dict(color=style["line_color"], dash=style["line_dash"]),
            legendgroup=f"e{elec}",
        ))

        annotation_lines.append(
            f"Électrode {elec} : R² = {r2:.4f}, pente = {slope:.3f}, ordonnée = {intercept:.3f}"
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


# ── Pic redox (ajout — onglet "Pic redox") ────────────────────────────────────

def redox_peaks_figure(cv_session: CVSession) -> go.Figure:
    """I(U) pour probe + chaque concentration, pics anodique/cathodique marqués
    (étoiles), + tableau Ipa/Epa/Ipc/Epc/ΔEp par concentration (incl. probe).

    Construit une figure à deux lignes : courbes I(U) en haut, tableau en bas.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.08,
        subplot_titles=["Voltammogrammes avec pics redox", "Paramètres extraits"],
    )

    rows_table: list = []

    scans_with_label = []
    if cv_session.probe is not None:
        scans_with_label.append(("Probe", 0.0, cv_session.probe))
    for grp in cv_session.groups:
        scans_with_label.append((f"{grp.concentration:.2e} M", grp.concentration, grp.scan))

    c_vals = [c for _, c, _ in scans_with_label if c > 0]
    c_min = min(c_vals) if c_vals else 1e-12
    c_max = max(c_vals) if c_vals else 1e-8

    for lbl, conc, scan in scans_with_label:
        color = "black" if conc <= 0 else _cv_conc_color(conc, c_min, c_max)
        fig.add_trace(go.Scatter(
            x=scan.E, y=scan.I * 1e6, mode="lines", name=lbl,
            line=dict(color=color, width=2),
            legendgroup=lbl,
        ), row=1, col=1)

        peaks = detect_redox_peaks(scan)
        fig.add_trace(go.Scatter(
            x=[peaks["Epa"], peaks["Epc"]],
            y=[peaks["Ipa"] * 1e6, peaks["Ipc"] * 1e6],
            mode="markers",
            name=f"{lbl} — pics",
            marker=dict(symbol="star", size=12, color=color, line=dict(width=1, color="black")),
            legendgroup=lbl,
            showlegend=False,
        ), row=1, col=1)

        rows_table.append({
            "Concentration": lbl,
            "Ipa (µA)": f"{peaks['Ipa']*1e6:.3f}",
            "Epa (V)": f"{peaks['Epa']:.4f}",
            "Ipc (µA)": f"{peaks['Ipc']*1e6:.3f}",
            "Epc (V)": f"{peaks['Epc']:.4f}",
            "ΔEp (V)": f"{peaks['delta_Ep']:.4f}",
        })

    if rows_table:
        cols = list(rows_table[0].keys())
        fig.add_trace(go.Table(
            header=dict(values=[f"<b>{c}</b>" for c in cols],
                        fill_color="#4472C4", font=dict(color="white", size=11), align="center"),
            cells=dict(
                values=[[r[c] for r in rows_table] for c in cols],
                fill_color=[["white", "#f5f5f5"] * (len(rows_table) // 2 + 1)],
                align="center", font=dict(size=10),
            ),
        ), row=2, col=1)

    fig.update_xaxes(title_text="Potentiel E (V)", row=1, col=1)
    fig.update_yaxes(title_text="Courant I (µA)", row=1, col=1)
    fig.update_layout(
        title="Pics redox — anodique (Ipa/Epa) et cathodique (Ipc/Epc)",
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        height=800,
    )
    return fig


def cv_params_table_multi(cv_sessions: dict) -> go.Figure:
    """Tableau Ipa/Epa/Ipc/Epc/ΔEp par concentration et par électrode.

    Reprend la logique de la table de redox_peaks_figure, bouclée sur
    cv_sessions.items() pour ajouter une colonne Électrode.
    """
    rows_table: list = []

    for elec, session in sorted(cv_sessions.items()):
        scans_with_label = []
        if session.probe is not None:
            scans_with_label.append(("Probe", session.probe))
        for grp in session.groups:
            scans_with_label.append((f"{grp.concentration:.2e} M", grp.scan))

        for lbl, scan in scans_with_label:
            peaks = detect_redox_peaks(scan)
            rows_table.append({
                "Électrode": f"E{elec}",
                "Concentration": lbl,
                "Ipa (µA)": f"{peaks['Ipa']*1e6:.3f}",
                "Epa (V)": f"{peaks['Epa']:.4f}",
                "Ipc (µA)": f"{peaks['Ipc']*1e6:.3f}",
                "Epc (V)": f"{peaks['Epc']:.4f}",
                "ΔEp (V)": f"{peaks['delta_Ep']:.4f}",
            })

    fig = go.Figure()
    if not rows_table:
        return fig

    cols = list(rows_table[0].keys())
    fig.add_trace(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in cols],
                    fill_color="#4472C4", font=dict(color="white", size=12), align="center"),
        cells=dict(
            values=[[r[c] for r in rows_table] for c in cols],
            fill_color=[["white", "#f5f5f5"] * (len(rows_table) // 2 + 1)],
            align="center",
        ),
    ))
    fig.update_layout(title="Paramètres extraits par concentration et électrode")
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
        concentrations, signals = [], []
        for grp in session.groups:
            if grp.concentration <= 0:
                continue
            mean_sig = np.nanmean(grp.delta_signal)
            if np.isfinite(mean_sig):
                concentrations.append(grp.concentration)
                signals.append(mean_sig)

        if len(concentrations) < 2:
            continue

        log_c = np.log10(concentrations)
        sig = np.array(signals)
        slope, intercept, r_value, _, _ = stats.linregress(log_c, sig)
        ax.plot(log_c, sig, markers[i % len(markers)],
                 label=f"Électrode {elec} — mesuré")
        x_fit = np.linspace(log_c.min(), log_c.max(), 200)
        ax.plot(x_fit, slope * x_fit + intercept, linestyles[i % len(linestyles)],
                 label=f"Électrode {elec} — régression (R²={r_value**2:.3f})")
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


def open_cv_calibration_matplotlib_window(cv_session: CVSession) -> None:
    """Ouvre une fenêtre matplotlib (bloquante) reproduisant cv_calibration_figure()."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    concentrations, signals = [], []
    for grp in cv_session.groups:
        if grp.concentration <= 0:
            continue
        mean_sig = np.nanmean(grp.delta_signal)
        if np.isfinite(mean_sig):
            concentrations.append(grp.concentration)
            signals.append(mean_sig)

    if len(concentrations) >= 2:
        log_c = np.log10(concentrations)
        sig = np.array(signals)
        slope, intercept, r_value, _, _ = stats.linregress(log_c, sig)
        ax.plot(log_c, sig, "o", label="Signal mesuré")
        x_fit = np.linspace(log_c.min(), log_c.max(), 200)
        ax.plot(x_fit, slope * x_fit + intercept, "--",
                 label=f"Régression (R²={r_value**2:.3f})")
    else:
        ax.text(0.5, 0.5, "Pas assez de points", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title("Calibration CV — Signal normalisé vs log([c])")
    ax.set_xlabel("log([c] / M)")
    ax.set_ylabel("|ΔI| / |I_probe|")
    ax.legend()
    fig.tight_layout()
    plt.show()
