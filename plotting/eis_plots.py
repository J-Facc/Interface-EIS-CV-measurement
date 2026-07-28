"""All EIS Plotly figures: Nyquist, Bode, DRT, parameter table, calibration."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import EISSession, EISSpectrum
from core.calibration import compute_calibration_all, compute_calibration_loglog
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


# ── Bode ───────────────────────────────────────────────────────────────────────────

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


# ── Parameters table ──────────────────────────────────────────────────────────

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

    def _rct_val(fit_results: dict, model: str):
        fr = fit_results.get(model)
        if fr is None or fr.Rct <= 0:
            return None
        return float(fr.Rct)

    def _chi2_str(fit_results: dict, model: str) -> str:
        fr = fit_results.get(model)
        if fr is None or fr.chi2_reduced is None or not np.isfinite(fr.chi2_reduced):
            return "—"
        return f"{fr.chi2_reduced:.2f}"

    has_comparison = "randles_full" in model_names and "drt_bayes" in model_names

    def _delta_str(fit_results: dict) -> str:
        r_randles = _rct_val(fit_results, "randles_full")
        r_drt = _rct_val(fit_results, "drt_bayes")
        if r_randles is None or r_drt is None:
            return "—"
        rel_err = abs(r_randles - r_drt) / abs(r_randles)
        return f"{rel_err * 100:.1f}%"

    step_col: list = []
    model_cols: list = [[] for _ in model_names]
    chi2_cols: list = [[] for _ in model_names]
    delta_col: list = []

    def _append_row(step_label: str, fit_results: dict) -> None:
        step_col.append(step_label)
        for i, m in enumerate(model_names):
            model_cols[i].append(_rct_str(fit_results, m))
            chi2_cols[i].append(_chi2_str(fit_results, m))
        if has_comparison:
            delta_col.append(_delta_str(fit_results))

    if session.bare is not None:
        _append_row("Bare", session.bare.fit_results)

    if session.probe is not None:
        _append_row("Probe", session.probe.fit_results)

    for grp in session.groups:
        _append_row(f"{grp.concentration:.2e} M", grp.fit_results)

    n_rows = len(step_col)
    row_colors = ["#EEF0F8" if i % 2 == 0 else "#FFFFFF" for i in range(n_rows)]
    header_values = ["Étape"] + [f"Rct — {m}" for m in model_names]
    cell_values = [step_col] + model_cols
    if has_comparison:
        header_values = header_values + ["Écart relatif Randles/DRT"]
        cell_values = cell_values + [delta_col]
    # Colonnes χ²_réduit par modèle (≈1 = adéquation en pondération sigma).
    header_values = header_values + [f"χ²ᵣ — {m}" for m in model_names]
    cell_values = cell_values + chi2_cols

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, fill_color="#4472C4",
                    font=dict(color="white", size=12), align="left"),
        cells=dict(values=cell_values,
                   fill_color=[row_colors] * len(header_values),
                   align="left", font=dict(size=11)),
    )])
    fig.update_layout(title="Rct par étape et modèle — Randles (paramétrique) vs DRT (model-free)")
    return fig


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


# ── Kramers-Kronig (fits/kk_validation.py) ───────────────────────────────────

def kk_figure(spectrum: EISSpectrum, kk_result: dict, label: str = "") -> go.Figure:
    """Nyquist (mesuré vs reconstruction KK) + résidus normalisés (%) en sous-graphes.

    Args:
        spectrum: Spectre EIS d'origine.
        kk_result: dict retourné par fits.kk_validation.kramers_kronig_check.
        label: Nom affiché dans le titre.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=["Nyquist — mesuré vs reconstruction KK", "Résidus normalisés (%)"],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=spectrum.Zre, y=spectrum.Zim, mode="markers", name="Mesuré",
        marker=dict(color="#1a56db", size=7),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=kk_result["Z_kk_re"], y=kk_result["Z_kk_im"], mode="lines", name="Reconstruction KK",
        line=dict(color="#dc2626", width=2),
    ), row=1, col=1)
    fig.update_xaxes(title_text="Z' (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' (Ω)", scaleanchor="x", row=1, col=1)

    f = np.asarray(spectrum.f, dtype=float)
    Zmod = np.sqrt(np.asarray(spectrum.Zre)**2 + np.asarray(spectrum.Zim)**2)
    Zmod = np.where(Zmod > 0, Zmod, 1e-30)
    res_re_pct = 100.0 * kk_result["residuals_re"] / Zmod
    res_im_pct = 100.0 * kk_result["residuals_im"] / Zmod

    fig.add_trace(go.Scatter(
        x=f, y=res_re_pct, mode="markers+lines", name="Résidu Re",
        line=dict(color="#1a56db", width=1), marker=dict(size=5),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=f, y=res_im_pct, mode="markers+lines", name="Résidu Im",
        line=dict(color="#db2777", width=1), marker=dict(size=5),
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=0.5), row=2, col=1)
    fig.update_xaxes(type="log", title_text="Fréquence (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Résidu (%)", row=2, col=1)

    verdict = "✅ KK validé" if kk_result["kk_passed"] else "❌ KK échoué"
    fig.update_layout(
        title=f"Validation Kramers-Kronig — {label} — {verdict} "
              f"(max résidu={kk_result['max_residual']*100:.2f}%)",
        legend=dict(orientation="h", y=-0.15),
    )
    apply_theme_to_figure(fig, "light")
    return fig


def drt_reconstruction_figure(spectrum: EISSpectrum, fit_result, label: str = "") -> go.Figure:
    """Nyquist mesuré vs reconstruit par le modèle DRT, avec ε affiché."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spectrum.Zre, y=spectrum.Zim, mode="markers", name="Mesuré",
        marker=dict(color="#1a56db", size=7),
    ))
    fig.add_trace(go.Scatter(
        x=fit_result.Zfit_re, y=fit_result.Zfit_im, mode="lines", name="Reconstruction DRT",
        line=dict(color="#dc2626", width=2),
    ))
    fig.update_xaxes(title_text="Z' (Ω)")
    fig.update_yaxes(title_text="-Z'' (Ω)", scaleanchor="x")

    err = fit_result.reconstruction_error
    err_str = f"ε = {err*100:.2f}%" if err is not None else "ε non disponible"
    fig.update_layout(
        title=f"Reconstruction DRT — {label} — {err_str}",
        legend=dict(orientation="h", y=-0.15),
    )
    apply_theme_to_figure(fig, "light")
    return fig


def calibration_drt_figure(session: EISSession, model_name: str = "drt_bayes") -> go.Figure:
    """Calibration log(Rct) vs log([c]) pour un modèle DRT, avec barres d'erreur et régression.

    Côte à côte : nuage de points + droite de régression (gauche), résidus (droite).
    """
    # Calcul délégué à core/calibration.py (calibration log-log Rct).
    cal = compute_calibration_loglog(session, model_name)
    if cal is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Pas assez de points (min. 2 concentrations positives avec fit DRT).",
            showarrow=False, font=dict(size=13),
        )
        apply_theme_to_figure(fig, "light")
        return fig

    log_c, log_rct, rcts, errs = cal.log_c, cal.y, cal.rcts, cal.errs
    log_c_line = np.linspace(log_c.min(), log_c.max(), 200)
    y_line = cal.slope * log_c_line + cal.intercept

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "log(Rct) vs log([c])", "Résidus de régression",
    ])

    # Barre d'erreur sur log10(Rct) : d(log10 Rct) = dRct/(Rct·ln10) avec
    # dRct = err·Rct → err/ln10 (les Rct se simplifient).
    yerr = np.asarray(errs) / np.log(10.0)
    fig.add_trace(go.Scatter(
        x=log_c, y=log_rct, mode="markers", name="Données",
        error_y=dict(type="data", array=yerr, visible=True),
        marker=dict(color="#1a56db", size=9),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=log_c_line, y=y_line, mode="lines",
        name=f"R²={cal.r2:.3f}  y={cal.slope:.3f}x+{cal.intercept:.3f}",
        line=dict(color="#dc2626", width=2),
    ), row=1, col=1)

    residuals = log_rct - (cal.slope * log_c + cal.intercept)
    fig.add_trace(go.Scatter(
        x=log_c, y=residuals, mode="markers", name="Résidus",
        marker=dict(color="#7c3aed", size=9), showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), row=1, col=2)

    fig.update_xaxes(title_text="log([c] / M)", row=1, col=1)
    fig.update_yaxes(title_text="log(Rct / Ω)", row=1, col=1)
    fig.update_xaxes(title_text="log([c] / M)", row=1, col=2)
    fig.update_yaxes(title_text="Résidu log(Rct)", row=1, col=2)

    fig.update_layout(
        title=f"Calibration DRT ({model_name}) — log(Rct) vs log([c])",
        legend=dict(orientation="h", y=-0.2),
    )
    apply_theme_to_figure(fig, "light")
    return fig


# ── Reconstructions Nyquist (Randles vs DRT) ──────────────────────────────────

def reconstruction_comparison_figure(sessions: dict) -> go.Figure:
    """Comparaison Randles vs DRT, mesurée sur le spectre probe de chaque électrode.

    Style de ligne distinct par électrode (solide E1, tirets E2…),
    couleur distincte par méthode (Randles / DRT).
    """
    fig = go.Figure()
    line_dashes = ["solid", "dash", "dot", "dashdot"]
    method_colors = {"randles_full": "#dc2626", "drt_bayes": "#1a56db"}

    any_data = False
    for idx, (e, session) in enumerate(sorted(sessions.items())):
        probe = session.probe
        if probe is None:
            continue
        dash = line_dashes[idx % len(line_dashes)]

        fig.add_trace(go.Scatter(
            x=probe.Zre, y=probe.Zim, mode="markers",
            name=f"E{e} — mesuré",
            marker=dict(color="black", size=6, symbol="circle" if idx == 0 else "x"),
        ))
        any_data = True

        fr_r = probe.fit_results.get("randles_full")
        if fr_r is not None:
            fig.add_trace(go.Scatter(
                x=fr_r.Zfit_re, y=fr_r.Zfit_im, mode="lines",
                name=f"E{e} — Randles",
                line=dict(color=method_colors["randles_full"], dash=dash, width=2),
            ))

        fr_d = probe.fit_results.get("drt_bayes")
        if fr_d is not None:
            fig.add_trace(go.Scatter(
                x=fr_d.Zfit_re, y=fr_d.Zfit_im, mode="lines",
                name=f"E{e} — DRT",
                line=dict(color=method_colors["drt_bayes"], dash=dash, width=2),
            ))

    if not any_data:
        fig.add_annotation(
            text="Aucun spectre probe disponible.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13),
        )

    fig.update_layout(
        title="Reconstructions Nyquist — Randles vs DRT (probe)",
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
        hovermode="closest",
    )
    apply_theme_to_figure(fig, "light")
    return fig


def drt_reconstruction_figure_dual(
    spectrum: EISSpectrum, fr_drt=None, fr_randles=None, label: str = "",
) -> go.Figure:
    """Mesuré + reconstruction Randles + reconstruction DRT (3 séries).

    Extension de drt_reconstruction_figure pour accepter un second FitResult
    (Randles) en plus de la DRT. fr_drt et/ou fr_randles peuvent être None.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spectrum.Zre, y=spectrum.Zim, mode="markers", name="Mesuré",
        marker=dict(color="#1a56db", size=7),
    ))
    err_parts = []
    if fr_randles is not None:
        fig.add_trace(go.Scatter(
            x=fr_randles.Zfit_re, y=fr_randles.Zfit_im, mode="lines",
            name="Reconstruction Randles",
            line=dict(color="#dc2626", width=2),
        ))
        if fr_randles.reconstruction_error is not None:
            err_parts.append(f"Randles ε={fr_randles.reconstruction_error*100:.2f}%")
    if fr_drt is not None:
        fig.add_trace(go.Scatter(
            x=fr_drt.Zfit_re, y=fr_drt.Zfit_im, mode="lines",
            name="Reconstruction DRT",
            line=dict(color="#16a34a", width=2, dash="dash"),
        ))
        if fr_drt.reconstruction_error is not None:
            err_parts.append(f"DRT ε={fr_drt.reconstruction_error*100:.2f}%")

    fig.update_xaxes(title_text="Z' (Ω)")
    fig.update_yaxes(title_text="-Z'' (Ω)", scaleanchor="x")
    err_str = " — " + ", ".join(err_parts) if err_parts else ""
    fig.update_layout(
        title=f"Reconstruction Randles vs DRT — {label}{err_str}",
        legend=dict(orientation="h", y=-0.15),
    )
    apply_theme_to_figure(fig, "light")
    return fig


def open_reconstruction_matplotlib_window(sessions: dict) -> None:
    """Ouvre une fenêtre matplotlib (bloquante) empilant les graphes de l'onglet 3.

    Empile : (a) comparaison moyenne probe toutes électrodes, (b) reconstruction
    par électrode sur le probe moyen (Randles + DRT).
    """
    import matplotlib.pyplot as plt

    n_graphs = 1 + len(sessions)
    fig, axes = plt.subplots(nrows=n_graphs, ncols=1, figsize=(9, 4.2 * n_graphs))
    if n_graphs == 1:
        axes = [axes]

    # (a) comparaison globale
    ax = axes[0]
    line_dashes = ["-", "--", ":", "-."]
    method_colors = {"randles_full": "#dc2626", "drt_bayes": "#1a56db"}
    for idx, (e, session) in enumerate(sorted(sessions.items())):
        probe = session.probe
        if probe is None:
            continue
        dash = line_dashes[idx % len(line_dashes)]
        ax.plot(probe.Zre, probe.Zim, "o", color="black", markersize=4, label=f"E{e} — mesuré")
        fr_r = probe.fit_results.get("randles_full")
        if fr_r is not None:
            ax.plot(fr_r.Zfit_re, fr_r.Zfit_im, dash, color=method_colors["randles_full"],
                     label=f"E{e} — Randles")
        fr_d = probe.fit_results.get("drt_bayes")
        if fr_d is not None:
            ax.plot(fr_d.Zfit_re, fr_d.Zfit_im, dash, color=method_colors["drt_bayes"],
                     label=f"E{e} — DRT")
    ax.set_title("Reconstructions Nyquist — comparaison toutes électrodes (probe)")
    ax.set_xlabel("Re(Z) (Ω)")
    ax.set_ylabel("-Im(Z) (Ω)")
    ax.legend(fontsize=7)

    # (b) par électrode
    for i, (e, session) in enumerate(sorted(sessions.items())):
        ax = axes[i + 1]
        probe = session.probe
        if probe is None:
            ax.set_title(f"Électrode {e} — aucune donnée")
            continue
        ax.plot(probe.Zre, probe.Zim, "o", color="black", markersize=4, label="Mesuré")
        fr_r = probe.fit_results.get("randles_full")
        if fr_r is not None:
            ax.plot(fr_r.Zfit_re, fr_r.Zfit_im, "-", color="#dc2626", label="Randles")
        fr_d = probe.fit_results.get("drt_bayes")
        if fr_d is not None:
            ax.plot(fr_d.Zfit_re, fr_d.Zfit_im, "--", color="#1a56db", label="DRT")
        ax.set_title(f"Reconstruction probe — Électrode {e}")
        ax.set_xlabel("Re(Z) (Ω)")
        ax.set_ylabel("-Im(Z) (Ω)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    plt.show()


def open_calibration_matplotlib_window(sessions: dict) -> None:
    """Ouvre une fenêtre matplotlib (bloquante) empilant les courbes de calibration EIS.

    Une sous-figure par électrode présente, une courbe par méthode (randles_full,
    drt_bayes), reproduisant la logique de calibration_figure() en matplotlib.
    """
    import matplotlib.pyplot as plt

    n_graphs = max(len(sessions), 1)
    fig, axes = plt.subplots(nrows=n_graphs, ncols=1, figsize=(8, 5 * n_graphs))
    if n_graphs == 1:
        axes = [axes]

    for i, (e, session) in enumerate(sorted(sessions.items())):
        ax = axes[i]
        # Calcul délégué à core/calibration.py (même source que calibration_figure).
        cals = compute_calibration_all(session)
        if not cals:
            ax.set_title(f"Électrode {e} — pas de calibration")
            continue
        for cal in cals:
            ax.plot(cal.log_c, cal.y, "o", label=f"{cal.model} (données)")
            log_c_line = np.linspace(cal.log_c.min(), cal.log_c.max(), 200)
            ax.plot(log_c_line, cal.slope * log_c_line + cal.intercept, "-",
                     label=f"{cal.model} R²={cal.r2:.3f}")
        ax.set_title(f"Calibration — Électrode {e}")
        ax.set_xlabel("log([c] / M)")
        ax.set_ylabel("|Rct_probe − Rct_c| / |Rct_probe|")
        ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
