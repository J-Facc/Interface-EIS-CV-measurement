"""Figures Plotly EIS : Nyquist et Bode (données nettoyées), fit du circuit, DRT (pic principal),
sélection de points.

Aucun import Streamlit, aucun calcul métier : les figures reçoivent des données déjà produites
(``core/display.py``, ``core/results.py``). Les figures d'étalonnage sont dans
``plotting/calibration_plots.py``.
"""

import numpy as np
import plotly.colors as _pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotting.theme import apply_theme_to_figure, get_theme


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


# ── Nyquist par électrode ─────────────────────────────────────────────────────────


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


# ── Calibration ─────────────────────────────────────────────────────────────


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


# ── DRT : vue agrégée (variabilité inter-réplicats) ──────────────────────────────────


# ── Calibration : log10(Rct) vs log10([c]), un jeu de points par méthode ─────────────


# ═════════════════════════════════════════════════════════════════════════════
# v3 — Nyquist d'un spectre avec sélection de points (exclusion manuelle)
# ═════════════════════════════════════════════════════════════════════════════

def nyquist_points_figure(f, Zre, Zim, excluded=None, title: str = "", theme_mode: str = "light") -> go.Figure:
    """Nyquist d'UN spectre nettoyé, points sélectionnables (lasso/boîte) ; les points
    exclus sont dessinés en croix rouges. ``point_index`` de la sélection = indice du point."""
    f, Zre, Zim = (np.asarray(a, dtype=float) for a in (f, Zre, Zim))
    excluded = sorted({int(i) for i in (excluded or []) if 0 <= int(i) < len(f)})
    fig = go.Figure(go.Scatter(
        x=Zre, y=Zim, mode="lines+markers", name="points", customdata=f,
        marker=dict(size=7), line=dict(width=1),
        hovertemplate="Re(Z) = %{x:.1f} Ω<br>−Im(Z) = %{y:.1f} Ω<br>f = %{customdata:.3e} Hz<extra></extra>",
    ))
    if excluded:
        fig.add_trace(go.Scatter(x=Zre[excluded], y=Zim[excluded], mode="markers", name="exclus",
                                 marker=dict(symbol="x", size=11, color="red"), hoverinfo="skip"))
    fig.update_layout(title=title, xaxis_title="Re(Z) (Ω)", yaxis_title="−Im(Z) (Ω)", dragmode="lasso",
                      height=420, hovermode="closest")
    apply_theme_to_figure(fig, theme_mode)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# v3 — DRT : γ(τ) vs τ (log), pic principal et zone intégrée (Rct_DRT)
# ═════════════════════════════════════════════════════════════════════════════

def drt_peak_figure(items: list, title: str = "DRT — γ(τ)", theme_mode: str = "light") -> go.Figure:
    """γ(τ) (Ω, échelle linéaire) vs τ (s, échelle log) ; pic principal marqué (★) et zone
    intégrée (aire = Rct_DRT) ombrée ; bande de crédibilité 95 % en mode 'sample'.

    Args:
        items: ``[(label, DRTRecord)]`` (``core.results``) — rien n'est recalculé ici.
    """
    theme = get_theme(theme_mode)
    fig = go.Figure()
    for k, (label, d) in enumerate(items):
        if d is None or d.tau is None or d.gamma is None:
            continue
        color = theme["colors"][k % len(theme["colors"])]
        tau, g = np.asarray(d.tau, float), np.asarray(d.gamma, float)
        order = np.argsort(tau)
        tau, g = tau[order], g[order]
        if d.gamma_lo is not None and d.gamma_hi is not None:
            lo, hi = np.asarray(d.gamma_lo, float)[order], np.asarray(d.gamma_hi, float)[order]
            fig.add_trace(go.Scatter(x=np.concatenate([tau, tau[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                                     fill="toself", line=dict(width=0), fillcolor=_hex_to_rgba(color, 0.15),
                                     name=f"{label} — IC 95 %", hoverinfo="skip", legendgroup=label))
        pk = d.peak
        if pk is not None and pk.ok:
            m = (tau >= pk.tau_lo) & (tau <= pk.tau_hi)
            fig.add_trace(go.Scatter(x=tau[m], y=g[m], fill="tozeroy", mode="none",
                                     fillcolor=_hex_to_rgba(color, 0.30), legendgroup=label, showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[pk.tau_peak], y=[pk.gamma_peak], mode="markers", legendgroup=label,
                                     marker=dict(symbol="star", size=13, color=color,
                                                 line=dict(color=theme["text"], width=1)),
                                     name=f"{label} — pic : Rct_DRT = {pk.rct:.4g} Ω",
                                     hovertemplate=(f"<b>{label}</b><br>τ = %{{x:.3e}} s<br>"
                                                    f"Rct_DRT = {pk.rct:.4g} Ω<extra></extra>")))
        fig.add_trace(go.Scatter(x=tau, y=g, mode="lines", name=f"{label} ({d.mode})", legendgroup=label,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"<b>{label}</b><br>τ = %{{x:.3e}} s<br>γ = %{{y:.4g}} Ω<extra></extra>"))
    for k, (label, d) in enumerate(items[:1]):
        if d is not None and d.peak is not None and d.peak.tau_window:
            lo, hi = d.peak.tau_window
            fig.add_vrect(x0=lo, x1=hi, fillcolor="gray", opacity=0.06, line_width=0,
                          annotation_text="fenêtre de recherche du pic", annotation_position="top left")
    fig.update_layout(title=title, xaxis=dict(title="τ (s)", type="log"), yaxis=dict(title="γ(τ) (Ω)"),
                      hovermode="closest", height=460, legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0),
                      margin=dict(r=160))
    apply_theme_to_figure(fig, theme_mode)
    fig.update_xaxes(type="log")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# v3 — Moyennes de réplicats (vue par canal, vue synthétique)
# ═════════════════════════════════════════════════════════════════════════════

def _rgba_any(color: str, alpha: float) -> str:
    """Couleur (#hex, rgb(…), « black ») → rgba translucide."""
    if color.startswith("#"):
        return _hex_to_rgba(color, alpha)
    if color == "black":
        return f"rgba(0,0,0,{alpha})"
    return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")


def _mean_color(item: dict, c_min: float, c_max: float) -> str:
    class _G:
        concentration = item.get("concentration") or 0.0
        step = item["step"] if item["step"] in ("bare", "probe") else "hybridization"
    return _group_color(_G, c_min, c_max)


def nyquist_means_figure(items: list, title: str = "", theme_mode: str = "light") -> go.Figure:
    """Nyquist de spectres MOYENS (``core.display`` : electrode_means / synthesis_view).

    Couleur = étape (concentration), style de trait = ``item["dash"]`` ; ``item["band"]`` : barres
    d'erreur ± écart-type sur Re et −Im."""
    fig = go.Figure()
    c_min, c_max = _concentration_range([type("G", (), {"concentration": i.get("concentration") or 0})
                                         for i in items])
    for it in items:
        m, color = it["mean"], _mean_color(it, c_min, c_max)
        band = it.get("band") and m["n"] > 1
        fig.add_trace(go.Scatter(
            x=m["Zre"], y=m["Zim"], mode="lines+markers", name=f"{it['label']} (n={m['n']})",
            line=dict(color=color, width=2, dash=it.get("dash", "solid")), marker=dict(color=color, size=4),
            error_x=dict(type="data", array=m["sd_re"], visible=bool(band), color=color, thickness=0.8, width=0),
            error_y=dict(type="data", array=m["sd_im"], visible=bool(band), color=color, thickness=0.8, width=0),
            customdata=m["f"], hovertemplate=(f"<b>{it['label']}</b><br>x = %{{x:.4g}}<br>y = %{{y:.4g}}"
                                              "<br>f = %{customdata:.3e} Hz<extra></extra>")))
    xt, yt = "Re(Z) (Ω)", "−Im(Z) (Ω)"
    fig.update_layout(title=title, xaxis=dict(title=xt, rangemode="tozero"),
                      yaxis=dict(title=yt, rangemode="tozero"), hovermode="closest", height=460,
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0), margin=dict(r=160))
    apply_theme_to_figure(fig, theme_mode)
    return fig


def bode_means_figure(items: list, title: str = "", theme_mode: str = "light") -> go.Figure:
    """|Z| moyen vs f (log-log) avec zone d'erreur ± écart-type (``item["band"]``)."""
    fig = go.Figure()
    c_min, c_max = _concentration_range([type("G", (), {"concentration": i.get("concentration") or 0})
                                         for i in items])
    for it in items:
        m, color = it["mean"], _mean_color(it, c_min, c_max)
        if it.get("band") and m["n"] > 1:
            lo = np.maximum(m["mod"] - m["sd_mod"], 1e-12)
            hi = m["mod"] + m["sd_mod"]
            fig.add_trace(go.Scatter(x=np.concatenate([m["f"], m["f"][::-1]]), y=np.concatenate([hi, lo[::-1]]),
                                     fill="toself", fillcolor=_rgba_any(color, 0.18),
                                     line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=it["label"]))
        fig.add_trace(go.Scatter(x=m["f"], y=m["mod"], mode="lines", name=f"{it['label']} (n={m['n']})",
                                 legendgroup=it["label"], line=dict(color=color, width=2, dash=it.get("dash", "solid")),
                                 hovertemplate=f"<b>{it['label']}</b><br>f = %{{x:.3e}} Hz<br>y = %{{y:.4g}}<extra></extra>"))
    ytype = "log"
    fig.update_layout(title=title, xaxis=dict(title="Fréquence (Hz)", type="log"),
                      yaxis=dict(title="|Z| (Ω)", type=ytype),
                      hovermode="closest", height=460,
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0), margin=dict(r=160))
    apply_theme_to_figure(fig, theme_mode)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type=ytype)
    return fig
