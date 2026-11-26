"""Figures matplotlib du rapport PDF — mêmes données que les figures Plotly, aucun calcul.

Backend ``Agg`` (aucune fenêtre). Chaque fonction dessine dans des ``Axes`` fournis par
``exports/report_pdf.py`` : groupes d'affichage de ``core.display``, ``DRTRecord`` de
``core.results``, courbes de ``core.calibration``, lignes de ``core.report_tables``.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_CMAP = plt.get_cmap("plasma")
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _step_color(step: str, conc, c_min: float, c_max: float):
    if conc and conc > 0:
        t = 0.5 if c_min >= c_max else (math.log10(conc) - math.log10(c_min)) / (math.log10(c_max) - math.log10(c_min))
        return _CMAP(0.85 * float(np.clip(t, 0, 1)))
    return "#6b7280" if step == "bare" else "black"


def _range(values):
    v = [c for c in values if c and c > 0]
    return (min(v), max(v)) if v else (1.0, 1.0)


def cv_axes(ax, groups: list, title: str = "", peaks: list = None) -> None:
    """``groups`` : sortie de ``core.display.cv_display_groups`` ; I affiché en µA."""
    c_min, c_max = _range([g.get("concentration") for g in groups])
    for g in groups:
        color = _step_color(g["step"], g.get("concentration"), c_min, c_max)
        for i, (_name, E, I) in enumerate(g["curves"]):
            ax.plot(E, np.asarray(I) * 1e6, color=color, lw=1.0 if len(g["curves"]) > 1 else 1.5,
                    alpha=0.8, label=g["label"] if i == 0 else None)
    for _n, e, i in peaks or []:
        ax.plot([e], [i * 1e6], "x", color="red", ms=6)
    ax.set_xlabel("E (V)")
    ax.set_ylabel("I (µA)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    if groups:
        ax.legend(fontsize=6, loc="best")


def nyquist_axes(ax, groups: list, title: str = "") -> None:
    """``groups`` : ``DisplayGroup`` de ``core.display.eis_display_groups`` (réplicats)."""
    c_min, c_max = _range([g.concentration for g in groups])
    for g in groups:
        color = _step_color(g.step, g.concentration, c_min, c_max)
        for i, sp in enumerate(g.replicates):
            ax.plot(sp.Zre, sp.Zim, "o-", color=color, ms=2, lw=0.8, alpha=0.8, label=g.label if i == 0 else None)
    ax.set_xlabel("Re(Z) (Ω)")
    ax.set_ylabel("−Im(Z) (Ω)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    if groups:
        ax.legend(fontsize=6, loc="best")


def bode_axes(ax_mod, ax_phase, groups: list, title: str = "") -> None:
    c_min, c_max = _range([g.concentration for g in groups])
    for g in groups:
        color = _step_color(g.step, g.concentration, c_min, c_max)
        for sp in g.replicates:
            zr, zi = np.asarray(sp.Zre, float), np.asarray(sp.Zim, float)
            ax_mod.loglog(sp.f, np.hypot(zr, zi), "-", color=color, lw=0.8, alpha=0.8)
            ax_phase.semilogx(sp.f, np.degrees(np.arctan2(zi, zr)), "-", color=color, lw=0.8, alpha=0.8)
    ax_mod.set_ylabel("|Z| (Ω)")
    ax_mod.set_title(title, fontsize=9)
    ax_phase.set_ylabel("−φ (°)")
    ax_phase.set_xlabel("f (Hz)")
    for a in (ax_mod, ax_phase):
        a.grid(alpha=0.3, which="both")


def drt_axes(ax, items: list, title: str = "") -> None:
    """``items`` : ``[(label, DRTRecord)]`` ; pic principal (★) et zone intégrée ombrée."""
    for k, (label, d) in enumerate(items):
        if d is None or d.tau is None:
            continue
        color = _COLORS[k % len(_COLORS)]
        order = np.argsort(d.tau)
        tau, g = np.asarray(d.tau)[order], np.asarray(d.gamma)[order]
        if d.gamma_lo is not None and d.gamma_hi is not None:
            ax.fill_between(tau, np.asarray(d.gamma_lo)[order], np.asarray(d.gamma_hi)[order], color=color, alpha=0.12)
        pk = d.peak
        if pk is not None and pk.ok:
            m = (tau >= pk.tau_lo) & (tau <= pk.tau_hi)
            ax.fill_between(tau[m], 0, g[m], color=color, alpha=0.3)
            ax.plot([pk.tau_peak], [pk.gamma_peak], "*", color=color, ms=9, mec="k", mew=0.5)
            label = f"{label} : Rct_DRT = {pk.rct:.4g} Ω"
        ax.semilogx(tau, g, color=color, lw=1.2, label=label)
    ax.set_xlabel("τ (s)")
    ax.set_ylabel("γ(τ) (Ω)")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    if items:
        ax.legend(fontsize=6, loc="best")


def calibration_axes(ax, curves: list, title: str = "", y_title: str = "") -> None:
    """Moyennes ± écart-type (ddof = 1), réplicats pâles, régression et R²."""
    for k, c in enumerate(curves):
        color = _COLORS[k % len(_COLORS)]
        inc = [p for p in c.points if p.included and np.isfinite(p.y)]
        if inc:
            ax.plot([math.log10(p.concentration) for p in inc], [p.y for p in inc], "o", color=color, alpha=0.3, ms=3)
        if not c.levels:
            continue
        x = np.array([lv.log_c for lv in c.levels])
        y = np.array([lv.mean for lv in c.levels])
        e = np.array([lv.std if np.isfinite(lv.std) else 0.0 for lv in c.levels])
        label = c.label + (f" (R² = {c.r2:.4f})" if c.has_regression else "")
        ax.errorbar(x, y, yerr=e, fmt="o", color=color, ms=5, capsize=3, label=label)
        if c.has_regression:
            xx = np.linspace(x.min() - 0.2, x.max() + 0.2, 20)
            ax.plot(xx, c.slope * xx + c.intercept, "--", color=color, lw=1)
    ax.set_xlabel("log10([c] / (mol/L))")
    ax.set_ylabel(y_title)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    if curves:
        ax.legend(fontsize=6, loc="best")


def _cell(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "—" if not math.isfinite(v) else f"{v:.4g}"
    return str(v)


def table_axes(ax, rows: list, columns: list, title: str = "", fontsize: float = 6.0,
               max_chars: int = 40) -> None:
    """Tableau (lignes de ``core.report_tables``) dans un axe sans repère."""
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")
    if not rows:
        ax.text(0.0, 0.9, "Aucune ligne.", fontsize=8)
        return
    cells = [[_cell(r.get(c))[:max_chars] for c in columns] for r in rows]
    t = ax.table(cellText=cells, colLabels=columns, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1.0, 1.15)
    for (i, _j), cell in t.get_celld().items():
        if i == 0:
            cell.set_facecolor(mcolors.to_rgba("#dde3ea"))


def text_axes(ax, lines: list, title: str = "") -> None:
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, loc="left", fontweight="bold")
    y = 0.95
    for line in lines:
        ax.text(0.0, y, line, fontsize=9, va="top", family="DejaVu Sans")
        y -= 0.035
