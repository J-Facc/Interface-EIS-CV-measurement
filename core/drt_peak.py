"""Rct par DRT = aire du pic PRINCIPAL de γ(τ) (v3) — post-traitement pur, aucun import Streamlit.

Sur γ(τ) en ln τ (``drt.engine`` : τ en secondes, γ en Ω, ∫ γ d(ln τ) = résistance) :

1. détection des pics (``scipy.signal.find_peaks``, hauteur ≥ ``rel_height`` × max γ) ;
2. candidats = pics dont τ est dans ``tau_window`` (``fit.drt.tau_window``, à calibrer) ;
3. pic retenu = le plus HAUT des candidats ; les autres candidats sont signalés ;
4. bornes d'intégration = minima locaux qui encadrent ce pic (descente depuis le sommet
   jusqu'à la première remontée de γ de chaque côté) ;
5. ``Rct_DRT = ∫ γ d(ln τ)`` entre ces bornes (trapèzes).

Le moteur DRT, lui, rapporte toujours son Rct « pic pénultième ±3 en ln τ »
(``FitResult.target_value``, drt/VALIDATION_REGLAGES.md) : il reste affiché en colonne de
contrôle. Ce module ne modifie pas le moteur.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

#: Hauteur minimale d'un pic, relative au maximum de γ (écarte les bosses numériques).
DEFAULT_REL_HEIGHT = 0.01


@dataclass
class MainPeak:
    """Pic principal de γ(τ) et son aire (Ω).

    Les indices se rapportent aux tableaux ``tau``/``gamma`` TELS QUE FOURNIS.
    ``ok = False`` : aucun pic dans la fenêtre — ``rct`` vaut NaN et ``message`` dit pourquoi.
    """

    ok: bool
    rct: float = float("nan")
    tau_peak: float = float("nan")
    gamma_peak: float = float("nan")
    idx_peak: Optional[int] = None
    idx_lo: Optional[int] = None
    idx_hi: Optional[int] = None
    tau_lo: float = float("nan")
    tau_hi: float = float("nan")
    tau_window: list = field(default_factory=list)
    other_candidates: list = field(default_factory=list)   # [{"tau", "gamma"}] autres pics de la fenêtre
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["MainPeak"]:
        if not d:
            return None
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("rct", "tau_peak", "gamma_peak", "tau_lo", "tau_hi"):
            if known.get(k) is None:
                known[k] = float("nan")        # NaN sérialisé en null (JSON)
        return cls(**known)


def _walk_down(g: np.ndarray, i: int, step: int) -> int:
    """Depuis le sommet ``i``, descend dans le sens ``step`` (±1) jusqu'au minimum local."""
    n = len(g)
    j = i
    while 0 <= j + step < n and g[j + step] <= g[j]:
        j += step
    return j


def main_peak(tau: Sequence[float], gamma: Sequence[float], tau_window: Sequence[float],
              rel_height: float = DEFAULT_REL_HEIGHT) -> MainPeak:
    """Pic principal de γ(τ) dans ``tau_window`` = [τ_min, τ_max] (s) et son aire."""
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    lo_w, hi_w = float(tau_window[0]), float(tau_window[1])
    base = MainPeak(ok=False, tau_window=[lo_w, hi_w])
    if tau.size < 3 or tau.size != gamma.size or not np.all(np.isfinite(tau)) or np.any(tau <= 0):
        base.message = "DRT absente ou invalide : Rct_DRT (pic principal) non calculable."
        return base
    order = np.argsort(tau)                    # travail en ln τ croissant
    t, g = tau[order], np.nan_to_num(gamma[order], nan=0.0)
    ln_t = np.log(t)
    gmax = float(np.max(g))
    if not gmax > 0:
        base.message = "γ(τ) nulle ou négative : aucun pic."
        return base
    peaks, _ = find_peaks(g, height=rel_height * gmax)
    cands = [int(p) for p in peaks if lo_w <= t[p] <= hi_w]
    if not cands:
        base.message = (f"Aucun pic de γ(τ) dans la fenêtre τ ∈ [{lo_w:.2g} ; {hi_w:.2g}] s "
                        "(fit.drt.tau_window) : Rct_DRT non calculé.")
        return base
    best = max(cands, key=lambda p: g[p])
    i_lo, i_hi = _walk_down(g, best, -1), _walk_down(g, best, +1)
    area = float(np.trapezoid(g[i_lo:i_hi + 1], ln_t[i_lo:i_hi + 1])) if i_hi > i_lo else float(g[best])
    others = [{"tau": float(t[p]), "gamma": float(g[p])} for p in cands if p != best]
    msg = ""
    if others:
        msg = (f"{len(others)} autre(s) pic(s) dans la fenêtre (τ = "
               + ", ".join(f"{o['tau']:.2g} s" for o in others) + ") : le plus haut a été retenu.")
    return MainPeak(ok=True, rct=area, tau_peak=float(t[best]), gamma_peak=float(g[best]),
                    idx_peak=int(order[best]), idx_lo=int(order[i_lo]), idx_hi=int(order[i_hi]),
                    tau_lo=float(t[i_lo]), tau_hi=float(t[i_hi]), tau_window=[lo_w, hi_w],
                    other_candidates=others, message=msg)
