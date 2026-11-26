"""Chargeur CV (v3) : séparateur et colonnes auto-détectés, courant en A, cycle retenu,
ordre chronologique (``load_cv_measurement``) ; détection des branches (``split_branches``)."""

import io

import numpy as np
import pandas as pd

from core.loader import parse_robust

_E_ALIASES = {"ewe", "e", "potential", "voltage"}
_I_ALIASES = {"i", "current", "<i>"}

# ── Détection de branches (CV-1) ─────────────────────────────────────────────
#
# Un voltammogramme cyclique est une BOUCLE : branche aller (E croissant) puis
# branche retour (E décroissant). Trier les points par potentiel croissant
# entrelace les deux branches et détruit la courbe (I n'est plus une fonction
# de E). La détection est donc faite sur l'ORDRE DE MESURE :
#
#   1. on parcourt E dans l'ordre du fichier et on repère les points de
#      rebroussement (zigzag avec hystérésis : E doit reculer d'au moins
#      _BRANCH_REVERSAL_FRACTION de l'étendue totale pour compter, ce qui
#      ignore le bruit de mesure) ;
#   2. les points de rebroussement découpent le signal en branches monotones ;
#   3. il y a cycle si on obtient au moins 2 branches, chacune d'au moins
#      _BRANCH_MIN_POINTS points. Un rebroussement isolé à 1-2 points est du
#      bruit ou un fichier désordonné, pas une branche.
#
# En v3, les points sont TOUJOURS gardés dans leur ordre de mesure (chronologique) ;
# ``split_branches`` sert à repérer plusieurs cycles quand le fichier n'a pas de colonne
# « cycle number ».
#
# Le point de rebroussement appartient à la branche qu'il termine. Un fichier à
# plusieurs cycles donne plus de deux branches (aller, retour, aller, ...).

_BRANCH_REVERSAL_FRACTION = 0.05
_BRANCH_MIN_POINTS = 5


def split_branches(E: np.ndarray) -> list[slice]:
    """Découpe un signal de potentiel (ordre de mesure) en branches monotones.

    Returns:
        Liste de slices couvrant tout `E`. Un seul slice = balayage simple
        (aucun cycle complet détecté, voir le commentaire de section).
    """
    E = np.asarray(E, dtype=float)
    n = len(E)
    whole = [slice(0, n)]
    if n < 2 * _BRANCH_MIN_POINTS:
        return whole
    span = float(np.max(E) - np.min(E))
    if span <= 0:
        return whole
    thr = _BRANCH_REVERSAL_FRACTION * span

    # Zigzag : `turns` reçoit l'indice de chaque extremum confirmé.
    turns: list[int] = []
    direction = 0            # 0 = pas encore décidée, +1 monte, -1 descend
    ext_i = 0                # indice de l'extremum courant dans le sens `direction`
    lo_i = hi_i = 0          # extrema provisoires tant que la direction est inconnue
    for i in range(1, n):
        if direction == 0:
            if E[i] > E[hi_i]:
                hi_i = i
            if E[i] < E[lo_i]:
                lo_i = i
            if E[hi_i] - E[lo_i] > thr:
                # direction fixée par l'extremum atteint en dernier
                direction = 1 if hi_i > lo_i else -1
                ext_i = hi_i if direction == 1 else lo_i
        elif direction == 1:
            if E[i] > E[ext_i]:
                ext_i = i
            elif E[ext_i] - E[i] > thr:
                turns.append(ext_i)
                direction, ext_i = -1, i
        else:
            if E[i] < E[ext_i]:
                ext_i = i
            elif E[i] - E[ext_i] > thr:
                turns.append(ext_i)
                direction, ext_i = 1, i

    bounds = [0, *(t + 1 for t in turns), n]
    branches = [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    if len(branches) < 2 or any(b.stop - b.start < _BRANCH_MIN_POINTS for b in branches):
        return whole
    return branches


def _detect_separator(sample: str) -> str:
    for sep in (",", ";", "\t"):
        if sep in sample:
            return sep
    return ","


def _find_column(columns: list[str], aliases: set[str]) -> str | None:
    # Match exact sur le nom nettoyé (unité après '/' retirée), p. ex.
    # "Ewe/V" -> "ewe", "I/mA" -> "i", "current/µA" -> "current".
    # Pas de repli "sous-chaîne" : il acceptait à tort des colonnes comme
    # "voltage_x" (contient "voltage"/"e") ou "time" (contient "i"), d'où de
    # faux positifs silencieux sur des fichiers sans vraie colonne E/I.
    for col in columns:
        if col.strip().lower().split("/")[0].strip() in aliases:
            return col
    return None


# ═════════════════════════════════════════════════════════════════════════════
# v3 — Nettoyage d'une CV pour une Measurement (ordre chronologique, choix du cycle)
# ═════════════════════════════════════════════════════════════════════════════

_CYCLE_ALIASES = {"cycle", "cycle number", "cyclenumber", "cycle_number"}


def _select_cycle(cycle: np.ndarray, which) -> tuple:
    """Masque du cycle retenu et son numéro. ``which`` : 'last' | 'first' | entier ≥ 1."""
    cycles = sorted({int(c) for c in cycle[np.isfinite(cycle)]})
    if not cycles:
        raise ValueError("colonne de cycle vide")
    if which in (None, "last"):
        num = cycles[-1]
    elif which == "first":
        num = cycles[0]
    else:
        num = int(which)
        if num not in cycles:
            raise ValueError(f"cycle {num} absent (cycles présents : {', '.join(map(str, cycles))})")
    return cycle == num, num, cycles


def load_cv_measurement(content, label: str, cycle="last") -> dict:
    """Nettoie une CV (export EC-Lab texte, ou CSV ``E, I, cycle`` de l'archive).

    * séparateur, décimales et colonnes auto-détectés (``core.robust_loader``) : Ewe/V,
      I ou <I> (mA, µA… convertis en **A**), numéro de cycle (« cycle number ») ;
    * le cycle ``cycle`` est conservé ('last' par défaut, 'first', ou un numéro) ;
    * l'ordre de mesure est conservé (ordre chronologique, JAMAIS trié par potentiel :
      une CV est une boucle).

    Returns:
        ``{"E", "I", "cycle" (np.ndarray), "cycle_kept": int, "cycles": [int], "warnings": [str]}``.

    Raises:
        ValueError: colonnes E/I introuvables, fichier EIS, cycle absent, < 5 points.
    """
    warnings: list = []
    pf = parse_robust(content)
    if pf is not None and pf.kind == "EIS":
        raise ValueError(f"Fichier {label} : spectre EIS détecté, pas une courbe CV.")
    if pf is not None and pf.kind == "CV" and pf.n_rows > 0:
        warnings.extend(pf.warnings)
        E = np.asarray(pf.columns["Ewe"], dtype=float)
        I = np.asarray(pf.columns["I"], dtype=float)
        cyc = np.asarray(pf.columns["cycle"], dtype=float) if "cycle" in pf.columns else None
    else:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
        sep = _detect_separator(text[:2000])
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        cols = [str(c) for c in df.columns]
        e_col = _find_column(cols, _E_ALIASES)
        i_col = _find_column(cols, _I_ALIASES)
        if e_col is None or i_col is None:
            raise ValueError(f"Fichier {label} : colonnes potentiel/courant introuvables (colonnes : {cols}).")
        c_col = next((c for c in cols if c.strip().lower() in _CYCLE_ALIASES), None)
        E = pd.to_numeric(df[e_col], errors="coerce").to_numpy(dtype=float)
        I = pd.to_numeric(df[i_col], errors="coerce").to_numpy(dtype=float)
        low = i_col.lower()
        if "ma" in low:
            I = I * 1e-3
        elif "µa" in low or "ua" in low:
            I = I * 1e-6
        cyc = pd.to_numeric(df[c_col], errors="coerce").to_numpy(dtype=float) if c_col else None

    if cyc is None:
        cyc = np.ones_like(E)
        if len(split_branches(E)) > 3:
            warnings.append("Aucune colonne « cycle number » : plusieurs cycles semblent présents, la courbe "
                            "entière est conservée (cycle 1).")
    mask = np.isfinite(E) & np.isfinite(I) & np.isfinite(cyc)
    if (~mask).any():
        warnings.append(f"{int((~mask).sum())} point(s) non numérique(s) retiré(s).")
    E, I, cyc = E[mask], I[mask], cyc[mask]
    try:
        keep, num, cycles = _select_cycle(cyc, cycle)
    except ValueError as exc:
        raise ValueError(f"Fichier {label} : {exc}.") from exc
    E, I, cyc = E[keep], I[keep], cyc[keep]
    if len(E) < 5:
        raise ValueError(f"Fichier {label} : moins de 5 points dans le cycle {num}.")
    return {"E": E, "I": I, "cycle": cyc.astype(float), "cycle_kept": int(num),
            "cycles": cycles, "warnings": warnings}
