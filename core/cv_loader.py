"""CV file loader: auto-detect separator, find E/I columns, return CVScan."""

import io
from typing import Optional

import numpy as np
import pandas as pd

from core.cv_models import CVScan
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
# Cycle détecté  -> points gardés dans leur ordre de mesure naturel, SANS tri.
# Pas de cycle   -> balayage simple : tri par E croissant (comportement
#                   historique, inoffensif puisque I y est déjà fonction de E).
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


def _order_points(E: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cycle complet : ordre de mesure conservé. Balayage simple : tri par E."""
    if len(split_branches(E)) > 1:
        return E, I
    order = np.argsort(E, kind="stable")
    return E[order], I[order]


def interp_on_reference(E_target: np.ndarray, E_ref: np.ndarray, I_ref: np.ndarray) -> np.ndarray:
    """Interpole la courbe de référence (E_ref, I_ref) aux potentiels E_target.

    Balayage simple des deux côtés : `np.interp`. Boucle : chaque branche de la
    cible est interpolée sur la branche de même rang de la référence (l'aller
    sur l'aller, le retour sur le retour) — un `np.interp` global sur un E non
    monotone n'a aucun sens (CV-1).

    Raises:
        ValueError: si cible et référence n'ont pas le même nombre de branches.
    """
    E_target = np.asarray(E_target, dtype=float)
    E_ref = np.asarray(E_ref, dtype=float)
    I_ref = np.asarray(I_ref, dtype=float)
    tb, rb = split_branches(E_target), split_branches(E_ref)
    if len(tb) == 1 and len(rb) == 1:
        order = np.argsort(E_ref, kind="stable")
        return np.interp(E_target, E_ref[order], I_ref[order])
    if len(tb) != len(rb):
        raise ValueError(
            f"Nombre de branches CV différent ({len(tb)} contre {len(rb)}) : "
            f"les courbes ne sont pas comparables point à point."
        )
    out = np.empty_like(E_target)
    for t, r in zip(tb, rb):
        order = np.argsort(E_ref[r], kind="stable")
        out[t] = np.interp(E_target[t], E_ref[r][order], I_ref[r][order])
    return out


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


def load_cv_file(
    content: bytes,
    label: str,
    concentration: float,
    step: str,
    warnings_out: Optional[list] = None,
) -> CVScan:
    """Parse a CV file and return a CVScan.

    Reads real EC-Lab ASCII exports via core.robust_loader.parse_eclab_file
    (FR decimal comma, tab/;/space delimiters, Windows encodings, name-based
    column mapping, mA→A unit conversion) and falls back to a generic pandas
    reader for other layouts.

    Args:
        content: Raw file bytes.
        label: Display label (typically the filename).
        concentration: Analyte concentration in mol/L.
        step: Measurement step ('probe', 'hybridization', 'bare').
        warnings_out: Optional list; parser warnings are appended to it when the
            robust parser is used, so callers can relay them to the UI.

    Returns:
        A CVScan with E in volts and I in amperes. A complete cycle (forward
        + return branch) keeps its measurement order; a simple sweep is sorted
        by ascending E (see « Détection de branches » above).

    Raises:
        ValueError: If E/I columns cannot be found, or if an EIS file was
            supplied to the CV loader.
    """
    # ── Parseur EC-Lab robuste en priorité ──────────────────────────────────
    pf = parse_robust(content)
    if pf is not None and pf.kind == "CV" and pf.n_rows > 0:
        if warnings_out is not None:
            warnings_out.extend(pf.warnings)
        E = np.asarray(pf.columns["Ewe"], dtype=float)
        I = np.asarray(pf.columns["I"], dtype=float)  # déjà converti en ampères
        mask = np.isfinite(E) & np.isfinite(I)
        E, I = E[mask], I[mask]
        E, I = _order_points(E, I)
        return CVScan(
            label=label,
            E=E,
            I=I,
            concentration=concentration,
            step=step,
            source_files=[label],
        )
    if pf is not None and pf.kind == "EIS":
        raise ValueError(
            f"Fichier {label} : spectre EIS détecté (colonnes {sorted(pf.columns)}), "
            f"pas une courbe CV. Utilisez le canal EIS."
        )

    # ── Repli : lecture pandas générique ────────────────────────────────────
    text = content.decode("utf-8", errors="replace")
    sep = _detect_separator(text[:2000])
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df.columns = [str(c) for c in df.columns]

    e_col = _find_column(list(df.columns), _E_ALIASES)
    i_col = _find_column(list(df.columns), _I_ALIASES)

    if e_col is None:
        raise ValueError(f"Colonne potentiel introuvable dans {label}. Colonnes : {list(df.columns)}")
    if i_col is None:
        raise ValueError(f"Colonne courant introuvable dans {label}. Colonnes : {list(df.columns)}")

    E = pd.to_numeric(df[e_col], errors="coerce").to_numpy(dtype=float)
    I = pd.to_numeric(df[i_col], errors="coerce").to_numpy(dtype=float)

    # Unit conversion based on column name
    col_lower = i_col.lower()
    if "ma" in col_lower:
        I = I * 1e-3
    elif "µa" in col_lower or "ua" in col_lower:
        I = I * 1e-6

    # Drop NaN rows
    mask = np.isfinite(E) & np.isfinite(I)
    E, I = E[mask], I[mask]

    # Cycle complet : ordre de mesure ; balayage simple : tri par E (CV-1)
    E, I = _order_points(E, I)

    return CVScan(
        label=label,
        E=E,
        I=I,
        concentration=concentration,
        step=step,
        source_files=[label],
    )


def average_cv_replicates(scans: list) -> CVScan:
    """Interpolate all scans onto the first scan's E grid and average point-by-point.

    Branch-aware for cycles (see interp_on_reference)."""
    if not scans:
        raise ValueError("Liste de scans vide.")
    if len(scans) == 1:
        return scans[0]

    ref = scans[0]
    E_grid = ref.E
    I_matrix = np.stack(
        [interp_on_reference(E_grid, s.E, s.I) for s in scans],
        axis=0,
    )
    I_mean = I_matrix.mean(axis=0)

    return CVScan(
        label="moyenne",
        E=E_grid,
        I=I_mean,
        concentration=ref.concentration,
        step=ref.step,
        source_files=[s.label for s in scans],
    )


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
