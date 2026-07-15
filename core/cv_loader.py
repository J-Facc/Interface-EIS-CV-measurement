"""CV file loader: auto-detect separator, find E/I columns, return CVScan."""

import io
from typing import Optional

import numpy as np
import pandas as pd

from core.cv_models import CVScan
from core.loader import parse_robust
from core.models import CVCurve

_E_ALIASES = {"ewe", "e", "potential", "voltage"}
_I_ALIASES = {"i", "current", "<i>"}


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
        A CVScan with E in volts and I in amperes, sorted by ascending E.

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
        order = np.argsort(E)
        return CVScan(
            label=label,
            E=E[order],
            I=I[order],
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

    # Sort by ascending potential
    order = np.argsort(E)
    E, I = E[order], I[order]

    return CVScan(
        label=label,
        E=E,
        I=I,
        concentration=concentration,
        step=step,
        source_files=[label],
    )


def load_cv_curve(content: bytes, label: str, warnings_out: Optional[list] = None) -> CVCurve:
    """Parse a CV file into a lightweight CVCurve(Ewe, I, label).

    Thin wrapper over load_cv_file for callers that only need the raw (Ewe, I)
    curve — e.g. to stash a detected CV file in session without plotting it,
    before a dedicated CV view exists. `I` is already in amperes. For full CV
    analysis (concentration, step, replicates), use load_cv_file → CVScan.

    Args:
        content: Raw file bytes.
        label: Display label (typically the filename).
        warnings_out: Optional list; parser warnings are appended to it.

    Returns:
        A CVCurve with Ewe in volts and I in amperes.
    """
    scan = load_cv_file(content, label=label, concentration=0.0, step="cv", warnings_out=warnings_out)
    return CVCurve(Ewe=scan.E, I=scan.I, label=label)


def average_cv_replicates(scans: list) -> CVScan:
    """Interpolate all scans onto the first scan's E grid and average point-by-point."""
    if not scans:
        raise ValueError("Liste de scans vide.")
    if len(scans) == 1:
        return scans[0]

    ref = scans[0]
    E_grid = ref.E
    I_matrix = np.stack(
        [np.interp(E_grid, s.E, s.I) for s in scans],
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
