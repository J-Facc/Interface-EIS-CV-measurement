"""CV file loader: auto-detect separator, find E/I columns, return CVScan."""

import io
import numpy as np
import pandas as pd

from core.cv_models import CVScan

_E_ALIASES = {"ewe", "e", "potential", "voltage"}
_I_ALIASES = {"i", "current", "<i>"}


def _detect_separator(sample: str) -> str:
    for sep in (",", ";", "\t"):
        if sep in sample:
            return sep
    return ","


def _find_column(columns: list[str], aliases: set[str]) -> str | None:
    for col in columns:
        if col.strip().lower().split("/")[0].strip() in aliases:
            return col
    # looser: alias is a substring
    for col in columns:
        lower = col.strip().lower()
        for alias in aliases:
            if alias in lower:
                return col
    return None


def load_cv_file(content: bytes, label: str, concentration: float, step: str) -> CVScan:
    """Parse a CV file and return a CVScan."""
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
