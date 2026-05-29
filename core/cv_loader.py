"""CV data loader: CSV/TXT import, column detection, unit conversion, replicate averaging."""

from __future__ import annotations

import io
from typing import Union

import numpy as np
import pandas as pd

from core.cv_models import CVScan
from core.logger import get_logger

log = get_logger("cv_loader")

_E_ALIASES = ["ewe", "e", "potential", "voltage", "ewe_v", "e_v"]
_I_ALIASES = ["i", "current", "<i>", "i_a", "i_ua", "i_ma", "courant"]


def _detect_separator(content: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ["\t", ";", ",", " "]:
            if sep in line:
                return sep
    return ","


def _find_header_row(lines: list[str]) -> int:
    keywords = {"ewe", "potential", "voltage", "current", "<i>", " i ", "courant"}
    for i, line in enumerate(lines[:40]):
        low = line.lower()
        if any(kw in low for kw in keywords):
            return i
    return 0


def _normalise(col: str) -> str:
    return col.strip().lower().replace(" ", "").replace("/", "").replace("(", "").replace(")", "")


def _find_columns(df: pd.DataFrame) -> tuple[str, str, bool]:
    """Return (E_col, I_col, is_uA) from the DataFrame."""
    norm = {c: _normalise(c) for c in df.columns}

    e_col = None
    for raw, n in norm.items():
        if any(n == alias or n.startswith(alias) for alias in _E_ALIASES):
            e_col = raw
            break

    i_col = None
    is_uA = False
    for raw, n in norm.items():
        if raw == e_col:
            continue
        if any(n == alias or n.startswith(alias) for alias in _I_ALIASES):
            i_col = raw
            is_uA = "µa" in raw.lower() or "ua" in raw.lower() or "µa" in raw or "uA" in raw
            break

    if e_col is None or i_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 2:
            log.warning("CV column names not recognised; using positional mapping (cols 0,1)")
            e_col, i_col = numeric[0], numeric[1]
            is_uA = False
        else:
            raise ValueError(
                f"Cannot identify E/I columns. Found: {list(df.columns)}"
            )

    return e_col, i_col, is_uA


def load_cv_file(
    content: Union[str, bytes],
    label: str,
    concentration: float,
    step: str,
) -> CVScan:
    """Parse one CSV/TXT file into a CVScan.

    Args:
        content: Raw file bytes or string.
        label: Display label (typically the filename).
        concentration: Analyte concentration in mol/L (0.0 for probe).
        step: 'probe' or 'hybridization'.

    Returns:
        CVScan sorted by increasing potential.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    lines = content.splitlines()
    header_row = _find_header_row(lines)
    sep = _detect_separator(content)

    try:
        df = pd.read_csv(
            io.StringIO(content),
            sep=sep,
            skiprows=header_row,
            comment="#",
            engine="python",
        )
    except Exception as e:
        raise ValueError(f"Fichier {label} : impossible de parser le CSV — {e}") from e

    e_col, i_col, is_uA = _find_columns(df)

    E = pd.to_numeric(df[e_col], errors="coerce").values
    I = pd.to_numeric(df[i_col], errors="coerce").values

    mask = np.isfinite(E) & np.isfinite(I)
    E, I = E[mask], I[mask]

    if is_uA:
        I = I * 1e-6
        log.info("CV %s: converted µA → A", label)

    order = np.argsort(E)
    E, I = E[order], I[order]

    if len(E) < 3:
        raise ValueError(f"Fichier {label} : moins de 3 points valides ({len(E)} trouvé)")

    return CVScan(
        label=label,
        E=E,
        I=I,
        concentration=concentration,
        step=step,
        source_files=[label],
    )


def average_cv_replicates(scans: list[CVScan]) -> CVScan:
    """Average multiple CVScan replicates on a common E grid.

    Args:
        scans: List of CVScan objects (same concentration and step).

    Returns:
        Averaged CVScan.
    """
    if not scans:
        raise ValueError("average_cv_replicates: empty scan list")
    if len(scans) == 1:
        return scans[0]

    E_ref = scans[0].E
    I_stack = []

    for sc in scans:
        if len(sc.E) == len(E_ref) and np.allclose(sc.E, E_ref, rtol=0.01):
            I_stack.append(sc.I)
        else:
            I_interp = np.interp(E_ref, sc.E, sc.I)
            I_stack.append(I_interp)

    ref = scans[0]
    return CVScan(
        label=ref.label + " (avg)",
        E=E_ref,
        I=np.mean(I_stack, axis=0),
        concentration=ref.concentration,
        step=ref.step,
        source_files=[s.label for s in scans],
    )
