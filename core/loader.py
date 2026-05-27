"""EIS data loader: CSV/TXT import, validation, sign correction, replicate averaging."""

import io
import numpy as np
import pandas as pd
from typing import Union

from core.models import EISSpectrum
from core.logger import get_logger

log = get_logger("loader")

_PARASITIC_FREQS_DEFAULT = [50.0, 100.0]
_PARASITIC_TOL_DEFAULT = 3.0


def _detect_separator(content: str) -> str:
    """Sniff the CSV column separator from the first non-comment line."""
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ["\t", ";", ",", " "]:
            if sep in line:
                return sep
    return ","


def _find_header_row(lines: list[str]) -> int:
    """Return the 0-based index of the header row for EC-Lab / generic CSV files."""
    keywords = {"frequency", "freq", "re(z)", "zreal", "z_re", "z'"}
    for i, line in enumerate(lines[:40]):
        low = line.lower()
        if any(kw in low for kw in keywords):
            return i
    return 0


def _normalise_col_name(col: str) -> str:
    return col.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


_COL_ALIASES = {
    "frequency_hz": ["frequency_hz", "freq", "frequency", "fhz", "f_hz"],
    "zreal_ohm": ["zreal_ohm", "zreal", "z_re", "z'", "rez", "re_z", "re_z_ohm", "reziohm"],
    "zimag_ohm": ["zimag_ohm", "zimag", "z_im", "z''", "imz", "im_z", "-im_z_ohm",
                  "-imziohm", "-imz", "-im_z"],
}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to standard names (frequency_hz, zreal_ohm, zimag_ohm).

    Falls back to positional mapping (cols 0, 1, 2) if name matching fails.

    Args:
        df: DataFrame with raw column names.

    Returns:
        DataFrame with standardised column names.

    Raises:
        ValueError: If columns cannot be identified.
    """
    norm_cols = {c: _normalise_col_name(c) for c in df.columns}
    rename_map = {}
    for std_name, aliases in _COL_ALIASES.items():
        for raw_col, normed in norm_cols.items():
            if normed in aliases and std_name not in rename_map.values():
                rename_map[raw_col] = std_name
                break

    if len(rename_map) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            log.warning("Column names not recognised; using positional mapping (cols 0,1,2)")
            rename_map = {
                numeric_cols[0]: "frequency_hz",
                numeric_cols[1]: "zreal_ohm",
                numeric_cols[2]: "zimag_ohm",
            }
        else:
            raise ValueError(
                f"Cannot identify frequency/Zreal/Zimag columns. "
                f"Found after normalisation: {list(norm_cols.values())}"
            )

    return df.rename(columns=rename_map)


def _clean_spectrum(
    f: np.ndarray,
    Zre: np.ndarray,
    Zim: np.ndarray,
    parasitic_freqs: list,
    tol: float,
) -> tuple:
    """Remove NaN/inf, suppress parasitic frequencies, correct sign, sort HF→BF.

    Args:
        f: Frequency array (Hz).
        Zre: Real impedance (Ω).
        Zim: Imaginary impedance (Ω).
        parasitic_freqs: List of frequencies to suppress (Hz).
        tol: Tolerance window around each parasitic frequency (Hz).

    Returns:
        Tuple (f, Zre, Zim) after cleaning.
    """
    mask = np.isfinite(f) & np.isfinite(Zre) & np.isfinite(Zim)
    f, Zre, Zim = f[mask], Zre[mask], Zim[mask]

    for fp in parasitic_freqs:
        keep = np.abs(f - fp) > tol
        f, Zre, Zim = f[keep], Zre[keep], Zim[keep]

    # EC-Lab convention: Zim exported as negative → make positive
    if len(Zim) > 0 and np.sum(Zim < 0) > np.sum(Zim > 0):
        Zim = -Zim
        log.info("Auto-corrected Zim sign (EC-Lab negative convention detected)")

    order = np.argsort(f)[::-1]  # descending: HF first
    return f[order], Zre[order], Zim[order]


def load_spectrum(
    content: Union[str, bytes],
    label: str,
    concentration: float = 0.0,
    step: str = "hybridization",
    config: dict = None,
) -> EISSpectrum:
    """Parse one CSV/TXT file into a validated EISSpectrum.

    Auto-detects separator and header row. Applies sign correction,
    parasitic frequency removal, and HF→BF sorting.

    Args:
        content: Raw file bytes or string.
        label: Display label (typically the filename).
        concentration: Analyte concentration in mol/L.
        step: Measurement step: 'bare', 'probe', or 'hybridization'.
        config: App config dict (for parasitic freq parameters).

    Returns:
        Validated EISSpectrum.

    Raises:
        ValueError: If fewer than 5 valid points remain after cleaning.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    parasitic_freqs = _PARASITIC_FREQS_DEFAULT
    tol = _PARASITIC_TOL_DEFAULT
    if config:
        fit_cfg = config.get("fit", {})
        parasitic_freqs = fit_cfg.get("n_freqs_parasites", _PARASITIC_FREQS_DEFAULT)
        tol = float(fit_cfg.get("tol_parasites", _PARASITIC_TOL_DEFAULT))

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

    try:
        df = _map_columns(df)
    except ValueError as e:
        raise ValueError(f"Fichier {label} : {e}") from e

    df = df[["frequency_hz", "zreal_ohm", "zimag_ohm"]].apply(pd.to_numeric, errors="coerce")

    f = df["frequency_hz"].values
    Zre = df["zreal_ohm"].values
    Zim = df["zimag_ohm"].values

    f, Zre, Zim = _clean_spectrum(f, Zre, Zim, parasitic_freqs, tol)

    if len(f) < 5:
        raise ValueError(
            f"Fichier {label} : moins de 5 points valides après nettoyage "
            f"(trouvé {len(f)} points)"
        )

    return EISSpectrum(
        label=label,
        f=f,
        Zre=Zre,
        Zim=Zim,
        concentration=concentration,
        step=step,
        n_points=len(f),
        source_files=[label],
    )


def average_replicates(spectra: list) -> EISSpectrum:
    """Average multiple EISSpectrum replicates point-by-point on a common frequency grid.

    If frequency grids differ, replicates are interpolated onto the first spectrum's grid.

    Args:
        spectra: List of EISSpectrum objects (same concentration and step).

    Returns:
        Averaged EISSpectrum.

    Raises:
        ValueError: If the list is empty.
    """
    if not spectra:
        raise ValueError("average_replicates: empty spectrum list")
    if len(spectra) == 1:
        return spectra[0]

    f_ref = spectra[0].f
    Zre_stack = []
    Zim_stack = []

    for sp in spectra:
        if len(sp.f) == len(f_ref) and np.allclose(sp.f, f_ref, rtol=0.01):
            Zre_stack.append(sp.Zre)
            Zim_stack.append(sp.Zim)
        else:
            # Interpolate onto reference grid (both sorted descending)
            sort_idx = np.argsort(sp.f)
            f_sorted = sp.f[sort_idx]
            Zre_interp = np.interp(f_ref, f_sorted, sp.Zre[sort_idx])
            Zim_interp = np.interp(f_ref, f_sorted, sp.Zim[sort_idx])
            Zre_stack.append(Zre_interp)
            Zim_stack.append(Zim_interp)

    ref = spectra[0]
    return EISSpectrum(
        label=ref.label + " (avg)",
        f=f_ref,
        Zre=np.mean(Zre_stack, axis=0),
        Zim=np.mean(Zim_stack, axis=0),
        concentration=ref.concentration,
        step=ref.step,
        n_points=len(f_ref),
        source_files=[s.label for s in spectra],
    )
