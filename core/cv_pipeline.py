"""CV analysis pipeline: load, group, average replicates, compute delta_signal."""

from __future__ import annotations

from itertools import groupby
from typing import Union

import numpy as np

from core.cv_loader import load_cv_file, average_cv_replicates
from core.cv_models import CVConcentrationGroup, CVSession
from core.logger import get_logger

log = get_logger("cv_pipeline")


def run_cv_pipeline(cv_assignments: list[dict]) -> CVSession:
    """Build a CVSession from raw file assignments.

    Args:
        cv_assignments: List of dicts with keys:
            content (bytes|str), filename (str), step (str), concentration (float).

    Returns:
        CVSession with probe scan and concentration groups.
    """
    # Group by (step, concentration)
    by_group: dict[tuple, list] = {}
    for a in cv_assignments:
        key = (a["step"], float(a["concentration"]))
        by_group.setdefault(key, []).append(a)

    probe_scan = None
    groups = []

    for (step, conc), assignments in sorted(by_group.items(), key=lambda x: (x[0][0], x[0][1])):
        scans = []
        for a in assignments:
            try:
                sc = load_cv_file(
                    content=a["content"],
                    label=a["filename"],
                    concentration=conc,
                    step=step,
                )
                scans.append(sc)
            except Exception as exc:
                log.warning("Skipping CV file %s: %s", a["filename"], exc)

        if not scans:
            continue

        averaged = average_cv_replicates(scans)

        if step == "probe":
            probe_scan = averaged
        else:
            groups.append((conc, averaged))

    session = CVSession(probe=probe_scan)

    if probe_scan is not None:
        for conc, scan in groups:
            E_min = max(probe_scan.E.min(), scan.E.min())
            E_max = min(probe_scan.E.max(), scan.E.max())
            E_grid = np.linspace(E_min, E_max, 500)

            I_probe_interp = np.interp(E_grid, probe_scan.E, probe_scan.I)
            I_conc_interp = np.interp(E_grid, scan.E, scan.I)

            with np.errstate(divide="ignore", invalid="ignore"):
                delta = np.where(
                    I_probe_interp != 0,
                    np.abs(I_probe_interp - I_conc_interp) / np.abs(I_probe_interp),
                    np.nan,
                )

            session.groups.append(
                CVConcentrationGroup(concentration=conc, scan=scan, delta_signal=delta)
            )
    else:
        for conc, scan in groups:
            session.groups.append(
                CVConcentrationGroup(
                    concentration=conc,
                    scan=scan,
                    delta_signal=np.full(len(scan.E), np.nan),
                )
            )

    return session
