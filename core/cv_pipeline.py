"""CV analysis pipeline: group files, load, average replicates, compute delta_signal."""

from collections import defaultdict
import numpy as np

from core.cv_loader import load_cv_file, average_cv_replicates
from core.cv_models import CVConcentrationGroup, CVSession


def run_cv_pipeline(cv_assignments: list) -> CVSession:
    """
    cv_assignments: list of dicts {content, filename, step, concentration}
    step = 'probe' ou 'hybridization'
    """
    # Group by (step, concentration)
    groups: dict = defaultdict(list)
    for item in cv_assignments:
        key = (item["step"], float(item["concentration"]))
        groups[key].append(item)

    # Load and average probe
    probe_items = groups.get(("probe", 0.0), [])
    if not probe_items:
        # probe may have non-zero concentration key — find any probe step
        probe_items = [v for (s, _), v in groups.items() if s == "probe"]
        probe_items = probe_items[0] if probe_items else []

    probe_scans = [
        load_cv_file(item["content"], item["filename"], item["concentration"], "probe")
        for item in probe_items
    ]
    probe_scan = average_cv_replicates(probe_scans) if probe_scans else None

    # Load hybridization groups
    concentration_groups: list = []
    for (step, conc), items in groups.items():
        if step != "hybridization":
            continue
        scans = [
            load_cv_file(item["content"], item["filename"], conc, "hybridization")
            for item in items
        ]
        avg_scan = average_cv_replicates(scans)

        # Compute delta_signal = |I_probe_interp - I_conc| / |I_probe_interp|
        if probe_scan is not None:
            I_probe_interp = np.interp(avg_scan.E, probe_scan.E, probe_scan.I)
            with np.errstate(invalid="ignore", divide="ignore"):
                delta = np.abs(I_probe_interp - avg_scan.I) / np.abs(I_probe_interp)
                delta = np.where(I_probe_interp == 0, np.nan, delta)
        else:
            delta = np.full_like(avg_scan.I, np.nan)

        concentration_groups.append(
            CVConcentrationGroup(
                concentration=conc,
                scan=avg_scan,
                delta_signal=delta,
            )
        )

    # Sort by concentration
    concentration_groups.sort(key=lambda g: g.concentration)

    return CVSession(probe=probe_scan, groups=concentration_groups)
