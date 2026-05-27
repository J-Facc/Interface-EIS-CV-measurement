"""Orchestrates the EIS pipeline: Import → Validate → Fit → Analyse."""

from datetime import datetime
from typing import Optional

from core.models import EISSession, EISSpectrum, ConcentrationGroup
from core.loader import load_spectrum, average_replicates
from core.logger import get_logger

log = get_logger("pipeline")


def run_pipeline(
    file_assignments: list,
    config: dict,
    active_models: Optional[list] = None,
) -> EISSession:
    """Run the full EIS analysis pipeline.

    Args:
        file_assignments: List of dicts, each with keys:
            - content (bytes): Raw file content.
            - filename (str): Original filename.
            - step (str): 'bare', 'probe', or 'hybridization'.
            - concentration (float): Analyte concentration (mol/L).
        config: App config dict from config_to_dict(load_config()).
        active_models: Model name list to run. None runs all registered models.

    Returns:
        EISSession with spectra and fit results populated.
    """
    from fits.registry import all_models, get_model

    session = EISSession(created_at=datetime.now(), config=config)

    models = (
        all_models()
        if active_models is None
        else [get_model(m) for m in active_models]
    )

    # Group file assignments by (step, concentration)
    groups: dict = {}
    for fa in file_assignments:
        key = (fa["step"], float(fa.get("concentration", 0.0)))
        groups.setdefault(key, []).append(fa)

    bare_spectra = []
    probe_spectra = []
    hybridization: dict = {}

    for (step, conc), fas in groups.items():
        loaded = []
        for fa in fas:
            try:
                sp = load_spectrum(
                    content=fa["content"],
                    label=fa["filename"],
                    concentration=conc,
                    step=step,
                    config=config,
                )
                loaded.append(sp)
                log.info(f"Loaded '{fa['filename']}' step={step} c={conc:.2e} M n={sp.n_points}")
            except Exception as e:
                log.error(f"Skipped '{fa['filename']}': {e}")

        if not loaded:
            continue

        averaged = average_replicates(loaded) if len(loaded) > 1 else loaded[0]

        if step == "bare":
            bare_spectra.append(averaged)
        elif step == "probe":
            probe_spectra.append(averaged)
        else:
            hybridization.setdefault(conc, []).append(averaged)

    session.bare = average_replicates(bare_spectra) if bare_spectra else None
    session.probe = average_replicates(probe_spectra) if probe_spectra else None

    for conc in sorted(hybridization.keys()):
        sp_list = hybridization[conc]
        spectrum = average_replicates(sp_list) if len(sp_list) > 1 else sp_list[0]

        fit_results = {}
        for model in models:
            try:
                fr = model.fit(spectrum, config)
                fit_results[model.name] = fr
                log.info(
                    f"Fit '{model.name}' [{conc:.2e} M]: "
                    f"Rct={fr.Rct:.1f} Ω chi2={fr.chi2:.3e} ok={fr.converged}"
                )
            except Exception as e:
                log.error(f"Fit '{model.name}' [{conc:.2e} M] failed: {e}")

        session.groups.append(
            ConcentrationGroup(
                concentration=conc,
                spectrum=spectrum,
                fit_results=fit_results,
            )
        )

    return session
