"""Orchestrates the EIS pipeline: Import → Validate → Fit → Analyse."""

from datetime import datetime
from typing import Optional

import numpy as np

from core.models import EISSession, EISSpectrum, ConcentrationGroup
from core.loader import load_spectrum, average_replicates
from core.logger import get_logger
from core.validator import validate_replicate_group

log = get_logger("pipeline")


def validate_session(replicate_groups: dict, config) -> dict:
    """
    Valide chaque groupe de réplicats avant analyse.

    Parameters
    ----------
    replicate_groups : dict label → {"f": list[np.ndarray],
                                      "zre": list[np.ndarray],
                                      "zim": list[np.ndarray]}
    config : AppSettings

    Returns
    -------
    dict label → ValidationResult
    """
    results = {}
    for label, group in replicate_groups.items():
        vr = validate_replicate_group(
            frequencies_list=group["f"],
            zre_list=group["zre"],
            zim_list=group["zim"],
            label=label,
            mu_threshold=getattr(config, "kk_mu_threshold", 0.85),
            residual_threshold_pct=getattr(config, "kk_residual_pct", 2.0),
        )
        results[label] = vr
    return results


def _build_weights(spectrum, config) -> np.ndarray:
    """
    Construit w(f) = 1/σ²(f) si σ(f) disponible,
    sinon retombe sur pondération Modulus uniforme (comportement actuel).
    """
    if spectrum.sigma_re is not None and spectrum.sigma_im is not None:
        sigma2 = np.asarray(spectrum.sigma_re)**2 + np.asarray(spectrum.sigma_im)**2
        return 1.0 / sigma2
    alpha = getattr(config, "alpha_noise", 0.001)
    Z_mod = np.sqrt(np.asarray(spectrum.Zre)**2 + np.asarray(spectrum.Zim)**2)
    return 1.0 / (alpha * Z_mod)**2


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

    for label, sp in [("bare", session.bare), ("probe", session.probe)]:
        if sp is None:
            continue
        for model in models:
            try:
                weights = _build_weights(sp, config)
                fr = model.fit(sp, config, weights=weights)
                sp.fit_results[model.name] = fr
                log.info(
                    f"Fit '{model.name}' [{label}]: "
                    f"Rct={fr.Rct:.1f} Ω chi2={fr.chi2:.3e} ok={fr.converged}"
                )
            except Exception as e:
                log.error(f"Fit '{model.name}' [{label}] failed: {e}")

    for conc in sorted(hybridization.keys()):
        sp_list = hybridization[conc]
        spectrum = average_replicates(sp_list) if len(sp_list) > 1 else sp_list[0]

        fit_results = {}
        for model in models:
            try:
                weights = _build_weights(spectrum, config)
                fr = model.fit(spectrum, config, weights=weights)
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
