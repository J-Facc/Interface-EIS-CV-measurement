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
    # NB : les seuils KK (mu, résidu %) sont laissés aux valeurs par défaut de
    # core/validator.py (MU_THRESHOLD, RESIDUAL_THRESHOLD_PCT), source unique.
    # L'ancien code lisait getattr(config, "kk_mu_threshold"/"kk_residual_pct")
    # sur un dict → renvoyait toujours le défaut, et ces clés n'existaient ni
    # dans le YAML ni dans Pydantic. Supprimé pour éviter une config fantôme.
    results = {}
    for label, group in replicate_groups.items():
        vr = validate_replicate_group(
            frequencies_list=group["f"],
            zre_list=group["zre"],
            zim_list=group["zim"],
            label=label,
        )
        results[label] = vr
    return results


def _characterize_error_structure_upfront(session, hybridization, config) -> None:
    """Caractérise (et persiste) la structure d'erreur d'Orazem une fois par run.

    Parcourt les spectres moyennés porteurs de réplicats (bare, probe, groupes de
    concentration) et déclenche resolve_error_structure sur le premier éligible :
    la caractérisation est alors persistée et servira à TOUS les fits du run (y
    compris les réplicats individuels, qui n'ont pas de σ propre). Si aucun
    spectre n'a assez de réplicats, on ne fait rien : les fits réutiliseront une
    caractérisation persistée antérieure, ou seront refusés (ErrorStructureUnavailable).
    """
    from fits.error_structure import resolve_error_structure, ErrorStructureUnavailable

    candidates = [session.bare, session.probe]
    for conc in sorted(hybridization.keys()):
        sp_list = hybridization[conc]
        candidates.append(sp_list[0] if sp_list else None)

    for sp in candidates:
        if sp is None or getattr(sp, "sigma_re", None) is None:
            continue
        try:
            es = resolve_error_structure(sp, config)
        except ErrorStructureUnavailable:
            continue
        if es.source == "characterized_now":
            log.info(
                f"Structure d'erreur caractérisée sur '{sp.label}' "
                f"({es.n_replicates} réplicats) et persistée."
            )
            return
    log.info(
        "Aucune caractérisation de structure d'erreur possible sur ce jeu "
        "(pas assez de réplicats) — repli sur coefficients persistés si disponibles."
    )


def _run_kk(spectrum, config, label: str) -> Optional[dict]:
    """Calcule la validation Kramers-Kronig (fits/kk_validation.py) pour un spectre."""
    from fits.kk_validation import kramers_kronig_check
    try:
        return kramers_kronig_check(spectrum, config)
    except Exception as e:
        log.error(f"kramers_kronig_check [{label}] failed: {e}")
        return None


def run_pipeline(
    file_assignments: list,
    config: dict,
    active_models: Optional[list] = None,
) -> tuple:
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
    replicate_groups: dict = {}
    # NOTE (ajout hors périmètre initial — réorganisation onglets EIS) :
    # conserve les réplicats individuels (avant moyenne) par (step, conc),
    # afin de pouvoir tracer DRT / reconstructions par réplicat (onglet DRT
    # et onglet Reconstructions Nyquist). Fits appliqués ci-dessous.
    replicate_spectra_by_key: dict = {}

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

        # Collect raw replicates for KK validation (before averaging)
        if step in ("bare", "probe"):
            grp_label = step
        else:
            grp_label = f"hyb_{conc:.2e}"
        replicate_groups[grp_label] = {
            "f":   [np.asarray(sp.f,   dtype=float) for sp in loaded],
            "zre": [np.asarray(sp.Zre, dtype=float) for sp in loaded],
            "zim": [np.asarray(sp.Zim, dtype=float) for sp in loaded],
        }
        replicate_spectra_by_key[(step, conc)] = loaded

        averaged = average_replicates(loaded) if len(loaded) > 1 else loaded[0]

        if step == "bare":
            bare_spectra.append(averaged)
        elif step == "probe":
            probe_spectra.append(averaged)
        else:
            hybridization.setdefault(conc, []).append(averaged)

    session.bare = average_replicates(bare_spectra) if bare_spectra else None
    session.probe = average_replicates(probe_spectra) if probe_spectra else None

    # ── Caractérisation de la structure d'erreur (Orazem) AVANT tout fit ──────
    # On caractérise sur le PREMIER spectre porteur d'assez de réplicats et on
    # persiste, afin que TOUS les fits (moyennes ET réplicats individuels, qui
    # n'ont pas de σ propre) partagent une même structure d'erreur — caractérisée
    # sur ce jeu, ou réutilisée depuis une caractérisation antérieure persistée.
    _characterize_error_structure_upfront(session, hybridization, config)

    def _fit_replicates(reps: list) -> list:
        """Applique tous les modèles actifs à chaque réplicat individuel."""
        for sp in reps:
            for model in models:
                try:
                    # Pas de poids explicites : le modèle résout lui-même la
                    # structure d'erreur d'Orazem (caractérisée ou réutilisée).
                    fr = model.fit(sp, config)
                    sp.fit_results[model.name] = fr
                except Exception as e:
                    log.error(f"Fit réplicat '{model.name}' [{sp.label}] failed: {e}")
        return reps

    if ("bare", 0.0) in replicate_spectra_by_key:
        session.bare_replicate_spectra = _fit_replicates(replicate_spectra_by_key[("bare", 0.0)])
    if ("probe", 0.0) in replicate_spectra_by_key:
        session.probe_replicate_spectra = _fit_replicates(replicate_spectra_by_key[("probe", 0.0)])

    for label, sp in [("bare", session.bare), ("probe", session.probe)]:
        if sp is None:
            continue
        kk = _run_kk(sp, config, label)
        for model in models:
            try:
                fr = model.fit(sp, config)
                if kk is not None:
                    fr.kk_passed = kk["kk_passed"]
                    fr.kk_residuals = kk
                sp.fit_results[model.name] = fr
                log.info(
                    f"Fit '{model.name}' [{label}]: "
                    f"Rct={fr.Rct:.1f} Ω chi2_red={fr.chi2_reduced:.3e} ok={fr.converged}"
                )
            except Exception as e:
                log.error(f"Fit '{model.name}' [{label}] failed: {e}")

    for conc in sorted(hybridization.keys()):
        sp_list = hybridization[conc]
        spectrum = average_replicates(sp_list) if len(sp_list) > 1 else sp_list[0]

        kk = _run_kk(spectrum, config, f"hyb_{conc:.2e}")
        fit_results = {}
        for model in models:
            try:
                fr = model.fit(spectrum, config)
                if kk is not None:
                    fr.kk_passed = kk["kk_passed"]
                    fr.kk_residuals = kk
                fit_results[model.name] = fr
                log.info(
                    f"Fit '{model.name}' [{conc:.2e} M]: "
                    f"Rct={fr.Rct:.1f} Ω chi2_red={fr.chi2_reduced:.3e} ok={fr.converged}"
                )
            except Exception as e:
                log.error(f"Fit '{model.name}' [{conc:.2e} M] failed: {e}")

        rep_spectra = _fit_replicates(replicate_spectra_by_key.get(("hybridization", conc), []))

        session.groups.append(
            ConcentrationGroup(
                concentration=conc,
                spectrum=spectrum,
                fit_results=fit_results,
                replicate_spectra=rep_spectra,
            )
        )

    # KK validation on raw replicates
    try:
        validation_results = validate_session(replicate_groups, config)
    except Exception as e:
        log.error(f"validate_session failed: {e}")
        validation_results = {}

    return session, validation_results
