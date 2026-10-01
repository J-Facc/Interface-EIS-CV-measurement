"""Orchestrateur EIS : chargement → measurement model + KK → fit Orazem + DRT.

Pour CHAQUE groupe de réplicats (bare, probe, chaque concentration), dans cet ordre :

1. chargement des fichiers → réplicats BRUTS (``EISSpectrum``), conservés tels quels
   à chaque étape (le spectre moyen les porte dans ``replicates``) ;
2. measurement model de Voigt sur les réplicats BRUTS (``core/measurement_model.py``) :
   structure d'erreur σ(ω) PUIS verdict Kramers-Kronig, calculés AVANT tout fit
   (l'ancien pipeline validait en aval des fits, AUDIT.md §5.5) — une seule analyse par
   groupe, qui fournit à la fois le verdict affiché et les poids du fit ;
3. fit Orazem du circuit utilisateur (``fits/orazem_fit.py``) sur CHAQUE réplicat (σ
   d'une mesure) ET sur la moyenne (σ/√n), puis agrégation par paramètre :
   incertitude intra-fit vs variabilité inter-réplicats (``OrazemGroupResult``) ;
4. DRT (``drt/engine.py``) sur CHAQUE réplicat ET sur la moyenne ; Rct DRT agrégé
   sur les réplicats (``GroupAnalysis.drt_target``).

Les ``FitResult`` sont rangés dans ``fit_results`` de chaque spectre (réplicat brut ou
moyenne) sous les clés ``circuit.model_name`` (« orazem ») et ``DRT_MODEL_NAME``
(« drt_bayes ») ; le statut et les agrégats du groupe dans ``GroupAnalysis``.

Sens des dépendances (AUDIT.md CPL-1)
-------------------------------------
::

    pages / ui / plotting / exports ──► core ──► drt ──► fits.result
                                         │                  ▲
                                         └────► fits ───────┘
                                                 └──► circuit

``core`` (orchestration, modèles de session, measurement model) dépend des bibliothèques
numériques ``fits``, ``drt`` et ``circuit`` ; AUCUN module de ``fits/``, ``drt/`` ou
``circuit/`` n'importe ``core``. ``FitResult`` (``fits/result.py``) et
``regression_stats`` ont été descendus dans ``fits/`` pour cela ; ``core.models``
ré-exporte ``FitResult``. Tous les imports sont au niveau module : plus d'import local
masquant un cycle. ``tests/test_architecture.py`` vérifie ce sens sur le code réel.

Erreurs (AUDIT.md ERR-1, ERR-3)
-------------------------------
Trois catégories, jamais confondues :

* **Saisie invalide** (circuit, guess/bornes, paramètre cible, réglages DRT) :
  ``InvalidAnalysisInput`` levée AVANT tout calcul.
* **Donnée utilisateur invalide** pour UN fichier, groupe ou spectre : résultat
  dégradé avec un message clair rangé DANS la session (pas seulement dans les logs) :

  - fichier illisible (``ValueError`` du loader) → ``session.load_errors`` ;
  - structure d'erreur non caractérisable (``ErrorStructureUnavailable``) → le
    groupe est ARRÊTÉ : ``GroupAnalysis.status = GROUP_ERROR_STRUCTURE_UNAVAILABLE``,
    ``message`` = phrase destinée à l'utilisateur, AUCUN fit (ni Orazem ni DRT) ;
  - circuit incompatible avec les données du groupe (``FitSpecificationError``, ex.
    plus de paramètres que d'observations) → ``GROUP_INVALID_INPUT``, groupe arrêté ;
  - DRT d'un spectre impossible — ``ValueError`` (spectre invalide) ou
    ``RuntimeError`` (échec de CmdStan), contrat de ``drt.engine.fit_drt`` →
    ``GroupAnalysis.drt_failures`` ; le reste du groupe est conservé ;
  - moteur DRT absent → ``session.messages``.
* **Bug logiciel** : toute autre exception REMONTE telle quelle. Ce module ne contient
  aucun ``except Exception``.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from typing import Optional

import numpy as np

import drt.engine as drt_engine
from circuit import CircuitError
from core.loader import average_replicates, load_spectrum
from core.logger import get_logger
from core.measurement_model import (
    ErrorStructureUnavailable,
    MeasurementModelOptions,
    analyze_replicates,
    common_frequency_grid,
)
from core.models import (
    GROUP_ERROR_STRUCTURE_UNAVAILABLE,
    GROUP_INVALID_INPUT,
    ConcentrationGroup,
    EISSession,
    EISSpectrum,
    FitResult,
    GroupAnalysis,
)
from core.validator import validation_from_analysis, validation_without_structure
from fits.orazem_fit import (
    AggregatedParameter,
    CircuitFit,
    FitOptions,
    FitSpecificationError,
    aggregate_parameter,
    compile_circuit_fit,
    fit_replicate_group,
)

log = get_logger("pipeline")

#: Clé de ``fit_results`` des DRT (``drt.engine.MODEL_NAME``).
DRT_MODEL_NAME = drt_engine.MODEL_NAME

_STEPS = ("bare", "probe", "hybridization")


class InvalidAnalysisInput(ValueError):
    """Saisie d'analyse invalide (circuit, guess/bornes, paramètre cible, réglages DRT).

    Levée AVANT tout calcul ; le message est destiné à l'utilisateur.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Saisie : circuit et DRT
# ─────────────────────────────────────────────────────────────────────────────

def build_circuit_fit(expression: str, parameters: dict, target_param: str) -> CircuitFit:
    """Compile le circuit utilisateur ; toute saisie invalide → ``InvalidAnalysisInput``.

    Args:
        expression: circuit (docs/CIRCUIT_UTILISATEUR.md).
        parameters: {nom: spécification} (``fits.orazem_fit`` : ParameterSpec, tuple
            (initial, lower, upper) ou dict initial/lower/upper/scale).
        target_param: paramètre servant de signal de calibration.
    """
    try:
        return compile_circuit_fit(expression, parameters, target_param)
    except (CircuitError, FitSpecificationError) as exc:
        raise InvalidAnalysisInput(f"Circuit ou paramètres invalides : {exc}") from exc


def circuit_fit_from_config(config: Optional[dict]) -> CircuitFit:
    """Circuit par défaut de la configuration (``fit.circuit``) → ``CircuitFit``.

    Dans ``fit.circuit.parameters``, une borne ``None`` signifie « non borné de ce côté ».
    """
    c = ((config or {}).get("fit") or {}).get("circuit") or {}
    if not c.get("expression"):
        raise InvalidAnalysisInput("Aucun circuit défini (fit.circuit.expression).")
    params = {}
    for name, p in (c.get("parameters") or {}).items():
        p = dict(p or {})
        spec = {
            "initial": p.get("initial"),
            "lower": -math.inf if p.get("lower") is None else p["lower"],
            "upper": math.inf if p.get("upper") is None else p["upper"],
        }
        if p.get("scale") is not None:
            spec["scale"] = p["scale"]
        params[name] = spec
    return build_circuit_fit(c["expression"], params, c.get("target_param"))


def _drt_request(config: Optional[dict], run_drt: Optional[bool],
                 mode: Optional[str]) -> Optional[dict]:
    """Arguments de ``drt.engine.fit_drt`` demandés, ou None si la DRT est désactivée.

    Raises:
        InvalidAnalysisInput: mode DRT inconnu.
    """
    dcfg = (((config or {}).get("fit") or {}).get("drt") or {})
    enabled = bool(dcfg.get("enabled", True)) if run_drt is None else bool(run_drt)
    if not enabled:
        return None
    kwargs = {"mode": mode or dcfg.get("mode") or drt_engine.DEFAULT_MODE}
    try:
        drt_engine.DRTSettings(**kwargs)
    except ValueError as exc:
        raise InvalidAnalysisInput(f"Réglages DRT invalides : {exc}") from exc
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Étapes d'un groupe
# ─────────────────────────────────────────────────────────────────────────────

def group_label(step: str, concentration: float) -> str:
    """Identifiant d'un groupe : 'bare', 'probe' ou 'hyb_<c>' (ex. 'hyb_1.00e-09')."""
    return step if step in ("bare", "probe") else f"hyb_{concentration:.2e}"


def _load_files(fas: list, step: str, conc: float, config: dict, session: EISSession) -> list:
    """Charge les fichiers d'un groupe. Fichier illisible → ``session.load_errors``."""
    loaded = []
    for fa in fas:
        try:
            sp = load_spectrum(content=fa["content"], label=fa["filename"],
                               concentration=conc, step=step, config=config)
        except ValueError as exc:          # contrat du loader : fichier illisible / invalide
            session.load_errors.append({"filename": fa["filename"], "message": str(exc)})
            log.warning(f"Fichier écarté '{fa['filename']}' : {exc}")
            continue
        loaded.append(sp)
        log.info(f"Chargé '{fa['filename']}' step={step} c={conc:.2e} M n={sp.n_points}")
    return loaded


def _display_mean(reps: list) -> EISSpectrum:
    """Spectre moyen d'AFFICHAGE quand le measurement model n'a pas abouti.

    Toujours un objet DISTINCT de ses réplicats (même pour un seul fichier), qui porte
    la liste des réplicats bruts.
    """
    if len(reps) > 1:
        return average_replicates(reps)
    r = reps[0]
    return EISSpectrum(
        label=f"{r.label} (avg)", f=np.array(r.f, dtype=float), Zre=np.array(r.Zre, dtype=float),
        Zim=np.array(r.Zim, dtype=float), concentration=r.concentration, step=r.step,
        n_points=len(r.f), source_files=[r.label], n_replicates=1, replicates=[r],
    )


def _mean_from_analysis(mm, reps: list, options: MeasurementModelOptions) -> EISSpectrum:
    """Spectre moyen = celui du measurement model (grille COMMUNE), en ordre HF→BF.

    C'est sur ce spectre que porte le fit Orazem de la moyenne et sa DRT : les deux
    voient exactement les mêmes données. ``sigma_re/sigma_im`` = écart-type BRUT
    inter-réplicats (ddof = 1) sur cette grille.
    """
    f_c, idx = common_frequency_grid(reps, options.freq_rtol, mm.label)
    zr = np.asarray([np.asarray(sp.Zre, dtype=float)[i] for sp, i in zip(reps, idx)])
    zj = np.asarray([np.asarray(sp.Zim, dtype=float)[i] for sp, i in zip(reps, idx)])
    ref = reps[0]
    return EISSpectrum(
        label=f"{ref.label} (avg)", f=f_c[::-1].copy(),
        Zre=np.asarray(mm.mean_Zre, dtype=float)[::-1].copy(),
        Zim=np.asarray(mm.mean_Zim, dtype=float)[::-1].copy(),
        concentration=ref.concentration, step=ref.step, n_points=len(f_c),
        source_files=[s.label for s in reps],
        sigma_re=zr.std(axis=0, ddof=1)[::-1].copy(), sigma_im=zj.std(axis=0, ddof=1)[::-1].copy(),
        n_replicates=len(reps), replicates=list(reps),
    )


def _reversed(fr: FitResult) -> FitResult:
    """Même FitResult, tableaux en ordre inverse (grille croissante → HF→BF)."""
    return dataclasses.replace(
        fr, Zfit_re=np.asarray(fr.Zfit_re)[::-1].copy(), Zfit_im=np.asarray(fr.Zfit_im)[::-1].copy(),
        residuals_re=np.asarray(fr.residuals_re)[::-1].copy(),
        residuals_im=np.asarray(fr.residuals_im)[::-1].copy(),
    )


def aggregate_drt_target(replicates: list) -> Optional[AggregatedParameter]:
    """Rct DRT agrégé sur les réplicats dont la DRT a CONVERGÉ.

    Avec une incertitude a posteriori par réplicat (mode 'sample') : même agrégation
    que le fit Orazem (``fits.orazem_fit.aggregate_parameter`` : intra-fit, inter-
    réplicats, Q de Cochran). En MAP ('optimize'), pas d'incertitude intra-fit : seule
    la variabilité inter-réplicats est rapportée (``std_within`` = NaN, « non
    calculée », jamais 0) et ``sem`` = s/√n.

    Returns:
        None si aucun réplicat ne porte de DRT.
    """
    fits = [sp.fit_results.get(DRT_MODEL_NAME) for sp in replicates]
    fits = [fr for fr in fits if fr is not None]
    if not fits:
        return None
    name = fits[0].target_param
    used = [fr for fr in fits if fr.converged and np.isfinite(fr.target_value)]
    excluded = len(replicates) - len(used)
    values = [float(fr.target_value) for fr in used]
    stds = [float(fr.target_std) for fr in used]
    if used and all(np.isfinite(s) and s > 0 for s in stds):
        return aggregate_parameter(name, values, stds, n_excluded=excluded)
    nan = float("nan")
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    sb = float(np.std(v, ddof=1)) if n >= 2 else nan
    return AggregatedParameter(
        name=name, n=n, mean=float(v.mean()) if n else nan, std_between=sb,
        std_within=nan, sem_within=nan, sem=sb / math.sqrt(n) if n >= 2 else nan,
        q=nan, q_pvalue=nan, values=v.tolist(), stds=stds, n_excluded=excluded,
    )


def _refresh_drt_target(analysis: GroupAnalysis, reps: list) -> None:
    analysis.drt_target = aggregate_drt_target(reps)
    modes = {sp.fit_results[DRT_MODEL_NAME].drt_mode for sp in reps
             if DRT_MODEL_NAME in sp.fit_results}
    msg = ("Rct DRT agrégé sur des réplicats calculés dans des modes DRT différents "
           f"({', '.join(sorted(m or '?' for m in modes))}) : relancez-les dans un même mode "
           "avant de comparer leurs incertitudes.")
    analysis.warnings = [w for w in analysis.warnings if not w.startswith("Rct DRT agrégé sur des")]
    if len(modes) > 1:
        analysis.warnings.append(msg)


def _run_drt(sp: EISSpectrum, drt_kwargs: dict, analysis: GroupAnalysis) -> None:
    """DRT d'un spectre ; spectre invalide ou échec de CmdStan → ``drt_failures``."""
    try:
        fr = drt_engine.fit_drt(sp, **drt_kwargs)
    except (ValueError, RuntimeError) as exc:     # contrat de drt.engine.fit_drt
        analysis.drt_failures[sp.label] = str(exc)
        msg = f"DRT de « {sp.label} » non calculée : {exc}"
        analysis.warnings.append(msg)
        log.warning(msg)
        return
    sp.fit_results[DRT_MODEL_NAME] = fr


def _analyze_group(label: str, reps: list, *, circuit: CircuitFit, fit_options: FitOptions,
                   mm_options: MeasurementModelOptions, drt_kwargs: Optional[dict]) -> tuple:
    """Measurement model + KK, PUIS fit Orazem et DRT, d'un groupe de réplicats bruts.

    Returns:
        (GroupAnalysis, spectre moyen).
    """
    analysis = GroupAnalysis(label=label)

    # ── 1. Measurement model sur les réplicats BRUTS : σ(ω) puis verdict KK ──────
    try:
        mm = analyze_replicates(reps, options=mm_options, label=label)
    except ErrorStructureUnavailable as exc:
        # ERR-1 : groupe ARRÊTÉ, statut explicite ; aucun fit, aucune pondération de repli.
        analysis.status = GROUP_ERROR_STRUCTURE_UNAVAILABLE
        analysis.message = exc.user_message
        analysis.validation = validation_without_structure(reps, label, exc.user_message)
        log.warning(exc.user_message)
        return analysis, _display_mean(reps)
    analysis.validation = validation_from_analysis(mm, label)
    mean_sp = _mean_from_analysis(mm, reps, mm_options)

    # ── 2. Fit Orazem : chaque réplicat (σ) + la moyenne (σ/√n), puis agrégation ──
    try:
        og = fit_replicate_group(circuit.Z_func, circuit.param_names, reps, mm, circuit.specs,
                                 circuit.target_param, options=fit_options,
                                 model_name=circuit.model_name)
    except FitSpecificationError as exc:
        analysis.status = GROUP_INVALID_INPUT
        analysis.message = (f"Groupe « {label} » : fit du circuit impossible sur ces données — "
                            f"{exc} Aucun fit n'a été réalisé pour ce groupe.")
        log.warning(analysis.message)
        return analysis, mean_sp
    analysis.orazem = og
    analysis.warnings.extend(og.warnings)
    for sp, fr in zip(reps, og.replicate_fits):
        sp.fit_results[circuit.model_name] = fr
    mean_sp.fit_results[circuit.model_name] = _reversed(og.mean_fit)
    t = og.target
    log.info(f"Orazem [{label}] {circuit.target_param} = {t.mean:.4g} ± {t.sem:.2g} "
             f"(n = {t.n}, inter = {t.std_between:.2g}, intra = {t.std_within:.2g})")

    # ── 3. DRT : chaque réplicat + la moyenne, puis Rct agrégé ───────────────────
    if drt_kwargs is not None:
        for sp in list(reps) + [mean_sp]:
            _run_drt(sp, drt_kwargs, analysis)
        _refresh_drt_target(analysis, reps)
    return analysis, mean_sp


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    file_assignments: list,
    config: Optional[dict],
    circuit: Optional[CircuitFit] = None,
    *,
    run_drt: Optional[bool] = None,
    drt_mode: Optional[str] = None,
) -> tuple:
    """Analyse EIS complète d'une électrode.

    Args:
        file_assignments: liste de dicts ``content`` (bytes), ``filename``, ``step``
            ('bare' | 'probe' | 'hybridization'), ``concentration`` (mol/L, lue pour
            'hybridization' seulement ; absente = 0.0).
        config: config de l'app (``config_to_dict(load_config())``).
        circuit: circuit à ajuster (``build_circuit_fit``) ; None → ``fit.circuit`` de
            la config.
        run_drt: lancer la DRT ; None → ``fit.drt.enabled`` de la config.
        drt_mode: 'optimize' | 'sample' ; None → ``fit.drt.mode`` de la config.

    Returns:
        ``(EISSession, {label de groupe: ValidationResult})`` — le verdict KK de chaque
        groupe, calculé avant ses fits, dans l'ordre d'arrivée des fichiers.

    Raises:
        InvalidAnalysisInput: circuit, guess/bornes, cible ou réglages DRT invalides
            (avant tout calcul). Toute autre exception est un bug et remonte.
    """
    config = config if config is not None else {}
    circuit_fit = circuit if circuit is not None else circuit_fit_from_config(config)
    if not isinstance(circuit_fit, CircuitFit):
        raise TypeError("circuit doit être un CircuitFit (voir build_circuit_fit).")
    drt_kwargs = _drt_request(config, run_drt, drt_mode)
    mm_options = MeasurementModelOptions.from_config(config)
    fit_options = FitOptions.from_config(config)

    session = EISSession(created_at=datetime.now(), config=config, circuit=circuit_fit.to_dict())
    if drt_kwargs is not None:
        ok, why = drt_engine.engine_available()
        if ok:
            session.drt_mode = drt_kwargs["mode"]
        else:
            session.messages.append(
                f"DRT non calculée : moteur indisponible — {why}. Installez l'extra DRT "
                "(requirements-drt.txt) et CmdStan : relancez launch.bat "
                "(ou python setup_drt_bayesien.py --ensure)."
            )
            drt_kwargs = None

    by_key: dict = {}
    for fa in file_assignments:
        step = fa.get("step")
        if step not in _STEPS:
            raise InvalidAnalysisInput(f"Étape inconnue {step!r} pour « {fa.get('filename')} » "
                                       f"(attendu : {', '.join(_STEPS)}).")
        conc = float(fa.get("concentration", 0.0)) if step == "hybridization" else 0.0
        by_key.setdefault((step, conc), []).append(fa)

    validation_results: dict = {}
    conc_groups = []
    for (step, conc), fas in by_key.items():
        reps = _load_files(fas, step, conc, config, session)
        if not reps:
            continue
        label = group_label(step, conc)
        analysis, mean_sp = _analyze_group(label, reps, circuit=circuit_fit, fit_options=fit_options,
                                           mm_options=mm_options, drt_kwargs=drt_kwargs)
        validation_results[label] = analysis.validation
        if step == "bare":
            session.bare, session.bare_replicate_spectra, session.bare_analysis = mean_sp, reps, analysis
        elif step == "probe":
            session.probe, session.probe_replicate_spectra, session.probe_analysis = mean_sp, reps, analysis
        else:
            conc_groups.append(ConcentrationGroup(concentration=conc, spectrum=mean_sp,
                                                  replicate_spectra=reps, analysis=analysis))
    session.groups = sorted(conc_groups, key=lambda g: g.concentration)
    return session, validation_results


# ─────────────────────────────────────────────────────────────────────────────
# Recalcul DRT ciblé — seul point d'entrée du recalcul (typiquement en 'sample')
# ─────────────────────────────────────────────────────────────────────────────

def _iter_session_spectra(session: EISSession):
    """Itère (label, spectrum) sur tous les spectres d'une session (moyennes + réplicats)."""
    if session.bare is not None:
        yield "bare", session.bare
    for i, sp in enumerate(session.bare_replicate_spectra or []):
        yield f"bare#{i}", sp
    if session.probe is not None:
        yield "probe", session.probe
    for i, sp in enumerate(session.probe_replicate_spectra or []):
        yield f"probe#{i}", sp
    for grp in session.groups or []:
        yield f"{grp.concentration:.2e}", grp.spectrum
        for i, sp in enumerate(grp.replicate_spectra or []):
            yield f"{grp.concentration:.2e}#{i}", sp


def _resolve_spectrum(session: EISSession, spectrum_id) -> Optional[EISSpectrum]:
    """Résout un spectre par identité (objet EISSpectrum) ou par label de session."""
    if isinstance(spectrum_id, EISSpectrum):
        return spectrum_id
    for label, sp in _iter_session_spectra(session):
        if sp is spectrum_id or label == spectrum_id:
            return sp
    return None


def recompute_drt(
    session: EISSession, spectrum_id, config: Optional[dict], mode: str = "sample"
) -> FitResult:
    """Relance UNIQUEMENT la DRT du spectre ciblé et remplace son FitResult.

    Seul point d'entrée du recalcul DRT à la demande (typiquement mode 'sample', HMC
    bayésien) : ``ui/`` appelle cette fonction et ne touche jamais à ``Inverter``. Le
    résultat remplace ``spectrum.fit_results['drt_bayes']`` ; si le spectre est un
    réplicat, le Rct DRT agrégé de son groupe est recalculé, et un échec DRT
    précédemment enregistré pour ce spectre est effacé.

    Args:
        session: session EIS courante.
        spectrum_id: l'objet ``EISSpectrum`` visé, ou son label de session ('bare',
            'probe', '1.00e-06', 'probe#0', '1.00e-06#2'…).
        config: config de l'app (seul ``fit.drt`` est lu ; le mode est forcé à ``mode``).
        mode: 'sample' (défaut) ou 'optimize'.

    Returns:
        Le nouveau ``FitResult`` DRT (également stocké dans ``spectrum.fit_results``).

    Raises:
        ValueError: spectre introuvable dans la session.
        InvalidAnalysisInput: mode inconnu.
        ValueError / RuntimeError de ``drt.engine.fit_drt`` (spectre invalide, moteur
            indisponible, échec de CmdStan) : propagées à l'appelant, qui les affiche.
    """
    spectrum = _resolve_spectrum(session, spectrum_id)
    if spectrum is None:
        raise ValueError(f"Spectre introuvable dans la session : {spectrum_id!r}.")
    drt_kwargs = _drt_request(config, True, mode)
    fr = drt_engine.fit_drt(spectrum, **drt_kwargs)
    spectrum.fit_results[DRT_MODEL_NAME] = fr

    owners = [(session.bare_analysis, session.bare, session.bare_replicate_spectra),
              (session.probe_analysis, session.probe, session.probe_replicate_spectra)]
    owners += [(g.analysis, g.spectrum, g.replicate_spectra) for g in session.groups]
    for analysis, mean_sp, reps in owners:
        is_rep = any(spectrum is r for r in reps or [])
        if analysis is None or not (is_rep or spectrum is mean_sp):
            continue
        if analysis.drt_failures.pop(spectrum.label, None) is not None:
            analysis.warnings = [w for w in analysis.warnings
                                 if not w.startswith(f"DRT de « {spectrum.label} » non calculée")]
        if is_rep:
            _refresh_drt_target(analysis, reps)
    log.info(f"recompute_drt [{spectrum.label}] mode={mode} {fr.target_param}={fr.target_value:.4g} "
             f"source={fr.params.get('rct_source')}")
    return fr
