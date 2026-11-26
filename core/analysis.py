"""Analyse complète en arrière-plan (v3) — aucun import Streamlit.

Pour chaque groupe EIS (canal, électrode, étape) :

1. **fit du circuit** (tâche ``eis_group_task``, dans un processus de calcul) :
   * ≥ ``fit.error_structure.min_replicates`` réplicats (3) : measurement model + verdict
     Kramers-Kronig PUIS fit Orazem pondéré par la structure d'erreur (``core.pipeline``) ;
   * moins de réplicats, ou structure d'erreur non caractérisable : le fit est QUAND MÊME
     calculé, en **repli** pondéré proportionnellement (σ = 1 %·|Z|) ; le verdict KK est
     « indéterminé » et le repli est signalé ;
   * un verdict KK « non conforme » n'exclut rien : il est signalé ;
2. **DRT** (tâche ``drt_task``) de chaque réplicat et du spectre moyen, puis extraction du
   pic principal (``core.drt_peak``) → Rct_DRT ;
3. **CV** : courant de pic anodique et ΔI/I (``core.cv``, rapide, dans le processus principal).

Tout est converti en enregistrements sérialisables (``core.results``) DANS le worker : rien
de ce qui revient au processus principal ne dépend des objets internes du pipeline.
``AnalysisRun`` orchestre le tout (``core.background.BackgroundRun``) ; les résultats ne sont
publiés qu'une fois TOUTES les tâches terminées. Un échec isolé est listé (``Failure``) et
n'arrête pas les autres.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import drt.engine as drt_engine
from core.background import BackgroundRun
from core.cv import cv_features
from core.drt_peak import main_peak
from core.experiment import KIND_EIS, Experiment, step_sort_key
from core.logger import get_logger
from core.measurement_model import MeasurementModelOptions
from core.models import GROUP_ERROR_STRUCTURE_UNAVAILABLE, GROUP_INVALID_INPUT, EISSpectrum
from core.pipeline import (
    InvalidAnalysisInput,
    _analyze_group,
    _display_mean,
    _drt_request,
    _fit_drt_timed,
    _log_drt_detail,
    build_circuit_fit,
    circuit_fit_from_config,
)
from core.results import (
    GROUP_FAILED,
    GROUP_FALLBACK,
    GROUP_OK,
    KK_CONFORM,
    KK_NONCONFORM,
    KK_UNDETERMINED,
    PROPORTIONAL_SIGMA,
    WEIGHTING_ORAZEM,
    WEIGHTING_PROPORTIONAL,
    AnalysisResults,
    DRTRecord,
    EISResults,
    Failure,
    FitRecord,
    GroupRecord,
    SpectrumRecord,
    group_key,
    mean_key,
)
from fits.orazem_fit import FitOptions, FitSpecificationError, aggregate_parameter, fit_spectrum

log = get_logger("analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit : description sérialisable ↔ CircuitFit
# ─────────────────────────────────────────────────────────────────────────────

def circuit_from_dict(circuit: Optional[dict], config: Optional[dict]):
    """``CircuitFit`` depuis ``CircuitFit.to_dict()`` (ou la config si ``circuit`` est None).

    Raises:
        InvalidAnalysisInput: circuit ou paramètres invalides.
    """
    if not circuit:
        return circuit_fit_from_config(config)
    params = {}
    for name, p in (circuit.get("parameters") or {}).items():
        spec = {"initial": p.get("initial"),
                "lower": -math.inf if p.get("lower") is None else p["lower"],
                "upper": math.inf if p.get("upper") is None else p["upper"]}
        if p.get("scale") is not None:
            spec["scale"] = p["scale"]
        params[name] = spec
    return build_circuit_fit(circuit["expression"], params, circuit.get("target_param"))


# ─────────────────────────────────────────────────────────────────────────────
# Tâche 1 : fit d'un groupe (exécutée dans un processus de calcul)
# ─────────────────────────────────────────────────────────────────────────────

def _spectrum(item: dict) -> EISSpectrum:
    f = np.asarray(item["f"], dtype=float)
    return EISSpectrum(label=item["key"], f=f, Zre=np.asarray(item["Zre"], dtype=float),
                       Zim=np.asarray(item["Zim"], dtype=float),
                       concentration=float(item.get("concentration") or 0.0),
                       step=item["pipeline_step"], n_points=len(f), source_files=[item["key"]])


def _record(sp: EISSpectrum, key: str, payload: dict, replicate: Optional[int]) -> SpectrumRecord:
    return SpectrumRecord(key=key, group_key=payload["gkey"], channel_id=payload["channel_id"],
                          electrode=payload["electrode"], step=payload["step"], replicate=replicate,
                          concentration=payload.get("concentration"),
                          f=np.asarray(sp.f, dtype=float).copy(), Zre=np.asarray(sp.Zre, dtype=float).copy(),
                          Zim=np.asarray(sp.Zim, dtype=float).copy())


def _aggregate_dict(agg: dict) -> dict:
    return {p: {"mean": float(a.mean), "std_between": float(a.std_between), "sem": float(a.sem), "n": int(a.n)}
            for p, a in agg.items()}


def _fallback(group: GroupRecord, payload: dict, reps: list, circuit, fit_options, reason: str):
    """Fit de chaque réplicat et de la moyenne, pondération proportionnelle σ = 1 %·|Z|."""
    group.status = GROUP_FALLBACK
    group.weighting = WEIGHTING_PROPORTIONAL
    group.kk_state = KK_UNDETERMINED
    group.kk_message = "Verdict Kramers-Kronig indéterminé : aucun niveau de bruit caractérisé."
    group.message = (f"{reason} Fit calculé en REPLI, pondération proportionnelle (σ = "
                     f"{PROPORTIONAL_SIGMA:.0%}·|Z|) : χ²ᵣ n'est pas un test d'adéquation et les "
                     "incertitudes intra-fit sont indicatives.")
    mean_sp = _display_mean(reps)
    records = []
    for sp, rep in [(r, payload["spectra"][i]["replicate"]) for i, r in enumerate(reps)] + [(mean_sp, None)]:
        key = sp.label if rep is not None else mean_key(payload["channel_id"], payload["electrode"], payload["step"])
        rec = _record(sp, key, payload, rep)
        rec.kk_message = group.kk_message
        zr, zj = np.asarray(sp.Zre, float), np.asarray(sp.Zim, float)
        sigma = np.maximum(PROPORTIONAL_SIGMA * np.hypot(zr, zj), 1e-12)
        try:
            fr = fit_spectrum(circuit.Z_func, circuit.param_names, sp.f, zr, zj, sigma, sigma, circuit.specs,
                              circuit.target_param, options=fit_options, error_structure=None,
                              model_name=circuit.model_name, label=key)
            fr.warnings.insert(0, "pondération proportionnelle (repli)")
            rec.fit = FitRecord.from_fit_result(fr, WEIGHTING_PROPORTIONAL)
        except (FitSpecificationError, ValueError) as exc:
            rec.fit_error = str(exc)
        records.append(rec)
    used = [r.fit for r in records[:-1] if r.fit is not None and r.fit.converged]
    if used:
        group.aggregate = _aggregate_dict({p: aggregate_parameter(p, [f.params[p] for f in used],
                                                                  [f.params_std[p] for f in used],
                                                                  n_excluded=len(reps) - len(used))
                                           for p in circuit.param_names})
    group.mean_key = records[-1].key
    return records


def eis_group_task(payload: dict) -> dict:
    """Fit Orazem (ou repli) d'un groupe de réplicats. Exécutée dans un worker.

    Args:
        payload: ``gkey, channel_id, electrode, step, concentration, biomarker, circuit (dict),
            config, spectra: [{key, replicate, f, Zre, Zim, pipeline_step, concentration}]``.

    Returns:
        ``{"group": GroupRecord, "spectra": [SpectrumRecord]}`` (réplicats puis moyenne).
    """
    config = payload["config"]
    circuit = circuit_from_dict(payload.get("circuit"), config)
    fit_options = FitOptions.from_config(config)
    mm_options = MeasurementModelOptions.from_config(config)
    reps = [_spectrum(it) for it in payload["spectra"]]
    group = GroupRecord(key=payload["gkey"], channel_id=payload["channel_id"], electrode=payload["electrode"],
                        step=payload["step"], concentration=payload.get("concentration"),
                        biomarker=payload.get("biomarker", ""), n_replicates=len(reps),
                        replicate_keys=[it["key"] for it in payload["spectra"]])
    min_rep = int(((config.get("fit") or {}).get("error_structure") or {}).get("min_replicates", 3))
    if len(reps) < min_rep:
        return {"group": group, "spectra": _fallback(
            group, payload, reps, circuit, fit_options,
            f"{len(reps)} réplicat(s) (< {min_rep}) : structure d'erreur non caractérisable.")}

    analysis, mean_sp = _analyze_group(group.key, reps, circuit=circuit, fit_options=fit_options,
                                       mm_options=mm_options, drt_kwargs=None)
    if analysis.status == GROUP_ERROR_STRUCTURE_UNAVAILABLE:
        for sp in reps:
            sp.fit_results.clear()
        return {"group": group, "spectra": _fallback(group, payload, reps, circuit, fit_options,
                                                     analysis.message or "Structure d'erreur indisponible.")}
    vr = analysis.validation
    mm = getattr(vr, "measurement_model", None)
    group.weighting = WEIGHTING_ORAZEM
    valid = getattr(vr, "all_valid", None)
    group.kk_state = KK_UNDETERMINED if valid is None else (KK_CONFORM if valid else KK_NONCONFORM)
    group.kk_message = str(getattr(vr, "kk_message", None) or getattr(mm, "kk_message", "") or "")
    group.warnings = [str(w) for w in analysis.warnings]
    records = []
    for i, sp in enumerate(reps):
        rec = _record(sp, sp.label, payload, payload["spectra"][i]["replicate"])
        if mm is not None and i < len(mm.kk_replicates):
            rec.kk_conform = mm.kk_replicates[i].conform
            rec.kk_message = mm.kk_replicates[i].verdict.message
        records.append(rec)
    mrec = _record(mean_sp, mean_key(group.channel_id, group.electrode, group.step), payload, None)
    if mm is not None:
        mrec.kk_conform, mrec.kk_message = mm.kk_mean.conform, mm.kk_mean.verdict.message
    records.append(mrec)
    group.mean_key = mrec.key
    if analysis.status == GROUP_INVALID_INPUT:
        group.status = GROUP_FAILED
        group.message = analysis.message or "Fit impossible sur ce groupe."
        for r in records:
            r.fit_error = group.message
        return {"group": group, "spectra": records}
    group.status = GROUP_OK
    og = analysis.orazem
    for rec, sp in zip(records[:-1], reps):
        fr = sp.fit_results.get(circuit.model_name)
        if fr is not None:
            rec.fit = FitRecord.from_fit_result(fr, WEIGHTING_ORAZEM)
    fr_mean = mean_sp.fit_results.get(circuit.model_name)
    if fr_mean is not None:
        mrec.fit = FitRecord.from_fit_result(fr_mean, WEIGHTING_ORAZEM)
    group.aggregate = _aggregate_dict(og.aggregate) if og is not None else {}
    return {"group": group, "spectra": records}


# ─────────────────────────────────────────────────────────────────────────────
# Tâche 2 : DRT d'un spectre (exécutée dans un processus de calcul)
# ─────────────────────────────────────────────────────────────────────────────

def drt_task(payload: dict) -> dict:
    """DRT + pic principal d'un spectre. ``{"key", "record": DRTRecord}`` ou ``{"key", "error"}``.

    Un spectre invalide ou un échec de CmdStan (contrat de ``drt.engine.fit_drt`` :
    ``ValueError``/``RuntimeError``) est rendu comme ``error`` ; tout autre problème remonte et
    est rapporté par ``BackgroundRun``.
    """
    sp = EISSpectrum(label=payload["key"], f=np.asarray(payload["f"], float),
                     Zre=np.asarray(payload["Zre"], float), Zim=np.asarray(payload["Zim"], float),
                     concentration=0.0, step="probe", n_points=len(payload["f"]))
    try:
        fr = _fit_drt_timed(sp, payload["drt_kwargs"])
    except (ValueError, RuntimeError) as exc:
        _log_drt_detail(payload["key"], exc)
        return {"key": payload["key"], "error": str(exc)}
    peak = main_peak(fr.drt_tau, fr.drt_gamma, payload["tau_window"])
    return {"key": payload["key"], "record": DRTRecord.from_fit_result(fr, peak)}


# ─────────────────────────────────────────────────────────────────────────────
# Construction des tâches
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_step(step: str) -> str:
    return step if step in ("bare", "probe") else "hybridization"


def build_group_payloads(experiment: Experiment, measurements: list, config: dict, circuit: dict) -> list:
    """Une charge utile ``eis_group_task`` par (canal, électrode, étape) EIS."""
    groups: dict = {}
    for m in measurements or []:
        if m.kind == KIND_EIS:
            groups.setdefault((m.channel_id, m.electrode, m.step), []).append(m)
    out = []
    for (ch_id, el, step) in sorted(groups, key=lambda k: (k[0], k[1], step_sort_key(k[2]))):
        ms = sorted(groups[(ch_id, el, step)], key=lambda m: m.replicate)
        ch = experiment.channel(ch_id)
        conc = ch.concentration_of(step) if ch else None
        out.append({
            "gkey": group_key(ch_id, el, step), "channel_id": ch_id, "electrode": el, "step": step,
            "concentration": conc, "biomarker": ch.biomarker if ch else "", "circuit": circuit, "config": config,
            "spectra": [{"key": m.id, "replicate": m.replicate, "f": m.data["f"], "Zre": m.data["Zre"],
                         "Zim": m.data["Zim"], "pipeline_step": _pipeline_step(step),
                         "concentration": conc} for m in ms],
        })
    return out


def drt_settings(config: dict, mode: Optional[str] = None, enabled: Optional[bool] = None) -> tuple:
    """(kwargs du moteur ou None si DRT désactivée, fenêtre τ du pic principal)."""
    kwargs = _drt_request(config, enabled, mode)
    tau_window = list(((config.get("fit") or {}).get("drt") or {}).get("tau_window") or [1e-5, 1e-1])
    return kwargs, tau_window


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisRun:
    """Analyse complète en arrière-plan ; ``results`` n'est publié qu'à la fin de TOUT.

    Args:
        experiment, measurements: données nettoyées.
        config: config de l'app (``config_to_dict``).
        circuit: ``CircuitFit.to_dict()`` saisi par l'utilisateur (None = défaut de la config).
        run_drt / drt_mode: None = ``fit.drt.enabled`` / ``fit.drt.mode`` de la config.
        executor_kind, max_workers: voir ``core.background``.

    Raises:
        InvalidAnalysisInput: circuit ou réglages DRT invalides (avant tout calcul).
    """

    def __init__(self, experiment: Experiment, measurements: list, config: dict, circuit: Optional[dict] = None,
                 run_drt: Optional[bool] = None, drt_mode: Optional[str] = None,
                 executor_kind: str = "process", max_workers: Optional[int] = None):
        circuit_obj = circuit_from_dict(circuit, config)        # valide AVANT tout calcul
        self.circuit = circuit_obj.to_dict()
        self.config = config
        self.experiment = experiment
        self.measurements = measurements
        self.drt_kwargs, self.tau_window = drt_settings(config, drt_mode, run_drt)
        self._results = AnalysisResults(eis=EISResults(circuit=self.circuit), config=config)
        if self.drt_kwargs is not None:
            ok, why = drt_engine.engine_available()
            if ok:
                self._results.eis.drt_mode = self.drt_kwargs["mode"]
            else:
                self._results.eis.drt_messages.append(
                    f"DRT non calculée : moteur indisponible — {why}. Relancez launch.bat "
                    "(ou python setup_drt_bayesien.py --ensure).")
                self.drt_kwargs = None
        self._drt: dict = {}
        mw = max_workers or ((config.get("analysis") or {}).get("max_workers"))
        if self.drt_kwargs and self.drt_kwargs.get("mode") == "sample" and not mw:
            # HMC : chaque calcul lance ses propres chaînes → pas plus de workers que de cœurs/chaînes.
            import os
            mw = max(1, (os.cpu_count() or 2) // int(self.drt_kwargs.get("chains", 4)))
        self.bg = BackgroundRun(executor_kind, mw)
        payloads = build_group_payloads(experiment, measurements, config, self.circuit)
        n_spectra = sum(len(p["spectra"]) for p in payloads)
        self.n_groups, self.n_drt = len(payloads), (n_spectra + len(payloads) if self.drt_kwargs else 0)
        self.bg.total = self.n_groups + self.n_drt
        for p in payloads:
            self.bg.submit(p["gkey"], eis_group_task, p,
                           on_result=self._on_group, on_error=self._group_error(p))
            if self.drt_kwargs:
                for it in p["spectra"]:
                    self._submit_drt(it["key"], it["f"], it["Zre"], it["Zim"])

    # ── Rappels ─────────────────────────────────────────────────────────────
    def _submit_drt(self, key, f, zre, zim) -> None:
        self.bg.submit(f"DRT {key}", drt_task,
                       {"key": key, "f": f, "Zre": zre, "Zim": zim, "drt_kwargs": self.drt_kwargs,
                        "tau_window": self.tau_window},
                       on_result=self._on_drt, on_error=lambda msg, k=key: self._drt.__setitem__(k, msg))

    def _on_group(self, out: dict) -> None:
        g: GroupRecord = out["group"]
        self._results.eis.groups[g.key] = g
        for rec in out["spectra"]:
            self._results.eis.spectra[rec.key] = rec
            if rec.fit_error:
                self._results.failures.append(Failure(rec.key, "fit", rec.fit_error))
        if self.drt_kwargs:
            mean = self._results.eis.spectra.get(g.mean_key) if g.mean_key else None
            if mean is not None:
                self._submit_drt(mean.key, mean.f, mean.Zre, mean.Zim)
            else:
                self.bg.total -= 1

    def _group_error(self, payload: dict):
        def _err(msg: str) -> None:
            self._results.failures.append(Failure(payload["gkey"], "fit", msg))
            if self.drt_kwargs:
                self.bg.total -= 1          # la DRT de la moyenne ne sera pas soumise
        return _err

    def _on_drt(self, out: dict) -> None:
        self._drt[out["key"]] = out.get("record") if "record" in out else out["error"]

    # ── Fin de l'analyse ────────────────────────────────────────────────────
    def _finalize(self) -> None:
        eis = self._results.eis
        for key, val in self._drt.items():
            rec = eis.spectra.get(key)
            if isinstance(val, DRTRecord):
                if rec is not None:
                    rec.drt = val
                    if not val.peak.ok:
                        self._results.failures.append(Failure(key, "drt", val.peak.message))
            else:
                if rec is not None:
                    rec.drt_error = str(val)
                self._results.failures.append(Failure(key, "drt", str(val)))
        cv_window = ((self.config.get("cv") or {}).get("peak_window_V")) or [0.0, 0.25]
        self._results.cv = cv_features(self.measurements, self.experiment, cv_window)
        for f in self._results.cv.values():
            if not f.ok:
                self._results.failures.append(Failure(f.key, "cv", f.message))
        self._finalized = True

    def poll(self) -> bool:
        finished = self.bg.poll()
        if finished and not self.bg.cancelled and not getattr(self, "_finalized", False):
            self._finalize()
        return finished

    def wait(self, timeout: Optional[float] = None) -> bool:
        ok = self.bg.wait(timeout)
        self.poll()
        return ok

    def cancel(self) -> None:
        self.bg.cancel()

    @property
    def progress(self) -> tuple:
        return self.bg.done, max(self.bg.total, self.bg.done)

    @property
    def finished(self) -> bool:
        return self.bg.finished and getattr(self, "_finalized", False)

    @property
    def cancelled(self) -> bool:
        return self.bg.cancelled

    @property
    def results(self) -> Optional[AnalysisResults]:
        """Résultats complets, ou None tant qu'une tâche est en cours (rien d'affiché avant la fin)."""
        return self._results if self.finished else None


class DRTRecomputeRun:
    """Recalcul de la DRT de spectres choisis (ex. en 'sample'), en arrière-plan.

    À la fin, chaque ``DRTRecord`` recalculé REMPLACE celui du spectre dans ``results`` ; un
    échec est signalé sans toucher au résultat existant.
    """

    def __init__(self, results: AnalysisResults, keys: list, config: dict, mode: str,
                 executor_kind: str = "process", max_workers: Optional[int] = None):
        self.results = results
        self.mode = mode
        self.drt_kwargs, self.tau_window = drt_settings(config, mode, True)
        self.outcomes: dict = {}
        if mode == "sample" and not max_workers:
            import os
            max_workers = max(1, (os.cpu_count() or 2) // int(self.drt_kwargs.get("chains", 4)))
        self.bg = BackgroundRun(executor_kind, max_workers)
        self.bg.total = len(keys)
        for k in keys:
            sp = results.eis.spectra[k]
            self.bg.submit(f"DRT {k}", drt_task, {"key": k, "f": sp.f, "Zre": sp.Zre, "Zim": sp.Zim,
                                                  "drt_kwargs": self.drt_kwargs, "tau_window": self.tau_window},
                           on_result=self._on_result, on_error=lambda msg, k=k: self.outcomes.__setitem__(k, msg))
        self._merged = False

    def _on_result(self, out: dict) -> None:
        self.outcomes[out["key"]] = out.get("record") if "record" in out else out["error"]

    def poll(self) -> bool:
        done = self.bg.poll()
        if done and not self._merged and not self.bg.cancelled:
            for k, val in self.outcomes.items():
                if isinstance(val, DRTRecord):
                    self.results.eis.spectra[k].drt = val
                    self.results.eis.spectra[k].drt_error = None
                    self.results.failures = [f for f in self.results.failures
                                             if not (f.key == k and f.stage == "drt")]
                    if not val.peak.ok:
                        self.results.failures.append(Failure(k, "drt", val.peak.message))
            self._merged = True
        return done

    def wait(self, timeout: Optional[float] = None) -> bool:
        ok = self.bg.wait(timeout)
        self.poll()
        return ok

    def cancel(self) -> None:
        self.bg.cancel()

    @property
    def progress(self) -> tuple:
        return self.bg.done, max(self.bg.total, self.bg.done)

    @property
    def finished(self) -> bool:
        return self.bg.finished and (self._merged or self.bg.cancelled)

    def errors(self) -> dict:
        return {k: v for k, v in self.outcomes.items() if not isinstance(v, DRTRecord)}


__all__ = ["AnalysisRun", "DRTRecomputeRun", "InvalidAnalysisInput", "eis_group_task", "drt_task",
           "build_group_payloads", "circuit_from_dict", "drt_settings"]
