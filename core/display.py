"""Mise en forme des mesures nettoyées pour l'affichage (v3) — aucun calcul, aucun fit.

Regroupe les ``Measurement`` d'une électrode par étape, dans l'ordre naturel (bare, probe,
conc_1…), pour les figures de la page Visualisation (``plotting.eis_plots`` /
``plotting.cv_plots``). Fonctionne de la même façon sur une session fraîche et sur une archive
rechargée (les mesures nettoyées sont dans l'archive).
"""

from __future__ import annotations

from core.experiment import (
    KIND_CV,
    KIND_EIS,
    STEP_BARE,
    STEP_PROBE,
    Experiment,
    format_concentration,
    step_sort_key,
)
from core.models import DisplayGroup


def step_display_label(step: str, concentration) -> str:
    if step == STEP_BARE:
        return "Bare"
    if step == STEP_PROBE:
        return "Probe"
    return f"{step} — {format_concentration(concentration)}"


def selection_options(measurements: list, experiment: Experiment, kind: str = None) -> dict:
    """Options disponibles pour les sélecteurs : {biomarqueur: {canal: {électrode: {étape: [réplicats]}}}}."""
    out: dict = {}
    for m in measurements or []:
        if kind and m.kind != kind:
            continue
        ch = experiment.channel(m.channel_id)
        if ch is None:
            continue
        (out.setdefault(ch.biomarker, {}).setdefault(m.channel_id, {}).setdefault(m.electrode, {})
         .setdefault(m.step, set()).add(m.replicate))
    for bio in out.values():
        for chd in bio.values():
            for eld in chd.values():
                for step in list(eld):
                    eld[step] = sorted(eld[step])
    return out


def _select(measurements, kind, channel_id, electrode, steps, replicates):
    sel = [m for m in measurements or [] if m.kind == kind and m.channel_id == channel_id
           and m.electrode == electrode and (steps is None or m.step in steps)
           and (replicates is None or m.replicate in replicates)]
    by_step: dict = {}
    for m in sel:
        by_step.setdefault(m.step, []).append(m)
    return [(s, sorted(by_step[s], key=lambda m: m.replicate)) for s in sorted(by_step, key=step_sort_key)]


def eis_display_groups(measurements: list, experiment: Experiment, channel_id: int, electrode: int,
                       steps=None, replicates=None) -> list:
    """``DisplayGroup`` (sans moyenne) par étape, pour ``nyquist_replicates_figure``/``bode_figure``."""
    ch = experiment.channel(channel_id)
    out = []
    for step, ms in _select(measurements, KIND_EIS, channel_id, electrode, steps, replicates):
        conc = ch.concentration_of(step) if ch else None
        reps = [m.to_spectrum(experiment) for m in ms]
        for sp, m in zip(reps, ms):
            sp.label = f"r{m.replicate}"
        out.append(DisplayGroup(label=step_display_label(step, conc), concentration=float(conc or 0.0),
                                step=step if step in (STEP_BARE, STEP_PROBE) else "hybridization",
                                mean=None, replicates=reps))
    return out


def cv_display_groups(measurements: list, experiment: Experiment, channel_id: int, electrode: int,
                      steps=None, replicates=None) -> list:
    """Groupes pour ``plotting.cv_plots.cv_figure`` : une entrée par étape, une courbe par réplicat."""
    ch = experiment.channel(channel_id)
    out = []
    for step, ms in _select(measurements, KIND_CV, channel_id, electrode, steps, replicates):
        conc = ch.concentration_of(step) if ch else None
        label = step_display_label(step, conc)
        out.append({"label": label, "step": step, "concentration": conc,
                    "curves": [(f"{label} r{m.replicate}", m.data["E"], m.data["I"]) for m in ms]})
    return out
