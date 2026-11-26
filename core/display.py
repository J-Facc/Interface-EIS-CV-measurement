"""Mise en forme des mesures nettoyées pour l'affichage (v3) — aucun calcul, aucun fit.

Regroupe les ``Measurement`` d'une électrode par étape, dans l'ordre naturel (bare, probe,
conc_1…), pour les figures de la page Visualisation (``plotting.eis_plots`` /
``plotting.cv_plots``). Fonctionne de la même façon sur une session fraîche et sur une archive
rechargée (les mesures nettoyées sont dans l'archive).
"""

from __future__ import annotations

import numpy as np

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


# ═════════════════════════════════════════════════════════════════════════════
# Moyennes de réplicats pour l'affichage (vue par canal, vue synthétique)
# ═════════════════════════════════════════════════════════════════════════════
# Moyennes et écarts-types (ddof = 1) POINT À POINT, uniquement pour les figures : aucun fit,
# aucune DRT. EIS : grille de fréquences du premier spectre (interpolation en log f si les grilles
# diffèrent, restreinte au recouvrement). CV : courbes de même longueur moyennées point à point
# (même protocole) ; sinon rééchantillonnées sur l'indice normalisé (ordre chronologique).

VIEW_SYNTHESIS = "synthese"


def mean_eis(spectra: list) -> dict:
    """``{"f", "Zre", "Zim", "sd_re", "sd_im", "n"}`` de spectres ``(f, Zre, Zim)``."""
    f0 = np.asarray(spectra[0][0], dtype=float)
    same = all(len(s[0]) == len(f0) and np.allclose(s[0], f0, rtol=1e-3) for s in spectra)
    if not same:
        lo = max(float(np.min(s[0])) for s in spectra)
        hi = min(float(np.max(s[0])) for s in spectra)
        f0 = f0[(f0 >= lo) & (f0 <= hi)]
    zr, zi = [], []
    for f, a, b in spectra:
        f, a, b = (np.asarray(x, dtype=float) for x in (f, a, b))
        if same:
            zr.append(a)
            zi.append(b)
        else:
            o = np.argsort(f)
            zr.append(np.interp(np.log(f0), np.log(f[o]), a[o]))
            zi.append(np.interp(np.log(f0), np.log(f[o]), b[o]))
    zr, zi = np.vstack(zr), np.vstack(zi)
    mod = np.hypot(zr, zi)
    n = len(spectra)
    sd = (lambda a: a.std(axis=0, ddof=1) if n > 1 else np.zeros(a.shape[1]))
    return {"f": f0, "Zre": zr.mean(axis=0), "Zim": zi.mean(axis=0), "sd_re": sd(zr), "sd_im": sd(zi),
            "mod": mod.mean(axis=0), "sd_mod": sd(mod), "n": n}


def mean_cv(curves: list) -> dict:
    """``{"E", "I", "sd", "n"}`` de courbes ``(E, I)`` (ordre chronologique conservé)."""
    m = min(len(c[0]) for c in curves)
    same = all(len(c[0]) == m for c in curves)
    es, is_ = [], []
    for E, I in curves:
        E, I = np.asarray(E, dtype=float), np.asarray(I, dtype=float)
        if not same:
            t, t0 = np.linspace(0, 1, m), np.linspace(0, 1, len(E))
            E, I = np.interp(t, t0, E), np.interp(t, t0, I)
        es.append(E)
        is_.append(I)
    es, is_ = np.vstack(es), np.vstack(is_)
    n = len(curves)
    return {"E": es.mean(axis=0), "I": is_.mean(axis=0),
            "sd": is_.std(axis=0, ddof=1) if n > 1 else np.zeros(m), "n": n}


def _mean_of(kind, ms):
    if kind == KIND_EIS:
        return mean_eis([(m.data["f"], m.data["Zre"], m.data["Zim"]) for m in ms])
    return mean_cv([(m.data["E"], m.data["I"]) for m in ms])


def mean_groups(measurements: list, experiment: Experiment, kind: str, channel_id: int,
                electrodes=(1, 2), steps=None, replicates=None) -> list:
    """Une moyenne par étape, réplicats des ``electrodes`` données regroupés.

    Returns:
        ``[{"label", "step", "concentration", "channel_id", "electrodes", "mean"}]`` (ordre naturel).
    """
    ch = experiment.channel(channel_id)
    by_step: dict = {}
    for m in measurements or []:
        if (m.kind == kind and m.channel_id == channel_id and m.electrode in electrodes
                and (steps is None or m.step in steps) and (replicates is None or m.replicate in replicates)):
            by_step.setdefault(m.step, []).append(m)
    out = []
    for step in sorted(by_step, key=step_sort_key):
        conc = ch.concentration_of(step) if ch else None
        out.append({"label": step_display_label(step, conc), "step": step, "concentration": conc,
                    "channel_id": channel_id, "electrodes": tuple(electrodes),
                    "mean": _mean_of(kind, by_step[step])})
    return out


def channel_mean_view(measurements: list, experiment: Experiment, kind: str, channel_id: int,
                      steps=None, replicates=None) -> list:
    """3ᵉ graphe d'un canal : moyenne électrode 1, moyenne électrode 2, moyenne des deux, par étape."""
    out = []
    for els, suffix, style in (((1,), "E1", "dash"), ((2,), "E2", "dot"), ((1, 2), "E1+E2", "solid")):
        for g in mean_groups(measurements, experiment, kind, channel_id, els, steps, replicates):
            g["label"] = f"{g['label']} — {suffix}"
            g["dash"] = style
            g["band"] = False
            out.append(g)
    return out


def synthesis_view(measurements: list, experiment: Experiment, kind: str, biomarkers=None) -> list:
    """Vue synthétique : pour chaque canal et chaque étape, moyenne des réplicats des DEUX électrodes,
    avec sa zone d'erreur (± écart-type)."""
    out = []
    for ch in experiment.channels:
        if biomarkers is not None and ch.biomarker not in biomarkers:
            continue
        for g in mean_groups(measurements, experiment, kind, ch.id):
            g["label"] = f"C{ch.id} {ch.biomarker} · {g['label']}"
            g["dash"] = ["solid", "dash", "dot", "dashdot"][(ch.id - 1) % 4]
            g["band"] = True
            out.append(g)
    return out
