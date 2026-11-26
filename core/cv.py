"""CV : courant de pic anodique et signal normalisé ΔI/I (v3) — logique pure, aucun import Streamlit.

* **Pic anodique** d'une courbe = maximum de I sur la branche ANODIQUE (E croissant, dans
  l'ordre de mesure) du cycle retenu au nettoyage, restreint à la fenêtre de potentiel
  ``cv.peak_window_V``. Pas de correction de ligne de base en v1 (option future : soustraire
  une droite ajustée sur le pied de pic avant le maximum — voir docs/ARCHITECTURE.md).
* **Signal normalisé** : ``ΔI/I = (I_probe − I) / I_probe``, par électrode, où ``I_probe`` est
  la MOYENNE des courants de pic des réplicats « probe » de CETTE électrode.
"""

from __future__ import annotations

import numpy as np

from core.experiment import KIND_CV, STEP_PROBE, Experiment
from core.results import CVFeature


def anodic_branch_mask(E) -> np.ndarray:
    """Points de la branche anodique (E croissant dans l'ordre de mesure)."""
    E = np.asarray(E, dtype=float)
    if E.size < 2:
        return np.zeros(E.size, dtype=bool)
    return np.gradient(E) > 0


def anodic_peak(E, I, window) -> tuple:
    """(I_pic (A), E_pic (V), ok, message) : maximum de I sur la branche anodique dans ``window``."""
    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)
    lo, hi = float(window[0]), float(window[1])
    mask = anodic_branch_mask(E) & (E >= lo) & (E <= hi) & np.isfinite(I)
    if not mask.any():
        return (float("nan"), float("nan"), False,
                f"aucun point de la branche anodique dans la fenêtre [{lo:g} ; {hi:g}] V (cv.peak_window_V)")
    idx = np.flatnonzero(mask)
    k = idx[int(np.argmax(I[idx]))]
    return float(I[k]), float(E[k]), True, ""


def normalized_delta(i_probe: float, i: float) -> float:
    """ΔI/I = (I_probe − I) / I_probe ; NaN si I_probe est nul ou non fini."""
    if not np.isfinite(i_probe) or i_probe == 0 or not np.isfinite(i):
        return float("nan")
    return float((i_probe - i) / i_probe)


def cv_features(measurements: list, experiment: Experiment, window) -> dict:
    """``{measurement_id: CVFeature}`` de toutes les mesures CV, normalisées par électrode."""
    feats = {}
    for m in measurements or []:
        if m.kind != KIND_CV:
            continue
        ch = experiment.channel(m.channel_id)
        i_pk, e_pk, ok, msg = anodic_peak(m.data["E"], m.data["I"], window)
        feats[m.id] = CVFeature(key=m.id, channel_id=m.channel_id, electrode=m.electrode, step=m.step,
                                replicate=m.replicate, concentration=ch.concentration_of(m.step) if ch else None,
                                i_peak=i_pk, e_peak=e_pk, ok=ok, message=msg)
    probes: dict = {}
    for f in feats.values():
        if f.step == STEP_PROBE and f.ok:
            probes.setdefault((f.channel_id, f.electrode), []).append(f.i_peak)
    for f in feats.values():
        vals = probes.get((f.channel_id, f.electrode))
        if not vals:
            if f.ok:
                f.message = "aucun pic « probe » valide sur cette électrode : ΔI/I non calculé"
            continue
        f.i_probe = float(np.mean(vals))
        f.delta_i_norm = normalized_delta(f.i_probe, f.i_peak)
    return feats
