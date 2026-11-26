"""Modèle de données de l'expérience (v3) — sérialisable en JSON, aucun import Streamlit.

Vocabulaire
-----------
* **Biomarqueur** : la cible mesurée (nom libre, persisté par ``core/user_data.py``).
* **Canal** : une voie microfluidique = **deux électrodes** (1 et 2), mesurées séparément.
  Chaque canal porte UN biomarqueur et la liste ordonnée des concentrations injectées
  (mol/L, ordre croissant d'injection).
* **Étape** (``step``) : ``bare`` (électrode nue), ``probe`` (sonde greffée), puis
  ``conc_1``, ``conc_2``… — ``conc_i`` ↔ ``Channel.concentrations[i-1]``.
* **Mesure** (``Measurement``) : UN fichier nettoyé = (canal, électrode, étape, réplicat,
  type EIS|CV). Les réplicats sont conservés individuellement : l'écart-type des
  étalonnages vient d'eux.

Le **mode** d'un canal est déduit, jamais choisi : 1 concentration → « une concentration
par canal » (l'étalonnage regroupe des canaux du même biomarqueur) ; plusieurs →
« plusieurs concentrations sur un canal » (étapes successives sur les mêmes électrodes).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from core.models import EISSpectrum
from core.version import APP_VERSION

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

KIND_EIS = "EIS"
KIND_CV = "CV"
KINDS = (KIND_EIS, KIND_CV)

STEP_BARE = "bare"
STEP_PROBE = "probe"
ELECTRODES = (1, 2)

MODE_SINGLE = "single"   # une concentration par canal
MODE_MULTI = "multi"     # plusieurs concentrations sur un canal
MODE_LABELS = {
    MODE_SINGLE: "une concentration par canal",
    MODE_MULTI: "plusieurs concentrations sur un canal",
}

_CONC_STEP_RE = re.compile(r"^conc_([1-9]\d*)$")


# ─────────────────────────────────────────────────────────────────────────────
# Étapes
# ─────────────────────────────────────────────────────────────────────────────

def conc_step(i: int) -> str:
    """Nom de l'étape de la i-ème concentration (1-indexée) : ``conc_i``."""
    if int(i) < 1:
        raise ValueError(f"indice de concentration invalide : {i}")
    return f"conc_{int(i)}"


def conc_index(step: str) -> Optional[int]:
    """Indice 1-based d'une étape ``conc_i`` ; None pour ``bare``/``probe``/inconnu."""
    m = _CONC_STEP_RE.match(str(step))
    return int(m.group(1)) if m else None


def is_valid_step_name(step: str) -> bool:
    return step in (STEP_BARE, STEP_PROBE) or conc_index(step) is not None


def step_sort_key(step: str) -> tuple:
    """Ordre naturel : bare, probe, conc_1, conc_2…"""
    if step == STEP_BARE:
        return (0, 0)
    if step == STEP_PROBE:
        return (1, 0)
    i = conc_index(step)
    return (2, i) if i is not None else (3, 0)


def format_concentration(c: Optional[float]) -> str:
    """« 1.00e-09 M » ; « — » si absente."""
    if c is None or not math.isfinite(float(c)):
        return "—"
    return f"{float(c):.2e} M"


def concentration_from_parts(mantissa: float, exponent: int) -> float:
    """Concentration saisie mantisse × 10^exposant (mol/L)."""
    return float(mantissa) * (10.0 ** int(exponent))


def concentration_parts(c: float) -> tuple:
    """(mantisse arrondie à 2 décimales, exposant) d'une concentration > 0."""
    c = float(c)
    if not (c > 0 and math.isfinite(c)):
        return 1.0, -9
    e = int(math.floor(math.log10(c)))
    m = round(c / 10.0 ** e, 2)
    if m >= 10.0:            # arrondi 9.999 → 10.0
        m, e = round(m / 10.0, 2), e + 1
    return m, e


# ─────────────────────────────────────────────────────────────────────────────
# Canal et expérience
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Channel:
    """Un canal (2 électrodes implicites), son biomarqueur et ses concentrations."""

    id: int
    biomarker: str
    concentrations: list = field(default_factory=list)

    @property
    def mode(self) -> str:
        return MODE_SINGLE if len(self.concentrations) <= 1 else MODE_MULTI

    def steps(self, include_bare: bool = True) -> list:
        """Étapes valides du canal, dans l'ordre : [bare,] probe, conc_1…conc_n."""
        out = [STEP_BARE] if include_bare else []
        out.append(STEP_PROBE)
        out += [conc_step(i + 1) for i in range(len(self.concentrations))]
        return out

    def has_step(self, step: str) -> bool:
        if step in (STEP_BARE, STEP_PROBE):
            return True
        i = conc_index(step)
        return i is not None and 1 <= i <= len(self.concentrations)

    def concentration_of(self, step: str) -> Optional[float]:
        """Concentration (mol/L) d'une étape ``conc_i`` ; None pour bare/probe ou étape absente."""
        i = conc_index(step)
        if i is None or not (1 <= i <= len(self.concentrations)):
            return None
        return float(self.concentrations[i - 1])

    def to_dict(self) -> dict:
        return {"id": int(self.id), "biomarker": self.biomarker,
                "concentrations": [float(c) for c in self.concentrations]}

    @classmethod
    def from_dict(cls, d: dict) -> "Channel":
        return cls(id=int(d["id"]), biomarker=str(d["biomarker"]),
                   concentrations=[float(c) for c in d.get("concentrations") or []])


@dataclass
class Experiment:
    """Configuration d'une expérience : biomarqueurs et canaux."""

    biomarkers: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    name: str = ""
    app_version: str = APP_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def channel(self, channel_id: int) -> Optional[Channel]:
        return next((c for c in self.channels if c.id == int(channel_id)), None)

    def channel_ids(self) -> list:
        return [c.id for c in self.channels]

    def channels_of(self, biomarker: str) -> list:
        return [c for c in self.channels if c.biomarker == biomarker]

    def to_dict(self) -> dict:
        return {"name": self.name, "biomarkers": list(self.biomarkers),
                "channels": [c.to_dict() for c in self.channels],
                "app_version": self.app_version, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        if not isinstance(d, dict) or "channels" not in d or "biomarkers" not in d:
            raise ValueError("Description d'expérience invalide (clés 'biomarkers' et 'channels' attendues).")
        return cls(biomarkers=[str(b) for b in d["biomarkers"]],
                   channels=[Channel.from_dict(c) for c in d["channels"]],
                   name=str(d.get("name") or ""),
                   app_version=str(d.get("app_version") or APP_VERSION),
                   created_at=str(d.get("created_at") or datetime.now().isoformat(timespec="seconds")))


def default_biomarker_for_channel(k: int, biomarkers: list, n_channels: int) -> Optional[str]:
    """Biomarqueur proposé par défaut pour le canal d'indice ``k`` (0-based).

    Si ``n_channels == len(biomarkers)``, le canal k reçoit le biomarqueur k ; sinon le
    premier biomarqueur.
    """
    if not biomarkers:
        return None
    if n_channels == len(biomarkers) and 0 <= k < len(biomarkers):
        return biomarkers[k]
    return biomarkers[0]


def validate_experiment(exp: Experiment) -> list:
    """Erreurs de saisie (liste vide = expérience valide). Messages destinés à l'utilisateur."""
    errors = []
    names = [str(b).strip() for b in exp.biomarkers]
    if not names:
        errors.append("Déclarez au moins un biomarqueur.")
    if any(not n for n in names):
        errors.append("Chaque biomarqueur doit avoir un nom non vide.")
    lowered = [n.lower() for n in names if n]
    dup = sorted({n for n in names if n and lowered.count(n.lower()) > 1})
    if dup:
        errors.append(f"Noms de biomarqueurs en double : {', '.join(dup)}.")
    if not exp.channels:
        errors.append("Déclarez au moins un canal.")
    ids = [c.id for c in exp.channels]
    if len(set(ids)) != len(ids):
        errors.append("Identifiants de canaux en double.")
    for ch in exp.channels:
        if ch.biomarker not in names:
            errors.append(f"Canal {ch.id} : biomarqueur « {ch.biomarker} » non déclaré.")
        if not ch.concentrations:
            errors.append(f"Canal {ch.id} : au moins une concentration est requise.")
        for i, c in enumerate(ch.concentrations, start=1):
            if not (isinstance(c, (int, float)) and math.isfinite(c) and c > 0):
                errors.append(f"Canal {ch.id} : la concentration {i} doit être > 0 (reçu {c!r}).")
    return errors


def experiment_summary_rows(exp: Experiment) -> list:
    """Lignes du récapitulatif tabulaire (une par canal)."""
    return [{
        "Canal": ch.id,
        "Biomarqueur": ch.biomarker,
        "Électrodes": "1 et 2",
        "Mode (déduit)": MODE_LABELS[ch.mode],
        "Nb concentrations": len(ch.concentrations),
        "Concentrations (mol/L)": ", ".join(f"{c:.2e}" for c in ch.concentrations),
    } for ch in exp.channels]


# ─────────────────────────────────────────────────────────────────────────────
# Mesure nettoyée
# ─────────────────────────────────────────────────────────────────────────────

def measurement_id(channel_id: int, electrode: int, step: str, replicate: int, kind: str) -> str:
    """Identifiant stable et lisible : ``EIS_C1_E2_conc_3_r1``."""
    return f"{kind}_C{int(channel_id)}_E{int(electrode)}_{step}_r{int(replicate)}"


def electrode_key(channel_id: int, electrode: int) -> str:
    """Clé d'électrode lisible : ``C1·E2``."""
    return f"C{int(channel_id)}·E{int(electrode)}"


@dataclass
class Measurement:
    """Une mesure NETTOYÉE.

    ``data`` : EIS → ``{"f", "Zre", "Zim"}`` (Hz, Ω, Zim = −Im(Z) > 0, HF→BF) ;
    CV → ``{"E", "I", "cycle"}`` (V, A, n° de cycle), ordre chronologique.
    """

    channel_id: int
    electrode: int
    step: str
    replicate: int
    kind: str
    source_filename: str
    data: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return measurement_id(self.channel_id, self.electrode, self.step, self.replicate, self.kind)

    @property
    def electrode_key(self) -> str:
        return electrode_key(self.channel_id, self.electrode)

    def to_spectrum(self, experiment: Optional[Experiment] = None) -> EISSpectrum:
        """Spectre EIS de la mesure, au format du pipeline (``core.pipeline``).

        Étape du pipeline : 'bare' / 'probe' / 'hybridization' (``conc_i``), concentration
        lue dans le canal (0.0 pour bare/probe).
        """
        if self.kind != KIND_EIS:
            raise ValueError(f"{self.id} n'est pas une mesure EIS.")
        conc = None
        if experiment is not None and experiment.channel(self.channel_id) is not None:
            conc = experiment.channel(self.channel_id).concentration_of(self.step)
        step = self.step if self.step in (STEP_BARE, STEP_PROBE) else "hybridization"
        f = np.asarray(self.data["f"], dtype=float)
        return EISSpectrum(label=self.id, f=f.copy(), Zre=np.asarray(self.data["Zre"], dtype=float).copy(),
                           Zim=np.asarray(self.data["Zim"], dtype=float).copy(),
                           concentration=float(conc or 0.0), step=step, n_points=len(f),
                           source_files=[self.source_filename])

    def meta_dict(self) -> dict:
        """Métadonnées sans les données (table d'assignation, manifest)."""
        return {"id": self.id, "channel_id": int(self.channel_id), "electrode": int(self.electrode),
                "step": self.step, "replicate": int(self.replicate), "kind": self.kind,
                "source_filename": self.source_filename, "warnings": list(self.warnings)}


def sort_measurements(measurements: list) -> list:
    """Tri stable : type, canal, électrode, étape (ordre naturel), réplicat."""
    return sorted(measurements, key=lambda m: (m.kind, m.channel_id, m.electrode,
                                               step_sort_key(m.step), m.replicate))
