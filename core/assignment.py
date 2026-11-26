"""Assignation manuelle des fichiers et nettoyage (v3) — aucun import Streamlit.

Une ligne d'assignation par fichier déposé : ``fichier | type | canal | électrode | étape |
réplicat | exclure``. Le pré-remplissage (:func:`guess_rows`) est un *best effort* tiré du
contenu (EIS/CV) et du nom du fichier ; l'utilisateur corrige toujours à la main.

:func:`check_assignment` liste les incohérences avec un message clair ; une erreur bloque le
nettoyage, un avertissement non. :func:`clean_files` produit les ``Measurement`` nettoyées :

* EIS : ``core.loader.load_spectrum`` (séparateur auto, signe de Zim EC-Lab, suppression
  des fréquences parasites 50/100 Hz, tri HF → BF), puis retrait des points exclus à la main ;
* CV : ``core.cv_loader.load_cv_measurement`` (colonnes auto, courant en A, cycle retenu,
  ordre chronologique).

Les réplicats sont conservés individuellement (aucun moyennage ici).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from core.cv_loader import load_cv_measurement
from core.experiment import (
    ELECTRODES,
    KIND_CV,
    KIND_EIS,
    KINDS,
    STEP_BARE,
    STEP_PROBE,
    Experiment,
    Measurement,
    conc_step,
    is_valid_step_name,
    sort_measurements,
)
from core.loader import load_spectrum, parse_robust
from core.logger import get_logger

log = get_logger("assignment")

LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"


@dataclass
class Issue:
    level: str
    message: str
    file_id: Optional[str] = None


@dataclass
class AssignmentRow:
    file_id: str
    filename: str
    kind: Optional[str] = None
    channel: Optional[int] = None
    electrode: Optional[int] = None
    step: Optional[str] = None
    replicate: Optional[int] = None
    exclude: bool = False
    detected_kind: Optional[str] = None
    file_concentration: Optional[float] = None   # concentration lue dans le nom (import par dossier)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AssignmentRow":
        def _int(v):
            try:
                return None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)
            except (TypeError, ValueError):
                return None
        return cls(file_id=str(d["file_id"]), filename=str(d.get("filename") or d["file_id"]),
                   kind=d.get("kind") or None, channel=_int(d.get("channel")),
                   electrode=_int(d.get("electrode")), step=d.get("step") or None,
                   replicate=_int(d.get("replicate")), exclude=bool(d.get("exclude") or False),
                   detected_kind=d.get("detected_kind") or None,
                   file_concentration=(None if d.get("file_concentration") is None
                                       else float(d["file_concentration"])))

    def key(self) -> tuple:
        return (self.kind, self.channel, self.electrode, self.step, self.replicate)


# ─────────────────────────────────────────────────────────────────────────────
# Détection du type et pré-remplissage
# ─────────────────────────────────────────────────────────────────────────────

def detect_kind(content: bytes) -> Optional[str]:
    """'EIS' ou 'CV' d'après les colonnes du fichier ; None si indéterminé."""
    pf = parse_robust(content)
    if pf is not None and pf.kind in KINDS:
        return pf.kind
    head = content[:4000].decode("utf-8", errors="replace").lower() if isinstance(content, bytes) else ""
    if "freq" in head and ("re(z)" in head or "zre" in head or "zreal" in head):
        return KIND_EIS
    if ("ewe" in head or "potential" in head) and ("<i>" in head or "current" in head):
        return KIND_CV
    return None


_RE_CHANNEL = re.compile(r"(?:canal|channel|chan|ch)[ _\-]?(\d+)", re.I)
_RE_ELECTRODE = re.compile(r"(?:^|[ _\-.])(?:electrode|elec|el|we|e)[ _\-]?([12])(?=$|[ _\-.])", re.I)
_RE_REPLICATE = re.compile(r"(?:^|[ _\-.])(?:rep|réplicat|replicat|r)[ _\-]?(\d+)(?=$|[ _\-.])", re.I)


def guess_from_filename(filename: str) -> dict:
    """Indices tirés du nom de fichier (peut ne rien trouver)."""
    stem = filename.rsplit(".", 1)[0]
    low = stem.lower()
    out: dict = {}
    if re.search(r"(?:^|[ _\-.])(?:peis|eis)(?=$|[ _\-.])", low):
        out["kind"] = KIND_EIS
    elif re.search(r"(?:^|[ _\-.])cv(?=$|[ _\-.])", low):
        out["kind"] = KIND_CV
    if re.search(r"(?:^|[ _\-.])(?:bare|nue?)(?=$|[ _\-.])", low):
        out["step"] = STEP_BARE
    elif re.search(r"(?:probe|sonde)", low):
        out["step"] = STEP_PROBE
    else:
        m = re.search(r"(?:^|[ _\-.])(?:conc|hyb)[ _\-]?(\d+)(?=$|[ _\-.])", low)
        if m and int(m.group(1)) >= 1:
            out["step"] = conc_step(int(m.group(1)))
    m = _RE_CHANNEL.search(stem)
    if m:
        out["channel"] = int(m.group(1))
    m = _RE_ELECTRODE.search(stem)
    if m:
        out["electrode"] = int(m.group(1))
    m = _RE_REPLICATE.search(stem)
    if m and int(m.group(1)) >= 1:
        out["replicate"] = int(m.group(1))
    return out


def guess_rows(files: list, experiment: Experiment) -> list:
    """Lignes d'assignation pré-remplies (best effort).

    Args:
        files: ``[(file_id, filename, detected_kind)]`` dans l'ordre de dépôt.
        experiment: configuration validée (canaux existants).

    Le type détecté dans le contenu prime sur le nom ; le canal par défaut est le premier
    canal déclaré ; l'électrode par défaut 1 ; un réplicat absent du nom est numéroté dans
    l'ordre de dépôt parmi les fichiers de même (type, canal, électrode, étape).
    """
    ids = experiment.channel_ids() if experiment else []
    rows, counters = [], {}
    for file_id, filename, detected in files:
        g = guess_from_filename(filename)
        ch = g.get("channel") if g.get("channel") in ids else (ids[0] if ids else None)
        step = g.get("step")
        if step and experiment and ch is not None and not experiment.channel(ch).has_step(step):
            step = None
        row = AssignmentRow(file_id=file_id, filename=filename, kind=detected or g.get("kind"),
                            channel=ch, electrode=g.get("electrode", 1), step=step,
                            replicate=g.get("replicate"), detected_kind=detected)
        rows.append(row)
    for row in rows:
        if row.replicate is None and row.step is not None:
            k = (row.kind, row.channel, row.electrode, row.step)
            taken = {r.replicate for r in rows if r.replicate is not None and
                     (r.kind, r.channel, r.electrode, r.step) == k}
            n = counters.get(k, 0) + 1
            while n in taken:
                n += 1
            counters[k] = n
            row.replicate = n
    return rows


def step_options(experiment: Experiment) -> list:
    """Étapes proposées dans la table : bare, probe, conc_1…conc_max (tous canaux)."""
    n = max((len(c.concentrations) for c in experiment.channels), default=0)
    return [STEP_BARE, STEP_PROBE] + [conc_step(i) for i in range(1, n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Contrôles de cohérence
# ─────────────────────────────────────────────────────────────────────────────

def _desc(r: AssignmentRow) -> str:
    return f"{r.kind} canal {r.channel} électrode {r.electrode} étape {r.step} réplicat {r.replicate}"


def check_assignment(rows: list, experiment: Experiment) -> list:
    """Incohérences de l'assignation. Erreurs : bloquent le nettoyage ; avertissements : non."""
    issues = []
    active = [r for r in rows if not r.exclude]
    if not active:
        issues.append(Issue(LEVEL_ERROR, "Aucun fichier à nettoyer : déposez des fichiers (ou décochez « exclure »)."))
        return issues
    for r in active:
        missing = [lbl for lbl, v in (("type", r.kind), ("canal", r.channel), ("électrode", r.electrode),
                                      ("étape", r.step), ("réplicat", r.replicate)) if v in (None, "")]
        if missing:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : fichier non assigné ({', '.join(missing)} "
                                             "manquant).", r.file_id))
            continue
        if r.kind not in KINDS:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : type « {r.kind} » inconnu (EIS ou CV).", r.file_id))
        if r.detected_kind and r.kind != r.detected_kind:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : assigné {r.kind} mais le contenu est un "
                                             f"fichier {r.detected_kind}.", r.file_id))
        if r.detected_kind is None:
            issues.append(Issue(LEVEL_WARNING, f"« {r.filename} » : type non reconnu dans le contenu — "
                                               "vérifiez qu'il s'agit bien d'un export EC-Lab.", r.file_id))
        ch = experiment.channel(r.channel)
        if ch is None:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : canal {r.channel} inexistant (canaux "
                                             f"déclarés : {', '.join(map(str, experiment.channel_ids()))}).",
                                r.file_id))
            continue
        if r.electrode not in ELECTRODES:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : électrode {r.electrode} invalide (1 ou 2).",
                                r.file_id))
        if not is_valid_step_name(r.step):
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : étape « {r.step} » invalide.", r.file_id))
        elif not ch.has_step(r.step):
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : étape {r.step} inexistante pour le canal "
                                             f"{ch.id}, qui déclare {len(ch.concentrations)} concentration(s).",
                                r.file_id))
        if r.replicate is not None and r.replicate < 1:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : réplicat {r.replicate} invalide (≥ 1).", r.file_id))

    complete = [r for r in active if None not in (r.kind, r.channel, r.electrode, r.step, r.replicate)]
    seen: dict = {}
    for r in complete:
        seen.setdefault(r.key(), []).append(r)
    for key, rs in seen.items():
        if len(rs) > 1:
            names = " et ".join(f"« {x.filename} »" for x in rs)
            issues.append(Issue(LEVEL_ERROR, f"Doublon : {names} sont tous assignés à {_desc(rs[0])}.",
                                rs[0].file_id))

    by_elec: dict = {}
    for r in complete:
        by_elec.setdefault((r.kind, r.channel, r.electrode), set()).add(r.step)
    for (kind, ch, el), steps in sorted(by_elec.items(), key=lambda x: str(x[0])):
        if STEP_PROBE not in steps:
            issues.append(Issue(LEVEL_WARNING, f"{kind} canal {ch} électrode {el} : aucune mesure « probe » — "
                                               "la normalisation (étalonnage) sera impossible pour cette "
                                               "électrode."))
        reps = {}
        for r in complete:
            if (r.kind, r.channel, r.electrode) == (kind, ch, el):
                reps.setdefault(r.step, []).append(r.replicate)
        for step, rr in reps.items():
            if sorted(rr) != list(range(1, len(rr) + 1)):
                issues.append(Issue(LEVEL_WARNING, f"{kind} canal {ch} électrode {el} étape {step} : réplicats "
                                                   f"{sorted(rr)} non consécutifs depuis 1."))
    return issues


def has_errors(issues: list) -> bool:
    return any(i.level == LEVEL_ERROR for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# Nettoyage
# ─────────────────────────────────────────────────────────────────────────────

def clean_eis(content: bytes, label: str, config: Optional[dict] = None,
              excluded_points: Optional[list] = None) -> tuple:
    """(data {"f","Zre","Zim"}, warnings) d'un fichier EIS nettoyé, points exclus retirés."""
    warnings: list = []
    sp = load_spectrum(content=content, label=label, config=config, warnings_out=warnings)
    f, zr, zi = np.asarray(sp.f, float), np.asarray(sp.Zre, float), np.asarray(sp.Zim, float)
    if excluded_points:
        keep = np.ones(len(f), dtype=bool)
        idx = [int(i) for i in excluded_points if 0 <= int(i) < len(f)]
        keep[idx] = False
        f, zr, zi = f[keep], zr[keep], zi[keep]
        warnings.append(f"{len(idx)} point(s) retiré(s) à la main.")
        if len(f) < 5:
            raise ValueError(f"Fichier {label} : moins de 5 points après retrait des points exclus.")
    return {"f": f, "Zre": zr, "Zim": zi}, warnings


def clean_files(rows: list, contents: dict, experiment: Experiment, config: Optional[dict] = None,
                excluded_points: Optional[dict] = None) -> tuple:
    """Nettoie les fichiers assignés (non exclus).

    Args:
        rows: lignes d'assignation (``AssignmentRow``) — doivent passer :func:`check_assignment`.
        contents: {file_id: bytes} (les .mpr ont déjà été convertis en texte).
        experiment: configuration validée.
        config: config de l'app (fréquences parasites, cycle CV).
        excluded_points: {file_id: [indices]} de points EIS retirés à la main.

    Returns:
        (measurements triées, issues) — un fichier illisible est signalé, les autres sont gardés.
    """
    issues = check_assignment(rows, experiment)
    if has_errors(issues):
        return [], issues
    cycle = ((config or {}).get("cv") or {}).get("cycle", "last")
    excluded_points = excluded_points or {}
    measurements = []
    for r in rows:
        if r.exclude:
            continue
        content = contents.get(r.file_id)
        if content is None:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : contenu introuvable (redéposez le fichier).",
                                r.file_id))
            continue
        try:
            if r.kind == KIND_EIS:
                data, warns = clean_eis(content, r.filename, config, excluded_points.get(r.file_id))
            else:
                d = load_cv_measurement(content, r.filename, cycle)
                data = {"E": d["E"], "I": d["I"], "cycle": d["cycle"]}
                warns = d["warnings"] + ([f"cycle {d['cycle_kept']} conservé sur {len(d['cycles'])}."]
                                         if len(d["cycles"]) > 1 else [])
        except ValueError as exc:
            issues.append(Issue(LEVEL_ERROR, f"« {r.filename} » : {exc}", r.file_id))
            log.warning(f"Nettoyage impossible de {r.filename} : {exc}")
            continue
        measurements.append(Measurement(channel_id=r.channel, electrode=r.electrode, step=r.step,
                                        replicate=r.replicate, kind=r.kind, source_filename=r.filename,
                                        data=data, warnings=list(warns)))
    return sort_measurements(measurements), issues


def measurement_summary_rows(measurements: list) -> list:
    """Aperçu des mesures nettoyées (une ligne par mesure)."""
    out = []
    for m in measurements:
        if m.kind == KIND_EIS:
            f = np.asarray(m.data["f"])
            span = f"{f.max():.3g} → {f.min():.3g} Hz" if f.size else "—"
        else:
            E = np.asarray(m.data["E"])
            span = f"{E.min():.3g} → {E.max():.3g} V" if E.size else "—"
        out.append({"Mesure": m.id, "Fichier": m.source_filename, "Type": m.kind, "Canal": m.channel_id,
                    "Électrode": m.electrode, "Étape": m.step, "Réplicat": m.replicate,
                    "Points": int(len(next(iter(m.data.values())))), "Plage": span,
                    "Remarques": " ; ".join(m.warnings)})
    return out
