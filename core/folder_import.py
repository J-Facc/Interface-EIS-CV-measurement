"""Import d'un DOSSIER de mesures par canal (v3) — aucun import Streamlit.

**Convention principale : noms EC-Lab** ``<partie libre>_<NN>_<CV|PEIS>_C<nn>.<ext>`` — la fin est
fixe (``NN`` = n° de réplicat, ``CV``/``PEIS`` = technique, ``Cnn`` = voie de l'instrument) ; la
partie libre, définie par l'utilisateur, contient TOUJOURS l'électrode (``w3`` → électrode 3) et
``bare``, ``probe`` ou une concentration (``10-18`` = 1e-18 M ; aussi ``1e-18``, ``5x10-12``…),
à une position quelconque. Ex. ``10-18_w3_01_CV_C01.mpr``, ``w4_probe_06_PEIS_C02.mpr``. Les
autres techniques (CA, OCV…) sont ignorées. Les numéros de réplicat du nom sont renumérotés
1, 2, 3… par (technique, électrode, étape).

Conventions historiques, toujours acceptées :

Arborescence attendue (celle du poste de mesure ; les dossiers intermédiaires sont facultatifs,
seuls les NOMS de fichiers font foi) ::

    <dossier du canal>/
    ├── probe/{eis,cv}/electrode_<N>/probe_<eis|cv>_e<N>_r<k>.<txt|csv|mpt|mpr>
    ├── calibration/{eis,cv}/electrode_<N>/<eis|cv>_e<N>_<c>M_r<k>.<ext>    ex. eis_e1_1.00e-08M_r2.txt
    └── bare/{eis,cv}/electrode_<N>/bare_<eis|cv>_e<N>_r<k>.<ext>            (facultatif)

Chaque nom donne la technique, l'électrode, le réplicat et, pour la calibration, la
concentration (mol/L). La concentration est rattachée à l'étape ``conc_i`` du canal par sa
VALEUR (tolérance relative 0,5 % : les noms arrondissent à 3 chiffres). Les ``.mpr`` sont
convertis en texte (``core.mpr_converter``) avant toute lecture.

Les fichiers dont le nom ne suit pas ces conventions (réglages .mps, journaux, exports annexes…)
sont IGNORÉS : ni lus, ni convertis, ni importés — seulement comptés (``FolderImport.ignored``).
Une même mesure présente sous plusieurs extensions (``.mpr`` + ``.txt`` exporté) n'est importée
qu'une fois, version texte de préférence (``.txt`` > ``.csv`` > ``.mpt`` > ``.mpr``).

Contrôles signalés : incohérence dossier/nom (technique, électrode, probe/calibration), plus de
deux électrodes dans le dossier d'un canal, concentration absente de la configuration du canal.
Des électrodes numérotées autrement que 1 et 2 (ex. 3 et 4 : numérotation globale de la puce)
sont renumérotées 1 et 2 dans l'ordre croissant, avec un avertissement.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Optional

from core.assignment import LEVEL_ERROR, LEVEL_WARNING, AssignmentRow, Issue, detect_kind
from core.experiment import KIND_CV, KIND_EIS, STEP_BARE, STEP_PROBE, Experiment, conc_step
from core.mpr_converter import is_mpr, mpr_to_csv_bytes

#: Extensions acceptées dans un dossier de canal.
EXTENSIONS = ("txt", "csv", "mpt", "mpr")
_EXT = r"\.(?:txt|csv|mpt|mpr)$"

RE_CALIB = re.compile(r"^(?P<tech>cv|eis)_e(?P<electrode>\d+)_(?P<conc>\d+(?:\.\d+)?e[+-]?\d+)M_r(?P<rep>\d+)"
                      + _EXT, re.IGNORECASE)
RE_PROBE = re.compile(r"^probe_(?P<tech>cv|eis)_e(?P<electrode>\d+)_r(?P<rep>\d+)" + _EXT, re.IGNORECASE)
RE_BARE = re.compile(r"^bare_(?P<tech>cv|eis)_e(?P<electrode>\d+)_r(?P<rep>\d+)" + _EXT, re.IGNORECASE)

#: Nom EC-Lab : <partie libre>_<réplicat>_<CV|PEIS>_C<voie de l'instrument>.<ext>
#: ex. 10-18_w3_01_CV_C01.mpr, w4_probe_06_PEIS_C02.mpr. Toute autre technique (CA, OCV…) est ignorée.
RE_ECLAB = re.compile(r"^(?P<free>.+?)_(?P<rep>\d+)_(?P<tech>CV|PEIS)_C(?P<chan>\d+)" + _EXT, re.IGNORECASE)
RE_ECLAB_ANY = re.compile(r"^.+_\d+_(?P<tech>[A-Za-z]+)_C\d+" + _EXT, re.IGNORECASE)
#: Électrode dans la partie libre : « w » collé au numéro (w3), délimité.
_RE_W = re.compile(r"(?:^|[^a-z0-9])w(\d+)(?![0-9])", re.IGNORECASE)
#: Concentration dans la partie libre : 10-18, 1E-18M, 1e-18, 5x10-12, 2.5e-9 (mol/L).
_RE_CONC = [
    re.compile(r"(?:^|[^0-9.])(?:(?P<m>\d+(?:[.,]\d+)?)\s*[x*])?10[-\u2212](?P<e>\d+)(?:\s*M)?(?![0-9])",
               re.IGNORECASE),
    re.compile(r"(?:^|[^0-9.])(?P<m>\d+(?:[.,]\d+)?)e[-\u2212](?P<e>\d+)(?:\s*M)?(?![0-9])", re.IGNORECASE),
]
_RE_STEP = {"bare": re.compile(r"bare", re.IGNORECASE), "probe": re.compile(r"probe|sonde", re.IGNORECASE)}

#: Préférence entre extensions quand une même mesure est présente plusieurs fois.
_EXT_PREFERENCE = (".txt", ".csv", ".mpt", ".mpr")

#: Tolérance relative du rattachement concentration du nom ↔ concentration configurée.
CONC_RTOL = 5e-3


@dataclass
class ParsedName:
    """Métadonnées tirées d'un chemin de fichier."""

    dataset: str                 # "probe" | "calibration" | "bare"
    kind: str                    # "EIS" | "CV"
    electrode: int               # numéro tel qu'écrit dans le nom
    replicate: int
    concentration: Optional[float] = None
    warnings: list = field(default_factory=list)
    instrument_channel: Optional[int] = None    # C01, C02… (nom EC-Lab), information seulement


def _parse_eclab(fname: str) -> tuple:
    """(ParsedName | None, raison) d'un nom EC-Lab ``<libre>_<rép>_<CV|PEIS>_C<nn>.<ext>``.

    Retourne (None, "") si le nom n'a pas la forme EC-Lab (on essaie alors les autres conventions).
    """
    m = RE_ECLAB.match(fname)
    if m is None:
        other = RE_ECLAB_ANY.match(fname)
        if other is not None:
            return None, f"technique {other['tech'].upper()} (seules CV et PEIS sont importées)"
        return None, ""
    free = m["free"]
    ws = {int(x) for x in _RE_W.findall(free)}
    if len(ws) != 1:
        return None, ("électrode « wN » introuvable dans le nom" if not ws
                      else f"plusieurs électrodes dans le nom ({', '.join(f'w{w}' for w in sorted(ws))})")
    free_wo_w = _RE_W.sub(" ", free)
    steps = [k for k, rx in _RE_STEP.items() if rx.search(free_wo_w)]
    concs = set()
    for rx in _RE_CONC:
        for cm in rx.finditer(free_wo_w):
            mant = float((cm["m"] or "1").replace(",", "."))
            concs.add(mant * 10.0 ** (-int(cm["e"])))
    if len(steps) + (1 if concs else 0) != 1 or len(concs) > 1:
        found = steps + [f"{c:.2e} M" for c in sorted(concs)]
        return None, ("ni bare, ni probe, ni concentration dans le nom" if not found
                      else f"étape ambiguë dans le nom ({', '.join(found)})")
    dataset = steps[0] if steps else "calibration"
    return ParsedName(dataset=dataset, kind=KIND_EIS if m["tech"].upper() == "PEIS" else KIND_CV,
                      electrode=ws.pop(), replicate=int(m["rep"]),
                      concentration=concs.pop() if concs else None,
                      instrument_channel=int(m["chan"])), ""


def classify_path(relpath: str) -> tuple:
    """(ParsedName | None, raison de l'ignorance). Nom EC-Lab d'abord, puis conventions historiques."""
    fname = PurePosixPath(relpath.replace("\\", "/")).name
    parsed, why = _parse_eclab(fname)
    if parsed is not None or why:
        return parsed, why
    parsed = parse_path(relpath)
    return (parsed, "") if parsed is not None else (None, "nom hors conventions (ni …_NN_CV_Cnn ni …_NN_PEIS_Cnn)")


def parse_path(relpath: str) -> Optional[ParsedName]:
    """Analyse ``relpath`` (chemin relatif, séparateurs / ou \\). None si le nom est inattendu.

    Les dossiers parents connus (probe/calibration/bare, eis/cv, electrode_<N>) sont confrontés
    au nom : une incohérence est rangée dans ``warnings``.
    """
    parts = [p for p in PurePosixPath(relpath.replace("\\", "/")).parts if p not in ("", ".")]
    if not parts:
        return None
    fname = parts[-1]
    for dataset, rx in (("calibration", RE_CALIB), ("probe", RE_PROBE), ("bare", RE_BARE)):
        m = rx.match(fname)
        if m:
            break
    else:
        return None
    out = ParsedName(dataset=dataset, kind=KIND_EIS if m["tech"].lower() == "eis" else KIND_CV,
                     electrode=int(m["electrode"]), replicate=int(m["rep"]),
                     concentration=float(m["conc"]) if dataset == "calibration" else None)
    folders = [p.lower() for p in parts[:-1]]
    for d in ("probe", "calibration", "bare"):
        if d in folders and d != dataset:
            out.warnings.append(f"incohérence dossier/nom : rangé dans « {d}/ » mais nommé comme « {dataset} »")
    for t in ("eis", "cv"):
        if t in folders and t != m["tech"].lower():
            out.warnings.append(f"incohérence dossier/nom : rangé dans « {t}/ » mais nommé « {m['tech']} »")
    for f in folders:
        em = re.fullmatch(r"electrode_(\d+)", f)
        if em and int(em.group(1)) != out.electrode:
            out.warnings.append(f"incohérence dossier/nom : dossier « {f} » mais électrode e{out.electrode} "
                                "dans le nom")
    return out


def match_concentration(c: float, concentrations: list) -> Optional[int]:
    """Indice 1-based de ``c`` parmi ``concentrations`` (tolérance relative CONC_RTOL), sinon None."""
    for i, ref in enumerate(concentrations, start=1):
        if ref > 0 and math.isclose(c, ref, rel_tol=CONC_RTOL):
            return i
    return None


@dataclass
class FolderImport:
    """Résultat de l'import du dossier d'UN canal."""

    channel_id: int
    raw_files: dict = field(default_factory=dict)     # {file_id: {"name", "content", "kind"}}
    rows: list = field(default_factory=list)          # [AssignmentRow]
    issues: list = field(default_factory=list)        # [Issue]
    concentrations_found: list = field(default_factory=list)   # mol/L, triées, uniques
    ignored: list = field(default_factory=list)       # [(chemin, raison)] — fichiers non importés

    def summary(self) -> dict:
        ok = [r for r in self.rows if not r.exclude]
        return {"fichiers": len(self.rows), "retenus": len(ok),
                "EIS": sum(r.kind == KIND_EIS for r in ok), "CV": sum(r.kind == KIND_CV for r in ok),
                "électrodes": sorted({r.electrode for r in ok if r.electrode}),
                "concentrations": len(self.concentrations_found)}


def same_concentrations(a: list, b: list) -> bool:
    """Mêmes concentrations, dans le même ordre, à CONC_RTOL près."""
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=CONC_RTOL) for x, y in zip(a, b))


def _unique_sorted(values: list) -> list:
    out = []
    for v in sorted(values):
        if not out or not math.isclose(v, out[-1], rel_tol=CONC_RTOL):
            out.append(v)
    return out


def import_channel_folder(uploads: list, experiment: Experiment, channel_id: int) -> FolderImport:
    """Lignes d'assignation et contenus d'un dossier de canal.

    Args:
        uploads: ``[(chemin relatif, octets)]`` (nom d'un fichier de ``st.file_uploader`` en mode
            dossier = son chemin dans le dossier choisi).
        experiment: configuration validée.
        channel_id: canal auquel TOUT le dossier appartient.
    """
    res = FolderImport(channel_id=channel_id)
    ch = experiment.channel(channel_id)
    # 1. Tri sur les NOMS seulement : un fichier inutile n'est ni lu ni converti.
    candidates: dict = {}
    for relpath, data in uploads:
        p, why = classify_path(relpath)
        if p is None:
            res.ignored.append((relpath, why))
            continue
        key = (p.dataset, p.kind, p.electrode, p.replicate, p.concentration)
        candidates.setdefault(key, []).append((relpath, data, p))
    selected = []
    for group in candidates.values():
        group.sort(key=lambda it: _EXT_PREFERENCE.index(PurePosixPath(it[0].replace("\\", "/")).suffix.lower()))
        selected.append(group[0])
        for relpath, _d, _p in group[1:]:
            res.ignored.append((relpath, f"même mesure que « {group[0][0]} » (version retenue)"))
    selected.sort(key=lambda it: it[0])

    parsed = []
    for relpath, data, p in selected:
        name = PurePosixPath(relpath.replace("\\", "/")).name
        if is_mpr(name):
            try:
                data = mpr_to_csv_bytes(data)
            except Exception as exc:             # lecteur .mpr tiers : message, jamais de crash
                res.issues.append(Issue(LEVEL_ERROR, f"« {relpath} » : conversion .mpr impossible — {exc}"))
                continue
        fid = f"C{channel_id}/{relpath}#{hashlib.sha256(data).hexdigest()[:8]}"
        detected = detect_kind(data)
        res.raw_files[fid] = {"name": relpath, "content": data, "kind": detected}
        parsed.append((fid, relpath, detected, p))

    # Numérotation des électrodes : 1 et 2 dans un canal.
    numbers = sorted({p.electrode for *_x, p in parsed})
    remap = {n: n for n in numbers}
    if len(numbers) > 2:
        res.issues.append(Issue(LEVEL_ERROR, f"Canal {channel_id} : {len(numbers)} électrodes dans le dossier "
                                             f"({', '.join(f'n°{n}' for n in numbers)}) — un canal n'en a que 2. "
                                             "Corrigez l'électrode dans le tableau ou importez un dossier par canal."))
    elif numbers and not set(numbers) <= {1, 2}:
        remap = {n: i for i, n in enumerate(numbers, start=1)}
        res.issues.append(Issue(LEVEL_WARNING, f"Canal {channel_id} : électrodes "
                                               + ", ".join(f"n°{n} → électrode {remap[n]}" for n in numbers)
                                               + " (renumérotées dans le canal)."))

    # Réplicats : numéro du nom (01, 06… — numéro de séquence EC-Lab) → 1, 2, 3… dans l'ordre, par
    # (technique, électrode, étape). Le numéro d'origine reste dans le nom du fichier.
    order: dict = {}
    for *_x, p in parsed:
        order.setdefault((p.kind, p.electrode, p.dataset, p.concentration), []).append(p.replicate)
    renum = {k: {r: i for i, r in enumerate(sorted(set(v)), start=1)} for k, v in order.items()}

    found = [p.concentration for *_x, p in parsed if p.concentration is not None]
    res.concentrations_found = _unique_sorted(found)
    missing: dict = {}
    for fid, relpath, detected, p in parsed:
        for w in p.warnings:
            res.issues.append(Issue(LEVEL_WARNING, f"« {relpath} » : {w}", fid))
        step = STEP_PROBE if p.dataset == "probe" else STEP_BARE if p.dataset == "bare" else None
        if p.dataset == "calibration":
            i = match_concentration(p.concentration, ch.concentrations if ch else [])
            if i is None:
                missing.setdefault(p.concentration, []).append(relpath)
            else:
                step = conc_step(i)
        res.rows.append(AssignmentRow(file_id=fid, filename=relpath, kind=p.kind, channel=channel_id,
                                      electrode=remap.get(p.electrode, p.electrode), step=step,
                                      replicate=renum[(p.kind, p.electrode, p.dataset, p.concentration)][p.replicate],
                                      detected_kind=detected,
                                      file_concentration=p.concentration))
    for c, paths in sorted(missing.items()):
        res.issues.append(Issue(LEVEL_ERROR, f"Canal {channel_id} : concentration {c:.2e} M absente de la "
                                             f"configuration du canal ({len(paths)} fichier(s), ex. « {paths[0]} ») — "
                                             "adoptez les concentrations lues dans les noms ou corrigez la configuration."))
    return res


def reassign_steps(rows: list, experiment: Experiment) -> list:
    """Recalcule l'étape ``conc_i`` des lignes portant une concentration lue dans le nom
    (après une modification des concentrations du canal). Les autres lignes sont inchangées."""
    out = []
    for r in rows:
        if r.file_concentration is not None and experiment.channel(r.channel) is not None:
            i = match_concentration(r.file_concentration, experiment.channel(r.channel).concentrations)
            r.step = conc_step(i) if i is not None else None
        out.append(r)
    return out
