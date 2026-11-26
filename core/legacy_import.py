"""Lecture des archives v2 (``core/experiment_io.py``) dans le modèle v3 — aucun import Streamlit.

Une archive v2 ne contient que les fichiers BRUTS (et les exclusions du prétraitement), jamais
de résultats : elle est importée comme une configuration + des fichiers assignés, à nettoyer
puis à analyser.

Correspondance v2 → v3 :

* électrode v2 n° e → canal ⌈e/2⌉, électrode 1 (e impair) ou 2 (e pair) ;
* toutes les électrodes v2 partageaient les mêmes concentrations → chaque canal les reçoit ;
* biomarqueur : le nom de l'expérience v2 (ou « Biomarqueur v2 ») ;
* « probe » → ``probe`` ; i-ème concentration → ``conc_i`` ; réplicat k (0-based) → k + 1 ;
* réplicats exclus au prétraitement v2 → lignes « exclure » cochées ; points EIS supprimés →
  points exclus (mêmes indices : même loader, même configuration).
* le jeu de « validation » v2 n'a pas d'équivalent v3 : il est ignoré, et signalé.
"""

from __future__ import annotations

import hashlib
import math

from core.assignment import AssignmentRow, detect_kind
from core.experiment import Channel, Experiment, conc_step
from core.experiment_io import load_experiment


def _fid(name: str, data: bytes) -> str:
    return f"{name}#{hashlib.sha256(data).hexdigest()[:8]}"


def import_v2(zip_bytes: bytes) -> dict:
    """``{"experiment", "raw_files", "rows", "excluded_points", "warnings"}`` d'une archive v2.

    Raises:
        ValueError: archive v2 illisible (message de ``core.experiment_io.load_experiment``).
    """
    exp2 = load_experiment(zip_bytes)
    n_elec = int(exp2.get("n_electrodes") or 1)
    concs = [float(c) for c in exp2.get("concentrations") or []]
    bio = (exp2.get("name") or "").strip() or "Biomarqueur v2"
    n_ch = math.ceil(n_elec / 2)
    exp = Experiment(biomarkers=[bio], channels=[Channel(id=k + 1, biomarker=bio, concentrations=list(concs))
                                                 for k in range(n_ch)], name=exp2.get("name") or "")
    excl = exp2.get("exclusions") or {}
    deleted = exp2.get("deleted_points") or {}
    raw, rows, points, warnings = {}, [], {}, []

    def _add(bio_file, kind_hint, e, ci, ri):
        if bio_file is None:
            return
        data = bio_file.getvalue() if hasattr(bio_file, "getvalue") else bytes(bio_file)
        name = getattr(bio_file, "name", None) or f"e{e}_{kind_hint}_{ci}_r{ri + 1}.txt"
        fid = _fid(name, data)
        kind = detect_kind(data) or kind_hint.upper()
        raw[fid] = {"name": name, "content": data, "kind": kind}
        step = "probe" if ci == "probe" else conc_step(int(ci) + 1)
        flags = ((excl.get(f"e{e}") or {}).get(kind_hint) or {}).get(ci) or []
        rows.append(AssignmentRow(file_id=fid, filename=name, kind=kind, channel=(e + 1) // 2,
                                  electrode=1 if e % 2 else 2, step=step, replicate=ri + 1,
                                  exclude=bool(ri < len(flags) and flags[ri]), detected_kind=detect_kind(data)))
        ci_str = "probe" if ci == "probe" else str(ci)
        pts = deleted.get(f"e{e}_{kind_hint}_c{ci_str}_r{ri}")
        if pts and kind_hint == "eis":
            points[fid] = sorted(int(i) for i in pts)

    for kind_hint in ("eis", "cv"):
        probe = ((exp2.get("probe") or {}).get(kind_hint) or {})
        cal = ((exp2.get("calibration") or {}).get(kind_hint) or {})
        for e in range(1, n_elec + 1):
            for ri, b in enumerate(probe.get(f"electrode_{e}") or []):
                _add(b, kind_hint, e, "probe", ri)
            for ci, reps in enumerate(cal.get(f"electrode_{e}") or []):
                for ri, b in enumerate(reps or []):
                    _add(b, kind_hint, e, ci, ri)
    if exp2.get("validation"):
        warnings.append("Le jeu « validation » de l'archive v2 n'a pas d'équivalent en v3 : il est ignoré.")
    if n_elec % 2:
        warnings.append(f"{n_elec} électrode(s) v2 : le dernier canal n'a qu'une électrode.")
    warnings.append(f"Archive v2 importée : {n_elec} électrode(s) → {n_ch} canal(aux) ; biomarqueur « {bio} ». "
                    "Vérifiez la configuration, puis nettoyez et lancez l'analyse.")
    return {"experiment": exp, "raw_files": raw, "rows": rows, "excluded_points": points, "warnings": warnings}
