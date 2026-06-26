"""core/experiment_io.py
======================
Sauvegarde et chargement d'une session expérimentale complète (ZIP).

Format ZIP v2 :
  experiment.yaml   — métadonnées + exclusions + file_map
  probe/eis/electrode_1/probe_eis_e1_r1.txt
  probe/cv/electrode_1/probe_cv_e1_r1.txt
  calibration/eis/electrode_1/eis_e1_1.00e-08M_r1.txt
  calibration/cv/electrode_1/cv_e1_1.00e-08M_r1.txt
  validation/eis/electrode_1/eis_e1_val_1.00e-08M_r1.txt  (optionnel)
  validation/cv/electrode_1/cv_e1_val_1.00e-08M_r1.txt   (optionnel)

Aucun import Streamlit — logique métier pure.
"""

from __future__ import annotations

import copy
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from core.logger import get_logger
from core.mpr_converter import is_mpr, mpr_to_csv_bytes

log = get_logger("experiment_io")


# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────

def save_experiment(
    experiment: Dict[str, Any],
    exclusions: Optional[Dict] = None,
    deleted_points: Optional[Dict] = None,
    preprocessing_done: bool = False,
) -> bytes:
    """
    Sérialise un dict experiment dans un ZIP en mémoire.

    Parameters
    ----------
    experiment : dict conforme à la structure définie dans pages/0_import.py
        probe       : {sig_type: {electrode_k: [BytesIO, ...]}}
        calibration : {sig_type: {electrode_k: [[BytesIO, ...], ...]}}
        validation  : {sig_type: {electrode_k: [[BytesIO, ...], ...]}} ou None
    exclusions : dict 2D {e_str: {modality: {ci: [bool]}}}
    deleted_points : dict {inner_label: [int]} — points exclus par spectre EIS
    preprocessing_done : bool

    Returns
    -------
    bytes — contenu du fichier ZIP
    """
    mode           = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations", [])
    n_elec         = int(experiment.get("n_electrodes", 2))
    sig_types      = _sig_types_for_mode(mode)

    file_map: Dict[str, Any] = {
        "probe": {},
        "calibration": {s: {} for s in ["eis", "cv"]},
        "validation":  {s: {} for s in ["eis", "cv"]},
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

        # ── probe : {sig_type: {electrode_k: [BytesIO, ...]}} ────────────
        probe = experiment.get("probe") or {}
        for sig_type in sig_types:
            elec_dict = probe.get(sig_type) or {}
            file_map["probe"][sig_type] = {}
            for e in range(1, n_elec + 1):
                elec_key = f"electrode_{e}"
                e_label  = f"e{e}"
                rep_list = elec_dict.get(elec_key) or []
                paths: List[str] = []
                for ri, bio in enumerate(rep_list):
                    if bio is None:
                        continue
                    data = _read_bytesio(bio)
                    if data:
                        fname    = f"probe_{sig_type}_{e_label}_r{ri + 1}.txt"
                        zip_path = f"probe/{sig_type}/{elec_key}/{fname}"
                        zf.writestr(zip_path, data)
                        paths.append(zip_path)
                file_map["probe"][sig_type][elec_key] = paths

        # ── calibration : {sig_type: {electrode_k: [[BytesIO,...], ...]}} ─
        calibration = experiment.get("calibration") or {}
        for sig_type in sig_types:
            elec_dict = calibration.get(sig_type) or {}
            file_map["calibration"][sig_type] = {}
            for e in range(1, n_elec + 1):
                elec_key  = f"electrode_{e}"
                e_label   = f"e{e}"
                conc_list = elec_dict.get(elec_key) or []
                elec_paths: List[List[str]] = []
                for ci, rep_list in enumerate(conc_list):
                    conc_val = concentrations[ci] if ci < len(concentrations) else 0.0
                    conc_str = f"{conc_val:.2e}M"
                    rep_paths: List[str] = []
                    for ri, bio in enumerate(rep_list or []):
                        if bio is None:
                            continue
                        data = _read_bytesio(bio)
                        if data:
                            fname    = f"{sig_type}_{e_label}_{conc_str}_r{ri + 1}.txt"
                            zip_path = f"calibration/{sig_type}/{elec_key}/{fname}"
                            zf.writestr(zip_path, data)
                            rep_paths.append(zip_path)
                    elec_paths.append(rep_paths)
                file_map["calibration"][sig_type][elec_key] = elec_paths

        # ── validation (optionnel) ────────────────────────────────────────
        validation = experiment.get("validation") or {}
        if validation:
            for sig_type in sig_types:
                elec_dict = (validation.get(sig_type) or {})
                file_map["validation"][sig_type] = {}
                for e in range(1, n_elec + 1):
                    elec_key  = f"electrode_{e}"
                    e_label   = f"e{e}"
                    conc_list = elec_dict.get(elec_key) or []
                    elec_paths_v: List[List[str]] = []
                    for ci, rep_list in enumerate(conc_list):
                        conc_val = concentrations[ci] if ci < len(concentrations) else 0.0
                        conc_str = f"{conc_val:.2e}M"
                        rep_paths_v: List[str] = []
                        for ri, bio in enumerate(rep_list or []):
                            if bio is None:
                                continue
                            data = _read_bytesio(bio)
                            if data:
                                fname    = f"{sig_type}_{e_label}_val_{conc_str}_r{ri + 1}.txt"
                                zip_path = f"validation/{sig_type}/{elec_key}/{fname}"
                                zf.writestr(zip_path, data)
                                rep_paths_v.append(zip_path)
                        elec_paths_v.append(rep_paths_v)
                    file_map["validation"][sig_type][elec_key] = elec_paths_v

        # ── métadonnées YAML ──────────────────────────────────────────────
        meta: Dict[str, Any] = {
            "name":               experiment.get("name", ""),
            "date":               experiment.get("date", datetime.now().strftime("%Y-%m-%d")),
            "mode":               mode,
            "concentrations":     [float(c) for c in concentrations],
            "n_electrodes":       n_elec,
            "n_replicats":        int(experiment.get("n_replicats", 3)),
            "preprocessing_done": bool(preprocessing_done),
            "file_map":           file_map,
        }

        # Exclusions
        if exclusions:
            excl_serial: Dict[str, Any] = {}
            for e_str, mod_dict in exclusions.items():
                if not mod_dict:
                    continue
                excl_serial[e_str] = {}
                for modality, ci_dict in mod_dict.items():
                    if not ci_dict:
                        continue
                    excl_serial[e_str][modality] = {}
                    for ci, excl_list in ci_dict.items():
                        ci_key = "probe" if ci == "probe" else str(ci)
                        excl_serial[e_str][modality][ci_key] = [bool(x) for x in excl_list]
            if excl_serial:
                meta["exclusions"] = excl_serial

        if deleted_points:
            meta["deleted_points"] = {
                k: sorted(int(i) for i in v)
                for k, v in deleted_points.items()
                if v
            }

        zf.writestr("experiment.yaml", yaml.dump(meta, allow_unicode=True))

    return buf.getvalue()


# alias pour clarté dans les pages
zip_experiment = save_experiment


# ─────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────

def load_experiment(zip_bytes: bytes) -> Dict[str, Any]:
    """
    Désérialise un ZIP (produit par save_experiment) en dict experiment.

    Supporte le format v2 (file_map dans experiment.yaml) et le format v1
    (rep_N.bin) pour la rétrocompatibilité.

    Returns
    -------
    dict avec les clés :
      name, date, mode, concentrations, n_electrodes, n_replicats,
      probe, calibration, validation,
      exclusions, deleted_points, preprocessing_done

    Raises
    ------
    ValueError si le ZIP ne contient pas experiment.yaml
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        names = set(zf.namelist())

        if "experiment.yaml" not in names:
            raise ValueError("ZIP invalide : 'experiment.yaml' introuvable.")

        meta = yaml.safe_load(zf.read("experiment.yaml").decode("utf-8"))

        n_elec     = int(meta.get("n_electrodes", 2))
        n_concs    = len(meta.get("concentrations", []))
        mode       = meta.get("mode", "both")
        sig_types  = _sig_types_for_mode(mode)
        file_map   = meta.get("file_map")

        def _read_bio(path: str) -> Optional[io.BytesIO]:
            """Lit un fichier du ZIP et retourne un BytesIO positionné en 0."""
            try:
                data = zf.read(path)
                name = path.split("/")[-1]
                if is_mpr(name):
                    try:
                        data = mpr_to_csv_bytes(data)
                        name = name[:-4] + ".csv"
                    except Exception as exc:
                        log.warning(f"Conversion .mpr échouée pour {path} : {exc}")
                bio  = io.BytesIO(data)
                bio.seek(0)
                bio.name = name
                return bio
            except KeyError:
                return None

        if file_map:
            # ── Format v2 : reconstruction via file_map ───────────────────
            probe: Dict[str, Any] = {}
            for sig_type in sig_types:
                probe[sig_type] = {}
                for e in range(1, n_elec + 1):
                    elec_key = f"electrode_{e}"
                    paths    = (file_map.get("probe", {})
                                       .get(sig_type, {})
                                       .get(elec_key, []))
                    probe[sig_type][elec_key] = [
                        b for p in paths if (b := _read_bio(p)) is not None
                    ]

            calibration: Dict[str, Any] = {st_: None for st_ in ["eis", "cv"]}
            for sig_type in sig_types:
                elec_dict: Dict[str, List] = {}
                for e in range(1, n_elec + 1):
                    elec_key  = f"electrode_{e}"
                    conc_list = (file_map.get("calibration", {})
                                         .get(sig_type, {})
                                         .get(elec_key, []))
                    elec_dict[elec_key] = [
                        [b for p in rep_paths if (b := _read_bio(p)) is not None]
                        for rep_paths in conc_list
                    ]
                calibration[sig_type] = elec_dict

            has_validation = bool(
                file_map.get("validation") and any(
                    file_map["validation"].get(s)
                    for s in sig_types
                )
            )
            validation: Optional[Dict] = None
            if has_validation:
                validation = {}
                for sig_type in sig_types:
                    elec_dict_v: Dict[str, List] = {}
                    for e in range(1, n_elec + 1):
                        elec_key  = f"electrode_{e}"
                        conc_list = (file_map.get("validation", {})
                                             .get(sig_type, {})
                                             .get(elec_key, []))
                        elec_dict_v[elec_key] = [
                            [b for p in rep_paths if (b := _read_bio(p)) is not None]
                            for rep_paths in conc_list
                        ]
                    validation[sig_type] = elec_dict_v

        else:
            # ── Format v1 : reconstruction via chemins rep_N.bin ─────────
            probe = {}
            for sig_type in sig_types:
                probe[sig_type] = {}
                for e in range(1, n_elec + 1):
                    elec_key = f"electrode_{e}"
                    reps = _load_rep_list_v1(zf, names, f"probe/{sig_type}/{elec_key}")
                    probe[sig_type][elec_key] = reps

            calibration = {st_: None for st_ in ["eis", "cv"]}
            for sig_type in sig_types:
                elec_dict = {}
                for e in range(1, n_elec + 1):
                    elec_key  = f"electrode_{e}"
                    conc_list = []
                    for ci in range(n_concs):
                        reps = _load_rep_list_v1(
                            zf, names,
                            f"calibration/{sig_type}/{elec_key}/conc_{ci}",
                        )
                        conc_list.append(reps)
                    elec_dict[elec_key] = conc_list
                calibration[sig_type] = elec_dict

            has_validation = any(n.startswith("validation/") for n in names)
            validation = None
            if has_validation:
                validation = {}
                for sig_type in sig_types:
                    elec_dict_v = {}
                    for e in range(1, n_elec + 1):
                        elec_key  = f"electrode_{e}"
                        conc_list = []
                        ci = 0
                        while True:
                            prefix = f"validation/{sig_type}/{elec_key}/conc_{ci}"
                            reps   = _load_rep_list_v1(zf, names, prefix)
                            if not reps:
                                break
                            conc_list.append(reps)
                            ci += 1
                        elec_dict_v[elec_key] = conc_list
                    validation[sig_type] = elec_dict_v

        # ── exclusions et points supprimés ────────────────────────────────
        raw_exclusions = meta.get("exclusions") or {}
        exclusions_out: Dict[str, Any] = {}
        if raw_exclusions:
            first_val = next(iter(raw_exclusions.values()), None)
            if isinstance(first_val, dict):
                for e_str, mod_dict in raw_exclusions.items():
                    exclusions_out[e_str] = {}
                    for modality, ci_dict in (mod_dict or {}).items():
                        exclusions_out[e_str][modality] = {}
                        for ci_key, excl_list in (ci_dict or {}).items():
                            ci = "probe" if ci_key == "probe" else int(ci_key)
                            exclusions_out[e_str][modality][ci] = list(excl_list)

        raw_deleted = meta.get("deleted_points") or {}
        deleted_points_out = {k: list(v) for k, v in raw_deleted.items()} if raw_deleted else {}

        return {
            "name":               meta.get("name", ""),
            "date":               meta.get("date", ""),
            "mode":               mode,
            "concentrations":     meta.get("concentrations", []),
            "n_electrodes":       n_elec,
            "n_replicats":        int(meta.get("n_replicats", 3)),
            "probe":              probe,
            "calibration":        calibration,
            "validation":         validation,
            "exclusions":         exclusions_out,
            "deleted_points":     deleted_points_out,
            "preprocessing_done": bool(meta.get("preprocessing_done", False)),
        }


# ─────────────────────────────────────────────
# Application des exclusions (utilitaire partagé)
# ─────────────────────────────────────────────

def apply_exclusions(experiment: Dict[str, Any], exclusions: Dict) -> Dict[str, Any]:
    """
    Applique les exclusions 2D sur un dict experiment brut (BytesIO lists).

    Parameters
    ----------
    experiment : dict experiment avec probe/calibration contenant des BytesIO
    exclusions : dict 2D {e_str: {modality: {ci: [bool]}}}

    Returns
    -------
    Copie profonde de experiment avec les réplicats exclus retirés.
    """
    exp_clean = copy.deepcopy(experiment)
    mode      = exp_clean.get("mode", "both")
    n_elec    = exp_clean.get("n_electrodes", 2)

    def _excl(e_str: str, modality: str, ci) -> list:
        return exclusions.get(e_str, {}).get(modality, {}).get(ci, [])

    for e_idx in range(1, n_elec + 1):
        e_str    = f"e{e_idx}"
        elec_key = f"electrode_{e_idx}"

        if mode in ("eis_only", "both"):
            probe_eis = ((exp_clean.get("probe") or {}).get("eis") or {})
            reps = probe_eis.get(elec_key) or []
            excl = _excl(e_str, "eis", "probe")
            probe_eis[elec_key] = [r for ri, r in enumerate(reps)
                                   if not (ri < len(excl) and excl[ri])]

            cal_eis = ((exp_clean.get("calibration") or {}).get("eis") or {})
            for ci, rep_list in enumerate(cal_eis.get(elec_key) or []):
                excl = _excl(e_str, "eis", ci)
                cal_eis[elec_key][ci] = [r for ri, r in enumerate(rep_list or [])
                                         if not (ri < len(excl) and excl[ri])]

        if mode in ("cv_only", "both"):
            probe_cv = ((exp_clean.get("probe") or {}).get("cv") or {})
            reps = probe_cv.get(elec_key) or []
            excl = _excl(e_str, "cv", "probe")
            probe_cv[elec_key] = [r for ri, r in enumerate(reps)
                                  if not (ri < len(excl) and excl[ri])]

            cal_cv = ((exp_clean.get("calibration") or {}).get("cv") or {})
            for ci, rep_list in enumerate(cal_cv.get(elec_key) or []):
                excl = _excl(e_str, "cv", ci)
                cal_cv[elec_key][ci] = [r for ri, r in enumerate(rep_list or [])
                                        if not (ri < len(excl) and excl[ri])]

    return exp_clean


def apply_point_exclusions(
    experiment: Dict[str, Any],
    point_exclusions: Dict[str, list],
    config: dict = None,
) -> Dict[str, Any]:
    """
    Retire les points individuels marqués comme supprimés (éditeur de points
    EIS) de chaque réplicat concerné, en ré-écrivant son contenu CSV/BytesIO
    sans les lignes correspondant aux indices supprimés.

    Les indices de point_exclusions correspondent à la position dans le
    tableau f/Zre/Zim APRÈS le nettoyage effectué par core.loader.load_spectrum
    (tri HF→BF, suppression fréquences parasites) — donc chaque réplicat
    concerné est relu via load_spectrum avant de retirer les points, puis
    ré-sérialisé en CSV propre.

    Parameters
    ----------
    experiment : dict experiment (probe/calibration avec BytesIO), typiquement
        déjà passé par apply_exclusions.
    point_exclusions : {label: [indices supprimés]}, label au format
        "e{elec}_eis_c{ci|probe}_r{ri}" (cf. pages/1_pretraitement.py::_dp_key,
        sans le préfixe "deleted_points_").
    config : config app (passé à load_spectrum pour cohérence du nettoyage
        parasite).

    Returns
    -------
    Copie profonde de experiment avec les BytesIO concernés ré-écrits sans
    les points supprimés. Les réplicats non concernés restent inchangés.
    """
    from core.loader import load_spectrum

    exp_clean = copy.deepcopy(experiment)
    mode = exp_clean.get("mode", "both")
    if mode not in ("eis_only", "both") or not point_exclusions:
        return exp_clean

    n_elec = exp_clean.get("n_electrodes", 2)

    def _rewrite_bio(bio, deleted_idx: list):
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        sp = load_spectrum(content, label="tmp", config=config)
        keep = [i for i in range(len(sp.f)) if i not in set(deleted_idx)]
        lines = ["frequency_Hz,Zreal_Ohm,Zimag_Ohm"]
        for i in keep:
            # Même convention de signe que sp.Zim (positive) pour éviter un
            # double retournement de signe à la relecture par load_spectrum.
            lines.append(f"{sp.f[i]},{sp.Zre[i]},{sp.Zim[i]}")
        new_bytes = ("\n".join(lines)).encode("utf-8")
        new_bio = io.BytesIO(new_bytes)
        new_bio.name = getattr(bio, "name", "edited.csv")
        return new_bio

    for e_idx in range(1, n_elec + 1):
        e_str = f"e{e_idx}"
        elec_key = f"electrode_{e_idx}"

        probe_eis = ((exp_clean.get("probe") or {}).get("eis") or {})
        reps = probe_eis.get(elec_key) or []
        for ri, bio in enumerate(reps):
            label = f"{e_str}_eis_cprobe_r{ri}"
            if point_exclusions.get(label):
                reps[ri] = _rewrite_bio(bio, point_exclusions[label])

        cal_eis = ((exp_clean.get("calibration") or {}).get("eis") or {})
        for ci, rep_list in enumerate(cal_eis.get(elec_key) or []):
            for ri, bio in enumerate(rep_list or []):
                label = f"{e_str}_eis_c{ci}_r{ri}"
                if point_exclusions.get(label):
                    rep_list[ri] = _rewrite_bio(bio, point_exclusions[label])

    return exp_clean


# ─────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────

def _read_bytesio(obj: Any) -> bytes:
    """Lit un io.BytesIO ou un UploadedFile Streamlit et le remet en position 0."""
    if obj is None:
        return b""
    try:
        obj.seek(0)
        data = obj.read()
        obj.seek(0)
        return data
    except Exception:
        return b""


def _load_rep_list_v1(
    zf: zipfile.ZipFile,
    names: set,
    prefix: str,
) -> List[io.BytesIO]:
    """Format v1 : charge rep_0.bin, rep_1.bin, … pour un préfixe donné."""
    reps: List[io.BytesIO] = []
    ri = 0
    while True:
        path = f"{prefix}/rep_{ri}.bin"
        if path not in names:
            break
        data = zf.read(path)
        bio  = io.BytesIO(data)
        bio.seek(0)
        bio.name = f"rep_{ri}.bin"
        reps.append(bio)
        ri += 1
    return reps


def _sig_types_for_mode(mode: str) -> List[str]:
    if mode == "eis_only":
        return ["eis"]
    if mode == "cv_only":
        return ["cv"]
    return ["eis", "cv"]
