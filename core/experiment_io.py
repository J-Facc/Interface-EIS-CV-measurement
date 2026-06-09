"""core/experiment_io.py
======================
Sauvegarde et chargement d'une session expérimentale complète (ZIP).

Format ZIP :
  experiment.yaml   — métadonnées (name, date, mode, concentrations, n_electrodes, n_replicats)
  probe/electrode_1/rep_0.bin  — bytes bruts des fichiers probe
  probe/electrode_2/rep_0.bin
  calibration/eis/electrode_1/conc_0_rep_0.bin
  calibration/cv/electrode_1/conc_0_rep_0.bin
  validation/eis/electrode_1/conc_0_rep_0.bin   (optionnel)
  validation/cv/electrode_1/conc_0_rep_0.bin    (optionnel)

Aucun import Streamlit — logique métier pure.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────

def save_experiment(experiment: Dict[str, Any]) -> bytes:
    """
    Sérialise un dict experiment dans un ZIP en mémoire.

    Parameters
    ----------
    experiment : dict conforme à la structure définie dans pages/0_import.py

    Returns
    -------
    bytes — contenu du fichier ZIP
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

        # ── métadonnées YAML ──────────────────────────────────────────────
        meta = {
            "name":         experiment.get("name", ""),
            "date":         experiment.get("date", datetime.now().strftime("%Y-%m-%d")),
            "mode":         experiment.get("mode", "both"),
            "concentrations": [float(c) for c in experiment.get("concentrations", [])],
            "n_electrodes": int(experiment.get("n_electrodes", 2)),
            "n_replicats":  int(experiment.get("n_replicats", 3)),
        }
        zf.writestr("experiment.yaml", yaml.dump(meta, allow_unicode=True))

        # ── probe ─────────────────────────────────────────────────────────
        probe = experiment.get("probe") or {}
        for elec_key, rep_list in probe.items():
            if not rep_list:
                continue
            for ri, bio in enumerate(rep_list):
                if bio is None:
                    continue
                data = _read_bytesio(bio)
                if data:
                    zf.writestr(f"probe/{elec_key}/rep_{ri}.bin", data)

        # ── calibration ───────────────────────────────────────────────────
        calibration = experiment.get("calibration") or {}
        for sig_type, elec_dict in calibration.items():
            if not elec_dict:
                continue
            for elec_key, conc_list in elec_dict.items():
                if not conc_list:
                    continue
                for ci, rep_list in enumerate(conc_list):
                    if not rep_list:
                        continue
                    for ri, bio in enumerate(rep_list):
                        if bio is None:
                            continue
                        data = _read_bytesio(bio)
                        if data:
                            zf.writestr(
                                f"calibration/{sig_type}/{elec_key}/conc_{ci}_rep_{ri}.bin",
                                data,
                            )

        # ── validation (optionnel) ────────────────────────────────────────
        validation = experiment.get("validation") or {}
        for sig_type, elec_dict in validation.items():
            if not elec_dict:
                continue
            for elec_key, conc_list in elec_dict.items():
                if not conc_list:
                    continue
                for ci, rep_list in enumerate(conc_list):
                    if not rep_list:
                        continue
                    for ri, bio in enumerate(rep_list):
                        if bio is None:
                            continue
                        data = _read_bytesio(bio)
                        if data:
                            zf.writestr(
                                f"validation/{sig_type}/{elec_key}/conc_{ci}_rep_{ri}.bin",
                                data,
                            )

    return buf.getvalue()


# alias pour clarté dans les pages
zip_experiment = save_experiment


# ─────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────

def load_experiment(zip_bytes: bytes) -> Dict[str, Any]:
    """
    Désérialise un ZIP (produit par save_experiment) en dict experiment.

    Tous les fichiers binaires sont renvoyés comme io.BytesIO,
    identiques à ce que Streamlit renvoie via st.file_uploader.

    Returns
    -------
    dict avec la même structure que st.session_state['experiment']

    Raises
    ------
    ValueError si le ZIP ne contient pas experiment.yaml
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        names = set(zf.namelist())

        if "experiment.yaml" not in names:
            raise ValueError("ZIP invalide : 'experiment.yaml' introuvable.")

        meta = yaml.safe_load(zf.read("experiment.yaml").decode("utf-8"))

        n_elec  = int(meta.get("n_electrodes", 2))
        n_concs = len(meta.get("concentrations", []))
        mode    = meta.get("mode", "both")

        # ── probe ─────────────────────────────────────────────────────────
        probe: Dict[str, List[Optional[io.BytesIO]]] = {}
        for e in range(1, n_elec + 1):
            elec_key = f"electrode_{e}"
            reps = _load_rep_list(zf, names, f"probe/{elec_key}")
            probe[elec_key] = reps

        # ── calibration ───────────────────────────────────────────────────
        sig_types = _sig_types_for_mode(mode)
        calibration: Dict[str, Any] = {st_: None for st_ in ["eis", "cv"]}
        for sig_type in sig_types:
            elec_dict: Dict[str, List] = {}
            for e in range(1, n_elec + 1):
                elec_key = f"electrode_{e}"
                conc_list = []
                for ci in range(n_concs):
                    reps = _load_rep_list(
                        zf, names,
                        f"calibration/{sig_type}/{elec_key}/conc_{ci}",
                    )
                    conc_list.append(reps)
                elec_dict[elec_key] = conc_list
            calibration[sig_type] = elec_dict

        # ── validation (optionnel) ────────────────────────────────────────
        has_validation = any(
            n.startswith("validation/") for n in names
        )
        validation: Optional[Dict] = None
        if has_validation:
            validation = {}
            for sig_type in sig_types:
                elec_dict = {}
                for e in range(1, n_elec + 1):
                    elec_key = f"electrode_{e}"
                    conc_list = []
                    ci = 0
                    while True:
                        prefix = f"validation/{sig_type}/{elec_key}/conc_{ci}"
                        reps = _load_rep_list(zf, names, prefix)
                        if not reps:
                            break
                        conc_list.append(reps)
                        ci += 1
                    elec_dict[elec_key] = conc_list
                validation[sig_type] = elec_dict

        return {
            "name":         meta.get("name", ""),
            "date":         meta.get("date", ""),
            "mode":         mode,
            "concentrations": meta.get("concentrations", []),
            "n_electrodes": n_elec,
            "n_replicats":  int(meta.get("n_replicats", 3)),
            "probe":        probe,
            "calibration":  calibration,
            "validation":   validation,
        }


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


def _load_rep_list(
    zf: zipfile.ZipFile,
    names: set,
    prefix: str,
) -> List[io.BytesIO]:
    """Charge tous les réplicats rep_0.bin, rep_1.bin, … pour un préfixe donné."""
    reps: List[io.BytesIO] = []
    ri = 0
    while True:
        path = f"{prefix}/rep_{ri}.bin"
        if path not in names:
            break
        data = zf.read(path)
        bio = io.BytesIO(data)
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
