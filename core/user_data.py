"""Liste persistante des biomarqueurs (``user_data/biomarkers.json``) — aucun import Streamlit.

Le fichier vit HORS du dossier de l'application (``core.config.user_data_dir``) : launch.bat
remplace ``eis_app/`` à chaque mise à jour, les noms saisis doivent y survivre.

Format : ``{"version": 1, "biomarkers": ["miR-21", "ARN-X", ...]}``. Un fichier absent ou
illisible donne une liste vide (jamais d'exception à la lecture : l'utilisateur peut toujours
saisir un nouveau nom) ; l'écriture est atomique (fichier temporaire puis remplacement).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from core.config import user_data_dir
from core.logger import get_logger

log = get_logger("user_data")

BIOMARKERS_FILENAME = "biomarkers.json"


def biomarkers_path(directory: Optional[Path] = None) -> Path:
    return Path(directory or user_data_dir()) / BIOMARKERS_FILENAME


def _clean(names: Iterable) -> list:
    """Noms nettoyés (espaces de bord retirés), non vides, sans doublon (insensible à la casse),
    dans l'ordre de première apparition."""
    out, seen = [], set()
    for n in names:
        n = str(n).strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def load_biomarkers(directory: Optional[Path] = None) -> list:
    """Biomarqueurs enregistrés ; [] si le fichier n'existe pas ou est illisible."""
    path = biomarkers_path(directory)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning(f"{path} illisible ({exc}) : liste de biomarqueurs ignorée.")
        return []
    names = raw.get("biomarkers") if isinstance(raw, dict) else raw
    return _clean(names) if isinstance(names, list) else []


def save_biomarkers(names: Iterable, directory: Optional[Path] = None) -> list:
    """Écrit la liste (nettoyée) de façon atomique et la retourne."""
    names = _clean(names)
    path = biomarkers_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".biomarkers_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "biomarkers": names}, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return names


def add_biomarkers(new_names: Iterable, directory: Optional[Path] = None) -> list:
    """Ajoute des noms à la liste persistante (sans doublon) et retourne la liste complète.

    N'écrit sur disque que si la liste change.
    """
    current = load_biomarkers(directory)
    merged = _clean(list(current) + list(new_names))
    if merged != current:
        try:
            save_biomarkers(merged, directory)
        except OSError as exc:      # dossier en lecture seule… : la saisie reste utilisable
            log.warning(f"Impossible d'enregistrer les biomarqueurs ({exc}).")
    return merged
