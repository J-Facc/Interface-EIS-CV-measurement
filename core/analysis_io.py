"""core/analysis_io.py
====================
Sauvegarde et rechargement d'une ANALYSE EIS complète (sessions + DRT calculées).

Complément de ``core/experiment_io.py`` (qui sauvegarde les fichiers de mesure bruts) :
ici on range le RÉSULTAT — sessions par électrode (spectres, verdicts KK, fits Orazem, DRT de
chaque réplicat et moyenne, diagnostics HMC), registre DRT, réglages — pour rouvrir
l'expérience plus tard sans refaire aucun calcul (la DRT HMC prend plusieurs minutes par spectre).

Format : un ZIP contenant
  manifest.json  — format, version, date, résumé lisible (électrodes, nombre de DRT)
  analysis.pkl   — ``dict`` des clés de ``st.session_state`` listées dans ``ANALYSIS_KEYS``

Sécurité : un pickle exécute du code à la lecture. Le fichier est donc relu par un
``Unpickler`` RESTREINT (``_RestrictedUnpickler``) qui n'accepte que les classes des modules
du projet, quelques types numpy/stdlib sans effet de bord ; tout autre objet lève
``AnalysisFileError``. Ne chargez malgré tout que des fichiers que vous avez produits.

Aucun import Streamlit — logique métier pure.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from datetime import datetime
from typing import Any, Dict

from core.pipeline import DRT_MODEL_NAME

FORMAT_NAME = "eis-analysis"
FORMAT_VERSION = 1

#: Clés de ``st.session_state`` qui composent une analyse sauvegardée.
ANALYSIS_KEYS = (
    "eis_sessions", "eis_validations", "eis_normalized", "eis_drt_store",
    "eis_config", "eis_inputs", "eis_failures",
)

_MANIFEST = "manifest.json"
_PAYLOAD = "analysis.pkl"
_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024     # garde-fou contre une « bombe » ZIP

#: Préfixes des modules du projet dont les CLASSES peuvent être reconstruites.
_PROJECT_PREFIXES = ("core.", "fits.", "drt.", "circuit.")
#: (module, nom) numpy / stdlib strictement nécessaires à la reconstruction des données.
_SAFE_GLOBALS = {
    ("builtins", n) for n in ("set", "frozenset", "complex", "slice", "range", "bytearray")
} | {
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("datetime", "datetime"),
    ("datetime", "date"), ("datetime", "timedelta"), ("datetime", "timezone"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer"),
}


class AnalysisFileError(ValueError):
    """Fichier d'analyse illisible, d'un autre format ou contenant des objets non autorisés."""


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _SAFE_GLOBALS or module == "numpy.dtypes":
            return super().find_class(module, name)
        if module.startswith(_PROJECT_PREFIXES):
            obj = super().find_class(module, name)
            if isinstance(obj, type) and getattr(obj, "__module__", "").startswith(_PROJECT_PREFIXES):
                return obj
        raise AnalysisFileError(f"Objet non autorisé dans le fichier d'analyse : {module}.{name}")


def count_drt(sessions: dict) -> tuple:
    """(spectres portant une DRT, spectres analysables) sur toutes les électrodes."""
    from core.drt_recompute import iter_targets      # import local : évite un cycle de chargement
    targets = iter_targets(sessions or {})
    done = sum(1 for t in targets if DRT_MODEL_NAME in t.spectrum.fit_results)
    return done, len(targets)


def save_analysis(state: Dict[str, Any]) -> bytes:
    """Sérialise l'analyse EIS courante (``st.session_state`` ou tout mapping) en ZIP.

    Raises:
        AnalysisFileError: aucune analyse à sauvegarder.
    """
    sessions = state.get("eis_sessions")
    if not sessions:
        raise AnalysisFileError("Aucune analyse EIS à sauvegarder.")
    payload = {k: state.get(k) for k in ANALYSIS_KEYS}
    done, total = count_drt(sessions)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "electrodes": sorted(int(e) for e in sessions),
        "drt_spectra_computed": done,
        "drt_spectra_total": total,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr(_PAYLOAD, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return buf.getvalue()


def load_analysis(zip_bytes: bytes) -> Dict[str, Any]:
    """Relit un ZIP produit par :func:`save_analysis`.

    Returns:
        ``{"manifest": dict, **{clé de ANALYSIS_KEYS: valeur}}``.

    Raises:
        AnalysisFileError: ZIP invalide, autre format, version inconnue, objet non autorisé.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())
            if not {_MANIFEST, _PAYLOAD} <= names:
                raise AnalysisFileError(
                    "Ce fichier n'est pas une analyse EIS sauvegardée (pour recharger des "
                    "fichiers de mesure, utilisez la page Import).")
            manifest = json.loads(zf.read(_MANIFEST).decode("utf-8"))
            if manifest.get("format") != FORMAT_NAME:
                raise AnalysisFileError("Format de fichier inconnu.")
            if manifest.get("version") != FORMAT_VERSION:
                raise AnalysisFileError(
                    f"Version {manifest.get('version')!r} non prise en charge "
                    f"(attendu : {FORMAT_VERSION}). Refaites l'analyse puis sauvegardez-la.")
            if zf.getinfo(_PAYLOAD).file_size > _MAX_PAYLOAD_BYTES:
                raise AnalysisFileError("Fichier d'analyse anormalement volumineux.")
            raw = zf.read(_PAYLOAD)
    except zipfile.BadZipFile as exc:
        raise AnalysisFileError("Fichier ZIP invalide ou corrompu.") from exc
    try:
        payload = _RestrictedUnpickler(io.BytesIO(raw)).load()
    except AnalysisFileError:
        raise
    except Exception as exc:       # pickle corrompu / classe renommée depuis la sauvegarde
        raise AnalysisFileError(
            f"Analyse illisible (fichier corrompu, ou sauvegardé par une version incompatible "
            f"de l'application) : {exc!r}") from exc
    if not isinstance(payload, dict) or not payload.get("eis_sessions"):
        raise AnalysisFileError("Le fichier ne contient aucune analyse EIS.")
    return {"manifest": manifest, **{k: payload.get(k) for k in ANALYSIS_KEYS}}
