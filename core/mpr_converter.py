"""Conversion automatique des fichiers binaires .mpr (EC-Lab/BioLogic) en CSV texte."""

from __future__ import annotations

import io
import os
import tempfile

import pandas as pd

from core.logger import get_logger

log = get_logger("mpr_converter")


def is_mpr(filename: str) -> bool:
    """True si le nom de fichier se termine par .mpr (insensible à la casse)."""
    return filename.lower().endswith(".mpr")


def _read_mpr_dataframe(tmp_path: str) -> pd.DataFrame:
    """Lit un fichier .mpr sur disque et retourne un DataFrame brut.

    Essaie eclabfiles en priorité (gère plus de variantes de format),
    puis galvani en fallback si eclabfiles n'est pas disponible ou échoue.

    Args:
        tmp_path: Chemin vers le fichier .mpr temporaire sur disque.

    Returns:
        DataFrame avec les colonnes natives EC-Lab (freq, Re(Z), -Im(Z),
        Ewe, I/<I>, etc. selon la technique).

    Raises:
        RuntimeError: Si ni eclabfiles ni galvani ne parviennent à lire
            le fichier.
    """
    try:
        import eclabfiles as ecf
        return ecf.to_df(tmp_path)
    except Exception as exc_ecf:
        log.warning(f"eclabfiles a échoué ({exc_ecf}), tentative avec galvani")
        try:
            from galvani import BioLogic
        except ImportError:
            # galvani est une dépendance OPTIONNELLE (cf. requirements-optional.txt) :
            # son setup.py casse le build sur les runtimes non-Debian, on ne peut
            # pas l'imposer en dépendance dure. Message actionnable plutôt qu'une
            # ImportError brute pour l'utilisateur.
            raise RuntimeError(
                f"Lecture du fichier .mpr impossible : le lecteur principal "
                f"eclabfiles a échoué ({exc_ecf}) et le lecteur de secours "
                f"'galvani' n'est pas installé. Installez-le "
                f"(pip install galvani) ou ré-exportez le fichier en CSV/TXT "
                f"depuis EC-Lab."
            ) from exc_ecf
        try:
            mpr = BioLogic.MPRfile(tmp_path)
            return pd.DataFrame(mpr.data)
        except Exception as exc_galvani:
            raise RuntimeError(
                f"Impossible de lire le fichier .mpr avec eclabfiles "
                f"({exc_ecf}) ni avec galvani ({exc_galvani}). "
                f"Ré-exportez le fichier en CSV/TXT depuis EC-Lab."
            ) from exc_galvani


def mpr_to_csv_bytes(raw_bytes: bytes) -> bytes:
    """Convertit le contenu binaire d'un fichier .mpr EC-Lab en CSV texte
    (séparateur tabulation, toutes les colonnes natives conservées telles
    quelles — pas de renommage, core/loader.py et core/cv_loader.py
    reconnaissent déjà ces noms de colonnes via leurs alias existants).

    Pour les colonnes de courant en mA (I, <I>), convertit en A (divise
    par 1000) avant écriture, car core/cv_loader.py ne fait aucune
    conversion d'unité et traite la colonne reconnue comme étant déjà en
    ampères.

    Args:
        raw_bytes: Contenu binaire brut du fichier .mpr.

    Returns:
        Contenu CSV encodé en UTF-8 (bytes), séparateur tabulation.

    Raises:
        RuntimeError: Si la lecture du .mpr échoue avec les deux backends.
    """
    with tempfile.NamedTemporaryFile(suffix=".mpr", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        df = _read_mpr_dataframe(tmp_path)
    finally:
        os.unlink(tmp_path)

    for col in ("I", "<I>"):
        if col in df.columns:
            df[col] = df[col].astype(float) / 1000.0

    buf = io.StringIO()
    df.to_csv(buf, index=False, sep="\t")
    return buf.getvalue().encode("utf-8")
