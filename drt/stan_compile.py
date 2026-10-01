# -*- coding: utf-8 -*-
"""drt/stan_compile.py — compile un modèle Stan sans jamais exposer un chemin non-ASCII à ``make``.

Le défaut. Sous Windows, ``cmdstanpy`` lance ``mingw32-make``, qui lance à son tour le shell
MSYS (``/usr/bin/sh``) avec le chemin du ``.stan`` dans la ligne de commande. Si ce chemin
contient un caractère hors ASCII (« C:\\Users\\x\\Desktop\\.Thèse\\… »), le shell le reçoit
décalé d'un encodage (« è » devient « Ã¨ ») et échoue en « No such file or directory » alors que
le fichier existe. Le correctif ne peut pas être dans l'application : la corruption a lieu dans
des binaires tiers (make, MSYS).

Ce qui a été examiné dans **cmdstanpy 1.3.0** avant de copier à la main (et pourquoi rien de
plus simple n'a été retenu) :

* ``CmdStanModel`` n'a **aucun** paramètre d'encodage ni de répertoire de compilation :
  ``stanc_options`` et ``cpp_options`` ne parlent qu'à stanc et au compilateur C++, jamais au
  shell que make lance ; ``compile_stan_file`` appelle ``make`` avec ``cwd=cmdstan_path()`` et
  pour cible le chemin POSIX de l'exécutable, à côté du ``.stan``.
* ``cmdstanpy.utils.filesystem.SanitizedOrTmpFilePath`` fait déjà la même chose (copie du
  ``.stan`` dans un dossier temporaire, compilation, recopie de l'exécutable) — mais ne se
  déclenche que sur les **espaces** (et ``~`` hors Windows), jamais sur un caractère non-ASCII ;
  et son dossier temporaire vient de ``tempfile.mkdtemp()``, donc de ``%TEMP%``, qui contient le
  nom d'utilisateur Windows, accentué ou non.
* ``PYTHONUTF8=1`` ne change que l'encodage des fichiers et des flux de Python : la ligne de
  commande qui atteint make est construite par Python en UTF-16 (CreateProcessW) puis
  *reconvertie* par make dans la page de code ANSI du système, hors de portée de Python.
* ``chcp 65001`` ne change que la page de code de la CONSOLE ; make et MSYS lisent leur ligne
  de commande dans la page de code ANSI du système (``ACP``), que ``chcp`` ne touche pas.
  Seul le réglage Windows « Prise en charge mondiale des langues Unicode UTF-8 » la change, au
  niveau machine, avec redémarrage : une application ne peut pas l'imposer.

Aucune de ces pistes n'agit sur la cause ; ce n'est pas vérifiable ici (aucun Windows), donc elles
ne sont pas retenues comme « fiables ». Reste ce que fait ce module : **le chemin vu par make est
ASCII par construction**. Un ``.stan`` au chemin non-ASCII est copié dans un cache ASCII, compilé
là, et c'est l'exécutable du cache que l'application lance — il n'est pas recopié à côté du
``.stan`` d'origine, ce qui ferait sortir de nouveau du cache l'exécutable à chaque démarrage.

Le cache est valide si le SHA-256 du ``.stan`` ET la version de CmdStan notés dans son tampon
(``<modèle>.stamp.json``) sont ceux d'aujourd'hui : jamais le nom ou la date seuls. Le tampon est
écrit APRÈS une compilation réussie, donc un échec ne laisse jamais un exécutable présumé bon.

Ce module n'importe pas ``cmdstanpy`` au chargement (``setup_drt_bayesien.py`` en dépend avant
que l'extra DRT soit forcément installé) : il le résout à l'appel.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from drt.cmdstan_version import installed_version

#: Suffixe de l'exécutable compilé par CmdStan (même règle que ``setup_drt_bayesien._exe_path``).
EXE_SUFFIX = ".exe" if os.name == "nt" else ""

#: Fichier posé à côté de l'exécutable du cache : SHA-256 du ``.stan`` et version de CmdStan.
STAMP_SUFFIX = ".stamp.json"

#: Variable d'environnement qui impose la racine du cache (doit être ASCII et inscriptible).
CACHE_ENV = "EIS_STAN_CACHE_DIR"

#: Message de référence — repris tel quel par toutes les erreurs de ce module.
PATH_ERROR_MESSAGE = (
    "le chemin d'installation contient des caractères accentués ou spéciaux, ce qui empêche la "
    "compilation du moteur DRT sur certains systèmes Windows ; déplacez le dossier vers un chemin "
    "sans accents (ex: C:\\EIS_Analyzer) et relancez"
)

#: Avertissement INFORMATIF affiché au démarrage (rien n'a échoué à ce stade).
INSTALL_PATH_WARNING = (
    "Attention : votre dossier d'installation contient des caractères accentués, ce qui peut "
    "poser problème pour la compilation du moteur DRT. Si l'installation du moteur DRT échoue, "
    "essayez de déplacer l'application vers un chemin sans accents."
)


class NonAsciiPathError(RuntimeError):
    """La compilation exige un chemin ASCII qu'on n'a pas pu fournir (ou qu'on ne peut pas)."""


def has_non_ascii(path) -> bool:
    """Vrai si ``path`` — tel que cmdstanpy le transmettra à make — contient un caractère non-ASCII.

    cmdstanpy passe ``os.path.realpath`` du fichier, pas la chaîne reçue : un lien ou un
    ``subst`` ASCII vers un dossier accentué donnerait donc, malgré les apparences, un chemin
    accentué à make. Les deux formes sont testées.
    """
    raw = os.fspath(path)
    return not (os.path.abspath(raw).isascii() and os.path.realpath(raw).isascii())


def _explain(detail: object, path: object) -> NonAsciiPathError:
    return NonAsciiPathError(f"Impossible de compiler le modèle Stan : {PATH_ERROR_MESSAGE}. "
                             f"(Chemin concerné : {path} ; détail technique : {detail})")


def _cmdstan_dir() -> Optional[str]:
    try:
        import cmdstanpy

        return str(cmdstanpy.cmdstan_path()) or None
    except Exception:  # noqa: BLE001 — pas de CmdStan enregistré : sans objet ici
        return None


def _default_cmdstan_parent() -> str:
    # Même dossier COURT que ``setup_drt_bayesien._default_cmdstan_dir``.
    if os.name == "nt":
        return os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "cmdstan")
    return os.path.join(os.path.expanduser("~"), ".cmdstan")


def _candidate_roots(cmdstan_dir: Optional[str]) -> List[str]:
    """Racines possibles du cache, par ordre de préférence (la première ASCII et inscriptible gagne).

    À côté de l'installation CmdStan d'abord (elle doit déjà être ASCII pour que make tourne),
    puis à côté de l'emplacement par défaut, enfin dans le dossier temporaire.
    """
    roots = []
    if os.environ.get(CACHE_ENV):
        roots.append(os.environ[CACHE_ENV])
    if cmdstan_dir:
        roots.append(os.path.join(os.path.dirname(os.path.normpath(cmdstan_dir)), "model_cache"))
    roots.append(os.path.join(_default_cmdstan_parent(), "model_cache"))
    roots.append(os.path.join(tempfile.gettempdir(), "eis_model_cache"))
    seen, unique = set(), []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _ascii_cache_root(cmdstan_dir: Optional[str]) -> str:
    """Première racine de cache ASCII et réellement inscriptible, sinon :class:`NonAsciiPathError`."""
    refused = []
    for root in _candidate_roots(cmdstan_dir):
        if has_non_ascii(root):
            refused.append(f"{root} (non ASCII)")
            continue
        try:
            os.makedirs(root, exist_ok=True)
            with tempfile.TemporaryFile(dir=root):     # sonde d'écriture réelle, pas os.access
                pass
        except OSError as exc:
            refused.append(f"{root} ({exc})")
            continue
        return root
    raise _explain("aucun dossier de cache ASCII utilisable : " + " ; ".join(refused),
                   "cache de compilation")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_dir(root: str, stan_file: str) -> Path:
    return Path(root) / Path(stan_file).stem


def _read_stamp(model_dir: Path, name: str) -> Optional[dict]:
    try:
        return json.loads((model_dir / (name + STAMP_SUFFIX)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _version_label(cmdstan_dir: Optional[str]) -> str:
    return (installed_version(cmdstan_dir) if cmdstan_dir else None) or "inconnue"


def cached_exe(stan_file: str, cmdstan_version: Optional[str] = None) -> Optional[str]:
    """Exécutable du cache ASCII s'il correspond au ``.stan`` ACTUEL (contenu) et à la version de
    CmdStan, sinon ``None``. Aucune écriture, aucune compilation.

    ``cmdstan_version`` : celle du CmdStan actif (``None`` → relue depuis ``cmdstanpy``).
    """
    cmdstan_dir = _cmdstan_dir()
    wanted_version = cmdstan_version or _version_label(cmdstan_dir)
    try:
        digest = _sha256(stan_file)
    except OSError:
        return None
    name = Path(stan_file).stem
    for root in _candidate_roots(cmdstan_dir):
        if has_non_ascii(root):
            continue
        model_dir = _model_dir(root, stan_file)
        exe = model_dir / (name + EXE_SUFFIX)
        if exe.is_file() and _read_stamp(model_dir, name) == {"stan_sha256": digest,
                                                              "cmdstan_version": wanted_version}:
            return str(exe)
    return None


def _check_cmdstan_dir_ascii(cmdstan_dir: Optional[str]) -> None:
    """Un CmdStan installé sous un chemin non-ASCII ne se contourne pas par une copie : make
    tourne DANS ce dossier. Le dire clairement plutôt que de laisser make échouer."""
    if cmdstan_dir and has_non_ascii(cmdstan_dir):
        raise NonAsciiPathError(
            "Impossible de compiler le modèle Stan : CmdStan est installé sous un chemin contenant "
            "des caractères accentués ou spéciaux, ce qui empêche la compilation du moteur DRT sur "
            f"certains systèmes Windows ({cmdstan_dir}). Réinstallez CmdStan dans un dossier sans "
            "accents (ex: C:\\cmdstan) : python setup_drt_bayesien.py --cmdstan-dir C:\\cmdstan")


def compile_stan_model(stan_file, force: bool = False):
    """``CmdStanModel(stan_file=…)`` sûr pour tout chemin : renvoie un ``CmdStanModel`` compilé.

    * Chemin ASCII : appel direct à cmdstanpy, inchangé (``force`` → ``force_compile=True``).
    * Chemin non-ASCII : voir le module — copie dans un cache ASCII, compilation là, rien n'est
      recompilé tant que le SHA-256 du ``.stan`` et la version de CmdStan n'ont pas changé.

    Raises:
        NonAsciiPathError: le chemin non-ASCII ne peut pas être contourné (cache impossible à
            écrire, CmdStan lui-même sous un chemin accentué…) — message explicite, jamais la
            sortie brute de make seule.
        Exception: toute vraie erreur de compilation (code Stan invalide…), inchangée.
    """
    import cmdstanpy

    stan_file = os.fspath(stan_file)
    cmdstan_dir = _cmdstan_dir()

    if not has_non_ascii(stan_file):
        kwargs = {"force_compile": True} if force else {}
        try:
            return cmdstanpy.CmdStanModel(stan_file=stan_file, **kwargs)
        except Exception:
            _check_cmdstan_dir_ascii(cmdstan_dir)      # la cause probable, si elle s'applique
            raise

    name = Path(stan_file).stem
    version = _version_label(cmdstan_dir)
    digest = _sha256(stan_file)
    root = _ascii_cache_root(cmdstan_dir)
    model_dir = _model_dir(root, stan_file)
    cached_stan, exe = model_dir / Path(stan_file).name, model_dir / (name + EXE_SUFFIX)
    wanted = {"stan_sha256": digest, "cmdstan_version": version}

    if not force and exe.is_file() and _read_stamp(model_dir, name) == wanted:
        # exe_file= : cmdstanpy ne compare alors plus de dates, il prend l'exécutable validé.
        return cmdstanpy.CmdStanModel(stan_file=str(cached_stan), exe_file=str(exe))

    _check_cmdstan_dir_ascii(cmdstan_dir)
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        # Tampon et exécutable périmés d'abord : un échec ne doit laisser aucun « prêt » trompeur.
        for stale in (model_dir / (name + STAMP_SUFFIX), exe, model_dir / (name + ".hpp")):
            if stale.exists():
                stale.unlink()
        shutil.copyfile(stan_file, cached_stan)
    except OSError as exc:
        raise _explain(exc, model_dir) from exc

    model = cmdstanpy.CmdStanModel(stan_file=str(cached_stan), force_compile=True)

    try:
        tmp = model_dir / (name + STAMP_SUFFIX + ".tmp")
        tmp.write_text(json.dumps(wanted), encoding="utf-8")
        os.replace(tmp, model_dir / (name + STAMP_SUFFIX))
    except OSError as exc:
        raise _explain(exc, model_dir) from exc
    return model
