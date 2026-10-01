"""
setup_drt_bayesien.py — installe/enregistre CmdStan (toolchain C++ incluse) puis
précompile les modèles Stan de la DRT bayésienne (drt/bayes_drt2), afin que la première analyse
ne paie pas le coût de compilation.

C'est LA logique d'installation de la DRT : ``launch.bat`` l'appelle depuis le venv du
lanceur (``--check`` puis, si besoin, ``--ensure``) au lieu de la dupliquer ; un
développeur peut l'appeler à la main.

**Version épinglée.** Les réglages de ``drt/engine.py`` ont été validés sur CmdStan
``PINNED_CMDSTAN_VERSION`` (``drt/cmdstan_version.py``). Ce script :

* enregistre CETTE version quand elle est installée, jamais « la plus récente du dossier » ;
* l'installe à côté d'une autre version déjà présente (rien n'est jamais désinstallé) ;
* ne se rabat sur une autre version que si l'installation de la version épinglée est
  impossible (hors ligne) — et le dit, dans ses logs comme dans l'UI ;
* note, à côté de chaque exécutable Stan compilé, la version de CmdStan qui l'a produit :
  cmdstanpy ne recompile qu'une source plus récente que son exécutable, donc sans cette
  trace un ``Series.exe`` bâti sous une autre version serait réutilisé en silence ;
* ne laisse jamais ``make`` voir un chemin non-ASCII (``C:\\Users\\x\\.Thèse\\…``) : le shell MSYS
  que lance mingw32-make le reçoit corrompu. Un ``.stan`` situé sous un tel chemin est compilé
  depuis un cache ASCII (``drt/stan_compile.py``, qui documente aussi pourquoi ni
  ``PYTHONUTF8`` ni ``chcp 65001`` ne suffisent). Si même cela est impossible, le message dit
  de déplacer le dossier — jamais la sortie brute de make.

Points d'entrée :

* :func:`main` — CLI (voir ci-dessous).
* :func:`check_drt_ready` — sans effet de bord ni réseau : la DRT est-elle prête, SUR LA
  VERSION VALIDÉE ?
* :func:`ensure_drt_ready` — **idempotent**, renvoie ``(ready, message)`` et ne lève
  jamais. ``app.py`` l'appelle avec ``allow_install=False`` : l'application n'installe
  RIEN d'elle-même (une installation lancée depuis le script d'une page bloquerait
  l'interface sans délai maximal, et hors ligne elle échouerait à chaque démarrage) ;
  l'installation est l'affaire du lanceur.

Usage CLI :
    python setup_drt_bayesien.py --check             # 0 si prêt, 1 sinon (aucune installation)
    python setup_drt_bayesien.py [--ensure]          # installe/compile SEULEMENT ce qui manque
    python setup_drt_bayesien.py --ensure --force    # recompile les modèles Stan
    options : --cmdstan-dir DIR

Codes de sortie :
    0  succès (ou, avec --check : prêt)
    1  --check : pas prêt (CmdStan absent, autre version que la version validée, modèles
       Stan non compilés pour la version active)
    3  cmdstanpy absent du venv
    4  échec install_cmdstan (téléchargement / SSL) alors que le réseau répond
    5  chemin cmdstan introuvable / invalide
    6  échec de la précompilation du modèle Stan (erreur de compilation, toolchain OK)
    7  échec de l'installation de CmdStan parce que le réseau est injoignable (hors ligne)
    8  toolchain C++ absente ou non fonctionnelle (make / compilateur introuvable)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import socket
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Le paquet drt/bayes_drt2 (Inverter + modèles .stan) est résolu depuis le dépôt,
# quel que soit le répertoire courant.
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from drt import stan_compile  # noqa: E402
from drt.cmdstan_version import (  # noqa: E402 — après l'ajout de REPO_DIR au chemin
    PINNED_CMDSTAN_VERSION, installed_version, version_warning)

# Modèles DRT « série » compilés d'avance. ``Inverter.fit(nonneg=…, outliers=False)``
# (drt/engine.py ne passe jamais ``outliers``) choisit ``Series_pos`` pour nonneg=True — le
# défaut du moteur, ``drt.engine.DEFAULT_NONNEG`` — et ``Series`` pour nonneg=False.
# ``tests/test_setup_drt.py`` rejoue cette sélection sur le vrai ``Inverter`` : un modèle
# utilisable qui manquerait ici fait échouer la suite.
STAN_TARGETS = ["Series.stan", "Series_pos.stan"]

#: Suffixe du fichier, posé à côté de chaque exécutable, qui nomme la version de CmdStan
#: l'ayant compilé (``Series.cmdstan-version``). Il porte le nom du modèle suivi d'un
#: point : ``launch.bat`` (``stan_cache``) le conserve avec l'exécutable (``Series.*``).
MARKER_SUFFIX = ".cmdstan-version"

(EXIT_OK, EXIT_NOT_READY, EXIT_NO_CMDSTANPY, EXIT_INSTALL, EXIT_BAD_PATH, EXIT_COMPILE,
 EXIT_OFFLINE, EXIT_TOOLCHAIN) = (0, 1, 3, 4, 5, 6, 7, 8)


class StanCompileError(RuntimeError):
    """La compilation d'UN modèle Stan a échoué (cause de ``precompile``, pas un autre échec)."""

    def __init__(self, model: str, original: BaseException):
        super().__init__(f"{type(original).__name__}: {original}")
        self.model = model
        self.original = original


def log(msg=""):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Console / redirection non UTF-8 (cp850, cp1252…) : un caractère hors de son jeu
        # ne doit jamais faire échouer l'installation — il est remplacé, pas propagé.
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(encoding, "replace").decode(encoding), flush=True)


def _is_ms_store_python():
    exe = sys.executable or ""
    return "WindowsApps" in exe or os.path.join("Packages", "PythonSoftwareFoundation") in exe


def _is_windows() -> bool:
    return os.name == "nt"


def _default_cmdstan_dir():
    # Dossier COURT pour éviter les chemins trop longs (Python MS Store, MAX_PATH).
    if os.name == "nt":
        return os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "cmdstan")
    return os.path.join(os.path.expanduser("~"), ".cmdstan")


def _resolve_parent(parent=None):
    return parent or os.environ.get("CMDSTAN_INSTALL_DIR") or _default_cmdstan_dir()


def _pinned_dir(parent) -> str:
    return os.path.join(parent, f"cmdstan-{PINNED_CMDSTAN_VERSION}")


def _latest_cmdstan(parent):
    """Renvoie le dossier ``cmdstan-X.Y.Z`` le plus récent sous ``parent``.

    Repli UNIQUEMENT : la version à utiliser est la version épinglée (voir :func:`register`).
    """
    candidates = []
    for path in glob.glob(os.path.join(parent, "cmdstan-*")):
        if not os.path.isdir(path):
            continue
        match = re.search(r"cmdstan-(\d+)\.(\d+)\.(\d+)", os.path.basename(path))
        key = tuple(int(x) for x in match.groups()) if match else (0, 0, 0)
        candidates.append((key, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _network_down(timeout: float = 5.0) -> bool:
    """True si github.com:443 est injoignable ET qu'aucun proxy n'est déclaré.

    Sert UNIQUEMENT à étiqueter un échec (« hors ligne » ≠ « échec réel »), jamais à en
    bloquer un : derrière un proxy, une connexion directe échoue alors que pip/requests
    passent très bien, donc on ne conclut « hors ligne » que sans variable de proxy.
    """
    if any(os.environ.get(v) for v in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")):
        return False
    try:
        with socket.create_connection(("github.com", 443), timeout=timeout):
            return False
    except OSError:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Toolchain C++ (make + compilateur)
# ─────────────────────────────────────────────────────────────────────────────
# cmdstanpy lance ``make`` (``mingw32-make`` sous Windows) par subprocess : l'outil doit être
# dans le PATH DU PROCESSUS qui compile. Or cmdstanpy 1.3.0 n'ajoute son toolchain RTools au
# PATH que dans ``install_cmdstan(compiler=True)`` — donc seulement dans le processus qui vient
# d'installer. Tout autre processus qui compile ensuite (relance du lanceur, application)
# échoue en « [WinError 2] Le fichier spécifié est introuvable » alors que RTools est bien là.
# D'où :func:`activate_toolchain`, appelée avant toute compilation ET à l'enregistrement.

def _toolchain_needs():
    """[(outils équivalents…), …] : au moins un outil de CHAQUE groupe doit fonctionner."""
    make = os.environ.get("MAKE") or ("mingw32-make" if _is_windows() else "make")
    compilers = ("g++",) if _is_windows() else ("g++", "clang++")
    return [(make,), compilers]


def _probe_tool(name: str):
    """``(fonctionne, détail)`` : l'outil est dans le PATH ET répond à ``--version``."""
    exe = shutil.which(name)
    if exe is None:
        return False, "introuvable dans le PATH"
    try:
        result = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{exe} n'a pas pu etre lance ({type(exc).__name__}: {exc})"
    if result.returncode != 0:
        return False, f"{exe} --version a rendu le code {result.returncode}"
    first = (result.stdout or result.stderr or "").strip().splitlines()
    return True, (first[0] if first else exe)


def toolchain_status():
    """``(ok, problèmes, trouvés)`` — AUCUNE installation.

    ``problèmes`` : une phrase par groupe d'outils sans aucun représentant fonctionnel ;
    ``trouvés`` : « outil : première ligne de --version » des outils qui répondent.
    """
    problems, found = [], []
    for group in _toolchain_needs():
        results = [(name,) + _probe_tool(name) for name in group]
        working = [(n, d) for n, ok, d in results if ok]
        if working:
            found.append(f"{working[0][0]} : {working[0][1]}")
        else:
            problems.append(" / ".join(group) + " : " + "; ".join(d for _n, _ok, d in results))
    return not problems, problems, found


def activate_toolchain(parent) -> bool:
    """Windows : met au PATH du process un toolchain RTools DÉJÀ installé (sans rien installer).

    Renvoie True si le PATH a été modifié. Sans effet hors Windows, ou si ``make`` et le
    compilateur répondent déjà, ou si aucun RTools n'est trouvé (ce n'est pas une erreur :
    c'est à :func:`_ensure_toolchain` de trancher, avec le bon message).
    """
    if not _is_windows():
        return False
    if toolchain_status()[0]:
        return False
    try:
        from cmdstanpy.utils import cxx_toolchain_path

        cxx_toolchain_path(None, parent)   # cherche sous parent\RTools40, ~\.cmdstan, C:\RTools40…
    except Exception:  # noqa: BLE001 — aucun RTools trouvé / cmdstanpy sans cette API
        return False
    return True


def _toolchain_message(problems) -> str:
    if _is_windows():
        hint = ("Installez-la : python -m cmdstanpy.install_cxx_toolchain --dir "
                f"{_default_cmdstan_dir()} (RTools 4.0 + mingw32-make), ou relancez launch.bat "
                "avec un acces reseau.")
    elif sys.platform == "darwin":
        hint = "Installez-la : xcode-select --install."
    else:
        hint = "Installez-la : sudo apt install build-essential (ou l'equivalent de votre distribution)."
    return ("Toolchain C++ absente ou non fonctionnelle (" + " | ".join(problems) + "). "
            "Les modeles Stan ne peuvent pas etre compiles ; ce n'est PAS une erreur dans les "
            "modeles. " + hint)


def _ensure_toolchain(parent, allow_install: bool):
    """S'assure que ``make`` et le compilateur fonctionnent AVANT toute compilation.

    Renvoie ``None`` si la toolchain fonctionne, sinon le message d'échec (la cause précise).
    Installe la toolchain (Windows) via ``install_cmdstan(compiler=True)`` quand c'est permis,
    puis RE-VÉRIFIE : une installation « réussie » dont les binaires ne sont pas au PATH est
    un échec, pas un succès.
    """
    activate_toolchain(parent)
    ok, problems, found = toolchain_status()
    if ok:
        log("[toolchain] OK : " + " ; ".join(found))
        return None
    log("[toolchain] non fonctionnelle : " + " | ".join(problems))
    if not allow_install:
        return _toolchain_message(problems)
    if not _is_windows():
        # cmdstanpy n'installe un compilateur que sous Windows : relancer install_cmdstan
        # ici ne changerait rien et coûterait plusieurs minutes.
        return _toolchain_message(problems)
    log("[toolchain] installation de la toolchain C++ (RTools 4.0) via install_cmdstan(compiler=True)...")
    try:
        install(parent)
    except Exception as exc:  # noqa: BLE001
        reason = "reseau injoignable" if _network_down() else "echec de l'installation"
        return (f"Toolchain C++ non installable ({reason} : {exc}). "
                + _toolchain_message(problems))
    activate_toolchain(parent)
    ok, problems, found = toolchain_status()
    if not ok:
        return ("Toolchain C++ installee mais ses binaires ne sont toujours pas utilisables "
                "depuis ce processus. " + _toolchain_message(problems))
    log("[toolchain] OK apres installation : " + " ; ".join(found))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CmdStan
# ─────────────────────────────────────────────────────────────────────────────
def install(parent):
    """Installe CmdStan ``PINNED_CMDSTAN_VERSION`` (et, sous Windows, sa toolchain)."""
    import cmdstanpy

    os.makedirs(parent, exist_ok=True)
    log(f"[cmdstan] Installation de CmdStan {PINNED_CMDSTAN_VERSION} dans : {parent}")
    log("[cmdstan] compiler=True -> toolchain RTools 4.0 (mingw-w64) installee par cmdstanpy sous Windows.")
    log("[cmdstan] Telechargement puis compilation : plusieurs minutes la premiere fois.")
    # version= EXPLICITE : sans elle install_cmdstan prendrait la DERNIERE version publiée,
    # que la DRT n'a pas été validée dessus. overwrite=False -> idempotent : ne retélécharge
    # pas si déjà présent. install_cmdstan RENVOIE un booléen au lieu de lever : l'ignorer
    # ferait passer un échec pour un succès jusqu'à l'étape suivante, avec un message sans rapport.
    ok = cmdstanpy.install_cmdstan(
        version=PINNED_CMDSTAN_VERSION, dir=parent, compiler=True, overwrite=False, progress=True)
    if not ok:
        raise RuntimeError("install_cmdstan a echoue (voir les messages ci-dessus)")


def register(parent):
    """Enregistre le chemin cmdstan pour la session (set_cmdstan_path).

    La version épinglée l'emporte TOUJOURS sur une version plus récente présente dans le
    même dossier. Une autre version n'est enregistrée qu'en repli, quand l'épinglée est
    absente : l'appelant en tire l'avertissement (:func:`version_warning`).
    """
    import cmdstanpy

    path = _pinned_dir(parent)
    if not os.path.isdir(path):
        path = _latest_cmdstan(parent)
    if path is None:
        raise RuntimeError(f"Aucun dossier cmdstan-* trouve dans {parent}")
    cmdstanpy.set_cmdstan_path(path)
    return path


def verify():
    import cmdstanpy

    path = cmdstanpy.cmdstan_path()
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"cmdstan_path() invalide : {path!r}")
    log(f"[cmdstan] cmdstan_path() = {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Modèles Stan
# ─────────────────────────────────────────────────────────────────────────────
def _stan_dir() -> str:
    from drt.bayes_drt2 import inversion

    return os.path.join(os.path.dirname(inversion.__file__), "stan_model_files")


def _exe_path(stan_dir: str, name: str) -> str:
    return os.path.join(stan_dir, name[:-len(".stan")] + (".exe" if os.name == "nt" else ""))


def _marker_path(stan_dir: str, name: str) -> str:
    return os.path.join(stan_dir, name[:-len(".stan")] + MARKER_SUFFIX)


def _compiled_with(stan_dir: str, name: str):
    """Version de CmdStan notée à côté de l'exécutable de ``name``, ``None`` si absente."""
    try:
        with open(_marker_path(stan_dir, name), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _is_compiled(stan_dir: str, name: str, stamp: str) -> bool:
    """Le modèle ``name`` est-il compilé POUR cette version de CmdStan et CE ``.stan`` ?"""
    if stan_compile.has_non_ascii(stan_dir):
        # Compilé dans le cache ASCII (stan_compile) : son tampon vérifie le contenu du .stan
        # ET la version, pas seulement la version comme le marqueur ci-dessous.
        return stan_compile.cached_exe(os.path.join(stan_dir, name), stamp) is not None
    return os.path.exists(_exe_path(stan_dir, name)) and _compiled_with(stan_dir, name) == stamp


def _models_to_compile(version, force: bool = False) -> list:
    """Modèles de ``STAN_TARGETS`` à (re)compiler pour CmdStan ``version``.

    À compiler : exécutable absent, OU non compilé sous cette version (marqueur absent ou
    différent — un exécutable d'origine inconnue n'est jamais présumé bon), OU ``force``.
    Sous un chemin non-ASCII, « exécutable » et « marqueur » sont ceux du cache ASCII.
    """
    if force:
        return list(STAN_TARGETS)
    try:
        stan_dir = _stan_dir()
    except Exception:  # noqa: BLE001 — paquet drt illisible : rien n'est prêt
        return list(STAN_TARGETS)
    stamp = version or "inconnue"
    return [name for name in STAN_TARGETS if not _is_compiled(stan_dir, name, stamp)]


def _tail(text: str, lines: int = 15, chars: int = 1500) -> str:
    out = "\n".join(str(text).strip().splitlines()[-lines:])
    return out[-chars:]


def precompile(force: bool = False):
    """Instancie l'Inverter puis compile les modèles Stan de drt/bayes_drt2 (sans sampling).

    Instancier l'Inverter valide l'import du paquet et résout le dossier des ``.stan``.
    ``CmdStanModel`` compile et met l'exécutable en cache ; aucun échantillonnage n'est lancé.
    La compilation passe par :func:`drt.stan_compile.compile_stan_model`, qui ne montre jamais
    à make un chemin non-ASCII.
    Un modèle n'est recompilé que s'il le faut (:func:`_models_to_compile`), ou si ``force``.

    Raises:
        StanCompileError: la compilation d'un modèle a échoué.
        drt.stan_compile.NonAsciiPathError: chemin non-ASCII impossible à contourner.
        RuntimeError: un ``.stan`` de ``STAN_TARGETS`` manque — jamais ignoré : le moteur
            peut le sélectionner.
    """
    import cmdstanpy

    from drt.bayes_drt2.inversion import Inverter

    Inverter()  # honore « instancie l'Inverter » + valide l'import du paquet
    stan_dir = _stan_dir()
    version = installed_version(cmdstanpy.cmdstan_path()) or "inconnue"
    todo = _models_to_compile(version, force=force)

    compiled = []
    for name in STAN_TARGETS:
        stan_file = os.path.join(stan_dir, name)
        if not os.path.exists(stan_file):
            raise RuntimeError(f"modele Stan introuvable : {stan_file}")
        if name not in todo:
            log(f"[stan] {name} deja compile avec CmdStan {version}")
            compiled.append(name)
            continue
        log(f"[stan] Compilation de {name} avec CmdStan {version} (1er passage : 1-3 min)...")
        try:
            # force : un exécutable plus récent que sa source mais bâti sous une autre version
            # de CmdStan serait sinon réutilisé tel quel par cmdstanpy.
            model = stan_compile.compile_stan_model(stan_file, force=True)
        except stan_compile.NonAsciiPathError:
            raise                              # déjà explicite : ni enveloppée, ni « erreur C++ »
        except Exception as exc:  # noqa: BLE001
            raise StanCompileError(name, exc) from exc
        if not stan_compile.has_non_ascii(stan_dir):   # sinon le tampon du cache ASCII en tient lieu
            with open(_marker_path(stan_dir, name), "w", encoding="utf-8") as fh:
                fh.write(version + "\n")
        log(f"[stan]   -> executable : {model.exe_file}")
        compiled.append(name)
    return compiled


def _explain_compile_failure(exc: StanCompileError):
    """``(code de sortie, message)`` d'un échec de compilation, avec sa cause PRÉCISE."""
    ok, problems, found = toolchain_status()
    if not ok:
        return EXIT_TOOLCHAIN, (
            f"Compilation de {exc.model} impossible : la toolchain C++ ne fonctionne plus "
            f"({' | '.join(problems)}). " + _toolchain_message(problems))
    return EXIT_COMPILE, (
        f"Erreur de compilation Stan/C++ de {exc.model} (la toolchain, elle, fonctionne : "
        f"{' ; '.join(found)}). Derniere sortie :\n{_tail(str(exc))}")


# ─────────────────────────────────────────────────────────────────────────────
# État
# ─────────────────────────────────────────────────────────────────────────────
def check_drt_ready(parent: str = None):
    """La DRT est-elle prête SUR LA VERSION VALIDÉE ? ``(ready, raison)`` — AUCUNE installation,
    AUCUN réseau.

    Prête = cmdstanpy importable, CmdStan ``PINNED_CMDSTAN_VERSION`` installé sous ``parent``
    (enregistré dans le process) et exécutables Stan de ``STAN_TARGETS`` compilés AVEC lui.
    Une autre version de CmdStan n'est jamais « prête » : la raison est alors l'avertissement
    explicite, et ``launch.bat`` enchaîne sur ``--ensure`` qui installe la version validée.
    (L'application, elle, tolère un repli sur une autre version tant qu'elle la signale : voir
    :func:`ensure_drt_ready`.)
    """
    parent = _resolve_parent(parent)
    try:
        import cmdstanpy  # noqa: F401
    except ImportError:
        return False, "cmdstanpy absent (pip install -r requirements-drt.txt)"
    try:
        path = register(parent)
        verify()
    except Exception as exc:
        return False, f"CmdStan introuvable ou invalide ({exc})"
    version = installed_version(path)
    warning = version_warning(version)
    if warning:
        return False, "ATTENTION : " + warning
    todo = _models_to_compile(version)
    if todo:
        return False, f"modeles Stan non compiles pour CmdStan {version} : {', '.join(todo)}"
    return True, f"DRT prete : CmdStan {version} enregistre et modeles Stan compiles."


def _try_register(parent):
    """``(chemin, None)`` ou ``(None, exception)`` — enregistre puis valide CmdStan."""
    try:
        path = register(parent)
        verify()
        return path, None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def _ensure(parent: str = None, force: bool = False, allow_install: bool = True):
    """Cœur de :func:`ensure_drt_ready` : renvoie ``(code_de_sortie, message)``."""
    parent = _resolve_parent(parent)

    try:
        import cmdstanpy  # noqa: F401
    except ImportError:
        return EXIT_NO_CMDSTANPY, (
            "cmdstanpy absent : installez l'extra DRT (pip install -r requirements-drt.txt)."
        )

    # 0. make tourne DANS le dossier de CmdStan : un chemin non-ASCII là ne se contourne pas par
    #    une copie. Mieux vaut le refuser AVANT un téléchargement de plusieurs minutes.
    if allow_install and stan_compile.has_non_ascii(parent):
        return EXIT_BAD_PATH, (
            f"Le dossier d'installation de CmdStan ({parent}) contient des caractères accentués "
            "ou spéciaux, ce qui empêche la compilation du moteur DRT sur certains systèmes "
            "Windows ; utilisez un dossier sans accents (ex: C:\\cmdstan : --cmdstan-dir "
            "C:\\cmdstan, ou la variable CMDSTAN_INSTALL_DIR) et relancez.")

    # 1. CmdStan. La version épinglée est celle qu'on veut ; si elle manque, on l'installe
    #    (téléchargement + build, plusieurs minutes ; idempotent via overwrite=False) — si on
    #    y est autorisé, et À CÔTÉ d'une autre version éventuelle, qu'on ne touche pas.
    path, err = _try_register(parent)
    if (path is None or installed_version(path) != PINNED_CMDSTAN_VERSION) and allow_install:
        try:
            install(parent)
        except Exception as exc:  # téléchargement / SSL / toolchain
            if path is None:
                if _network_down():
                    return EXIT_OFFLINE, (
                        f"Hors ligne : CmdStan ne peut pas etre telecharge ({exc}). "
                        "La DRT restera indisponible jusqu'au prochain lancement avec reseau."
                    )
                return EXIT_INSTALL, (
                    f"Echec d'installation de CmdStan : {exc}. Verifiez le proxy/SSL (aide : "
                    "python setup_drt_bayesien.py --help) ou la toolchain C++."
                )
            log(f"[cmdstan] CmdStan {PINNED_CMDSTAN_VERSION} non installable ({exc}) : repli sur "
                f"la version deja presente ({installed_version(path) or 'inconnue'}).")
        else:
            path, err = _try_register(parent)
            if path is None:
                return EXIT_BAD_PATH, f"CmdStan installe mais inutilisable : {err}"
    if path is None:
        return EXIT_BAD_PATH, f"CmdStan absent ({err}) : lancez launch.bat ou " \
                              "python setup_drt_bayesien.py --ensure."

    version = installed_version(path)
    warning = version_warning(version)
    if warning:
        log("[cmdstan] ATTENTION : " + warning)

    # 2. Le PATH de CE process doit contenir la toolchain : compiler, ici ou plus tard
    #    (application), en dépend. Sans effet réseau ni installation.
    activate_toolchain(parent)

    # 3. Compiler les modèles Stan SEULEMENT s'ils manquent ou ont été bâtis sous une autre
    #    version de CmdStan — et seulement après avoir vérifié la toolchain.
    todo = _models_to_compile(version, force=force)
    if todo:
        if not allow_install:
            return EXIT_COMPILE, "Modeles Stan non compiles : lancez launch.bat ou " \
                                 "python setup_drt_bayesien.py --ensure."
        problem = _ensure_toolchain(parent, allow_install=True)
        if problem is not None:
            return EXIT_TOOLCHAIN, problem
        try:
            precompile(force=force)
        except StanCompileError as exc:
            return _explain_compile_failure(exc)
        except stan_compile.NonAsciiPathError as exc:
            return EXIT_COMPILE, str(exc)
        except Exception as exc:  # noqa: BLE001 — hors compilation C++ : import, .stan absent…
            return EXIT_COMPILE, (
                f"Echec de preparation des modeles Stan (avant la compilation C++) : "
                f"{type(exc).__name__}: {exc}")

    message = f"DRT prete : CmdStan {version} enregistre et modeles Stan compiles."
    if warning:
        message += " ATTENTION : " + warning
    return EXIT_OK, message


def ensure_drt_ready(parent: str = None, force: bool = False, allow_install: bool = True):
    """Prépare la DRT (idempotent) : CmdStan enregistré + modèles Stan compilés.

    Ne fait que ce qui manque. Ne lève jamais : renvoie ``(ready: bool, message: str)``
    pour que l'app reste fonctionnelle (DRT désactivée proprement) si la toolchain échoue.
    ``allow_install=False`` : ne fait que ré-enregistrer un CmdStan DÉJÀ installé et
    constater l'état — aucun téléchargement, aucune compilation.

    Une autre version que la version validée reste UTILISABLE (``ready`` vrai) tant que ses
    modèles sont compilés : le message commence alors par l'avertissement, que l'application
    affiche. C'est ``--check`` qui, lui, exige la version validée.
    """
    code, message = _ensure(parent=parent, force=force, allow_install=allow_install)
    return code == EXIT_OK, message


def _ssl_hint():
    return (
        "\n"
        "Piste en cas d'echec SSL derriere un proxy d'entreprise :\n"
        "  - Definir le proxy :\n"
        "        set HTTPS_PROXY=http://utilisateur:motdepasse@proxy:port\n"
        "        set HTTP_PROXY=http://utilisateur:motdepasse@proxy:port\n"
        "  - Interception TLS (certificat interne) -> pointer vers le CA bundle :\n"
        "        set REQUESTS_CA_BUNDLE=C:\\chemin\\vers\\ca-bundle.pem\n"
        "        set SSL_CERT_FILE=C:\\chemin\\vers\\ca-bundle.pem\n"
        "  puis relancer launch.bat.\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DRT bayesienne : verifie / installe CmdStan " + PINNED_CMDSTAN_VERSION +
                    " (toolchain C++) et precompile les modeles Stan. Idempotent.",
        epilog=_ssl_hint(), formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="verifie seulement (aucune installation, aucun reseau) : 0 = pret, 1 = non")
    mode.add_argument("--ensure", action="store_true",
                      help="installe/compile seulement ce qui manque (comportement par defaut)")
    parser.add_argument("--force", action="store_true", help="recompile les modeles Stan")
    parser.add_argument(
        "--cmdstan-dir", default=None,
        help="Dossier parent d'installation de cmdstan (defaut : C:\\cmdstan sous Windows)",
    )
    args = parser.parse_args(argv)

    if args.check:
        ready, reason = check_drt_ready(args.cmdstan_dir)
        log(reason)
        return EXIT_OK if ready else EXIT_NOT_READY

    if _is_ms_store_python():
        log("[info] Python Microsoft Store detecte -> dossier court obligatoire pour cmdstan.")
    if stan_compile.has_non_ascii(REPO_DIR):
        log("[info] " + stan_compile.INSTALL_PATH_WARNING)
        log("[info] Les modeles Stan seront compiles depuis un cache ASCII ; en cas d'echec, "
            "deplacez l'application, par exemple vers C:\\EIS_Analyzer.")

    code, message = _ensure(parent=args.cmdstan_dir, force=args.force, allow_install=True)
    log()
    log(("SUCCES : " if code == EXIT_OK else "ECHEC : ") + message)
    if code in (EXIT_INSTALL, EXIT_COMPILE):
        log(_ssl_hint())
    return code


if __name__ == "__main__":
    sys.exit(main())
