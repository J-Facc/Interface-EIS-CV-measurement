"""
setup_drt_bayesien.py — installe/enregistre CmdStan (toolchain C++ incluse) puis
précompile les modèles Stan de la DRT bayésienne (drt/bayes_drt2), afin que la première analyse
ne paie pas le coût de compilation.

C'est LA logique d'installation de la DRT : ``launch.bat`` l'appelle depuis le venv du
lanceur (``--check`` puis, si besoin, ``--ensure``) au lieu de la dupliquer ; un
développeur peut l'appeler à la main.

Points d'entrée :

* :func:`main` — CLI (voir ci-dessous).
* :func:`check_drt_ready` — sans effet de bord ni réseau : la DRT est-elle prête ?
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
    1  --check : pas prêt
    3  cmdstanpy absent du venv
    4  échec install_cmdstan (téléchargement / SSL / toolchain) alors que le réseau répond
    5  chemin cmdstan introuvable / invalide
    6  échec de la précompilation du modèle Stan
    7  échec de l'installation de CmdStan parce que le réseau est injoignable (hors ligne)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import socket
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Le paquet drt/bayes_drt2 (Inverter + modèles .stan) est résolu depuis le dépôt,
# quel que soit le répertoire courant.
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Modèles DRT « série » compilés d'avance : nonneg=False -> Series,
# nonneg=True -> Series_pos. Couvre les deux cas d'usage par défaut.
STAN_TARGETS = ["Series.stan", "Series_pos.stan"]

#: Version de CmdStan avec laquelle la DRT a été validée (drt/VALIDATION_REGLAGES.md ;
#: la CI utilise la même). Épinglée : sans elle, install_cmdstan prendrait la dernière.
PINNED_CMDSTAN_VERSION = "2.36.0"

EXIT_OK, EXIT_NOT_READY, EXIT_NO_CMDSTANPY, EXIT_INSTALL, EXIT_BAD_PATH, EXIT_COMPILE, EXIT_OFFLINE = (
    0, 1, 3, 4, 5, 6, 7)


def log(msg=""):
    print(msg, flush=True)


def _is_ms_store_python():
    exe = sys.executable or ""
    return "WindowsApps" in exe or os.path.join("Packages", "PythonSoftwareFoundation") in exe


def _default_cmdstan_dir():
    # Dossier COURT pour éviter les chemins trop longs (Python MS Store, MAX_PATH).
    if os.name == "nt":
        return os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "cmdstan")
    return os.path.join(os.path.expanduser("~"), ".cmdstan")


def _resolve_parent(parent=None):
    return parent or os.environ.get("CMDSTAN_INSTALL_DIR") or _default_cmdstan_dir()


def _latest_cmdstan(parent):
    """Renvoie le dossier ``cmdstan-X.Y.Z`` le plus récent sous ``parent``."""
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


def install(parent):
    import cmdstanpy

    os.makedirs(parent, exist_ok=True)
    log(f"[cmdstan] Installation de CmdStan {PINNED_CMDSTAN_VERSION} dans : {parent}")
    log("[cmdstan] compiler=True -> mingw-w64 sous Windows (evite d'exiger RTools).")
    log("[cmdstan] Telechargement puis compilation : plusieurs minutes la premiere fois.")
    # overwrite=False -> idempotent : ne retélécharge pas si déjà présent.
    # install_cmdstan RENVOIE un booléen au lieu de lever : l'ignorer ferait passer un
    # échec pour un succès jusqu'à l'étape suivante, avec un message sans rapport.
    ok = cmdstanpy.install_cmdstan(
        version=PINNED_CMDSTAN_VERSION, dir=parent, compiler=True, overwrite=False, progress=True)
    if not ok:
        raise RuntimeError("install_cmdstan a echoue (voir les messages ci-dessus)")


def register(parent):
    """Enregistre le chemin cmdstan pour la session (set_cmdstan_path)."""
    import cmdstanpy

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


def _stan_dir() -> str:
    from drt.bayes_drt2 import inversion

    return os.path.join(os.path.dirname(inversion.__file__), "stan_model_files")


def precompile():
    """Instancie l'Inverter puis compile les modèles Stan de drt/bayes_drt2 (sans sampling).

    Instancier l'Inverter valide l'import du paquet et résout le dossier
    des ``.stan``. La compilation via ``CmdStanModel`` met en cache l'exécutable ;
    aucun échantillonnage n'est lancé.
    """
    from cmdstanpy import CmdStanModel

    from drt.bayes_drt2.inversion import Inverter

    Inverter()  # honore « instancie l'Inverter » + valide l'import du paquet
    stan_dir = _stan_dir()

    compiled = []
    for name in STAN_TARGETS:
        stan_file = os.path.join(stan_dir, name)
        if not os.path.exists(stan_file):
            log(f"[stan] introuvable, ignore : {name}")
            continue
        log(f"[stan] Compilation de {name} (1er passage : peut durer 1-3 min)...")
        model = CmdStanModel(stan_file=stan_file)
        model.compile()  # idempotent : ne recompile pas si l'exe est a jour
        log(f"[stan]   -> executable : {model.exe_file}")
        compiled.append(name)

    if not compiled:
        raise RuntimeError("Aucun modele Stan compile (fichiers .stan manquants ?)")
    return compiled


def _series_exe_exists() -> bool:
    """True si les exécutables compilés de ``STAN_TARGETS`` existent tous (``Series_pos`` :
    modèle par défaut du moteur, ``drt.engine.DEFAULT_NONNEG`` ; ``Series`` : nonneg=False)."""
    try:
        stan_dir = _stan_dir()
    except Exception:
        return False
    for name in STAN_TARGETS:
        exe = os.path.join(stan_dir, name)[:-len(".stan")] + (".exe" if os.name == "nt" else "")
        if not os.path.exists(exe):
            return False
    return True


def check_drt_ready(parent: str = None):
    """La DRT est-elle prête ? ``(ready, raison)`` — AUCUNE installation, AUCUN réseau.

    Prête = cmdstanpy importable, un CmdStan valide sous ``parent`` (enregistré dans le
    process) et les exécutables Stan de ``STAN_TARGETS`` présents.
    """
    parent = _resolve_parent(parent)
    try:
        import cmdstanpy  # noqa: F401
    except ImportError:
        return False, "cmdstanpy absent (pip install -r requirements-drt.txt)"
    try:
        register(parent)
        verify()
    except Exception as exc:
        return False, f"CmdStan introuvable ou invalide ({exc})"
    if not _series_exe_exists():
        return False, "modeles Stan non compiles"
    return True, "DRT prete : CmdStan enregistre et modeles Stan compiles."


def _ensure(parent: str = None, force: bool = False, allow_install: bool = True):
    """Cœur de :func:`ensure_drt_ready` : renvoie ``(code_de_sortie, message)``."""
    parent = _resolve_parent(parent)

    try:
        import cmdstanpy  # noqa: F401
    except ImportError:
        return EXIT_NO_CMDSTANPY, (
            "cmdstanpy absent : installez l'extra DRT (pip install -r requirements-drt.txt)."
        )

    # 1. S'assurer que le chemin CmdStan est connu du process (register instantané).
    #    S'il échoue, CmdStan n'est pas installé -> l'installer (téléchargement + build,
    #    plusieurs minutes ; idempotent via overwrite=False) — si on y est autorisé.
    try:
        register(parent)
        verify()
    except Exception as exc_register:
        if not allow_install:
            return EXIT_BAD_PATH, f"CmdStan absent ({exc_register}) : lancez launch.bat ou " \
                                  "python setup_drt_bayesien.py --ensure."
        try:
            install(parent)
        except Exception as exc:  # téléchargement / SSL / toolchain
            if _network_down():
                return EXIT_OFFLINE, (
                    f"Hors ligne : CmdStan ne peut pas etre telecharge ({exc}). "
                    "La DRT restera indisponible jusqu'au prochain lancement avec reseau."
                )
            return EXIT_INSTALL, (
                f"Echec d'installation de CmdStan : {exc}. Verifiez le proxy/SSL (aide : "
                "python setup_drt_bayesien.py --help) ou la toolchain C++."
            )
        try:
            register(parent)
            verify()
        except Exception as exc:
            return EXIT_BAD_PATH, f"CmdStan installe mais inutilisable : {exc}"

    # 2. Compiler les modèles Stan SEULEMENT si leurs exécutables manquent.
    if force or not _series_exe_exists():
        if not allow_install:
            return EXIT_COMPILE, "Modeles Stan non compiles : lancez launch.bat ou " \
                                 "python setup_drt_bayesien.py --ensure."
        try:
            precompile()
        except Exception as exc:
            return EXIT_COMPILE, (
                f"Echec de compilation du modele Stan : {exc}. Verifiez la toolchain "
                "C++ (install_cmdstan(compiler=True)) puis relancez."
            )

    return EXIT_OK, "DRT prete : CmdStan enregistre et modeles Stan compiles."


def ensure_drt_ready(parent: str = None, force: bool = False, allow_install: bool = True):
    """Prépare la DRT (idempotent) : CmdStan enregistré + modèles Stan compilés.

    Ne fait que ce qui manque. Ne lève jamais : renvoie ``(ready: bool, message: str)``
    pour que l'app reste fonctionnelle (DRT désactivée proprement) si la toolchain échoue.
    ``allow_install=False`` : ne fait que ré-enregistrer un CmdStan DÉJÀ installé et
    constater l'état — aucun téléchargement, aucune compilation.
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
        description="DRT bayesienne : verifie / installe CmdStan (toolchain C++) et "
                    "precompile les modeles Stan. Idempotent.",
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

    code, message = _ensure(parent=args.cmdstan_dir, force=args.force, allow_install=True)
    log()
    log(("SUCCES : " if code == EXIT_OK else "ECHEC : ") + message)
    if code in (EXIT_INSTALL, EXIT_COMPILE):
        log(_ssl_hint())
    return code


if __name__ == "__main__":
    sys.exit(main())
