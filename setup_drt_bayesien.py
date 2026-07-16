"""
setup_drt_bayesien.py — installe/enregistre CmdStan (toolchain C++ incluse) puis
précompile le modèle Stan vendoré du DRT bayésien, afin que la première analyse
ne paie pas le coût de compilation.

Appelé par ``setup_drt_bayesien.bat``. Ce module n'est **jamais** importé par
l'application et n'est **jamais** exécuté au démarrage (``launch.bat`` ne
l'appelle pas). Il est prévu pour un lancement manuel, une seule fois.

Usage :
    python setup_drt_bayesien.py [--cmdstan-dir DIR] [--path-out FICHIER]

Codes de sortie :
    0  succès
    3  cmdstanpy absent du venv
    4  échec install_cmdstan (téléchargement / SSL / toolchain)
    5  chemin cmdstan introuvable / invalide
    6  échec de la précompilation du modèle Stan
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Le paquet vendoré (Inverter + modèles .stan) est résolu depuis le dépôt,
# quel que soit le répertoire courant.
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Modèles DRT « série » compilés d'avance : nonneg=False -> Series,
# nonneg=True -> Series_pos. Couvre les deux cas d'usage par défaut.
STAN_TARGETS = ["Series.stan", "Series_pos.stan"]


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


def install(parent):
    import cmdstanpy

    os.makedirs(parent, exist_ok=True)
    log(f"[cmdstan] Installation dans : {parent}")
    log("[cmdstan] compiler=True -> mingw-w64 sous Windows (evite d'exiger RTools).")
    # overwrite=False -> idempotent : ne retélécharge pas si déjà présent.
    cmdstanpy.install_cmdstan(dir=parent, compiler=True, overwrite=False)


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


def precompile():
    """Instancie l'Inverter puis compile les modèles Stan vendorés (sans sampling).

    Instancier l'Inverter valide l'import du paquet vendoré et résout le dossier
    des ``.stan``. La compilation via ``CmdStanModel`` met en cache l'exécutable ;
    aucun échantillonnage n'est lancé.
    """
    from cmdstanpy import CmdStanModel

    from vendor.bayes_drt2 import inversion
    from vendor.bayes_drt2.inversion import Inverter

    Inverter()  # honore « instancie l'Inverter » + valide l'import du paquet
    stan_dir = os.path.join(os.path.dirname(inversion.__file__), "stan_model_files")

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
        "  puis relancer setup_drt_bayesien.bat.\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Setup DRT bayesien (cmdstan + toolchain C++ + precompilation Stan)"
    )
    parser.add_argument(
        "--cmdstan-dir", default=None,
        help="Dossier parent d'installation de cmdstan (defaut : dossier court)",
    )
    parser.add_argument(
        "--path-out", default=None,
        help="Fichier ou ecrire le chemin cmdstan resolu (consomme par le .bat)",
    )
    args = parser.parse_args()

    parent = (
        args.cmdstan_dir
        or os.environ.get("CMDSTAN_INSTALL_DIR")
        or _default_cmdstan_dir()
    )

    if _is_ms_store_python():
        log("[info] Python Microsoft Store detecte -> dossier court obligatoire pour cmdstan.")

    try:
        import cmdstanpy  # noqa: F401
    except ImportError:
        log("ERREUR : cmdstanpy absent du venv. Lancez d'abord : pip install cmdstanpy")
        return 3

    try:
        install(parent)
    except Exception as exc:  # téléchargement / SSL / toolchain
        log(f"ERREUR pendant install_cmdstan : {exc}")
        log(_ssl_hint())
        return 4

    try:
        register(parent)
        cmdstan_path = verify()
    except Exception as exc:
        log(f"ERREUR : impossible d'enregistrer/verifier le chemin cmdstan : {exc}")
        return 5

    # Écrire le chemin résolu pour que le .bat puisse le rendre persistant (CMDSTAN).
    if args.path_out:
        try:
            with open(args.path_out, "w", encoding="utf-8") as handle:
                handle.write(cmdstan_path)
        except OSError as exc:
            log(f"[avert] impossible d'ecrire {args.path_out} : {exc}")

    try:
        precompile()
    except Exception as exc:
        log(f"ERREUR pendant la precompilation du modele Stan : {exc}")
        return 6

    log()
    log("SUCCES : cmdstan installe, chemin enregistre, modele Stan precompile.")
    log(f"        CMDSTAN = {cmdstan_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
