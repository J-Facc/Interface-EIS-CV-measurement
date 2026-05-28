"""
EIS Analyzer Launcher — sans Git
Mise à jour via téléchargement ZIP depuis l'API GitHub.
"""
import os
import sys
import ssl
import socket
import subprocess
import time
import webbrowser
import zipfile
import shutil
import json
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
GITHUB_USER   = "J-Facc"
GITHUB_REPO   = "Interface-EIS-CV-measurement"
BRANCH        = "main"
BASE_DIR      = Path(__file__).parent
APP_DIR       = BASE_DIR / "eis_app"
VENV_DIR      = BASE_DIR / ".venv"
VERSION_FILE  = BASE_DIR / ".version_hash"   # stocke le SHA du dernier commit connu
APP_ENTRY     = APP_DIR / "app.py"
PORT          = 8501
TIMEOUT_START = 30   # secondes max pour attendre Streamlit
# ───────────────────────────────────────────────────────────────────────────────

ZIP_URL    = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/zipball/{BRANCH}"
COMMIT_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/commits/{BRANCH}"

# Contexte SSL sans vérification — nécessaire sur certains réseaux d'entreprise
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def show_error_and_exit(msg: str):
    """Affiche une boîte d'erreur tkinter et quitte proprement."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("EIS Analyzer — Erreur", msg)
    root.destroy()
    sys.exit(1)


def update_status(msg: str):
    print(f"[EIS Launcher] {msg}")
    try:
        from splash import update_label
        update_label(msg)
    except Exception:
        pass


def is_online() -> bool:
    """Vérifie la connectivité en tentant de joindre l'API GitHub directement."""
    urls = ["https://api.github.com", "https://github.com"]
    for url in urls:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "EIS-Analyzer-Launcher"}
            )
            urllib.request.urlopen(req, timeout=5, context=SSL_CTX)
            return True
        except Exception:
            continue
    return False


def get_remote_sha() -> str | None:
    """Récupère le SHA du dernier commit sur BRANCH via l'API GitHub."""
    try:
        req = urllib.request.Request(
            COMMIT_URL,
            headers={"Accept": "application/vnd.github.v3+json",
                     "User-Agent": "EIS-Analyzer-Launcher"}
        )
        with urllib.request.urlopen(req, timeout=5, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
            return data["sha"]
    except Exception as e:
        update_status(f"Impossible de vérifier la version distante : {e}")
        return None


def get_local_sha() -> str | None:
    """Lit le SHA de la dernière version installée."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    return None


def download_and_extract_zip():
    """Télécharge le ZIP du repo et extrait son contenu dans APP_DIR."""
    zip_path = BASE_DIR / "repo_update.zip"
    tmp_dir  = BASE_DIR / "repo_tmp"

    update_status("Téléchargement de la mise à jour...")
    req = urllib.request.Request(
        ZIP_URL,
        headers={"Accept": "application/vnd.github.v3+json",
                 "User-Agent": "EIS-Analyzer-Launcher"}
    )
    with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp, open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    update_status("Extraction...")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # GitHub ZIP crée toujours 1 sous-dossier : "user-repo-sha/"
    # rglob pour trouver app.py quelle que soit la profondeur
    app_candidates = list(tmp_dir.rglob("app.py"))
    if not app_candidates:
        all_files = [str(p.relative_to(tmp_dir)) for p in tmp_dir.rglob("*")]
        raise RuntimeError(
            "app.py introuvable dans le ZIP.\nContenu :\n" + "\n".join(all_files[:30])
        )

    repo_root = app_candidates[0].parent
    update_status(f"Racine détectée : {repo_root.name}")

    if APP_DIR.exists():
        shutil.rmtree(APP_DIR)
    shutil.copytree(repo_root, APP_DIR)

    zip_path.unlink(missing_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    update_status(f"eis_app/ créé : {APP_DIR.exists()} — app.py : {APP_ENTRY.exists()}")


def check_and_update():
    """Compare les SHA local et distant, télécharge si nécessaire."""
    update_status("Vérification de la connexion...")
    online = is_online()
    update_status(f"Connexion : {'OK' if online else 'hors ligne'}")

    if not online:
        update_status("Hors ligne — utilisation de la version locale.")
        return

    update_status("Vérification de la version distante...")
    remote_sha = get_remote_sha()
    update_status(f"SHA distant : {remote_sha[:8] if remote_sha else 'inconnu'}")

    if remote_sha is None:
        update_status("Impossible de vérifier — utilisation version locale.")
        return

    local_sha = get_local_sha()
    update_status(f"SHA local : {local_sha[:8] if local_sha else 'aucun'}")

    if local_sha == remote_sha and APP_DIR.exists():
        update_status("Application déjà à jour.")
        return

    if not APP_DIR.exists():
        update_status("Première utilisation — téléchargement...")
    else:
        update_status("Nouvelle version — mise à jour...")

    try:
        download_and_extract_zip()
        VERSION_FILE.write_text(remote_sha)
    except Exception as e:
        update_status(f"Erreur téléchargement : {e}")
        if not APP_DIR.exists():
            show_error_and_exit(
                f"Impossible de télécharger l'application.\n\nErreur : {e}\n\n"
                "Vérifiez votre connexion internet et réessayez."
            )


def python_exe() -> Path:
    """Retourne le chemin de python dans le venv."""
    p = VENV_DIR / "Scripts" / "python.exe"   # Windows
    if not p.exists():
        p = VENV_DIR / "bin" / "python"         # Linux/Mac
    return p


def pip_exe() -> Path:
    p = VENV_DIR / "Scripts" / "pip.exe"
    if not p.exists():
        p = VENV_DIR / "bin" / "pip"
    return p


def streamlit_exe() -> Path:
    p = VENV_DIR / "Scripts" / "streamlit.exe"
    if not p.exists():
        p = VENV_DIR / "bin" / "streamlit"
    return p


def ensure_venv():
    if not VENV_DIR.exists():
        update_status("Création de l'environnement Python (une seule fois)...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)


def pip_install():
    req_file = APP_DIR / "requirements.txt"
    if not req_file.exists():
        update_status("Avertissement : requirements.txt introuvable.")
        return
    update_status("Vérification des dépendances...")
    subprocess.run(
        [str(pip_exe()), "install", "-r", str(req_file),
         "--quiet", "--disable-pip-version-check"],
        check=True
    )


def wait_for_streamlit(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def launch_streamlit() -> subprocess.Popen:
    return subprocess.Popen(
        [str(streamlit_exe()), "run", str(APP_ENTRY),
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def main():
    update_status("Démarrage de EIS Analyzer...")

    # 1. Mise à jour si en ligne
    check_and_update()

    # Sécurité : si eis_app/ absent malgré tout, retenter le téléchargement
    if not APP_DIR.exists():
        update_status("eis_app/ absent — nouvelle tentative de téléchargement...")
        try:
            download_and_extract_zip()
            remote_sha = get_remote_sha()
            if remote_sha:
                VERSION_FILE.write_text(remote_sha)
        except Exception as e:
            show_error_and_exit(
                f"Impossible de créer eis_app/.\n\nErreur : {e}\n\n"
                "Vérifiez votre connexion internet."
            )

    # Debug : lister le contenu de APP_DIR
    if APP_DIR.exists():
        contents = [str(p.name) for p in APP_DIR.iterdir()]
        update_status(f"Contenu eis_app/ : {contents[:10]}")

    # 2. Vérifier que l'app existe
    if not APP_ENTRY.exists():
        if APP_DIR.exists():
            py_files = [str(p.relative_to(APP_DIR)) for p in APP_DIR.rglob("*.py")]
            detail = "Fichiers .py trouvés :\n" + "\n".join(py_files[:20])
        else:
            detail = "Le dossier eis_app/ n'existe pas."
        show_error_and_exit(
            f"Application introuvable : {APP_ENTRY}\n\n{detail}\n\n"
            "Vérifiez votre connexion internet."
        )

    # 3. Environnement Python + dépendances
    ensure_venv()
    pip_install()

    # 4. Lancement Streamlit
    update_status("Lancement de l'interface...")
    proc = launch_streamlit()

    # 5. Attendre Streamlit puis ouvrir le navigateur
    update_status("Ouverture du navigateur...")
    if wait_for_streamlit(PORT, TIMEOUT_START):
        webbrowser.open(f"http://localhost:{PORT}")
        update_status("Application prête !")
    else:
        webbrowser.open(f"http://localhost:{PORT}")

    # 6. Rester en vie tant que Streamlit tourne
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
