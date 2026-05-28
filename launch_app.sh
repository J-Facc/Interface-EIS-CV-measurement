#!/bin/bash
set -e

VENV_DIR=".venv"
PORT=8501

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/3] Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[2/3] Vérification des dépendances..."
pip install -r requirements.txt --quiet

echo "[3/3] Lancement de l'application sur http://localhost:$PORT"
streamlit run app.py --server.port "$PORT" --browser.gatherUsageStats false
