# EIS Analyzer — Launcher (sans Git)

Ce launcher permet de distribuer EIS Analyzer comme un **exécutable autonome** (.exe) qui se met à jour automatiquement via l'API GitHub, sans nécessiter Git sur la machine de l'utilisateur.

## Architecture

```
launcher/
├── launcher.py                  ← Script principal
├── splash.py                    ← Fenêtre tkinter de statut
├── build.py                     ← Génération du .exe via PyInstaller
├── requirements_launcher.txt
└── README_build.md
```

## Construire le .exe

```bash
cd launcher
pip install pyinstaller
python build.py
```

Le fichier `dist/EIS_Analyzer.exe` est prêt à être distribué.

## Comportement

| Situation | Comportement |
|-----------|-------------|
| Premier lancement, en ligne | Télécharge le ZIP → extrait → installe deps → lance |
| Lancement suivant, même version | "Déjà à jour" → lance directement |
| Lancement suivant, nouvelle version | Re-télécharge le ZIP → remplace → lance |
| Hors ligne, app présente | Lance la version locale sans mise à jour |
| Hors ligne, première utilisation | Message d'erreur explicite |

## Publier sur GitHub Releases

1. Aller sur `https://github.com/J-Facc/Interface-EIS-CV-measurement/releases`
2. "Draft a new release" → tag `v1.0.0`
3. Uploader `launcher/dist/EIS_Analyzer.exe`
4. Publier

## Zéro dépendance externe

- Pas de Git requis
- Pas d'installation requise par l'utilisateur final
- Python est embarqué dans le .exe via PyInstaller
