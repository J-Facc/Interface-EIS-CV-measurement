"""
Génère EIS_Analyzer.exe avec PyInstaller.
Exécuter une seule fois :
    pip install pyinstaller
    python build.py

Le .exe produit est dans dist/EIS_Analyzer.exe
"""
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--noconsole",
    "--name", "EIS_Analyzer",
    "--add-data", "splash.py;.",
    "launcher.py"
], check=True)

print("\n✅ dist/EIS_Analyzer.exe prêt.")
print("   1. Uploader sur GitHub Releases.")
print("   2. Placer sur le bureau — aucune autre installation requise.")
