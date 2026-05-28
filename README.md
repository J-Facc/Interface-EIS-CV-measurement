# EIS Analyzer

Application web locale (Python/Streamlit) pour l'analyse de spectres
d'impédance électrochimique (EIS) appliquée à des biosenseurs microfluidiques
ADN/ARN.

## Démarrage rapide

### Windows
```bat
launch_app.bat
```

### Linux / macOS
```bash
chmod +x launch_app.sh
./launch_app.sh
```

L'application s'ouvre automatiquement sur **http://localhost:8501**.

## Installation manuelle

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Utilisation

1. **Importer** vos fichiers CSV/TXT dans la sidebar.
2. **Assigner** chaque fichier à une étape : `bare`, `probe` ou `hybridization`.
3. Pour les fichiers `hybridization`, saisir la concentration en mantisse × 10ˣ M.
4. **Sélectionner** les modèles de fit.
5. Cliquer sur **▶ Analyser**.
6. Explorer les onglets : Nyquist · Bode · DRT · Paramètres · Calibration · Export.

## Format des fichiers d'entrée

- CSV ou TXT exporté depuis EC-Lab (format `.mpt` en CSV)
- Colonnes attendues : `frequency_hz`, `zreal_ohm`, `zimag_ohm`
- Séparateur auto-détecté : virgule, tabulation, point-virgule ou espace
- Correction automatique du signe de Zim (convention EC-Lab)

## Modèles de fit

| Modèle | Description |
|--------|-------------|
| Fit circulaire | Lecture géométrique — aucun paramètre physique |
| Randles contraint | Re fixé, ZD0 ∝ Fv^(−1/3), 3 paramètres libres |
| Randles complet | 8 paramètres libres, pondération Modulus |
| DRT Tikhonov | Distribution des temps de relaxation, λ auto |

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le détail des modules et des flux de données.
