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
6. Explorer les onglets : Validation KK · Courbes DRT · Reconstructions Nyquist · Calibration · Export.

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
| DRT Tikhonov + NNLS (`drt_tikhonov`) | **DRT principale**, model-free : appliquée directement sur les données expérimentales déposées via l'import, λ sélectionné automatiquement par L-curve, γ(τ) ≥ 0 par NNLS |
| DRT FFT/Wiener spectre idéal (`drt_fft_ideal`) | Outil d'étude théorique des lois d'échelle MAD — recalcule la DRT exacte du modèle Randles déjà fitté, n'analyse pas les données brutes de façon indépendante |

DRT Tikhonov et DRT FFT/Wiener implémentent les deux méthodes distinctes
décrites dans Bissessur, Man, Gamby, *Use of an approach with a distribution
of relaxation times for impedance analysis of a channel electrode in
microfluidics*, Phys. Rev. E **113**, 025502 (2026), DOI: 10.1103/fn2s-z364
(sections III.B et III.C respectivement). L'onglet "Reconstructions Nyquist"
compare Rct_randles et Rct_drt (paramètre et reconstruction), l'onglet
"Calibration" trace une régression log(Rct) vs log([c]) séparée pour chaque
méthode.

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le détail des modules et des flux de données.
