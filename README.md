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

1. **Importer** l'expérience (probe, concentrations, réplicats par électrode).
2. **Prétraiter** (exclusions de réplicats/points) et valider : le verdict
   Kramers-Kronig de chaque groupe est calculé à ce stade.
3. Page **EIS seule** : définir le circuit équivalent (expression, guess/bornes,
   paramètre cible) et le mode DRT, puis lancer l'analyse.
4. Explorer les onglets : Validation KK · Résultats par groupe · Courbes DRT ·
   Reconstructions Nyquist · Calibration, puis la page Export.

## Format des fichiers d'entrée

- CSV ou TXT exporté depuis EC-Lab (format `.mpt` en CSV)
- Colonnes attendues : `frequency_hz`, `zreal_ohm`, `zimag_ohm`
- Séparateur auto-détecté : virgule, tabulation, point-virgule ou espace
- Correction automatique du signe de Zim (convention EC-Lab)

## Analyse EIS (`core/pipeline.py`)

Pour CHAQUE groupe de réplicats (probe, chaque concentration), dans cet ordre :

| Étape | Module | Ce qui est produit |
|-------|--------|--------------------|
| 1. Measurement model + Kramers-Kronig | `core/measurement_model.py` | structure d'erreur σ(ω) et verdict KK, sur les réplicats **bruts**, avant tout fit ([MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md)) |
| 2. Fit Orazem du circuit utilisateur | `fits/orazem_fit.py`, `circuit/` | CNLS pondéré 1/σ² de **chaque réplicat** et de la **moyenne** (σ/√n) ; paramètre cible agrégé : incertitude intra-fit vs variabilité inter-réplicats ([docs/CIRCUIT_UTILISATEUR.md](docs/CIRCUIT_UTILISATEUR.md)) |
| 3. DRT bayésienne | `drt/engine.py` (clone identifié de [bayes-drt2](https://github.com/jdhuang-csm/bayes-drt2), `drt/PROVENANCE.md`) | γ(τ) de **chaque réplicat** et de la **moyenne** ; Rct = arc de transfert (pic pénultième, convention Bissessur) ; mode `optimize` (MAP, ~1 s) ou `sample` (HMC, R̂/divergences/ESS + intervalles, 2 à 5 min par spectre) |

Un groupe dont la structure d'erreur n'est pas caractérisable (moins de 3
réplicats…) est **arrêté** avec un message explicite : aucun fit, aucune pondération
arbitraire.

Le calcul DRT compile des modèles Stan via CmdStan : la toolchain C++ est
installée automatiquement au premier lancement (`install_cmdstan(compiler=True)`,
sans exiger RTools), puis mise en cache. La figure DRT trace `ln(γ/γ₀)` en fonction
de `ln(τ/τ₀)` (logarithme népérien, γ₀ = 1 Ω, τ₀ = 1 s), avec un badge indiquant le
mode (`optimize`/`sample`). Référence DRT : Bissessur, Man, Gamby, *Use of an
approach with a distribution of relaxation times for impedance analysis of a
channel electrode in microfluidics*, Phys. Rev. E **113**, 025502 (2026), DOI:
10.1103/fn2s-z364.

## Tests

```bash
python -m pytest tests/ -v
```

## Architecture

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le détail des modules et des flux de données.
