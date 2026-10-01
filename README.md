# EIS Analyzer

Application web locale (Python/Streamlit) pour l'analyse de spectres
d'impédance électrochimique (EIS) appliquée à des biosenseurs microfluidiques
ADN/ARN.

## Démarrage rapide

### Windows (utilisateur) — un seul fichier

1. Installez **Python 3.12 ou plus récent** depuis https://www.python.org/downloads/
   (cochez *Add python.exe to PATH* ; le stub du Microsoft Store ne convient pas).
2. Téléchargez [`launch.bat`](launch.bat), placez-le dans un **dossier court** (par exemple
   `C:\EIS`) et double-cliquez dessus.

C'est tout : le dépôt est public, aucun jeton n'est nécessaire. À chaque lancement, `launch.bat`

* vérifie la dernière version sur GitHub (validation TLS normale) et ne la télécharge que si
  elle a changé ; l'ancienne version n'est remplacée qu'**après** extraction et contrôle de la
  nouvelle ;
* **hors ligne, ou si cette vérification échoue pour quelque raison que ce soit**, il lance
  directement la version déjà installée, sans rien tenter d'autre (le message distingue
  « HORS LIGNE » d'un « ECHEC REEL ») ;
* crée l'environnement Python (`venv\`) et n'installe les dépendances (versions épinglées) que
  si elles ont changé ;
* installe au **premier lancement** le moteur DRT (CmdStan 2.36.0 dans `C:\cmdstan` et
  compilation des modèles Stan : plusieurs minutes, une seule fois ; les modèles compilés
  survivent aux mises à jour). Si cette étape échoue, l'application démarre sans DRT ;
* ouvre le navigateur sur http://localhost:8501 (ou le premier port libre jusqu'à 8510)
  **seulement quand le serveur répond**.

Fermez la fenêtre noire pour arrêter l'application. Le journal du lanceur est `launcher.log`.

### Développement (Linux, macOS, ou Windows sans le lanceur)

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt    # exécution + pytest
pip install -r requirements-drt.txt    # facultatif : DRT (cvxopt, cmdstanpy)
python setup_drt_bayesien.py --ensure  # facultatif : CmdStan + modèles Stan (idempotent)
streamlit run app.py
```

Sans l'extra DRT, l'application fonctionne : l'analyse par circuit reste complète et
l'onglet DRT l'indique. L'application elle-même n'installe jamais rien ; l'installation est
l'affaire de `launch.bat` ou de `setup_drt_bayesien.py`.

## Utilisation

1. **Importer** l'expérience (probe, concentrations, réplicats par électrode).
2. **Prétraiter** (exclusions de réplicats/points) et valider : le verdict
   Kramers-Kronig de chaque groupe est calculé à ce stade.
3. Page **EIS seule** : définir le circuit équivalent (expression, guess/bornes,
   paramètre cible) et le mode DRT, puis lancer l'analyse.
4. Explorer les onglets : Validation KK · Résultats par groupe · Courbes DRT ·
   Reconstructions Nyquist · Calibration, puis la page Export.

## Format des fichiers d'entrée

- CSV ou TXT exporté depuis EC-Lab (ou `.mpr` BioLogic, converti à l'import)
- Colonnes attendues : `frequency_hz`, `zreal_ohm`, `zimag_ohm`
- Séparateur auto-détecté : virgule, tabulation, point-virgule ou espace
- Correction automatique du signe de Zim (convention EC-Lab)

## Analyse EIS (`core/pipeline.py`)

Pour CHAQUE groupe de réplicats (probe, chaque concentration), dans cet ordre :

| Étape | Module | Ce qui est produit |
|-------|--------|--------------------|
| 1. Measurement model + Kramers-Kronig | `core/measurement_model.py` | structure d'erreur σ(ω) et verdict KK, sur les réplicats **bruts**, avant tout fit ([docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)) |
| 2. Fit Orazem du circuit utilisateur | `fits/orazem_fit.py`, `circuit/` | CNLS pondéré 1/σ² de **chaque réplicat** et de la **moyenne** (σ/√n) ; paramètre cible agrégé : incertitude intra-fit vs variabilité inter-réplicats ([docs/CIRCUIT_UTILISATEUR.md](docs/CIRCUIT_UTILISATEUR.md)) |
| 3. DRT bayésienne | `drt/engine.py` (clone identifié de [bayes-drt2](https://github.com/jdhuang-csm/bayes-drt2), `drt/PROVENANCE.md`) | γ(τ) de **chaque réplicat** et de la **moyenne** ; Rct = arc de transfert (pic pénultième, convention Bissessur) ; mode `optimize` (MAP, ~1 s) ou `sample` (HMC, R̂/divergences/ESS + intervalles, 2 à 5 min par spectre) |

Un groupe dont la structure d'erreur n'est pas caractérisable (moins de 3
réplicats…) est **arrêté** avec un message explicite : aucun fit, aucune pondération
arbitraire.

Le calcul DRT compile des modèles Stan via CmdStan : la toolchain C++ est installée par
`launch.bat` au premier lancement (`install_cmdstan(compiler=True)`, sans exiger RTools), puis
mise en cache — voir [docs/DRT_BAYESIENNE.md](docs/DRT_BAYESIENNE.md) pour lire les intervalles de
crédibilité et les diagnostics. La figure DRT trace `ln(γ/γ₀)` en fonction
de `ln(τ/τ₀)` (logarithme népérien, γ₀ = 1 Ω, τ₀ = 1 s), avec un badge indiquant le
mode (`optimize`/`sample`). Référence DRT : Bissessur, Man, Gamby, *Use of an
approach with a distribution of relaxation times for impedance analysis of a
channel electrode in microfluidics*, Phys. Rev. E **113**, 025502 (2026), DOI:
10.1103/fn2s-z364.

## Tests

```bash
python -m pytest tests/ -v                # ajouter -m "not slow" pour exclure le HMC lent
```

## Documentation

| Document | Contenu |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | arborescence, flux de données, règles de modularité, lanceur |
| [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md) | measurement model, structure d'erreur d'Orazem, Kramers-Kronig, fit |
| [docs/CIRCUIT_UTILISATEUR.md](docs/CIRCUIT_UTILISATEUR.md) | écrire son propre circuit équivalent |
| [docs/DRT_BAYESIENNE.md](docs/DRT_BAYESIENNE.md) | DRT, HMC/NUTS, lecture des intervalles de crédibilité |
| [drt/PROVENANCE.md](drt/PROVENANCE.md), [drt/VALIDATION_REGLAGES.md](drt/VALIDATION_REGLAGES.md) | origine du code DRT, validation des réglages |

Les anciens rapports d'audit sont dans l'historique git : `git show 60371c7:AUDIT.md`.
