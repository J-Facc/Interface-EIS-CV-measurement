# Architecture — EIS Analyzer

## Vue d'ensemble

EIS Analyzer est une application web locale (Python/Streamlit) pour l'analyse
de spectres d'impédance électrochimique (EIS) appliquée à des biosenseurs
microfluidiques ADN/ARN.

## Structure des modules

```
eis_analyzer/
├── app.py                   Point d'entrée Streamlit unique
├── launch_app.bat/.sh       Lanceurs automatiques (venv + pip + streamlit)
├── requirements.txt
├── config/
│   └── default.yaml         Paramètres physiques, géométrie, fit, export
├── core/                    Logique métier — AUCUN import Streamlit
│   ├── config.py            Chargement YAML + validation Pydantic
│   ├── models.py            EISSpectrum, FitResult, ConcentrationGroup, EISSession
│   ├── loader.py            Import CSV/TXT, détection auto, validation, moyenne
│   ├── logger.py            Logging centralisé (fichier + console)
│   └── pipeline.py          Orchestrateur Import → Fit → EISSession
├── fits/                    Système plugin — 1 fichier = 1 modèle
│   ├── base.py              BaseFitModel (ABC) : interface commune
│   ├── physics.py           Fonctions physiques partagées (ZD, Randles, etc.)
│   ├── registry.py          Découverte et instanciation automatiques
│   ├── circular_fit.py      Fit géométrique circulaire
│   ├── randles_constrained.py  Randles 3 paramètres libres
│   ├── randles_full.py      Randles 8 paramètres libres
│   └── drt_tikhonov.py      DRT par régularisation Tikhonov
├── plotting/
│   ├── theme.py             Palettes jour/nuit + apply_theme_to_figure
│   └── eis_plots.py         7 graphes Plotly (Nyquist, Bode, DRT, table, calibration)
├── exports/
│   └── exporter.py          CSV, PNG, HTML, YAML session
├── ui/
│   ├── sidebar.py           Upload, assignation, modèles, paramètres physiques
│   └── tabs.py              6 onglets d'analyse
└── tests/
    ├── test_loader.py
    ├── test_physics.py
    └── test_fits.py
```

## Règles d'architecture

1. **Séparation des responsabilités** : `core/` ne contient jamais d'imports Streamlit.
   `fits/` ne contient jamais d'imports UI. `plotting/` ne modifie jamais l'état.

2. **Système plugin** : ajouter un modèle de fit = créer un fichier dans `fits/`
   qui sous-classe `BaseFitModel`. Aucun autre fichier n'est modifié. La découverte
   est automatique via `fits/registry.py`.

3. **État centralisé** : tout l'état de l'application est dans
   `st.session_state['session']` (objet `EISSession`). Pas de variables globales
   mutables.

4. **Configuration YAML** : toutes les constantes physiques et les bornes de fit
   sont dans `config/default.yaml`, validées par Pydantic (`core/config.py`).

## Flux de données

```
Fichiers CSV  →  core/loader.py  →  EISSpectrum
                                      ↓
                               core/pipeline.py
                                      ↓
                    fits/*.py  →  FitResult (par modèle)
                                      ↓
                               EISSession
                                      ↓
                   ui/tabs.py  →  plotting/eis_plots.py
                                      ↓
                              Graphes Plotly / Export
```

## Modèles de fit

| Nom | Paramètres libres | Méthode |
|-----|-------------------|---------|
| `circular` | 0 (géométrique) | Moindres carrés algébriques |
| `randles_constrained` | 3 (Rct, Qdl, α) | TRF scipy |
| `randles_full` | 8 | TRF scipy, pondération Modulus |
| `drt_tikhonov` | grille τ | Tikhonov + L-curve |

## Physique implémentée

Toutes les formules sont dans `fits/physics.py` et suivent :
- Poujouly (2022) pour l'impédance de diffusion-convection en microcanal
- Deslouis et al. pour les régimes LF/HF unifiés
- Brug (1984) pour la capacité de double couche effective (CPE)
