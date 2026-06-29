# EIS Analyzer — Architecture & Contexte général
## Document de référence — Maintenance & Développement

> Version courante : v3 — Repo : https://github.com/J-Facc/Interface-EIS-CV-measurement
> Dernière mise à jour : 09/06/2026

---

## 1. Objectif de l'interface

EIS Analyzer est une application web locale (Python/Streamlit) pour l'analyse de spectres d'impédance électrochimique (EIS) appliquée à des biosenseurs microfluidiques ADN/ARN sur électrodes Pt. Elle extrait la résistance de transfert de charge **Rct** par 4 méthodes comparatives et produit une courbe de calibration log(Rct) vs log([c]).

**Signal de détection :** Rct encode la concentration cible via :
```
θ_EIS = 1 − Rct,bare / Rct,ap
log(Rct_norm) = a × log([c]) + b       LOD ≈ 10⁻¹⁷ M
```

---

## 2. Architecture des fichiers

```
Interface-EIS-CV-measurement/          ← racine du repo GitHub
│
├── app.py                             ← point d'entrée Streamlit (navigation v3)
├── requirements.txt                   ← dépendances Python
├── launch.bat                         ← lanceur Windows (auto-update + venv)
├── .version                           ← SHA GitHub du dernier update local
│
├── config/
│   └── default.yaml                   ← tous les paramètres physiques et de fit
│
├── core/                              ← logique métier pure (JAMAIS d'import Streamlit)
│   ├── __init__.py
│   ├── models.py                      ← dataclasses : EISSpectrum, FitResult, EISSession
│   ├── loader.py                      ← import CSV/TXT, validation, moyennage réplicats
│   ├── pipeline.py                    ← orchestrateur Import → Fit → Analyse
│   ├── config.py                      ← chargement YAML + Pydantic AppSettings
│   ├── logger.py                      ← logging centralisé
│   ├── validator.py                   ← validation KK (lin-KK via impedance.py)
│   ├── cv_loader.py                   ← chargement fichiers CV
│   ├── cv_models.py                   ← CVScan, CVConcentrationGroup
│   ├── cv_pipeline.py                 ← pipeline traitement CV
│   └── experiment_io.py               ← sauvegarde/chargement session complète (ZIP)
│
├── fits/                              ← système plugin : 1 fichier = 1 modèle
│   ├── __init__.py
│   ├── base.py                        ← BaseFitModel (ABC) — interface commune
│   ├── physics.py                     ← fonctions physiques partagées (ZD, alpha_h, Brug)
│   ├── registry.py                    ← FitRegistry : découverte automatique des modèles
│   ├── randles_full.py                ← Randles complet (8 paramètres, least_squares)
│   ├── drt_tikhonov.py                ← DRT model-free (Tikhonov ordre 2 + NNLS) — DRT principale
│   ├── drt_fft.py                     ← DRT FFT/Wiener sur spectre IDÉAL Randles (étude MAD, pas indépendante)
│   └── kk_validation.py               ← validation Kramers-Kronig (Lin-KK, circuits de Voigt)
│
├── plotting/
│   ├── __init__.py
│   ├── theme.py                       ← palettes jour / nuit
│   └── eis_plots.py                   ← 7 figures Plotly interactives
│
├── exports/
│   ├── __init__.py                    ← ⚠️ obligatoire sinon ModuleNotFoundError
│   └── exporter.py                    ← export CSV, PNG, HTML, YAML session
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py                     ← upload fichiers + saisie concentrations + toggle thème
│   └── tabs.py                        ← 6 onglets (render_tabs)
│
├── tests/
│   ├── __init__.py
│   └── test_fits.py                   ← tests pytest
│
├── logs/                              ← créé automatiquement
├── sessions/                          ← créé automatiquement
└── .github/
    └── workflows/
        └── validate.yml               ← CI : syntax check à chaque push
```

---

## 2b. Navigation v3 — Structure des pages

```
app.py  →  st.navigation({
  "Données":     [0_import.py, 1_pretraitement.py],
  "Analyse":     [A_eis.py,    B_cv.py           ],
  "Comparaison": [C_comparatif.py                 ],
  "Inférence":   [D_inference.py                  ],
})
```

### Flux obligatoire

```
pages/0_import.py
  → st.session_state['experiment']
  → st.session_state['import_validated'] = True

pages/1_pretraitement.py
  → st.session_state['exclusions']
  → st.session_state['experiment_clean']

pages/A_eis.py, B_cv.py, C_comparatif.py, D_inference.py
  ← lisent st.session_state['experiment_clean']
  ← affichent st.warning + st.stop() si experiment_clean absent
```

### Structure de st.session_state['experiment']

```python
{
  "name":         str,
  "date":         str,           # YYYY-MM-DD
  "mode":         "eis_only" | "cv_only" | "both",
  "concentrations": [float, ...],
  "n_electrodes": int,
  "n_replicats":  int,
  "probe": {
    "eis": {"electrode_1": [BytesIO, ...], "electrode_2": [...]},
    "cv":  {"electrode_1": [BytesIO, ...], ...},
  },
  "calibration": {
    "eis": {"electrode_1": [[BytesIO, ...], ...], ...},
    "cv":  {"electrode_1": [[BytesIO, ...], ...], ...},
  },
  "validation": { ... } | None,
}
```

### core/experiment_io.py

Sérialise/désérialise l'experiment dict en ZIP :
- `save_experiment(experiment) -> bytes`  (alias `zip_experiment`)
- `load_experiment(zip_bytes) -> dict`

Format ZIP : `experiment.yaml` + `probe/electrode_N/rep_R.bin` + `calibration/sig/electrode_N/conc_C_rep_R.bin`

---

## 3. Règles de modularité — CRITIQUES

| Couche | Règle absolue |
|--------|--------------|
| `core/` | Jamais d'import Streamlit |
| `fits/` | Jamais d'import UI ni Streamlit |
| `plotting/` | Reçoit des données, ne les calcule pas |
| `ui/` | Appelle `core/` et `fits/`, n'implémente pas de physique |
| `exports/` | Doit avoir un `__init__.py` sinon Python ne le trouve pas |

**Ajouter un modèle de fit** = créer un fichier dans `fits/` héritant de `BaseFitModel`. Aucun autre fichier à modifier.

---

## 4. Flux de données

```
Fichiers CSV/TXT (EC-Lab export)
        ↓
core/loader.py
  • auto-détection séparateur
  • correction signe Zim (convention EC-Lab)
  • suppression 50/100 Hz parasites
  • tri HF → BF
  • moyennage réplicats
        ↓
core/models.py → EISSpectrum { label, f[], Zre[], Zim[], concentration, step }
        ↓
core/pipeline.py → run_analysis()
        ↓
fits/ — 4 modèles en parallèle
  randles_classique   → FitResult { params, Zfit[], chi2, Rct }
  randles_contraint   → FitResult
  drt_fit             → FitResult
  circulaire_fit      → FitResult
        ↓
core/models.py → EISSession { bare, probe, groups[] }
        ↓
plotting/eis_plots.py → 7 figures Plotly
        ↓
ui/tabs.py → 6 onglets Streamlit
```

---

## 5. Les modèles de fit

| Modèle | Fichier | Paramètres libres | Méthode |
|--------|---------|-------------------|---------|
| Randles complet | `randles_full.py` | 8 (Re, R'e, Cb, Rct, Qdl, α, R_D, τ_d) | `scipy.optimize.least_squares` (pondération Modulus) |
| DRT Tikhonov + NNLS | `drt_tikhonov.py` | λ (régularisation, sélection auto par L-curve) | NNLS, **model-free** |
| DRT FFT/Wiener (spectre idéal) | `drt_fft.py` | hérités du fit Randles + filtre W | FFT + filtre Wiener, sur spectre **idéal** |
| Circulaire | `circular.py` (référencé dans `ui/sidebar.py`, non présent dans `fits/` à ce jour) | 0 — lecture géométrique | Fit circulaire algébrique |

**Circuit physique (Randles modifié) :**
```
Re — [ R'e // Cb ] — [ Rct // CPE(Qdl, α) ] — ZD(ω)
```

### Deux méthodes DRT distinctes — ne jamais les confondre

Référence : Bissessur, Man, Gamby, *Use of an approach with a distribution of
relaxation times for impedance analysis of a channel electrode in
microfluidics*, Phys. Rev. E **113**, 025502 (2026), DOI: 10.1103/fn2s-z364.

- **`fits/drt_tikhonov.py` (DRT principale, section III.B "DRT with
  DRTtools")** — model-free, appliquée DIRECTEMENT sur les données
  expérimentales brutes déposées via l'onglet Import (`pages/0_import.py`).
  Aucune hypothèse de circuit équivalent : grille τ log-espacée dimensionnée
  sur la plage de fréquences réelles, noyau de Fredholm discret,
  régularisation de Tikhonov d'ordre 2, λ sélectionné par L-curve, résolution
  NNLS (γ(τ) ≥ 0). C'est le modèle `"drt_tikhonov"` utilisé par défaut dans
  les graphes DRT, la reconstruction Nyquist et la calibration.

- **`fits/drt_fft.py` (modèle `"drt_fft_ideal"`, section III.C "DRT with
  DFT")** — reconstruit la DRT EXACTE d'un spectre **idéal**, c'est-à-dire le
  modèle Randles déjà fitté (`randles_full.py`) réévalué sur une grille
  log-ω dense, puis déconvolué par FFT/filtre Wiener. Cette méthode n'est
  PAS indépendante du fit Randles — elle sert exclusivement à l'étude
  théorique des lois d'échelle MAD (Maxima Asymptotic Dynamics) sur des cas
  contrôlés, pas à produire un Rct à comparer à Rct_randles.

### Comparaison Randles vs DRT (model-free) — trois niveaux

1. **Paramètre** — `plotting/eis_plots.py::params_table_figure` affiche, pour
   chaque étape (bare/probe/concentration), Rct_randles, Rct_drt et leur
   écart relatif. Branché dans l'onglet "3️⃣ Reconstructions Nyquist".
2. **Reconstruction Nyquist** — `reconstruction_comparison_figure` et
   `drt_reconstruction_figure_dual` superposent points expérimentaux,
   courbe Randles et courbe DRT Tikhonov pour un même spectre.
3. **Calibration** — `calibration_figure` trace une régression log-log
   distincte (avec son propre R²) par modèle présent dans la session,
   donc une courbe pour `randles_full` et une pour `drt_tikhonov`,
   visibles simultanément sur le même graphe avec légende séparée.

---

## 6. Les 6 onglets de l'interface

| Onglet | Contenu |
|--------|---------|
| **1️⃣ Validation KK** | Diagnostic Kramers-Kronig (Lin-KK) par réplicat |
| **2️⃣ Courbes DRT** | Distribution γ(τ) vs log(τ) — `drt_tikhonov` (model-free, principale) |
| **3️⃣ Reconstructions Nyquist** | Table Rct_randles vs Rct_drt (écart relatif) + Nyquist mesuré/Randles/DRT superposés |
| **4️⃣ Calibration** | log(Rct_norm) vs log([c]) + régression + R², une courbe par modèle (Randles et DRT séparées) |
| **Export** | CSV params, CSV spectres DRT, PNG, HTML, YAML session |

---

## 7. Launcher Windows (launch.bat)

**Fonctionnement à chaque double-clic :**
```
1. Vérifier SHA GitHub vs SHA local (.version)
   → identique + eis_app/ présent : sauter le téléchargement
   → différent ou absent : télécharger ZIP GitHub + extraire dans eis_app/
2. Créer venv/ si absent (une seule fois)
3. pip install requirements.txt
4. cd eis_app/ + streamlit run app.py
```

**Fichiers créés localement (à côté du .bat) :**
```
EIS-CV_analyzer/
├── launch.bat          ← seul fichier à distribuer
├── .version            ← SHA du dernier commit installé
├── venv/               ← environnement Python (créé automatiquement)
└── eis_app/            ← code de l'app (extrait du ZIP GitHub)
```

**Prérequis machine :** Python installé (Microsoft Store ou python.org)

---

## 8. Paramètres physiques (config/default.yaml)

| Symbole | Valeur | Unité | Description |
|---------|--------|-------|-------------|
| T | 298 | K | Température |
| D | 7.2×10⁻¹⁰ | m²/s | Diffusion Fe(CN)₆³⁻ |
| Fv | 0.5×10⁻⁹ | m³/s | Débit (0.5 µL/s) |
| h | 60×10⁻⁶ | m | Hauteur canal |
| d | 300×10⁻⁶ | m | Largeur canal |
| xe | 30×10⁻⁶ | m | Largeur électrode WE |
| S | 9×10⁻⁹ | m² | Surface active |
| C0 | 20 | mM | Concentration Fe(CN)₆ |

---

## 9. Erreurs fréquentes et solutions

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError: No module named 'exports'` | `exports/__init__.py` absent | Créer ce fichier vide |
| `ImportError: cannot import name 'render_tabs'` | Streamlit lancé hors de `eis_app/` | `cd /d "%APP_DIR%"` avant streamlit |
| `OSError: No such file or directory ... streamlit` | Chemin trop long (MS Store Python) | Utiliser venv local (chemin court) |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Proxy d'entreprise | `ssl.CERT_NONE` dans les requêtes urllib |
| Fenêtre .bat qui se ferme | Variable `%VAR%` non évaluée après `cd` | Utiliser `!VAR!` + `setlocal enabledelayedexpansion` |

---

## 10. Commandes utiles

```bash
# Lancer les tests
python -m pytest tests/ -v

# Vérifier la syntaxe de app.py
python -m py_compile app.py

# Forcer une mise à jour (supprimer le SHA local)
del .version

# Réinstaller le venv from scratch
rmdir /s /q venv
```

---

*Document rédigé le 29/05/2026 — à importer dans tout nouveau projet Claude pour la maintenance*
