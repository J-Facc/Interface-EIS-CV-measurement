# EIS Analyzer — Architecture & Contexte général
## Document de référence — Maintenance & Développement

> Repo : https://github.com/J-Facc/Interface-EIS-CV-measurement
> Navigation multipage `st.navigation` (pages/) — EIS **et** CV.
> Dernière mise à jour : 01/10/2026 (étape 5 de la refonte : pipeline, sens des dépendances)

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
├── app.py                             ← point d'entrée Streamlit (st.navigation, pages/)
├── requirements.txt                   ← dépendances dures
├── requirements-optional.txt          ← dépendances optionnelles (galvani, lecteur .mpr de secours)
├── launch.bat                         ← lanceur Windows (auto-update + venv)
├── .version                           ← SHA GitHub du dernier update local
│
├── config/
│   └── default.yaml                   ← paramètres physiques et de fit (bornes typées list[float])
│
├── core/                              ← logique métier et orchestration (JAMAIS d'import Streamlit)
│   ├── models.py                      ← EISSpectrum (porte ses réplicats bruts), GroupAnalysis,
│   │                                     ConcentrationGroup, EISSession ; ré-exporte FitResult
│   ├── pipeline.py                    ← orchestrateur EIS : MM + KK → fit Orazem → DRT, par groupe
│   ├── measurement_model.py           ← measurement model de Voigt : structure d'erreur + KK
│   ├── validator.py                   ← verdict KK affiché (ValidationResult) + dérive
│   ├── results_table.py               ← tables intra-fit / inter-réplicats / diagnostics HMC (UI + export)
│   ├── app_state.py                   ← clés session_state partagées et leur réinitialisation (B-STATE)
│   ├── loader.py · robust_loader.py   ← import EIS EC-Lab
│   ├── config.py                      ← YAML + Pydantic (circuit par défaut, DRT)
│   ├── calibration.py                 ← calibration EIS/CV — source unique
│   ├── cv_*.py                        ← CV (chargement, pics, pipeline)
│   ├── experiment_io.py · mpr_converter.py · logger.py
│
├── fits/                              ← bibliothèque numérique du fit (n'importe JAMAIS core)
│   ├── result.py                      ← FitResult (contrat commun fit Orazem / DRT)
│   ├── orazem_fit.py                  ← fit Orazem d'un circuit LIBRE (par réplicat + moyenne, agrégation)
│   ├── regression_stats.py            ← covariance par SVD, conditionnement, identifiabilité
│   ├── kk_validation.py               ← critère KK unique (kk_verdict), Lin-KK indicatif
│   └── physics.py                     ← Z_D (référence de ZD_bounded), θ_EIS…
│
├── drt/                               ← DRT (n'importe JAMAIS core ; seulement fits.result)
│   ├── engine.py                      ← moteur durci (HMC/MAP, gardes, diagnostics)
│   ├── diagnostics.py                 ← seuils R-hat / ESS / divergences / qualité
│   └── bayes_drt2/                    ← clone identifié de bayes-drt2 (PROVENANCE.md)
│
├── circuit/                           ← circuit utilisateur (Python restreint, liste blanche AST)
│
├── plotting/                          ← reçoit des données, ne calcule rien
│   ├── __init__.py
│   ├── theme.py                       ← palettes clair / sombre (dark défini mais non câblé aujourd'hui)
│   ├── eis_plots.py                   ← figures Plotly EIS (+ fenêtres matplotlib)
│   ├── cv_plots.py                    ← figures CV
│   └── kk_plots.py                    ← figures validation KK
│
├── exports/                          ← ⚠️ paquet source versionné (PAS ignoré par .gitignore)
│   ├── __init__.py
│   └── exporter.py                    ← export CSV, PNG, HTML, YAML, ZIP session
│
├── ui/
│   ├── __init__.py
│   └── tabs.py                        ← render_eis_tabs / render_cv_tabs (appelés par les pages)
│
├── pages/                             ← pages st.navigation
│   ├── __init__.py
│   ├── 0_import.py                    ← import de l'expérience
│   ├── 1_pretraitement.py             ← exclusions/nettoyage → experiment_clean
│   ├── A_eis.py                       ← analyse EIS (fits, KK, calibration, diagnostics)
│   ├── B_cv.py                        ← analyse CV
│   ├── D_inference.py                 ← inférence/prédiction
│   └── E_export.py                    ← export
│
├── tests/                            ← pytest (fits, physics, loader, cv, drt, config, calibration, diagnostics…)
├── logs/  · sessions/                 ← créés automatiquement (ignorés par git)
└── .github/workflows/validate.yml     ← CI : compileall + pytest (Python 3.12)
```

---

## 2b. Navigation v3 — Structure des pages

```
app.py  →  st.navigation({
  "Données":   [0_import.py, 1_pretraitement.py],
  "Analyse":   [A_eis.py,    B_cv.py           ],
  "Export":    [E_export.py                     ],
  "Inférence": [D_inference.py                  ],
})
```
> Il n'y a pas de page « Comparaison » / `C_comparatif.py`. Les onglets EIS/CV
> sont rendus par `ui/tabs.py::render_eis_tabs` / `render_cv_tabs`, appelés
> depuis `pages/A_eis.py` et `pages/B_cv.py` (il n'existe plus de `render_tabs`).

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

### Sens des dépendances (AUDIT.md CPL-1)

```
pages / ui / plotting / exports ──► core ──► drt ──► fits.result
                                     │                  ▲
                                     └────► fits ───────┘
                                             └──► circuit (feuille)
```

| Couche | Règle absolue |
|--------|--------------|
| `core/` | Jamais d'import Streamlit. Importe `fits`/`drt`/`circuit` **au niveau module** (aucun import local masquant un cycle) |
| `fits/`, `drt/`, `circuit/` | N'importent **jamais** `core`, `ui`, `pages`, `plotting`, `exports` ni Streamlit. `drt` n'importe de `fits` que `fits.result` ; `circuit` n'importe rien du projet |
| `plotting/` | Reçoit des données, ne les calcule pas |
| `ui/` | Appelle `core/`, n'implémente pas de physique |
| `exports/` | Doit avoir un `__init__.py` sinon Python ne le trouve pas |

`tests/test_architecture.py` vérifie ces règles sur l'arbre d'imports réel.

Il n'y a plus de registre de plugins (`fits/registry.py`, `fits/base.py` supprimés à
l'étape 5) : un seul moteur de fit (Orazem sur circuit utilisateur), une seule DRT,
appelés explicitement par `core/pipeline.py`.

---

## 4. Flux de données

```
Fichiers CSV/TXT (EC-Lab export)
        ↓
core/loader.py → réplicats BRUTS (EISSpectrum), conservés à chaque étape
        ↓  (par groupe : probe, chaque concentration — et bare si fourni)
core/measurement_model.analyze_replicates   ← sur les réplicats BRUTS, AVANT tout fit
  • σ(ω) = structure d'erreur d'Orazem du groupe
  • verdict Kramers-Kronig (réplicats + moyenne)      → ValidationResult (onglet KK)
  • ErrorStructureUnavailable → groupe ARRÊTÉ (GroupAnalysis.status, message affiché)
        ↓
fits/orazem_fit.fit_replicate_group
  • fit de CHAQUE réplicat (σ) et de la moyenne (σ/√n) du circuit utilisateur
  • agrégation : moyenne, s inter-réplicats, √v̄ intra-fit, incertitude de la moyenne, Q de Cochran
        ↓
drt/engine.fit_drt sur CHAQUE réplicat ET la moyenne (mode fit.drt.mode)
  • Rct DRT agrégé sur les réplicats ; recompute_drt(…, mode='sample') à la demande
        ↓
core/models.py → EISSession { probe, groups[], *_replicate_spectra, *_analysis (GroupAnalysis) }
        ↓
core/results_table.py → tables intra/inter + diagnostics HMC   ·   core/calibration.py → régressions
        ↓
ui/tabs.py::render_eis_tabs (pages/A_eis.py)   ·   exports/exporter.py (pages/E_export.py)
```

Erreurs : saisie invalide → `InvalidAnalysisInput` avant tout calcul ; donnée invalide →
résultat dégradé + message rangé dans la session (`load_errors`, `GroupAnalysis.status/
message/drt_failures`, `messages`) ; bug → l'exception remonte (affichée comme erreur
logicielle par la page EIS). Détail : docstring de `core/pipeline.py`.

---

## 5. Le fit (`fits/orazem_fit.py`) et la DRT (`drt/engine.py`)

| Méthode | Paramètres | Sortie (`fit_results` de chaque spectre) |
|---------|-----------|-------------------------------------------|
| Fit Orazem, clé `"orazem"` | circuit utilisateur (`fit.circuit` par défaut : Randles complet à 8 paramètres) ; guess/bornes par paramètre ; paramètre cible désigné | `FitResult` : `target_param/target_value/target_std` (intra-fit), χ²ᵣ et son intervalle attendu, `fit_diagnostics` (κ, rang, identifiabilité, bornes actives) |
| DRT, clé `"drt_bayes"` | réglages documentés de `drt/engine.py` (`drt/VALIDATION_REGLAGES.md`) ; mode `fit.drt.mode` | `FitResult` : γ(τ), Rct (pic pénultième, ±3 en ln τ), `target_std` a posteriori (HMC), `drt_diagnostics` (R̂, ESS, divergences, alertes) |

Méthode d'Orazem et sources : [MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md). Circuit
utilisateur : [docs/CIRCUIT_UTILISATEUR.md](docs/CIRCUIT_UTILISATEUR.md). DRT :
`drt/PROVENANCE.md`, `drt/VALIDATION_REGLAGES.md`.

**Convention d'impédance** : le loader stocke `Zim = -Im(Z) > 0` ; bayes_drt2 attend
`Z'' < 0`, d'où `Z = Zre - 1j·Zim` dans `drt/engine.py`.

---

## 6. Onglets de l'analyse EIS (`ui/tabs.py::render_eis_tabs`)

En tête de la page EIS : statut explicite de l'analyse (groupes arrêtés, fichiers
écartés, moteur DRT absent — jamais « ✅ Analyse terminée » si un groupe n'a aucun fit)
et diagnostics d'ajustement.

| Onglet | Contenu |
|--------|---------|
| **Validation KK** | Verdict du measurement model par groupe (calculé avant le fit), résidus |
| **Résultats par groupe** | Par groupe, côte à côte : circuit (valeur ± intra-fit par réplicat, χ²ᵣ) et DRT (Rct ± a posteriori, **R̂ max, divergences, ESS**) ; agrégats moyenne / inter-réplicats / intra-fit / incertitude de la moyenne ; diagnostics du fit de la moyenne |
| **Courbes DRT** | ln(γ/γ₀) vs ln(τ/τ₀) des moyennes et des réplicats d'un groupe ; diagnostics HMC par spectre ; recalcul bayésien d'un spectre (`core.pipeline.recompute_drt`) |
| **Reconstructions Nyquist** | Paramètre cible circuit vs DRT ; reconstructions moyenne et réplicat |
| **Calibration** | Signal normalisé vs log([c]) + régression (via `core/calibration.py`), une courbe par méthode |
| **Export** (page dédiée `E_export.py`) | paramètres (colonnes par modèle), résultats par réplicat/groupe, DRT, ZIP |

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
| `OSError: No such file or directory ... streamlit` | Chemin trop long (MS Store Python) | Utiliser venv local (chemin court) |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Proxy d'entreprise | `ssl.CERT_NONE` dans les requêtes urllib |
| Fenêtre .bat qui se ferme | Variable `%VAR%` non évaluée après `cd` | Utiliser `!VAR!` + `setlocal enabledelayedexpansion` |

> **Historique** — le `ModuleNotFoundError: No module named 'exports'`, longtemps
> « soigné » en recréant `exports/__init__.py`, avait pour cause réelle une ligne
> `exports/` dans `.gitignore` : le paquet source était ignoré, donc absent du ZIP
> téléchargé par `launch.bat`. Corrigé (`.gitignore` nettoyé) — le rituel n'a plus
> lieu d'être. De même, `render_tabs` n'existe plus (voir `ui/tabs.py`).

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
