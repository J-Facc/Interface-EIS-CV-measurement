# EIS Analyzer — Architecture & Contexte général
## Document de référence — Maintenance & Développement

> Repo : https://github.com/J-Facc/Interface-EIS-CV-measurement
> Navigation multipage `st.navigation` (pages/) — EIS **et** CV.
> Dernière mise à jour : 13/07/2026 (réalignée sur l'état réel du code — audit)

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
├── core/                              ← logique métier pure (JAMAIS d'import Streamlit)
│   ├── __init__.py
│   ├── models.py                      ← dataclasses : EISSpectrum, FitResult (+ warnings), EISSession
│   ├── loader.py                      ← import CSV/TXT EIS, validation, moyennage réplicats
│   ├── pipeline.py                    ← orchestrateur Import → Fit → Analyse (EIS)
│   ├── config.py                      ← chargement YAML + Pydantic AppSettings
│   ├── calibration.py                 ← calibration EIS/CV (signal normalisé, log-log) — source unique
│   ├── logger.py                      ← logging centralisé
│   ├── validator.py                   ← validation KK inter-réplicats (Lin-KK natif)
│   ├── cv_loader.py                   ← chargement fichiers CV
│   ├── cv_models.py                   ← CVScan, CVConcentrationGroup, CVSession
│   ├── cv_peaks.py                    ← détection de pics CV
│   ├── cv_pipeline.py                 ← pipeline traitement CV
│   ├── mpr_converter.py               ← conversion .mpr (BioLogic) → CSV (eclabfiles ; galvani en secours)
│   └── experiment_io.py               ← sauvegarde/chargement session complète (ZIP)
│
├── fits/                              ← système plugin : 1 fichier = 1 modèle (BaseFitModel)
│   ├── __init__.py
│   ├── base.py                        ← BaseFitModel (ABC) — interface commune
│   ├── physics.py                     ← fonctions physiques partagées (Z_D, Z_randles_full, θ_EIS…)
│   ├── registry.py                    ← découverte auto + discovery_errors() (modèles non chargés)
│   ├── randles_full.py                ← Randles complet (8 paramètres, least_squares) + diagnostics
│   ├── drt_fit.py                     ← moteur DRT (wrapper bayes-drt2) : ridge (aperçu) + HMC (référence, incertitudes)
│   └── kk_validation.py               ← validation Kramers-Kronig (Lin-KK, circuits de Voigt)
│
│   (le paquet de calcul DRT lui-même est vendoré dans vendor/bayes_drt2/)
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
core/pipeline.py → run_pipeline()
        ↓
fits/ — fits paramétriques, appliqués SÉQUENTIELLEMENT (pas en parallèle)
  randles_full   → FitResult { params, Zfit[], chi2_reduced, Rct, reconstruction_error, warnings }
        ↓
core/models.py → EISSession { bare, probe, groups[] }
        ↓
plotting/eis_plots.py → figures Plotly  ·  core/calibration.py → régressions
        ↓
ui/tabs.py::render_eis_tabs → onglets Streamlit (appelé par pages/A_eis.py)

DRT (hors pipeline) — moteur dédié fits/drt_fit.py (wrapper bayes-drt2) :
  drt_preview(spectrum) → DRTResult (ridge, aperçu live, pas d'intervalles)
  run_drt_bayes_batch(session) → HMC parallèle → DRTResult { γ, γ_lo, γ_hi,
     Rp, Rp_lo, Rp_hi, Z_fit } mis en cache + persisté dans sessions/
```

---

## 5. Les modèles de fit

| Modèle | Fichier | Paramètres libres | Méthode |
|--------|---------|-------------------|---------|
| Randles complet (`name="randles_full"`) | `randles_full.py` | 8 (Re, R'e, Cb, Rct, Qdl, α, R_D, τ_d) | `scipy.optimize.least_squares` (pondération Modulus) + diagnostics (résidu/butée) |

> Les fits paramétriques sont découverts automatiquement par `fits/registry.py`
> (sous-classes de `BaseFitModel`). Un modèle dont l'import échoue est signalé
> par `registry.discovery_errors()` et affiché dans l'UI, pas masqué.
> **La DRT n'est plus un plugin du registre** : c'est un moteur dédié
> (`fits/drt_fit.py`, voir §5bis) appelé directement par l'onglet DRT.

**Circuit physique (Randles modifié) :**
```
Re — [ R'e // Cb ] — [ Rct // CPE(Qdl, α) ] — ZD(ω)
```

## 5bis. Le moteur DRT (`fits/drt_fit.py`, wrapper `vendor/bayes_drt2/`)

DRT unique, bâtie sur le paquet **bayes-drt2** vendoré (inversion hiérarchique
bayésienne, Jake Huang — voir `vendor/README.md`, `THIRD_PARTY_LICENSES.md`).
Deux méthodes, un seul paquet de calcul :

- **`drt_preview(spectrum)` — aperçu instantané (ridge).**
  `Inverter.ridge_fit(freq, Z, hyper_lambda=True)` : ridge hyperparamétrique,
  rapide, sans cmdstan. Affiché en direct pour tous les spectres, et sert
  d'**initialiseur** du HMC. Pas d'intervalles (`gamma_lo/hi = None`).

- **`drt_bayes(spectrum)` / `run_drt_bayes_batch(session)` — référence (HMC).**
  `Inverter.fit(freq, Z, mode='sample', init_from_ridge=True)` : échantillonnage
  HMC qui produit en plus les **intervalles de crédibilité** (γ_lo/γ_hi à
  2.5/97.5 %, Rp_lo/Rp_hi). Le batch parallélise (`ProcessPoolExecutor`,
  `max_workers` = nombre de cœurs) sur tous les spectres non encore en cache.

**Convention d'impédance** : le loader stocke `Zim = -Im(Z) > 0` ; bayes-drt2
attend `Z'' < 0`, d'où `Z = spectrum.Zre - 1j·spectrum.Zim`.

**Cache + persistance** : clé = hash de `(freq, Z)` arrondis. Le DRTResult HMC est
mis en cache mémoire et persisté en YAML dans `sessions/drt_cache/`. Au
rechargement d'une session, un HMC déjà en cache **n'est jamais recalculé**.

**Garde-fous** :
- L'import de `vendor.bayes_drt2` est protégé (`try/except`) : extra DRT non
  installé (`cvxopt`/`cmdstanpy` absents) → DRT désactivée proprement, pas de
  crash (`drt_fit.bayes_available()` renvoie `False`).
- `drt_bayes` teste `cmdstanpy.cmdstan_path()` **sans compiler à la volée** :
  cmdstan absent → repli sur le ridge + statut « lancez setup_drt_bayesien.bat ».
  L'app reste fonctionnelle en aperçu ; seul le HMC est bloqué.

Dépendances : `requirements-drt.txt` (`cvxopt` + `cmdstanpy`). `requirements.txt`
de base reste léger.

Référence DRT (contexte capteur) : Bissessur, Man, Gamby, Phys. Rev. E **113**,
025502 (2026), DOI: 10.1103/fn2s-z364.

---

## 6. Onglets de l'analyse EIS (`ui/tabs.py::render_eis_tabs`)

Rendus par `pages/A_eis.py` (page **EIS seule**) ; la page **CV seule** utilise
`render_cv_tabs`. En tête de page EIS, un bandeau **« Diagnostics d'ajustement »**
remonte les avertissements de fit (non convergé, résidu élevé, paramètre en butée).

| Onglet | Contenu |
|--------|---------|
| **Validation KK** | Diagnostic Kramers-Kronig (Lin-KK) par réplicat |
| **Courbes DRT** | γ(τ) vs log₁₀(τ) — ridge en direct pour tous les spectres ; bouton « Calculer DRT bayésienne » → HMC + bande d'incertitude γ_lo–γ_hi (`fits/drt_fit.py`) |
| **Reconstructions Nyquist** | Table Rct_randles + reconstruction Nyquist mesuré/Randles |
| **Calibration** | Signal normalisé vs log([c]) + régression + R² (via `core/calibration.py`), une courbe par modèle |
| **Export** (page dédiée `E_export.py`) | CSV params, PNG, HTML, YAML, ZIP session |

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
