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
│   ├── drt_fit.py                     ← plugin DRT (drt_bayes, wrapper bayes_drt2) : MAP « optimize » (défaut) / HMC « sample »
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

DRT (plugin du pipeline) — fits/drt_fit.py::DRTBayesModel (wrapper bayes_drt2) :
  pipeline → DRTBayesModel().fit(sp) mode 'optimize' (MAP) → FitResult(drt_tau,
     drt_gamma, drt_mode='optimize', Rct par pic Bissessur)
  recompute_drt(session, spectrum_id, mode='sample') → HMC → FitResult avec
     intervalles (drt_gamma_lo/hi), à la demande depuis l'onglet DRT
```

---

## 5. Les modèles de fit

| Modèle | Fichier | Paramètres libres | Méthode |
|--------|---------|-------------------|---------|
| Randles complet (`name="randles_full"`) | `randles_full.py` | 8 (Re, R'e, Cb, Rct, Qdl, α, R_D, τ_d) | `scipy.optimize.least_squares` (pondération Modulus) + diagnostics (résidu/butée) |
| DRT (`name="drt_bayes"`) | `drt_fit.py` | 0 — γ(τ) model-free | `bayes_drt2` / `Inverter` — MAP « optimize » (défaut) / HMC « sample » (voir §5bis) |

> Les modèles sont découverts automatiquement par `fits/registry.py`
> (sous-classes de `BaseFitModel`). Un modèle dont l'import échoue est signalé
> par `registry.discovery_errors()` et affiché dans l'UI, pas masqué.
> **La DRT est de nouveau un plugin du registre** (`drt_bayes`) : elle est lancée
> par le pipeline en mode « optimize » comme les autres fits (voir §5bis).

**Circuit physique (Randles modifié) :**
```
Re — [ R'e // Cb ] — [ Rct // CPE(Qdl, α) ] — ZD(ω)
```

## 5bis. Le plugin DRT (`fits/drt_fit.py::DRTBayesModel`, wrapper `vendor/bayes_drt2/`)

DRT unique, bâtie **exclusivement** sur le paquet **bayes_drt2** vendoré (classe
`Inverter`, inversion hiérarchique bayésienne, Jake Huang — voir `vendor/README.md`,
`THIRD_PARTY_LICENSES.md`). Plugin `BaseFitModel` (`name="drt_bayes"`) découvert par
le registre et lancé par le pipeline. Deux modes, un seul paquet de calcul :

- **`mode='optimize'` — MAP Stan (défaut).**
  `Inverter.fit(freq, Z, mode='optimize')` : estimation du maximum a posteriori
  (L-BFGS-B). **Lancée par le pipeline sur chaque spectre**, comme les autres fits.
  Pas d'intervalles (`drt_gamma_lo/hi = None`).

- **`mode='sample'` — HMC bayésien (à la demande).**
  `Inverter.fit(freq, Z, mode='sample')` : échantillonnage HMC produisant en plus
  les **intervalles de crédibilité** (`drt_gamma_lo/hi` à 2.5/97.5 %). Jamais
  automatique : uniquement via `core.pipeline.recompute_drt(session, spectrum_id,
  config, mode='sample')` (seul point d'entrée), déclenché par le bouton de l'onglet
  DRT. Le résultat remplace le `FitResult` DRT du spectre dans la session.

**Récupération** : `gamma = inv.predict_distribution('DRT')`,
`tau = inv.distributions['DRT']['tau']` (garde-fou `KeyError`). Les deux modes
compilent des modèles Stan → CmdStan requis (aucun chemin sans compilation).

**Convention d'impédance** : le loader stocke `Zim = -Im(Z) > 0` ; bayes_drt2 attend
`Z'' < 0`, d'où `Z = spectrum.Zre - 1j·spectrum.Zim`. Tri HF→BF sur les **vraies
fréquences** lues (jamais reconstruites par `logspace`).

**Rct (grandeur de calibration)** : extrait de l'**arc de transfert de charge** par
la convention Bissessur (pic pénultième de γ(τ), ∫γ dlnτ sur ±3 en ln τ). Repli sur
`Rp` (aire totale) seulement si aucun pic n'est exploitable, **toujours signalé**
(`params['rct_source']` + `warnings`) pour ne pas confondre les deux grandeurs.

**Garde-fous** :
- L'import de `vendor.bayes_drt2` est protégé (`try/except`) : extra DRT non
  installé (`cvxopt`/`cmdstanpy`) → DRT désactivée proprement (`bayes_available()`
  renvoie `False`). Le fit DRT du pipeline échoue alors sans casser les autres.
- Toolchain : `app.py` appelle `setup_drt_bayesien.ensure_drt_ready()` **au premier
  lancement** (idempotent) → `install_cmdstan(compiler=True)` (installe mingw-w64,
  n'exige pas RTools) + compilation de `Series.stan`, mis en cache. Aucune étape
  manuelle ; la compilation n'a pas lieu au clic de l'utilisateur.

Dépendances : `requirements-drt.txt` (`cvxopt` + `cmdstanpy`). `requirements.txt`
de base reste léger (DRT désactivée proprement sans l'extra).

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
| **Courbes DRT** | ln(γ/γ₀) vs ln(τ/τ₀) (ln népérien, γ₀=1 Ω, τ₀=1 s) — MAP « optimize » pour tous les spectres, badge de mode ; bouton « Recalculer en bayésien (sample) » → HMC + bande d'incertitude labellisée (`core.pipeline.recompute_drt`) |
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
