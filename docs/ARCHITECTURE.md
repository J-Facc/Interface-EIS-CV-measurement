# EIS Analyzer — Architecture

> Document de référence, aligné sur le code (le code fait foi). Dépôt :
> https://github.com/J-Facc/Interface-EIS-CV-measurement — application web locale
> (Python 3.12+, Streamlit) pour l'analyse de spectres d'impédance (EIS) et de
> voltammétrie cyclique (CV) de biosenseurs microfluidiques ADN/ARN.
>
> Documents associés : [MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md) (méthode d'Orazem, KK),
> [CIRCUIT_UTILISATEUR.md](CIRCUIT_UTILISATEUR.md) (écrire son circuit),
> [DRT_BAYESIENNE.md](DRT_BAYESIENNE.md) (DRT), `drt/PROVENANCE.md`,
> `drt/VALIDATION_REGLAGES.md`. Les anciens rapports d'audit ne sont plus dans l'arbre :
> `git show 60371c7:AUDIT.md` (et `AUDIT_REPORT.md`) les restituent.

---

## 1. Ce que fait l'application

Pour chaque groupe de mesures — sonde seule (« probe »), puis chaque concentration de la
cible — l'application :

1. caractérise le bruit de mesure (measurement model, structure d'erreur d'Orazem) et
   vérifie la cohérence Kramers-Kronig des spectres ;
2. ajuste le **circuit équivalent défini par l'utilisateur** (fit Orazem pondéré 1/σ²) sur
   chaque réplicat et sur leur moyenne ;
3. calcule la **DRT bayésienne** (distribution des temps de relaxation) de chaque réplicat
   et de la moyenne ;
4. en déduit, pour le paramètre désigné par l'utilisateur (par défaut `Rct`) et pour le
   Rct de la DRT, une valeur par groupe avec deux incertitudes distinctes (intra-fit,
   inter-réplicats), puis une **calibration** : régression linéaire du signal normalisé
   sur log10([c]).

**Signal de calibration** (`core/calibration.py`, source unique) :
`signal = |Rct_probe − Rct_c| / |Rct_probe|`, régressé sur `log10([c])` (régression OLS,
`scipy.stats.linregress`). La référence est la **sonde** (pas l'électrode nue). Une
référence « électrode nue » peut être importée, mais elle n'est qu'**affichée** sur les
Nyquist (`EISSession.bare_reference`) et n'entre dans aucun calcul. Une seconde régression
log10(Rct) vs log10([c]) existe pour la DRT. L'application ne calcule ni taux de
recouvrement ni limite de détection.

Le circuit n'est **pas** fixe : l'expression par défaut est un Randles complet
(`config/default.yaml`, `fit.circuit`), mais l'utilisateur écrit le sien dans la page EIS
(expression, valeur initiale et bornes de chaque paramètre, paramètre cible).

---

## 2. Arborescence

```
Interface-EIS-CV-measurement/
├── app.py                    point d'entrée Streamlit : session_state, état de la DRT, navigation
├── launch.bat                LANCEUR UNIQUE Windows (mise à jour, venv, dépendances, DRT) — §8
├── setup_drt_bayesien.py     vérifie/installe CmdStan et compile les modèles Stan (appelé par launch.bat)
├── requirements.txt          dépendances d'exécution, épinglées (Python >= 3.12)
├── requirements-drt.txt      extra DRT épinglé : cvxopt, cmdstanpy
├── requirements-optional.txt galvani (lecteur .mpr de secours)
├── requirements-dev.txt      requirements.txt + pytest (tests / CI)
├── pytest.ini · .gitattributes · .gitignore · THIRD_PARTY_LICENSES.md
├── .github/workflows/validate.yml   CI : compileall + pytest ; job DRT réelle (cvxopt + CmdStan)
│
├── config/default.yaml       acquisition, fit (structure d'erreur, circuit par défaut, DRT)
│
├── core/                     logique métier et orchestration — JAMAIS d'import Streamlit
│   ├── pipeline.py           orchestrateur EIS : MM + KK → fit Orazem → DRT, par groupe
│   ├── measurement_model.py  measurement model de Voigt : structure d'erreur σ(ω) + verdict KK
│   ├── validator.py          verdict KK affiché au prétraitement, dérive inter-réplicats
│   ├── results_table.py      tables intra-fit / inter-réplicats / diagnostics HMC (UI + export)
│   ├── calibration.py        régressions de calibration EIS et CV — source unique
│   ├── models.py             EISSpectrum, GroupAnalysis, ConcentrationGroup, EISSession
│   ├── cv_models.py · cv_loader.py · cv_pipeline.py · cv_peaks.py   voltammétrie cyclique
│   ├── loader.py · robust_loader.py · mpr_converter.py   import EC-Lab (texte, .mpr)
│   ├── experiment_io.py      sauvegarde/chargement d'une expérience (ZIP v2, lecture v1)
│   ├── app_state.py          clés session_state partagées et leur invalidation
│   ├── config.py             YAML + Pydantic
│   └── logger.py             journal fichier + console (écrit dans logs/)
│
├── fits/                     bibliothèque numérique du fit — n'importe JAMAIS core
│   ├── orazem_fit.py         fit d'un circuit LIBRE (par réplicat + moyenne, agrégation)
│   ├── regression_stats.py   covariance par SVD, conditionnement, identifiabilité
│   ├── kk_validation.py      critère KK unique (kk_verdict), Lin-KK indicatif
│   ├── physics.py            Z_D (référence de l'élément ZD_bounded)
│   └── result.py             FitResult : contrat commun fit Orazem / DRT
│
├── circuit/                  circuit utilisateur : Python restreint, liste blanche AST (aucun eval)
│   ├── parser.py · elements.py · registry_elements.py
│
├── drt/                      DRT — n'importe de fits que fits.result
│   ├── engine.py             moteur durci (HMC par défaut, MAP en aperçu, gardes)
│   ├── diagnostics.py        seuils R-hat / ESS / divergences / qualité de reconstruction
│   ├── bayes_drt2/           clone identifié de bayes-drt2 (PROVENANCE.md) + modèles Stan
│   ├── validation/           scripts et résultats bruts de la validation des réglages
│   ├── PROVENANCE.md · VALIDATION_REGLAGES.md
│
├── plotting/                 reçoit des données, ne calcule rien (eis_plots, cv_plots, kk_plots, theme)
├── exports/exporter.py       CSV, YAML, ZIP de session
├── ui/tabs.py                render_eis_tabs / render_cv_tabs (appelés par les pages)
├── pages/                    0_import · 1_pretraitement · A_eis · B_cv · E_export
├── docs/                     ARCHITECTURE · MEASUREMENT_MODEL · CIRCUIT_UTILISATEUR · DRT_BAYESIENNE
└── tests/                    pytest : circuit, fit, KK, DRT, pipeline, loader, CV, export, lanceur…
```

Créés à l'exécution et ignorés par git : `logs/`, `sessions/`.

---

## 3. Règles de modularité

```
pages / ui / plotting / exports ──► core ──► drt ──► fits.result
                                     │                  ▲
                                     └────► fits ───────┘
                                             └──► circuit (feuille)
```

| Couche | Règle |
|---|---|
| `core/` | Jamais d'import Streamlit. Importe `fits`/`drt`/`circuit` au niveau module (aucun import local masquant un cycle). |
| `fits/`, `drt/`, `circuit/` | N'importent jamais `core`, `ui`, `pages`, `plotting`, `exports` ni Streamlit. `drt` n'importe de `fits` que `fits.result` ; `circuit` n'importe rien du projet. |
| `plotting/` | Reçoit des données, ne les calcule pas (les régressions viennent de `core/calibration.py`). |
| `ui/` | Appelle `core/`, n'implémente pas de physique. |
| `exports/` | Doit contenir `__init__.py`, et n'est pas ignoré par `.gitignore`. |

`tests/test_architecture.py` vérifie ces règles sur l'arbre d'imports réel. Il n'existe plus de
registre de plugins : un seul moteur de fit (Orazem sur circuit utilisateur), une seule DRT,
appelés explicitement par `core/pipeline.py`.

---

## 4. Navigation et état

```
app.py → st.navigation({
  "Données":  [0_import.py, 1_pretraitement.py],
  "Analyse":  [A_eis.py,    B_cv.py           ],
  "Export":   [E_export.py                    ],
})
```

Les onglets EIS/CV sont rendus par `ui/tabs.py::render_eis_tabs` / `render_cv_tabs`, appelés
depuis `A_eis.py` et `B_cv.py`.

Flux obligatoire (clés de `st.session_state`, `core/app_state.py`) :

```
0_import.py          → experiment, import_validated
1_pretraitement.py   → exclusions, experiment_clean, preprocessing_done, validation_results
A_eis.py / B_cv.py   ← lisent experiment_clean ; st.warning + st.stop() si absent
                     → eis_sessions, eis_validations, eis_normalized, eis_config, cv_sessions
E_export.py          ← lit experiment_clean et les résultats d'analyse
```

Toute modification des DONNÉES (import, rechargement de ZIP, re-prétraitement) invalide
**tous** les résultats d'analyse (`reset_analysis_results`) : jamais d'analyse périmée affichée.
L'état de la DRT (`drt_ready`, `drt_ready_msg`) est constaté une fois par process par `app.py`,
sans rien installer.

Structure de `experiment` :

```python
{
  "name": str, "date": "YYYY-MM-DD",
  "mode": "eis_only" | "cv_only" | "both",
  "concentrations": [float, ...], "n_electrodes": int, "n_replicats": int,
  "probe":       {"eis": {"electrode_1": [BytesIO, ...]}, "cv": {...}},
  "calibration": {"eis": {"electrode_1": [[BytesIO, ...], ...]}, "cv": {...}},
  "validation": {...} | None,
}
```

`core/experiment_io.py` : `save_experiment(experiment) -> bytes` et `load_experiment(zip_bytes)`.
Format v2 : `experiment.yaml` + `probe/{eis,cv}/electrode_N/*.txt` +
`calibration/{eis,cv}/electrode_N/*.txt` (+ `validation/…`). Le format v1 (`.bin`) reste lisible.

---

## 5. Flux de données de l'analyse EIS

```
Fichiers CSV/TXT (export EC-Lab) ou .mpr
        ↓  core/loader.py (+ robust_loader.py)
réplicats BRUTS (EISSpectrum), conservés à chaque étape
        ↓  par groupe : probe, chaque concentration (et bare si fourni, affichage seul)
core/measurement_model.analyze_replicates          — sur les réplicats BRUTS, AVANT tout fit
   • σ(ω) = structure d'erreur d'Orazem du groupe (≥ 3 réplicats)
   • verdict Kramers-Kronig (réplicats + moyenne)   → ValidationResult (onglet « Measurement model & fit Orazem »)
   • ErrorStructureUnavailable → groupe ARRÊTÉ (GroupAnalysis.status + message)
        ↓
fits/orazem_fit.fit_replicate_group
   • fit de CHAQUE réplicat (poids 1/σ²) et de la moyenne (σ/√n) du circuit utilisateur
   • agrégation : moyenne, écart inter-réplicats, incertitude intra-fit, incertitude de la
     moyenne, Q de Cochran
        ↓
drt/engine.fit_drt sur CHAQUE réplicat ET la moyenne (mode fit.drt.mode)
   • Rct DRT agrégé sur les réplicats ; recompute_drt(…, mode='sample') à la demande
        ↓
core/models.py → EISSession { probe, groups[], *_replicate_spectra, *_analysis (GroupAnalysis) }
        ↓
core/results_table.py → tables intra / inter / diagnostics HMC   ·   core/calibration.py → régressions
        ↓
ui/tabs.py (pages/A_eis.py)   ·   exports/exporter.py (pages/E_export.py)
```

Erreurs : saisie invalide → `InvalidAnalysisInput` avant tout calcul ; donnée invalide →
résultat dégradé + message rangé dans la session (`load_errors`, `GroupAnalysis.status/message`,
`messages`) ; bug → l'exception remonte et la page l'affiche comme erreur logicielle. Une page
n'affiche jamais « analyse terminée » si un groupe n'a aucun fit. Détail : docstring de
`core/pipeline.py`.

**Convention d'impédance** : le loader stocke `Zim = −Im(Z) > 0` ; bayes_drt2 attend `Z'' < 0`,
d'où `Z = Zre − 1j·Zim` dans `drt/engine.py`.

**Analyse CV** (`core/cv_pipeline.py`) : groupes par (étape, concentration), moyenne de
réplicats par interpolation, `delta_signal = |I_probe − I_c| / |I_probe|`, régression sur
log10([c]). Une voltammétrie cyclique est une boucle : `core/cv_loader.py` détecte les branches
sur l'ordre de mesure et ne trie jamais par potentiel (qui entrelacerait aller et retour).

---

## 6. Le circuit, le fit et la DRT

| Méthode | Paramètres | Sortie (`fit_results` de chaque spectre) |
|---|---|---|
| Fit Orazem, clé `"orazem"` | circuit utilisateur (`fit.circuit`, par défaut un Randles complet à 8 paramètres) ; valeur initiale et bornes par paramètre ; paramètre cible désigné | `FitResult` : `target_param/target_value/target_std` (intra-fit), χ²ᵣ et son intervalle attendu, `fit_diagnostics` (conditionnement, rang, identifiabilité, bornes actives) |
| DRT, clé `"drt_bayes"` | réglages constants de `drt/engine.py` (justifiés par `drt/VALIDATION_REGLAGES.md`) ; mode `fit.drt.mode` | `FitResult` : γ(τ), Rct (pic pénultième, ±3 en ln τ), `target_std` a posteriori (HMC), `drt_diagnostics` (R̂, ESS, divergences, alertes) |

Le circuit par défaut (`config/default.yaml`) :

```
Re + parallel(Re_prime + parallel(R(Rct) + ZD_bounded(R_D, tau_d), Q(Qdl, alpha)), C(Cb))
```

soit `Z_eq = R'e + (Rct + Z_D) ∥ CPE(Qdl, α)` puis `Z = Re + Z_eq ∥ Cb`, avec `Z_D` diffusion
bornée (formulation Bissessur). C'est un point de départ, pas une contrainte : toute expression
valide de [CIRCUIT_UTILISATEUR.md](CIRCUIT_UTILISATEUR.md) est ajustée de la même façon.

Le fit est un moindres carrés non linéaire borné (`scipy.optimize.least_squares`, multi-départ),
pondéré par `w = 1/σ²` ; la covariance vient de `(JᵀJ)⁻¹` par SVD. La méthode et ses sources sont
dans [MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md) ; la DRT dans [DRT_BAYESIENNE.md](DRT_BAYESIENNE.md).

---

## 7. Onglets de l'analyse EIS (`ui/tabs.py::render_eis_tabs`)

En tête de page : statut explicite de l'analyse (groupes arrêtés, fichiers écartés, moteur DRT
absent). Les résultats sont présentés en **trois onglets exactement**, chacun découpé par
électrode (sous-onglets, comme la page CV) puis par groupe de réplicats :

| Onglet | Contenu |
|---|---|
| **1 · Visualisation** | Données mesurées seulement, aucun fit. Nyquist et Bode (|Z| et −phase) : réplicats superposés (traits fins) + moyenne (trait épais), une couleur par concentration, légende cliquable par groupe, filtre de groupes, bascule prétraitées / brutes (avant exclusions), référence « électrode nue » en pointillés, et une vue « Normalisé E1 + E2 » |
| **2 · Measurement model & fit Orazem** | Récapitulatif de l'électrode (verdict KK, éléments de Voigt, fiabilité du fit, nombre d'alertes), puis pour le groupe choisi, **dans l'ordre de lecture** : (1) measurement model et verdict KK — conforme / non conforme / indéterminé —, éléments de Voigt retenus et points hors bande par spectre, résidus, structure d'erreur ; (2) fit Orazem — bandeau **rouge** si κ dépasse `fits.orazem_fit.CONDITION_NUMBER_WARN` (≈ 6,7·10⁷) ou si un paramètre n'est pas identifiable, alertes de fit affichées d'emblée (jamais dans un expander), paramètre cible mis en évidence (★), Nyquist expérimental + circuit ajusté et résidus, tableau des paramètres (intra-fit, inter-réplicats, incertitude de la moyenne, fit du spectre moyen), diagnostics de CHAQUE fit (κ, rang/P, non identifiables, dérivée unilatérale, bornes actives, χ²ᵣ avec son intervalle attendu, départs convergés). Un groupe arrêté (moins de 3 réplicats…) affiche son verdict « indéterminé » et la cause, pas une erreur ni un tableau vide |
| **3 · DRT** | Par groupe : tableau des diagnostics de CHAQUE spectre, toujours affiché (R̂ max, divergences, ESS bulk/tail avec leurs seuils, E-BFMI, Rct ± σ a posteriori, IC 95 % de Rct, origine de Rct, Rp) ; vue agrégée (γ(τ) moyenne + bande LARGE de variabilité inter-réplicats, min–max) ou vue d'un réplicat / du spectre moyen (γ(τ) + bande FINE de crédibilité HMC, diagnostics en grand, alertes) ; échec de calcul d'un spectre nommé en rouge avec son motif (le pipeline continue sans DRT pour ce spectre) ; rappel permanent sur les intervalles de Rp et de Rct ; recalcul bayésien d'un spectre |

Plus de calibration ni de reconstructions dans la page EIS : les valeurs correspondantes
(calibration EIS, reconstructions) restent produites par la page Export.

La page **Export** (`E_export.py`) produit : paramètres (colonnes par modèle), résultats par
réplicat et par groupe, DRT, reconstructions, calibration EIS/CV, YAML de session, ZIP complet.
Il n'y a pas d'export PNG/HTML de figures.

---

## 8. Lanceur Windows (`launch.bat`)

Seul fichier à distribuer. Le dépôt est public : aucun jeton ; la validation TLS est normale.

```
1. Version distante : API GitHub publique (commits/main → SHA), TLS 1.2, délai 15 s.
   ÉCHEC, quelle qu'en soit la cause (réseau, DNS, délai, erreur API) → message « HORS LIGNE »
   ou « ECHEC REEL », puis lancement DIRECT de la version locale : ni mise à jour, ni pip, ni CmdStan.
2. SHA différent : téléchargement du zipball de CE SHA dans %TEMP%\eis_launcher.
3. Extraction dans .eis_update\ ; vérification (app.py, requirements*.txt, setup_drt_bayesien.py,
   Series.stan…) ; restauration des modèles Stan compilés depuis stan_cache\ si le .stan est
   identique octet pour octet ; PUIS seulement : eis_app → eis_app.old, nouveau dossier → eis_app
   (retour arrière automatique si le déplacement échoue). logs\ et sessions\ sont conservés.
4. venv\ (Python >= 3.12 exigé : numpy 2.5.x) ; pip install -r requirements.txt puis
   requirements-drt.txt, SEULEMENT si l'empreinte de ces fichiers a changé (.deps_hash).
5. Moteur DRT : setup_drt_bayesien.py --check ; si besoin --ensure — plusieurs minutes la première
   fois. La version de CmdStan (2.36.0, `drt/cmdstan_version.py`) est la source unique : elle est
   ENREGISTRÉE de préférence à toute version plus récente du même dossier, et installée à côté
   d'une autre version sans jamais la supprimer. Avant toute compilation, la toolchain C++
   (`mingw32-make` + `g++`) est vérifiée, installée si besoin, puis re-vérifiée au PATH (cmdstanpy
   ne l'ajoute au PATH que dans le process qui installe : `activate_toolchain` le refait pour
   chaque process). Series.stan ET Series_pos.stan sont compilés ; un fichier
   `<modèle>.cmdstan-version` note la version de CmdStan qui a produit chaque exécutable (cmdstanpy
   ne recompile qu'une source plus récente que son exécutable, sans regarder la version).
   Un échec ici ne bloque jamais le démarrage : l'application tourne sans DRT.
6. Port libre de 8501 à 8510 ; Streamlit démarre ; le navigateur s'ouvre APRÈS la réponse de
   /_stcore/health.
```

Fichiers créés à côté du `.bat` : `eis_app\`, `venv\`, `stan_cache\`, `.version`, `.deps_hash`,
`launcher.log`. Aucune écriture dans le registre (les chemins longs ne sont pas nécessaires :
venv court, CmdStan dans `C:\cmdstan` ; un avertissement s'affiche si le dossier du lanceur est
très profond).

Règles d'écriture du script (gardées par `tests/test_launcher.py`) : `!VAR!` exclusivement dans
les blocs `( … )` ; `ROOT` capturé avant l'expansion retardée ; `!errorlevel!` testé après chaque
commande significative ; fichier en CRLF (`.gitattributes`).

**Développement (Linux/macOS, ou Windows sans le lanceur)** :

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt                       # + requirements-drt.txt pour la DRT
python setup_drt_bayesien.py --ensure                     # CmdStan + modèles Stan (optionnel)
streamlit run app.py
```

---

## 9. Tests et CI

```bash
python -m pytest tests/ -v                 # tout
python -m pytest tests/ -m "not slow"      # sans le HMC lent
python -m compileall app.py setup_drt_bayesien.py core fits drt circuit plotting exports ui pages tests
```

`validate.yml` : job `validate` (Python 3.12, `requirements-dev.txt`, compileall + pytest) et job
`drt` (cvxopt + CmdStan réels : les tests DRT, sautés sans CmdStan, y sont exécutés). Limite
connue : deux paramétrisations du test HMC lent peuvent échouer selon la machine (voir
`drt/VALIDATION_REGLAGES.md` §7).

---

## 10. Erreurs fréquentes

| Symptôme | Cause | Solution |
|---|---|---|
| `OSError … streamlit` ou erreurs de chemin long | dossier du lanceur trop profond | déplacer `launch.bat` dans un dossier court (`C:\EIS`) |
| `SSL: CERTIFICATE_VERIFY_FAILED` (pip, CmdStan) | proxy d'entreprise qui intercepte le TLS | définir `HTTPS_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE` (aide : `python setup_drt_bayesien.py --help`) ; la validation TLS n'est jamais désactivée |
| « Python 3.12 ou plus récent est introuvable » | Python absent, trop ancien, ou stub du Microsoft Store | installer Python depuis python.org |
| Onglet DRT : « moteur indisponible » | extra pip ou CmdStan non installé (premier lancement hors ligne) | relancer `launch.bat` avec réseau, ou `python setup_drt_bayesien.py --ensure` |
| `setup_drt_bayesien.py` code 8 : « Toolchain C++ absente ou non fonctionnelle » | `mingw32-make` / `g++` introuvables au PATH (RTools absent ou non installable hors ligne) | relancer `launch.bat` avec réseau, ou `python -m cmdstanpy.install_cxx_toolchain --dir C:\cmdstan` |
| Code 6 : « Erreur de compilation Stan/C++ » | la toolchain fonctionne, la compilation d'un modèle échoue (dernière sortie affichée) | lire la sortie affichée ; chemin trop long ? déplacer `launch.bat` dans `C:\EIS` |
| Onglet DRT : « CmdStan x.y.z est utilisé, alors que les réglages… 2.36.0 » | une autre version de CmdStan est la seule utilisable (2.36.0 non installable hors ligne) | relancer `launch.bat` avec réseau : la 2.36.0 est installée à côté, l'autre n'est pas touchée |
| « DRT de « … » non calculée : échec à l'étape … » | exception pendant le calcul DRT | le traceback complet est dans `logs\eis_analyzer.log` (étape, origine, spectre, réglages, version de CmdStan) |
| Groupe « arrêté » | moins de 3 réplicats : structure d'erreur non caractérisable | fournir au moins 3 réplicats indépendants par groupe |
| Fenêtre `.bat` qui se ferme / variable vide | `%VAR%` lu dans un bloc `( … )` | `!VAR!` + `enabledelayedexpansion` (règle de `launch.bat`) |
