# AUDIT_REPORT — EIS Analyzer : préparation d'une « DRT bayésienne bayes_drt2 en 5ᵉ modèle »

> Audit en LECTURE SEULE. Dépôt audité : `J-Facc/Interface-EIS-CV-measurement`, branche
> `claude/zealous-mendel-17kf3f` = `main` au commit `060b97f` (merge PR #95), arbre propre avant et après.
> Seul fichier créé dans le dépôt : ce rapport (non commité). Tout le reste (venv, CmdStan, copie du dépôt
> pour les tests, script de chronométrage) vit dans le scratchpad, hors dépôt.
> Plate-forme d'audit : Linux x86-64, Python 3.13.16. **Windows n'a pas été testé** (voir §Limites).

---

## 0. Résultat principal : la prémisse du chantier est périmée

La demande suppose un dépôt avec `ARCHITECTURE_v2.md`, `REFERENCE_MAINTENANCE.html`, une classe `BaseFitModel`,
un `registry.py`, un `fits/drt_fit.py` (Tikhonov), « 4 fits » et « 7 figures ». **Rien de cela n'existe dans le
dépôt actuel, ni dans son historique** (`git log --all` ne trouve ni `ARCHITECTURE_v2.md` ni
`REFERENCE_MAINTENANCE.html`). Pire pour le plan : **`bayes_drt2` est déjà intégré**, de bout en bout.

| Élément supposé par la demande | Réalité au `HEAD` |
|---|---|
| `BaseFitModel` + `registry.py` (plugins) | Supprimés par le refactor `e0c5afa` (2026-10-01) : `fits/base.py`, `fits/registry.py`, `fits/drt_fit.py`, `vendor/` n'existent plus. `docs/ARCHITECTURE.md:123` : « Il n'existe plus de registre de plugins ». |
| `drt_fit.py` Tikhonov | Remplacé par `drt/engine.py` (moteur durci autour de `drt.bayes_drt2.Inverter`). |
| 4 fits | 2 méthodes d'ajustement : fit Orazem d'un circuit libre (`fits/orazem_fit.py`) et DRT bayésienne (`drt/engine.py`), plus measurement model + Kramers-Kronig (`core/measurement_model.py`). |
| `bayes_drt2` à installer | Copié dans `drt/bayes_drt2/` (clone identifié, commit amont `99d5b60`, BSD-3, 3 patchs : `drt/PROVENANCE.md`). Modes `optimize` ET `sample` déjà câblés. |
| DRT à ajouter à l'UI/export | Onglet « 3 · DRT » (`ui/tabs.py:592`), recalcul HMC à la demande (`ui/tabs.py:574-589`), export `export/drt/drt_values.csv` (`exports/exporter.py:241,461`), calibration `drt_bayes`, installation CmdStan par `launch.bat`, job CI `drt` réel. |

Aussi : `tests/test_architecture.py:24,75-88` **échoue si on recrée** `fits/base.py`, `fits/registry.py` ou
`fits/drt_fit.py` (il vérifie leur absence). Exécuter les trois prompts tels que cadrés violerait donc un test existant.

Hypothèse la plus probable : le plan a été rédigé depuis une copie locale / une documentation antérieures au
commit `21bebf8` (« DRT unique via bayes_drt2/Inverter (plugin BaseFitModel, optimize/sample ») puis `e0c5afa`.
**À confirmer par vous** : `git log --oneline -3` sur votre poste doit montrer `060b97f`.

---

## 1. Tableau des constats

Gravité calculée **par rapport à l'objectif annoncé** (intégrer bayes_drt2 comme 5ᵉ modèle).

| # | Constat | Gravité | Fichier:ligne |
|---|---|---|---|
| 1 | La DRT bayes_drt2 (optimize + sample) est déjà intégrée : moteur, pipeline, UI, export, launcher, CI. Refaire l'intégration = doublon / régression. | **Bloquant** | `drt/engine.py:431`, `core/pipeline.py:304-315,361-364`, `ui/tabs.py:592`, `exports/exporter.py:241` |
| 2 | `BaseFitModel`, registre, `fits/drt_fit.py`, `ARCHITECTURE_v2.md`, `REFERENCE_MAINTENANCE.html` n'existent pas ; recréer les modules supprimés fait échouer un test. | **Bloquant** | `tests/test_architecture.py:24,75-88`; commit `e0c5afa` |
| 3 | `Inverter().fit(..., mode='optimize')` avec les réglages amont donne un résultat non physique : **Rp = −5856 Ω** sur un Randles de Rct = 3000 Ω (mesuré). Le moteur durci (`nonneg`, `init_from_ridge`, graine) donne Rct = 3070 Ω. | Important | `drt/engine.py:75-106` ; mesure §E |
| 4 | `Inverter.__init__` a un dict `distributions` par défaut **mutable et partagé** (état conservé d'un fit à l'autre) ; non patché, contourné par le moteur. Tout appel direct à `Inverter()` réintroduit le bug. | Important | `drt/bayes_drt2/inversion.py:33-34` ; contournement `drt/engine.py:153-169` |
| 5 | La config ne rejette **aucune clé inconnue** (Pydantic v2, `extra` non défini → ignoré ; mesuré). Une clé `fit.drt.chains` / une faute de frappe est silencieusement perdue. Et `DRTSettings` (config) ne porte que `enabled` et `mode` : chaînes, tirages, graine, `adapt_delta`, `nonneg` ne sont pas configurables. | Important | `core/config.py:51-65,105-107` ; `core/pipeline.py:166` |
| 6 | Durée de calcul et horodatage de la DRT non enregistrés. Aucun champ dédié, mais `drt_diagnostics` est un `dict` libre → pas besoin d'étendre la dataclass. | Important | `fits/result.py:70` ; `drt/engine.py:580-597` |
| 7 | L'analyse complète s'exécute **de façon synchrone dans le script Streamlit** (spinner, pas de progression ni d'annulation). En mode `sample` : 2–5 min × (réplicats + moyenne) × groupes × électrodes (≈ 56 spectres pour 2 électrodes × 7 groupes × 4 = 2 à 5 h). Un rerun ou une navigation interrompt le script et perd le travail. | Important | `pages/A_eis.py:350-379` |
| 8 | Une exception « bug » sur une électrode fait perdre **toutes** les électrodes (le dict `sessions` n'est affecté qu'après la boucle). | Important | `pages/A_eis.py:352-378` vs `:380` |
| 9 | CI : le job `drt` n'exécute que 3 fichiers de test ; `tests/test_cmdstan_version.py:215` (marqué « job CI drt ») n'est jamais exécuté sur un vrai CmdStan en CI. Aucun runner Windows alors que la cible utilisateur est Windows (`launch.bat`, mingw). Aucun lint/typage. | Important | `.github/workflows/validate.yml:63`, `tests/test_cmdstan_version.py:215` |
| 10 | Les résultats d'analyse vivent dans `st.session_state` : ils survivent aux reruns, pas à un rafraîchissement du navigateur ni à un redémarrage ; la sauvegarde ZIP (`experiment_io`) ne conserve que les entrées. | Important (si besoin de persistance) | `pages/A_eis.py:380-383` ; `core/experiment_io.py` |
| 11 | Doc : `docs/ARCHITECTURE.md` cite `core/cv_peaks.py`, absent du dépôt (retiré avec « pics redox », commit `0781026`). | Mineur | `docs/ARCHITECTURE.md:69` |
| 12 | `export_drt_csv` n'exporte ni les bornes de crédibilité (`drt_gamma_lo/hi`), ni la durée ; `ln(γ + 1e-300)` ramène les γ = 0 (cas `nonneg`) à −690,8. | Mineur | `exports/exporter.py:241-275` (l.270) |
| 13 | 48 `SyntaxWarning: invalid escape sequence` à la compilation du code tiers (`plotting.py` 26, `equiv_circuit.py` 11, `inversion.py` 8, `utils.py` 3). Sans effet aujourd'hui ; non patchés (politique de provenance). | Mineur | `drt/bayes_drt2/*.py` |
| 14 | `docs/validation_derive/` contient un script `.py` sans `__init__.py` (dossier de scripts, non importé). | Mineur | `docs/validation_derive/run_drift_study.py` |
| 15 | `_drt_status()` est mis en cache **par process** : installer CmdStan pendant que l'app tourne n'est vu qu'après redémarrage (volontaire, documenté dans le commentaire). | Mineur | `app.py:36-52` |
| 16 | `except Exception: pass` avale toute erreur de lecture de la référence « électrode nue » (affichage seul). | Mineur | `pages/A_eis.py:153` |
| 17 | `eis_inputs` / `eis_config` ne sont pas dans `ANALYSIS_RESULT_KEYS` : non remis à zéro par `reset_analysis_results`. | Mineur | `core/app_state.py:27,38` ; `pages/A_eis.py:383` |
| 18 | `nyquist_figure_electrode` n'est utilisée que par les tests (code mort côté application). | Mineur | `plotting/eis_plots.py:210` |
| 19 | `load_config` retombe silencieusement sur les défauts si `config/default.yaml` est absent ; un fichier vide lève une `ValidationError` peu lisible. | Mineur | `core/config.py:122-126` |
| 20 | Launcher : en cas d'échec TLS/proxy de `pip`, le message distingue « HORS LIGNE » / « ECHEC REEL » mais ne mentionne pas `HTTPS_PROXY`/`REQUESTS_CA_BUNDLE` (la consigne n'est que dans `docs/ARCHITECTURE.md` §10). `BRANCH=main` : rien d'une branche de travail n'atteint les utilisateurs avant fusion. | Mineur | `launch.bat:35,427-437` |
| 21 | Détection de « drift » : 33–37 % des fréquences signalées sur des réplicats synthétiques **stationnaires** (non investigué plus loin ; la calibration est le sujet de `docs/validation_derive/`). | À noter | `core/validator.py` (`_detect_drift`, l.376) |

---

## 2. Réponses point par point

### A. Structure et règles

**1. `__init__.py`.** 11 : `circuit`, `core`, `drt`, `drt/bayes_drt2`, `drt/validation`, `exports`, `fits`, `pages`,
`plotting`, `tests`, `ui`. Dossiers contenant des `.py` sans `__init__.py` : la racine (`app.py`,
`setup_drt_bayesien.py`, normal) et `docs/validation_derive/` (script d'étude, constat 14). `exports/` en a bien un
(règle de `ARCHITECTURE.md` §3 respectée).

**2. `streamlit` dans `core/` et `fits/`.** Aucun import. Les occurrences sont des docstrings/commentaires
(`core/experiment_io.py:14,510`, `core/results_table.py:3,187`, `core/app_state.py:4`, `core/validator.py:21`,
`drt/engine.py:5,167`, `circuit/parser.py:227`). La règle est en outre gardée par `tests/test_architecture.py`
(passe). **Aucune violation.**

**3. Fichiers/fonctions annoncés mais absents.**
- `ARCHITECTURE_v2.md`, `REFERENCE_MAINTENANCE.html` : absents, y compris de l'historique. La doc réelle est
  `docs/ARCHITECTURE.md` (+ `README.md`, `docs/*.md`).
- `core/cv_peaks.py` : cité (`docs/ARCHITECTURE.md:69`), absent.
- « 7 figures » de `plotting/eis_plots.py` : la doc actuelle n'en annonce aucun nombre. Réellement **11 fonctions
  figure** : `nyquist_figure`, `nyquist_figure_electrode`, `nyquist_normalized_figure`, `drt_figure`,
  `drt_aggregate_figure`, `calibration_figure`, `calibration_loglog_figure`, `nyquist_replicates_figure`,
  `bode_figure`, `fit_nyquist_figure`, `fit_residuals_figure`. Toutes utilisées par l'app (`ui/tabs.py`,
  `pages/1_pretraitement.py`) sauf `nyquist_figure_electrode` (tests seuls).
- Tout le reste de l'arborescence de `ARCHITECTURE.md` §2 existe (vérifié fichier par fichier).

### B. Contrats

**4. `BaseFitModel`.** N'existe pas (voir §0). Les contrats réels sont des fonctions :
- DRT : `drt.engine.fit_drt(spectrum, *, mode='sample', nonneg=True, init_from_ridge=True, random_seed=1234,
  chains=4, warmup=500, samples=2000, adapt_delta=0.99, max_iter=50000, model_name='drt_bayes') -> FitResult`
  (`drt/engine.py:431-435`). `spectrum` = tout objet portant `f`, `Zre`, `Zim` (`Zim = −Im Z > 0`). Lève
  `ValueError` (entrée invalide) ou `RuntimeError`/`DRTComputationError` (moteur absent ou échec CmdStan).
- Orazem : `fits.orazem_fit.fit_replicate_group(Z_func, param_names, reps, mm, specs, target, *, options,
  model_name)` (`fits/orazem_fit.py:659`).
- Pas de `predict()` / `method` / `display_name` : le nom de modèle est la clé de `fit_results`
  (`'orazem'`, `'drt_bayes'` = `drt.engine.MODEL_NAME`, `drt/engine.py:66`).

**5. Champs et stockage.** `FitResult` (`fits/result.py:20-86`) :
- tau / gamma : `drt_tau`, `drt_gamma` (l.58-59) ; bornes 2,5/97,5 % : `drt_gamma_lo/hi` (l.61-62, `sample` seulement) ;
  **mode** : `drt_mode` (l.60, `'optimize'|'sample'`) ; diagnostics : `drt_diagnostics` (dict libre, l.70).
- **Durée : aucun champ, aucune mesure.** Peut se loger dans `drt_diagnostics` sans toucher la dataclass.
- `EISSpectrum.fit_results: dict[str, FitResult]` (`core/models.py:59`) ; `EISSession.drt_mode` (`core/models.py:224`) ;
  agrégats dans `GroupAnalysis.drt_target / drt_failures` (`core/models.py:90-124`).
- L'ancien `drt_fit.py` (Tikhonov) est supprimé ; aujourd'hui `fit_drt` renvoie tau/gamma dans le `FitResult`
  (`drt/engine.py:606-628`).

**6. `registry.py`.** Ni découverte automatique ni enregistrement manuel : **appels explicites** dans
`core/pipeline.py` (l.342 pour Orazem, l.361-364 pour la DRT).

**7. `run_pipeline()` (`core/pipeline.py:372`).** **Séquentiel**, dans le processus, sans thread ni
multiprocessing (aucune occurrence de `Thread`/`multiprocessing`/`concurrent` dans `core/`, `ui/`, `pages/`,
`drt/engine.py`). Ordre : pour chaque groupe (l.432-446) → measurement model + KK → fit Orazem de chaque réplicat
et de la moyenne → DRT de chaque réplicat et de la moyenne. Isolation des erreurs :
- DRT : `ValueError`/`RuntimeError` par spectre → `GroupAnalysis.drt_failures`, le reste continue (l.304-315) ;
- structure d'erreur non caractérisable → groupe arrêté (l.330-336) ; circuit incompatible → groupe arrêté
  (l.345-350) ;
- toute autre exception **remonte** (docstring l.56-57) → arrête l'électrode, voire toute l'analyse côté page
  (constat 8).
Résultats : `spectrum.fit_results[model_name]` (l.315, 354-355), `GroupAnalysis`, `EISSession`, puis
`st.session_state['eis_sessions']` (`pages/A_eis.py:380`).

**8. Configuration.** `core/config.py:113` `load_config()` → `yaml.safe_load` → `AppSettings.model_validate`
(chemin par défaut `config/default.yaml`) ; `config_to_dict()` (l.129) produit le dict passé à `run_pipeline`.
Appelé à l'import de `pages/A_eis.py:29`. **Une clé inconnue ne fait PAS échouer la validation** (mesuré :
`{"fit":{"drt":{"mode":"sample","chains":8}},"inconnu":3}` accepté, `chains` absent du `model_dump`). En
revanche un `fit.drt.mode` invalide lève bien une `ValidationError` (l.59-64).

### C. UI et état

**9. Onglet DRT.** `ui/tabs.py::_render_drt_tab` (l.592), appelé par `render_eis_tabs` (l.730) avec les sessions
de `st.session_state['eis_sessions']`. Lit `drt_version_warning` (l.604) et `eis_config` (l.616). Aucun
`st.cache_*` dans `ui/` ni `core/` ; seuls `app.py:36,55` utilisent `st.cache_resource` (état de la toolchain DRT,
par process). **Déclencheurs de calcul** : (a) première visite de la page quand `eis_sessions` est vide ;
(b) bouton « ↺ Relancer l'analyse » (`pages/A_eis.py:337-340`) ; (c) bouton « 🎲 Recalculer en HMC » d'un spectre
(`ui/tabs.py:580-589`, appelle `core.pipeline.recompute_drt` puis `st.rerun()`). Changer circuit/mode DRT n'écrase
rien : un bandeau demande de relancer (`pages/A_eis.py:390-392`). **Les résultats survivent à un rerun** (state de
session) ; non à un rafraîchissement/redémarrage (constat 10). Les quatre onglets sont exécutés à chaque rerun
(comportement de `st.tabs`), `drt_diagnostic_rows` est calculé deux fois par groupe (`ui/tabs.py:508,621-623`).

**10. Export.** `export_drt_csv` (`exports/exporter.py:241`) : colonnes `electrode, label, concentration,
replicate_idx, drt_mode, ln_tau, ln_gamma` ; un spectre = une série (`replicate_idx` = indice ou `'avg'`) ; écrit
dans le ZIP à `export/drt/drt_values.csv` (l.457-461). Les diagnostics HMC et Rct sont exportés par
`results_table.replicate_rows` (CSV « résultats par réplicat »). Non exportés : bornes de crédibilité, durée (constat 12).

### D. Environnement (commandes réellement exécutées)

**11. Python / paquets.** Système : Python 3.13.16 ; `pip list` global : `numpy 2.5.3`, `pandas 3.0.5`,
`pydantic 2.13.5`, `PyYAML 6.0.1` seulement ; **absents globalement** : scipy, plotly, streamlit, lmfit, pystan,
cvxopt. `pip check` : « No broken requirements found ». J'ai donc créé un venv hors dépôt avec
`requirements-dev.txt` + `requirements-drt.txt` : `numpy 2.5.3`, `scipy 1.18.1`, `pandas 3.0.6`, `plotly 7.1.0`,
`streamlit 1.64.0`, `matplotlib 3.11.2`, `pydantic 2.13.5`, `cvxopt 1.3.3`, `cmdstanpy 1.3.0`, `pytest 9.1.1` ;
`pip check` propre. **`lmfit` et `pystan` ne sont ni installés ni importés nulle part** (la DRT passe par
`cmdstanpy`) : ils ne sont pas à ajouter.

**12. `requirements*.txt` vs imports.** Balayage AST de tout le dépôt : racines tierces importées par l'application
= `cmdstanpy, eclabfiles, galvani (optionnel, secours .mpr), matplotlib, numpy, pandas, plotly, pydantic, scipy,
streamlit, yaml` ; `cvxopt` par `drt/bayes_drt2/inversion.py`. Tout est couvert par `requirements.txt`,
`requirements-drt.txt`, `requirements-optional.txt`, `requirements-dev.txt` (tests : `pytest`). Aucun import non
déclaré, aucune dépendance déclarée inutilisée. Tout est épinglé.

**13. `bayes_drt2`.** Pas installé comme paquet : `python -c "import bayes_drt2.inversion"` →
`ModuleNotFoundError` — **c'est voulu**. Il est **vendorisé** sous `drt/bayes_drt2/` ;
`import drt.bayes_drt2.inversion` fonctionne (dans le venv). Origine : `jdhuang-csm/bayes-drt2` commit
`99d5b603…`, 3 patchs, SHA-256 publiés (`drt/PROVENANCE.md`) ; le remote git du dépôt est uniquement
`origin = J-Facc/Interface-EIS-CV-measurement` (copie, pas de sous-module). `numpy 2.5.3` : `np.trapz` **absent**,
`np.trapezoid` présent ; plus aucun `np.trapz` dans le dépôt (patch 1).

**14. Compilateur.** Cette machine est Linux : `g++`, `gcc`, `make` présents dans `/usr/bin` ; `C:\rtools44`
non applicable. À noter : le projet n'utilise **pas** RTools 4.4 — `launch.bat` / `setup_drt_bayesien.py`
installent la toolchain mingw de **RTools 4.0** via `install_cmdstan(compiler=True)` (CmdStan 2.36.0 dans
`C:\cmdstan`). Tester `C:\rtools44` ne serait donc pas pertinent. J'ai installé CmdStan 2.36.0 dans le scratchpad
(192 s) et compilé les modèles Stan sans erreur.

**15. Compilation et tests.**
- `compile()` sur les 89 `.py` suivis : **0 échec** (48 `SyntaxWarning` dans le code tiers, constat 13).
- `pytest tests/ -m "not slow"` (copie du dépôt, **sans** CmdStan) : **744 réussis, 17 sautés, 10 désélectionnés
  (`slow`)**, 0 échec, 360 s (mesure faussée : la compilation de CmdStan tournait en parallèle). Les 17 sautés sont
  tous des tests « DRT réelle » qui exigent CmdStan.
- Avec CmdStan 2.36.0 installé, les 5 fichiers DRT (`test_drt_engine`, `test_drt_fitresult_contract`,
  `test_pipeline`, `test_cmdstan_version`, `test_drt_failure_diagnostics`, `-m "not slow"`) : **139 réussis, 0 sauté**,
  169 s. Donc les 17 sautés passent une fois CmdStan présent.
- **Non exécutés** : les 10 tests `slow` (HMC, 2–5 min chacun).

**16. `launch.bat`** (lecture statique ; non exécutable sous Linux).
- `enabledelayedexpansion` : oui, et `ROOT` capturé **avant** l'activation (`launch.bat:28-30`).
- `cd` vers `eis_app` : oui, `cd /d "!APP_DIR!"` juste avant `streamlit run` (l.267).
- SSL/TLS : jamais désactivé ; TLS 1.2 ajouté (`-bor`) aux appels PowerShell ; pas de gestion de proxy propre
  (constat 20).
- Échec de `pip` : `requirements.txt` → message HORS LIGNE / ECHEC REEL, `exit /b 1` → `FAIL` + `pause`
  (l.427-437) ; `requirements-drt.txt` → code 2, l'app démarre **sans DRT** (l.438-449, 227-235) ; `.deps_hash`
  n'est écrit qu'en cas de succès, donc nouvel essai au lancement suivant. Échec de la vérification GitHub →
  lancement de la version locale (l.121-132). Échec du moteur DRT → jamais bloquant (l.455-489).
- `tests/test_launcher.py` (règles d'écriture, CRLF : `file launch.bat` → CRLF, 509 lignes) passe.

**17. `.github/workflows/validate.yml`.** Deux jobs, déclenchés sur `push main` et `pull_request`.
`validate` (Python 3.12, ubuntu) : `pip install -r requirements-dev.txt`, `compileall` (app, setup, core, fits, drt,
circuit, plotting, exports, ui, pages, tests), `pytest tests/ -v` — **sans cvxopt ni CmdStan** (les tests DRT réels y
sont sautés, comme reproduit ci-dessus). `drt` : installe dev + drt, met `~/.cmdstan` en cache, installe CmdStan
2.36.0, **garde-fou** `engine_available()`, puis lance seulement `test_pipeline`, `test_drt_engine`,
`test_drt_fitresult_contract` (HMC lents inclus, timeout 150 min). Lacunes : constat 9.

### E. Test réel (script hors dépôt `bench_fits.py`, spectre synthétique)

Spectre : Randles Rct = 3000 Ω, 60 points, 1e5→1e-1 Hz, bruit de structure d'Orazem, 3 réplicats. CPU de ce bac à
sable ; ordres de grandeur, pas des performances Windows.

| Étape | Durée | Résultat |
|---|---|---|
| `load_spectrum` (1 fichier) | 0,00 s | 60 points |
| Measurement model + KK (3 réplicats) | 0,85 s | — |
| Fit Orazem Randles (3 réplicats + moyenne) | 5,85 s | Rct = 2998 Ω, 3/3 convergés |
| `drt.engine.fit_drt` optimize, 1ᵉʳ appel | 26,6 s | inclut la compilation de `Series_pos.stan` |
| `drt.engine.fit_drt` optimize, à chaud | 0,95 s | Rct DRT = 3070 Ω (+2,3 %, cohérent avec le biais de ≈ +2 % annoncé), convergé, 0 alerte |
| 4 spectres DRT optimize | 3,74 s | — |
| **`Inverter().fit(mode='optimize')` direct (réglages amont)** | 27,9 s (dont ≈ 25 s de compilation de `Series.stan`) | **Rp = −5856 Ω** (non physique) |
| `run_pipeline` 9 fichiers (probe + 2 conc. × 3 réplicats), **sans** DRT | 23,5 s | — |
| `run_pipeline` 9 fichiers, DRT `optimize` | 35,8 s | +12 s pour 12 spectres ; `fit_results` = `['drt_bayes','orazem']` |

`mode='sample'` volontairement **non exécuté** (consigne). Une ligne de mon script (`g.label` sur
`ConcentrationGroup`) a levé une `AttributeError` après toutes les mesures : erreur de mon script, pas du dépôt.

---

## 3. Écarts entre documentation et code

| Document | Affirmation | Réalité |
|---|---|---|
| Demande d'audit | `ARCHITECTURE_v2.md`, `REFERENCE_MAINTENANCE.html` | Inexistants (ni arbre, ni historique). Doc réelle : `docs/ARCHITECTURE.md`. |
| Demande d'audit | `BaseFitModel`, `registry.py`, `drt_fit.py`, « 4 fits », « 7 figures » | Supprimés/inexistants ; 2 méthodes de fit ; 11 fonctions figure. |
| Demande d'audit | `pystan`, `lmfit`, `Rtools44` | Non utilisés ; la DRT passe par `cmdstanpy` + toolchain RTools 4.0 installée par le launcher. |
| `docs/ARCHITECTURE.md:69` | `core/cv_peaks.py` | Absent. |
| `tests/test_cmdstan_version.py:215` | « (job CI « drt ») » | Le job `drt` n'exécute pas ce fichier (`validate.yml:63`). |
| `docs/ARCHITECTURE.md`, `README.md`, `docs/DRT_BAYESIENNE.md` (modes, réglages, onglets, export, launcher, CI) | — | **Conformes au code** sur tous les points vérifiés (valeurs par défaut `optimize`/`sample`, versions épinglées, ordre du pipeline, règles de dépendances). |

---

## 4. Impact sur l'intégration bayes_drt2

### Ce qui est prêt (rien à construire)
- Moteur DRT durci, deux modes, diagnostics HMC, gardes qualité, extraction de Rct (convention Bissessur), échec
  isolé par spectre : `drt/engine.py`, `drt/diagnostics.py`.
- Contrat de résultat commun `FitResult` (tau, gamma, bandes, mode, diagnostics) et rangement par spectre.
- Orchestration par réplicat + moyenne, agrégation inter-réplicats, recalcul HMC ciblé (`recompute_drt`).
- UI complète (onglet DRT, diagnostics, recalcul), calibration DRT, export CSV/ZIP/YAML.
- Dépendances (`requirements-drt.txt`), installation CmdStan + toolchain + compilation (`launch.bat`,
  `setup_drt_bayesien.py`), CI avec job DRT réel, 139 tests DRT verts sur CmdStan 2.36.0.

### À corriger / trancher AVANT toute nouvelle tâche
1. **Confirmer la version de départ** (`git log` local = `060b97f` ?). Si votre copie est ancienne, la mettre à jour
   avant de réécrire quoi que ce soit.
2. **Réécrire les trois prompts** : ils décrivent une architecture qui n'existe plus (voir ci-dessous).
3. Décider du vrai besoin : s'il s'agit de **consolider** l'existant (constats 5-10, 12), c'est un chantier
   d'améliorations ciblées, pas une intégration.

### Ajustements pour les 3 prompts suivants

**Prompt backend**
- Ne **pas** créer `fits/base.py`, `fits/registry.py`, `fits/drt_fit.py` ni `BaseFitModel` (le test d'architecture
  échouerait) ; ne pas ajouter `bayes_drt2` à `requirements*.txt` (vendorisé) ; ne **jamais** appeler `Inverter()`
  nu — passer par `drt.engine.fit_drt` (constats 3-4).
- Travaux utiles : (a) étendre `DRTSettings` de `core/config.py:51` et `_drt_request` (`core/pipeline.py:155`) si l'on
  veut exposer chaînes/tirages/graine/`adapt_delta`/`nonneg`, **et** passer la config en `extra='forbid'` (ou au
  moins avertir) pour que les clés inconnues ne soient plus perdues en silence (constat 5) ; (b) chronométrer
  chaque `fit_drt` et ranger `duration_s` + horodatage dans `drt_diagnostics` (`drt/engine.py:580`), sans toucher à
  `FitResult` (constat 6) ; (c) isoler les erreurs par électrode dans `pages/A_eis.py` (constat 8).
- Tests : compléter les fichiers existants ; tout **nouveau** fichier de test exigeant CmdStan doit être ajouté à la
  liste du job `drt` (`validate.yml:63`), sinon il est sauté en CI (constat 9).

**Prompt UI**
- L'onglet DRT, le choix de mode (`pages/A_eis.py:316`) et le recalcul HMC existent. Ne rien reconstruire : traiter le
  vrai problème de l'UI, à savoir le calcul `sample` **bloquant** dans le script (constat 7) — exécution hors du
  thread du script (thread/sous-processus + barre de progression et annulation) ou, à défaut, limiter `sample` au
  recalcul spectre par spectre déjà proposé (`ui/tabs.py:574-589`).
- Persistance : si les résultats doivent survivre à un rafraîchissement, ajouter une sauvegarde/rechargement des
  sessions (constat 10) ; `st.session_state` ne suffit pas.
- Éviter le double calcul de `drt_diagnostic_rows` par rerun (`ui/tabs.py:508,621`).

**Prompt launcher + doc**
- Le launcher installe déjà CmdStan 2.36.0 et la toolchain ; **ne pas** ajouter de test `C:\rtools44` (le projet
  s'appuie sur RTools 4.0 via cmdstanpy). Ajouter, si souhaité, un conseil proxy/TLS dans le message d'échec de
  `pip` (`launch.bat:427-437`) (constat 20). Toute correction doit être **fusionnée sur `main`** (`launch.bat:35`).
- Doc : corriger `docs/ARCHITECTURE.md:69` (retirer `cv_peaks.py`), supprimer toute référence à
  `ARCHITECTURE_v2.md` / `REFERENCE_MAINTENANCE.html`, documenter `duration_s` si ajouté.
- CI : ajouter `tests/test_cmdstan_version.py` (et `test_drt_failure_diagnostics.py`) au job `drt` ; un runner Windows
  est le seul moyen d'exercer le chemin mingw/accents que les utilisateurs empruntent.

---

## 5. Verdict

**NO-GO pour l'intégration telle que cadrée.** Raison : la prémisse est fausse. `bayes_drt2` (modes `optimize` et
`sample`) est déjà intégré et validé dans ce dépôt, et l'architecture « plugin `BaseFitModel` + registre » décrite
par la demande a été supprimée volontairement (`e0c5afa`) ; un test (`tests/test_architecture.py:75-88`) interdit de
la recréer. Lancer les trois prompts en l'état dupliquerait du code testé et réintroduirait les défauts déjà
corrigés (appel nu à `Inverter()` : Rp négatif, état partagé).

**GO pour un chantier recadré de consolidation** (constats 5, 6, 7, 8, 9, 10, 12), sans blocage technique :
l'environnement est sain (744 tests verts sans CmdStan, 139 DRT verts avec), les dépendances sont complètes et
épinglées, et le code respecte ses règles de modularité.

## Limites de l'audit
- Aucune exécution sous Windows : `launch.bat`, le chemin mingw/RTools et l'installation réelle du launcher n'ont
  été examinés que statiquement (+ `tests/test_launcher.py`, qui passe).
- `mode='sample'` et les 10 tests `slow` non exécutés ; les durées HMC citées sont celles de la documentation du dépôt.
- Chronométrages sur données synthétiques, sur un seul bac à sable Linux.
- Aucune donnée EC-Lab réelle n'est présente dans le dépôt ; le « spectre d'exemple » vient de `tests/synthetic_data.py`.
