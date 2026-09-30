# AUDIT — EIS Analyzer (Interface-EIS-CV-measurement) avant refonte

> Audit exhaustif **en lecture seule** du code réel, préparé pour la refonte annoncée
> (fit unique « méthode Orazem » sur circuit défini par l'utilisateur, DRT via `bayes_drt2`,
> suppression du code mort, calculs transparents et sourcés).
>
> - **Date** : 2026-09-30 · **HEAD audité** : `b91b8ba` (merge PR #80, 2026-09-11) · branche `claude/hopeful-fermi-285uqw`
> - **Aucun fichier du dépôt n'a été modifié**, à l'exception de ce `AUDIT.md` (qui **remplace** l'audit du
>   2026-07-13, resté récupérable : `git show ae89a82:AUDIT.md` ; ses conclusions utiles sont résumées en Annexe B).
> - **Méthode** : lecture intégrale du code de production (`app.py`, `core/`, `fits/`, `plotting/`, `exports/`,
>   `ui/`, `pages/`, `setup_drt_bayesien.*`, lanceurs, CI, `config/`), lecture des docs, scan statique
>   (pyflakes, vulture, AST maison pour les usages par symbole), **exécution réelle** des tests, de la couverture
>   et de scénarios de reproduction dans une **copie hors dépôt** (voir Annexe A — reproductibilité).
> - **Convention de citation** : `chemin:ligne` = numéro de ligne du fichier à `b91b8ba`. Les constats **mesurés**
>   sont marqués 🔬 (avec la façon de les reproduire), les constats **lus dans le code** sont marqués 📖, les
>   points **non vérifiables** ici sont marqués ⚠️.

---

## 0. Lecture préalable — les prémisses de la mission ne correspondent pas au dépôt

L'énoncé de la mission décrit une architecture antérieure. **À corriger avant de planifier la refonte**
(le rapport `AUDIT_REPORT.md` du 2026-07-30 faisait déjà ce constat ; il est confirmé et complété ici) :

| Prémisse de l'énoncé | Réalité observée dans le code | Preuve |
|---|---|---|
| « 4 modèles de fit, à réduire à 1 » | **Il n'y a déjà plus que 2 plugins** : `randles_full` (fit paramétrique) et `drt_bayes` (DRT). | `fits/registry.py:18-53` (skip L29) ; `fits/randles_full.py:33` ; `fits/drt_fit.py:330` |
| Fichiers `randles_classique.py`, `randles_contraint.py`, `circulaire_fit.py` | **N'ont jamais existé dans l'historique git** (`git log --all --name-only` : aucune occurrence). Existaient : `circular_fit.py`, `randles_constrained.py` (supprimés au commit `e315f30`, 2026-06-22) et `drt_tikhonov.py` / `drt_fft.py` / `_pydrttools/` (supprimés au commit `6806aba`, 2026-07-16). | `git log --diff-filter=D --name-only -- fits` |
| « Fit unique méthode Orazem » à créer | **Déjà en place** : `randles_full` est pondéré *uniquement* par la structure d'erreur d'Orazem (`w = 1/σ²`, `absolute_sigma` toujours vrai). Ce qui n'existe pas : le **circuit défini par l'utilisateur** (le circuit est codé en dur : 8 paramètres). | `fits/weighting.py:1-25`, `fits/randles_full.py:17,107-147` |
| « Remplacer le module DRT par `bayes_drt2` (jdhuang-csm) » | **Déjà fait** : `fits/drt_fit.py` est un wrapper de `vendor/bayes_drt2/` (v0.2 vendorée, BSD-3, sha256 vérifiés). L'ancien DRT (Tikhonov/FFT/pyDRTtools) n'existe plus. | `fits/drt_fit.py:1-40,60-66` ; `vendor/README.md` |
| Répertoires `tests/`, `ui/`, `exports/` | Existent. Mais `ui/` ne contient qu'**un** fichier (`ui/tabs.py`) et la « logique UI » est surtout dans `pages/` (2 520 lignes). | inventaire §1 |
| Documents de référence `REFERENCE_MAINTENANCE.html` et `ARCHITECTURE_v2.md` | **Ces noms n'existent pas** (ni dans l'arbre, ni dans l'historique). Correspondants réels : `Reference_maintenance` (HTML **sans extension**, « v2 · 29/05/2026 », partiellement patché) et `ARCHITECTURE.md` (« MàJ 13/07/2026 »). Le fichier `project` (« Brief de refonte v3 », 08/06/2026) cite bien les deux noms d'origine (`project:12-13`). `ARCHITECTURE_v3.md`, `REFERENCE_MAINTENANCE_v3.html`, `SCIENTIFIC_METHODS.md` promis par `AUDIT_REPORT.md:128,343-349` **n'ont jamais été livrés**. | `git log --all -- ARCHITECTURE_v2.md …` → vide |
| Lanceur `launch.bat` « avec étape cmdstan à ajouter » | Une préparation CmdStan **existe déjà** mais ailleurs : `app.py:50-70` appelle `setup_drt_bayesien.ensure_drt_ready()` au premier lancement, et `setup_drt_bayesien.bat` fait l'installation manuelle. `launch.bat` n'installe **ni** `requirements-drt.txt` **ni** cmdstan. | §7 |

**Conséquence** : la refonte n'est pas « supprimer 3 modèles + brancher bayes_drt2 » mais :
(a) **généraliser** `randles_full` en moteur de fit sur circuit utilisateur, (b) **fiabiliser** la DRT déjà branchée
(elle est aujourd'hui *silencieusement fausse* avec les réglages actuels — §4), (c) nettoyer le résiduel (config morte,
modèles fantômes dans l'UI, docs), (d) traiter les défauts de robustesse relevés ici (§10).

---

## Sommaire
1. Inventaire fichier par fichier · 2. Code mort · 3. Dépendances · 4. Module DRT actuel · 5. Modules de fit,
`base.py`, `registry.py` · 6. Couplage des couches · 7. Lanceur `launch.bat` · 8. Tests · 9. Incohérences code ↔ docs ·
10. Conclusion priorisée · Annexes (A reproductibilité et scripts, B état des audits précédents)


---

## 1. Inventaire fichier par fichier

Légende des usages : *prod* = appelé depuis un autre fichier de production ; *tests* = appelé seulement par `tests/` ;
« ∅ » = aucune référence hors du fichier. Les usages ont été obtenus par recherche de mots entiers sur tout le dépôt
(hors `vendor/`), puis vérifiés à la main pour les cas ambigus. Volume : ≈ 10 200 lignes Python de production
(dont 2 520 en `pages/`) + ≈ 1 700 lignes de tests, hors `vendor/` (6 966 lignes Python vendorées + 13 modèles Stan).

### 1.1 Racine, configuration, CI

| Fichier | Lignes | Rôle réel observé | Remarques |
|---|--:|---|---|
| `app.py` | 95 | Point d'entrée Streamlit : `set_page_config` (L11), init de 9 clés `session_state` (L22-40), **bootstrap CmdStan + `Series.stan` au premier lancement** via `st.cache_resource` (L50-70), navigation `st.navigation` de 6 pages (L76-93). | 2 des 9 clés sont mortes (`eis_session` L25, `eis_validation` L27) ; page « Prédiction » = stub. |
| `config/default.yaml` | 68 | Paramètres YAML validés par Pydantic (`core/config.py`). | **Sections entières mortes** — §2.3. |
| `pytest.ini` | 3 | Déclare le marqueur `slow`. | |
| `.github/workflows/validate.yml` | 21 | CI : Python 3.12, `pip install -r requirements.txt`, `compileall app.py core fits plotting exports ui pages`, `pytest tests/`. | Ne compile ni `vendor/` ni `tests/` ni `setup_drt_bayesien.py` ; **ne teste pas la DRT réelle** (pas de cvxopt/cmdstan) ; pas de lint ni de couverture (§8). |
| `.gitignore` | 19 | Ignore `.venv/`, `logs/`, `sessions/`, `config/error_structure.json`… | **Manque** `token.txt`, `.version`, `venv/`, `eis_app/` (§7). |
| `requirements*.txt` | 15/20/12 | Dépendances dures / extra DRT / extra `.mpr` (galvani). | §3. |
| `launch.bat` / `launch_app.bat` / `launch_app.sh` | 125/27/18 | Lanceur Windows auto-update + venv (repo privé + token) ; lanceur Windows simple ; lanceur Unix. | §7. |
| `setup_drt_bayesien.py` / `.bat` | 280/126 | Installe/enregistre CmdStan (+ toolchain), pré-compile `Series.stan` et `Series_pos.stan` ; `ensure_drt_ready()` est appelée par `app.py`. | Le `.bat` (L10-12) affirme « JAMAIS appelé au démarrage » alors que `app.py` fait l'équivalent (§7.4). |
| `project` (655 lignes, **sans extension**) | 655 | Texte Markdown « Brief de refonte v3 » (08/06/2026) — cahier des charges d'une refonte **UI déjà réalisée**. | Fichier orphelin ; nom trompeur (n'est pas du Python). |
| `Reference_maintenance` (**sans extension**) | 789 | Page HTML « Référence Maintenance v2 » (29/05/2026). | Obsolète en grande partie (§9). |
| `ARCHITECTURE.md`, `METHODES.md`, `MEASUREMENT_MODEL.md`, `README.md`, `AUDIT.md`, `AUDIT_REPORT.md`, `THIRD_PARTY_LICENSES.md`, `vendor/README.md` | 361/985/103/77/—/362/15/105 | Documentation. | §9. |

### 1.2 `core/` — logique métier (aucun import Streamlit : vérifié)

| Fichier | Lignes | Rôle réel observé | Symboles publics → usages |
|---|--:|---|---|
| `models.py` | 163 | Dataclasses `EISSpectrum` (L9), `CVCurve` (L47), `FitResult` (L69), `ConcentrationGroup` (L132), `EISSession` (L146). | `CVCurve` : prod `cv_loader.py` seulement pour `load_cv_curve` (lui-même tests seulement). Champs morts : §2.4. |
| `loader.py` | 408 | Import EIS : parseur EC-Lab robuste en priorité (`parse_robust` L48) puis repli pandas (L295-339) ; nettoyage (NaN, fréquences parasites 50/100 Hz, correction de signe, tri HF→BF) (`_clean_spectrum` L156) ; reconstruction d'axe fréquence si absent (L36) ; `average_replicates` (L342) qui calcule aussi σ inter-réplicats (L387-390). | `load_spectrum` (prod : pipeline, experiment_io, exporter, pages ×3 ; tests) ; `average_replicates` (prod ×5). `parse_robust`/`_detect_separator` sont aussi importés par `cv_loader.py`. |
| `robust_loader.py` | 208 | Lecture ASCII EC-Lab : encodages, délimiteur, en-tête, mapping des colonnes **par nom** (`_classify` L102), signe de Im(Z), unités de courant, discrimination EIS/CV (`parse_eclab_file` L130). Sans dépendance au reste du projet. | `parse_eclab_file` (prod : loader ; tests) ; `ParsedFile`. |
| `cv_loader.py` | 173 | Import CV : parseur robuste puis repli pandas ; **tri par potentiel croissant** (L74, L118) ; moyenne de réplicats par interpolation (L151). | `load_cv_file` (prod ×4), `average_cv_replicates` (prod ×4). `load_cv_curve` (L131) : **tests seulement**. |
| `cv_models.py` | 31 | `CVScan`, `CVConcentrationGroup`, `CVSession`. | Tous utilisés. |
| `cv_peaks.py` | 61 | Pics redox : Savitzky-Golay puis argmax/argmin. | `detect_redox_peaks` (prod : cv_plots, ui/tabs). **Aucun test** (21 % de couverture). |
| `cv_pipeline.py` | 65 | Orchestration CV : groupe par (step, conc.), moyenne, `delta_signal = \|I_probe−I_c\|/\|I_probe\|` par interpolation (L316-322). | `run_cv_pipeline` (prod : B_cv). |
| `pipeline.py` | 350 | Orchestration EIS : chargement, moyenne, **caractérisation Orazem préalable** (L47), KK (L83), fits séquentiels par modèle sur bare/probe/concentrations (L215-262), fits de réplicats (L188-208, **DRT exclue**), validation de réplicats (L15) ; `recompute_drt` (L306) pour le HMC à la demande. | `run_pipeline` (prod : A_eis), `recompute_drt` (prod : ui/tabs), `_iter_session_spectra`/`_resolve_spectrum` (internes). **0 % de couverture.** |
| `validator.py` | 402 | Validation KK par réplicat (`validate_spectrum` L100), plage valide commune, détection de dérive inter-réplicats (`_detect_drift` L328), σ empirique (`_compute_sigma` L371). | `validate_replicate_group` (prod : pipeline, 1_pretraitement) ; `ValidationResult` (prod : plotting/kk_plots, ui/tabs). **0 % de couverture.** `ValidationResult.sigma_*` jamais lu (§2.4). |
| `calibration.py` | 178 | Source unique des régressions de calibration : signal normalisé (L58), log-log (L92), CV (L133), « tous modèles » (L164). | `compute_calibration_all` (exporter, eis_plots), `compute_calibration_loglog` (eis_plots), `compute_cv_calibration` (exporter). `compute_calibration` : interne + tests. |
| `config.py` | 163 | Schéma Pydantic de `config/default.yaml` ; `load_config` (L138), `config_to_dict` (L154). | Sections `physics`, `geometry`, `conditions`, `export` **jamais lues** (§2.3). |
| `experiment_io.py` | 552 | Sauvegarde/chargement d'une expérience en ZIP (`save_experiment` L37, `load_experiment` L197, formats v1 et v2), exclusions de réplicats (`apply_exclusions` L379) et de points (`apply_point_exclusions` L432). | `zip_experiment` (L190) : alias **jamais utilisé**. `_sig_types_for_mode` (L547) **dupliqué** dans `pages/0_import.py:468`. 16 % de couverture (seule `apply_point_exclusions` testée). |
| `mpr_converter.py` | 102 | Convertit un `.mpr` BioLogic en CSV texte (eclabfiles, galvani en secours). | `is_mpr`, `mpr_to_csv_bytes` (prod : 0_import, experiment_io). |
| `logger.py` | 38 | Logging fichier + console. | **Effet de bord à l'import** : crée `logs/` (L8) — tout module important `core.logger` écrit dans l'arborescence de l'app. |

### 1.3 `fits/` — modèles et méthodes numériques

| Fichier | Lignes | Rôle réel observé | Symboles publics → usages |
|---|--:|---|---|
| `base.py` | 51 | ABC `BaseFitModel` : `fit(spectrum, config, weights=None)`, `initial_guess`, `bounds`. | Sous-classée par `RandlesFullModel`, `DRTBayesModel`. `initial_guess`/`bounds` de `DRTBayesModel` retournent `{}`/`({},{})` (L341-347). |
| `registry.py` | 74 | Découverte automatique des sous-classes de `BaseFitModel` par `pkgutil` ; mémorise les échecs d'import (`discovery_errors`). | `all_models` (pipeline, A_eis), `get_model` (pipeline), `discovery_errors` (A_eis). `skip` (L29) n'exclut pas `error_structure` (importé pour rien). |
| `physics.py` | 110 | Fonctions physiques : `Z_D` (diffusion bornée), `Z_randles_full`, `Rct_bare_theory`, `Cdl_brug`, `theta_EIS`. | `Z_D`, `Z_randles_full` : prod. **`Rct_bare_theory` (L66) : ∅. `Cdl_brug` (L84), `theta_EIS` (L100) : tests seulement.** |
| `weighting.py` | 75 | Poids du CNLS : `resolve_weights` (L40) → `w = 1/σ²`, avec `σ_i = σ_struct/√n_replicates` (L68-73). | prod : randles_full. |
| `error_structure.py` | 451 | Structure d'erreur d'Orazem : dataclass `ErrorStructure`, σ empirique (directe L118 ou par résidus Lin-KK L143), régression NNLS des coefficients (L189), **persistance JSON append-only** (L347) et rechargement du dernier (L385), `resolve_error_structure` (L407). | prod : pipeline, weighting. `field` importé inutilement (L44). |
| `randles_full.py` | 268 | Plugin `randles_full` : 8 paramètres, `least_squares` TRF borné, covariance `(JᵀJ)⁻¹`, χ² réduit + intervalle attendu, garde-fous (non convergé, χ² hors intervalle, résidu relatif > 10 %, paramètre collé à une borne). | Chargé dynamiquement par le registre ; classe utilisée par 4 fichiers de tests. |
| `kk_validation.py` | 132 | Lin-KK (circuit de Voigt linéaire, M fixe) `lin_kk` (L22), `kramers_kronig_check` (L86), et un **wrapper de compatibilité** `linKK` (L126) dont `c` et `fit_type` sont ignorés. | `lin_kk` : error_structure ; `kramers_kronig_check` : pipeline ; `linKK` : validator. |
| `drt_fit.py` | 359 | Wrapper de `vendor.bayes_drt2.Inverter` : `fit_drt` (L208), extraction Rct par pic (L114), plugin `DRTBayesModel` (L322). Import protégé (L60-66). | `bayes_available`/`import_error`/`cmdstan_available` (app, ui/tabs). `DRTBayesModel.predict` (L356) : ∅. |

### 1.4 `plotting/`, `exports/`, `ui/`

| Fichier | Lignes | Rôle réel observé | Symboles sans usage |
|---|--:|---|---|
| `plotting/eis_plots.py` | 1083 | ~20 constructeurs de figures Plotly EIS + 2 fenêtres matplotlib (`plt.show()` L1049, L1083). | `bode_figure` (L327), `kk_figure` (L755), `drt_reconstruction_figure` (L809) : **∅**. Variable inutile `rcts` (L849). Thème `"light"` codé en dur (16 appels `get_theme`/`apply_theme_to_figure`). |
| `plotting/cv_plots.py` | 464 | Figures CV + 2 fenêtres matplotlib. **Recalcule `linregress` en local** (L157, 224, 412, 450). | `cv_figure_electrode` (L22), `open_cv_calibration_matplotlib_window` (L432) : **∅** ; `cv_calibration_figure` (L203) : tests seulement. |
| `plotting/kk_plots.py` | 254 | Figures de résidus KK et tableau de synthèse. | Imports `List`, `Optional` inutiles (L16), variable `n_rep` inutile (L73). **0 % de couverture.** |
| `plotting/theme.py` | 55 | Palettes clair/sombre. | Le mode `dark` n'est jamais sélectionné. |
| `exports/exporter.py` | 517 | Exports CSV/YAML/ZIP. | `export_spectra_csv` (L57), `export_figure_html` (L84), `export_figure_png` (L89) : **∅** ; `export_cv_calibration_csv` (L168) : tests + import inutile ; `export_cv_calibration_csv_from_result` (L192) : branche morte (§2.6). |
| `ui/tabs.py` | 554 | Rendu des onglets EIS (KK, DRT, Reconstructions, Calibration) et CV (4 onglets). | 6 imports inutiles (L5, 9, 10, 24 ×2, 34) ; variable `avg_spectrum` inutile (L368). Aucun test. |

### 1.5 `pages/`

| Fichier | Lignes | Rôle réel observé |
|---|--:|---|
| `0_import.py` | 598 | Saisie de l'expérience (mode EIS/CV/les deux, concentrations, électrodes, réplicats), envoi des fichiers (conversion `.mpr`), chargement/sauvegarde ZIP. Produit `session_state["experiment"]` (dict de `BytesIO`). |
| `1_pretraitement.py` | 1220 | Exclusion de réplicats, éditeur de suppression de points, superpositions Nyquist/CV, **validation KK** (`_run_kk_validation` L754), export de graphiques (fenêtre matplotlib), production de `experiment_clean`. Contient de la logique métier (`_apply_point_mask` L654, `_combine_eis_two` L934…). `_collect_deleted_points` (L902) : ∅ ; imports inutiles `datetime`, `List` (L18-19). |
| `A_eis.py` | 487 | Page EIS : sélection des modèles (**dont 2 fantômes**, L350-354), paramètres « physiques » (**inertes**, L371-379), lancement de `run_pipeline` par électrode, diagnostics, 3 Nyquist, vue normalisée (**logique métier dans la page**, L178-241), `render_eis_tabs`. |
| `B_cv.py` | 129 | Page CV : assemble les assignations, `run_cv_pipeline`, `render_cv_tabs`. Accès strict `experiment["calibration"]["cv"]` (L22, sans `.get`) ; `run_cv_pipeline` (L118) non protégé par `try` contrairement à la page EIS. |
| `E_export.py` | 81 | Boutons de téléchargement. **Bouton « Paramètres fit Randles » non filtré → CSV mal aligné** (§2.5, bug B-EXP). |
| `D_inference.py` | 5 | **Stub** (« sera développée ultérieurement »), pourtant exposé dans la navigation (`app.py:89-91`). |

### 1.6 `vendor/bayes_drt2/` (code tiers, BSD-3)

`inversion.py` (4 645 lignes, `Inverter`), `matrices.py`, `peak_fit.py`, `plotting.py`, `file_load.py`, `utils.py`,
+ 13 modèles Stan. 📖 Les sha256 des 6 fichiers « non modifiés » de `vendor/README.md:74-81` **ont été recalculés : identiques**.
Seuls 2 fichiers sont patchés (`matrices.py`, `peak_fit.py` : `np.trapz → np.trapezoid`) ; ⚠️ le diff par rapport à
l'amont n'est pas vérifiable ici (pas d'accès au commit amont — le README le reconnaît : « commit non récupérable »).
Imports lourds **au niveau module** dans `inversion.py:1-12` : `cvxopt`, `cmdstanpy`, `matplotlib.pyplot`.
`vendor/README.md:4` parle d'« EduGradia/Interface » (copier-coller d'un autre projet). Le code vendoré émet des
`DeprecationWarning` (séquences d'échappement invalides, ex. `inversion.py:1453`) qui deviennent des `SyntaxWarning`
en Python 3.12.

---

## 2. Code mort, résidus et incohérences internes

Méthodes : (1) pyflakes, (2) vulture (confiance ≥ 60 %, faux positifs Pydantic écartés à la main), (3) recherche de
mots entiers de chaque symbole public sur tout le dépôt, (4) lecture. **Aucun** `TODO`/`FIXME`/`XXX`/`HACK` n'a été
trouvé et **aucun bloc de code commenté** n'a été détecté (regex sur `# def|class|import|return|if…|x = f(…)`) : le dépôt
est propre sur ces deux points. Restent des commentaires historiques (§2.7).

### 2.1 Fonctions, classes, constantes sans appelant

| # | Élément | Statut | Action possible |
|--:|---|---|---|
| 1 | `fits/physics.py:66` `Rct_bare_theory` | ∅ (aucune référence, tests compris) | Supprimer (ou réutiliser si θ_EIS est rétabli) |
| 2 | `fits/physics.py:84` `Cdl_brug`, `:100` `theta_EIS` | tests seulement (`tests/test_physics.py`) | Idem — cf. §9 (θ_EIS documenté mais non calculé) |
| 3 | `fits/drt_fit.py:356` `DRTBayesModel.predict` | ∅ ; de plus refait un fit complet | Supprimer |
| 4 | `fits/kk_validation.py:126` `linKK` (wrapper de compat. de la lib `impedance` retirée) | 1 appelant (`core/validator.py:121-139`) ; paramètres `c` et `fit_type` **ignorés** (vulture 100 %) | Appeler `lin_kk` directement |
| 5 | `exports/exporter.py:57` `export_spectra_csv` | ∅ | Supprimer |
| 6 | `exports/exporter.py:84` `export_figure_html`, `:89` `export_figure_png` | ∅ — et c'est le **seul** usage de `kaleido` (§3) | Supprimer |
| 7 | `exports/exporter.py:168` `export_cv_calibration_csv` | tests + import inutile (`ui/tabs.py:34`) + branche morte `:504` | Supprimer avec la branche |
| 8 | `exports/exporter.py:192` `export_cv_calibration_csv_from_result` | atteignable seulement par `:500` (`isinstance(cv_session, dict) and "groups" in cv_session`), **jamais vraie** : `cv_sessions` est `{int: CVSession}` (`pages/B_cv.py:114-123`) ; la docstring vise `session_state['cv_result']`/`['cv_ols']`, clés qu'aucune page ne produit | Supprimer |
| 9 | `plotting/eis_plots.py:327` `bode_figure`, `:755` `kk_figure`, `:809` `drt_reconstruction_figure` | ∅ (les deux derniers sont importés sans être appelés : `ui/tabs.py:10`) | Supprimer |
| 10 | `plotting/cv_plots.py:22` `cv_figure_electrode`, `:432` `open_cv_calibration_matplotlib_window` | ∅ | Supprimer |
| 11 | `plotting/cv_plots.py:203` `cv_calibration_figure` | tests seulement (`tests/test_cv.py`) ; import inutile `ui/tabs.py:24` | Supprimer avec ses tests, ou fusionner avec la variante `_multi` |
| 12 | `core/cv_loader.py:131` `load_cv_curve` + `core/models.py:47` `CVCurve` | tests seulement (`tests/test_loader.py:281`) | Supprimer |
| 13 | `core/experiment_io.py:190` `zip_experiment` | alias jamais utilisé (l'ARCHITECTURE.md:149 le mentionne) | Supprimer |
| 14 | `pages/1_pretraitement.py:902` `_collect_deleted_points` | ∅ | Supprimer |
| 15 | `tests/test_fits.py:49` `_zarc_spectrum` | ∅ | Supprimer |
| 16 | `pages/D_inference.py` (5 lignes) + entrée de navigation `app.py:89-91` | Stub visible de l'utilisateur | Retirer de la navigation tant qu'inexistant |
| 17 | Attributs de classe `description` (`fits/randles_full.py:35`, `fits/drt_fit.py:334`), `method`, `display_name` (`fits/drt_fit.py:332-333`), `BaseFitModel.description` (`fits/base.py:16`) | aucun lecteur | Utiliser dans l'UI ou supprimer |
| 18 | `fits/registry.py:29` : `skip` n'inclut pas `error_structure` | `error_structure.py` est importé et inspecté à chaque découverte | Ajouter au `skip` |

Faux positifs vulture à **ne pas** supprimer : validateurs Pydantic `core/config.py:80,87`, classes `RandlesFullModel`/`DRTBayesModel`
(découvertes par `pkgutil`/`dir()`), `ParsedFile.f/Zre/Zim` (propriétés).

### 2.2 Imports et variables inutilisés (pyflakes/vulture)

`fits/error_structure.py:44` (`field`) · `fits/weighting.py:32` (ré-export volontaire, `# noqa`) ·
`plotting/kk_plots.py:16` (`List`, `Optional`) · `plotting/kk_plots.py:73` (`n_rep`) · `plotting/eis_plots.py:849` (`rcts`) ·
`ui/tabs.py:5` (`numpy`), `:9` (`CVSession`), `:10` (`drt_reconstruction_figure`), `:24` (`cv_calibration_figure`,
`open_cv_calibration_matplotlib_window`), `:34` (`export_cv_calibration_csv`), `:368` (`avg_spectrum`) ·
`pages/1_pretraitement.py:18-19` (`datetime`, `List`) · `tests/test_cv.py:8` (`CVScan`) · `tests/test_diagnostics.py:39` (`Z`).
(`setup_drt_bayesien.py:165,240` : `import cmdstanpy` sert de test de présence — légitime.)

### 2.3 Configuration morte (YAML + Pydantic) et paramètres inertes dans l'UI

| Clé | YAML | Pydantic | Consommateur ? |
|---|---|---|---|
| `physics.*` (T, F, R, n, C0, D_FeIII, D_FeII) | `default.yaml:2-11` | `core/config.py:10-17` | **Aucun** (le seul code physique qui les utiliserait, `Rct_bare_theory`, est ∅) |
| `geometry.*` (xe, h, d, S_WE) | `:12-18` | `:20-24` | **Aucun** |
| `conditions.Fv` | `:19-24` | `:27-28` | **Aucun** |
| `fit.n_monte_carlo` | `:45` | `:104` | **Aucun** |
| `fit.drt_wiener_W`, `fit.drt_n_z` | `:60-61` | `:107-108` | **Aucun** (vestiges du DRT Wiener/FFT supprimé) ; injectés par `tests/test_fits.py:67-71` |
| `export.dpi/fig_width/fig_height` | `:65-68` | `:112-115` | **Aucun** |
| ✅ utilisées : `acquisition.*` (loader), `fit.error_structure.*`, `fit.n_freqs_parasites`, `fit.tol_parasites` (`core/loader.py:236-237`), `fit.max_iter` (`randles_full.py:147`), `fit.bounds_randles_full`, `fit.drt.mode` (`drt_fit.py:181-189`), `fit.drt_kk_tol` (`kk_validation.py`) | | | |

**Conséquence UI** : `pages/A_eis.py:371-379` affiche six champs « Paramètres physiques » (Fv, xe, h, d, T, C0) →
`_merge_overrides` (`A_eis.py:69-87`) les écrit dans `cfg["physics"|"geometry"|"conditions"]` → **personne ne les lit**.
L'utilisateur peut les modifier sans aucun effet sur les résultats. 📖

**Bornes triplées** : les bornes de `randles_full` sont écrites trois fois — `fits/randles_full.py:79-88` (défauts en dur),
`core/config.py:31-45` (défauts Pydantic) et `config/default.yaml:46-54` — avec un test qui vérifie leur cohérence
(`tests/test_config.py`). À réduire à une source pour le futur moteur de circuit.

### 2.4 Champs de dataclass sans consommateur

`EISSpectrum.validation`, `.f_min_valid`, `.f_max_valid` (`core/models.py:34,39,40`) · `FitResult.Rct_sigma` (`:93`) ·
`FitResult.drt_S`, `.drt_lnGamma` (`:105-106` ; **produits** par `drt_fit.py:309-310`, jamais lus) ·
`FitResult.kk_passed`, `.kk_residuals` (`:112-113` ; **écrits** par `core/pipeline.py:223-224,243-244`, jamais lus par l'UI ni les exports) ·
`FitResult.chi2_is_valid_test`, `.chi2_reduced_ci` (`:126,128` ; écrits par `randles_full.py:198-202`, aucun lecteur — la garde
correspondante est déjà exprimée par `warnings`) · `FitResult.params_std` (`:78` ; ne sert qu'à dériver `Rct_std`) ·
`ValidationResult.sigma_re/sigma_im` (`core/validator.py:73-74`, calculés par `_compute_sigma` `:371-402`, **jamais lus** ; ce calcul
applique en outre un plancher de 0,1 % du module `:396-400` que le loader a explicitement banni pour son propre σ, `loader.py:381-386`).

### 2.5 Clés `session_state` mortes ou non réinitialisées

- `eis_session` (`app.py:25`) et `eis_validation` (`app.py:27`) : **initialisées puis remises à `None`**
  (`pages/0_import.py:116`, `pages/1_pretraitement.py:886`) mais **jamais lues**. Les vraies clés sont `eis_sessions`,
  `eis_validations`, `eis_normalized`, `cv_sessions`.
- 🔬 **Conséquence (bug B-STATE)** : ces vraies clés ne sont **jamais remises à zéro** quand l'utilisateur recharge un
  ZIP (`0_import.py:116`), revalide l'import (`0_import.py:568-575`) ou re-valide le prétraitement (`1_pretraitement.py:884-888`).
  `A_eis.py:413` (`if not st.session_state.get("eis_sessions")`) et `B_cv.py:111` (`if "cv_sessions" not in …`) réutilisent
  donc l'**ancienne analyse** tant que l'utilisateur ne clique pas sur « Relancer l'analyse » (`A_eis.py:408`, `B_cv.py:107`) :
  résultats affichés/exportés sur d'anciennes données, sans avertissement. (Lu dans le code ; non rejoué dans un navigateur.)
- **Variante plantante (B-STATE-b)** : la validation d'un nouvel import (`0_import.py:552-575`) supprime `experiment_clean` mais **ne remet pas
  `preprocessing_done` à `False`** (seul le chargement d'un ZIP le fait, `:126`). Après un premier prétraitement, ré-importer puis ouvrir « EIS seule »
  passe la garde `A_eis.py:330` et déclenche `KeyError` sur `st.session_state["experiment_clean"]` (`A_eis.py:340`, idem `B_cv.py:105`). (Lu dans le code ; non rejoué.)

### 2.6 Branches mortes ou inatteignables

- `exports/exporter.py:500-505` : deux branches sur trois de la sélection de l'export CV (§2.1 #7-8).
- `exports/exporter.py:127` : filtre `_lc_*`/`_gcv_*` sur les paramètres — préfixes de l'ancien DRT Tikhonov (L-curve/GCV), plus produits.
- `fits/randles_full.py:128-129` : « tableau unique hérité » pour `weights` ; le pipeline ne passe jamais `weights` (`pipeline.py:202-204`),
  seuls 2 tests le font (`tests/test_fits.py:169`, `tests/test_measurement_model.py:186`).
- `pages/A_eis.py:337`, `pages/B_cv.py:102` : `return` après `st.stop()` (qui lève) — inatteignable.
- `plotting/theme.py:13-21` : palette « dark » jamais sélectionnée (tous les appels passent `"light"`).
- `fits/randles_full.py:198` : `chi2_is_valid_test = True` constant (drapeau toujours vrai).
- `pages/A_eis.py:350-369` : voir §5.3 — deux modèles fantômes dans l'UI.

### 2.7 Commentaires et docstrings périmés

Marqueurs « NOTE (ajout hors périmètre initial …) » : `core/models.py:138,154`, `core/pipeline.py:132` · docstring de test parlant
de « FFT/Wiener DRT » : `tests/test_fits.py:30` · `pages/0_import.py:4` (« les autres pages lisent `experiment` » : elles lisent
`experiment_clean`) · `core/validator.py:39-40` (commentaire « (Zre_fit − Zre_exp) » alors que `lin_kk` calcule `Z − Z_fit`,
`fits/kk_validation.py:69` : `res_re = Z.real − Z_fit_re`) · `setup_drt_bayesien.bat:10-12` (« JAMAIS appelé au démarrage »). Le corps de `METHODES.md`
est lui aussi périmé (§9).

### 2.8 Fichiers orphelins

`project` (brief de refonte UI exécutée), `Reference_maintenance` (HTML sans extension, obsolète), `launch_app.bat` (doublon
fonctionnel de `launch.bat`, cité seul par `README.md:11`), `pages/D_inference.py` (stub). Sous `vendor/bayes_drt2/stan_model_files/`,
seuls `Series.stan` et `Series_pos.stan` sont atteignables par `fit_drt` (une seule distribution « DRT », `outliers=False`) ; les 11 autres
sont de la charge utile vendorée, pas du code mort du projet.

---

## 3. Dépendances

Aucun `pyproject.toml`/`setup.py`, **aucun lockfile**, aucune contrainte de version Python déclarée. Toutes les bornes sont des
minima (`>=`), **aucune version n'est épinglée**. 🔬 En installant `requirements.txt` à neuf (2026-09-30) pip a résolu les versions les plus
récentes — streamlit 1.64.0, numpy 2.4.6, scipy 1.17.1, pandas **3.0.6**, plotly **7.1.0**, matplotlib 3.11.2, pydantic 2.13.5,
kaleido 1.4.0, scikit-learn 1.9.1, eclabfiles 0.4.1, PyYAML 6.0.3, pytest 9.1.1 (+ cvxopt 1.3.3, cmdstanpy 1.3.0, CmdStan 2.36.0) —
et la suite de tests passe (§8). C'est une bonne nouvelle pour la compatibilité, mais **le résultat dépend du jour d'installation**
(l'installation pip a lieu à *chaque* lancement de `launch.bat`, L97-99).

| Paquet | Déclaré | Épinglé ? | Importé par (production) | Verdict |
|---|---|---|---|---|
| streamlit | `requirements.txt:1` `>=1.32.0` | non | `app.py`, `pages/*`, `ui/tabs.py` | **Borne fausse** 🔬 : `st.navigation`/`st.Page` (`app.py:76-93`) n'existent **pas** dans la roue 1.32.0 (absent de `streamlit/navigation/`), présents en 1.36.0 ; `st.plotly_chart(..., width='stretch')` (21 appels : `1_pretraitement.py` ×5, `A_eis.py` ×3, `ui/tabs.py` ×13) n'apparaît dans la signature qu'à partir de **1.51.0** (absent en 1.50.0). Minimum réel ≈ **1.51** (détection par inspection des roues, non exécutée). |
| numpy | `:2` `>=2.0` | non | partout ; `np.trapezoid` (`drt_fit.py:172`, patch vendor) | Borne correcte (numpy ≥ 2 requis par `np.trapezoid`). |
| scipy | `:3` `>=1.12.0` | non | `calibration`, `cv_peaks`, `error_structure` (nnls), `randles_full` (least_squares), `cv_plots` | Utilisé. |
| pandas | `:4` `>=2.2.0` | non | `loader`, `cv_loader`, `mpr_converter`, `1_pretraitement` (+ vendor) | Utilisé (fonctionne avec pandas 3.0.6). |
| plotly | `:5` `>=5.20.0` | non | `plotting/*`, `1_pretraitement` | Utilisé (fonctionne avec 7.1.0). |
| matplotlib | `:6` `>=3.8.0` | non | fenêtres `plt.show()` (`eis_plots.py:999,1058`, `cv_plots.py:390,434`), `1_pretraitement.py`, **vendor** (import module) | Utilisé — dépendance dure de l'import du vendor. |
| pyyaml | `:7` `>=6.0` | non | `config`, `experiment_io`, `exporter` | Utilisé. |
| pydantic | `:8` `>=2.0` | non | `core/config.py` | Utilisé. |
| **kaleido** | `:9` `>=0.2.1` | non | *seulement* `exporter.export_figure_png` (`:95`), fonction **∅** | **Inutilisé** (kaleido 1.x exige de plus un Chrome). Retirer. |
| **pytest** | `:10` `>=8.0` | non | tests | Dépendance de **dev** placée dans les dépendances d'exécution (installée à chaque lancement chez l'utilisateur). |
| **scikit-learn** | `:11` `>=1.3.0` | non | **aucun import** (`grep -r "sklearn"` vide) | **Inutilisé** — lourd (+ joblib, threadpoolctl). Retirer. |
| eclabfiles | `:12` `>=0.4.1` | non | `core/mpr_converter.py:39` (import différé) | Utilisé (lecture `.mpr`). |
| galvani | `requirements-optional.txt:12` `>=0.5.0` | non | `mpr_converter.py:44` (repli) | Optionnel assumé (build cassé hors Debian, cf. commentaire L1-11). |
| cvxopt | `requirements-drt.txt:19` `>=1.3.0` | non | *uniquement* le vendor (`inversion.py:7`) | Requis par `bayes_drt2` (import module). |
| cmdstanpy | `requirements-drt.txt:20` `>=1.2.0` | non | `drt_fit.py:89`, `setup_drt_bayesien.py`, vendor `inversion.py:11` | Requis (DRT `optimize` **et** `sample`). 🔬 Avec 1.3.0 : ≈ 15 `FutureWarning` par fit MAP (`stan_variable()` renverra bientôt un `ndarray` même pour un scalaire) — 60 lignes de bruit pour 4 spectres — et **risque de rupture** du vendor si ce comportement change. |
| *(absent)* Python | — | — | `core/cv_loader.py:24` (`str \| None` évalué à l'exécution) | **Python ≥ 3.10 requis mais non déclaré** ni testé par les lanceurs. CI en 3.12 seulement. |

Points de cohérence : `requirements-drt.txt` n'est installé par **aucun** lanceur (`launch.bat:97-99` n'installe que `requirements.txt`) ;
`setup_drt_bayesien.bat:54` installe `cvxopt cmdstanpy` **sans version** (duplication non synchronisée avec `requirements-drt.txt`) ;
la CI n'installe pas l'extra DRT. Non listés mais nécessaires en pratique pour la DRT : une toolchain C++ (mingw sous Windows) et
CmdStan (~ centaines de Mo, hors pip).

---

## 4. Module DRT actuel (`fits/drt_fit.py`) — état réel avant remplacement/durcissement

### 4.1 Algorithme et appel

`fit_drt` (`drt_fit.py:208`) : (1) construit `Z = Zre − j·Zim` (convention `Zim = −Im(Z) > 0`, L223), (2) trie HF→BF sur les fréquences mesurées
(L226-228), (3) `Inverter().fit(freq, Z, mode=mode)` (**L231, seul paramètre transmis : `mode`**), (4) lit `tau` et `γ(τ)` via
`predict_distribution` (L234-241), (5) en mode `sample` uniquement, percentiles 2,5/97,5 % (L246-255), (6) extrait `Rct` par pic (L259) ou
**repli sur `Rp`** (L260-268), (7) reconstruit `Z` (L273-277) et calcule un « misfit » relatif (L192-205), (8) empaquette un `FitResult` (L295-316).
`bayes_drt2` = inversion **hiérarchique bayésienne** (lignée Ciucci-Chen) : la DRT est développée sur des fonctions de base gaussiennes
(10 pts/décade, grille dérivée des fréquences mesurées, d'après le commentaire de `core/config.py:48-52`) avec hyperparamètres estimés par un modèle Stan ; `optimize` = MAP (L-BFGS), `sample` =
HMC (`vendor/bayes_drt2/inversion.py:1326`).

### 4.2 Paramètres effectifs

| Paramètre | Valeur utilisée | Origine |
|---|---|---|
| `mode` | `optimize` (pipeline) / `sample` (bouton) | `config.fit.drt.mode` (`drt_fit.py:181-189`), forcé par `recompute_drt` (`pipeline.py:335-340`) |
| `nonneg` | **False** | défaut vendor (`inversion.py:1326`) — non transmis |
| `init_from_ridge` | **False** | défaut vendor — non transmis |
| `outliers` | False | défaut vendor |
| `random_seed` | 1234 | défaut vendor — non transmis (déjà signalé m5, `AUDIT_REPORT.md`) |
| `max_iter` (optimize) | 50 000 | défaut vendor |
| `warmup`, `samples`, `chains` (sample) | 200, 200, 2 | défauts vendor |
| `sigma_min` | 0,002 | défaut vendor |
| Modèle Stan | `Series.stan` | choisi par `_get_stan_model` (`inversion.py:1828-1880`) ; pré-compilé aussi : `Series_pos.stan` |
| Pondération d'Orazem | **non utilisée** (`weights` ignoré, `drt_fit.py:352`) | la DRT a son propre modèle d'erreur |
| Clés de config héritées | `drt_wiener_W`, `drt_n_z` : **mortes** (§2.3) | |

### 4.3 Extraction de `Rct` (convention « Bissessur »)

`_extract_rct_peak` (L114-175) : maxima locaux de γ dans une fenêtre `l = max(1, n//15)` au-dessus de `1e-3·max(γ)` sur la partie « cœur » (marges `n//20`), puis
**avant-dernier maximum** = arc de transfert de charge, et intégrale trapèze de γ sur ±3 en ln τ. Un seul pic → `peak_single` (+ avertissement) ;
aucun → `None` → repli sur `Rp` (aire totale) avec avertissement (L261-268). Hypothèse implicite : exactement (arc de transfert, diffusion) ⇒ au plus 2 constantes de temps.

### 4.4 Mesures 🔬 (reproductibles : Annexe A ; environnement Linux / Python 3.11 / CmdStan 2.36.0 / cmdstanpy 1.3.0 / cvxopt 1.3.3)

| Cas | `Inverter.fit(mode='optimize')` réglages actuels (défaut) | `init_from_ridge=True` | `nonneg=True` |
|---|---|---|---|
| Scénario du test du dépôt (2 RC, 71 pts, 1e5→1e-2 Hz), via `DRTBayesModel().fit` | Rp = **130,30 Ω** (vrai 130) ✅ ; Rct = 80,05 (arc à 0,1 s, cf. DRT-10) | 130,1 ✅ | 130,3 ✅ |
| Mêmes 2 RC, **60 pts, 1e5→1e-1 Hz, sans bruit** | Rp = **−159,4 Ω** ❌, γ_min = −250,8, erreur de reconstruction max **363 %** | 130,1 ✅ (0,2 %) | 130,2 ✅ (0,75 %) |
| Idem avec 0,5 % de bruit, graine par défaut (1234) | Rp = −159,1 ❌ | 129,4 ✅ | 130,7 ✅ |
| Idem, graines 1 / 2 / 3 | 129,5 ✅ / 129,4 ✅ / **−159,2 ❌** | — | — |
| **Spectres de type Randles** (Re 200, R'e 20, Cb 1 nF, Qdl 2 µF, α 0,9, R_D = 0,3·Rct, τ_d 0,5 s ; Rct = 3000 / 3500 / 4200 / 5200 ; 60 pts ; bruit 0,5 %) ; Rp attendu = R'e + 1,3·Rct = 3920 / 4570 / 5480 / 6780 | Rp = **−5678 / −6552 / −7925 / −9278** ❌ ; erreur de reconstruction ≈ 380 % | 3916 / 4555 / 5462 / 6774 ✅ (≤ 0,4 %) | 3905 / 4560 / 5461 / 6761 ✅ (≤ 0,4 %) |
| Même jeu de type Randles **via `run_pipeline` complet** (3 réplicats par concentration) | `Rct_drt` = **−6080 / −7130 / −8410 / −10067 Ω** (4/4 négatifs), `converged=True`, `warnings=[]`, `chi2_reduced` = 5,3-6,1 | — | — |
| Même jeu, `randles_full` | Rct = 2999,5 / 3497,3 / 4201,6 / 5193,6 (écart ≤ 0,13 % vs 3000/3500/4200/5200) ✅ | | |

Temps : `pytest tests/test_drt.py -m "not slow"` = 26,9 s (dont compilation de `Series.stan`) ; 4 fits MAP dans `run_pipeline` ≈ 1 s chacun après
compilation (pipeline complet 13 s) ; test HMC (`-m slow`) = 32,7 s.

**Lecture** : avec les réglages actuels, le MAP tombe dans un optimum dégénéré (DRT à valeurs négatives énormes) sur des spectres de type
Randles réalistes, **sans qu'aucun garde-fou du dépôt ne le voie**. Les deux réglages alternatifs testés (`init_from_ridge=True`, `nonneg=True`)
ont donné un résultat correct (Rp à ≤ 0,6 % de la valeur attendue, erreur de reconstruction ≤ 1,7 %) sur 17 des 18 essais (9 essais par réglage) ; l'exception est la grille 80 points 1e6→1e-1 Hz sans bruit,
où `nonneg=True` (comme le défaut) donne Rp ≈ 140 (+8 %) alors que `init_from_ridge=True` donne 130,1. `init_from_ridge=True` est correct sur les 9 essais, `nonneg=True` sur 8 sur 9. ⚠️ Non établis : la cause racine (initialisation aléatoire de Stan + DRT non contrainte
sont soupçonnées, non démontrées) et la dépendance aux versions (CmdStan/cmdstanpy/numpy du poste Windows de l'utilisateur ne sont pas ceux de ce
banc) — à **rejouer sur le poste cible** avant toute décision de réglage.

### 4.5 Limites et défauts constatés

| ID | Sév. | Constat | Réf. |
|---|---|---|---|
| DRT-1 | **Critique** | MAP non fiable avec les réglages actuels (voir §4.4). | `drt_fit.py:231` |
| DRT-2 | **Critique** | Aucune garde qualité : `converged=True` **codé en dur**, `warnings` vide si un pic est trouvé, ni test de `Rp ≤ 0`, de γ négatif dominant, ni d'erreur de reconstruction — alors que `randles_full` a 4 gardes (I7). | `drt_fit.py:306,258-270` |
| DRT-3 | Majeur | La calibration DRT **retire silencieusement** tout point dont `Rct ≤ 0` (`core/calibration.py:50`) : avec DRT-1 la courbe log-log peut perdre tous ses points sans message (`calibration_drt_figure` n'affiche alors « rien »). Idem pour la calibration Randles : `_collect_points` ne regarde ni `converged`, ni `warnings`, ni `kk_passed`. | `calibration.py:42-55` |
| DRT-4 | Majeur | Le champ `params["tau_Rct"]` contient **ln τ**, pas τ (`_extract_rct_peak` renvoie `ln_tau[peak_idx]`, L175 ; le test le valide, `tests/test_drt.py:41-47`). Il est exporté tel quel (`export_params_csv`). | `drt_fit.py:175,286` |
| DRT-5 | Majeur | `chi2_reduced` a **deux sémantiques** : test d'adéquation pondéré (Randles) vs moyenne du carré de l'erreur relative (DRT, L201). `params_table_figure` et `export_params_csv` les placent dans la même colonne. | `drt_fit.py:192-205` |
| DRT-6 | Majeur | Deux modèles d'erreur incompatibles dans une même comparaison : Orazem (Randles) vs modèle hiérarchique interne (DRT, `weights` ignoré) ; les réplicats n'ont pas de DRT (`pipeline.py:196-200`), donc **aucune incertitude inter-réplicats** sur `Rct_drt`. | `drt_fit.py:349-354` |
| DRT-7 | Majeur | Le bootstrap `_bootstrap_drt` (`app.py:50-70`) télécharge/compile CmdStan **dans le thread du script** sans délai maximal ; et `st.cache_resource` **mémorise aussi l'échec** `(False, msg)` (la fonction ne lève pas) : pas de nouvelle tentative avant redémarrage du processus. ⚠️ Comportement sous Windows non testé ici. | `app.py:50-70`, `setup_drt_bayesien.py:150-199` |
| DRT-8 | Majeur | L'extra pip n'est installé par aucun lanceur : sur une installation neuve `bayes_available()` est faux, l'onglet DRT est en erreur (`ui/tabs.py:155-162`) et `pages/A_eis.py:369` ajoute pourtant `drt_bayes` à chaque analyse avant de le retirer avec un avertissement (`A_eis.py:397-406`). | §7.4 |
| DRT-9 | Moyen | Reproductibilité : `random_seed` non transmis (défaut vendor) ; les résultats de DRT-1 dépendent de la graine. | `drt_fit.py:231` |
| DRT-10 | **Majeur** | Le Rct « pic pénultième » n'est validé que sur des gaussiennes parfaites (`tests/test_drt.py:26-80`). 🔬 Sur la DRT MAP *correcte* de 2 RC (γ = 105,8 à τ = 1 ms ; 182,1 à τ = 0,1 s), `_local_maxima` trouve **4 maxima** dont 2 ondulations à γ ≈ 0,2 (seuil = 1e-3·max = 0,18) ; le « pénultième » retenu est le pic à 0,1 s (80 Ω) et **non** celui à 1 ms (50 Ω) que la convention prétend viser (arc de transfert avant la queue de diffusion). Le résultat dépend donc d'une ondulation de 0,1 % du maximum. Aucune garde sur γ négatif, τ hors plage physique, ≥ 3 pics. | `drt_fit.py:114-175` (seuil L145) |
| DRT-11 | Moyen | La plage de fréquences KK-valide (`validator.py`) n'est jamais appliquée avant fit/DRT : `EISSpectrum.f_min_valid/f_max_valid` sont inutilisés (§2.4). | `models.py:39-40` |
| DRT-12 | Mineur | Bruit de journal (60 `FutureWarning` cmdstanpy / 4 fits) ; imports lourds au niveau module du vendor (`cvxopt`, `cmdstanpy`, `matplotlib`). | `inversion.py:1-12` |
| DRT-13 | Mineur | `DRTBayesModel.predict` (∅), attributs `method`/`display_name` (∅), clés de config `drt_wiener_W`/`drt_n_z` (mortes). | §2 |

### 4.6 Tests existants

`tests/test_drt.py` (6 tests) : 4 tests purs sur `_local_maxima`/`_extract_rct_peak` (gaussiennes synthétiques) ; 1 test d'intégration `optimize`
(deux pics à ±0,3 décade, `Rct > 0`, longueurs) ; 1 test `sample` (`γ_lo ≤ γ ≤ γ_hi`, marqué `slow`). Les deux derniers sont **sautés sans CmdStan**
(donc en CI). 🔬 Avec CmdStan ils passent (5 passed en 26,9 s ; le `slow` en 32,7 s) — **mais aucun n'aurait vu DRT-1** : ils ne vérifient ni
l'ordre de grandeur de `Rp`/`Rct`, ni la positivité, ni l'erreur de reconstruction, et le scénario testé (71 points, 1e5→1e-2 Hz) est l'un
de ceux où le réglage par défaut réussit. Aucun test sur `recompute_drt`, `run_pipeline` + DRT, ni sur le repli `rp_fallback` de bout en bout
(le commit `1d3aac5` « DRT jamais lancée par le pipeline » a été corrigé sans test de non-régression).

### 4.7 Ce que cela change pour l'intégration « `bayes_drt2` »

- Le package est **déjà vendoré** (v0.2, BSD-3, provenance par sha256, deux patchs numpy 2). Reste à décider : conserver le vendoring ou revenir à une
  dépendance installable (⚠️ non vérifié ici : existence d'une distribution installable et lien avec un commit précis de `jdhuang-csm/bayes-drt2`).
- La partie à concevoir n'est pas « brancher » mais **encadrer** : réglages (`nonneg`/`init_from_ridge`/graine) validés sur une matrice de cas, garde-fous
  qualité (§4.5 DRT-2), gestion explicite d'échec, homogénéisation des sorties (`chi2_reduced`, `tau_Rct`), reproductibilité, et non-régression en CI avec
  CmdStan (aujourd'hui absent de la CI).

---

## 5. Modules de fit, `base.py`, `registry.py` — quoi garder pour le fit Orazem unique

### 5.1 Correspondance avec les 3 modules « à supprimer »

| Module cité dans l'énoncé | État réel | Où en est la logique |
|---|---|---|
| `randles_classique.py` (Differential Evolution, 7-8 params) | **Inexistant**, jamais versionné. Pas de Differential Evolution ni de méthode stochastique nulle part (`grep differential_evolution` vide) ; `least_squares` TRF est déterministe. | Remplacé par `fits/randles_full.py` (8 paramètres). |
| `randles_contraint.py` (3 paramètres, Re et ZD0 fixés) | **Supprimé** (`randles_constrained.py`, commit `e315f30`). | Plus aucune trace de code ; subsiste dans `README.md:51-52` et `pages/A_eis.py:351-352`. |
| `circulaire_fit.py` (Kasa) | **Supprimé** (`circular_fit.py`, `e315f30`). | Idem. |

### 5.2 Ce qui est **générique** dans `randles_full.py` / `base.py` / `registry.py` (à conserver ou à déplacer dans le futur moteur)

| Élément | Réf. | Pourquoi le garder |
|---|---|---|
| Contrat `FitResult` : `params`, `params_std`, `Zfit_re/im`, `residuals_re/im`, `chi2_reduced`, `converged`, `warnings`, provenance de la structure d'erreur (`error_structure_source/timestamp/coeffs`) | `core/models.py:69-128` | Interface commune UI/export. À **élaguer** (champs morts §2.4) et à **découpler de `Rct`** (voir §5.4). |
| Convention de signe `Zim = −Im(Z) > 0` de bout en bout, et résidu `−Z.imag − Zim` | `randles_full.py:136-145`, `loader.py:188-190`, `drt_fit.py:222-223` | Bug B1 historique (25-30 % de biais sur Rct) ; test d'équivalence `tests/test_measurement_model.py:274`. |
| Pondération `w = 1/σ²`, `σ_i = σ_struct/√n_replicates`, `absolute_sigma` toujours vrai (covariance non rééchelonnée) | `weighting.py:68-75`, `randles_full.py:158-170` | Cœur de la « méthode Orazem » actuelle. |
| Structure d'erreur : σ empirique → régression NNLS de (α, β, γ, δ) → persistance → refus si non caractérisable | `error_structure.py:118-451` | Cœur, testé (10 tests). **À revoir** : persistance (§5.5). |
| Solveur : `scipy.optimize.least_squares` (TRF, bornes, `ftol=xtol=1e-10`, `max_nfev` configurable) | `randles_full.py:147-156` | Générique ; le calcul de la covariance `(JᵀJ)⁻¹` l'est aussi (`:165-170`). |
| Gardes d'ajustement (I7) : non convergé, χ²ᵣ hors de `[1 ± 2√(2/dof)]`, résidu relatif RMS > 10 %, paramètre à < 1 % d'une borne | `randles_full.py:194-234` | Ce qui aurait détecté B1 ; **à généraliser et à imposer aussi à la DRT** (§4.5 DRT-2). |
| Diagnostic de chargement des plugins `discovery_errors()` | `registry.py:57-64`, `A_eis.py:397-406` | Évite le « modèle évaporé » (I6). À conserver si on garde un registre. |
| Refus explicite (`ErrorStructureUnavailable`) plutôt que σ arbitraire | `error_structure.py:63`, `:446-451` | Bon principe — mais il faut que **l'UI l'affiche** (§5.5 ERR-1). |

### 5.3 Ce qui est **spécifique** et à jeter/refaire

| Élément | Réf. | Pourquoi |
|---|---|---|
| `_PARAM_NAMES` (8 noms), `Z_randles_full`, `Z_D` (diffusion bornée « Bissessur ») | `randles_full.py:17`, `physics.py:10-63` | Circuit codé en dur : à remplacer par un constructeur de Z(ω) piloté par le circuit utilisateur. `Z_D` reste réutilisable comme **élément** de bibliothèque. |
| Heuristiques `initial_guess` (`Re ≥ 100 Ω`, `Rct ≥ 500 Ω`, `R'e = 5 % Re`, `Cb = 1 nF`, `Qdl = 1 µF`, `α = 0,85`, `τ_d = 1/(2π f_min)`) | `randles_full.py:51-67` | Planchers en dur propres aux capteurs de la thèse ; un fit générique a besoin d'un guess fourni/estimé par élément. Mono-départ (pas de multi-start). |
| Bornes par défaut, **écrites 3 fois** | `randles_full.py:79-88`, `config.py:31-45`, `default.yaml:46-54` | Une seule source dans le futur moteur. |
| Champs `Rct`, `Rct_std` obligatoires du `FitResult` et `reconstruction_error` | `models.py:88-89` | Couplage de tout le pipeline (calibration, exports, UI) à un paramètre nommé `Rct`. Avec un circuit libre, l'utilisateur doit **désigner** le paramètre de calibration. |
| Registre par `pkgutil` pour 2 plugins | `registry.py` | Sur-ingénierie pour ≤ 2 méthodes ; si le fit devient un moteur unique + DRT, un appel explicite suffit. |
| Modèles fantômes dans l'UI | `pages/A_eis.py:350-354` (`"circular"`, `"randles_constrained"`) | Cases à cocher (défaut décoché) qui sélectionnent des modèles **inexistants** : cochées, elles déclenchent l'avertissement « modèle indisponible » (`A_eis.py:397-406`). `README.md:51-52` les documente aussi. |

### 5.4 Défauts de `randles_full.fit` à ne pas reproduire

| ID | Constat | Réf. |
|---|---|---|
| FIT-1 | `except Exception:` **avale toute erreur** (y compris un bug de programmation) et renvoie `x_fit = x0` (le guess initial) avec `converged=False`. Le pipeline logue quand même `Rct=… Ω` et la calibration (`_collect_points`, `calibration.py:50`) **ne filtre pas `converged`** : un guess non ajusté peut entrer dans la régression, seul un bandeau « non convergé » le trahit. | `randles_full.py:172-175` |
| FIT-2 | Covariance : `np.linalg.inv(JᵀJ)` sans conditionnement et `np.sqrt(np.abs(diag))` — un écart-type est fabriqué même si la variance est négative (mal-conditionnement à 8 paramètres corrélés sur plusieurs décades). Préférer SVD/`pinv` + nombre de conditionnement rapporté. | `randles_full.py:165-170` |
| FIT-3 | Pas d'échelle de paramètres (`x_scale`) alors que les paramètres couvrent 1e-12…1e9 ; mono-départ. ⚠️ Effet sur la robustesse non mesuré ici (récupération à ≤ 0,13 % sur mes 4 jeux synthétiques, mais ce sont des cas favorables). | `randles_full.py:150-154` |
| FIT-4 | Garde « paramètre en butée » : seuil relatif à la **valeur de la borne** (`1 % × \|borne\|`), peu parlant pour des bornes très petites (`Cb` 1e-12) ou très asymétriques. | `randles_full.py:229-234` |
| FIT-5 | `weights=` : API à deux formes (tuple ou tableau « hérité ») utilisée par 2 tests seulement. | `randles_full.py:121-131` |

### 5.5 Défauts du pipeline autour du fit (à traiter avant/pendant la refonte)

| ID | Sév. | Constat | Réf. |
|---|---|---|---|
| ERR-1 | **Critique** | 🔬 Sans réplicats **et** sans structure persistée (premier usage, ou après une mise à jour de `launch.bat` qui efface `config/error_structure.json`, §7), `resolve_error_structure` lève `ErrorStructureUnavailable`, que `run_pipeline` **attrape et journalise seulement** (`log.error`). Reproduit : `run_pipeline` sur 1 fichier probe + 2 concentrations → 2 groupes créés, `fit_results` **vides**, aucune exception ; `A_eis.py:448` affiche « ✅ Analyse terminée — 2 groupe(s) » et `_collect_fit_diagnostics` (`A_eis.py:44-66`) n'a rien à signaler. | `pipeline.py:206,230,250` ; `A_eis.py:436-448` |
| ERR-2 | Majeur | Persistance = fichier JSON **partagé, en append, jamais purgé**, dans l'arborescence de l'app. 🔬 5 entrées ajoutées par run de 4 spectres moyennés (10 après 2 runs) car chaque `resolve_error_structure` avec réplicats appelle `persist` (`error_structure.py:432-434`). État global caché : le résultat d'un fit dépend de l'historique des analyses précédentes (la « dernière structure » sert aux spectres sans réplicats ; les réplicats individuels de bare/probe sont ajustés avec la structure du **premier** spectre caractérisé — `pipeline.py:47-80` — donc pas la leur, ceux des concentrations avec celle de leur groupe, persistée juste avant). Incompatible avec l'objectif de transparence/reproductibilité. | `error_structure.py:347-382,407-451`, `pipeline.py:47-80,188-208` |
| ERR-3 | Majeur | Tous les `except Exception` du pipeline convertissent une erreur en « spectre sans fit » : chargement (`:151`), fits (`:206,230,250`), KK (`:88`), validation (`:267`). Un même défaut logiciel est indiscernable d'un fichier invalide côté utilisateur. | `pipeline.py` |
| ERR-4 | Majeur | Faux positifs de la détection de dérive : 🔬 sur des réplicats **stationnaires** (même spectre, bruit gaussien indépendant de 0,5 %), `_detect_drift` annonce « Drift détecté sur 72 %/75 %/82 %/88 % des fréquences » (4 groupes). Le critère CV = σ/moyenne des résidus KK inter-réplicats vaut ~1 pour du bruit blanc, donc dépasse toujours le seuil 0,5. L'alerte est affichée en rouge (`ui/tabs.py:88-89`). | `validator.py:328-368` (seuil `:93`) |
| ERR-5 | Majeur | `Lin-KK` : `M = min(100, max(2, 2n//3))` **fixe** (`kk_validation.py:44`) ; le critère `c = 0.85` de Schönleber passé par `validator.py:135` est **ignoré** (`linKK` L126-132) — le commentaire de l'appelant laisse croire à une recherche itérative de M qui n'existe pas (la docstring de `linKK` L127-130 dit d'ailleurs que `c` n'est pas utilisé). Avec ~40 éléments RC pour 60 points, les résidus sont petits par construction : test KK peu discriminant. Ajustement non pondéré (Schönleber pondère par 1/\|Z\|). ⚠️ Effet sur la sensibilité non mesuré. | `kk_validation.py:44,126-132` |
| ERR-6 | Moyen | Deux critères KK différents coexistent : `RESIDUAL_THRESHOLD_PCT = 2 %` par point (`validator.py:85`) et `drt_kk_tol = 0,05` sur le max (`kk_validation.py`), avec deux verdicts (`kk_passed` — jamais lu — et `is_valid`). | §2.4 |
| ERR-7 | Moyen | Aucune restriction du fit à la plage KK-valide (les champs `f_min_valid/f_max_valid` de `EISSpectrum` ne sont jamais renseignés ni lus). | `models.py:39-40` |

### 5.6 Autres défauts fonctionnels relevés en chemin (hors fit/DRT)

| ID | Sév. | Constat | Réf. |
|---|---|---|---|
| CV-1 | **Majeur (à confirmer)** | `load_cv_file` **trie chaque courbe par potentiel croissant**. Un voltammogramme cyclique est une *boucle* (branche aller + branche retour) : le tri entrelace les deux branches, `detect_redox_peaks` lisse (Savitzky-Golay) un signal qui alterne de signe, et `average_cv_replicates` / `run_cv_pipeline` interpolent `I(E)` avec `np.interp` sur un axe qui n'est pas une fonction. 🔬 Boucle synthétique (aller : pic +5 µA à 0,25 V ; retour : pic −4 µA à 0,19 V ; 400 points, colonnes `Ewe/V`, `<I>/mA`) : après chargement, **223 changements de signe** de dI/dE (≈ 4 attendus) ; pics rendus `Ipa = −1,1 µA` et `Ipc = +1,6 µA` (**signes inversés** ; vrais +5 / −4 µA), `ΔEp = 72 mV` (vrai 60 mV). ⚠️ Ne concerne que des fichiers contenant des cycles complets ; sur des balayages simples le tri est inoffensif — **à confirmer avec l'utilisateur** (nature des fichiers CV). | `cv_loader.py:74,118` ; `cv_peaks.py` ; `cv_loader.py:151-173` ; `cv_pipeline.py:316-322` |
| CV-2 | Moyen | `delta_signal = \|I_probe − I_c\| / \|I_probe\|` n'écarte que le cas exact `I_probe == 0` : autour d'un passage par zéro du courant du probe le rapport explose, et ces valeurs entrent dans `np.nanmean` de la calibration CV. ⚠️ Non mesuré. | `cv_pipeline.py:317-320`, `calibration.py:148` |
| B-STATE | Majeur | Résultats d'analyse périmés réutilisés après rechargement/re-prétraitement ; `KeyError` après un ré-import (§2.5). | `A_eis.py:340,413`, `B_cv.py:105,111`, `0_import.py:552-575` |
| B-EXP | Majeur | CSV de paramètres mal aligné (§8.4). | `exporter.py:37-52`, `E_export.py:51-56` |

---

## 6. Couplage `core/` · `fits/` · `plotting/` · `ui/` (règles de `ARCHITECTURE.md:156-166`)

### 6.1 Règle demandée — « `core/` et `fits/` n'importent jamais Streamlit » : **aucune violation**

Vérification par AST (imports de module **et** imports locaux) et par recherche textuelle : `streamlit` n'est importé que par 8 fichiers —
`app.py`, `ui/tabs.py`, `pages/{0_import,1_pretraitement,A_eis,B_cv,D_inference,E_export}.py`. Les seules occurrences dans `core/`, `fits/`,
`plotting/`, `exports/` sont des **mots dans des commentaires/docstrings** (`core/models.py:147`, `core/pipeline.py:314`, `exports/exporter.py:194`).
`core/`, `fits/`, `plotting/`, `exports/` n'importent ni `ui` ni `pages`.

### 6.2 Matrice des dépendances internes (imports réels, AST)

| De → Vers | Imports | Nature |
|---|---|---|
| `core` → `fits` | 5 | **Tous locaux** (fonctions) : `pipeline.py:57,85,112,329`, `validator.py:121` |
| `fits` → `core` | 6 | Module : `base.py:4`, `drt_fit.py:48-49`, `randles_full.py:15`, `registry.py:7`, `error_structure.py:52` |
| `fits` → `vendor` | 1 | Local, protégé : `drt_fit.py:61` |
| `plotting` → `core` | 4 | Module : `cv_plots.py:9-10`, `eis_plots.py:7-8` |
| `exports` → `core` | 5 | 3 module + 2 locaux |
| `ui` → `core`/`fits`/`plotting`/`exports` | 4/1/3/1 | Module (`tabs.py:8-9,21-24,33-34`) |
| `pages` → `core`/`fits`/`plotting`/`ui`/`exports` | 15/1/2/2/1 | `pages/A_eis.py:397` importe `fits.registry` localement |
| `app.py` → `fits`, `setup_drt_bayesien`, `streamlit` | local ×2, module | `app.py:59,63,9` |

### 6.3 Écarts par rapport aux règles d'architecture (hors Streamlit)

| ID | Constat | Réf. | Statut |
|---|---|---|---|
| CPL-1 | **Dépendance circulaire `core` ↔ `fits`** masquée par des imports locaux : `core.pipeline`/`core.validator` appellent `fits.*`, tandis que `fits.*` importe `core.models`/`core.logger` au niveau module. Toute réorganisation de paquets casserait les imports différés silencieusement (au premier appel). | §6.2 | Nouveau |
| CPL-2 | `plotting/cv_plots.py` **recalcule** `stats.linregress` 4 fois au lieu de consommer `core/calibration.compute_cv_calibration` (risque de divergence figure ↔ export CSV). | `cv_plots.py:157,224,412,450` | Déjà signalé (M5, `AUDIT_REPORT.md`) — **toujours ouvert** |
| CPL-3 | **Logique métier dans des pages** : normalisation Nyquist point à point (`\|Z_probe − Z_c\|/Z_probe`, moyenne inter-électrodes) dans `pages/A_eis.py:178-241`, stockée dans `session_state["eis_normalized"]` puis lue par l'export ; masques de points, combinaisons d'électrodes, recalculs de moyennes dans `pages/1_pretraitement.py` (`_apply_point_mask` L654, `_combine_eis_two` L934, `_avg_eis_reps` L922…). Non testé, non réutilisable. | pages | Nouveau |
| CPL-4 | Duplication de logique entre pages et `core/` : `_sig_types_for_mode` (`core/experiment_io.py:547` et `pages/0_import.py:468`), chargement de spectres par électrode (`A_eis._load_electrode_spectra` L90, `1_pretraitement._load_eis_spectra` L196, `_bio_to_eis` L117), **deux exécutions de la validation KK** (`pipeline.validate_session` et `1_pretraitement._run_kk_validation` L754). | | Nouveau |
| CPL-5 | `pages` importent des symboles **privés** de `plotting` (`_spectrum_label`, `_nyquist_figure`, `_SUP_MAP`) ; `_SUP_MAP` est défini deux fois (`eis_plots.py:14`, `1_pretraitement.py:38`). | `A_eis.py:14`, `1_pretraitement.py:32,38` | Nouveau |
| CPL-6 | `plotting/` ouvre des **fenêtres matplotlib** (`plt.show()`) depuis un serveur Streamlit (`eis_plots.py:1049,1083`, `cv_plots.py:429,464`) : ne fonctionne que sur le poste qui héberge le serveur avec un backend graphique. ⚠️ Non testé (pas d'écran ici). | | Nouveau |
| CPL-7 | `core/logger.py:8` crée `logs/` **à l'import** ; `error_structure.py:57` persiste dans `config/` ; ces écritures dans l'arborescence de l'app sont détruites par chaque mise à jour du lanceur (§7). | | Nouveau |
| CPL-8 | `ui/tabs.py` importe `fits.drt_fit` (pour `bayes_available`, `cmdstan_available`) : la règle « ui n'appelle que core/fits » est respectée, mais l'UI connaît un détail de la DRT (pas d'`Inverter` : conforme à la règle de `drt_fit.py:14-16`). | `tabs.py:21` | Acceptable |
| CPL-9 | `pages/B_cv.py:22` accède à `experiment["calibration"]["cv"]` sans `.get` (KeyError en mode « EIS seule » ?) ; `run_cv_pipeline` (L118) n'est pas protégé par `try`, contrairement au pipeline EIS (`A_eis.py:421-438`). ⚠️ Lu dans le code, non rejoué dans le navigateur. | | Nouveau |

---

## 7. Lanceur `launch.bat` — relecture ligne par ligne

`launch.bat` (125 lignes) est le **seul fichier distribué** (`ARCHITECTURE.md:299`) : à côté de lui il crée `eis_app/`, `venv/`, `.version` et exige `token.txt`.

### 7.1 Déroulé et lignes

| Lignes | Action |
|--:|---|
| 1-13 | `@echo off`, `setlocal enabledelayedexpansion`, constantes : `GITHUB_USER=J-Facc`, `GITHUB_REPO`, `BRANCH=main`, `APP_DIR=%~dp0eis_app`, `VENV_DIR=%~dp0venv`, `VERSION_FILE=%~dp0.version`, `TOKEN_FILE=%~dp0token.txt`, `PORT=8501`. |
| 20 | `reg add HKLM\…\FileSystem /v LongPathsEnabled … /f >nul 2>&1` à **chaque** lancement. |
| 24-39 | Refuse de démarrer sans `token.txt` (dépôt privé) ; lit la 1re ligne (`set /p`, L34). |
| 41-60 | Interroge `api.github.com/repos/…/commits/main` (PowerShell, L44) → `REMOTE_SHA` ; compare à `.version` (L47-60). |
| 61-86 | Télécharge le `zipball` (L67), **supprime `eis_app/`** (L75), extrait (L77), `move` du dossier racine du zip (L78-81), écrit `.version` (L85). |
| 88-108 | Crée le `venv` si `streamlit.exe` absent (L92-96), `pip install -r eis_app\requirements.txt` à chaque lancement (L97-107). |
| 110-125 | Vérifie `app.py`, ouvre le navigateur (L121), `cd /d eis_app` (L122), `streamlit run app.py --server.port 8501 --server.headless true` (L123). |

### 7.2 Points fragiles

| ID | Sév. | Lignes | Constat |
|---|---|---|---|
| L-1 | **Critique** (sécurité) | 44, 67 | `[Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}` **désactive toute validation TLS** pour des requêtes qui envoient `Authorization: Bearer <PAT>` **et** téléchargent le code ensuite exécuté. Un intermédiaire réseau (proxy d'entreprise, Wi-Fi public) peut voler le jeton et injecter un ZIP piégé. Choix documenté (`ARCHITECTURE.md:332`, `Reference_maintenance:733`) mais contredit par `setup_drt_bayesien.bat:114-125`, qui recommande le CA bundle (`SSL_CERT_FILE`). |
| L-2 | **Critique** (secret) | 33-34, 44, 67 ; `.gitignore` | PAT en clair dans `token.txt` ; il est **interpolé (`%GITHUB_TOKEN%`) dans la ligne de commande PowerShell** (visible dans la liste des processus). `.gitignore` (19 lignes) **ne contient ni `token.txt`, ni `.version`, ni `venv/`, ni `eis_app/`** : si le `.bat` est exécuté depuis un clone du dépôt (il est à la racine), le jeton peut être committé. `set /p` (L34) est fragile : BOM, UTF-16 (Bloc-notes), espaces de fin → jeton invalide *sans message précis*. |
| L-3 | Majeur | 44, 51-53, 67-71, 77, 79, 83-84, 20 | **Erreurs silencieuses en cascade** : `catch { '' }` + `2^>nul` (L44) transforme TLS, DNS, proxy, 401/403, quota d'API en un unique « Hors ligne ou token invalide » (L52) puis lancement de la version locale ; `Expand-Archive … >nul 2>&1` (L77), `move … >nul 2>&1` (L79), `reg add … >nul 2>&1` (L20) masquent aussi leurs échecs. TLS 1.2 n'est pas forcé (PowerShell 5.1 sur .NET ancien → échec pris pour « hors ligne »). |
| L-4 | **Majeur** (perte d'état) | 75-86 | `rmdir /s /q "%APP_DIR%"` **avant** de savoir si l'extraction réussira ; aucun retour arrière ; `.version` est écrit (L85) et « Mise a jour appliquee. » affiché (L86) **sans vérifier** le résultat de l'extraction. Ce nettoyage détruit aussi tout ce que l'app écrit dans son arborescence : `config/error_structure.json` (→ ERR-1 après chaque mise à jour), `logs/`, `sessions/`, et les **exécutables Stan compilés** `vendor/bayes_drt2/stan_model_files/*` (recompilation de 1 à 3 min par modèle au prochain démarrage, cf. `setup_drt_bayesien.py:127`). |
| L-5 | Moyen | 44, 47-49, 55-60, 68-71, 78-81, 92-107 | `%VAR%` vs `!VAR!` : correct **aujourd'hui** seulement parce que les variables lues en `%…%` dans les blocs `( … )` sont des **constantes** (`TEMP`, `APP_DIR`, `VERSION_FILE`) ou définies par des lignes précédentes hors bloc. Toute variable *modifiée dans* un bloc puis relue en `%…%` dans ce bloc serait vide : c'est le piège qui attend un futur bloc « cmdstan » (résultats de `python -c`, `set /p`, `errorlevel`). Les tests `!errorlevel!` (L68, 100) sont, eux, corrects. `!` dans un chemin (`%~dp0`) est aussi consommé par l'expansion retardée. |
| L-6 | Moyen | 78 | `for /d %%i in (%TEMP%\eis_tmp\*)` : `%TEMP%` **non quoté**. Avec un chemin contenant des espaces (nom d'utilisateur « Jean Dupont » si `TEMP` n'est pas en forme 8.3), la boucle ne trouve rien → `EXTRACTED` sans `move` → `eis_app` **déjà supprimé** (L75) et absent. ⚠️ Non reproduit (dépend de la forme de `TEMP` sur le poste). |
| L-7 | Majeur | 51-54, 97-107 | **Sans réseau** : la mise à jour est ignorée (correct), mais `pip install -r requirements.txt` (L99) est relancé **à chaque démarrage**, non silencieux, sans `--timeout`/`--retries` ; il ne réussit hors ligne que si tout est déjà satisfait. Sinon L100-104 sort en « Verifiez votre connexion » et **interdit le démarrage** d'une version locale par ailleurs utilisable. |
| L-8 | Moyen | 94-95, 100 | `python -m venv` (L94) et `pip install streamlit` (L95) ne testent pas `errorlevel`. Aucun `where python`, aucune version minimale (Python ≥ 3.10 requis, §3). Le stub Microsoft Store de `python.exe` échoue, et l'erreur n'apparaît qu'en L100 avec un message trompeur (« connexion »). |
| L-9 | Mineur | 12, 121, 123 | Le navigateur est ouvert (L121) **avant** le démarrage du serveur ; port 8501 figé — s'il est occupé, Streamlit refuse de démarrer alors que le navigateur ouvre l'ancienne instance. |
| L-10 | Moyen | 20 | Écriture `HKLM` à chaque lancement : exige des droits administrateur (sinon échec masqué) et modifie la configuration de la machine. |
| L-11 | Moyen (chaîne d'approvisionnement) | 7, 65 | Le code exécuté est celui de `main` HEAD à l'instant du lancement ; aucune vérification d'intégrité, aucun épinglage de version, aucun retour arrière (L75) ; `pip` sans versions ni hashes (§3). |
| L-12 | Moyen (doc) | — | `README.md:11` et `launch_app.bat` sont présentés comme le démarrage rapide Windows ; `launch.bat` (token, auto-update) n'est documenté que dans `ARCHITECTURE.md:287-308` / `Reference_maintenance:662-700`, sans mention de `token.txt` ni du dépôt privé (M6, toujours ouvert). `launch_app.bat:20` : `pip install --quiet` sans test d'erreur ; `start` avant le serveur (L23). `launch_app.sh` (`set -e`, L15) échoue proprement. |

### 7.3 Ce qui est correct (à préserver)

`setlocal enabledelayedexpansion` (L2) ; chemins relatifs à `%~dp0` ; `if not exist "%TOKEN_FILE%"` avec message actionnable (L24-32) ;
repli sur la version locale si la vérification échoue (L51-54) ; `if !errorlevel! neq 0` après `pip install -r` (L100-104) ;
contrôle final de `app.py` (L112-116) ; `streamlit --server.headless true` (pas de double ouverture du navigateur).

### 7.4 Préparation CmdStan / `bayes_drt2` : état actuel et contraintes pour l'étape à ajouter

| Élément | Constat | Réf. |
|---|---|---|
| Deux mécanismes concurrents | `setup_drt_bayesien.bat` (manuel : pip + `install_cmdstan` + précompilation + `setx CMDSTAN`) **et** `app.py` (automatique : `ensure_drt_ready()` sans l'étape pip). L'en-tête du `.bat` dit « JAMAIS appelé au démarrage » ; `requirements-drt.txt:1-18` et `ARCHITECTURE.md:258-261` promettent une installation **automatique et sans étape manuelle** — fausse pour la partie pip : `ensure_drt_ready` retourne `False, "installez l'extra DRT (pip install -r requirements-drt.txt)"` si `cmdstanpy` manque (`setup_drt_bayesien.py:164-170`) et `app.py:61-62` s'arrête avant si `bayes_available()` est faux. | `setup_drt_bayesien.bat:10-12,54`, `app.py:50-70` |
| Où sont les artefacts | CmdStan : `%SystemDrive%\cmdstan` (`setup_drt_bayesien.py:56`, hors de `eis_app/` : survit aux mises à jour) — nécessite le droit d'écrire à la racine de `C:`. Modèles compilés : à côté des `.stan` **dans `eis_app/vendor/…`** : détruits par L75. | `setup_drt_bayesien.py:53-57,138-147` |
| Versions | `install_cmdstan()` sans version → dernière CmdStan ; `cmdstanpy>=1.2.0` sans plafond ; le code vendoré date de 2020 et lit des scalaires Stan (avertissement `FutureWarning`, §3). Le banc de cet audit utilise CmdStan 2.36.0 / cmdstanpy 1.3.0. | `requirements-drt.txt:20` |
| `setx CMDSTAN` | Persistant pour les **futures** sessions seulement (le processus courant a `set CMDSTAN=`, L83 du `.bat`) ; `register()` (`setup_drt_bayesien.py:85-93`) retrouve de toute façon `cmdstan-*` sous le dossier par défaut. | `.bat:83-84` |
| Durée / blocage | Téléchargement + compilation (`compiler=True` : toolchain C++ sous Windows) puis 2 compilations Stan de 1-3 min : dans `app.py` c'est un `st.cache_resource` sans délai maximal (DRT-7). ⚠️ Comportement Windows non vérifié depuis ce banc Linux. | `setup_drt_bayesien.py:127` |
| Réseau/proxy | `SSL_HINT` (`.bat:114-125`, `.py:202-213`) recommande proxy + CA bundle — l'inverse de la désactivation TLS de `launch.bat` (L-1). | |

**Exigences pour une étape « cmdstan/bayes_drt2 » dans `launch.bat`** : exécuter dans le venv (`!VENV_DIR!\Scripts\python.exe`) ; utiliser **exclusivement `!VAR!`** dans les blocs ;
tester `errorlevel` après chaque commande et distinguer « hors ligne » de « échec » ; être **idempotente et hors ligne-tolérante** (ne rien faire si `cmdstan_path()` est valide
et si les modèles compilés existent) ; ne pas s'exécuter *après* le `rmdir` de L75 sans déplacer les modèles compilés hors de `eis_app/` (ou ne plus effacer tout `eis_app/`) ;
épingler les versions (`cmdstanpy`, CmdStan) ; journaliser dans un fichier ; ne pas dépendre de `token.txt` pour le téléchargement de CmdStan (hôte public différent) ;
prévoir un délai maximal et un message de progression (l'étape peut durer plusieurs minutes) ; ne pas réintroduire la désactivation TLS.

---

## 8. Tests existants (`tests/`)

🔬 Exécutés dans une copie hors dépôt avec les dépendances les plus récentes (Annexe A) : **82 tests collectés (79 fonctions) — 80 passent, 2 sautés** sans CmdStan ;
avec CmdStan : **81 passent + 1 test `slow` (HMC) qui passe** (32,7 s). Durée totale ≈ 4-8 s. Couverture d'instructions mesurée par `pytest-cov`.

### 8.1 Couverture réelle par module (sans CmdStan ; entre parenthèses : avec CmdStan)

| Zone | Couverture | Lignes non couvertes notables |
|---|--:|---|
| `core/models`, `cv_models`, `cv_pipeline`, `fits/base`, `fits/weighting`, `plotting/theme` | 100 % | — |
| `core/config` 96 % · `fits/kk_validation` 96 % · `core/logger` 95 % · `core/calibration` 92 % · `fits/randles_full` 92 % · `core/cv_loader` 93 % | 92-96 % | `randles_full.py:129,169-175,221` (branche d'exception FIT-1 non testée) |
| `core/robust_loader` 87 % · `core/loader` 83 % · `fits/registry` 88 % · `fits/physics` 82 % | 82-88 % | `loader.py:139-148` (positionnel), `:371-376` (interpolation de grille) |
| `fits/error_structure` | 79 % | lecture/écriture JSON dégradée (`:358-365,393-399`), ré-échantillonnage de grille (`:123-135,165-167`) |
| `fits/drt_fit` | 51 % (**82 %**) | intégration réelle, `rp_fallback`, IC |
| `core/mpr_converter` | 70 % | lecture réelle d'un `.mpr` (seul le chemin d'erreur est testé) |
| `core/cv_peaks` | 21 % | pratiquement tout (`detect_redox_peaks` L31-57) |
| `core/experiment_io` | **16 %** | `save_experiment`, `load_experiment` (v1 et v2), `apply_exclusions` |
| `exports/exporter` | **18 %** | tous les exports sauf les CSV de calibration ; `export_params_csv` (bug B-EXP), `export_full_zip` |
| `plotting/eis_plots` 18 % · `cv_plots` 27 % · `kk_plots` **0 %** | | figures non testées (sauf 3 tests de cohérence calibration/bare) |
| **`core/pipeline`** | **0 %** | orchestrateur EIS, `recompute_drt`, `_characterize_error_structure_upfront` |
| **`core/validator`** | **0 %** | validation KK par réplicat, dérive (ERR-4) |
| `ui/tabs`, `pages/*`, `app.py`, `setup_drt_bayesien.py` | **0 %** (jamais importés) | tout |
| **Total (fits+core+plotting+exports+ui+pages+scripts)** | **29 %** des instructions (4 573) | |

### 8.2 Ce qui est testé vs ce qui ne l'est pas

**Bien couvert** : parseurs EC-Lab et signe de Im(Z) (19 tests `test_loader.py`, dont l'absence de double inversion et la reconstruction de fréquence) ·
récupération de Rct de bout en bout loader→fit (`test_fits.py:112`, écart ≤ 5 %) · pondération d'Orazem (régression NNLS, χ²ᵣ ≈ 1 avec σ connu, covariance
non rééchelonnée, refus sans caractérisation, persistance, équivalence de signe : `test_measurement_model.py`, 10 tests) · gardes de fit et registre
(`test_diagnostics.py`) · cohérence figure ↔ export ↔ `core/calibration` (`test_calibration.py`) · invariants « référence nue = affichage seul » (`test_bare_reference.py`) ·
bornes de config (`test_config.py`) · CV loader/pipeline (14 tests).

**Non testé, alors que la refonte y touchera** : `run_pipeline` de bout en bout et son traitement d'erreurs (ERR-1, ERR-3) · `recompute_drt` · `validator` (ERR-4) ·
persistance concurrente/croissante de la structure d'erreur (ERR-2) · les exports (bug B-EXP) et le ZIP · `experiment_io` (aller-retour ZIP) · normalisation Nyquist (`A_eis.py:178`) ·
détection de pics CV · qualité **numérique** de la DRT (ordre de grandeur, positivité — DRT-1) · lanceurs et bootstrap CmdStan.

### 8.3 Qualité des tests : observations

- `tests/conftest.py:24-38` (`autouse`) **pré-remplit une structure d'erreur persistée** pour tous les tests : les fits « sans réplicats » y réussissent toujours. Le chemin
  réel du premier usage (refus → fit vide, ERR-1) n'est donc jamais exercé hors de `test_measurement_model.py:243,261` (qui teste la fonction, pas le pipeline).
- Tests de la DRT : voir §4.6 — n'auraient détecté ni DRT-1, ni DRT-10, ni DRT-4.
- Toutes les fixtures sont **synthétiques** ; aucune mesure réelle de référence (« golden ») ni comparaison à une implémentation indépendante (impedance.py, résultats publiés)
  pour Lin-KK ou la DRT. Pour un objectif de « calculs transparents avec sources », c'est le manque principal.
- Reliques : docstring « FFT/Wiener DRT » (`test_fits.py:30`), `_DRT_CONFIG` avec clés mortes (`test_fits.py:67-71`), helper `_zarc_spectrum` inutilisé (`:49`), imports inutiles (`test_cv.py:8`, `test_diagnostics.py:39`).
- CI (`validate.yml`) : Python 3.12 seul, pas de couverture ni de seuil, pas de lint/typage, pas de job avec `cvxopt`/CmdStan (donc les 2 tests DRT réels ne tournent jamais en CI), `compileall` ne couvre ni `vendor/`, ni `tests/`, ni `setup_drt_bayesien.py`.

### 8.4 Bug reproduit par exécution (hors couverture des tests) — B-EXP

`pages/E_export.py:51-56` (« Paramètres fit Randles (CSV) ») appelle `export_params_csv(sessions)` **sur la session non filtrée** ; la fonction écrit l'en-tête **une seule fois** à partir
de la première ligne puis y aligne des lignes de modèles différents (`exports/exporter.py:37-52`). 🔬 Avec un groupe portant `randles_full` et `drt_bayes` :

```
electrode,concentration,model,Rct,Rct_std,chi2_reduced,converged,Re,Re_prime,Cb,Qdl,alpha,R_D,tau_d
1,1e-09,randles_full,4,0.1,1,True,1,2,3,5,6,7,8
1,1e-09,drt_bayes,10.0,0,1,True,11.0,-5.2,80,peak_penultimate,optimize
```
Sur la 2ᵉ ligne, `Rp=11.0` apparaît sous « Re », `tau_Rct` (en ln τ) sous « Re_prime », `n_tau` sous « Cb », et `rct_source`/`drt_mode` sous « Qdl »/« alpha » ; il manque 3 colonnes.
Seul l'export ZIP filtre `randles_full` avant l'appel (`exporter.py:479-485`). Fichier téléchargé nommé `parametres_randles.csv` (`E_export.py:55`).

---

## 9. Incohérences code réel ↔ documents de référence

Documents comparés : `Reference_maintenance` (« REFERENCE_MAINTENANCE.html » dans l'énoncé) et `ARCHITECTURE.md` (« ARCHITECTURE_v2.md »), plus `README.md`, `METHODES.md`,
`vendor/README.md`, `requirements-drt.txt`, `setup_drt_bayesien.bat`. Le code fait foi. Les écarts déjà relevés par `AUDIT_REPORT.md` (M1-M7, m1-m6, c1-c4) sont **repris s'ils sont toujours
ouverts** : aucun n'a été corrigé dans les documents depuis (`ARCHITECTURE.md` toujours « 4 méthodes » L12, « Modulus » L208).

| ID | Document (réf.) | Affirmation | Réalité (réf. code) |
|---|---|---|---|
| D-1 | `Reference_maintenance:449,588-591,495-498` | 4 méthodes : Randles DE, Randles contraint, DRT, circulaire Kasa ; fichiers `randles_classique.py`, `randles_contraint.py`, `circulaire_fit.py` | 2 plugins ; fichiers inexistants (§0, §5.1) |
| D-2 | `Reference_maintenance:542` | « 4 fits en parallèle : randles_full · drt_bayes » (incohérent en interne) | Fits **séquentiels** (`pipeline.py:139-262`) ; 2 modèles |
| D-3 | `Reference_maintenance:437` | Pile « … lmfit » | `lmfit` retiré (`AUDIT.md` ancien C6) ; non importé |
| D-4 | `Reference_maintenance:711` ; `ARCHITECTURE.md:208` ; `README.md:53` ; `METHODES.md:344,378,404-435,849,960` | Pondération « Modulus », `fit.alpha_noise`, modes `modulus`/`sigma`, `weight_mode` | Supprimés : pondération **unique** d'Orazem (`weighting.py:1-25`) ; seul `MEASUREMENT_MODEL.md` est juste |
| D-5 | `ARCHITECTURE.md:12` | « 4 méthodes comparatives » | 2 (`registry`) — contredit le §5 du même document |
| D-6 | `ARCHITECTURE.md:16-17`, `Reference_maintenance:453` | Signal θ_EIS = 1 − Rct,bare/Rct,ap ; « LOD ≈ 10⁻¹⁷ M » | Signal calculé = `\|Rct_probe − Rct_c\|/Rct_probe` (`calibration.py:79`), référence **probe** ; `bare_reference` en **affichage seul** (`models.py:157-163`) ; `theta_EIS` non appelée ; **aucun calcul de LOD** dans le code |
| D-7 | `ARCHITECTURE.md:219` | Schéma `Re — [R'e // Cb] — [Rct // CPE] — ZD` | Réel (`physics.py:41-63`) : `Z_eq = R'e + (Rct+Z_D)/[1+Qdl(jω)^α(Rct+Z_D)]`, `Z = Re + Z_eq/(1+jω Cb Z_eq)` : Z_D **dans** la branche Rct//CPE, Cb en parallèle de l'ensemble |
| D-8 | `ARCHITECTURE.md:36-90` | Arborescence | Omet `core/robust_loader.py`, `fits/weighting.py`, `fits/error_structure.py`, `vendor/`, `setup_drt_bayesien.*`, `requirements-drt.txt`, `THIRD_PARTY_LICENSES.md`, `project`, `METHODES.md`… ; cite `.version` (créé par le lanceur, pas dans le dépôt) |
| D-9 | `ARCHITECTURE.md:119` | Le flux cite `C_comparatif.py` | N'existe pas (contredit L104-106) |
| D-10 | `ARCHITECTURE.md:152` | ZIP : `probe/electrode_N/rep_R.bin` | Format v2 : `experiment.yaml` + `probe/{eis,cv}/electrode_N/*.txt` (`experiment_io.py:5-12`) ; le `.bin` est le v1 (`_load_rep_list_v1`) |
| D-11 | `ARCHITECTURE.md:72,283` | Exports PNG, HTML | `export_figure_png/html` jamais branchées (§2.1 #6) |
| D-12 | `ARCHITECTURE.md:258-261`, `README.md:56-58`, `Reference_maintenance:729`, `requirements-drt.txt:10-18` | DRT « installée automatiquement, aucune étape manuelle » | Vrai pour CmdStan/compilation (`app.py:50-70`), **faux pour le pip** de `cvxopt`/`cmdstanpy` (§7.4, DRT-8) |
| D-13 | `setup_drt_bayesien.bat:10-12` | « JAMAIS appelé au démarrage » | `app.py` fait `ensure_drt_ready()` au démarrage |
| D-14 | `ARCHITECTURE.md:287-308`, `Reference_maintenance:662-700` | Flux du lanceur ; « seul fichier à distribuer » ; prérequis : Python | Exige `token.txt` (dépôt privé), écrit `HKLM`, désactive TLS, efface `eis_app/` (§7) ; M6 toujours ouvert ; `Reference_maintenance:735` propose « rendre le repo public » |
| D-15 | `ARCHITECTURE.md:332`, `Reference_maintenance:733` | « `ssl.CERT_NONE` dans les requêtes urllib » | Le lanceur utilise PowerShell + `ServerCertificateValidationCallback` (L44,67), pas urllib |
| D-16 | `Reference_maintenance:512,554,656,728` | `ui/tabs.py → render_tabs()` 6 onglets ; erreur `ImportError: render_tabs` | `render_tabs` n'existe plus ; `render_eis_tabs` (4 onglets) et `render_cv_tabs` (4 onglets) (`ui/tabs.py:443,479`) |
| D-17 | `Reference_maintenance:503,550` | `eis_plots.py` : 7 figures dont « Bode » | ~20 constructeurs ; `bode_figure` jamais appelée |
| D-18 | `Reference_maintenance:632,726` | `ModuleNotFoundError: exports` → « créer `exports/__init__.py` » | Cause réelle = `.gitignore` (V3, corrigé) ; `ARCHITECTURE.md:335-339` le dit, le HTML non |
| D-19 | `ARCHITECTURE.md:6` vs `:361` | MàJ 13/07/2026 vs « rédigé le 29/05/2026 » | Dates incohérentes |
| D-20 | `ARCHITECTURE.md:314-323` | Tableau des paramètres physiques (T, D, Fv, h, d, xe, S, C0) présenté comme configuration | Ces paramètres n'ont **aucun effet** (§2.3) |
| D-21 | `README.md:51-53` | Modèles : « Fit circulaire », « Randles contraint », « Randles complet … pondération Modulus » | 2 modèles ; pondération d'Orazem |
| D-22 | `README.md:9-12` | Démarrage rapide Windows = `launch_app.bat` | Le lanceur documenté ailleurs est `launch.bat` ; comportements différents (§7.2 L-12) |
| D-23 | `METHODES.md` (citations `fichier:ligne`) | Ex. `randles_full.py:residuals:126-135`, `fit:140-144`, `weighting.py:52-56` | Décalées d'≈ 10 lignes ou périmées (réel : `randles_full.py:136-145`, `147-156` ; `weighting.py` = 75 lignes, autre contenu) |
| D-24 | `METHODES.md:4-6` vs `ARCHITECTURE.md:6` | `METHODES.md` tient l'architecture pour « réputée obsolète » ; `ARCHITECTURE.md` se dit « réalignée sur l'état réel » | Les deux ont tort sur la pondération (D-4) : aucune doc de méthode n'est entièrement fiable hors `MEASUREMENT_MODEL.md` |
| D-25 | `vendor/README.md:4` | « Rien ici ne fait partie du code applicatif EduGradia/Interface » | Copier-coller d'un autre projet |
| D-26 | `project:12-13` | « Lire `ARCHITECTURE_v2.md` et `REFERENCE_MAINTENANCE.html` » ; règle « ne jamais modifier `core/ fits/ exports/ config/` » | Fichiers absents ; la règle est caduque (cahier des charges d'une refonte UI déjà réalisée) |
| D-27 | `AUDIT_REPORT.md:343-349` | Livraisons prévues : `ARCHITECTURE_v3.md`, `REFERENCE_MAINTENANCE_v3.html`, `SCIENTIFIC_METHODS.md` | Jamais livrées |
| D-28 | `requirements.txt:1` | `streamlit>=1.32.0` | Minimum réel ≈ 1.51 (§3) |
| D-29 | `pages/0_import.py:4`, `A_eis.py:1-6` | « pages lisent `experiment` » ; « calibration OLS log(Rct) vs log([c]) » | Elles lisent `experiment_clean` ; le signal est `\|ΔRct\|/Rct_probe` régressé sur log10[c] (le log-log ne concerne que la DRT) |

**Documents fiables** : `MEASUREMENT_MODEL.md` (pondération d'Orazem), `vendor/README.md` (provenance, sha256 vérifiés), `requirements-optional.txt` (justification de galvani),
`fits/*.py` (docstrings alignées sur le code), `THIRD_PARTY_LICENSES.md`.

---

## 10. Conclusion — liste priorisée

### 10.0 Synthèse en cinq lignes

1. Le code est **globalement sain et bien cloisonné** (aucun import Streamlit dans `core/` ni `fits/` ; 80 tests verts + 2 tests DRT sautés sans CmdStan, avec les dépendances les plus récentes ; Randles retrouve Rct à ≤ 0,13 % sur mes jeux synthétiques).
2. La refonte annoncée est **en grande partie déjà faite** : il reste à généraliser le circuit (fit Orazem sur circuit utilisateur), à **durcir la DRT** et à éliminer les résidus (§0).
3. Le défaut le plus grave est la **DRT actuellement fausse en silence** avec les réglages par défaut (Rp < 0, Rct négatif) et sans aucune garde (DRT-1/2/3).
4. Trois autres **échecs silencieux** existent : fits vides affichés « Analyse terminée » (ERR-1), résultats périmés (B-STATE), CSV de paramètres mal aligné (B-EXP) — plus, à confirmer, les pics CV faux si les fichiers sont des boucles (CV-1).
5. Le lanceur cumule deux risques majeurs (TLS désactivé avec envoi du jeton ; effacement complet de `eis_app/` à chaque mise à jour) qui touchent directement l'étape cmdstan à ajouter.

### 10.1 Supprimable sans risque

*Vérifié : aucun appelant (tests compris) — ≈ 400 lignes de fonctions mortes, plus configuration et dépendances.*

| Prio | Élément | Réf. | Risque |
|--:|---|---|---|
| P0 | Fonctions sans aucun appelant : `Rct_bare_theory`, `DRTBayesModel.predict`, `export_spectra_csv`, `export_figure_html`, `export_figure_png`, `bode_figure`, `kk_figure`, `drt_reconstruction_figure`, `cv_figure_electrode`, `open_cv_calibration_matplotlib_window`, `_collect_deleted_points`, alias `zip_experiment`, helper de test `_zarc_spectrum` | §2.1 #1,3,5,6,9,10,13,14,15 | Nul |
| P0 | Imports/variables inutilisés | §2.2 | Nul |
| P0 | Config morte : `fit.n_monte_carlo`, `fit.drt_wiener_W`, `fit.drt_n_z` (+ injection dans `test_fits.py:67-71`), bloc `export.*` | §2.3 | Nul |
| P0 | Sections `physics`, `geometry`, `conditions` (YAML + classes Pydantic) **et** les 6 champs « Paramètres physiques » de `A_eis.py:371-379` + `_merge_overrides` (L69-87) | §2.3 | Nul sur les résultats (aucun lecteur) ; changement visible dans l'UI |
| P0 | Modèles fantômes `"circular"` / `"randles_constrained"` (`A_eis.py:351-352`, `:402`) et lignes correspondantes de `README.md:51-52` | §5.3 | Nul |
| P0 | Clés `session_state` `eis_session`, `eis_validation` (`app.py:25,27`, `0_import.py:116`, `1_pretraitement.py:886`) — **à remplacer par la réinitialisation des vraies clés** (B-STATE) | §2.5 | Nul (mais corriger B-STATE en même temps) |
| P0 | Branches d'export CV mortes + `export_cv_calibration_csv_from_result` ; filtre `_lc_/_gcv_` (`exporter.py:127`) | §2.6 | Nul |
| P0 | Dépendances `scikit-learn`, `kaleido` ; `pytest` → dépendances de dev | §3 | Nul |
| P0 | Page « Prédiction » (`pages/D_inference.py`) retirée de la navigation tant qu'elle est un stub | `app.py:89-91` | Nul |
| P0 | Fichiers `project`, `Reference_maintenance` (à archiver/remplacer), doublon `launch_app.bat` (à décider) | §2.8 | Nul (documentaire) |
| P1 | *Tests seulement* : `Cdl_brug`, `theta_EIS` (⚠️ **décision produit** : θ_EIS est documenté partout mais non calculé — supprimer *ou* l'implémenter), `load_cv_curve` + `CVCurve`, `export_cv_calibration_csv`, `cv_calibration_figure` (avec leurs tests) | §2.1 #2,7,11,12 | Faible |
| P1 | Wrapper `linKK` → appel direct de `lin_kk` (`validator.py:121-139`) ; `error_structure` dans le `skip` du registre | §2.1 #4,18 | Faible |
| P1 | Champs de dataclass morts (`Rct_sigma`, `drt_S`, `drt_lnGamma`, `kk_residuals`, `EISSpectrum.validation/f_min_valid/f_max_valid`, `ValidationResult.sigma_*`) | §2.4 | Faible (décider d'abord si la restriction à la plage KK-valide, ERR-7, sera implémentée) |
| P1 | Commentaires « NOTE (ajout hors périmètre initial) » et docstrings périmées | §2.7 | Nul |

**À ne pas supprimer par erreur** : validateurs Pydantic (`config.py:80,87`), classes de modèle découvertes dynamiquement, propriétés de `ParsedFile`, `fits/weighting.py:32` (ré-export voulu), les 13 modèles Stan vendorés, `MEASUREMENT_MODEL.md`.

### 10.2 À conserver / adapter

| Élément | Adaptation attendue |
|---|---|
| `randles_full` : solveur `least_squares` borné, covariance `(JᵀJ)⁻¹`, χ²ᵣ + intervalle, gardes I7 (§5.2) | Base du moteur générique ; remplacer `Z_randles_full`/`_PARAM_NAMES` par un circuit utilisateur ; corriger FIT-1…FIT-5 ; guess/bornes **par élément** ; multi-départ ; conditionnement rapporté |
| `weighting.py` + `error_structure.py` (méthode d'Orazem) et `MEASUREMENT_MODEL.md` | Conserver la méthode ; **supprimer l'état global caché** (fichier JSON append-only, ERR-2) au profit d'une structure explicite portée par la session/l'export ; faire **échouer visiblement** l'absence de structure (ERR-1) |
| `FitResult` | Élaguer (§2.4) ; ajouter le **paramètre désigné pour la calibration** au lieu de `Rct` en dur ; unifier la sémantique de `chi2_reduced` (DRT-5) ; `converged` réel pour la DRT |
| Convention de signe `Zim = −Im(Z) > 0` et tests associés (`test_loader.py`, `test_measurement_model.py:274`) | Inchangée ; garder ces tests intacts |
| `robust_loader.py` / `loader.py` | Conserver ; corriger le tri des CV (CV-1) après confirmation |
| `core/calibration.py` | Conserver comme source unique ; y rattacher `cv_plots` (CPL-2) ; ne plus ignorer `converged`/`warnings` (FIT-1, DRT-3) |
| `validator.py` / `kk_validation.py` | Corriger la dérive (ERR-4), le choix de M (ERR-5), unifier les critères (ERR-6) ; documenter la source de chaque formule |
| `vendor/bayes_drt2` + `fits/drt_fit.py` | Conserver le vendoring **ou** passer à une dépendance ; dans tous les cas : réglages validés (§4.4), gardes qualité, seed explicite, sorties homogènes (DRT-1…13) |
| `setup_drt_bayesien.py` (`install`, `register`, `verify`, `precompile`, `ensure_drt_ready`) | Réutiliser dans le lanceur ; unifier avec `app.py`, épingler CmdStan/cmdstanpy, sortir les modèles compilés de `eis_app/` |
| `tests/` (structure, fixtures synthétiques, `test_bare_reference.py`) | Conserver et **étendre** (§10.4 étape 2) |

### 10.3 Bloquants et points risqués pour la refonte

| # | Risque | Pourquoi bloquant | Réf. |
|--:|---|---|---|
| R1 | **Cadrage périmé** de la mission (4 modèles, 3 fichiers à supprimer, intégration bayes_drt2 à faire) | Planifier sur cette base produirait des travaux inutiles et ignorerait les vrais chantiers | §0 |
| R2 | **DRT fausse en silence** (réglages par défaut) et sans garde ; Rct heuristique instable ; calibration qui écarte les points invalides sans avertir | Toute comparaison Randles/DRT et toute calibration DRT sont potentiellement fausses | DRT-1/2/3/10 |
| R3 | **Fits vides « réussis »** quand la structure d'erreur manque, situation **réamorcée à chaque mise à jour** du lanceur | Le futur fit unique dépend entièrement de cette structure | ERR-1, L-4 |
| R4 | **Lanceur** : TLS désactivé + jeton en clair (non ignoré par git) ; `eis_app/` effacé avant extraction (perte de l'état, des modèles Stan compilés) ; pip à chaque lancement (bloque hors ligne) | Ajouter une étape cmdstan longue et réseau-dépendante sur cette base amplifie chaque défaut | L-1…L-11 |
| R5 | **Résultats périmés** après rechargement/re-prétraitement ; **CSV de paramètres mal aligné** | Erreurs d'analyse/d'export non détectables par l'utilisateur | B-STATE, B-EXP |
| R6 | **Filet de tests absent** sur exactement ce que la refonte touche : `pipeline` 0 %, `validator` 0 %, exports 18 %, `experiment_io` 16 %, UI/pages 0 % ; aucune donnée réelle de référence | Impossible de refactorer « à comportement constant » | §8 |
| R7 | **CV** : tri par potentiel d'une boucle (à confirmer) | Peut invalider tous les résultats CV | CV-1 |
| R8 | **Dépendances non épinglées** (résolution du jour, à chaque lancement) ; bornes fausses (`streamlit`) ; API cmdstanpy en évolution ; vendor sans commit amont identifié | Reproductibilité des résultats et stabilité de la DRT | §3 |
| R9 | **Couplage** `core` ↔ `fits` masqué par des imports locaux ; logique métier dans les pages ; `Rct` câblé dans `FitResult`/calibration/exports | Freine tout moteur de circuit générique | CPL-1/3/4, §5.3 |
| R10 | **Documentation contradictoire** (deux docs de référence fausses sur la pondération, le nombre de modèles, le signal de calibration, l'installation DRT) | L'objectif « sources citées, calculs transparents » n'a pas de base documentaire fiable ; θ_EIS/LOD sont annoncés mais non calculés | §9 |
| R11 | **Point de conception à anticiper** : un « circuit défini par l'utilisateur » implique d'interpréter une description saisie ; ne pas évaluer de texte libre (`eval`) — aucun mécanisme de ce type n'existe aujourd'hui, à concevoir dès le départ | Sécurité | — |

### 10.4 Ordre de travail suggéré

1. **Recadrer** la refonte (R1) et trancher les décisions ouvertes (ci-dessous).
2. **Filet de sécurité d'abord** : tests de caractérisation sur les jeux synthétiques de l'Annexe A (pipeline complet avec/sans réplicats, exports, `validator`, `experiment_io`, DRT sur la matrice §4.4, CV en boucle) et job CI avec `cvxopt` + CmdStan ; épingler les dépendances.
3. **Corriger les échecs silencieux** : DRT-1/2/3, ERR-1, B-STATE, B-EXP, (CV-1 après confirmation).
4. **Sécuriser le lanceur** (L-1, L-2, L-4, L-7) puis y ajouter l'étape cmdstan selon §7.4.
5. **Nettoyer** (P0 puis P1 de §10.1) et corriger la documentation (§9).
6. **Refonte** : moteur de fit sur circuit utilisateur, DRT durcie, transparence (sources par méthode).

### 10.5 Décisions à prendre / informations manquantes

- Les fichiers CV sont-ils des **cycles complets** ou des balayages simples ? (CV-1)
- Le signal cible est-il θ_EIS (référence électrode nue) ou le ΔRct normalisé au probe actuel ? Faut-il calculer une LOD ? (D-6)
- Le dépôt reste-t-il **privé avec jeton** ? Quelle politique TLS/proxy sur les postes cibles ? (L-1, L-2)
- Rejouer la matrice DRT (§4.4) avec les **versions de CmdStan/cmdstanpy du poste Windows** avant de choisir `nonneg`/`init_from_ridge`.
- Où doit vivre la structure d'erreur d'Orazem (session, fichier de campagne exporté, base locale) et comment est-elle versionnée ? (ERR-2)
- Conserver le **vendoring** de `bayes_drt2` ou revenir à un paquet installable (existence et épinglage à vérifier) ?
- Le lanceur `launch.bat` et `launch_app.bat` : en garder un seul ?

---

## Annexe A — Reproductibilité des mesures 🔬

**Principe** : rien n'a été exécuté dans le dépôt audité (`git status` vide en fin d'audit ; `core/logger.py` crée `logs/` à l'import et
`error_structure.py` écrit `config/error_structure.json`, ce qui aurait pollué l'arborescence). Tout a tourné dans une **copie** avec un venv dédié.

```bash
WORK=/chemin/vers/un/dossier/temporaire
cp -r Interface-EIS-CV-measurement $WORK/repo && python3 -m venv $WORK/venv
$WORK/venv/bin/pip install -r $WORK/repo/requirements.txt vulture pyflakes pytest-cov cvxopt cmdstanpy
$WORK/venv/bin/python -c "import cmdstanpy; cmdstanpy.install_cmdstan(dir='$WORK/cmdstan', version='2.36.0', cores=4)"
export CMDSTAN=$WORK/cmdstan/cmdstan-2.36.0 PYTHONPATH=$WORK/repo ; cd $WORK/repo
$WORK/venv/bin/pyflakes app.py core fits plotting exports ui pages tests setup_drt_bayesien.py
$WORK/venv/bin/vulture app.py core fits plotting exports ui pages setup_drt_bayesien.py --min-confidence 60
$WORK/venv/bin/python -m pytest tests -q -p no:cacheprovider -m "not slow" --cov=core --cov=fits --cov=plotting --cov=exports --cov=ui --cov=pages --cov-report=term-missing
$WORK/venv/bin/python -m pytest tests/test_drt.py -v -p no:cacheprovider -m slow      # HMC : 32,7 s
```

**Environnement du banc** : Linux, Python 3.11.15 ; streamlit 1.64.0, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, plotly 7.1.0, matplotlib 3.11.2, pydantic 2.13.5, PyYAML 6.0.3,
kaleido 1.4.0, scikit-learn 1.9.1, eclabfiles 0.4.1, pytest 9.1.1, cvxopt 1.3.3, cmdstanpy 1.3.0, CmdStan 2.36.0 (g++ système). **Non reproduit** : Windows, RTools/mingw, `launch.bat`
(les constats §7 sont des lectures de code, sauf mention), navigateur/Streamlit interactif, versions CmdStan du poste cible.

Inspection des roues Streamlit (§3) : `pip download streamlit==<v> --no-deps` puis présence de `streamlit/navigation/` (1.32.0 : absent ; 1.36.0 : présent) et de la signature
`width:` dans `streamlit/elements/plotly_chart.py` (1.50.0 : absent ; 1.51.0 : présent).

Scripts utilisés (exécutés avec `PYTHONPATH` sur la copie ; `CMDSTAN` défini pour ceux qui appellent la DRT).

### A.1 — CSV de paramètres mal aligné (B-EXP, §8.4)

```python
import numpy as np, warnings
from core.models import EISSession, ConcentrationGroup, EISSpectrum, FitResult
from exports.exporter import export_params_csv
def sp(): 
    f=np.logspace(5,-1,40); return EISSpectrum(label="x",f=f,Zre=np.ones(40),Zim=np.ones(40),concentration=1e-9,step="hybridization",n_points=40)
s=sp()
z=np.ones(40)
fr1=FitResult(model_name="randles_full",params={"Re":1,"Re_prime":2,"Cb":3,"Rct":4,"Qdl":5,"alpha":6,"R_D":7,"tau_d":8},params_std={},Zfit_re=z,Zfit_im=z,chi2_reduced=1,residuals_re=z,residuals_im=z,Rct=4,Rct_std=0.1,converged=True)
fr2=FitResult(model_name="drt_bayes",params={"Rct":10.0,"Rp":11.0,"tau_Rct":-5.2,"n_tau":80,"rct_source":"peak_penultimate","drt_mode":"optimize"},params_std={},Zfit_re=z,Zfit_im=z,chi2_reduced=1,residuals_re=z,residuals_im=z,Rct=10,Rct_std=0,converged=True)
sess=EISSession(); sess.groups=[ConcentrationGroup(concentration=1e-9,spectrum=s,fit_results={"randles_full":fr1,"drt_bayes":fr2})]
print(export_params_csv({1:sess}).decode())
```

### A.2 — Pipeline complet sur jeu synthétique de type Randles (Rct, DRT, dérive) (§4.4, ERR-4)

```python
import numpy as np, time, io, json
from core.config import load_config, config_to_dict
from core.pipeline import run_pipeline
from fits.physics import Z_randles_full
rng=np.random.default_rng(0)
f=np.logspace(5,-1,60); w=2*np.pi*f
def make(Rct, noise=0.005, seed=0):
    r=np.random.default_rng(seed)
    Z=Z_randles_full(w,Re=200.,Re_prime=20.,Cb=1e-9,Rct=Rct,Qdl=2e-6,alpha=0.9,R_D=Rct*0.3,tau_d=0.5)
    Zre=Z.real*(1+noise*r.standard_normal(len(f))); Zim=(-Z.imag)*(1+noise*r.standard_normal(len(f)))
    s="freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm\n"+"\n".join(f"{a:.6g}\t{b:.6g}\t{c:.6g}" for a,b,c in zip(f,Zre,Zim))
    return s.encode()
fa=[]
for k in range(3): fa.append(dict(content=make(3000,seed=k),filename=f"probe_r{k}.txt",step="probe",concentration=0.0))
for ci,(c,R) in enumerate([(1e-9,3500),(1e-8,4200),(1e-7,5200)]):
    for k in range(3): fa.append(dict(content=make(R,seed=10+ci*3+k),filename=f"c{ci}_r{k}.txt",step="hybridization",concentration=c))
cfg=config_to_dict(load_config())
t=time.time()
sess,val=run_pipeline(fa,cfg,active_models=["randles_full","drt_bayes"])
print("pipeline s:",round(time.time()-t,1))
for g in [("probe",sess.probe)]+[(f"{x.concentration:.0e}",x.spectrum) for x in sess.groups]:
    for m,fr in g[1].fit_results.items():
        print(g[0],m,"Rct=%.1f"%fr.Rct,"chi2r=%.3g"%fr.chi2_reduced,"conv",fr.converged,"warn",fr.warnings[:2], {k:(round(v,3) if isinstance(v,float) else v) for k,v in fr.params.items() if k in("tau_Rct","rct_source","Rp")})
print("replicates fit_results probe:", [list(s.fit_results) for s in sess.probe_replicate_spectra])
print("val keys",list(val))
```

### A.3 — Matrice DRT : scénario du test du dépôt, grilles, bruit, graines (§4.4)

```python
import numpy as np, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from vendor.bayes_drt2.inversion import Inverter
from core.models import EISSpectrum
import fits.drt_fit as drt
def spec(n=71,f0=5,f1=-2,noise=0.0,seed=0):
    f=np.logspace(f0,f1,n); w=2*np.pi*f
    Z=10+50/(1+1j*w*1e-3)+80/(1+1j*w*1e-1)
    r=np.random.default_rng(seed); Z=Z*(1+noise*(r.standard_normal(n)+1j*r.standard_normal(n)))
    return f,Z
f,Z=spec()
sp=EISSpectrum(label="t",f=f,Zre=Z.real,Zim=-Z.imag,concentration=0,step="probe",n_points=len(f))
fr=drt.DRTBayesModel().fit(sp,{"fit":{"drt":{"mode":"optimize"}}})
print("REPO TEST SCENARIO: Rct=%.2f Rp=%.2f src=%s chi2=%.3g recon_err=%.3g (truth Rp=130, arcs 50/80)"%(fr.Rct,fr.params["Rp"],fr.params["rct_source"],fr.chi2_reduced,fr.reconstruction_error))
def run(label,f,Z,**kw):
    inv=Inverter(); inv.fit(f,Z,mode='optimize',**kw)
    tau=inv.distributions['DRT']['tau']; g=inv.predict_distribution('DRT',tau=tau); Zf=inv.predict_Z(f)
    print("%-38s Rp=%9.1f gmin=%9.1f maxrelerr=%.3g"%(label,inv.predict_Rp(),g.min(),np.max(np.abs(Zf-Z)/np.abs(Z))))
for n,f0,f1 in [(71,5,-2),(60,5,-1),(40,5,-1),(80,6,-1)]:
    f,Z=spec(n,f0,f1)
    run(f"default n={n} f=1e{f0}..1e{f1}",f,Z)
    run(f"  init_from_ridge",f,Z,init_from_ridge=True)
    run(f"  nonneg",f,Z,nonneg=True)
f,Z=spec(60,5,-1,noise=0.005)
run("noisy 0.5% default",f,Z); run("noisy 0.5% init_from_ridge",f,Z,init_from_ridge=True); run("noisy 0.5% nonneg",f,Z,nonneg=True)
for s in (1,2,3): run(f"noisy default seed{s}",f,Z,random_seed=s)
```

### A.4 — Matrice DRT : spectres de type Randles (§4.4)

```python
import numpy as np, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from fits.physics import Z_randles_full
from vendor.bayes_drt2.inversion import Inverter
f=np.logspace(5,-1,60); w=2*np.pi*f
def spec(Rct,noise,seed):
    r=np.random.default_rng(seed)
    Z=Z_randles_full(w,Re=200.,Re_prime=20.,Cb=1e-9,Rct=Rct,Qdl=2e-6,alpha=0.9,R_D=Rct*0.3,tau_d=0.5)
    return Z.real*(1+noise*r.standard_normal(60)) + 1j*Z.imag*(1+noise*r.standard_normal(60))
def run(label,Z,**kw):
    inv=Inverter(); inv.fit(f,Z,mode='optimize',**kw)
    tau=inv.distributions['DRT']['tau']; g=inv.predict_distribution('DRT',tau=tau); Zf=inv.predict_Z(f)
    print("%-34s Rp=%9.1f gmin=%9.1f maxrelerr=%.3g"%(label,inv.predict_Rp(),g.min(),np.max(np.abs(Zf-Z)/np.abs(Z))))
for R in (3000,3500,4200,5200):
    Z=spec(R,0.005,R)
    print("Rct vrai=",R," => Rp attendu ~", int(200*0+20+R*1.3))
    run("  default",Z); run("  init_from_ridge",Z,init_from_ridge=True); run("  nonneg",Z,nonneg=True)
```

### A.5 — Maxima détectés sur une DRT propre de 2 RC (DRT-10)

```python
import numpy as np, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
from vendor.bayes_drt2.inversion import Inverter
import fits.drt_fit as d
f=np.logspace(5,-2,71); w=2*np.pi*f
Z=10+50/(1+1j*w*1e-3)+80/(1+1j*w*1e-1)
inv=Inverter(); inv.fit(f,Z,mode='optimize')
tau=np.asarray(inv.distributions['DRT']['tau']); g=np.asarray(inv.predict_distribution('DRT',tau=tau))
n=len(g); margin=max(1,n//20); core=slice(margin,n-margin); thr=g[core].max()*1e-3
mx=[i+margin for i in d._local_maxima(g[core],l=max(1,n//15),threshold=thr)]
print("n_tau",n,"tau range %.2e..%.2e"%(tau.min(),tau.max()))
print("maxima detected (tau, gamma):",[("%.2e"%tau[i],round(float(g[i]),1)) for i in mx])
print("Rct chosen:",d._extract_rct_peak(tau,g)[0:3])
```

### A.6 — Boucle CV synthétique triée par potentiel (CV-1)

```python
import numpy as np
from core.cv_loader import load_cv_file
from core.cv_peaks import detect_redox_peaks
n=200
E=np.concatenate([np.linspace(-0.2,0.6,n),np.linspace(0.6,-0.2,n)])
def peak(E,E0,w): return np.exp(-((E-E0)/w)**2)
I=np.concatenate([ 5e-6*peak(E[:n],0.25,0.06)+1e-7*E[:n], -4e-6*peak(E[n:],0.19,0.06)-1e-7*E[n:] ])
txt="Ewe/V\t<I>/mA\n"+"\n".join(f"{e:.6f}\t{i*1e3:.9f}" for e,i in zip(E,I))
sc=load_cv_file(txt.encode(),"loop",0.0,"probe")
d=np.diff(sc.I); print("points",len(sc.I),"E monotone:",bool(np.all(np.diff(sc.E)>=0)))
print("sign changes of dI/dE after load:",int(np.sum(np.sign(d[1:])!=np.sign(d[:-1]))),"(a clean loop sorted by branch would have ~4)")
print("peaks:",{k:round(v,7) for k,v in detect_redox_peaks(sc).items()})
print("truth: Ipa=5e-6@0.25, Ipc=-4e-6@0.19")
```

### A.7 — Pipeline sans réplicats ni structure persistée (ERR-1)

```python
import numpy as np, os, tempfile, logging
from core.config import load_config, config_to_dict
from core.pipeline import run_pipeline
from fits.physics import Z_randles_full
f=np.logspace(5,-1,60); w=2*np.pi*f
def make(Rct):
    Z=Z_randles_full(w,Re=200.,Re_prime=20.,Cb=1e-9,Rct=Rct,Qdl=2e-6,alpha=0.9,R_D=Rct*0.3,tau_d=0.5)
    return ("freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm\n"+"\n".join(f"{a:.6g}\t{b:.6g}\t{c:.6g}" for a,b,c in zip(f,Z.real,-Z.imag))).encode()
cfg=config_to_dict(load_config()); cfg["fit"]["error_structure"]["persistence_path"]=os.path.join(tempfile.mkdtemp(),"none.json")
fa=[dict(content=make(3000),filename="probe.txt",step="probe",concentration=0.0)]+[dict(content=make(R),filename=f"c{c}.txt",step="hybridization",concentration=c) for c,R in ((1e-9,3500),(1e-8,4200))]
sess,val=run_pipeline(fa,cfg,active_models=["randles_full"])
print("RESULT groups:",len(sess.groups),"| probe.fit_results:",list(sess.probe.fit_results),"| groups fit_results:",[list(g.fit_results) for g in sess.groups])
```


---

## Annexe B — État des audits précédents (`AUDIT.md` du 2026-07-13 remplacé ; `AUDIT_REPORT.md` du 2026-07-30 conservé)

**Ancien `AUDIT.md`** (commit `ae89a82`, récupérable par `git show ae89a82:AUDIT.md`) — fil rouge : « un résultat faux, jamais annoncé ». Corrigés et **toujours vérifiés présents** dans le code :
B1 (signe du résidu imaginaire de Randles, `randles_full.py:141-145`, garde `tests/test_measurement_model.py:274`), B2/C7 (`)` orphelin et échec de `pip` dans `launch.bat`), B3/C6 (`matplotlib` déclaré, `lmfit` retiré),
I1 (fixtures en convention loader), I2/I2b (bornes typées), I3 (CI), I4, I5/I5b (`core/calibration.py`), I6 (`discovery_errors`), I7 (`FitResult.warnings` + bandeau UI), I8 (orphelins `ui/`), V1 (matching de colonnes CV), V3 (`.gitignore`/`exports/`).
Restés ouverts à l'époque : C1 (config morte → **toujours ouvert**, §2.3), C2 (thème sombre → **ouvert**), C3 et C4 et C12 (liés à l'ancien DRT → **disparus** avec sa suppression), C5 (fonctions physiques non branchées → **ouvert**, §2.1), C8 (perf → obsolète).
Sa section « À ne pas toucher » reste valable pour la correction de signe/fréquences parasites du loader et la convention Bissessur (avec la réserve DRT-10) ; ses passages sur `pyDRTtools`, `drt_fft.py`, `drt_tikhonov.py` sont caducs.

**`AUDIT_REPORT.md`** (documentaire) — statut au HEAD `b91b8ba`, vérifié : **M1** (4 méthodes, `ARCHITECTURE.md:12`) ouvert · **M2** (Modulus, `:208`) ouvert · **M3/M4** (θ_EIS/LOD) ouverts (D-6) · **M5** (`cv_plots` recalcule la régression) ouvert (CPL-2) ·
**M6** (token non documenté) ouvert (D-14) · **M7** (arborescence incomplète) ouvert (D-8) · **m1** (`C_comparatif.py`, `ARCHITECTURE.md:119`) ouvert · **m2** (config morte) ouvert (§2.3) · **m3** (schéma du circuit) ouvert (D-7) ·
**m4** (Rct « pénultième ») **confirmé et aggravé** (DRT-10) · **m5** (`random_seed`) ouvert (DRT-9) · **m6** (`skip` du registre) ouvert · **c3** (`DRT_MODEL_NAME` défini après usage, `pipeline.py:277` vs `:199`) ouvert · **c4** (dates) ouvert (D-19).
Ses livrables de phases 2 et 3 (`ARCHITECTURE_v3.md`, `REFERENCE_MAINTENANCE_v3.html`, `SCIENTIFIC_METHODS.md`) n'existent pas dans le dépôt.

**Ce que cet audit ajoute** : DRT-1/2/3/4/5/6/7/8/10/11 (mesurés), ERR-1/2/3/4/5/6/7, FIT-1…5, CPL-1/3/4/5/6/7/9, B-STATE, B-EXP, CV-1/2, L-1…L-12 et l'analyse du bootstrap CmdStan, la revue de couverture, la matrice de dépendances internes, les bornes réelles de `streamlit`, la vérification des sha256 du vendor, et la correction du cadrage de la mission (§0).

*Fin de l'audit. Aucun fichier source, test, configuration ou document du dépôt n'a été modifié, hormis ce `AUDIT.md`.*
