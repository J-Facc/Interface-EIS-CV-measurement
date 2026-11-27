# EIS Analyzer — Architecture (v3)

> Document de référence, aligné sur le code (le code fait foi). Dépôt :
> https://github.com/J-Facc/Interface-EIS-CV-measurement — application web locale
> (Python 3.12+, Streamlit, Plotly, matplotlib) pour l'analyse de spectres d'impédance (EIS) et de
> voltammétrie cyclique (CV) de biosenseurs microfluidiques ADN/ARN sur électrodes Pt.
>
> Documents associés : [REFERENCE_MAINTENANCE.html](REFERENCE_MAINTENANCE.html) (aide-mémoire),
> [MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md), [CIRCUIT_UTILISATEUR.md](CIRCUIT_UTILISATEUR.md),
> [DRT_BAYESIENNE.md](DRT_BAYESIENNE.md), `drt/PROVENANCE.md`, `drt/VALIDATION_REGLAGES.md`.
> L'architecture v2 (pages Import / Prétraitement / EIS seule / CV seule / Export) est dans
> l'historique git (`git show 305ad88:docs/ARCHITECTURE.md`).

---

## 1. Parcours utilisateur

| # | Page (`ui/`) | Rôle | Calcule ? |
|---|---|---|---|
| 1 | **Configuration** (`page_config.py`) | biomarqueurs, canaux, concentrations ; charger une archive | non |
| 2 | **Import & nettoyage** (`page_import.py`) | dépôt, assignation manuelle, contrôles, nettoyage | nettoyage seul |
| 3 | **Visualisation** (`page_viz.py`) | par électrode : vue synthétique (moyennes ± écart-type de chaque électrode, filtre bare/probe/concentrations) ou une électrode (réplicats puis moyennes) ; données brutes, sans normalisation (réservée à l'extraction des concentrations) | moyennes d'affichage seulement |
| 4 | **Analyse** (`page_analysis.py` + `calibration_tabs.py`) | fit Randles + DRT en arrière-plan, CV, étalonnages | oui |
| 5 | **Export** (`page_export.py`) | archive .zip, rapport PDF, rapport HTML | non (assemblage) |

`app.py` : `st.set_page_config`, état DRT (CmdStan ré-enregistré, jamais installé), toggle jour/nuit
des figures (sidebar), navigation `st.navigation` vers les fonctions `render()` des pages
(`ui/nav.py` garde le registre des pages pour les liens).

---

## 2. Arborescence

```
Interface-EIS-CV-measurement/
├── app.py                    point d'entrée Streamlit (navigation, thème, état DRT)
├── launch.bat                LANCEUR UNIQUE Windows (mise à jour, venv, dépendances, DRT) — §10
├── setup_drt_bayesien.py     vérifie/installe CmdStan et compile les modèles Stan
├── requirements*.txt         dépendances épinglées (matplotlib : rapport PDF)
├── config/default.yaml       acquisition, fit (circuit, DRT, tau_window), cv, analysis
│
├── core/                     logique métier — JAMAIS d'import Streamlit
│   ├── experiment.py         Experiment, Channel, Measurement ; étapes bare/probe/conc_i ; mode déduit
│   ├── user_data.py          biomarkers.json persistant (hors du dossier de l'app)
│   ├── config.py             YAML + Pydantic ; user_data_dir()
│   ├── app_state.py          clés de session_state et invalidations (logique pure)
│   ├── assignment.py         assignation (pré-remplissage, contrôles) et nettoyage EIS + CV
│   ├── folder_import.py      import d'un dossier par canal (noms de fichiers → assignation)
│   ├── loader.py · robust_loader.py · mpr_converter.py · cv_loader.py   lecture EC-Lab
│   ├── display.py            regroupement des mesures et moyennes de réplicats pour l'affichage (ni fit ni DRT)
│   ├── background.py         pool de processus, progression, erreurs isolées, annulation
│   ├── analysis.py           AnalysisRun (fit par groupe, DRT par spectre, CV) ; DRTRecomputeRun
│   ├── pipeline.py           orchestration d'UN groupe : measurement model + KK → fit Orazem → DRT
│   ├── measurement_model.py · validator.py   structure d'erreur, verdict Kramers-Kronig
│   ├── drt_peak.py           Rct_DRT = aire du pic principal de γ(τ)
│   ├── cv.py                 pic anodique, ΔI/I
│   ├── calibration.py        étalonnages (source unique) ; courbes v2 conservées
│   ├── results.py            résultats SÉRIALISABLES (groupes, spectres, fits, DRT, CV, échecs)
│   ├── report_tables.py      tableaux uniques (UI, CSV, PDF, HTML)
│   ├── legacy_import.py · experiment_io.py   lecture des archives v2
│   ├── models.py             EISSpectrum, GroupAnalysis… (pipeline)
│   └── version.py · logger.py
│
├── fits/ · circuit/ · drt/   bibliothèques numériques (inchangées en v3) — n'importent jamais core
├── plotting/                 reçoit des données, ne calcule rien
│   ├── eis_plots.py          Nyquist/Bode (réplicats), fit, DRT (pic + zone), sélection de points
│   ├── cv_plots.py · calibration_plots.py · theme.py      Plotly
│   └── mpl_plots.py          matplotlib (rapport PDF), mêmes données
├── exports/                  archive.py · report_pdf.py · report_html.py
├── ui/                       pages + circuit_editor.py, archive_loader.py, calibration_tabs.py, nav.py
├── docs/                     ARCHITECTURE (ce fichier) · REFERENCE_MAINTENANCE.html · …
└── tests/                    pytest (dont tests/data/cv_bare_2cycles.txt : CV réelle anonymisée)
```

Créés à l'exécution et ignorés par git : `logs/`, `user_data/` (en développement), `sessions/`.

---

## 3. Règles de modularité

```
ui / plotting / exports ──► core ──► drt ──► fits.result
                             │                  ▲
                             └────► fits ───────┘
                                     └──► circuit (feuille)
```

| Couche | Règle |
|---|---|
| `core/` | Jamais d'import Streamlit. |
| `fits/`, `drt/`, `circuit/` | N'importent jamais `core`, `ui`, `plotting`, `exports` ni Streamlit. |
| `plotting/` | Reçoit des données, ne calcule rien (étalonnages : `core/calibration.py`). |
| `ui/` | Appelle `core/`, n'implémente pas de physique. La page Visualisation n'importe ni fit ni DRT (test). |
| Tous les dossiers | Un `__init__.py` dans chaque paquet. |

`tests/test_architecture.py` vérifie ces règles sur l'arbre d'imports réel.

---

## 4. Modèle de données (`core/experiment.py`, `core/results.py`)

```
Experiment  { name, biomarkers: [str], channels: [Channel], app_version, created_at }
Channel     { id, biomarker, concentrations: [float] }           # 2 électrodes implicites
            mode = "single" (1 concentration) | "multi" (plusieurs) — DÉDUIT
Measurement { channel_id, electrode (1|2), step, replicate, kind ("EIS"|"CV"),
              source_filename, data, warnings }
            step ∈ {bare, probe, conc_1, conc_2, …}   (conc_i ↔ Channel.concentrations[i-1])
            data : EIS {f, Zre, Zim} (HF→BF, Zim = −Im Z > 0) ; CV {E, I (A), cycle}
```

Résultats (`core/results.py`), produits dans les workers, seuls lus par l'UI, l'archive et les
rapports :

```
AnalysisResults { eis: EISResults, cv: {id: CVFeature}, failures: [Failure], config, … }
EISResults      { groups: {C1_E1_probe: GroupRecord}, spectra: {clé: SpectrumRecord}, circuit, drt_mode }
GroupRecord     statut (ok | fallback | failed), pondération, verdict KK, agrégats par paramètre
SpectrumRecord  réplicat (EIS_C1_E1_probe_r1) ou moyenne (…_mean) : f, Zre, Zim, KK, FitRecord, DRTRecord
DRTRecord       γ(τ), IC (sample), MainPeak (Rct_DRT), Rct moteur (contrôle), diagnostics
CVFeature       I_pic, E_pic, I_probe, ΔI/I
```

État de session (`core/app_state.py`) : `experiment` → `raw_files`, `assignment`,
`excluded_points` → `measurements` → `analysis_run` → `results`. Toute modification en amont
invalide l'aval ; un rechargement d'archive pose tout d'un coup (`results_source = "archive"`).

---

## 5. Configuration, import et nettoyage

* **Biomarqueurs persistants** : `core.config.user_data_dir()` = `<lanceur>/user_data/` quand
  l'app tourne dans `<lanceur>/eis_app/` (que launch.bat remplace à chaque mise à jour), sinon
  `<dépôt>/user_data/` ; `EIS_USER_DATA_DIR` force le dossier (tests).
* **Import par dossier de canal** (`core/folder_import.py`, uploader Streamlit en mode
  `"directory"`, sans filtre d'extension). Convention principale, noms EC-Lab
  `<libre>_<NN>_<CV|PEIS>_C<nn>.<ext>` : fin fixe (réplicat NN, technique, voie Cnn de
  l'instrument) ; la partie libre contient `w<N>` (électrode) et `bare`, `probe` ou une
  concentration (`10-18`, `1e-18`, `5x10-12`…) à position quelconque (`classify_path`). Les
  numéros de réplicat (numéros de séquence EC-Lab : 01, 06…) sont renumérotés 1, 2, 3… par
  (technique, électrode, étape). Autres techniques (CA, OCV…) et noms hors conventions : ignorés,
  avec leur raison. Conventions historiques toujours acceptées : arborescence `probe|calibration|bare/{eis,cv}/electrode_N/` ; noms
  `probe_<eis|cv>_eN_rK`, `<eis|cv>_eN_<c>M_rK`, `bare_<eis|cv>_eN_rK` (`.txt/.csv/.mpt/.mpr`, `.mpr`
  converti). La concentration du nom est rattachée à `conc_i` par sa valeur (tolérance 0,5 %) ;
  les concentrations lues peuvent être adoptées pour le canal (`reassign_steps`). Tout fichier au
  nom hors conventions (réglages `.mps`, journaux, notes…) est IGNORÉ sans être lu ni converti
  (seulement listé) ; une mesure présente en `.mpr` et en `.txt` n'est importée qu'une fois (texte
  de préférence). Signalés : incohérence dossier/nom, plus de 2 électrodes,
  concentration absente de la configuration ; électrodes 3/4 renumérotées 1/2 (avertissement).
* **Assignation** (`core/assignment.py`) : une ligne par fichier
  `fichier | type | canal | électrode | étape | réplicat | exclure`, pré-remplie d'après le contenu
  (EIS/CV) et le nom (`canal2_E1_conc_3_r2…`), toujours corrigeable. Contrôles : fichier non
  assigné, doublon, `conc_i` inexistante pour le canal, type incohérent avec le contenu (erreurs,
  bloquantes) ; électrode sans `probe`, réplicats non consécutifs (avertissements).
* **Nettoyage EIS** : `core.loader.load_spectrum` (séparateur, signe de Zim, 50/100 Hz, tri HF→BF)
  puis retrait des points exclus au lasso. **Nettoyage CV** : `core.cv_loader.load_cv_measurement`
  (colonnes auto, courant en A, cycle `cv.cycle` — dernier par défaut —, ordre chronologique).
* **Revue des spectres EIS** (page Import, §3) : un graphe à la fois (fragment Streamlit) ; la
  sélection au lasso est supprimée à la validation, qui passe au spectre suivant ; boutons
  Précédent/Suivant. Chaque fichier n'est lu qu'une fois (`load_eis_base`, cache réutilisé par le
  nettoyage final `clean_files(..., eis_cache)`).
* **Réplicats conservés** : aucun moyennage au nettoyage. `core.loader.average_replicates` reste
  un utilitaire (spectre moyen d'affichage du pipeline).

---

## 6. Analyse (`core/analysis.py`, `core/background.py`)

```
« Lancer l'analyse » → AnalysisRun (validation du circuit AVANT tout calcul)
   ├─ une tâche eis_group_task par (canal, électrode, étape)          [processus de calcul]
   │     ≥ 3 réplicats : measurement model + KK → fit Orazem pondéré (core.pipeline._analyze_group)
   │     sinon / structure indisponible : REPLI σ = 1 %·|Z| (signalé, KK « indéterminé »)
   │     → GroupRecord + SpectrumRecord (réplicats + moyenne), convertis DANS le worker
   ├─ une tâche drt_task par spectre (réplicats dès le départ, moyenne après son groupe)
   │     drt.engine.fit_drt → core.drt_peak.main_peak → DRTRecord
   └─ fin de TOUTES les tâches → CV (core.cv) → AnalysisResults publié
```

* **Pool** : `ProcessPoolExecutor`, contexte `spawn` (identique Windows/Linux), workers =
  `os.cpu_count() − 1` (min. 1, `analysis.max_workers`), ramenés à `cœurs / chaînes` en mode HMC.
  L'initialisation de chaque worker ré-enregistre CmdStan. `EIS_EXECUTOR=sync|thread` (tests).
* **Progression** : la page interroge `run.poll()` dans un fragment Streamlit (`run_every = 1 s`) ;
  barre + compteur « n/N tâches » ; rien n'est affiché avant la fin de tout.
* **Erreurs isolées** : une exception dans une tâche devient une ligne `Failure` (élément, étape,
  cause) ; les autres continuent. « Annuler » retire les tâches non démarrées.
* **Recalcul DRT** (onglet DRT) : `DRTRecomputeRun` sur les spectres choisis, en `sample` ou
  retour en `optimize` ; seul le `DRTRecord` de ces spectres est remplacé.
* **Rct par DRT** (`core/drt_peak.py`) : pics de γ(τ) (`scipy.signal.find_peaks`, ≥ 1 % du max)
  dans `fit.drt.tau_window` (défaut [1e-5 ; 1e-1] s, à calibrer : τ_ct ≈ Rct·Qdl) ; le plus haut est
  retenu (les autres sont signalés) ; intégration trapèzes de γ·d(ln τ) entre les minima locaux
  qui l'encadrent. Validé sur Rs + (Rct ∥ C) synthétique à ±5 % (test avec le moteur réel).
* **CV** (`core/cv.py`) : maximum de I sur la branche anodique (E croissant) dans
  `cv.peak_window_V` (défaut [0 ; 0,25] V, calé sur des balayages ±0,2 V), sans correction de ligne
  de base (option future : soustraire une droite ajustée sur le pied du pic) ;
  ΔI/I = (I_probe − I)/I_probe, I_probe = moyenne des pics « probe » de la même électrode.

### Étalonnages (`core/calibration.py`)

1. chaque réplicat est normalisé par la moyenne « probe » de **son** électrode :
   EIS `log10(Rct/Rct_probe)` — Rct par **DRT (pic principal)** ou par le **fit Randles** —,
   CV `ΔI/I` ;
2. par concentration : moyenne ± écart-type entre réplicats (`ddof = 1` ; 1 réplicat → pas de
   barre, signalé) ;
3. régression linéaire des moyennes sur log10([c]) (≥ 3 concentrations), R².

Vues : **par électrode** et **regroupé par biomarqueur** (une courbe par biomarqueur). Les deux
modes de canal sont traités de même : chaque point (canal, électrode, étape, réplicat) devient
(concentration, valeur normalisée), puis tout est regroupé par concentration. Courbes v2
conservées (`legacy_curves`, spectre moyen, |Rct_probe − Rct_c|/Rct_probe et log10 Rct, mode strict).

## 7. Le circuit, le fit et la DRT

| Méthode | Paramètres | Sortie (`fit_results` de chaque spectre) |
|---|---|---|
| Fit Orazem, clé `"orazem"` | circuit utilisateur (`fit.circuit`, par défaut un Randles complet à 8 paramètres) ; valeur initiale et bornes par paramètre ; paramètre cible désigné | `FitResult` : `target_param/target_value/target_std` (intra-fit), χ²ᵣ et son intervalle attendu, `fit_diagnostics` (conditionnement, rang, identifiabilité, bornes actives) |
| DRT, clé `"drt_bayes"` | réglages constants de `drt/engine.py` (justifiés par `drt/VALIDATION_REGLAGES.md`) ; mode `fit.drt.mode` | `FitResult` : γ(τ), Rct du moteur (pic pénultième, ±3 en ln τ — **colonne de contrôle** en v3), `target_std` a posteriori (HMC), `drt_diagnostics` ; puis `core/drt_peak.py` : **Rct_DRT = aire du pic principal** |

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

---

## 8. Archive .zip (`exports/archive.py`, schéma v3)

```
experience_<date>.zip
├── manifest.json        schema_version (3), app_version, expérience, table d'assignation,
│                        config (imbriquée et aplatie), fichiers + SHA-256
├── data/eis|cv/C<c>_E<e>_<étape>_r<k>.csv     données nettoyées
└── results/  analysis.json · spectra/<clé>.csv (+ Zfit) · drt/<clé>.csv (τ, γ, IC, zone du pic)
              drt_peaks.csv · fits_randles.csv · cv_features.csv · calibration_eis.csv · calibration_cv.csv
```

Nombres écrits avec `repr` (aller-retour EXACT). `read_archive` vérifie la version de schéma et
chaque SHA-256 puis reconstruit expérience, mesures et résultats **sans aucun recalcul** (test :
fonctions de fit et de DRT interdites pendant le rechargement) ; les étalonnages sont recalculés
à l'affichage à partir des enregistrements (arithmétique pure, identique). Archive invalide →
`ArchiveError` (message clair). Les archives **v2** (`experiment.yaml`) sont lues par
`core/legacy_import.py` : électrode v2 *e* → canal ⌈e/2⌉, électrode 1/2 ; exclusions et points
supprimés repris ; à nettoyer puis analyser.

## 9. Rapports

* **PDF** (`exports/report_pdf.py`, `matplotlib.backends.backend_pdf.PdfPages`, backend Agg, A4) :
  `pdf_plan()` liste les pages — garde, CV par canal, Nyquist+Bode par canal, DRT par canal,
  tableau des Rct (paysage, 28 lignes/page), étalonnage EIS par biomarqueur (sources DRT et fit),
  étalonnage CV, échecs. Figures : `plotting/mpl_plots.py`.
* **HTML** (`exports/report_html.py`) : fichier autonome, plotly.js embarqué une seule fois,
  sommaire, mêmes figures Plotly que l'application et mêmes tableaux.
* La page Export génère chaque fichier **à la demande** puis propose le téléchargement. L'export
  PNG n'existe pas (kaleido inutile).

## 10. Lanceur Windows (`launch.bat`)

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
`launcher.log`, et `user_data\` (créé par l'application : `biomarkers.json`) — hors de `eis_app\`,
donc jamais touché par une mise à jour. `core\app_state.py` fait partie des fichiers que le lanceur
exige dans une mise à jour : il ne doit pas être supprimé (garde-fou : `tests/test_app_state.py`). Aucune écriture dans le registre (les chemins longs ne sont pas nécessaires :
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

---

## 11. Tests et CI

```bash
python -m pytest tests/ -v                 # tout
python -m pytest tests/ -m "not slow"      # sans le HMC lent
python -m compileall app.py setup_drt_bayesien.py core fits drt circuit plotting exports ui tests
```

Tests v3 : `test_experiment_model` (modèle, persistance des biomarqueurs, mise à jour simulée),
`test_assignment` (contrôles de cohérence, nettoyage, CV réelle à 2 cycles), `test_drt_peak`
(aire du pic, fenêtre, moteur réel ±5 %), `test_analysis_background` (processus, échec isolé, rien
avant la fin, recalcul DRT), `test_calibration_v3` (3 cas de concentrations, ddof = 1, 1 réplicat,
CV ΔI/I), `test_archive_v3` (aller-retour exact sans fit, archives corrompues), `test_reports_v3`
(pages du PDF, figures du HTML, archives v2), `test_ui_pages` (pages en AppTest), `test_app_state`.
`validate.yml` : job `validate` (Linux), job `drt` (CmdStan réel, tests lents inclus),
`validate-windows`.

## 12. Erreurs fréquentes

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
| Groupe en « repli » (🟠) | moins de 3 réplicats, ou structure d'erreur non caractérisable : fit pondéré σ = 1 %·|Z|, χ²ᵣ non interprétable | fournir au moins 3 réplicats indépendants par groupe |
| « Aucun pic de γ(τ) dans la fenêtre » | `fit.drt.tau_window` ne couvre pas l'arc de transfert de charge | élargir/décaler la fenêtre dans `config/default.yaml` (τ_ct ≈ Rct·Qdl) |
| « aucun point de la branche anodique dans la fenêtre » (CV) | `cv.peak_window_V` hors de la plage balayée | ajuster `cv.peak_window_V` |
| Archive refusée « Somme de contrôle invalide » | fichier modifié ou corrompu après export | ré-exporter l'archive depuis l'application |
| Fenêtre `.bat` qui se ferme / variable vide | `%VAR%` lu dans un bloc `( … )` | `!VAR!` + `enabledelayedexpansion` (règle de `launch.bat`) |
