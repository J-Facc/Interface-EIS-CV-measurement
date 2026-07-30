# AUDIT_REPORT.md — EIS Analyzer (Interface-EIS-CV-measurement)

> Audit **en lecture seule** du code réel (le code fait foi), confronté aux deux
> documents de référence présents dans le dépôt : `ARCHITECTURE.md` (en-tête
> « MàJ 13/07/2026 — réalignée sur l'état réel ») et `Reference_maintenance`
> (HTML, style v2).
> Aucun fichier source n'a été modifié. Les correctifs proposés sont listés, **pas
> appliqués**.
> Date : 2026-07-30 · Branche : `claude/eis-analyzer-audit-jupdad`
> Périmètre : `core/`, `fits/`, `plotting/`, `exports/`, `ui/`, `pages/`, `config/`,
> `launch.bat`, `.github/`, `app.py`, `setup_drt_bayesien.py`.

---

## 0. Synthèse

Le **code** est mûr et robuste : un audit antérieur (`AUDIT.md`, 2026-07-13) a
éliminé les échecs silencieux (fits non convergés muets, `exports/` ignoré par
git, pondération arbitraire…). Les règles de modularité sont **presque** toutes
respectées : **aucun** `import streamlit` dans `core/` ni `fits/`, tous les
`__init__.py` présents, `ui/` ne fait pas de physique. La physique du modèle de
Randles (`fits/physics.py`) est cohérente dimensionnellement et la convention de
signe de `Zim` est traitée proprement de bout en bout.

Le problème dominant n'est **pas** dans le code mais dans la **désynchronisation
documentation ↔ code**, à deux vitesses :

- **`Reference_maintenance` (HTML) est intégralement obsolète.** Il décrit une
  architecture à **4 méthodes** (`randles_classique.py` en Differential Evolution,
  un Randles contraint, `circulaire_fit.py` en Kasa, une DRT NNLS/ridge) et une
  **pondération « Modulus »/`alpha_noise`**. **Aucun** de ces fichiers ni de ces
  méthodes n'existe dans le code réel, qui n'a plus que **2 modèles** :
  `randles_full` (`scipy.optimize.least_squares`, pondération **structure d'erreur
  d'Orazem**) et `drt_bayes` (wrapper `vendor/bayes_drt2/Inverter`).
- **`ARCHITECTURE.md` est partiellement réaligné** (il annonce bien 2 modèles au §5)
  mais conserve des affirmations mortes : « 4 méthodes » en intro, « pondération
  Modulus » dans le tableau, un `LOD ≈ 10⁻¹⁷ M` et une équation de calibration
  `θ_EIS` qui **ne correspondent pas** au signal réellement calculé.

Point notable pour le lecteur de ce rapport : **l'énoncé de la mission d'audit
lui-même** (« 4 méthodes : Randles DE, Randles contraint, DRT NNLS+L-curve,
circulaire Kasa », « fits stochastiques Differential Evolution ») décrit
**l'ancienne architecture**, celle du HTML obsolète — pas le code. Le code
n'a **ni Differential Evolution, ni Kasa, ni NNLS/L-curve, ni Randles contraint**.
`METHODES.md` (2026, à jour) et `MEASUREMENT_MODEL.md` sont, eux, fidèles au code
et le disent explicitement (« documentation d'architecture réputée obsolète »).

Une seule violation de modularité côté code : `plotting/cv_plots.py` recalcule
lui-même les régressions de calibration CV (`stats.linregress`) au lieu de passer
par `core/calibration.py`, alors que `plotting/eis_plots.py` et
`exports/exporter.py` délèguent correctement à la source unique.

**Bilan par sévérité :** 1 BLOQUANT (documentaire), 7 MAJEUR, 6 MINEUR,
4 COSMÉTIQUE. **Aucun défaut de correction scientifique bloquant** dans le code
lui-même. Les correctifs sont surtout des mises à jour documentaires (livrables des
Phases 2/3) + un refactor de `cv_plots.py` + un nettoyage de config morte.

---

## 1. PHASE 0 — Arborescence réelle vs documentée

### 1.1 Arborescence réelle observée (hors `.git`, `vendor/bayes_drt2/` replié)

```
Interface-EIS-CV-measurement/
├── app.py                       ← entrée Streamlit (st.navigation)
├── setup_drt_bayesien.py/.bat   ← préparation CmdStan + Series.stan
├── launch.bat                   ← lanceur Windows (auto-update, repo PRIVÉ + token.txt)
├── launch_app.bat / launch_app.sh
├── requirements.txt · requirements-drt.txt · requirements-optional.txt
├── pytest.ini
├── ARCHITECTURE.md · Reference_maintenance(.html) · project(brief) ← docs de réf.
├── AUDIT.md · METHODES.md · MEASUREMENT_MODEL.md · README.md · THIRD_PARTY_LICENSES.md
├── config/  default.yaml
├── core/    __init__ · models · loader · robust_loader · pipeline · config ·
│            calibration · logger · validator · cv_loader · cv_models · cv_peaks ·
│            cv_pipeline · mpr_converter · experiment_io
├── fits/    __init__ · base · physics · registry · weighting · error_structure ·
│            randles_full · drt_fit · kk_validation
├── plotting/ __init__ · theme · eis_plots · cv_plots · kk_plots
├── exports/ __init__ · exporter
├── ui/      __init__ · tabs
├── pages/   __init__ · 0_import · 1_pretraitement · A_eis · B_cv · D_inference · E_export
├── tests/   (14 fichiers pytest) + conftest
├── vendor/  bayes_drt2/ (paquet DRT vendoré, + stan_model_files/*.stan)
└── .github/workflows/validate.yml   ← CI : compileall + pytest (Python 3.12)
```

### 1.2 Écarts vs `ARCHITECTURE.md` §2

| Écart | Détail |
|-------|--------|
| **Fichiers `fits/` absents du doc** | `ARCHITECTURE.md:52-61` liste `base, physics, registry, randles_full, drt_fit, kk_validation` mais **omet `fits/weighting.py` et `fits/error_structure.py`** — le cœur même de la pondération d'Orazem qu'il décrit par ailleurs. |
| **Fichier `core/` absent du doc** | `ARCHITECTURE.md:39-50` **omet `core/robust_loader.py`**, qui est pourtant le parseur EC-Lab réel appelé en priorité par `loader.py`. |
| **Docs non cartographiés** | `AUDIT.md`, `METHODES.md`, `MEASUREMENT_MODEL.md`, `project` (brief), `setup_drt_bayesien.*` ne figurent pas dans l'arbre §2 (mineur : ce sont des annexes). |

### 1.3 Écarts vs `Reference_maintenance` (HTML) §arborescence

| Écart | Détail |
|-------|--------|
| **Fichiers cités INEXISTANTS** | `Reference_maintenance:495` cite `fits/randles_classique.py` (« Differential Evolution »), `:498` `fits/circulaire_fit.py` (« Kasa »), `:588-591` un tableau de 4 modèles. **Aucun de ces fichiers n'existe** (`ls` → *No such file*). |
| **Modules réels absents** | Le HTML ignore `weighting.py`, `error_structure.py`, `drt_fit.py` (bayes_drt2), `robust_loader.py`. |

---

## 2. Findings par sévérité

> Convention : `[DOC]` = écart documentaire, `[SCI]` = exactitude scientifique,
> `[MODULARITÉ]`, `[CONFIG]`, `[CODE]`, `[REPRO]` = reproductibilité,
> `[LAUNCHER]`. Chaque finding cite `fichier:ligne`.

### 🟥 BLOQUANT

#### B1 — [DOC] `Reference_maintenance` (HTML) décrit une architecture qui n'existe plus
- **Où :** `Reference_maintenance:449` (« compare 4 méthodes »), `:453`
  (θ_EIS/LOD), `:493-498` (arbre `fits/` avec `randles_classique.py`,
  `circulaire_fit.py`), `:588-591` (tableau 4 modèles : DE, Kasa), `:706-711`
  (`fit.alpha_noise`, « Pondération Modulus »).
- **Description :** ce document est **la** référence maintenance du projet, et il
  est faux sur le cœur scientifique. Il annonce 4 méthodes d'extraction de Rct
  (Randles DE, Randles contraint, circulaire Kasa, DRT ridge) et une pondération
  Modulus paramétrée par `alpha_noise`. Le code réel n'a **que 2 modèles**
  (`fits/registry.py` ne découvre que `randles_full` et `drt_bayes`), la
  pondération est **exclusivement** la structure d'erreur d'Orazem
  (`fits/weighting.py:1-25`, `fits/error_structure.py`), et la clé `alpha_noise`
  n'existe plus (supprimée, cf. `fits/weighting.py:4-5`). Un mainteneur qui suit ce
  doc cherchera des fichiers absents et un mode de pondération supprimé.
- **Correctif proposé :** remplacer intégralement par `REFERENCE_MAINTENANCE_v3.html`
  (Phase 2) reflétant les 2 modèles réels et la pondération d'Orazem ; conserver
  `Reference_maintenance` comme archive v2 ou le supprimer.

*(Aucun défaut de niveau BLOQUANT n'a été trouvé dans le code exécutable :
l'application se lance, les fits produisent des résultats corrects et signalent
leurs échecs.)*

---

### 🟧 MAJEUR

#### M1 — [DOC] `ARCHITECTURE.md` : « 4 méthodes » en intro contredit son propre §5 (2 modèles)
- **Où :** `ARCHITECTURE.md:12` (« extrait Rct par **4 méthodes** comparatives »)
  vs `ARCHITECTURE.md:206-209` (tableau **2 modèles** : `randles_full`, `drt_bayes`).
- **Description :** incohérence interne. Le code confirme 2 modèles (`fits/registry.py`,
  `fits/randles_full.py:33`, `fits/drt_fit.py:330`).
- **Correctif :** remplacer « 4 méthodes » par « 2 méthodes comparatives (Randles
  complet + DRT bayésienne) » dans l'intro et le §1.

#### M2 — [DOC] `ARCHITECTURE.md` annonce « pondération Modulus » alors que le code n'a QUE Orazem
- **Où :** `ARCHITECTURE.md:208` (« `least_squares` (**pondération Modulus**) »)
  et `Reference_maintenance:711` (`fit.alpha_noise` / « Pondération Modulus »).
- **Réalité code :** `fits/randles_full.py:3-6` et `:116-131` → pondération UNIQUE
  = structure d'erreur d'Orazem, `w = 1/σ²`, `absolute_sigma=True` ; `fits/weighting.py:3-5`
  déclare explicitement les modes « modulus » et « sigma » **supprimés**.
- **Correctif :** remplacer « pondération Modulus » par « pondération = structure
  d'erreur d'Orazem (measurement model), `w = 1/σ²` » dans le §5 et les tableaux de
  config.

#### M3 — [SCI/DOC] Le signal de calibration documenté (θ_EIS, bare) ≠ signal réellement calculé (probe)
- **Où doc :** `ARCHITECTURE.md:16-17` et `Reference_maintenance:453` :
  `θ_EIS = 1 − Rct,bare / Rct,ap` ; `log(Rct_norm) = a·log([c]) + b`.
- **Où code :** `core/calibration.py:79` calcule
  `signal = |Rct_probe − Rct_c| / |Rct_probe|` (référence **probe**, **pas bare**,
  **pas** de log sur le signal), régressé sur `log10([c])` (`:80-81`) ;
  `compute_calibration_loglog:106-107` calcule `log10(Rct)` **brut** (pas
  `Rct_norm`). La fonction `theta_EIS()` (`fits/physics.py:100-110`) — la formule
  affichée dans le doc — n'est appelée **nulle part** hors `tests/test_physics.py`.
  Le champ `EISSession.bare_reference` est même explicitement « AFFICHAGE SEUL,
  JAMAIS utilisé dans les calculs » (`core/models.py:158-163`).
- **Impact :** un lecteur croit que la grandeur de détection est le taux de
  couverture θ_EIS référencé à l'électrode nue ; le code produit un ΔRct normalisé
  référencé au probe. Interprétation scientifique faussée.
- **Correctif :** aligner la doc sur le signal réel (`|ΔRct|/Rct_probe` et
  `log10(Rct)`), **ou** — si θ_EIS est la cible voulue — ouvrir un ticket pour
  brancher `theta_EIS()` dans `compute_calibration`. À décider (choix produit).

#### M4 — [SCI/DOC] `LOD ≈ 10⁻¹⁷ M` annoncé mais aucun calcul de LOD dans le code
- **Où :** `ARCHITECTURE.md:17`, `Reference_maintenance:453`.
- **Réalité :** aucune occurrence de LOD / limite de détection dans le code
  (`grep -ri "lod|limite de d" --include=*.py` → néant). La calibration s'arrête à
  (pente, ordonnée, R², p-value, stderr) dans `core/calibration.py`.
- **Correctif :** soit retirer/annoter le chiffre comme valeur cible externe (non
  calculée par l'app), soit implémenter un `compute_lod()` (p. ex. 3σ_blanc/pente)
  dans `core/calibration.py`. Documenter clairement l'origine du 10⁻¹⁷ M
  (littérature capteur ?), sinon `[NON SOURCÉ]`.

#### M5 — [MODULARITÉ] `plotting/cv_plots.py` recalcule la calibration au lieu de déléguer à `core/`
- **Où :** `plotting/cv_plots.py:157, 224, 412, 450` → `stats.linregress(log_c, sig)`
  recalculé localement, alors que `core/calibration.py:133 compute_cv_calibration`
  existe et est utilisée par `exports/exporter.py:179,230`.
- **Impact :** viole la règle « `plotting/` reçoit des données, ne calcule rien »
  (`ARCHITECTURE.md:63,163`) et crée un **risque de divergence** figure ↔ export CSV
  (deux implémentations de la même régression). `eis_plots.py:654,839,1068` délègue
  correctement, `cv_plots.py` fait exception.
- **Correctif :** faire consommer à `cv_calibration_figure` /
  `cv_calibration_multi_figure` le résultat de `compute_cv_calibration` (pente,
  ordonnée, r²) au lieu de refaire le `linregress`.

#### M6 — [LAUNCHER/DOC] `launch.bat` exige `token.txt` (repo privé), non documenté
- **Où :** `launch.bat:24-39` (abandon si `token.txt` absent/vide), `:65-67`
  (`zipball` authentifié `Authorization: Bearer <token>`), `BRANCH=main` (`:7`).
- **Réalité doc :** `ARCHITECTURE.md:308` ne mentionne que « Prérequis : Python
  installé » — rien sur `token.txt` ni sur le caractère privé du dépôt. Le §7
  décrit un flux public (`.version` vs SHA) sans le token obligatoire.
- **Impact :** un utilisateur qui suit la doc **ne peut pas lancer** (arrêt
  immédiat `ERREUR : token.txt introuvable`).
- **Correctif :** documenter dans le §7 la présence obligatoire de `token.txt` à
  côté du `.bat`, sa provenance (PAT GitHub à portée `repo`), et le fait que le
  dépôt est privé.

#### M7 — [DOC arbo] `ARCHITECTURE.md` §2 omet des modules centraux réels
- **Où :** `ARCHITECTURE.md:52-61` (bloc `fits/`) sans `weighting.py` ni
  `error_structure.py` ; `:39-50` (bloc `core/`) sans `robust_loader.py`.
- **Impact :** l'arborescence de référence n'inclut pas le parseur EC-Lab réel ni
  les deux fichiers qui portent la pondération d'Orazem — pourtant décrite ailleurs
  dans le même document.
- **Correctif :** ajouter les trois fichiers à l'arbre §2 (fait dans
  `ARCHITECTURE_v3.md`).

---

### 🟨 MINEUR

#### m1 — [DOC] `ARCHITECTURE.md` §2b « Flux obligatoire » cite encore `C_comparatif.py`
- **Où :** `ARCHITECTURE.md:119` liste `pages/A_eis.py, B_cv.py, C_comparatif.py,
  D_inference.py`, alors que `:104-106` (même section) affirme qu'il n'existe pas de
  page Comparaison. `pages/C_comparatif.py` n'existe pas (`ls` → absent).
- **Correctif :** retirer `C_comparatif.py` de la ligne 119.

#### m2 — [CONFIG] Clés de config mortes (vestiges du DRT ridge/Wiener supprimé)
- **Où :** `config/default.yaml:60` `drt_wiener_W`, `:61` `drt_n_z`,
  `core/config.py:104` `n_monte_carlo`, `:107` `drt_wiener_W`, `:108` `drt_n_z`.
- **Réalité :** aucune de ces clés n'est lue par une logique de calcul (seulement
  par `tests/test_fits.py:69-70`, qui les injecte sans les utiliser). La DRT est
  désormais bayes_drt2 (`fits/drt_fit.py`), sans filtre de Wiener ni grille `n_z`.
  ⚠️ Ne PAS toucher `drt_kk_tol` (`default.yaml:62`) : **elle est utilisée** par
  `fits/kk_validation.py:105`.
- **Correctif :** supprimer `drt_wiener_W`, `drt_n_z`, `n_monte_carlo` du YAML et de
  Pydantic (`FitSettings`), et de `tests/test_fits.py`.

#### m3 — [SCI/DOC] Schéma ASCII du circuit trompeur vs implémentation réelle
- **Où doc :** `ARCHITECTURE.md:219` → `Re — [ R'e // Cb ] — [ Rct // CPE(Qdl, α) ] — ZD(ω)`.
- **Où code :** `fits/physics.py:41-63` implémente en réalité
  `Z_eq = R'e + (Rct + Z_D)/[1 + Qdl·(jω)^α·(Rct + Z_D)]` puis
  `Z = Re + Z_eq/[1 + jω·Cb·Z_eq]`, c.-à-d. **Cb en parallèle de tout `{R'e +
  (Rct+Z_D)//CPE}`** et **Z_D placé À L'INTÉRIEUR de la branche de transfert**
  `(Rct+Z_D)//CPE`.
- **Analyse :** le **code est physiquement correct** (Warburg en série avec Rct,
  l'ensemble sous la capacité de double couche — Randles canonique). C'est le
  **schéma du doc** qui induit en erreur : il place Z_D en élément terminal série et
  Cb autour de `R'e` seul. `[À VÉRIFIER]` uniquement côté doc, pas côté code.
- **Correctif :** redessiner le schéma pour refléter `physics.py` (Z_D dans la
  branche Rct//CPE ; Cb en parallèle de la branche complète).

#### m4 — [SCI] Extraction Rct DRT par « pic pénultième » : heuristique à domaine de validité étroit
- **Où :** `fits/drt_fit.py:151-152` (`peak_idx = maxima[-2]`).
- **Description :** on suppose que l'avant-dernier maximum de γ(τ) est **toujours**
  l'arc de transfert de charge (le dernier = diffusion BF). Si un 3ᵉ pic apparaît
  (adsorption, artefact de grille, seconde constante de temps), la sélection peut
  viser le mauvais arc. Le risque est **atténué** (fenêtre « cœur » `:140-148`,
  warnings `:157-164`, repli Rp signalé `:261-268`), mais l'hypothèse reste forte.
  `[À VÉRIFIER]` scientifiquement (à documenter dans SCIENTIFIC_METHODS.md).
- **Correctif :** documenter explicitement l'hypothèse « ≤ 2 constantes de temps » ;
  envisager une sélection par position en τ (borne physique de τ_ct) plutôt que par
  rang.

#### m5 — [REPRO] `drt_fit.py` ne fixe pas explicitement `random_seed` pour `Inverter.fit`
- **Où :** `fits/drt_fit.py:231` `inv.fit(freq_sorted, Z_sorted, mode=mode)` — pas
  de `random_seed` transmis. Le vendor a un défaut `random_seed=1234`
  (`vendor/bayes_drt2/inversion.py:1329`).
- **Description :** la reproductibilité du mode `'sample'` (HMC) et du MAP dépend
  donc d'un **défaut implicite du paquet vendoré**. Fonctionne aujourd'hui, mais une
  mise à jour du vendor changeant ce défaut casserait silencieusement la
  reproductibilité. (Le point « seeds/DE » de l'énoncé d'audit est par ailleurs
  **sans objet** : il n'y a **aucun** Differential Evolution ; `randles_full` utilise
  `least_squares` TRF déterministe depuis un guess fixe, `randles_full.py:150-154`.)
- **Correctif :** passer explicitement `random_seed` (lu depuis la config) à
  `inv.fit`, et l'exposer dans `config/default.yaml`.

#### m6 — [CODE] `fits/registry.py` n'exclut pas `error_structure` de la découverte
- **Où :** `fits/registry.py:29` `skip = {"base", "physics", "registry",
  "kk_validation", "weighting"}` — sans `error_structure`.
- **Description :** `fits/error_structure.py` est donc importé pendant la découverte
  des modèles. Inoffensif (il n'expose aucune sous-classe `BaseFitModel`), mais
  incohérent avec l'intention du `skip` (ne parcourir que les vrais plugins) et
  coûte un import inutile.
- **Correctif :** ajouter `"error_structure"` au set `skip`.

---

### ⬜ COSMÉTIQUE

#### c1 — [DOC] `Fv` noté différemment selon les docs (même valeur)
- `ARCHITECTURE.md:318` « 0.5×10⁻⁹ » vs `Reference_maintenance:706` / `config/default.yaml:20`
  « 5×10⁻¹⁰ ». Numériquement identiques (5·10⁻¹⁰ m³/s = 0,5 µL/s). Uniformiser la
  notation.

#### c2 — [DOC] Terminologie flottante « log(Rct) » vs « log(Rct_norm) »
- `ARCHITECTURE.md:12` « log(Rct) vs log([c]) » vs `:17` « log(Rct_norm) ».
  Choisir une seule formulation, cohérente avec `core/calibration.py` (qui trace
  `log10(Rct)` brut en variante DRT).

#### c3 — [CODE] `DRT_MODEL_NAME` défini après son usage lexical
- `core/pipeline.py:277` définit `DRT_MODEL_NAME` alors qu'il est référencé
  `:199` dans la fonction imbriquée `_fit_replicates`. Résolu au runtime (global),
  **sans bug**, mais peu lisible. Remonter la constante en tête de module.

#### c4 — [DOC] Dates incohérentes dans `ARCHITECTURE.md`
- En-tête `:6` « MàJ 13/07/2026 » vs pied de page `:361` « rédigé le 29/05/2026 ».
  Harmoniser (et dater la v3 au 30/07/2026).

---

## 3. Points vérifiés CONFORMES (contrôles passés)

| Contrôle | Résultat | Preuve |
|----------|----------|--------|
| `import streamlit` dans `core/` ou `fits/` | **Aucun** (règle respectée) | `grep -rn streamlit core/ fits/` → vide |
| `__init__.py` présents | `core, fits, plotting, exports, ui, pages, tests, vendor` : tous OK | `ls */__init__.py` |
| `ui/` n'implémente pas de physique | OK — délègue à `core.pipeline.recompute_drt`, `plotting`, `exports` | `ui/tabs.py:22`, pas de `linregress`/`least_squares`/`Inverter` |
| `exports/` délègue les calculs | OK — utilise `core/calibration.py` | `exports/exporter.py:14,156,179,230` |
| `eis_plots.py` délègue la calibration | OK — source unique | `plotting/eis_plots.py:8,654,839,1068` |
| Convention de signe `Zim = −Im(Z) > 0` | Cohérente loader→fit→DRT→KK | `loader.py:188`, `randles_full.py:141-144,183-184`, `drt_fit.py:222-223,277`, `kk_validation.py:103` |
| Cohérence dimensionnelle `Z_D`, CPE, Brug, θ | OK (Ω, S·sᵅ, sans dim.) | `physics.py:10-110` |
| Limite BF de `Z_D` (ω→0 → R_D) | Gérée (`tanh(x)/x → 1`) | `physics.py:21-25` |
| Robustesse loader (fichier vide / < 5 pts / CV en entrée) | Levée d'erreur explicite | `loader.py:274-278, 289-293, 324-328` |
| Concentration nulle/négative en calibration | Ignorée (log10 sur `> 0` seulement) | `calibration.py:43-44, 47, 143` |
| Réplicats manquants (N=1) | `average_replicates` renvoie le spectre unique | `loader.py:358-359` |
| Structure d'erreur : refus explicite si non caractérisable | `ErrorStructureUnavailable` (pas de σ arbitraire) | `error_structure.py:63-65, 446-451` |
| Découverte des modèles + erreurs non masquées | `discovery_errors()` exposé | `registry.py:36-40, 57-64` |
| DRT désactivée proprement sans l'extra | `bayes_available()` → False, import protégé | `drt_fit.py:60-66, 72-74` |
| CI (compileall + pytest, Python 3.12) | OK ; tests DRT lourds sautés sans CmdStan | `.github/workflows/validate.yml`, `tests/test_drt.py:15-19` |
| `launch.bat` : `setlocal enabledelayedexpansion`, chemins longs, SSL proxy | Présents | `launch.bat:2, 20, 44/67 (cert callback)` |
| `render_tabs` supprimé, pas de `C_comparatif.py` | Confirmé | `grep render_tabs` → vide ; `ls pages/C_comparatif.py` → absent |
| `experiment_io` : `save_experiment`/`zip_experiment`/`load_experiment` | Conformes au doc | `experiment_io.py:37, 190, 197` |

*Note `Fv` : la valeur `config/default.yaml:20` (5·10⁻¹⁰ m³/s) est **conforme** aux
deux docs ; seule la notation diffère (cf. c1). Pas de divergence numérique.*

---

## 4. Ce que je recommande pour la suite (Phases 2 & 3)

1. **`ARCHITECTURE_v3.md`** — corriger M1, M2, M3, M4, M7, m1, m3, c1, c2, c4 ;
   arborescence complète (avec `weighting.py`, `error_structure.py`,
   `robust_loader.py`) ; changelog listant ces écarts.
2. **`REFERENCE_MAINTENANCE_v3.html`** — réécriture complète (B1), même charte
   graphique que `Reference_maintenance` (fonts Syne/JetBrains Mono/Lora, palette
   `--ink/--paper/--accent`), 2 modèles réels, pondération Orazem, `token.txt` (M6).
3. **`SCIENTIFIC_METHODS.md`** (Phase 3) — documenter/justifier le circuit de Randles
   réel (m3), les **2** méthodes d'extraction (Randles `least_squares` + DRT
   bayes_drt2, **pas** 4), le prétraitement (signe, 50/100 Hz, moyennage), la
   calibration **réellement implémentée** (M3), et traiter le LOD (M4) en
   `[NON SOURCÉ]` tant que non calculé/sourcé.
4. **Correctifs code (hors périmètre Phases 2/3, à valider séparément) :** M5
   (`cv_plots` → `compute_cv_calibration`), m2 (config morte), m5 (`random_seed`
   explicite), m6 (`skip` registry), c3 (constante).

---

*Fin du rapport d'audit — Phase 1. En lecture seule : aucun fichier source modifié.
En attente de validation avant Phases 2 (architecture v3) et 3 (méthodes
scientifiques sourcées).*
