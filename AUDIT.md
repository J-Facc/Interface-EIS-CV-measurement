# AUDIT — EIS Analyzer (Interface-EIS-CV-measurement)

> Audit de fiabilisation **avant** modifications fonctionnelles.
> Phases 1 & 2 = lecture seule. Ce document est le livrable de la Phase 3.
> Les correctifs sont appliqués **un par un après validation** ; voir le
> **Journal des correctifs** (§2bis) pour l'état à jour et les commits.
> Date : 2026-07-13 · Branche : `claude/eis-analyzer-audit-a2ml22`

---

## 1. Résumé exécutif (10 lignes)

Le projet est globalement bien structuré : le cloisonnement des couches est **respecté**
(aucun import Streamlit dans `core/`, `fits/`, `plotting/`, `exports/`), chaque paquet a son
`__init__.py`, et la suite de tests est plus fournie que ne le dit la doc (7 fichiers de test,
pas seulement `test_fits.py`). **Le principal défaut identifié — et corrigé (B1) —** était le
résidu imaginaire du fit Randles au **mauvais signe** vs la convention réelle du loader (Zim
positif) : biais systématique ~25-30 % sur Rct, avec **effondrement sur la borne basse (100 Ω)
aux Rct élevés**, et calibration dégradée (pente 0.126 vs 0.150 vraie, R² 0.80). Après correction :
Rct à ±1 %, χ² divisé par 10³, pente 0.152 / R² 1.00 (mesuré, cf. §2bis). Les tests ne
l'attrapaient pas car leurs fixtures utilisaient la convention de signe *inverse* du loader.
S'y ajoutaient : un `)` parasite dans `launch.bat`, `matplotlib` absent de `requirements.txt`,
une CI qui ne lançait jamais les tests, une désynchro config↔code (`bounds_randles_full`),
un piège de parsing YAML (I2b), un trou de validation dans `cv_loader`, de la config morte,
de la duplication (calibration réimplémentée ≥4 fois), et une forte dérive documentaire.
**État de santé : les bloquants/importants prioritaires sont traités (voir §2bis) ; restent
surtout des items de propreté (duplication, code mort, doc).**

---

## 2bis. Journal des correctifs (branche `claude/eis-analyzer-audit-a2ml22`)

| Commit | Items | Résumé |
|--------|-------|--------|
| `fix(randles)` | **B1**, I1 | Signe du résidu imaginaire corrigé (`-Z.imag - spectrum.Zim`) ; fixtures `test_fits.py` passées en convention loader (Zim>0) ; test bout-en-bout loader→fit (±5 %, 4 Rct). |
| `fix(launcher/deps)` | **B2**, C7, **B3**, C6 | `)` orphelin supprimé ; `pip install` échoue proprement ; `matplotlib` ajouté ; `lmfit` (inutilisé) retiré. |
| `ci` | **I3** | Renommé `validate.yml` ; Python 3.12 ; compile tous les paquets ; lance `pytest` ; tourne aussi sur PR. |
| `fix(ci)` | **V1** cv_loader, dép. galvani | Corruption silencieuse de colonnes CV fermée (voir V1) ; `galvani` sorti des dépendances dures (build cassé en CI). |
| `fix(mpr)` | galvani (compromis) | `galvani` **déplacé** en dépendance optionnelle (`requirements-optional.txt`) — pas supprimé : c'est un **compromis assumé** (CI verte vs robustesse `.mpr`, voir note ci-dessous). Message actionnable si eclabfiles échoue et galvani absent. |
| `fix(config)` | **I2**, **I2b** | Clés `bounds_randles_full` réalignées `R_D`/`tau_d` ; bornes typées `list[float]` ; `1.0e9→1.0e+9` ; tests `test_config.py`. |
| `fix(pipeline)` | **I4** | `alpha_noise` lu par clé ; suppression des `getattr` KK fantômes. |
| `refactor(calibration)` | **I5** | `core/calibration.py` source unique ; `calibration_figure`, `calibration_drt_figure`, fenêtre matplotlib et `export_calibration_csv` branchés dessus ; `plotting/` ne calcule plus la calibration EIS ; test figure==export==core. Reste I5b (calibration CV). |

**Mesure B1 avant/après** (jeu synthétique à vérité-terrain, chemin réel loader→pipeline) :

| | pente | ordonnée | R² | χ² typique |
|---|--:|--:|--:|--:|
| vérité-terrain | 0.150 | 2.250 | 1.000 | — |
| avant (bug) | 0.126 | 2.018 | 0.800 | 6e5–4e6 |
| après (corrigé) | 0.152 | 2.284 | 1.000 | 1e3–5e3 |

Collatéral vérifié : `drt_fft` consommait des paramètres Randles effondrés (2083/100/100/100 →
2968/3865/4783/5690/6589 après) ; `drt_tikhonov` **n'a pas** ce bug (flip explicite `:131`,
reconstruction 0.08 % en convention loader).

**Compromis galvani (assumé).** `core/mpr_converter._read_mpr_dataframe` bascule sur `galvani`
**dans un `except` sur échec d'`eclabfiles`** — c'est donc un vrai lecteur de secours, pas un
choix de format. Le retirer des dépendances dures **n'est pas sans perte** : un `.mpr` qu'`eclabfiles`
ne sait pas lire ne sera plus récupéré automatiquement. Choix retenu : **CI verte > robustesse `.mpr`
de secours**, car galvani ne peut pas s'installer sur le runtime CI (setup.py `install_layout`). galvani
est donc **déplacé en optionnel** (`requirements-optional.txt`, `pip install -r requirements-optional.txt`)
et non supprimé ; l'utilisateur qui en a besoin l'installe et le fallback revit. Message d'erreur rendu
actionnable si `eclabfiles` échoue et galvani absent (au lieu d'une `ImportError` brute).

Reste non traité (propreté) : I5b (calibration CV), I6, I7, I8, C1–C12.

---

## 2. Tableau des problèmes

> Statut : ✅ résolu (§2bis) · ⬜ à faire.

| ID | Sévérité | Fichier:ligne | Problème | Correction proposée | Risque de régression |
|----|----------|---------------|----------|---------------------|----------------------|
| B1 ✅ | ~~BLOQUANT~~ → **IMPORTANT** | `fits/randles_full.py:117` | Résidu imaginaire de l'optimiseur `(Z.imag - spectrum.Zim)` : `spectrum.Zim` positif (loader), `Z.imag` physique (négatif) → mauvais signe, incohérent avec le χ² l.151 (lui correct). Impact **réel mesuré** : biais systématique ~25-30 % sur Rct + **effondrement sur 100 Ω aux Rct élevés** + calibration dégradée (pente 0.126 vs 0.150, R² 0.80). *(La 1re annexe « 188 vs 5000 » était un cas cherry-pické non représentatif.)* | **Appliqué** : `(-Z.imag - spectrum.Zim)`. Après : Rct ±1 %, χ²/10³, pente 0.152/R² 1.00. | Change tous les Rct_randles dans le bon sens ; couvert par le test bout-en-bout. |
| B2 ✅ | **BLOQUANT** | `launch.bat:89` | `)` orphelin non apparié après le bloc `if/else` → erreur de parsing batch. | **Appliqué** : ligne supprimée. | Faible. |
| B3 ✅ | **IMPORTANT** | `requirements.txt` | `matplotlib` utilisé (`plotting/eis_plots.py:1030,1227,1286`, `plotting/cv_plots.py`, `pages/1_pretraitement.py`) mais **absent** → `ModuleNotFoundError` sur les fenêtres matplotlib. | **Appliqué** : `matplotlib>=3.8.0` ajouté. | Faible. |
| I1 ✅ | **IMPORTANT** | `tests/test_fits.py:41,56,116` | Fixtures en `Zim=Z.imag` (**négatif**), inverse de la convention loader → tests verts alors que le chemin réel était cassé (B1). Aucun test loader→fit. | **Appliqué** : fixtures en `Zim=-Z.imag` + `test_randles_recovers_rct_end_to_end` (±5 %). | Résolu avec B1. |
| I2 ✅ | **IMPORTANT** | `core/config.py:29-37` vs `config/default.yaml` | Pydantic déclarait `ZD0`/`D_eff` ; le YAML et `randles_full` utilisent `R_D`/`tau_d`. **Précision (mesurée)** : seuls **R_D et tau_d** n'étaient pas lus (retombaient sur le fallback en dur, qui vaut par chance la même valeur) ; les 6 autres bornes (Re, Re_prime, Cb, **Rct**, Qdl, alpha) **étaient bien lues du YAML**. La borne Rct=100 vient donc du YAML, pas du fallback. | **Appliqué** : clés réalignées `R_D`/`tau_d` ; bornes typées `list[float]` ; tests `test_config.py`. | Faible. |
| I2b ✅ | **IMPORTANT** | `config/default.yaml:33,36,37` | **Piège résolveur float YAML 1.1** : `1.0e9`, `1.0e6`, `1.0e3` (exposant sans signe) parsés comme **chaînes**, et les champs Pydantic `list` nus ne les coerçaient pas. Inoffensif aujourd'hui (scipy coerce via `asarray(dtype=float)` : fit OK), mais fragile. Balayage complet : seules ces 3 valeurs touchées. | **Appliqué** : `→ 1.0e+9/1.0e+6/1.0e+3` + typage `list[float]` (Pydantic coerce) + test `isinstance(b,float)`. | Faible. |
| I3 ✅ | **IMPORTANT** | `.github/workflows/build_exe.yml` | CI = seulement `py_compile app.py` : **jamais pytest**, autres modules non compilés, nom trompeur, Python 3.11 (projet : 3.12). | **Appliqué** : renommé `validate.yml`, Python 3.12, `compileall` de tous les paquets, `pytest`, trigger PR. | Faible. |
| I4 ✅ | **IMPORTANT** | `core/pipeline.py:38-39,53` | `getattr(config_dict, ...)` → toujours le défaut. `alpha_noise` YAML ignoré ; `kk_mu_threshold`/`kk_residual_pct` inexistants (ni YAML ni Pydantic). | **Appliqué** : `alpha_noise` lu par clé ; `getattr` KK fantômes supprimés (seuils = défauts `validator.py`, source unique). | Faible. |
| V1 ✅ | **BLOQUANT** | `core/cv_loader.py:24-29` | **Corruption silencieuse de données** (même classe que B1, pas un simple « test rouge ») : le repli « sous-chaîne » de `_find_column` avec les alias 1-lettre `"e"`/`"i"` matchait `"time"→I` (le **temps** chargé comme courant !), `"temperature"→E`, `"voltage_x"→E`. Un CV pouvait donc être analysé avec les mauvaises colonnes, sans erreur. Révélé par `test_missing_e_column_raises` quand la CI a branché pytest. | **Appliqué** : repli sous-chaîne supprimé, match exact (unité `/…` retirée) uniquement. Formats EC-Lab légitimes couverts. | Faible. |
| V2 ✅ | — (vérifié) | `core/loader.py:51-88` | **Contre-vérification** : le loader **EIS** utilise-t-il le même matching permissif ? **Non** — match **exact** (`normed in aliases`), pas de sous-chaîne. Vérifié empiriquement : colonnes `time`/`temperature` distractrices ignorées, `f`/`Zre`/`Zim` correctement identifiés ; en-têtes inconnus → `ValueError` propre. **Pas le bug V1.** Reste un repli **positionnel** (3 premières colonnes numériques) mais **avec avertissement** (pas silencieux) → risque moindre, laissé tel quel. | Aucune action requise. | — |
| I5 ✅ | **IMPORTANT** | `plotting/eis_plots.py` + `exports/exporter.py` | **Duplication + violation de couche** : « signal normalisé + `linregress` » réimplémenté dans `calibration_figure`, `calibration_drt_figure` (log-log), la fenêtre matplotlib et `export_calibration_csv`. `plotting/` **calculait**. | **Appliqué** : `core/calibration.py` (`compute_calibration`, `compute_calibration_loglog`, `compute_calibration_all`) — source unique ; 4 sites EIS branchés dessus ; `stats` retiré de `eis_plots`. Test `test_calibration.py` : figure == export == core (pente/ordonnée/R²). **Reste I5b** : les 2 exports de calibration **CV** (`exporter.py:214,281`) — famille de données distincte (`CVSession`), à factoriser à part. | Moyen — couvert par tests. |
| I6 | **IMPORTANT** | `fits/registry.py:19-21` | `_discover` fait `except Exception: continue` : si un modèle échoue à l'import (ex. `cvxopt` absent), il **disparaît silencieusement** de l'app, sans message. | Logger le nom du module + l'exception avant de continuer. | Faible. |
| I7 | **IMPORTANT** | `fits/randles_full.py:139` + `core/pipeline.py:164,189,210` | Échecs de fit avalés : `except Exception` renvoie `x0`/`converged=False` sans signaler à l'UI ; le pipeline se contente de `log.error`. L'utilisateur voit un Rct sans savoir que le fit a échoué. | Propager un état d'échec au FitResult et l'afficher (badge/`st.warning`). | Faible. |
| I8 | **IMPORTANT** | `ui/sidebar.py`, `ui/data_input.py` | **Fichiers orphelins** : `render_sidebar` / `data_input` ne sont importés nulle part (seul `ui/tabs.py` est utilisé, via `pages/A_eis.py` et `B_cv.py`). Code mort complet. | Supprimer après confirmation. | Faible (à confirmer qu'aucune page future ne les vise). |
| C1 | COSMÉTIQUE | `core/config.py:41-43,52,60-63,13-26` | **Config morte** : `n_monte_carlo`, `drt.{n_tau,tau_min,tau_max}`, `export.{dpi,fig_width,fig_height}`, `physics.{D_FeII,D_FeIII}`, tout `geometry`/`conditions` ne sont lus par aucune logique d'analyse. | Retirer du YAML/Pydantic, ou brancher réellement. | Faible. |
| C2 | COSMÉTIQUE | `plotting/eis_plots.py:325,798-806` etc. | `get_theme("light")` / `apply_theme_to_figure(fig,"light")` **codés en dur** partout → le thème sombre (`plotting/theme.py:14-21`) et le « toggle thème » de la doc sont morts. | Passer le mode en paramètre, ou supprimer la palette dark. | Faible. |
| C3 | COSMÉTIQUE | `plotting/eis_plots.py:417-482` | `drt_lambda_diag_figure` lit des params (`_lc_lambdas`, `_gcv_*`, `lambda_lcurve`, `_str_lambda_method`) que `drt_tikhonov` ne produit jamais → figure toujours « non disponible », méthode λ toujours vide dans les titres. | Supprimer la figure ou produire ces params dans le modèle. | Faible. |
| C4 | COSMÉTIQUE | `fits/drt_fft.py:74-94` ↔ `fits/drt_tikhonov.py:85-96` + extraction Rct | `_local_maxima` et l'extraction Rct (convention Bissessur : avant-dernier pic, trapèze ±3 ln τ) sont **dupliqués** entre les deux modèles DRT. | Remonter dans `fits/physics.py` (ou `fits/_drt_common.py`). | Faible. |
| C5 | COSMÉTIQUE | `fits/physics.py:100-110,66-81,84-97` | `theta_EIS`, `Rct_bare_theory`, `Cdl_brug` **uniquement appelés par les tests**, jamais par l'app (la calibration utilise `|Rct_probe−Rct|/|Rct_probe|`, pas `θ_EIS`). `theta_EIS` sans garde `Rct_ap=0` ; `Rct_bare_theory` code en dur R,F,n au lieu de la config. | Décider : brancher ou marquer explicitement « théorique/tests ». | Faible. |
| C6 ✅ | COSMÉTIQUE | `requirements.txt` | `lmfit` déclaré mais **jamais importé** (le fit utilise `scipy.optimize.least_squares`). | **Appliqué** : retiré. | Faible. |
| C7 ✅ | COSMÉTIQUE | `launch.bat:81-88` | Échec de `pip install` non testé → l'app démarre avec des dépendances cassées. | **Appliqué** : `if !errorlevel! neq 0 (… & pause & exit /b 1)`. | Faible. |
| C8 | COSMÉTIQUE | `core/pipeline.py:159-211`, `fits/drt_fft.py:127` | « 4 fits en parallèle » (doc) est en réalité **séquentiel** (3 modèles), et `drt_fft` **re-fitte Randles** en interne → Randles calculé 2× par spectre. Pas de `@st.cache_data` → re-fit complet à chaque rerun Streamlit. | Réutiliser le FitResult Randles ; envisager cache/parallélisme si le temps le justifie. | Moyen. |
| C9 | COSMÉTIQUE | `ARCHITECTURE.md`, `Reference_maintenance`, `project` | Forte **dérive doc** : v3 décrit `render_tabs`/6 onglets/`C_comparatif.py`/modèles `randles_classique,randles_contraint,circulaire_fit`/`validate.yml` — rien de tout ça n'existe. `project` et `Reference_maintenance` (sans extension) non documentés à la racine. | Réécrire ARCHITECTURE.md sur l'état réel (pages `st.navigation`, 3 modèles). | Nul (doc). |
| C10 | COSMÉTIQUE | `fits/drt_tikhonov.py:106` | `label = "DRT Tikhonov + NNLS"` alors que la résolution est un **QP** cvxopt (pas NNLS). | Renommer « DRT Tikhonov (QP) ». | Nul. |
| C11 | COSMÉTIQUE | `plotting/eis_plots.py:868` | `yerr = errs*rcts/(rcts*ln10)` : les `rcts` se simplifient → `errs/ln10`. Trompeur. | Simplifier. | Nul. |
| C12 | COSMÉTIQUE | `fits/_pydrttools/basics.py:682-707`, `parameter_selection.py:374` | `print()` de debug résiduels — **code vendoré (pyDRTtools, MIT)**. Voir « À ne pas toucher ». | Ne pas modifier (upstream). | — |

---

## 3. Plan de nettoyage ordonné

Chaque étape laisse le projet fonctionnel et est validable indépendamment.

1. **Corriger le signe du fit Randles (B1)** — le seul changement qui touche un résultat physique.
   Corriger `randles_full.py:117`, puis **corriger d'abord les fixtures de test (I1)** pour qu'elles
   reflètent la convention loader, et ajouter un test `loader → RandlesFullModel().fit` qui vérifie
   Rct sur spectre synthétique positif. → Valider sur un vrai jeu EC-Lab avant de continuer.
2. **Réparer le launcher & les dépendances (B2, B3, C6, C7)** — `launch.bat:89`, ajouter `matplotlib`,
   retirer `lmfit`, faire échouer proprement `pip install`. Sans risque physique.
3. **Fiabiliser la CI (I3)** — lancer pytest + compiler tous les modules ; c'est le garde-fou de tout le reste.
4. **Cohérence config↔code (I2, I4, C1)** — aligner `bounds_randles_full` (R_D/tau_d), lire la config par
   clé et non `getattr`, supprimer la config morte. Faire **après** la CI pour être couvert.
5. **Robustesse silencieuse (I6, I7)** — remonter les échecs d'import de modèles et de fit à l'utilisateur.
6. **Factorisation calibration (I5)** — créer `core/calibration.py`, y déplacer signal-norm + régression,
   brancher figures + exports dessus. Supprime la duplication **et** la violation de couche `plotting/`.
7. **Factorisation DRT (C4)** — remonter `_local_maxima` + extraction Rct Bissessur en commun.
8. **Suppression du code mort (I8, C2, C3, C5)** — après confirmation : `ui/sidebar.py`, `ui/data_input.py`,
   palette dark, `drt_lambda_diag_figure`, fonctions physiques non branchées.
9. **Doc (C9, C10, C11)** — réaligner ARCHITECTURE.md, renommer le label DRT, simplifier `yerr`.

Ordre logique : **fiabilité (1-3) → cohérence (4-5) → propreté (6-9)**.

---

## 4. « À ne pas toucher » (probablement volontaire — me demander avant)

- **`core/loader.py:113-120`** — suppression des fréquences parasites 50/100 Hz et **correction du signe
  Zim** (convention EC-Lab). Volontaire et documenté. Le loader est correct (il produit Zim>0) ; c'était
  `randles_full` qui appliquait le mauvais signe côté résidu (B1, désormais **corrigé** — voir §2bis).
- **`fits/_pydrttools/`** — code **vendoré** de pyDRTtools (Ciucci lab, MIT, cf. `THIRD_PARTY_LICENSES.md`).
  Les `print()` (C12), le style et la structure sont ceux de l'upstream : **ne pas refactorer** pour rester
  fidèle à l'outil de référence cité (Bissessur PRE 2026).
- **`fits/drt_fft.py` — DRT sur spectre *idéal*** : n'analyse PAS les données brutes indépendamment
  (re-fit Randles → grille dense → FFT/Wiener). C'est **voulu** (section III.C du papier, étude des lois MAD),
  pas un bug. Ne pas « corriger » en le branchant sur les données expérimentales.
- **Extraction Rct « avant-dernier pic local » + trapèze ±3 en ln(τ)** (`drt_tikhonov.py:232-249`,
  `drt_fft.py:193-210`) — **convention Bissessur**, pas une heuristique arbitraire.
- **`drt_tikhonov.py:155-207` — recherche scalaire bornée + rGCV + back-off de λ** : contournements
  numériques **assumés** (SLSQP fragile, GCV instable sur spectres bruités épars). Ne pas « simplifier »
  vers le GCV/optimal_lambda natif sans en mesurer l'effet.
- **`plotting/theme.py`** garde une palette **dark** inutilisée (C2) : à supprimer *seulement* si tu confirmes
  qu'aucun retour du thème sombre n'est prévu.

---

## Annexe — vérification du bug B1 (résolu)

Balayage de Rct sur un demi-cercle propre, **via le chemin réel loader→fit** (`Zim = -Im(Z) > 0`),
code **avant** correction :

| Rct vrai | Rct ajusté (avant) | écart |
|--:|--:|--:|
| 1 000 | 747 | −25 % |
| 5 000 | **100** | effondrement (min. local) |
| 20 000 | 14 900 | −25 % |
| 80 000 | 58 010 | −27 % |

Le signe correct (`-Z.imag - spectrum.Zim`, cohérent avec le χ² l.151) rétablit Rct à **±1 %** et
divise le χ² par ~10³. Sur une série de calibration synthétique : pente **0.126 → 0.152** (vraie 0.150),
R² **0.80 → 1.00**. La 1re version de cette annexe ne montrait que le cas Rct=5000 (188 Ω), un
minimum local **non représentatif** : le biais typique est ~25-30 %, pas un ordre de grandeur — d'où
l'observation « courbe du bon ordre de grandeur » sur données réelles. Corrigé en commit `fix(randles)`
+ fixtures/test (`fix(randles)` / I1). ✅

---

*Correctifs prioritaires (B1, B2, B3, I1–I4, I2b, cv_loader, C6, C7) appliqués et poussés.
Restent des items de propreté (I5–I8, C1–C12) — à traiter sur validation.*
