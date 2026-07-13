# AUDIT — EIS Analyzer (Interface-EIS-CV-measurement)

> Audit de fiabilisation **avant** modifications fonctionnelles.
> Phases 1 & 2 = **lecture seule**. Ce document est le livrable de la Phase 3.
> **Aucune correction n'a été appliquée** — en attente de validation par point.
> Date : 2026-07-13 · Branche : `claude/eis-analyzer-audit-a2ml22`

---

## 1. Résumé exécutif (10 lignes)

Le projet est globalement bien structuré : le cloisonnement des couches est **respecté**
(aucun import Streamlit dans `core/`, `fits/`, `plotting/`, `exports/`), chaque paquet a son
`__init__.py`, et la suite de tests est plus fournie que ne le dit la doc (7 fichiers de test,
pas seulement `test_fits.py`). **Mais un bug bloquant fausse le résultat principal de l'app :**
le résidu imaginaire du fit Randles a le **mauvais signe** par rapport à la convention réelle
du loader (Zim positif), ce qui rend **Rct_randles complètement faux sur des données réelles**
(vérifié numériquement : 188 Ω au lieu de 5000 Ω). Les tests ne l'attrapent pas car leurs
spectres synthétiques utilisent la convention de signe *inverse* de celle produite par le loader.
S'y ajoutent : un `)` parasite dans `launch.bat`, `matplotlib` utilisé mais absent de
`requirements.txt`, une CI qui ne lance jamais les tests, une désynchro config↔code
(`bounds_randles_full`), de la config morte, de la duplication (calibration réimplémentée ≥4 fois),
et une forte dérive documentaire (ARCHITECTURE.md décrit une v3 qui n'existe plus).
**État de santé : fonctionnel en apparence, mais le chiffre-clé (Rct) est faux — à corriger en priorité absolue.**

---

## 2. Tableau des problèmes

| ID | Sévérité | Fichier:ligne | Problème | Correction proposée | Risque de régression |
|----|----------|---------------|----------|---------------------|----------------------|
| B1 | **BLOQUANT** | `fits/randles_full.py:117` | Résidu imaginaire de l'optimiseur `(Z.imag - spectrum.Zim)` : `spectrum.Zim` est en **convention positive** (loader), `Z.imag` est physique (négatif). Le signe est faux → l'optimiseur ajuste le modèle sur `+Im`. **Vérifié : Rct=188 Ω au lieu de 5000 Ω sur données loader.** Incohérent avec le χ² ligne 151 (`spectrum.Zim + Z_fit.imag`, lui correct). | Résidu = `(-Z.imag - spectrum.Zim)`. Uniformiser la convention entre optimiseur et χ². | Élevé — change tous les Rct_randles (mais dans le bon sens : ils deviennent justes). À valider sur données réelles. |
| B2 | **BLOQUANT** | `launch.bat:89` | `)` orphelin non apparié après le bloc `if/else` d'install des dépendances → erreur de parsing batch (`) was unexpected`). | Supprimer la ligne 89. | Faible. |
| B3 | **IMPORTANT** | `requirements.txt` | `matplotlib` utilisé (`plotting/eis_plots.py:1030,1227,1286`, `plotting/cv_plots.py`, `pages/1_pretraitement.py`) mais **absent** de requirements → `ModuleNotFoundError` dès qu'on ouvre une fenêtre matplotlib. | Ajouter `matplotlib>=3.8`. | Faible. |
| I1 | **IMPORTANT** | `tests/test_fits.py:41` (et `test_physics`, etc.) | Les spectres synthétiques stockent `Zim=Z.imag` (**négatif**), à l'inverse de la convention `EISSpectrum.Zim` positive du loader. Résultat : les tests « passent » alors que le chemin réel est cassé (B1). Aucun test loader→fit bout-en-bout, aucun test appelant `RandlesFullModel().fit` directement. | Générer les fixtures avec `Zim=-Z.imag` ; ajouter un test loader→randles vérifiant Rct. | Moyen — révèle B1, donc « casse » volontairement les tests jusqu'à correction. |
| I2 | **IMPORTANT** | `core/config.py:36-37` vs `config/default.yaml:36-37` | Le modèle Pydantic `BoundsRandlesFull` déclare `ZD0`/`D_eff`, le YAML fournit `R_D`/`tau_d`. Pydantic (`extra='ignore'`) **jette silencieusement** `R_D`/`tau_d` ; `randles_full.bounds` lit `R_D`/`tau_d` absents du dict dumpé → **retombe sur les bornes en dur** (l.67-76). Les bornes YAML de R_D/tau_d sont donc **ignorées**, et `ZD0`/`D_eff` ne sont lus nulle part. | Aligner les clés Pydantic sur `R_D`/`tau_d`. | Faible (met en cohérence). |
| I3 | **IMPORTANT** | `.github/workflows/build_exe.yml` | La CI ne fait que `py_compile app.py` : **ne lance jamais pytest**, ne compile pas les autres modules, nom trompeur (aucun build exe), Python 3.11 alors que le projet cible 3.12. | Ajouter `pytest tests/ -q` ; compiler tout ; renommer `validate.yml` ; Python 3.12. | Faible. |
| I4 | **IMPORTANT** | `core/pipeline.py:38-39,53` | `getattr(config, "alpha_noise"/"kk_mu_threshold"/"kk_residual_pct", ...)` sur un **dict** → renvoie toujours le défaut. `alpha_noise` configuré est ignoré ; `kk_mu_threshold`/`kk_residual_pct` n'existent ni dans le YAML ni dans Pydantic. | Lire via `config["fit"]["alpha_noise"]` ; ajouter/retirer les clés KK proprement. | Faible. |
| I5 | **IMPORTANT** | `plotting/eis_plots.py:614,860` + `exports/exporter.py:177,214,281` | **Duplication + violation de couche** : la logique « signal normalisé + régression `linregress` » est réimplémentée ≥4 fois (2 figures Plotly, 1 matplotlib, plusieurs exports CSV). `plotting/` **calcule** (interdit par la règle « plotting ne calcule rien »). | Extraire une fonction unique `core/calibration.py::compute_calibration(session, model)` ; figures et exports la consomment. | Moyen (refactor transverse). |
| I6 | **IMPORTANT** | `fits/registry.py:19-21` | `_discover` fait `except Exception: continue` : si un modèle échoue à l'import (ex. `cvxopt` absent), il **disparaît silencieusement** de l'app, sans message. | Logger le nom du module + l'exception avant de continuer. | Faible. |
| I7 | **IMPORTANT** | `fits/randles_full.py:139` + `core/pipeline.py:164,189,210` | Échecs de fit avalés : `except Exception` renvoie `x0`/`converged=False` sans signaler à l'UI ; le pipeline se contente de `log.error`. L'utilisateur voit un Rct sans savoir que le fit a échoué. | Propager un état d'échec au FitResult et l'afficher (badge/`st.warning`). | Faible. |
| I8 | **IMPORTANT** | `ui/sidebar.py`, `ui/data_input.py` | **Fichiers orphelins** : `render_sidebar` / `data_input` ne sont importés nulle part (seul `ui/tabs.py` est utilisé, via `pages/A_eis.py` et `B_cv.py`). Code mort complet. | Supprimer après confirmation. | Faible (à confirmer qu'aucune page future ne les vise). |
| C1 | COSMÉTIQUE | `core/config.py:41-43,52,60-63,13-26` | **Config morte** : `n_monte_carlo`, `drt.{n_tau,tau_min,tau_max}`, `export.{dpi,fig_width,fig_height}`, `physics.{D_FeII,D_FeIII}`, tout `geometry`/`conditions` ne sont lus par aucune logique d'analyse. | Retirer du YAML/Pydantic, ou brancher réellement. | Faible. |
| C2 | COSMÉTIQUE | `plotting/eis_plots.py:325,798-806` etc. | `get_theme("light")` / `apply_theme_to_figure(fig,"light")` **codés en dur** partout → le thème sombre (`plotting/theme.py:14-21`) et le « toggle thème » de la doc sont morts. | Passer le mode en paramètre, ou supprimer la palette dark. | Faible. |
| C3 | COSMÉTIQUE | `plotting/eis_plots.py:417-482` | `drt_lambda_diag_figure` lit des params (`_lc_lambdas`, `_gcv_*`, `lambda_lcurve`, `_str_lambda_method`) que `drt_tikhonov` ne produit jamais → figure toujours « non disponible », méthode λ toujours vide dans les titres. | Supprimer la figure ou produire ces params dans le modèle. | Faible. |
| C4 | COSMÉTIQUE | `fits/drt_fft.py:74-94` ↔ `fits/drt_tikhonov.py:85-96` + extraction Rct | `_local_maxima` et l'extraction Rct (convention Bissessur : avant-dernier pic, trapèze ±3 ln τ) sont **dupliqués** entre les deux modèles DRT. | Remonter dans `fits/physics.py` (ou `fits/_drt_common.py`). | Faible. |
| C5 | COSMÉTIQUE | `fits/physics.py:100-110,66-81,84-97` | `theta_EIS`, `Rct_bare_theory`, `Cdl_brug` **uniquement appelés par les tests**, jamais par l'app (la calibration utilise `|Rct_probe−Rct|/|Rct_probe|`, pas `θ_EIS`). `theta_EIS` sans garde `Rct_ap=0` ; `Rct_bare_theory` code en dur R,F,n au lieu de la config. | Décider : brancher ou marquer explicitement « théorique/tests ». | Faible. |
| C6 | COSMÉTIQUE | `requirements.txt:8` | `lmfit` déclaré mais **jamais importé** (le fit utilise `scipy.optimize.least_squares`). | Retirer `lmfit`. | Faible. |
| C7 | COSMÉTIQUE | `launch.bat:81-88` | Échec de `pip install -r requirements.txt` non testé (errorlevel seulement affiché) → l'app démarre avec des dépendances cassées. | `if !errorlevel! neq 0 (echo ... & pause & exit /b 1)`. | Faible. |
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
  Zim** (convention EC-Lab). Volontaire et documenté. ⚠️ **Ne pas confondre avec B1** : le loader est
  correct (il produit Zim>0) ; c'est `randles_full` qui applique le mauvais signe côté résidu.
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

## Annexe — vérification du bug B1 (reproductible)

Spectre Randles synthétique (Rct_vrai = 5000 Ω), construit dans la **convention du loader** (`Zim = -Im(Z) > 0`) :

| Cas | Rct ajusté |
|-----|-----------|
| Code actuel (`Z.imag - spectrum.Zim`) sur données loader (Zim>0) | **188.6 Ω** ❌ |
| Convention des tests (`Zim = Z.imag`, négatif) | 4869 Ω ✅ |
| Données loader (Zim>0) + résidu **corrigé** (`-Z.imag - spectrum.Zim`) | 4869 Ω ✅ |

Conclusion : le fit ne fonctionne aujourd'hui **que** sur la convention (négative) des fixtures de test ;
sur les données réelles produites par le loader, `Rct_randles` est faux. Corriger le signe **et** les fixtures.

---

*Fin de l'audit — STOP. En attente de ta validation, point par point, avant toute modification.*
