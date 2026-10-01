# Provenance — `drt/bayes_drt2/`

Code tiers : **bayes-drt2** (Jake Huang, Colorado School of Mines), inversion hiérarchique
bayésienne de spectres d'impédance (DRT/DDT). Copié tel quel depuis un clone git identifié,
puis **trois patchs**, chacun justifié ci-dessous :

1. `np.trapz` → `np.trapezoid` — **compatibilité**, indispensable avec numpy 2.5.3 (sinon aucun
   chemin ne s'exécute) ; c'est le **seul** patch de compatibilité nécessaire avec les versions épinglées ;
2. `Inverter.fit(..., adapt_delta=0.9)` — **exposition** d'un réglage NUTS codé en dur, défaut amont
   inchangé ; nécessaire au réglage par défaut retenu dans `drt/VALIDATION_REGLAGES.md` (§3) ;
3. `from cmdstanpy import CmdStanModel` → `from ..stan_compile import compile_stan_model as
   CmdStanModel` — **robustesse** : un chemin d'installation non-ASCII (« .Thèse ») n'atteint plus
   jamais `mingw32-make` (§3, patch 3). Une seule ligne ; les sites d'appel ne changent pas.

## 1. Origine

| | |
|---|---|
| **Dépôt** | https://github.com/jdhuang-csm/bayes-drt2 |
| **Branche** | `main` (seule branche ; aucun tag) |
| **Commit** | **`99d5b603d98469a6382ddde5210a1bfdc0d76b3d`** (= `HEAD` = `refs/heads/main` au moment du clone, vérifié par `git ls-remote`) |
| **Arbre (tree)** | `554744912f05671cb476ae4d9a0cc614ae62e8ff` |
| **Auteur / date du commit** | jdhuang `<jdhuang@mines.edu>`, 2025-01-11 12:02:30 +0100 — « Correct ridge initialization and predict_dist arg order » |
| **Version déclarée** | `0.2` (`setup.py`) |
| **Date du clone** | **2026-09-30 12:43 UTC** (`git clone https://github.com/jdhuang-csm/bayes-drt2`, dossier temporaire hors dépôt) |
| **Licence** | BSD 3-Clause — texte intégral en §5 et dans `drt/bayes_drt2/LICENSE` |
| **Citation demandée par l'auteur** | Huang, J., Papac, M., O'Hayre, R. (2020). *Towards robust autonomous impedance spectroscopy analysis: a calibrated hierarchical Bayesian approach for electrochemical impedance spectroscopy (EIS) inversion.* Electrochimica Acta 367, 137493. https://doi.org/10.1016/j.electacta.2020.137493 |

Historique amont complet (8 commits) : `1d3023d` Initial commit (2022-09-24) → … → `82c91fb` Fixed
multi-distribution fit (2022-10-06) → `a23b372` merge → **`99d5b60`** (2025-01-11).
Le dernier commit modifie **uniquement** `inversion.py` : dans `Inverter.fit`, la préparation des
matrices (`_prep_matrices`) est déplacée **après** l'initialisation ridge (lecture du diff :
`ridge_fit`, appelé par l'initialisation, prépare lui-même ses matrices et modifiait donc l'état
préparé juste avant pour le MAP), et `predict_distribution(tau, name)` devient
`predict_distribution(name, tau)` — l'ordre utilisé par `fits/drt_fit.py`.

### Lien avec l'ancien `vendor/bayes_drt2/` (commit « introuvable »)

`vendor/README.md` déclarait le commit amont « non récupérable ». Il l'est maintenant : après le seul
patch 1, les fichiers de `drt/bayes_drt2/` étaient **identiques octet pour octet** à ceux de
`vendor/bayes_drt2/` (`cmp` sur les 7 `.py`, `LICENSE` et les 13 `.stan` vendorés ; depuis le
patch 2, `inversion.py` en diffère des deux seules lignes de ce patch). Le vendoring correspondait donc
exactement à **`99d5b60` + le même patch `np.trapz → np.trapezoid`** — c'est d'ailleurs le seul
commit amont dont `inversion.py` a le sha256 `9a730ac4…` publié par `vendor/README.md`
(`1d3023d` : `f45f6f7b…`, `e9a9014`→`e920c8e` : `2a6b20a3…`, `82c91fb`/`a23b372` : `71177aee…`).
Conséquence : les défauts mesurés par `AUDIT.md` §4.4 sur le vendoring sont ceux de la **dernière**
version amont, pas d'une version périmée.

Le vendoring omettait deux fichiers amont, présents ici : `equiv_circuit.py` et
`stan_model_files/Parallel_fitY_SA.stan` (voir §3).

## 2. Contenu copié et empreintes

Copie **intégrale** du dossier `bayes_drt2/` du commit (22 fichiers) + `LICENSE` de la racine amont.
Non copiés (hors paquet) : `README.md`, `installation.txt`, `setup.py`, `MANIFEST.in`.

`git hash-object` = identifiant de blob git (vérifiable par `git ls-tree -r 99d5b60` sur un clone) ;
les blobs non patchés **sont** ceux de l'amont.

| Fichier | Blob git (ici) | Blob amont `99d5b60` | sha256 (ici) |
|---|---|---|---|
| `__init__.py` (vide) | `e69de29b` | = | `e3b0c442…b855` |
| `equiv_circuit.py` | `f6930ddf` | = | `e53e7f2a…b73d` |
| `file_load.py` | `7f286a00` | = | `c4f8fdfa…1fd2` |
| **`inversion.py`** (patchs 2 et 3) | `63f64779` | `421e2d84` | `8a8f4569…1215` (amont `9a730ac4…9e70`) |
| **`matrices.py`** (patch 1) | `c2c32cf2` | `81908307` | `43aed2f5…b2c2` (amont `f65803ac…ecdb`) |
| **`peak_fit.py`** (patch 1) | `51e6126e` | `90aee12d` | `c38b9c17…abaf` (amont `82c22bc7…b158`) |
| `plotting.py` | `4f9a65f3` | = | `3dd07137…aa94` |
| `utils.py` | `482c249b` | = | `effad4bc…ea83` |
| `LICENSE` | `344b0732` | = | `e67efbea…f5d5` |
| `stan_model_files/*.stan` (14) | tous = amont | = | — |

Les 14 modèles Stan (`Parallel`, `Parallel_fitY`, `Parallel_fitY_SA`, `Parallel_outliers`,
`Series`, `Series_outliers`, `Series_pos`, `Series_pos_outliers`, `Series-Parallel`,
`Series-Parallel_outliers`, `Series-Parallel_pos`, `Series-Parallel_pos_outliers`,
`Series-2Parallel`, `Series-2Parallel_pos`) sont inchangés. Seuls `Series.stan` et `Series_pos.stan`
sont utilisés par une DRT simple (`Inverter._get_stan_model`, `nonneg` → suffixe `_pos`).

Artefacts de compilation : `CmdStanModel(stan_file=…)` compile l'exécutable **à côté** du `.stan`
(`stan_model_files/Series`, `Series.exe` sous Windows). Ils sont exclus par `.gitignore`
(`drt/bayes_drt2/stan_model_files/*` sauf `*.stan`).

## 3. Patchs appliqués

Environnement de vérification (celui épinglé à l'étape 1 : `requirements.txt` + job CI `drt`) :
Linux, Python 3.12.3, **numpy 2.5.3, scipy 1.18.1**, pandas 3.0.6, matplotlib 3.11.2,
cvxopt 1.3.3, cmdstanpy 1.3.0, CmdStan 2.36.0.

### Méthode de recherche des patchs nécessaires

1. **Statique** : parcours AST de tous les modules ; chaque chaîne d'attributs `np.*`, `pd.*`, `plt.*`,
   `cvxopt.*` et chaque `from scipy… import …` est résolue dans les versions installées.
   Résultat : **seul `np.trapz` est irrésolu** (7 appels actifs : `matrices.py` ×3, `peak_fit.py` ×4).
2. **Dynamique** : le code amont **non patché** est exécuté sur tous les chemins utilisés par le
   projet (`ridge_fit` ; `fit(mode='optimize')` avec défaut, `init_from_ridge`, `nonneg`, les deux,
   `outliers='auto'` ; `fit(mode='sample')` ; `predict_distribution` avec et sans `percentile`,
   `predict_Z`, `predict_Rp` avec et sans `percentile`), avertissements capturés.
   - **Sans patch** : les 8 chemins échouent, tous sur
     `AttributeError: module 'numpy' has no attribute 'trapz'` (`np.trapz`, déprécié en numpy 2.0,
     n'existe plus en 2.5.3).
   - **Avec le seul patch 1** : les 8 chemins aboutissent. Aucune `DeprecationWarning` /
     `FutureWarning` Python émise par numpy, scipy, pandas ou cvxopt.

### Patch 1 (appliqué) — `np.trapz` → `np.trapezoid` (numpy ≥ 2.4 : `np.trapz` supprimé)

`np.trapezoid` est le nouveau nom de `np.trapz` (numpy 2.0) : même signature, même résultat.
Remplacement des 9 occurrences textuelles `np.trapz(` (7 appels + 2 lignes de commentaire, remplacées
aussi pour qu'un `grep np.trapz` reste vide) — commande exacte :
`sed -i 's/np\.trapz(/np.trapezoid(/g' matrices.py peak_fit.py`.
La chaîne `integrate_method='trapz'` (nom d'option interne, pas un appel numpy) est laissée telle quelle.

```diff
--- a/bayes_drt2/matrices.py   (amont 99d5b60, blob 81908307)
+++ b/drt/bayes_drt2/matrices.py (blob c2c32cf2)
@@ -226,16 +226,16 @@ def construct_A(frequencies, part, tau=None, basis='gaussian', fit_inductance=Fa
             quad_limits = (-20, 20)
         #		  elif part=='imag':
         #			  y = np.arange(-5,5,0.1)
-        #			  c = [np.trapz(func(y,w_n,t_0),x=y) for w_n in omega]
-        #			  r = [np.trapz(func(y,w_0,t_m),x=y) for t_m in 1/omega]
+        #			  c = [np.trapezoid(func(y,w_n,t_0),x=y) for w_n in omega]
+        #			  r = [np.trapezoid(func(y,w_0,t_m),x=y) for t_m in 1/omega]
 
         if integrate_method == 'quad':
             c = [quad(func, quad_limits[0], quad_limits[1], args=(w_n, t_0, epsilon), epsabs=1e-4)[0] for w_n in omega]
             r = [quad(func, quad_limits[0], quad_limits[1], args=(w_0, t_m, epsilon), epsabs=1e-4)[0] for t_m in tau]
         elif integrate_method == 'trapz':
             y = np.linspace(-20, 20, 1000)
-            c = [np.trapz(func(y, w_n, t_0, epsilon), x=y) for w_n in omega]
-            r = [np.trapz(func(y, w_0, t_m, epsilon), x=y) for t_m in tau]
+            c = [np.trapezoid(func(y, w_n, t_0, epsilon), x=y) for w_n in omega]
+            r = [np.trapezoid(func(y, w_0, t_m, epsilon), x=y) for t_m in tau]
         if r[0] != c[0]:
             print(r[0], c[0])
             raise Exception('First entries of first row and column are not equal')
@@ -260,7 +260,7 @@ def construct_A(frequencies, part, tau=None, basis='gaussian', fit_inductance=Fa
                            in tau]
             elif integrate_method == 'trapz':
                 y = np.linspace(-20, 20, 1000)
-                A[n, :] = [np.trapz(func(y, w_n, t_m, epsilon), x=y) for t_m in tau]
+                A[n, :] = [np.trapezoid(func(y, w_n, t_m, epsilon), x=y) for t_m in tau]
 
     return A
```

```diff
--- a/bayes_drt2/peak_fit.py   (amont 99d5b60, blob 90aee12d)
+++ b/drt/bayes_drt2/peak_fit.py (blob 51e6126e)
@@ -162,7 +162,7 @@ def fit_pos_peaks(tau, gamma, Rp, weights=None, check_shoulders=False, prom_rthr
         width = properties['widths'][i]
         start = int(peak - width)
         end = int(peak + width)
-        R = np.trapz(gamma[start:end], np.log(tau[start:end]))
+        R = np.trapezoid(gamma[start:end], np.log(tau[start:end]))
         t0 = tau[peak]
         alpha = 0.99  # 1 is symmetric
         beta = 0.8  # may be able to estimate from width
@@ -242,7 +242,7 @@ def fit_pos_peaks(tau, gamma, Rp, weights=None, check_shoulders=False, prom_rthr
                 width = new_peak_widths[i]
                 start = int(peak - width)
                 end = int(peak + width)
-                R = np.trapz((gamma - gamma_fit)[start:end], np.log(tau[start:end]))
+                R = np.trapezoid((gamma - gamma_fit)[start:end], np.log(tau[start:end]))
                 if R <= 0:
                     R = gamma[peak]
                 t0 = tau[peak]
@@ -278,7 +278,7 @@ def fit_pos_peaks(tau, gamma, Rp, weights=None, check_shoulders=False, prom_rthr
             # initialize new peak at largest misfit
             gamma_fit = evaluate_fit_distribution(x_filter, tau)
             peak = np.argmax(gamma - gamma_fit)
-            R = np.trapz((gamma - gamma_fit), np.log(tau))
+            R = np.trapezoid((gamma - gamma_fit), np.log(tau))
             if R <= 0:
                 R = gamma[peak]
             t0 = tau[peak]
@@ -416,7 +416,7 @@ def constrained_peak_fit(tau, gamma, tau0_guess, Rp, nonneg, lntau_uncertainty=3
         peak_width = 4  # peak width in ln_tau space
         start = np.argmin(np.abs(tau - (tau0_guess[n] * np.exp(-peak_width / 2))))
         end = np.argmin(np.abs(tau - (tau0_guess[n] * np.exp(peak_width / 2))))
-        R = np.trapz(gamma[start:end], np.log(tau[start:end]))
+        R = np.trapezoid(gamma[start:end], np.log(tau[start:end]))
         t0 = tau0_guess[n]
         alpha = 0.99
         beta = 0.8
```

Vérification : la matrice MAP de `drt/VALIDATION_REGLAGES.md` (§2) reproduit les nombres de
`AUDIT.md` §4.4 (mesurés avec numpy 2.4.6 / scipy 1.17.1 sur le vendoring patché de la même façon) :
identiques à la précision publiée pour les réglages `default` et `nonneg` ; pour `init_from_ridge`,
deux valeurs Randles diffèrent de 1,3 Ω et 1,7 Ω (≤ 0,04 %), les autres sont identiques.

### Patch 2 (appliqué) — exposer `adapt_delta` dans `Inverter.fit` (défaut amont 0,9 inchangé)

**Pas une incompatibilité** : un paramètre de NUTS codé en dur (`adapt_delta=0.9`, cible
d'acceptation de l'adaptation du pas) devient un argument de `Inverter.fit`, **de même valeur par
défaut** — tout appel existant se comporte à l'identique (les matrices MAP/HMC de
`drt/VALIDATION_REGLAGES.md` §2-3 ont d'ailleurs été produites avant ce patch, aux réglages amont).

**Pourquoi il est nécessaire** : avec la DRT contrainte ≥ 0 retenue par défaut, `adapt_delta=0.9`
laisse des transitions divergentes sur les spectres à arcs RC idéaux — 3 à 24 par ajustement de
4 chaînes × 500 tirages (4/4 essais « 2 RC » ; 0/4 essais Randles) —, donc l'alerte
« divergences > 0 » ; un warmup de 1000 n'y change rien (6, 14, 15 divergences), 0,95 en laisse
encore 13 sur un cas, **0,99 n'en laisse aucune** sur les 4 cas testés (R-hat ≤ 1,014) —
`drt/VALIDATION_REGLAGES.md` §3.
Sans ce patch, le seul moyen serait de réécrire `Inverter.fit` hors du paquet.

```diff
--- a/bayes_drt2/inversion.py   (amont 99d5b60, blob 421e2d84)
+++ b/drt/bayes_drt2/inversion.py (blob 24039a7f)
@@ -1332,7 +1332,7 @@ class Inverter:
 			# Sampling control
 			warmup=200, samples=200, chains=2,
 			add_stan_data={}, model_str=None,
-			fitY=False, SA=False, SASY=False):
+			fitY=False, SA=False, SASY=False, adapt_delta=0.9):
 		"""
 		Fit the defined distribution(s) using the calibrated hierarchical Bayesian model.
 		Model may be fitted either via optimization (maximum a posteriori estimate) or HMC sampling.
@@ -1478,7 +1478,7 @@ class Inverter:
 			self.stan_mcmc = model.sample(dat, iter_warmup=warmup, iter_sampling=samples, chains=chains,
 										  seed=random_seed,
 										  inits=init,
-										  adapt_delta=0.9)
+										  adapt_delta=adapt_delta)
 			self._sample_result = self.stan_mcmc.stan_variables()
 
 		# extract coefficients
```

### Patch 3 (appliqué) — compiler par `drt/stan_compile.py` (chemin d'installation non-ASCII)

**Pas une incompatibilité de bibliothèque : un défaut de Windows.** Sous Windows, cmdstanpy lance
`mingw32-make`, qui lance le shell MSYS avec le chemin du `.stan` ; si ce chemin contient un
caractère hors ASCII (`C:\Users\x\Desktop\.Thèse\…`), le shell le reçoit décalé (« è » → « Ã¨ ») et
échoue en « No such file or directory ». `Inverter` compile `stan_model_files/*.stan` à la demande
(tout modèle absent du cache, ex. `Parallel`), donc le chemin d'installation de l'utilisateur
atteint make à l'exécution, pas seulement à l'installation.

Le patch remplace l'import par un alias vers `compile_stan_model` : même appel
`CmdStanModel(stan_file=…)`, mais un `.stan` sous un chemin non-ASCII est copié, compilé et mis en
cache dans un dossier ASCII (clé : SHA-256 du `.stan` + version de CmdStan). Un chemin ASCII passe
tel quel à cmdstanpy. Détails, pistes écartées (`PYTHONUTF8`, `chcp 65001`, `cpp_options`,
`SanitizedOrTmpFilePath`) et garde-fous : docstring de `drt/stan_compile.py`,
`tests/test_stan_compile.py` (qui échoue si un autre `CmdStanModel(` apparaît hors de ce module).

```diff
--- a/drt/bayes_drt2/inversion.py
+++ b/drt/bayes_drt2/inversion.py
@@ -12,1 +12,3 @@
-from cmdstanpy import CmdStanModel
+# Patch 3 (drt/PROVENANCE.md) : meme appel `CmdStanModel(stan_file=...)` que l'amont, mais un
+# chemin non-ASCII n'atteint jamais make (drt/stan_compile.py). Les sites d'appel sont inchanges.
+from ..stan_compile import compile_stan_model as CmdStanModel
```

### Écarts constatés mais **non patchés** (non nécessaires — décision documentée)

| Constat | Pourquoi pas de patch |
|---|---|
| `equiv_circuit.py` importe `.misc_to_migrate`, **absent de l'amont** : le module n'est pas importable. | Aucun module du paquet ni du projet ne l'importe (`inversion.py` non plus). Conservé pour que la copie soit intégrale et vérifiable par blob ; ne jamais l'importer. |
| ~60 `SyntaxWarning: invalid escape sequence` (`'\.'`, `'\O'`, `'\m'`… dans des regex et des libellés LaTeX de `inversion.py`, `plotting.py`, `utils.py`, `equiv_circuit.py`), émis à la première compilation du bytecode sous Python ≥ 3.12. | Avertissement seulement : Python conserve le `\` tel quel, les chaînes sont **identiques** à l'exécution. Deviendrait nécessaire le jour où une version de Python en fait une `SyntaxError` (pas le cas en 3.12/3.13). |
| cmdstanpy 1.3.0 journalise (logger `cmdstanpy`, **pas** `warnings`) « The default behavior of CmdStanMLE.stan_variable() will change… » pour chaque variable scalaire lue par `fit(mode='optimize')` (`AUDIT.md` DRT-12). | Message de journal, comportement inchangé en 1.3.0 (retourne un `float`). `drt/engine.py` fait taire ce logger sous le niveau ERROR. À revérifier si cmdstanpy est monté en version (2.0 annoncé). |
| `stanc` (CmdStan 2.36) signale des divisions entières `N/2` dans `Series*.stan` (« Found int division »). | Voulu par le modèle (N = 2 × nombre de fréquences, toujours pair) ; résultat exact. |
| `Inverter.fit` ne transmet pas `max_treedepth` (défaut CmdStan : 10) ni `show_progress` (barres de progression tqdm sur la sortie standard). | Non nécessaires : la saturation de profondeur est une question d'efficacité, surveillée via R-hat/ESS (`drt/diagnostics.py`) ; les barres ne sont que du bruit de console. |

## 4. Mise à jour future

1. `git clone https://github.com/jdhuang-csm/bayes-drt2` ; noter `git rev-parse HEAD` et la date.
2. Remplacer `drt/bayes_drt2/` par `bayes_drt2/` + `LICENSE` ; réappliquer le patch 1 (commande `sed` ci-dessus)
   le patch 2 et le patch 3 (diffs ci-dessus), puis **refaire la recherche statique + dynamique** du §3 avec les
   versions épinglées du moment.
3. Rejouer `drt/validation/run_matrix.py` (MAP et HMC) et comparer à `drt/VALIDATION_REGLAGES.md`
   **avant** de modifier le réglage par défaut de `drt/engine.py`.
4. Mettre à jour ce fichier (commit, date, blobs, diffs).

## 5. Licence (texte intégral, recopié de `LICENSE` au commit `99d5b60`)

```
BSD 3-Clause License

Copyright (c) 2020, jdhuang-csm
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
