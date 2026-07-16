# Vendored third-party code

Code tiers copié dans le dépôt (« vendoré ») pour un usage contrôlé et
reproductible. Rien ici ne fait partie du code applicatif EduGradia/Interface ;
ces fichiers ne sont pas modifiés par le projet en dehors du correctif documenté.

---

## `bayes_drt2/` — bayes-drt2 (Jake Huang, Colorado School of Mines)

Inversion hiérarchique bayésienne de données d'impédance électrochimique (DRT).

| | |
|---|---|
| **Source** | https://github.com/jdhuang-csm/bayes-drt2 |
| **Version** | `0.2` (déclarée dans `setup.py`) |
| **Branche** | `main` |
| **Commit** | *non récupérable* — l'API GitHub (`api.github.com`) est bloquée par la politique d'egress de l'environnement (HTTP 403). Provenance assurée à la place par les sha256 par fichier ci-dessous. |
| **Récupéré le** | 2026-07-16 (UTC), via `raw.githubusercontent.com` (seul hôte GitHub autorisé ; `codeload.github.com` renvoie 403) |
| **Licence** | BSD 3-clause — voir `bayes_drt2/LICENSE` |

### Contenu

Modules Python purs (7) — suffisants pour l'import et pour `ridge_fit` :

```
bayes_drt2/__init__.py      (patch compat numpy, voir plus bas — vide à l'origine)
bayes_drt2/inversion.py     Inverter, ridge_fit / map_fit / bayes_fit, predict_*
bayes_drt2/matrices.py      construction des matrices A/L/M (PATCHÉ)
bayes_drt2/peak_fit.py      analyse de pics de la DRT           (PATCHÉ)
bayes_drt2/utils.py
bayes_drt2/plotting.py
bayes_drt2/file_load.py
bayes_drt2/LICENSE
```

Modèles Stan (`bayes_drt2/stan_model_files/*.stan`) — **best-effort**, 13 fichiers.
Ils ne servent **que** pour le HMC (`map_fit` / `bayes_fit`) et **ne sont pas
requis pour `ridge_fit`**. Le listing du dossier n'étant pas accessible (API
bloquée), le jeu récupéré peut être incomplet — les noms sont construits
dynamiquement dans `inversion._get_stan_model` (`Series`/`Parallel`/… ×
`_pos`/`_outliers`/`_fitY`/…). Les manquants pourront être ajoutés au besoin.
Les artefacts compilés (`.exe`, `.pkl`) ne sont pas vendorés : ils sont générés
localement par `cmdstanpy`/`cmdstan` lors du premier HMC.

### Modification appliquée (compat numpy ≥ 2) — UNIQUE changement

bayes-drt2 (2020) appelle `np.trapz`, **supprimé dans numpy 2.0** (renommé
`np.trapezoid`). Le projet, lui, utilise `np.trapezoid` (`fits/drt_fft.py`,
`fits/drt_tikhonov.py`) et **exige donc numpy ≥ 2** : un simple pin `numpy<2`
répare bayes-drt2 mais casse le projet (conflit mutuellement exclusif).

**Correctif retenu — remplacement en source** (préféré au monkeypatch global
`np.trapz = np.trapezoid`, qui muterait l'objet `numpy` partagé par tout le
process) : les 9 occurrences `np.trapz(` → `np.trapezoid(` dans
`matrices.py` (×5, dont 2 en commentaire) et `peak_fit.py` (×4).
`np.trapezoid` est un renommage strict de `np.trapz` (même signature, même
résultat numérique — vérifié : `ridge_fit` donne un τ de pic et un Rp
identiques avant/après). Le paquet vendoré requiert donc numpy ≥ 2, ce qui est
aligné avec le projet.

> Alternative équivalente (non retenue) : ajouter en tête de
> `bayes_drt2/__init__.py` :
> ```python
> import numpy as _np
> if not hasattr(_np, "trapz"):
>     _np.trapz = _np.trapezoid
> ```
> Ici `__init__.py` est laissé **vide** (identique à l'amont).

### Provenance (sha256 des fichiers vendorés)

Les fichiers non modifiés correspondent bit à bit à l'amont `raw` (vérifié).

```
# non modifiés (== amont) :
9a730ac429448c9f2815bbeed8fae7523c0cbe43e2f50054e0f356bd2cef9e70  inversion.py
effad4bc417ace086ad83c63f8694659b47d2a522d1c562c284a81be2a99ea83  utils.py
3dd071372eaf6dd234ba071874e3f4a83c1abde4a41321918351fcacba86aa94  plotting.py
c4f8fdfa5184877d171efbba6c1f596fd73927061cd6aeb91c03516e9edb1fd2  file_load.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  __init__.py  (vide)
e67efbead1ceb1c4e56f3aeadcd9a1e1be1e49010847363fae0c6b4f97f2d5f5  LICENSE

# patchés (np.trapz -> np.trapezoid) :
43aed2f5ec21935c1ad53375cded02bdbdcaaf9e31b86e82197316711d6fb2c2  matrices.py   (amont : f65803ac4f09a5903c902cc19489390dedc3d2f15c64ba7077dd748e223decdb)
c38b9c172fe88685a531a5650bbbf9992484de539807199c51cf4a067aeaabaf  peak_fit.py   (amont : 82c22bc77678f50e691631bf35fec63addc5e058199d8e195d7153702471b158)
```

### Usage

```python
from vendor.bayes_drt2.inversion import Inverter
inv = Inverter()
inv.ridge_fit(freq, Z, hyper_lambda=True)   # Z convention : Z'' < 0
#   Z = spectrum.Zre - 1j*spectrum.Zim      # le loader stocke Zim = -Im(Z) > 0
tau   = inv.distributions['DRT']['tau']
gamma = inv.predict_distribution('DRT', tau=tau)
Rp    = inv.predict_Rp()
```

`ridge_fit` ne nécessite **pas** cmdstan. Le HMC (`map_fit`/`bayes_fit`)
requiert `cmdstanpy` **et** une installation cmdstan (`install_cmdstan()`).

### Dépendances runtime

`numpy>=2`, `scipy`, `pandas`, `cvxopt`, `matplotlib` (+ `cmdstanpy` pour le HMC).
