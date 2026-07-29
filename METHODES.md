# METHODES.md — Recensement des méthodes numériques et formules de calcul

> **Objet.** Ce document recense **toutes** les méthodes numériques et **toutes** les
> formules utilisées dans un calcul, telles qu'elles sont **écrites dans le code réel**
> (pas dans la documentation d'architecture, réputée obsolète). Chaque formule est
> transcrite en notation lisible, accompagnée de l'extrait de code qui l'implémente
> et sourcée en `fichier:ligne`.
>
> **Nature du document.** Recensement factuel. Il ne corrige rien. Les divergences
> possibles entre implémentation et formule attendue sont isolées dans la section
> finale [« Points à vérifier »](#12-points-à-vérifier), séparée du recensement.
>
> **Note de périmètre.** Le cahier des charges citait `fits/drt_tikhonov.py`,
> `fits/drt_fft.py` et `core/cv_loader.py`. Les deux premiers **n'existent pas** dans
> le code : le calcul DRT est réalisé par `fits/drt_fit.py`, un *wrapper* du paquet
> vendoré `vendor/bayes_drt2/`. Ce document documente le code réellement présent.
> Les lignes citées correspondent à l'état du dépôt au moment de l'audit.

---

## 1. Table des matières

1. [Table des matières](#1-table-des-matières)
2. [Conventions transverses](#2-conventions-transverses)
3. [Prétraitement du signal — `core/loader.py`, `core/robust_loader.py`, `core/cv_loader.py`](#3-prétraitement-du-signal)
4. [Fonctions physiques partagées — `fits/physics.py`](#4-fonctions-physiques-partagées--fitsphysicspy)
5. [Fit du circuit de Randles — `fits/randles_full.py`](#5-fit-du-circuit-de-randles--fitsrandles_fullpy)
6. [Pondération du CNLS — `fits/weighting.py`](#6-pondération-du-cnls--fitsweightingpy)
7. [DRT — `fits/drt_fit.py` (+ `vendor/bayes_drt2`)](#7-drt--fitsdrt_fitpy--vendorbayes_drt2)
8. [Validation Kramers-Kronig — `fits/kk_validation.py`, `core/validator.py`](#8-validation-kramers-kronig)
9. [Calibration — `core/calibration.py`](#9-calibration--corecalibrationpy)
10. [Traitement CV — `core/cv_pipeline.py`, `core/cv_peaks.py`](#10-traitement-cv)
11. [Chaîne de calcul de bout en bout](#11-chaîne-de-calcul-de-bout-en-bout)
12. [Tableau récapitulatif](#12-tableau-récapitulatif)
13. [Points à vérifier](#13-points-à-vérifier)

---

## 2. Conventions transverses

Ces conventions sont partagées par plusieurs modules ; elles sont énoncées ici pour
éviter la répétition.

| Convention | Valeur dans le code | Source |
|---|---|---|
| Signe de l'imaginaire **stocké** (`EISSpectrum.Zim`) | `Zim = -Im(Z) > 0` (demi-cercle capacitif au-dessus de l'axe réel) | `core/robust_loader.py:41`, `fits/drt_fit.py:19-22` |
| Signe de l'imaginaire **physique** (modèle) | `Im(Z) < 0` pour un circuit R‑C dissipatif | `fits/randles_full.py:131-134` |
| Pulsation | `ω = 2·π·f` (rad/s), f en Hz | `fits/randles_full.py:100`, `fits/kk_validation.py:41` |
| Base des logarithmes en calibration | **log₁₀** (`np.log10`) | `core/calibration.py:80,105,154` |
| Tri des points EIS après nettoyage | HF → BF (fréquence décroissante) | `core/loader.py:164` |

---

## 3. Prétraitement du signal

### 3.1 Détection du délimiteur de colonnes — `core/robust_loader.py:_detect_delimiter:62`

Règle de priorité (pas une formule numérique, mais une décision qui conditionne le
décodage décimal) : tabulation → point‑virgule → virgule → découpage sur espaces.

```python
if "\t" in header_line: return "\t"
if ";"  in header_line: return ";"
if ","  in header_line: return ","      # en-tête séparée par virgules => CSV US
return None                             # None => str.split() sur espaces
```

**Conséquence numérique** (`robust_loader.py:139`) : `do_decimal_fix = delim != ","`.
Si le délimiteur **n'est pas** la virgule, les virgules décimales des données sont
converties en points (`s.replace(",", ".")`, ligne 169). Décodage FR ↔ US.

### 3.2 Correction du signe de Im(Z) — deux mécanismes distincts

**(a) Par nom de colonne — `core/robust_loader.py:_classify:110-113`.**
Le signe est déduit du **libellé** de la colonne :

```python
if "im(z)" in t:
    # "-im(z)" déjà positif ; "im(z)" à inverser
    sign = +1.0 if t.lstrip().startswith("-im(z)") or "-im(z)" in t else -1.0
    return "Zim", {"sign": sign}
```

Appliqué à chaque valeur ligne 172 : `val *= meta["sign"]` → convention de sortie
`-Im(Z) > 0`.

- Entrée : token d'en‑tête (str) + valeur brute mesurée.
- Sortie : `Zim` en convention `-Im(Z) > 0`.

**(b) Par heuristique de majorité — `core/loader.py:_clean_spectrum:160-162`.**
Utilisé **uniquement** sur le chemin de repli pandas (le parseur robuste passe
`correct_sign=False` pour éviter un double flip, `loader.py:224`) :

```python
if correct_sign and len(Zim) > 0 and np.sum(Zim < 0) > np.sum(Zim > 0):
    Zim = -Zim
```

> Formule de décision : `flip ⇔ card{Zim < 0} > card{Zim > 0}`.

### 3.3 Suppression des fréquences parasites — `core/loader.py:_clean_spectrum:155-157`

```python
for fp in parasitic_freqs:
    keep = np.abs(f - fp) > tol
    f, Zre, Zim = f[keep], Zre[keep], Zim[keep]
```

> Point conservé ⇔ `|f − f_p| > tol` pour **chaque** fréquence parasite `f_p`.

- `parasitic_freqs` : défaut `[50.0, 100.0]` Hz (`loader.py:16`, `config/default.yaml:28`).
- `tol` : défaut `3.0` Hz (`loader.py:17`, `config/default.yaml:29`).
- Provenance : config (`fit.n_freqs_parasites`, `fit.tol_parasites`).

### 3.4 Nettoyage et tri — `core/loader.py:_clean_spectrum:152-165`

```python
mask = np.isfinite(f) & np.isfinite(Zre) & np.isfinite(Zim)   # NaN/inf retirés
...
order = np.argsort(f)[::-1]   # décroissant : HF d'abord
return f[order], Zre[order], Zim[order]
```

Rejet si `len(f) < 5` après nettoyage (`loader.py:226`, `loader.py:276`).

### 3.5 Moyennage des réplicats EIS — `core/loader.py:average_replicates:294-358`

Moyenne point‑à‑point sur une grille de fréquence commune (celle du 1ᵉʳ spectre).
Interpolation linéaire des réplicats dont la grille diffère (`loader.py:322-328`) :

```python
Zre_interp = np.interp(f_ref, f_sorted, sp.Zre[sort_idx])
Zim_interp = np.interp(f_ref, f_sorted, sp.Zim[sort_idx])
```

Test d'égalité de grille : `len(sp.f)==len(f_ref) and np.allclose(sp.f, f_ref, rtol=0.01)`
(`loader.py:318`).

**Statistiques inter‑réplicats** (`loader.py:337-344`) — alimentent la pondération « sigma » :

$$\bar{Z}_{re}(f) = \frac{1}{N}\sum_i Z_{re}^{(i)}(f), \qquad
  \sigma_{re}(f) = \operatorname{std}_{\text{ddof}=1}\big(Z_{re}^{(i)}(f)\big)$$

(idem pour la partie imaginaire), puis **plancher** anti‑poids‑infini :

$$\sigma \leftarrow \max\!\big(\sigma,\; 0{,}001 \cdot |\bar{Z}|\big),
  \qquad |\bar{Z}| = \sqrt{\bar{Z}_{re}^2 + \bar{Z}_{im}^2}$$

```python
sigma_re = np.std(Zre_stack, axis=0, ddof=1)
sigma_im = np.std(Zim_stack, axis=0, ddof=1)
Zmod_mean = np.sqrt(Zre_mean ** 2 + Zim_mean ** 2)
floor = 0.001 * Zmod_mean
sigma_re = np.maximum(sigma_re, floor)
sigma_im = np.maximum(sigma_im, floor)
```

- **Convention** : `ddof=1` (estimateur non biaisé). Plancher = 0,1 % du module moyen.
- **Référence citée dans le code** : « approche Measurement Model, Orazem »
  (`loader.py:333`).

### 3.6 Prétraitement CV — `core/cv_loader.py`

**Conversion d'unités de courant** (`cv_loader.py:107-111`, repli pandas) :

```python
if "ma" in col_lower:   I = I * 1e-3
elif "µa" in col_lower or "ua" in col_lower:   I = I * 1e-6
```

Sur le chemin robuste, la conversion utilise la table
`_I_UNIT_TO_A = {"a":1.0, "ma":1e-3, "ua":1e-6, "µa":1e-6, "na":1e-9}`
(`robust_loader.py:23`, appliquée `robust_loader.py:174` : `val *= _I_UNIT_TO_A[unit]`).

**Tri** : par potentiel croissant, `order = np.argsort(E)` (`cv_loader.py:118`).

**Moyennage CV** (`cv_loader.py:average_cv_replicates:151-173`) : interpolation de tous
les scans sur la grille `E` du premier, puis moyenne :

```python
I_matrix = np.stack([np.interp(E_grid, s.E, s.I) for s in scans], axis=0)
I_mean = I_matrix.mean(axis=0)
```

---

## 4. Fonctions physiques partagées — `fits/physics.py`

### 4.1 Impédance de diffusion bornée `Z_D` — `physics.py:Z_D:10-25`

$$Z_D(\omega) = R_D \cdot \frac{\tanh\!\big(\sqrt{j\,\omega\,\tau_d}\big)}{\sqrt{j\,\omega\,\tau_d}},
  \qquad \lim_{\omega\to 0} Z_D = R_D$$

```python
x = np.sqrt(1j * omega * tau_d)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(np.abs(x) < 1e-8, 1.0, np.tanh(x) / x)
return R_D * ratio
```

- Entrées : `omega` (rad/s, calculé), `R_D` (Ω, fit), `tau_d` (s, fit).
- Sortie : impédance complexe (Ω).
- **Méthode numérique** : limite `tanh(x)/x → 1` pour `|x| < 1e-8` (règle de L'Hôpital,
  évite 0/0 en BF).
- **Référence citée** : formulation « bounded‑diffusion (Bissessur) pour canal
  microfluidique » (`physics.py:3-4, 11`).

### 4.2 Circuit de Randles complet `Z_randles_full` — `physics.py:Z_randles_full:28-63`

$$Z_{eq} = R'_e + \frac{R_{ct} + Z_D}{1 + Q_{dl}\,(j\omega)^{\alpha}\,(R_{ct} + Z_D)},
  \qquad
  Z = R_e + \frac{Z_{eq}}{1 + j\omega\,C_b\,Z_{eq}}$$

```python
jw = 1j * omega
Zd = Z_D(omega, R_D, tau_d)
num = Rct + Zd
denom = 1.0 + Qdl * (jw ** alpha) * (Rct + Zd)
Z_eq = Re_prime + num / denom
return Re + Z_eq / (1.0 + jw * Cb * Z_eq)
```

- Élément CPE implicite : le terme `Q_{dl}·(jω)^α` est l'admittance d'un CPE
  (`Z_CPE = 1 / (Q·(jω)^α)`), placé en parallèle sur la branche `R_ct + Z_D`.
- Entrées : 8 paramètres (voir §5), tous issus du fit ; `omega` calculé.
- Sortie : `Z` complexe (Ω), convention physique `Im(Z) < 0`.

### 4.3 Rct théorique électrode nue `Rct_bare_theory` — `physics.py:Rct_bare_theory:66-81`

$$R_{ct}^{\text{bare}} = \frac{R\,T}{F^{2}\,n\,S\,k^{0}\,C_{0}}$$

```python
R_gas = 8.314
F = 96485.0
n = 1
return (R_gas * T) / (F ** 2 * n * S * k0 * C0)
```

- **Constantes en dur** : `R = 8.314`, `F = 96485.0`, `n = 1` (⚠ dupliquées avec
  `config/default.yaml:physics` — voir §13).
- Entrées : `T` (K), `S` (m²), `k0` (m/s), `C0` (mol/m³).
- Sortie : Rct (Ω).
- **Référence citée** : « Butler‑Volmer kinetics » (`physics.py:67`).
- **Statut** : fonction **non appelée** par le pipeline principal (uniquement par les
  tests — voir §13).

### 4.4 Capacité de double couche effective `Cdl_brug` — `physics.py:Cdl_brug:84-97`

$$C_{dl}^{eq} = \left[\, Q_{dl}\left(\frac{1}{R_e + R'_e} + \frac{1}{R_{ct}}\right)^{\alpha - 1} \right]^{1/\alpha}$$

```python
return (Qdl * (1.0 / (Re + Re_prime) + 1.0 / Rct) ** (alpha - 1.0)) ** (1.0 / alpha)
```

- **Référence citée** : « Brug formula » (`physics.py:85`).
- **Statut** : fonction **non appelée** par le pipeline principal (voir §13).

### 4.5 Taux de recouvrement `theta_EIS` — `physics.py:theta_EIS:100-110`

$$\theta = 1 - \frac{R_{ct}^{\text{bare}}}{R_{ct}^{\text{ap}}}$$

```python
return 1.0 - Rct_bare / Rct_ap
```

- Entrées : `Rct_bare`, `Rct_ap` (Ω).
- Sortie : recouvrement (0 à 1).
- **Statut** : fonction **non appelée** par le pipeline principal (voir §13).

---

## 5. Fit du circuit de Randles — `fits/randles_full.py`

### 5.1 Paramètres, valeurs initiales, bornes

**Paramètres libres (8)** (`randles_full.py:11`) :
`Re, Re_prime, Cb, Rct, Qdl, alpha, R_D, tau_d`.

**Valeurs initiales** (`randles_full.py:initial_guess:34-60`) — dérivées des extrema
du spectre :

```python
Re_est  = max(min(spectrum.Zre), 100.0)
Rct_est = max(max(spectrum.Zre) - Re_est, 500.0)
tau_d_est = 1.0 / (2.0 * np.pi * f_min)          # inverse du coude BF
# Re_prime=max(Re_est*0.05,1.0), Cb=1e-9, Qdl=1e-6, alpha=0.85, R_D=0.2*Rct_est
```

> `R_e^{(0)} = \max(\min Z_{re}, 100)`, `R_{ct}^{(0)} = \max(\max Z_{re} - R_e^{(0)}, 500)`,
> `\tau_d^{(0)} = 1/(2\pi f_{min})`.

**Bornes** (`randles_full.py:bounds:62-88`) : lues dans `config.fit.bounds_randles_full`,
sinon défauts codés. Extrait des défauts (`config/default.yaml:32-40`) :

| Param | Borne basse | Borne haute |
|---|---|---|
| Re | 100 | 1e5 |
| Re_prime | 1 | 1e5 |
| Cb | 1e-12 | 1e-4 |
| Rct | 100 | 1e9 |
| Qdl | 1e-12 | 1e-4 |
| alpha | 0.6 | 1.0 |
| R_D | 10 | 1e6 |
| tau_d | 1e-4 | 1e3 |

### 5.2 Résidus pondérés (CNLS complexe) — `randles_full.py:residuals:126-135`

$$r = \Big[\; (Z_{re}^{\text{fit}} - Z_{re}^{\text{exp}})\cdot\sqrt{w_{re}}\;;\;\;
        (-Z_{im}^{\text{fit}} - Z_{im}^{\text{exp}})\cdot\sqrt{w_{im}}\;\Big]$$

```python
Z = Z_randles_full(omega, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d)
return np.concatenate([
    (Z.real - spectrum.Zre) * sw_re,
    (-Z.imag - spectrum.Zim) * sw_im,   # -Z.imag comparé à Zim (>0)
])
```

- `sw_re = √w_re`, `sw_im = √w_im` (`randles_full.py:123-124`).
- **Convention de signe** : `spectrum.Zim` est en convention `-Im(Z) > 0`, `Z.imag`
  est l'imaginaire physique (< 0) ; le résidu compare donc `-Z.imag` à `spectrum.Zim`.

### 5.3 Optimiseur — `randles_full.py:fit:140-144`

```python
result = least_squares(
    residuals, x0, bounds=(blo, bhi),
    max_nfev=max_iter, method="trf",
    ftol=1e-10, xtol=1e-10,
)
```

- **Algorithme** : `scipy.optimize.least_squares`, méthode **TRF**
  (Trust Region Reflective, gère les bornes).
- **Critères d'arrêt** : `ftol=1e-10`, `xtol=1e-10`, `max_nfev = config.fit.max_iter`
  (défaut 10000, `config/default.yaml:30`).
- Repli en cas d'exception : `x_fit = x0`, `converged = False` (`randles_full.py:165-168`).

### 5.4 Covariance et écarts‑types des paramètres — `randles_full.py:155-163`

$$\operatorname{cov} = (J^{\top}J)^{-1} \times
  \begin{cases}
    1 & \text{mode « sigma » } (\text{absolute\_sigma=True})\\[4pt]
    \dfrac{2\,\text{cost}}{2N - P} & \text{mode « modulus » (rééchelonné)}
  \end{cases}
  \qquad \sigma_k = \sqrt{|\operatorname{cov}_{kk}|}$$

```python
dof = max(2 * len(spectrum.f) - len(_PARAM_NAMES), 1)
cov = np.linalg.inv(J.T @ J)
if not absolute_sigma:
    cov = cov * (2.0 * result.cost / dof)
std = np.sqrt(np.abs(np.diag(cov)))
```

- `J = result.jac` : jacobienne **pondérée** au point solution.
- `result.cost = ½·Σr²` → la SSR pondérée vaut `2·result.cost` (d'où le facteur 2,
  commenté « fix √2 », `randles_full.py:154`).
- `N = len(spectrum.f)`, `P = 8` ; degrés de liberté `2N − P`.
- Repli `LinAlgError` → écarts‑types nuls (`randles_full.py:162-163`).

### 5.5 χ² réduit pondéré — `randles_full.py:185-186`

$$\chi^2_{\text{red}} = \frac{1}{2N-P}\sum_i
  \big(w_{re,i}\,\Delta_{re,i}^2 + w_{im,i}\,\Delta_{im,i}^2\big)$$

avec `res_re = Zre − Z_fit.real` et `res_im = Zim + Z_fit.imag` (`randles_full.py:176-177`).

```python
res_re = spectrum.Zre - Z_fit.real
res_im = spectrum.Zim + Z_fit.imag
dof = max(2 * len(spectrum.f) - len(_PARAM_NAMES), 1)
chi2_reduced = float(np.sum(w_re * res_re ** 2 + w_im * res_im ** 2) / dof)
```

- **Interprétation** (commentée `randles_full.py:181-184`) : `χ²_red ≈ 1` est un test
  d'adéquation **uniquement** en mode « sigma » (poids = 1/σ² mesurés) ; en mode
  « modulus » rééchelonné, ce n'est qu'une métrique de misfit relative.

### 5.6 Résidu relatif RMS (diagnostic) — `randles_full.py:189-190`

$$\text{rel\_residual} = \sqrt{\frac{1}{N}\sum_i
   \frac{\Delta_{re,i}^2 + \Delta_{im,i}^2}{Z_{re,i}^2 + Z_{im,i}^2 + 10^{-30}}}$$

```python
Zmod2 = spectrum.Zre ** 2 + spectrum.Zim ** 2 + 1e-30
rel_residual = float(np.sqrt(np.mean((res_re ** 2 + res_im ** 2) / Zmod2)))
```

Sert au garde‑fou `_REL_RESIDUAL_WARN = 0.10` (résidu relatif > 10 % → « ajustement
médiocre », `randles_full.py:14, 195-199`) et est exporté comme
`reconstruction_error` (`randles_full.py:219`).

### 5.7 Diagnostics de bornes — `randles_full.py:200-205`

Un paramètre est signalé « collé à la borne » si
`|value − borne| ≤ 0.01 · max(|borne|, 1e-30)` (`_BOUND_PROXIMITY = 0.01`,
`randles_full.py:15`).

---

## 6. Pondération du CNLS — `fits/weighting.py`

Deux modes sélectionnés par `config.fit.weight_mode` (`weighting.py:resolve_weights:24-56`).

### 6.1 Mode « modulus » (défaut) — `weighting.py:52-56`

$$w_{re} = w_{im} = \frac{1}{(\alpha_{\text{noise}}\cdot |Z|)^2},
  \qquad |Z| = \sqrt{Z_{re}^2 + Z_{im}^2}$$

```python
alpha = float(fit_cfg.get("alpha_noise", 0.001))
Zmod = np.sqrt(spectrum.Zre**2 + spectrum.Zim**2)
w = 1.0 / (alpha * Zmod) ** 2
return w, w, False        # absolute_sigma = False
```

- `alpha_noise` : niveau de bruit relatif **arbitraire**, défaut `0.001`
  (`config/default.yaml:27`). Les poids **ne sont pas** de vraies 1/variance →
  `absolute_sigma = False`.

### 6.2 Mode « sigma » (Measurement Model, Orazem) — `weighting.py:44-49`

$$w_{re} = \frac{1}{\sigma_{re}^2}, \qquad w_{im} = \frac{1}{\sigma_{im}^2}$$

```python
sre = np.asarray(sigma_re, dtype=float)
sim = np.asarray(sigma_im, dtype=float)
w_re = 1.0 / sre ** 2
w_im = 1.0 / sim ** 2
return w_re, w_im, True    # absolute_sigma = True
```

- `σ_re(f)`, `σ_im(f)` proviennent du moyennage inter‑réplicats (§3.5). Nécessite ≥ 2
  réplicats ; sinon **repli automatique** sur « modulus » (`weighting.py:44`, condition
  `mode == "sigma" and has_sigma`).
- **Référence citée** : « Measurement Model, Orazem » (`weighting.py:10`).

---

## 7. DRT — `fits/drt_fit.py` (+ `vendor/bayes_drt2`)

> **Nature.** `fits/drt_fit.py` expose le plugin `DRTBayesModel(BaseFitModel)`
> (`name="drt_bayes"`), *wrapper* de la classe `Inverter` du paquet **tiers vendoré**
> `vendor/bayes_drt2/` (bayes_drt2, Jake Huang, Colorado School of Mines, v0.2 —
> `vendor/README.md`). C'est **l'unique** moteur DRT de l'app : plus aucune
> implémentation Tikhonov / NNLS / L-curve / ridge ne subsiste. Le plugin est
> découvert par `fits/registry.py` et lancé par `core/pipeline.py` comme les autres
> fits.
>
> **Deux modes, un seul paquet de calcul.**
> - `mode='optimize'` — estimation du **maximum a posteriori** (MAP, L-BFGS-B).
>   **Défaut** : calculé par le pipeline sur chaque spectre **moyenné**. Pas
>   d'intervalles (`drt_gamma_lo/hi = None`).
> - `mode='sample'` — **HMC bayésien**, avec **intervalles de crédibilité**
>   (`drt_gamma_lo/hi`). Jamais automatique : uniquement à la demande via
>   `core.pipeline.recompute_drt(session, spectrum_id, config, mode='sample')`.
>
> Les **deux** modes compilent un modèle Stan via CmdStan : la toolchain C++ est un
> prérequis (installée automatiquement au premier lancement par
> `setup_drt_bayesien.ensure_drt_ready()` → `install_cmdstan(compiler=True)`). Il n'y a
> plus de chemin sans compilation.
>
> **Patch numpy ≥ 2 du vendored.** bayes_drt2 (2020) appelait `np.trapz`, supprimé en
> numpy 2.0 (renommé `np.trapezoid`). Le seul correctif appliqué au paquet vendoré est le
> remplacement en source des occurrences `np.trapz(` → `np.trapezoid(` dans `matrices.py`
> et `peak_fit.py` — `vendor/README.md` § « Modification appliquée ».

### 7.1 Convention d'impédance et tri des fréquences — `drt_fit.py::fit_drt`

Le loader stocke `Zim = -Im(Z) > 0` ; bayes_drt2 attend la convention physique
`Im(Z) < 0` :

$$Z = Z_{re} - j\,Z_{im}\quad(\text{car } Z_{im} = -\text{Im}(Z) > 0)$$

```python
Z = spectrum.Zre - 1j * spectrum.Zim
order = np.argsort(freq)[::-1]      # tri HF→BF sur les VRAIES fréquences lues
inv.fit(freq[order], Z[order], mode=mode)
```

Les fréquences ne sont **jamais** reconstruites par `np.logspace` : bayes_drt2 place
les pics en τ à partir des fréquences mesurées (une décade au-delà de chaque borne,
10 points/décade), la position des pics en dépend directement.

### 7.2 Récupération de γ(τ) — `drt_fit.py::fit_drt`

```python
inv = Inverter()
inv.fit(freq, Z, mode=mode)                              # 'optimize' (défaut) ou 'sample'
tau   = inv.distributions["DRT"]["tau"]                  # garde-fou KeyError
gamma = inv.predict_distribution("DRT")                  # médiane a posteriori
if mode == "sample":                                      # intervalles HMC uniquement
    gamma_lo = inv.predict_distribution("DRT", tau=tau, percentile=2.5)
    gamma_hi = inv.predict_distribution("DRT", tau=tau, percentile=97.5)
```

- **Garde-fou** : l'accès à `inv.distributions["DRT"]` est protégé par `try/except
  KeyError`, qui relève une erreur explicite inspectant `inv.distributions.keys()`.
- **Régularisation** : entièrement déterminée dans `vendor/bayes_drt2` (le wrapper
  n'active aucun preset ni λ manuel).
- **Référence** : lignée hiérarchique bayésienne DRT (Ciucci & Chen) telle
  qu'implémentée dans bayes_drt2.

### 7.3 Rct (grandeur de calibration) — extraction par pic, convention Bissessur

`FitResult.Rct` de la DRT est le **Rct de l'arc de transfert de charge**, pas la
résistance de polarisation totale. Il est extrait de γ(τ) (`_extract_rct_peak`) :

1. Détection des maxima locaux de γ (`_local_maxima`, fenêtre glissante, seuil relatif).
2. Sélection du **pic pénultième** (avant-dernier maximum ⇒ arc de transfert, la queue
   de diffusion BF étant le dernier pic) ; s'il n'y a qu'un pic, il est retenu **et
   signalé**.
3. $R_{ct} = \int \gamma\,\mathrm{d}\ln\tau$ par trapèze (`np.trapezoid`) sur **±3 en
   ln(τ)** autour du pic.

$$R_{ct} = \int_{\ln\tau_{pic}-3}^{\ln\tau_{pic}+3} \gamma(\tau)\,\mathrm{d}\ln\tau$$

- **Repli signalé** : si aucun pic n'est exploitable, `Rct` retombe sur `Rp` (aire
  totale sous γ, diffusion + transfert), **toujours** signalé via
  `params['rct_source'] = 'rp_fallback'` **et** `FitResult.warnings`, pour que la
  calibration $\theta_{EIS} = 1 - R_{ct,bare}/R_{ct,ap}$ ne soit jamais calculée en
  silence sur une grandeur différente. `params['rct_source']` vaut sinon
  `'peak_penultimate'` ou `'peak_single'`.
- **Rp (aire totale)** — `inv.predict_Rp()`, somme analytique des coefficients RBF
  (× √π/ε), stockée dans `params['Rp']` : c'est l'aire sous **toute** la DRT, distincte
  du Rct d'arc ci-dessus.

### 7.4 Reconstruction et métriques — `drt_fit.py::_reconstruction_metrics`

`Zfit` est reconstruit par `inv.predict_Z(freq)` (reconverti en `Zim = -Im(Z) > 0`).
Modèle-libre : pas de χ² réduit pondéré classique, mais un misfit normalisé par le
module, stocké dans `FitResult.chi2_reduced` :

$$\chi^2_{\text{norm}} = \frac{1}{N}\sum_k \frac{\Delta_{re,k}^2 + \Delta_{im,k}^2}{|Z_k|^2},\qquad
\varepsilon_{recon} = \frac{1}{N}\sum_k \frac{\sqrt{\Delta_{re,k}^2 + \Delta_{im,k}^2}}{|Z_k|}$$

### 7.5 Convention de tracé — `plotting/eis_plots.py::drt_figure`

La figure DRT trace $\ln(\gamma/\gamma_0)$ en fonction de $\ln(\tau/\tau_0)$ —
**logarithme népérien** (base e, `np.log`), **jamais** `log10` — avec les constantes
de normalisation dimensionnelle **explicites** $\gamma_0 = 1\,\Omega$ et
$\tau_0 = 1\,\mathrm{s}$ (`DRT_GAMMA0_OHM`, `DRT_TAU0_S`). Elles valent 1 (donc ne
changent pas la valeur numérique) mais adimensionnalisent l'argument du log et
figurent sur les axes. Un **badge de mode** distingue « DRT MAP (optimize) » de
« DRT bayésienne (sample) » ; en mode `sample`, la bande d'incertitude
(`drt_gamma_lo/hi`) est transformée de la même façon et identifiée dans la légende.

### 7.6 Résultat — `FitResult` (core/models.py)

Le plugin renvoie un `FitResult` standard (comme les autres modèles) : `drt_tau`,
`drt_gamma`, `drt_mode` (`'optimize'`/`'sample'`), `drt_gamma_lo/hi`, `Rct` (arc),
`Zfit_re/im`, `reconstruction_error`, `params['rct_source']`. La DRT est calculée par
le pipeline sur les spectres **moyennés** uniquement ; un réplicat peut être recalculé
à la demande via `recompute_drt`.

## 8. Validation Kramers-Kronig

### 8.1 Circuit de Voigt linéaire (Lin‑KK) — `fits/kk_validation.py:lin_kk:22-76`

Modèle ajusté :

$$Z_{\text{fit}}(j\omega) = R_0 + \sum_{k=1}^{M} \frac{R_k}{1 + j\omega\tau_k}
   \;\big[\; + \tfrac{1}{j\omega C} \text{ si add\_cap}\;\big]$$

**Grille des τ** (fixée, pas d'optimisation non linéaire) :

$$\tau_k = \text{geomspace}\!\left(\frac{1}{\omega_{max}}, \frac{1}{\omega_{min}}, M\right),
  \qquad M = \min\!\big(\text{max\_M},\ \max(2,\ \lfloor 2n/3\rfloor)\big)$$

```python
M = int(min(max_M, max(2, 2 * n // 3)))
tau = np.geomspace(1.0 / omega.max(), 1.0 / omega.min(), M)
```

**Construction des matrices de conception** (`kk_validation.py:47-59`) — colonne 0 = R₀,
colonnes 1..M = éléments de Voigt, colonne finale optionnelle = capacité série :

```python
A_re[:, 0] = 1.0 ;  A_im[:, 0] = 0.0
for k in range(M):
    wt = omega * tau[k]
    denom = 1.0 + wt ** 2
    A_re[:, k+1] = 1.0 / denom          #  Re: 1/(1+(ωτ)²)
    A_im[:, k+1] = -wt / denom          #  Im: -ωτ/(1+(ωτ)²)
if add_cap:
    A_im = ... , -1.0 / omega           #  terme capacitif série -1/(ωC)
```

**Résolution** — moindres carrés linéaires sur Re/Im empilés (`kk_validation.py:61-63`) :

$$x = \arg\min_x \|A x - b\|_2^2, \qquad
  A = \begin{bmatrix} A_{re}\\ A_{im}\end{bmatrix},\
  b = \begin{bmatrix} Z_{re}\\ Z_{im}\end{bmatrix}$$

```python
A = np.vstack([A_re, A_im]); b = np.concatenate([Z.real, Z.imag])
x, *_ = np.linalg.lstsq(A, b, rcond=None)
```

- **Méthode** : `np.linalg.lstsq` (linéaire, pas d'itération sur M).
- **Références citées** : Boukamp 1995 / Schönleber et al. 2014 (`kk_validation.py:11`).

### 8.2 Score µ de Schönleber — `kk_validation.py:72-74`

$$\mu = \frac{\sum_{k : R_k < 0} |R_k|}{\sum_k |R_k|}$$

```python
Rk = x[1:M + 1]
total = np.sum(np.abs(Rk))
mu = float(np.sum(np.abs(Rk[Rk < 0])) / total) if total > 0 else 0.0
```

- **Interprétation** : masse des `R_k` négatifs / masse totale. µ élevé ⇒ sur‑ajustement.
- **Note** : le paramètre `c` (critère µ) de l'interface `linKK` **n'est pas utilisé**
  (pas de recherche itérative de M) — conservé pour compatibilité de signature
  (`kk_validation.py:126-132`).

### 8.3 Résidu KK relatif et verdict — `kk_validation.py:kramers_kronig_check:86-123`

$$\text{rel\_residual}_i = \frac{\sqrt{r_{re,i}^2 + r_{im,i}^2}}{|Z_i|},
  \qquad \text{kk\_passed} = \big(\max_i \text{rel\_residual}_i < \text{tol}\big)$$

```python
Zmod = np.where(Zmod > 0, Zmod, 1e-30)
rel_residual = np.sqrt(res_re ** 2 + res_im ** 2) / Zmod
max_residual = float(np.max(rel_residual))
kk_passed = bool(max_residual < tol)
```

- `tol = config.fit.drt_kk_tol`, défaut `0.05` (`config/default.yaml:48`).
- Convention d'entrée : `Z = Zre − j·|Zim|` (force `Im(Z) < 0`, `kk_validation.py:103`).

### 8.4 Validation de spectre et de groupe — `core/validator.py`

**Résidus normalisés en %** (`validator.py:157-159`) :

$$r_{re}\%= \frac{r_{re}}{|Z_{exp}|}\times 100, \qquad r_{im}\% = \frac{r_{im}}{|Z_{exp}|}\times 100$$

**χ² pseudo** (`validator.py:162`) : `chi2_pseudo = mean(r_re%² + r_im%²)`.

**Masque de validité** (`validator.py:165-166`) : point valide ⇔
`|r_re%| < seuil ET |r_im%| < seuil`, avec `RESIDUAL_THRESHOLD_PCT = 2.0`
(`validator.py:85`).

**Seuils de verdict** (`validator.py:82-93, 174-194`) :

| Grandeur | Seuil | Effet |
|---|---|---|
| `MU_THRESHOLD` | 0.85 | µ > 0.85 ⇒ invalide (sur‑ajustement) |
| `INVALID_FRACTION_WARN` | 0.10 | ≥ 10 % points hors tol ⇒ avertissement |
| `INVALID_FRACTION_REJECT` | 0.25 | ≥ 25 % points hors tol ⇒ invalide |
| `DRIFT_CV_THRESHOLD` | 0.5 | CV des résidus inter‑réplicats |

- **Référence citée** : seuil µ de Schönleber 2014 (`validator.py:81`).
- **Note** : `validate_spectrum` appelle `linKK(..., c=0.85, max_M=100, fit_type="complex", add_cap=True)` (`validator.py:132-139`) ; `c` est ignoré (§8.2).

**Plage KK‑valide** (`validator.py:_find_valid_range:216-244`) : plus long bloc contigu de
points valides (algorithme de plus long *run* de `True`).

**Plage commune d'un groupe** (`validator.py:300-301`) — intersection :

$$f_{min}^{\text{commun}} = \max_i f_{min}^{(i)}, \qquad
  f_{max}^{\text{commun}} = \min_i f_{max}^{(i)}$$

**Détection de drift inter‑réplicats** (`validator.py:_detect_drift:328-368`) :
coefficient de variation des résidus KK Im, interpolés sur grille log commune :

$$\text{CV}(f) = \begin{cases}\dfrac{\operatorname{std}_i r_{im}^{(i)}(f)}{\operatorname{mean}_i |r_{im}^{(i)}(f)|} & \text{si mean} > 0.1\\ 0 & \text{sinon}\end{cases}$$

Drift signalé si `fraction{CV > cv_threshold} > 0.20` (`validator.py:359`).

```python
res_interp = np.interp(np.log10(f_ref), np.log10(kk.frequencies), kk.residuals_im)
...
cv = np.where(mean_res > 0.1, std_res / mean_res, 0.0)
fraction_drifted = (cv > cv_threshold).mean()
```

**σ empirique inter‑réplicats** (`validator.py:_compute_sigma:371-402`) : identique en forme
à §3.5 (`np.std(..., ddof=1)`, plancher `0.001·|Z̄|`). Interpolation sur la grille du 1ᵉʳ
réplicat.

---

## 9. Calibration — `core/calibration.py`

### 9.1 Signal normalisé vs log₁₀([c]) — `calibration.py:compute_calibration:58-89`

$$\text{signal}_i = \frac{\big|\,R_{ct}^{\text{probe}} - R_{ct,i}\,\big|}{\big|\,R_{ct}^{\text{probe}}\,\big|},
  \qquad y = \text{signal},\ \ x = \log_{10}([c])$$

Régression linéaire OLS : `slope, intercept, r, p, stderr = stats.linregress(x, y)`.

```python
signal = [abs(probe_rct - r) / abs(probe_rct) for r in rcts]
log_c = np.log10(concs)
reg = stats.linregress(log_c, signal)
# r2 = reg.rvalue ** 2
```

- Entrées : `R_ct` par concentration (issu du fit, §5), `R_ct^probe` (fit du probe).
- Points retenus : `concentration > 0` **et** `fit.Rct > 0`, ≥ 2 points
  (`calibration.py:42-55, 76`).
- Sortie : pente, ordonnée, `r2 = rvalue²`, p‑value, stderr.

### 9.2 Calibration log‑log (DRT) — `calibration.py:compute_calibration_loglog:92-115`

$$\log_{10}(R_{ct}) = a\cdot \log_{10}([c]) + b$$

```python
log_c = np.log10(concs)
log_rct = np.log10(rcts)
reg = stats.linregress(log_c, log_rct)
```

- `probe_rct = NaN` pour cette variante.

### 9.3 Calibration CV — `calibration.py:compute_cv_calibration:133-161`

$$\bar{s}_c = \operatorname{nanmean}\big(\text{delta\_signal}_c\big),
  \qquad \bar{s} = a\cdot\log_{10}([c]) + b$$

```python
mean_sig = np.nanmean(grp.delta_signal)
...
log_c = np.log10(concs)
reg = stats.linregress(log_c, signals)
```

- Retient les concentrations `> 0` avec moyenne finie ; ≥ 2 points.

---

## 10. Traitement CV

### 10.1 Signal différentiel `delta_signal` — `core/cv_pipeline.py:45-52`

$$\Delta(E) = \frac{\big|\,I_{\text{probe}}^{\text{interp}}(E) - I_c(E)\,\big|}{\big|\,I_{\text{probe}}^{\text{interp}}(E)\,\big|},
  \qquad \Delta = \text{NaN si } I_{\text{probe}}^{\text{interp}} = 0$$

```python
I_probe_interp = np.interp(avg_scan.E, probe_scan.E, probe_scan.I)
with np.errstate(invalid="ignore", divide="ignore"):
    delta = np.abs(I_probe_interp - avg_scan.I) / np.abs(I_probe_interp)
    delta = np.where(I_probe_interp == 0, np.nan, delta)
```

- Le courant probe est interpolé sur la grille `E` de la concentration (`np.interp`).
- Alimente la calibration CV (§9.3).

### 10.2 Détection des pics redox — `core/cv_peaks.py:detect_redox_peaks:18-61`

**Lissage** Savitzky‑Golay puis extrema globaux :

```python
I_smooth = savgol_filter(I, window_length=wl, polyorder=min(polyorder, wl-1))
idx_a = int(np.argmax(I_smooth))     # pic anodique
idx_c = int(np.argmin(I_smooth))     # pic cathodique
```

$$I_{pa} = I[\arg\max \tilde{I}],\quad I_{pc} = I[\arg\min \tilde{I}],\quad
  \Delta E_p = E_{pa} - E_{pc}$$

- **Méthode** : filtre Savitzky‑Golay (`window_length=9`, `polyorder=3` par défaut,
  fenêtre réduite si scan court, `cv_peaks.py:40-49`). Extremums lus sur le courant
  **lissé**, valeurs `I`/`E` reportées sur le courant **brut**.
- **Convention** : approche « un seul couple redox » (max global / min global), pas de
  détection multi‑pics.

---

## 11. Chaîne de calcul de bout en bout

Vue globale reliant chaque formule à l'étape suivante (branche EIS → Rct → calibration).

```
Fichier EC-Lab brut (f, Re(Z), Im(Z) ; virgule décimale, unités, signe variable)
        │
        │  robust_loader.parse_eclab_file
        │   • détection délimiteur  .................... §3.1  robust_loader.py:62
        │   • fix décimal virgule→point  ............... §3.1  robust_loader.py:169
        │   • signe Im(Z) par nom de colonne  ......... §3.2a robust_loader.py:110
        │   • unités courant → A (CV)  ................. §3.6  robust_loader.py:174
        ▼
loader.load_spectrum / _clean_spectrum
        │   • NaN/inf retirés  ......................... §3.4  loader.py:152
        │   • suppression 50/100 Hz (|f−f_p|>tol)  .... §3.3  loader.py:155
        │   • (repli) signe par majorité  ............. §3.2b loader.py:160
        │   • tri HF→BF  .............................. §3.4  loader.py:164
        │   • rejet si < 5 points
        ▼
average_replicates  (si ≥ 2 réplicats)
        │   • moyenne point-à-point (interp si grille ≠)  §3.5  loader.py:337
        │   • σ_re(f), σ_im(f) ddof=1, plancher 0.1%|Z̄|  §3.5  loader.py:339-344
        ▼
      ┌─────────────────────────────┬─────────────────────────────┐
      ▼                             ▼                             ▼
validate_replicate_group      resolve_weights                  (DRT, branche //)
 • Lin-KK Voigt  ...... §8.1   • modulus 1/(α|Z|)² .. §6.1     _impedance Z=Zre−jZim §7.1
 • µ Schönleber  ...... §8.2   • sigma  1/σ²  ....... §6.2     fit(mode=optimize) MAP  §7.2
 • résidu %, verdict .. §8.4     (choisit absolute_sigma)      Rct pic (Bissessur) ... §7.3
 • drift inter-rép.  .. §8.4                                   sample (HMC) à la demande §7
                                                              Rp=Σcoef·√π/ε (RBF)  . §7.3
                                    │
                                    ▼
                       RandlesFullModel.fit
                        • Z_randles_full(ω,·)  ....... §4.2  physics.py:28
                        •   dont Z_D bounded diff.  .. §4.1  physics.py:10
                        • résidus pondérés (CNLS)  ... §5.2  randles_full.py:126
                        • least_squares TRF  ......... §5.3  randles_full.py:140
                        • cov = (JᵀJ)⁻¹ [×χ²red]  .... §5.4  randles_full.py:155
                        • χ²_red, rel_residual  ...... §5.5-5.6
                                    │
                                    ▼   Rct (Ω) par concentration
                       compute_calibration(_loglog)
                        • signal = |ΔRct|/Rct_probe .. §9.1  calibration.py:79
                        •   ou log10(Rct)  ........... §9.2  calibration.py:106
                        • linregress vs log10([c])  .. §9   calibration.py:81
                                    ▼
                       pente / ordonnée / R² / p / stderr  →  courbe de calibration
```

**Branche CV (parallèle)** : `cv_loader` → `average_cv_replicates` (§3.6) →
`delta_signal = |I_probe−I_c|/|I_probe|` (§10.1) → `compute_cv_calibration`
(moyenne vs log₁₀[c], §9.3). Pics redox (§10.2) calculés à part pour affichage.

**Fonctions physiques hors chaîne** : `Rct_bare_theory` (§4.3), `Cdl_brug` (§4.4),
`theta_EIS` (§4.5) sont définies mais **non branchées** dans le pipeline (voir §13).

---

## 12. Tableau récapitulatif

| Grandeur | Formule | Fichier:ligne | Méthode | Réf citée |
|---|---|---|---|---|
| Fix décimal | `,`→`.` si delim≠`,` | robust_loader.py:169 | conditionnel | — |
| Signe Im(Z) (nom) | `sign=+1 si "-im(z)" sinon −1` | robust_loader.py:110-113 | par libellé | — |
| Signe Im(Z) (majorité) | flip si `#{Zim<0}>#{Zim>0}` | loader.py:160 | heuristique | EC‑Lab |
| Suppression parasites | conserver `|f−f_p|>tol` | loader.py:155 | masque | — |
| Tri EIS | `argsort(f)[::-1]` | loader.py:164 | HF→BF | — |
| Moyenne réplicats | `mean_axis0` (+interp) | loader.py:337 | — | — |
| σ inter‑réplicats | `std(ddof=1)`, plancher `0.001·|Z̄|` | loader.py:339-344 | Measurement Model | Orazem |
| Z_D | `R_D·tanh(√(jωτ_d))/√(jωτ_d)` | physics.py:10-25 | limite L'Hôpital | Bissessur |
| Z Randles | `Re+Z_eq/(1+jωC_b Z_eq)` (voir §4.2) | physics.py:28-63 | — | Randles |
| Rct théorique | `RT/(F²·n·S·k0·C0)` | physics.py:66-81 | (hors chaîne) | Butler‑Volmer |
| Cdl Brug | `[Q·(1/(Re+Re')+1/Rct)^{α−1}]^{1/α}` | physics.py:84-97 | (hors chaîne) | Brug |
| θ recouvrement | `1 − Rct_bare/Rct_ap` | physics.py:100-110 | (hors chaîne) | — |
| Poids modulus | `1/(α_noise·|Z|)²` | weighting.py:55 | absolute_sigma=False | — |
| Poids sigma | `1/σ_re²`, `1/σ_im²` | weighting.py:47-48 | absolute_sigma=True | Orazem |
| Résidus CNLS | `(ΔZre)√w_re`, `(−Zim_fit−Zim)√w_im` | randles_full.py:126-135 | — | — |
| Optimiseur | least_squares TRF, ftol=xtol=1e‑10 | randles_full.py:140 | TRF (bornes) | scipy |
| Covariance | `(JᵀJ)⁻¹·[2·cost/dof si !abs_sigma]` | randles_full.py:155-160 | — | — |
| χ²_red | `Σ(w_re Δre²+w_im Δim²)/(2N−P)` | randles_full.py:186 | — | — |
| rel_residual | `√mean((Δre²+Δim²)/(|Z|²+1e‑30))` | randles_full.py:190 | — | — |
| DRT MAP (défaut) | `fit(mode='optimize')`, γ_lo/γ_hi=None | drt_fit.py::fit_drt | MAP Stan (L-BFGS-B) | bayes_drt2 (lignée Ciucci‑Chen) |
| DRT HMC (à la demande) | `fit(mode='sample')` + percentiles 2.5/97.5 % | drt_fit.py::fit_drt, pipeline::recompute_drt | HMC | bayes_drt2 |
| DRT Rct (arc) | `∫ γ dlnτ` par trapèze sur ±3 lnτ autour du pic pénultième | drt_fit.py::_extract_rct_peak | trapèze (convention Bissessur), repli Rp signalé | Bissessur PRE 2026 |
| DRT Rp (aire totale) | `Σ coef · √π/ε` (somme RBF, **pas** trapèze) | inversion.py::predict_Rp | analytique | bayes_drt2 |
| DRT misfit | `mean((Δre²+Δim²)/|Z|²)` (Re/Im empilés) | drt_fit.py::_reconstruction_metrics | — | — |
| Lin‑KK modèle | `R0+ΣR_k/(1+jωτ_k)[+1/jωC]` | kk_validation.py:22-76 | lstsq linéaire | Boukamp/Schönleber |
| grille τ (KK) | `geomspace(1/ω_max,1/ω_min,M)`, `M=min(max_M,⌊2n/3⌋)` | kk_validation.py:44-45 | fixe | — |
| µ Schönleber | `Σ|R_k<0|/Σ|R_k|` | kk_validation.py:72-74 | — | Schönleber 2014 |
| résidu KK | `√(r_re²+r_im²)/|Z| < tol` | kk_validation.py:110-112 | — | — |
| drift CV | `std/mean` résidus, seuil 0.5 | validator.py:355 | — | — |
| plage commune | `max(f_min)`, `min(f_max)` | validator.py:300-301 | intersection | — |
| Calib. signal | `|Rct_probe−Rct|/|Rct_probe|` vs log₁₀[c] | calibration.py:79-81 | linregress | — |
| Calib. log‑log | `log10(Rct)` vs `log10[c]` | calibration.py:105-107 | linregress | — |
| Calib. CV | `nanmean(Δ)` vs log₁₀[c] | calibration.py:148-155 | linregress | — |
| Δ signal CV | `|I_probe−I_c|/|I_probe|` | cv_pipeline.py:49 | interp + ratio | — |
| Pics CV | argmax/argmin de `savgol(I)`, `ΔE_p=Epa−Epc` | cv_peaks.py:47-60 | Savitzky‑Golay | — |

---

## 13. Points à vérifier

> **Section neutre.** Liste des endroits où l'implémentation **pourrait** diverger d'une
> formule attendue ou mérite une vérification physique/mathématique. Aucune conclusion
> n'est tranchée ici — ce sont des points d'attention factuels, pas des corrections.

1. **Constantes physiques dupliquées `Rct_bare_theory`.**
   `R=8.314`, `F=96485.0`, `n=1` sont codés **en dur** dans la fonction
   (`physics.py:78-80`) alors que les mêmes constantes existent dans
   `config/default.yaml:physics` (`R`, `F`, `n`, `C0`, `T`). Toute modification de la
   config n'affecterait pas cette fonction. À vérifier : cohérence voulue ou risque de
   divergence.

2. **Fonctions physiques non branchées.**
   `Rct_bare_theory` (§4.3), `Cdl_brug` (§4.4) et `theta_EIS` (§4.5) ne sont appelées
   **que par les tests** (`tests/test_physics.py`), jamais par `core/pipeline.py` ni par
   la calibration. Le recouvrement θ et le Rct théorique ne participent donc pas à la
   chaîne de calcul aboutissant à la calibration. À confirmer : est‑ce l'intention ?

3. **Constantes de géométrie / diffusion inutilisées.**
   `geometry` (`xe`, `h`, `d`, `S_WE`), `conditions` (`Fv`) et `physics` (`D_FeIII`,
   `D_FeII`, `C0`, `T`, `k0`…) sont chargées dans les settings Pydantic
   (`core/config.py:13-22, 76-77`) mais **aucune formule du code de calcul ne les
   consomme** (vérifié par recherche globale hors tests/vendored). Le `tau_d` du modèle
   Randles est un paramètre **ajusté**, pas dérivé de `D` et de la géométrie. À vérifier :
   la diffusion attendue est‑elle censée être contrainte par ces constantes ?

4. **Signe Im(Z) : pas de double correction, mais risque propre à l'heuristique de
   majorité (chemin pandas).**
   Trace complète en §7.6 : **aucun** chemin n'applique deux fois la correction. Le
   chemin robuste corrige par nom de colonne puis passe `correct_sign=False`
   (`loader.py:224`) ; le chemin de repli pandas applique **uniquement** l'heuristique de
   majorité `flip ⇔ #{Zim<0}>#{Zim>0}` (`loader.py:160`). Le point d'attention
   B1/V1 résiduel n'est donc **pas** un double flip mais le fait qu'un spectre passant par
   le repli pandas et présentant un `Im(Z)` qui **change de signe** (p. ex. contribution
   inductive HF) avec `#{Zim<0} ≈ #{Zim>0}` pourrait être mal orienté. Le mapping pandas
   `-im_z`/`im_z` → `zimag_ohm` (`loader.py:83-84`) n'utilise **pas** le préfixe `-` du
   libellé pour corriger le signe, contrairement au chemin robuste. À vérifier : ce cas
   de bord est‑il atteignable en pratique (fichiers réels EC‑Lab passant par le repli) ?

5. **Base des logarithmes.**
   Toute la calibration utilise `log₁₀` (`np.log10`, §9). La grille τ du Lin‑KK et de la
   DRT utilise des espacements géométriques (`geomspace`). À vérifier : cohérence
   attendue des unités de pente (pente vs `log₁₀[c]`, pas `ln[c]`).

6. **Paramètre `c` de `linKK` ignoré.**
   `validate_spectrum` passe `c=0.85` (`validator.py:135`) mais `lin_kk` **n'utilise
   pas** ce paramètre : il n'y a **pas** de recherche itérative de M à la Schönleber
   (M est fixé à `min(max_M, ⌊2n/3⌋)`, §8.1). Le µ est calculé sur ce M fixe. À vérifier :
   l'algorithme Schönleber attendu (ajout itératif d'éléments RC jusqu'au critère µ)
   n'est pas implémenté ; l'implémentation résout un système linéaire à M fixe.

7. **Facteur 2 dans le rééchelonnement de covariance.**
   `cov = (JᵀJ)⁻¹ · (2·result.cost/dof)` (`randles_full.py:160`). Le facteur 2 vient de
   `result.cost = ½·Σr²` (SciPy). Le commentaire le note « fix √2 »
   (`randles_full.py:154`). À vérifier : l'expression donne bien `s² = SSR_pondérée/dof`
   avec `dof = 2N−P`.

8. **Signe du résidu imaginaire.**
   Deux écritures coexistent : dans `residuals` c'est `(-Z.imag - spectrum.Zim)·sw_im`
   (`randles_full.py:134`), dans le χ² post‑fit c'est `res_im = spectrum.Zim + Z_fit.imag`
   (`randles_full.py:177`). Les deux sont algébriquement `Zim − (−Z.imag)`. À vérifier :
   cohérence de signe entre les deux (elles semblent équivalentes mais méritent une
   relecture croisée).

9. **DRT — modes optimize/sample, Rct par pic, régularisation dans le vendored.**
   Le mode par défaut est le **MAP Stan** (`fit(mode='optimize')`), calculé par le
   pipeline sur les spectres moyennés ; le **HMC** (`fit(mode='sample')`, intervalles de
   crédibilité) est lancé **à la demande** via `recompute_drt`. Les deux compilent un
   modèle Stan (CmdStan requis, installé au 1er lancement). Le `Rct` de la DRT (§7.3) est
   celui de l'**arc de transfert de charge** (pic pénultième, `∫ γ dlnτ` sur ±3 lnτ) — pas
   la résistance de polarisation totale ; un repli sur **Rp** (aire totale via
   `predict_Rp` = `Σ coef·√π/ε`, somme analytique des coefficients RBF) n'a lieu qu'à
   défaut de pic et est **toujours signalé** (`params['rct_source']` + `warnings`). Le
   misfit de reconstruction (§7.4) est calculé sur `inv.predict_Z(freq)`, un spectre
   reconstruit. La régularisation (λ) est déterminée **entièrement dans
   `vendor/bayes_drt2`**, pas dans le code applicatif : toute vérification doit donc porter
   sur le paquet vendoré.

10. **Plancher σ à 0,1 % du module.**
    Le plancher `σ ← max(σ, 0.001·|Z̄|)` (§3.5, §8.4) est appliqué **par point**.
    À vérifier : en mode « sigma », ce plancher fixe une borne inférieure de poids
    `w ≤ 1/(0.001·|Z|)²` — sa valeur (0,1 %) est un choix codé en dur (`loader.py:342`,
    `validator.py:398`), non configurable.

11. **`alpha_noise` arbitraire en mode modulus.**
    `alpha_noise=0.001` (`config/default.yaml:27`) est décrit dans le code comme un
    niveau de bruit relatif **arbitraire** (`weighting.py:5-8`). En mode « modulus », la
    valeur absolue des poids n'a donc pas de sens statistique et `χ²_red≈1` n'est **pas**
    un test d'adéquation (commenté explicitement, `randles_full.py:181-184`). À garder à
    l'esprit lors de l'interprétation de `chi2_reduced`.

12. **Détection de pics CV mono‑couple.**
    `detect_redox_peaks` prend le max **global** et le min **global** du courant lissé
    (§10.2). Pour un voltammogramme à plusieurs couples redox, cela ne détecte qu'un
    seul pic anodique/cathodique. À vérifier : hypothèse « un seul couple » (Fe(CN)₆)
    conforme au cas d'usage.

13. **Convention de tri et interpolation dans le moyennage.**
    `average_replicates` interpole les réplicats sur la grille du **premier** spectre
    (`loader.py:313, 325`) après un tri croissant local, alors que les spectres sont
    stockés HF→BF (décroissant). `np.interp` requiert une abscisse croissante ; le tri
    `sort_idx = np.argsort(sp.f)` (`loader.py:323`) l'assure pour la source, mais
    `f_ref` (grille cible) reste en ordre décroissant. À vérifier : `np.interp` avec un
    `x` cible décroissant renvoie‑t‑il les valeurs attendues ici (comportement de
    `np.interp` sur `xp` croissant / `x` non trié).

---

*Fin du recensement. Document généré en lecture seule ; aucun fichier de code n'a été
modifié.*
