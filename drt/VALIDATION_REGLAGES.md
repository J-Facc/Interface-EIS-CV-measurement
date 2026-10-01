# Validation des réglages de la DRT — `drt/bayes_drt2` (clone `99d5b60`)

> Rejeu **intégral** de la matrice numérique de `AUDIT.md` Annexe A.3 et A.4 (mode MAP, comme l'audit)
> sur le clone identifié `drt/bayes_drt2/` (voir `drt/PROVENANCE.md`), étendu au mode **HMC** — devenu le
> mode par défaut — sur un périmètre restreint (§3.0), puis choix argumenté du réglage par défaut de
> `drt/engine.py`.
>
> Date : 2026-09-30 · Tout est reproductible par les scripts de `drt/validation/` ; les résultats bruts
> (une ligne JSON par essai, environnement compris) sont versionnés à côté.

## 0. Résumé

**Réglage retenu** (constantes de `drt/engine.py`) :

| Paramètre | Valeur | Pourquoi (section) |
|---|---|---|
| `DEFAULT_MODE` | `'sample'` (HMC / NUTS) | décision utilisateur ; seul mode avec diagnostics de convergence et intervalles |
| `DEFAULT_NONNEG` | `True` | sans elle, HMC ne converge pas sur des arcs RC nets (R-hat jusqu'à 2,3) et le MAP tombe dans des optima à Rp < 0 (§2, §3) |
| `DEFAULT_INIT_FROM_RIDGE` | `True` | indispensable au MAP (un cas à +7,8 % sans elle) ; neutre en HMC (§2, §3.4) |
| `DEFAULT_CHAINS` / `DEFAULT_WARMUP` / `DEFAULT_SAMPLES` | 4 / 500 / 2000 | R-hat fiable (≥ 4 chaînes) et ESS ≥ 100 par chaîne (§3.2, §3.3) |
| `DEFAULT_ADAPT_DELTA` | 0,99 (patch 2) | 0,9 amont : 3 à 24 divergences par ajustement ; 0,99 : aucune (§3.2) |
| `DEFAULT_RANDOM_SEED` | 1234, explicite et enregistré | reproductibilité (DRT-9) |
| `RCT_PEAKS_IN_MEASURED_WINDOW` | `True` | sans elle, Rct_DRT = arc de **diffusion** sur les spectres de type Randles, **quel que soit le réglage** (−47 à −64 %) (§5) |

Résultat : sur les 12 cas de l'audit (dont les 7 où le MAP amont rendait Rp < 0 en silence), le réglage
retenu donne en HMC Rp > 0, **aucune alerte** (12/12), 0 divergence, R-hat ≤ 1,015 ; en MAP aussi (12/12).
Sur 6 essais supplémentaires à d'autres graines (Randles), 2 alertes `ess_tail_low` — signalées, valeurs
inchangées (§6). Limites : biais positif de la moyenne a posteriori de Rp (+0,6 à +1,5 %) avec un IC 95 %
de Rp qui n'inclut pas la valeur vraie ; biais de méthode de Rct ≈ +2 % sur Randles ; ~2 à 4,5 min par
spectre sur 4 cœurs (§7). Le périmètre de la matrice HMC comparative est restreint (§3.0, décision
utilisateur).

## 1. Protocole

**Données — identiques aux scripts de l'audit** (reconstruites par `drt/validation/run_matrix.py`) :

* **A.3** — 2 RC : `Z = 10 + 50/(1+jω·1 ms) + 80/(1+jω·0,1 s)`, **Rp vrai = 130 Ω** ; grilles
  71 pts 1e5→1e-2 Hz (= scénario de `tests/test_drt.py`), 60 pts 1e5→1e-1, 40 pts 1e5→1e-1,
  80 pts 1e6→1e-1, sans bruit ; 60 pts avec bruit complexe 0,5 % (graine de bruit 0) et graine Stan
  1234 (défaut amont), 1, 2, 3.
* **A.4** — Randles complet (ancien `fits.physics.Z_randles_full`, aujourd'hui l'expression par défaut de `fit.circuit` construite par `circuit.parse_circuit` — identique à 1e-12 : Re 200, R'e 20, Cb 1 nF, Qdl 2 µF,
  α 0,9, R_D = 0,3·Rct, τ_d 0,5 s), Rct = 3000 / 3500 / 4200 / 5200 Ω, 60 pts 1e5→1e-1 Hz, bruit
  0,5 % (graine = Rct). **Rp vrai = R'e + Rct + R_D = 20 + 1,3·Rct** (Z(0) − Z(∞)). ⚠️ Les 20 Ω de
  R'e relaxent à τ ≈ R'e·Cb ≈ 2·10⁻⁸ s, hors de la fenêtre mesurée (τ ≥ 1,6·10⁻⁶ s) : les données ne
  peuvent pas les distinguer de R∞, d'où une ambiguïté de ≈ 0,3-0,5 % sur la « vérité » de Rp.

**Réglages comparés** (options de `Inverter.fit`) : `default` (aucune option = réglages amont),
`init_from_ridge`, `nonneg`, `nonneg+ridge` (les deux).

**Métriques** : `Rp` (`Inverter.predict_Rp()`) et son écart ; `γ_min` = min de γ(τ) sur la grille ;
**erreur de reconstruction max** = maxᵢ |Z_fit − Z|/|Z| ; en HMC : R-hat rang-normalisé split
(`stansummary`), ESS bulk/tail, divergences, itérations à profondeur d'arbre max, E-BFMI, et si Rp vrai
est dans l'IC 95 % a posteriori de Rp. « Correct » dans les décomptes = Rp > 0 et |écart| ≤ 1 %.

**Environnement** : Linux, Python 3.12.3, numpy 2.5.3, scipy 1.18.1 (épinglés, `requirements.txt`),
cvxopt 1.3.3, cmdstanpy 1.3.0, CmdStan 2.36.0 (versions du job CI `drt`), 4 cœurs.

| Fichier de résultats | Script | Contenu |
|---|---|---|
| `validation/results_optimize.jsonl` | `run_matrix.py --mode optimize` | §2 : MAP, 12 cas × 4 réglages |
| `validation/results_sample.jsonl` | `run_matrix.py --mode sample` | §3.1 : HMC réglages amont (2 × 200/200) — 48 essais calculés, 28 retenus (§3.0) |
| `validation/results_engine_budget500.jsonl` (+ γ(τ) en `.npz`) | `run_engine_study.py --warmup 500 --samples 500 --adapt-delta 0.9` | §3.2, §4, §5 : 4 chaînes × 500/500 |
| `validation/results_adapt.jsonl` | `run_matrix.py --adapt-delta …` | §3.2 : warmup / `adapt_delta` |
| `validation/results_engine_samples1000.jsonl` (+ `.npz`) | `run_engine_study.py --samples 1000` | §3.3 : réglage retenu à 1000 tirages |
| `validation/results_engine_final_*.jsonl` (+ `.npz`) | `run_engine_study.py` (défauts du moteur) | §6 : réglage retenu, MAP et HMC, graines |

`python drt/validation/report.py <fichier>` régénère les tableaux ci-dessous (`--hmc-scope
results_optimize.jsonl` pour la matrice HMC restreinte, `--engine` pour les études via le moteur).

⚠️ `results_engine_budget500.*` a été produit par une **version intermédiaire** du moteur : pic de Rct
cherché sur toute la grille (règle historique) et saturation de profondeur comptée comme alerte
(`treedepth`). Ses Rp, diagnostics et γ(τ) sont valides ; ses champs `rct*` et `alerts` ne reflètent pas
le moteur final — §5 recalcule Rct à partir des γ(τ) enregistrés. Tous les autres fichiers `engine`
correspondent au moteur final (seul le nombre de tirages diffère pour `samples1000`).

## 2. Mode MAP (`optimize`) — reproduction de l'audit

#### Rp (écart à la vraie valeur)

| Cas | Rp vrai | `default` | `init_from_ridge` | `nonneg` | `nonneg+ridge` |
|---|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | 130 | 130.3 (+0.2 %) | 130.1 (+0.1 %) | 130.3 (+0.2 %) | 130.1 (+0.1 %) |
| A3-60pts-1e5-1e-1 | 130 | -159.4 (-222.6 %) ❌ | 130.1 (+0.1 %) | 130.2 (+0.2 %) | 130.2 (+0.1 %) |
| A3-40pts-1e5-1e-1 | 130 | 130.1 (+0.1 %) | 130.2 (+0.1 %) | 130.5 (+0.4 %) | 130.2 (+0.1 %) |
| A3-80pts-1e6-1e-1 | 130 | 139.8 (+7.5 %) ❌ | 130.0 (+0.0 %) | 140.2 (+7.8 %) ❌ | 130.1 (+0.1 %) |
| A3-60pts-bruit0.5% | 130 | -159.1 (-222.4 %) ❌ | 129.4 (-0.4 %) | 130.7 (+0.6 %) | 130.5 (+0.4 %) |
| A3-60pts-bruit0.5%-seed1 | 130 | 129.5 (-0.4 %) | 129.4 (-0.5 %) | 130.7 (+0.5 %) | 130.5 (+0.4 %) |
| A3-60pts-bruit0.5%-seed2 | 130 | 129.4 (-0.4 %) | 129.4 (-0.5 %) | 130.6 (+0.4 %) | 130.4 (+0.3 %) |
| A3-60pts-bruit0.5%-seed3 | 130 | -159.2 (-222.5 %) ❌ | 129.4 (-0.5 %) | 130.7 (+0.5 %) | 130.5 (+0.4 %) |
| A4-Randles-Rct3000 | 3920 | -5677.9 (-244.8 %) ❌ | 3917.3 (-0.1 %) | 3905.2 (-0.4 %) | 3908.1 (-0.3 %) |
| A4-Randles-Rct3500 | 4570 | -6552.3 (-243.4 %) ❌ | 4553.3 (-0.4 %) | 4559.9 (-0.2 %) | 4569.8 (-0.0 %) |
| A4-Randles-Rct4200 | 5480 | -7925.4 (-244.6 %) ❌ | 5461.9 (-0.3 %) | 5461.1 (-0.3 %) | 5475.3 (-0.1 %) |
| A4-Randles-Rct5200 | 6780 | -9277.5 (-236.8 %) ❌ | 6773.5 (-0.1 %) | 6760.5 (-0.3 %) | 6772.7 (-0.1 %) |

#### γ_min (Ω) / erreur de reconstruction max (%)

| Cas | `default` | `init_from_ridge` | `nonneg` | `nonneg+ridge` |
|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | -4.55 / 0.34 | -4.53 / 0.20 | 0.00 / 0.76 | 0.00 / 1.54 |
| A3-60pts-1e5-1e-1 | -250.85 / 362.71 | -4.62 / 0.19 | 0.00 / 0.74 | 0.00 / 0.76 |
| A3-40pts-1e5-1e-1 | -2.79 / 0.27 | -2.64 / 0.26 | 0.00 / 0.86 | 0.00 / 0.82 |
| A3-80pts-1e6-1e-1 | -5.99 / 2.00 | -5.89 / 0.17 | 0.00 / 1.20 | 0.00 / 0.77 |
| A3-60pts-bruit0.5% | -250.82 / 363.39 | -1.00 / 1.41 | 0.00 / 1.65 | 0.00 / 1.62 |
| A3-60pts-bruit0.5%-seed1 | -1.30 / 1.39 | -2.04 / 1.45 | 0.00 / 1.62 | 0.00 / 1.64 |
| A3-60pts-bruit0.5%-seed2 | -2.27 / 1.44 | -2.12 / 1.45 | 0.00 / 1.65 | 0.00 / 1.67 |
| A3-60pts-bruit0.5%-seed3 | -250.61 / 363.45 | -1.95 / 1.45 | 0.00 / 1.64 | 0.00 / 1.64 |
| A4-Randles-Rct3000 | -8291.73 / 384.11 | -62.61 / 1.06 | 0.27 / 1.02 | 0.36 / 1.07 |
| A4-Randles-Rct3500 | -9607.70 / 377.48 | -17.43 / 1.03 | 0.57 / 1.05 | 0.20 / 1.03 |
| A4-Randles-Rct4200 | -13823.96 / 391.77 | -15.22 / 1.22 | 0.13 / 1.15 | 0.10 / 1.23 |
| A4-Randles-Rct5200 | -13615.74 / 371.55 | -19.67 / 1.15 | 0.02 / 1.13 | 0.29 / 1.11 |

#### Décompte

| Réglage | Rp correct | Rp < 0 | erreur recon. max > 10 % |
|---|---|---|---|
| `default` | 4/12 | 7/12 | 7/12 |
| `init_from_ridge` | 12/12 | 0/12 | 0/12 |
| `nonneg` | 11/12 | 0/12 | 0/12 |
| `nonneg+ridge` | 12/12 | 0/12 | 0/12 |

**Comparaison avec `AUDIT.md` §4.4** : identique à la précision publiée pour `default` (−159,4 ; −159,1 ;
129,5 / 129,4 / −159,2 ; −5678 / −6552 / −7925 / −9278 ; 80 pts ≈ 140) et `nonneg` (3905 / 4560 / 5461 /
6761 ; 80 pts ≈ 140) ; `init_from_ridge` identique sauf deux valeurs Randles (3917 vs 3916, 4553 vs 4555 :
≤ 0,04 %, attribuables au passage numpy 2.4.6 → 2.5.3 / scipy 1.17.1 → 1.18.1 dans la solution ridge
itérative). **Le défaut DRT-1 est donc reproduit sur le clone** — normal : le vendoring était le même code
(`drt/PROVENANCE.md` §1). L'audit comptait 9 essais par réglage ; la matrice ci-dessus en compte 12 (les
graines 1/2/3 sont jouées pour tous les réglages) : `init_from_ridge` 12/12, `nonneg` 11/12 — même
exception que l'audit (80 pts, 1e6→1e-1 Hz) — et la combinaison `nonneg+ridge` 12/12.

Lecture de l'exception `nonneg` (80 pts) : Rp = 140 alors que la reconstruction est bonne (1,2 %) — les
10 Ω de R∞ sont déplacés dans une relaxation à τ < 1/(2π·10⁶ Hz), indiscernable d'une résistance série
aux fréquences mesurées. L'initialisation ridge place le MAP dans le bon bassin.

## 3. Mode HMC (`sample`)

### 3.0 Périmètre de la matrice HMC (décision utilisateur du 2026-09-30)

La matrice HMC n'a **pas** à reproduire les 4 réglages sur tous les scénarios avec l'exhaustivité du MAP.
Justification (décision de l'utilisateur, reprise telle quelle) : *le réglage par défaut est déjà écarté
sans ambiguïté par le MAP — échec net et systématique dans l'audit et dans la reproduction du §2.* Le
périmètre retenu est donc :

| | Réglage | Scénarios | Rôle |
|---|---|---|---|
| (a) | `default` (amont) | **3** : `A3-60pts-1e5-1e-1`, `A3-60pts-bruit0.5%`, `A4-Randles-Rct3000` — un par famille où le MAP amont échoue (grille sans bruit, bruit, Randles) | confirmer qu'il échoue aussi en HMC, sans plus |
| (b) | `init_from_ridge`, `nonneg` | **tous** (12) | les deux candidats réels |
| (c) | `nonneg+ridge` | seulement les scénarios où `init_from_ridge` et `nonneg` **divergent en MAP** (l'un correct, l'autre non, §2) : **`A3-80pts-1e6-1e-1`** seul | départager là où les candidats divergent |

Le scénario de (c) est calculé, pas choisi à la main : `report.py --hmc-scope results_optimize.jsonl`
applique la règle aux résultats MAP. « 13 scénarios » dans l'audit = nos 12 + le scénario du test du dépôt
compté deux fois par l'Annexe A.3 (via `DRTBayesModel` puis via `Inverter` sur la grille 71 pts
1e5→1e-2 Hz : mêmes données, même code — fusionnés ici en `A3-repo-71pts-1e5-1e-2`).

**Traçabilité** : la matrice complète (4 × 12 = 48 essais) avait déjà été calculée quand ce périmètre a été
fixé. Les 48 essais restent dans `validation/results_sample.jsonl` (rien n'est effacé), mais **seuls les
28 du périmètre** figurent dans les tableaux et fondent les conclusions de §3.1 ; les tableaux se
régénèrent par `python drt/validation/report.py drt/validation/results_sample.jsonl --hmc-scope
drt/validation/results_optimize.jsonl`. Les études **ciblées** qui suivent (§3.2-3.4 : budget,
`adapt_delta`, nombre de tirages, initialisation ; §4-5 : localisation de Rp, extraction de Rct) ne sont
pas des comparaisons exhaustives : chacune garde son propre petit jeu de cas, indiqué dans sa section. La
validation du réglage **retenu** (§6) porte, elle, sur tous les scénarios.

### 3.1 Matrice HMC — réglages amont d'échantillonnage (2 chaînes × 200 warmup / 200 tirages, `adapt_delta` 0,9)

#### Rp (écart à la vraie valeur)

| Cas | Rp vrai | `default` | `init_from_ridge` | `nonneg` | `nonneg+ridge` |
|---|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | 130 | — | 130.3 (+0.3 %) | 131.1 (+0.9 %) | — |
| A3-60pts-1e5-1e-1 | 130 | 130.3 (+0.2 %) | 130.5 (+0.4 %) | 131.3 (+1.0 %) ❌ | — |
| A3-40pts-1e5-1e-1 | 130 | — | 130.4 (+0.3 %) | 131.9 (+1.5 %) ❌ | — |
| A3-80pts-1e6-1e-1 | 130 | — | 130.5 (+0.4 %) | 131.1 (+0.8 %) | 131.0 (+0.8 %) |
| A3-60pts-bruit0.5% | 130 | 129.8 (-0.1 %) | 129.9 (-0.1 %) | 131.9 (+1.4 %) ❌ | — |
| A3-60pts-bruit0.5%-seed1 | 130 | — | 129.9 (-0.1 %) | 131.9 (+1.5 %) ❌ | — |
| A3-60pts-bruit0.5%-seed2 | 130 | — | 129.7 (-0.2 %) | 131.9 (+1.5 %) ❌ | — |
| A3-60pts-bruit0.5%-seed3 | 130 | — | 129.7 (-0.2 %) | 131.9 (+1.5 %) ❌ | — |
| A4-Randles-Rct3000 | 3920 | 3911.4 (-0.2 %) | 3912.3 (-0.2 %) | 3943.2 (+0.6 %) | — |
| A4-Randles-Rct3500 | 4570 | — | 4562.0 (-0.2 %) | 4601.4 (+0.7 %) | — |
| A4-Randles-Rct4200 | 5480 | — | 5472.7 (-0.1 %) | 5510.6 (+0.6 %) | — |
| A4-Randles-Rct5200 | 6780 | — | 6782.2 (+0.0 %) | 6828.3 (+0.7 %) | — |

#### γ_min (Ω) / erreur de reconstruction max (%)

| Cas | `default` | `init_from_ridge` | `nonneg` | `nonneg+ridge` |
|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | — | -7.01 / 0.46 | 0.04 / 1.85 | — |
| A3-60pts-1e5-1e-1 | -8.47 / 0.35 | -8.74 / 0.52 | 0.04 / 1.87 | — |
| A3-40pts-1e5-1e-1 | — | -4.39 / 0.69 | 0.06 / 2.56 | — |
| A3-80pts-1e6-1e-1 | — | -6.28 / 0.46 | 0.02 / 1.60 | 0.02 / 1.54 |
| A3-60pts-bruit0.5% | -3.90 / 1.43 | -2.48 / 1.44 | 0.04 / 2.15 | — |
| A3-60pts-bruit0.5%-seed1 | — | -2.05 / 1.43 | 0.04 / 2.15 | — |
| A3-60pts-bruit0.5%-seed2 | — | -3.47 / 1.41 | 0.04 / 2.10 | — |
| A3-60pts-bruit0.5%-seed3 | — | -2.04 / 1.45 | 0.04 / 2.22 | — |
| A4-Randles-Rct3000 | -29.00 / 1.07 | -21.91 / 1.10 | 2.75 / 2.38 | — |
| A4-Randles-Rct3500 | — | -14.00 / 1.00 | 3.02 / 2.70 | — |
| A4-Randles-Rct4200 | — | -9.92 / 1.11 | 3.19 / 3.19 | — |
| A4-Randles-Rct5200 | — | -75.30 / 1.33 | 3.85 / 4.02 | — |

#### Diagnostics : R-hat max · divergences · itérations à profondeur max (/400) · ESS bulk min · Rp vrai ∈ IC 95 % ?

| Cas | `default` | `init_from_ridge` | `nonneg` | `nonneg+ridge` |
|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | — | 2.307 · 0 · 398 · 3 · oui | 1.069 · 0 · 0 · 24 · **non** | — |
| A3-60pts-1e5-1e-1 | 2.156 · 0 · 398 · 3 · oui | 1.856 · 0 · 398 · 3 · oui | 1.063 · 0 · 0 · 32 · **non** | — |
| A3-40pts-1e5-1e-1 | — | 2.150 · 0 · 398 · 3 · oui | 1.048 · 2 · 0 · 59 · **non** | — |
| A3-80pts-1e6-1e-1 | — | 2.008 · 0 · 398 · 3 · oui | 1.048 · 0 · 0 · 97 · **non** | 1.042 · 2 · 0 · 77 · **non** |
| A3-60pts-bruit0.5% | 1.131 · 0 · 357 · 13 · oui | 1.179 · 0 · 390 · 11 · oui | 1.047 · 1 · 0 · 54 · **non** | — |
| A3-60pts-bruit0.5%-seed1 | — | 1.091 · 0 · 398 · 25 · oui | 1.044 · 1 · 0 · 61 · **non** | — |
| A3-60pts-bruit0.5%-seed2 | — | 1.061 · 0 · 398 · 29 · oui | 1.046 · 2 · 0 · 66 · **non** | — |
| A3-60pts-bruit0.5%-seed3 | — | 1.122 · 0 · 392 · 17 · oui | 1.031 · 0 · 0 · 77 · **non** | — |
| A4-Randles-Rct3000 | 1.059 · 0 · 7 · 39 · oui | 1.065 · 0 · 0 · 56 · oui | 1.087 · 0 · 5 · 30 · **non** | — |
| A4-Randles-Rct3500 | — | 1.047 · 0 · 1 · 53 · oui | 1.290 · 0 · 282 · 6 · **non** | — |
| A4-Randles-Rct4200 | — | 1.074 · 0 · 7 · 26 · oui | 1.052 · 0 · 146 · 71 · **non** | — |
| A4-Randles-Rct5200 | — | 1.054 · 0 · 2 · 54 · oui | 1.030 · 0 · 37 · 95 · **non** | — |

#### Décompte (correct = Rp > 0 et |écart| ≤ 1 %)

| Réglage | Rp correct | Rp < 0 | erreur recon. max > 10 % | R-hat > 1,05 | divergences > 0 | prof. max > 0 | Rp vrai hors IC 95 % |
|---|---|---|---|---|---|---|---|
| `default` | 3/3 | 0/3 | 0/3 | 3/3 | 0/3 | 3/3 | 0/3 |
| `init_from_ridge` | 12/12 | 0/12 | 0/12 | 11/12 | 0/12 | 11/12 | 0/12 |
| `nonneg` | 6/12 | 0/12 | 0/12 | 5/12 | 4/12 | 4/12 | 12/12 |
| `nonneg+ridge` | 1/1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |

**Lecture.**

1. **HMC ne tombe jamais dans l'optimum dégénéré du MAP** (Rp > 0 dans 28/28 essais), y compris avec le
   réglage `default` : DRT-1 est une pathologie de l'optimiseur L-BFGS partant d'une initialisation
   aléatoire, pas du modèle. **Le réglage `default` échoue néanmoins aussi en HMC, autrement** : Rp est
   juste, mais les chaînes ne convergent pas (R-hat 1,06 à 2,16 sur les 3 scénarios, profondeur d'arbre
   saturée sur les 3, ESS bulk 3 à 39) — un résultat dont les diagnostics interdisent de le croire.
2. **DRT non contrainte (`default`, `init_from_ridge`) : estimations justes mais chaînes NON convergées**
   sur les arcs RC idéaux — 357 à 398 itérations sur 400 à profondeur d'arbre maximale, R-hat 1,06 à
   2,31 (> 1,05 dans 10/10 essais 2 RC), ESS bulk 3 à 29. La justesse de Rp y est fortuite et l'IC n'a
   pas de valeur. Sur les spectres Randles (arcs CPE élargis), convergence approximative (R-hat 1,05-1,07).
3. **DRT ≥ 0 (`nonneg`) : pas de saturation sur les 2 RC, R-hat 1,03-1,07** (jusqu'à 1,29 sur un cas
   Randles), mais avec 2 × 200 tirages R-hat dépasse encore 1,05 dans 5 cas sur 12, et des divergences
   apparaissent (4/12).
4. **Biais de la moyenne a posteriori de Rp sous `nonneg` : +0,8 à +1,5 % (2 RC), +0,6 à +0,7 %
   (Randles), IC 95 % n'incluant PAS la vérité dans 13/13 essais** (`nonneg` et `nonneg+ridge`) — voir §4.
5. **Scénario divergent (c), `A3-80pts-1e6-1e-1`** : en MAP, `nonneg` seul donne Rp = 140,2 (+7,8 %)
   et `init_from_ridge` 130,0 ; la combinaison donne 130,1 (§2). En HMC, `init_from_ridge` ne converge
   pas (R-hat 2,01), `nonneg` et la combinaison si (R-hat 1,048 et 1,042), au même biais (+0,8 %).
   → En HMC, seule une DRT ≥ 0 converge ; en MAP, `nonneg` seul a un échec que l'initialisation ridge
   corrige : **la combinaison est le seul réglage correct dans les deux modes sur ce scénario.**

### 3.2 Budget d'échantillonnage, `adapt_delta` et profondeur d'arbre

Étude ciblée (4 cas, 4 réglages) : 4 chaînes (recommandation de Vehtari et al. 2021 pour que R-hat soit
fiable) × 500 warmup / 500 tirages, `adapt_delta` 0,9 (via le moteur, `results_engine_budget500.jsonl`) :

| Cas | Réglage | R-hat · div · prof. max (/2000) · ESS bulk/tail | Rp |
|---|---|---|---|
| A3-repo-71pts | `default` | 1.626 · 0 · 1996 · 7/28 | 130.2 |
| A3-repo-71pts | `init_from_ridge` | 1.877 · 0 · 1996 · 6/61 | 130.3 |
| A3-repo-71pts | `nonneg` | 1.015 · 24 · 1 · 341/252 | 131.1 |
| A3-repo-71pts | `nonneg+ridge` | 1.011 · 3 · 0 · 414/151 | 131.1 |
| A3-60pts-bruit0.5% | `default` | 1.046 · 0 · 1996 · 126/91 | 129.8 |
| A3-60pts-bruit0.5% | `init_from_ridge` | 1.065 · 0 · 1996 · 53/41 | 129.8 |
| A3-60pts-bruit0.5% | `nonneg` | 1.008 · 5 · 0 · 485/326 | 132.0 |
| A3-60pts-bruit0.5% | `nonneg+ridge` | 1.009 · 8 · 0 · 479/511 | 131.9 |
| A4-Randles-Rct3000 | `default` | 1.026 · 0 · 1 · 375/241 | 3913.8 |
| A4-Randles-Rct3000 | `init_from_ridge` | 1.013 · 0 · 127 · 293/269 | 3912.6 |
| A4-Randles-Rct3000 | `nonneg` | 1.014 · 0 · 0 · 304/216 | 3944.5 |
| A4-Randles-Rct3000 | `nonneg+ridge` | 1.020 · 0 · 1 · 279/200 | 3945.4 |
| A4-Randles-Rct5200 | `default` | 1.023 · 0 · 1 · 232/199 | 6785.8 |
| A4-Randles-Rct5200 | `init_from_ridge` | 1.021 · 0 · 0 · 214/288 | 6784.4 |
| A4-Randles-Rct5200 | `nonneg` | 1.019 · 0 · 0 · 280/336 | 6828.9 |
| A4-Randles-Rct5200 | `nonneg+ridge` | 1.012 · 0 · 0 · 386/353 | 6830.1 |

* Quadrupler le budget **ne fait pas converger la DRT non contrainte** sur les 2 RC (1996/2000
  itérations saturées dans les 4 essais ; R-hat 1,63-1,88 sans bruit, ESS bulk 53-126 avec bruit) :
  c'est structurel, pas un manque de tirages.
* Sous `nonneg`, R-hat ≈ 1,01, mais **3 à 24 divergences** subsistent sur les 2 RC.

`nonneg+ridge`, 4 chaînes × 500 tirages : effet du warmup et d'`adapt_delta` (`results_adapt.jsonl` ;
ESS min sur paramètres et `lp__`) :

| Cas | warmup | `adapt_delta` | div | prof. max (/2000) | R-hat | ESS bulk / tail | E-BFMI min | durée |
|---|---|---|---|---|---|---|---|---|
| A3-repo-71pts | 1000 | 0,90 | 6 | 0 | 1.009 | 380 / 562 | 0.85 | 85 s |
| A3-80pts-1e6-1e-1 | 1000 | 0,90 | 15 | 0 | 1.008 | 447 / 532 | 0.79 | 56 s |
| A3-60pts-bruit0.5% | 1000 | 0,90 | 14 | 0 | 1.011 | 493 / 292 | 0.75 | 44 s |
| A4-Randles-Rct3500 | 1000 | 0,90 | 0 | 1 | 1.011 | 353 / 185 | 0.71 | 76 s |
| A3-repo-71pts | 500 | 0,95 | 0 | 661 | 1.010 | 397 / 482 | 0.75 | 102 s |
| A3-80pts-1e6-1e-1 | 500 | 0,95 | 13 | 0 | 1.009 | 466 / 370 | 0.76 | 60 s |
| A3-60pts-bruit0.5% | 500 | 0,95 | 0 | 0 | 1.010 | 435 / 367 | 0.75 | 37 s |
| A4-Randles-Rct3500 | 500 | 0,95 | 0 | 748 | 1.015 | 362 / 312 | 0.78 | 72 s |
| A3-repo-71pts | 500 | **0,99** | **0** | 1996 | 1.014 | 270 / 200 | 0.85 | 98 s |
| A3-80pts-1e6-1e-1 | 500 | **0,99** | **0** | 0 | 1.007 | 465 / 337 | 0.85 | 64 s |
| A3-60pts-bruit0.5% | 500 | **0,99** | **0** | 0 | 1.010 | 458 / 464 | 0.80 | 50 s |
| A4-Randles-Rct3500 | 500 | **0,99** | **0** | 1989 | 1.014 | 277 / 313 | 0.69 | 86 s |

* Allonger le warmup ne supprime pas les divergences ; **`adapt_delta` = 0,99 les supprime toutes**
  (0/4), au prix d'un pas plus petit, donc de trajectoires qui saturent la profondeur 10 sur 2 cas.
  → `DEFAULT_ADAPT_DELTA = 0.99`, ce qui exige le **patch 2** (`drt/PROVENANCE.md`) : `Inverter.fit`
  codait `adapt_delta=0.9` en dur.
* Avec 500 tirages/chaîne, l'ESS reste sous 400 (= 100 × 4 chaînes, Vehtari et al. 2021) sur 2 cas sur 4
  → augmenter le nombre de tirages (l'ESS croît ~ linéairement avec lui) : voir §3.3.
* **Profondeur d'arbre maximale : rapportée, pas une alerte.** La documentation Stan (« Runtime warnings
  and convergence problems ») la classe comme un problème d'efficacité, non de validité — au contraire des
  divergences. Ses conséquences (exploration lente) sont exactement ce que mesurent R-hat et l'ESS, qui
  sont gardés. Vérifié : chacun des 32 essais saturés de §3.1-3.2 (18 de la matrice, 9 + 5 des études
  ciblées) portait aussi une alerte R-hat ou
  ESS ; inversement, avec le budget retenu, des ajustements saturés à 100 % ont 0 divergence,
  R-hat ≤ 1,02 et ESS ≥ 400 (§6) — la saturation seule ne dit rien de la validité.

### 3.3 Nombre de tirages : 1000 ne suffit pas, 2000 retenu

Premier essai du réglage final avec 1000 tirages/chaîne (`results_engine_samples1000.jsonl`, 10 cas
avant interruption) :

| Cas | R-hat max (variable) | div | prof. max (/4000) | ESS bulk / tail min | alertes | durée |
|---|---|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | 1.015 (Rinf_raw) | 0 | 3996 | 482 / 406 | aucune | 172 s |
| A3-60pts-1e5-1e-1 | 1.004 (x[4]) | 0 | 1700 | 806 / 938 | aucune | 121 s |
| A3-40pts-1e5-1e-1 | 1.012 (lp__) | 0 | 1141 | 555 / 625 | aucune | 118 s |
| A3-80pts-1e6-1e-1 | 1.004 (ups_raw[1]) | 0 | 0 | 900 / 914 | aucune | 122 s |
| A3-60pts-bruit0.5% | 1.007 (lp__) | 0 | 0 | 994 / 865 | aucune | 80 s |
| A3-60pts-bruit0.5%-seed1 | 1.007 (lp__) | 0 | 945 | 889 / 979 | aucune | 122 s |
| A3-60pts-bruit0.5%-seed2 | 1.005 (x[78]) | 0 | 67 | 942 / 890 | aucune | 88 s |
| A3-60pts-bruit0.5%-seed3 | 1.005 (x[76]) | 0 | 1 | 908 / 873 | aucune | 71 s |
| A4-Randles-Rct3000 | 1.021 (x[42]) | 0 | 3996 | 356 / 254 | ess_bulk_low, ess_tail_low | 142 s |
| A4-Randles-Rct3500 | 1.014 (x[43]) | 0 | 3984 | 557 / 395 | ess_tail_low | 140 s |

Sur les spectres de type Randles, la profondeur d'arbre est saturée à 100 % (conséquence d'`adapt_delta`
0,99) et l'ESS reste sous 400. Le pire R-hat porte sur un coefficient de la DRT **près du pic de
transfert** (x[42], x[43]) : le faible ESS concerne donc la grandeur d'intérêt elle-même, et restreindre
le périmètre de la garde à certains paramètres ne serait pas justifié. → **`DEFAULT_SAMPLES = 2000`**
(ESS attendu ≈ ×2). Le coût est assumé (fiabilité > vitesse).

### 3.4 `init_from_ridge` en HMC

Étude ciblée (§3.2, 4 cas). À budget égal, `nonneg` et `nonneg+ridge` donnent le même postérieur à l'erreur Monte-Carlo près (Rp 131,1 /
131,1 ; 132,0 / 131,9 ; 3944,5 / 3945,4 ; 6828,9 / 6830,1 — §3.2), avec moins de divergences au départ
de la solution ridge (3 vs 24, 8 vs 5 : non concluant). Elle est **indispensable au MAP** (§2). Réserve :
toutes les chaînes partent du même vecteur de coefficients x (les autres paramètres — hyperparamètres,
modèle d'erreur — restent initialisés aléatoirement par chaîne), ce qui affaiblit la surdispersion
initiale sur laquelle repose R-hat ; l'égalité des postérieurs obtenus avec et sans cette initialisation
indique qu'elle ne masque pas ici de mode. Un seul réglage pour les deux modes : `nonneg=True`,
`init_from_ridge=True`.

## 4. Pourquoi Rp est biaisé sous `nonneg` en HMC (et pas Rct)

Aire de γ(τ) moyenne a posteriori, par région de τ (fenêtre mesurée = [1/(2π f_max), 1/(2π f_min)]) —
`results_engine_budget500.jsonl`, 4 × 500 :

| Cas | Réglage | aire dans la fenêtre | aire τ > τ_max (BF) | aire τ < τ_min (HF) | Rp |
|---|---|---|---|---|---|
| A3-60pts-bruit0.5% (vrai 130) | `default` | 129.8 | −0.4 | 0.4 | 129.8 |
| A3-60pts-bruit0.5% | `nonneg+ridge` | 130.7 | 0.6 | 0.5 | 131.9 |
| A4-Randles-Rct3000 (vrai 3920) | `default` | 3896.0 | 5.9 | 11.5 | 3913.8 |
| A4-Randles-Rct3000 | `nonneg+ridge` | 3903.4 | 17.9 | 21.9 | 3945.4 |
| A4-Randles-Rct5200 (vrai 6780) | `nonneg+ridge` | 6764.8 | 27.9 | 34.0 | 6830.1 |

La moyenne a posteriori d'un coefficient contraint ≥ 0 est **strictement positive** même là où la
vraie DRT est nulle ; sommées sur ~80 fonctions de base, ces contributions gonflent Rp. L'essentiel de
l'excès se loge **hors de la fenêtre mesurée**, où rien dans les données ne contraint γ (au-delà de τ_max
une relaxation ne contribue plus à Z aux fréquences mesurées ; en deçà de τ_min elle est
indiscernable de R∞). Le MAP (le mode, et non la moyenne) n'a pas ce biais (§2 : 130,1-130,5).
L'IC 95 % de Rp, calculé sur ces mêmes tirages, est donc **décalé** et n'est pas une incertitude fiable
de Rp. **Rct**, intégré sur ±3 en ln τ autour d'un pic situé dans la fenêtre, n'est quasiment pas
affecté (§5, §6 : 2 RC 50,1-50,2 pour 50).

## 5. Extraction de Rct : restriction à la fenêtre mesurée

La règle historique (`fits/drt_fit._extract_rct_peak`, conservée : pic **pénultième** au-dessus de
1e-3·max, bords écartés, aire ±3 en ln τ) retient les maxima **hors de la fenêtre mesurée**. Or γ(τ) y
porte de petites bosses (≈ 0,2 % du max, ex. Randles 3000 : 6 Ω à 5·10⁻⁷ s et 4 Ω à 6,3 s pour un pic de
2280 Ω) : la bosse BF devient le « dernier » pic et le pénultième tombe sur la **diffusion**.

HMC, 4 × 500 (γ enregistrés, `results_engine_budget500/`) :

| Cas | Réglage | règle historique : Rct (τ du pic) | fenêtre mesurée : Rct (τ du pic) | cible |
|---|---|---|---|---|
| A3-60pts-bruit0.5% | `default` | 49.9 (0.001 s) | 49.9 (0.001 s) | 50 @ 1 ms |
| A3-60pts-bruit0.5% | `nonneg` | 80.6 (0.1 s) ❌ | 50.2 (0.001 s) | 50 @ 1 ms |
| A3-60pts-bruit0.5% | `nonneg+ridge` | 80.6 (0.1 s) ❌ | 50.2 (0.001 s) | 50 @ 1 ms |
| A3-repo-71pts | tous | 50.0-50.1 (0.001 s) | 50.0-50.1 (0.001 s) | 50 @ 1 ms |
| A4-Randles-Rct3000 | `default` | 1081.2 (0.2 s) ❌ | 3073.2 (0.0032 s) | 3000 |
| A4-Randles-Rct3000 | `init_from_ridge` | 1082.4 (0.2 s) ❌ | 3072.6 (0.0032 s) | 3000 |
| A4-Randles-Rct3000 | `nonneg` | 1082.5 (0.2 s) ❌ | 3060.5 (0.0032 s) | 3000 |
| A4-Randles-Rct3000 | `nonneg+ridge` | 1082.1 (0.2 s) ❌ | 3059.5 (0.0032 s) | 3000 |
| A4-Randles-Rct5200 | `default` | 2756.7 (0.2 s) ❌ | 5299.3 (0.0063 s) | 5200 |
| A4-Randles-Rct5200 | `init_from_ridge` | 2755.6 (0.2 s) ❌ | 5297.8 (0.0063 s) | 5200 |
| A4-Randles-Rct5200 | `nonneg` | 2766.2 (0.2 s) ❌ | 5321.1 (0.0063 s) | 5200 |
| A4-Randles-Rct5200 | `nonneg+ridge` | 2766.9 (0.2 s) ❌ | 5319.4 (0.0063 s) | 5200 |

Et en **MAP** avec le réglage retenu (`results_engine_final_optimize/`) : règle historique 1071 / 1359 /
1808 / 2760 Ω pour Rct = 3000 / 3500 / 4200 / 5200 (pic de diffusion à 0,2 s) ; fenêtre mesurée
3053 / 3577 / 4301 / 5268 Ω (pic à 3-5 ms).

**Conséquence importante hors du périmètre initial** : même les réglages MAP que l'audit a validés sur Rp
(`init_from_ridge`, `nonneg`) donnaient, avec l'extraction actuelle de `fits/drt_fit.py`, un
`Rct_drt` **faux de −47 à −64 %** sur les spectres de type Randles. D'où
`RCT_PEAKS_IN_MEASURED_WINDOW = True` dans `drt/engine.py` — seule modification de la règle, justifiée
physiquement (un pic hors fenêtre est une extrapolation, pas une mesure). Le biais résiduel (+1,3 à
+2,4 % sur Randles, tous réglages) est celui de la convention ±3 ln τ, qui englobe une partie de la
structure de diffusion : pour une diffusion bornée isolée, les relaxations sont à
τ_k = 4τ_d/((2k−1)²π²) = 0,20 s, 23 ms, 8 ms… — les deux dernières à moins de 3 unités de ln τ du pic
de transfert (3-6 ms). Estimation indicative : ici la diffusion est imbriquée dans la branche de
transfert du circuit, sa DRT exacte n'est pas une somme de pics séparés.

## 6. Réglage retenu — validation finale via `drt/engine.py`

Défauts du moteur : HMC, `nonneg=True`, `init_from_ridge=True`, 4 chaînes × 500 warmup / 2000 tirages,
`adapt_delta` 0,99, graine 1234, pic de Rct dans la fenêtre mesurée.

HMC, graine 1234 (`results_engine_final_sample.jsonl`) :

| Cas | Réglage | graine | Rp (écart) | aire hors fenêtre BF / HF | Rct (IC 95 %) | τ_Rct | cible | erreur recon. max | R-hat · div · prof. max · ESS bulk/tail | alertes | durée |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | `nonneg+ridge` | 1234 | 131.1 (+0.85 %) | 0.3 / 0.5 | 50.1 (49.9–50.2) | 0.001 s | 50 @ 1e-03 s | 1.83 % | 1.005 · 0 · 7996 · 1024/1327 | aucune | 268 s |
| A3-60pts-1e5-1e-1 | `nonneg+ridge` | 1234 | 131.3 (+0.99 %) | 0.4 / 0.5 | 50.1 (49.9–50.2) | 0.001 s | 50 @ 1e-03 s | 1.82 % | 1.002 · 0 · 3448 · 1577/2013 | aucune | 214 s |
| A3-40pts-1e5-1e-1 | `nonneg+ridge` | 1234 | 131.8 (+1.41 %) | 0.6 / 0.7 | 50.1 (49.9–50.4) | 0.001 s | 50 @ 1e-03 s | 2.59 % | 1.006 · 0 · 2284 · 1268/1514 | aucune | 200 s |
| A3-80pts-1e6-1e-1 | `nonneg+ridge` | 1234 | 131.1 (+0.82 %) | 0.3 / 0.4 | 50.0 (49.9–50.2) | 0.001 s | 50 @ 1e-03 s | 1.61 % | 1.002 · 0 · 0 · 2053/2264 | aucune | 160 s |
| A3-60pts-bruit0.5% | `nonneg+ridge` | 1234 | 131.9 (+1.48 %) | 0.6 / 0.5 | 50.2 (49.9–50.5) | 0.001 s | 50 @ 1e-03 s | 2.16 % | 1.004 · 0 · 0 · 1973/1923 | aucune | 126 s |
| A3-60pts-bruit0.5%-seed1 | `nonneg+ridge` | 1 | 131.9 (+1.49 %) | 0.6 / 0.5 | 50.2 (49.9–50.5) | 0.001 s | 50 @ 1e-03 s | 2.17 % | 1.002 · 0 · 1887 · 1842/1630 | aucune | 205 s |
| A3-60pts-bruit0.5%-seed2 | `nonneg+ridge` | 2 | 131.9 (+1.48 %) | 0.6 / 0.5 | 50.2 (49.9–50.5) | 0.001 s | 50 @ 1e-03 s | 2.17 % | 1.002 · 0 · 131 · 1779/1717 | aucune | 145 s |
| A3-60pts-bruit0.5%-seed3 | `nonneg+ridge` | 3 | 131.9 (+1.48 %) | 0.6 / 0.5 | 50.2 (49.9–50.5) | 0.001 s | 50 @ 1e-03 s | 2.17 % | 1.002 · 0 · 1 · 1821/2000 | aucune | 115 s |
| A4-Randles-Rct3000 | `nonneg+ridge` | 1234 | 3944.2 (+0.62 %) | 17.3 / 21.3 | 3060.0 (3040.1–3078.3) | 0.0032 s | 3000 | 2.45 % | 1.006 · 0 · 7994 · 847/631 | aucune | 225 s |
| A4-Randles-Rct3500 | `nonneg+ridge` | 1234 | 4599.2 (+0.64 %) | 17.5 / 23.6 | 3573.9 (3551.7–3596.7) | 0.004 s | 3500 | 2.57 % | 1.004 · 0 · 7974 · 1045/946 | aucune | 218 s |
| A4-Randles-Rct4200 | `nonneg+ridge` | 1234 | 5514.0 (+0.62 %) | 22.4 / 27.6 | 4283.7 (4252.3–4320.1) | 0.005 s | 4200 | 3.24 % | 1.014 · 0 · 7996 · 612/636 | aucune | 214 s |
| A4-Randles-Rct5200 | `nonneg+ridge` | 1234 | 6830.4 (+0.74 %) | 27.9 / 34.1 | 5319.1 (5272.8–5381.6) | 0.0063 s | 5200 | 4.05 % | 1.005 · 0 · 7996 · 903/559 | aucune | 213 s |

**Graine 1234** : 12 essais, **12/12 sans aucune alerte** ; écart de Rp +0.62 à +1.49 % ; écart de Rct à la cible +0.09 à +2.29 % ; R-hat max 1.014 ; divergences 0 au total ; ESS bulk min 612, tail min 559 ; E-BFMI min 0.70 ; durée 115-268 s. Cible de Rct dans l'IC 95 % : 8/12 ; Rp vrai dans l'IC 95 % : 0/12.

Robustesse à la graine Stan — graines 1 et 2 sur les cas Randles (`results_engine_final_sample_seeds.jsonl`) ; balayage prévu sur 8 essais, **arrêté à 6 sur instruction de l'utilisateur** lors de la décision de périmètre du §3.0 (pas de balayage supplémentaire de la combinaison hors du scénario divergent) ; les 6 essais faits sont conservés :

| Cas | Réglage | graine | Rp (écart) | aire hors fenêtre BF / HF | Rct (IC 95 %) | τ_Rct | cible | erreur recon. max | R-hat · div · prof. max · ESS bulk/tail | alertes | durée |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A4-Randles-Rct3000 | `nonneg+ridge` | 1 | 3943.9 (+0.61 %) | 17.2 / 21.2 | 3060.1 (3040.0–3078.4) | 0.0032 s | 3000 | 2.46 % | 1.008 · 0 · 7996 · 645/492 | aucune | 224 s |
| A4-Randles-Rct3000 | `nonneg+ridge` | 2 | 3943.9 (+0.61 %) | 17.3 / 21.1 | 3059.6 (3039.5–3078.0) | 0.0032 s | 3000 | 2.44 % | 1.004 · 0 · 7991 · 1153/842 | aucune | 214 s |
| A4-Randles-Rct3500 | `nonneg+ridge` | 1 | 4599.7 (+0.65 %) | 17.7 / 23.7 | 3574.8 (3552.0–3596.8) | 0.004 s | 3500 | 2.59 % | 1.009 · 0 · 7996 · 534/352 | ess_tail_low | 216 s |
| A4-Randles-Rct3500 | `nonneg+ridge` | 2 | 4600.4 (+0.66 %) | 17.8 / 24.2 | 3574.8 (3552.4–3597.0) | 0.004 s | 3500 | 2.56 % | 1.010 · 0 · 7996 · 564/444 | aucune | 219 s |
| A4-Randles-Rct4200 | `nonneg+ridge` | 1 | 5514.5 (+0.63 %) | 22.0 / 28.2 | 4284.1 (4253.9–4319.9) | 0.005 s | 4200 | 3.23 % | 1.008 · 0 · 7996 · 565/413 | aucune | 212 s |
| A4-Randles-Rct4200 | `nonneg+ridge` | 2 | 5513.9 (+0.62 %) | 21.9 / 27.7 | 4284.9 (4253.7–4321.2) | 0.005 s | 4200 | 3.23 % | 1.009 · 0 · 7996 · 583/257 | ess_tail_low | 216 s |

**Graines 1 et 2** : 6 essais, **4/6 sans aucune alerte** ; écart de Rp +0.61 à +0.66 % ; écart de Rct à la cible +1.99 à +2.14 % ; R-hat max 1.010 ; divergences 0 au total ; ESS bulk min 534, tail min 257 ; E-BFMI min 0.73 ; durée 212-224 s. Cible de Rct dans l'IC 95 % : 0/6 ; Rp vrai dans l'IC 95 % : 0/6.

**Ensemble HMC** : 18 essais, **16/18 sans aucune alerte** ; écart de Rp +0.61 à +1.49 % ; écart de Rct à la cible +0.09 à +2.29 % ; R-hat max 1.014 ; divergences 0 au total ; ESS bulk min 534, tail min 257 ; E-BFMI min 0.70 ; durée 115-268 s. Cible de Rct dans l'IC 95 % : 8/18 ; Rp vrai dans l'IC 95 % : 0/18.


#### Mode `optimize` (aperçu) avec le même réglage (`results_engine_final_optimize.jsonl`)

| Cas | Réglage | graine | Rp (écart) | aire hors fenêtre BF / HF | Rct (IC 95 %) | τ_Rct | cible | erreur recon. max | R-hat · div · prof. max · ESS bulk/tail | alertes | durée |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | `nonneg+ridge` | 1234 | 130.1 (+0.05 %) | 0.0 / 0.0 | 49.9 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.54 % | — | aucune | 2 s |
| A3-60pts-1e5-1e-1 | `nonneg+ridge` | 1234 | 130.2 (+0.13 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 0.76 % | — | aucune | 1 s |
| A3-40pts-1e5-1e-1 | `nonneg+ridge` | 1234 | 130.2 (+0.13 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 0.82 % | — | aucune | 1 s |
| A3-80pts-1e6-1e-1 | `nonneg+ridge` | 1234 | 130.1 (+0.11 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 0.77 % | — | aucune | 1 s |
| A3-60pts-bruit0.5% | `nonneg+ridge` | 1234 | 130.5 (+0.42 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.62 % | — | aucune | 1 s |
| A3-60pts-bruit0.5%-seed1 | `nonneg+ridge` | 1 | 130.5 (+0.42 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.64 % | — | aucune | 1 s |
| A3-60pts-bruit0.5%-seed2 | `nonneg+ridge` | 2 | 130.4 (+0.34 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.67 % | — | aucune | 1 s |
| A3-60pts-bruit0.5%-seed3 | `nonneg+ridge` | 3 | 130.5 (+0.40 %) | 0.0 / 0.0 | 50.1 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.64 % | — | aucune | 1 s |
| A4-Randles-Rct3000 | `nonneg+ridge` | 1234 | 3908.1 (-0.30 %) | 5.8 / 2.0 | 3053.4 (nan–nan) | 0.0032 s | 3000 | 1.07 % | — | aucune | 1 s |
| A4-Randles-Rct3500 | `nonneg+ridge` | 1234 | 4569.8 (-0.00 %) | 7.9 / 1.5 | 3577.0 (nan–nan) | 0.004 s | 3500 | 1.03 % | — | aucune | 1 s |
| A4-Randles-Rct4200 | `nonneg+ridge` | 1234 | 5475.3 (-0.09 %) | 11.5 / 2.5 | 4300.8 (nan–nan) | 0.005 s | 4200 | 1.23 % | — | aucune | 1 s |
| A4-Randles-Rct5200 | `nonneg+ridge` | 1234 | 6772.7 (-0.11 %) | 8.9 / 2.4 | 5267.8 (nan–nan) | 0.005 s | 5200 | 1.11 % | — | aucune | 1 s |

**MAP, réglage retenu** : 12 essais, **12/12 sans aucune alerte** ; écart de Rp -0.30 à +0.42 % ; écart de Rct à la cible -0.22 à +2.40 %

Contre-épreuve — mêmes cas en MAP avec les réglages AMONT (`nonneg=False, init_from_ridge=False`), via le moteur : les optima dégénérés de DRT-1 sont désormais **tous signalés** :

| Cas | Réglage | graine | Rp (écart) | aire hors fenêtre BF / HF | Rct (IC 95 %) | τ_Rct | cible | erreur recon. max | R-hat · div · prof. max · ESS bulk/tail | alertes | durée |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3-repo-71pts-1e5-1e-2 | `default` | 1234 | 130.3 (+0.23 %) | 0.0 / 0.3 | 80.1 (nan–nan) | 0.1 s | 50 @ 1e-03 s | 0.34 % | — | aucune | 4 s |
| A3-60pts-1e5-1e-1 | `default` | 1234 | -159.4 (-222.59 %) | 0.0 / 0.1 | -160.9 (nan–nan) | 0.002 s | 50 @ 1e-03 s | 362.71 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 2 s |
| A3-40pts-1e5-1e-1 | `default` | 1234 | 130.1 (+0.07 %) | 0.1 / 0.0 | 49.9 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 0.27 % | — | aucune | 3 s |
| A3-80pts-1e6-1e-1 | `default` | 1234 | 139.8 (+7.50 %) | -0.1 / 0.5 | 80.2 (nan–nan) | 0.1 s | 50 @ 1e-03 s | 2.00 % | — | aucune | 4 s |
| A3-60pts-bruit0.5% | `default` | 1234 | -159.1 (-222.36 %) | 0.0 / 0.1 | -160.6 (nan–nan) | 0.002 s | 50 @ 1e-03 s | 363.39 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 2 s |
| A3-60pts-bruit0.5%-seed1 | `default` | 1 | 129.5 (-0.40 %) | -0.5 / 0.2 | 49.9 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.39 % | — | aucune | 3 s |
| A3-60pts-bruit0.5%-seed2 | `default` | 2 | 129.4 (-0.45 %) | -0.4 / 0.2 | 49.8 (nan–nan) | 0.001 s | 50 @ 1e-03 s | 1.44 % | — | aucune | 2 s |
| A3-60pts-bruit0.5%-seed3 | `default` | 3 | -159.2 (-222.46 %) | 0.0 / 0.1 | -160.7 (nan–nan) | 0.002 s | 50 @ 1e-03 s | 363.45 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 1 s |
| A4-Randles-Rct3000 | `default` | 1234 | -5677.9 (-244.84 %) | 1.7 / 2.0 | -6162.6 (nan–nan) | 0.004 s | 3000 | 384.11 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 1 s |
| A4-Randles-Rct3500 | `default` | 1234 | -6552.3 (-243.38 %) | 1.7 / 4.5 | -7136.4 (nan–nan) | 0.005 s | 3500 | 377.48 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 1 s |
| A4-Randles-Rct4200 | `default` | 1234 | -7925.4 (-244.62 %) | 2.3 / 5.2 | -8465.2 (nan–nan) | 0.005 s | 4200 | 391.77 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 2 s |
| A4-Randles-Rct5200 | `default` | 1234 | -9277.5 (-236.84 %) | 2.1 / 12.8 | -10142.5 (nan–nan) | 0.008 s | 5200 | 371.55 % | — | rp_nonpositive, reconstruction_error, gamma_negative | 1 s |


## 7. Choix et limites connues

**Pourquoi ce réglage.** Dans l'ordre des critères (sur la matrice HMC restreinte du §3.0 et les études
ciblées) :

1. *Ne jamais être faux en silence* (DRT-1/DRT-2) : en HMC, seul `nonneg` produit des chaînes dont les
   diagnostics permettent de **croire** le résultat ; la DRT non contrainte donne un Rp juste mais des
   chaînes non mélangées (R-hat jusqu'à 2,3 à 2 × 200, encore 1,6-1,9 à 4 × 500) — le moteur l'alerterait à chaque arc RC net,
   ce qui revient à ne pas avoir de DRT utilisable.
2. *Justesse de la grandeur de calibration* : Rct (pic dans la fenêtre) à ≤ 0,4 % sur les 2 RC et
   +1,3 à +2,4 % sur Randles — même biais de méthode que les autres réglages.
3. *Cohérence MAP / HMC* : le même réglage est correct en MAP (12/12) et en HMC ; sur le seul scénario
   où les deux candidats divergent en MAP (80 pts, §3.1 point 5), c'est le seul correct dans les deux modes.
4. *Convergence démontrée* : sur 18 ajustements HMC, 0 divergence, R-hat ≤ 1,015, E-BFMI ≥ 0,70 ; ESS
   ≥ 400 sur 16/18 (§6).

**Limites connues (cas où le réglage échoue ou reste imparfait).**

* **Rp en HMC biaisé de +0,6 à +1,5 %** (moyenne a posteriori sous contrainte ≥ 0, masse hors fenêtre —
  §4) et **IC 95 % de Rp ne contenant pas la vraie valeur** : ne pas présenter cet IC comme l'incertitude
  de Rp ; Rp reste utile comme ordre de grandeur et comme repli signalé (`rct_source = 'rp_fallback'`).
* **Rct : biais de méthode ≈ +2 % sur Randles** (fenêtre ±3 ln τ qui inclut une partie de la diffusion),
  commun à tous les réglages ; l'IC 95 % de Rct est l'incertitude a posteriori de **cette aire**, pas de
  la résistance de transfert du circuit (il n'inclut pas 3000 Ω pour Rct = 3000 Ω).
* **Pas de DRT négative** : `nonneg=True` interdit par construction les contributions négatives
  (boucles inductives, pseudo-inductance d'adsorption). Sur de telles données, la garde « erreur de
  reconstruction > 10 % » doit alerter ; non testé ici faute de jeu synthétique correspondant.
* **ESS de queue parfois sous 400** : 2 essais sur 18 (Randles 3500 graine 1 : 352 ; Randles 4200
  graine 2 : 257) lèvent `ess_tail_low` avec 2000 tirages. Alerte de **précision** des quantiles extrêmes
  (IC), pas de validité : R-hat ≤ 1,011, 0 divergence, Rp et Rct identiques à ±0,1 % aux autres graines.
  Réponse documentée : relancer avec `samples=4000` (ou changer de graine) ; l'interface doit afficher
  l'alerte, pas la masquer.
* **Échecs connus du test lent — à ne pas prendre pour une régression.** Constatés le 2026-10-01,
  identiques sur `1749d98` et après l'étape 4 (Linux, 4 cœurs, CmdStan 2.36.0, cmdstanpy 1.3.0,
  cvxopt 1.3.3), dans `tests/test_drt_engine.py::test_default_hmc_is_correct_and_converged_on_drt1_cases`,
  graine Stan 1234 (défaut du moteur) :
  - `[A3-60pts-sans-bruit]` (2 RC, 60 pts 1e5→1e-1 Hz, sans bruit) : **1 divergence** (R-hat max
    1,0037) → échoue sur `divergences == 0` ;
  - `[A4-Randles-Rct3500]` (bruit 0,5 %, graine de bruit 3500) : **`ess_tail_low`, ESS tail min 344 <
    400** → échoue sur `alerts == []` (même alerte de précision que le point précédent).

  Au §6, ces deux cas passent sans alerte à la même graine et aux mêmes versions : une graine Stan ne
  fixe donc pas le résultat d'une machine à l'autre (cause probable : compilation de CmdStan et
  arithmétique flottante — non démontrée). Le test s'arrête avant de vérifier Rct : dans cet
  environnement, ces deux cas ne le couvrent pas. Un échec de ces deux seules paramétrisations, avec
  ces symptômes, n'est pas une régression ; un autre cas, ou un autre symptôme, en est une.
* **Coût** : 115 à 268 s par spectre sur 4 cœurs (saturation de profondeur à `adapt_delta` 0,99 sur les
  spectres à arcs nets). `mode='optimize'` (~1 s) reste disponible comme aperçu, sans diagnostic de
  convergence ni intervalle.
* **Seuil de reconstruction (10 %)** calé à bruit 0,5 % (pire cas correct : 4,05 %) : risque de fausse
  alerte à bruit fort non évalué.
* **DRT-10 partiellement ouvert** : une ondulation située **dans** la fenêtre peut encore être prise pour
  un pic (observé en MAP non contraint : Rct = 80 Ω au lieu de 50 Ω sur 2 des 12 cas ; pas observé avec
  le réglage retenu).
* **Réglages non retenus, échecs non détectés par les gardes** : MAP `default` sur 80 pts (Rp +7,5 %,
  aucune alerte) ; MAP `nonneg` seul sur 80 pts (+7,8 %). Les gardes détectent Rp ≤ 0, reconstruction
  > 10 % et aire négative > 25 %, pas un décalage modéré de Rp à reconstruction correcte.
* **Portée** : spectres synthétiques (2 RC idéaux, Randles/CPE/diffusion bornée), bruit 0,5 %, 40 à
  80 points, 1e6→1e-2 Hz. **Non vérifié** : bruit plus fort, points aberrants, dérive (spectres non KK),
  plus de trois constantes de temps, Windows / versions CmdStan du poste cible (déjà signalé par l'audit).
  Premier test réel sous Windows : CmdStan 2.39.0 se trouvait dans le dossier d'installation et était
  celui enregistré. **Toute version autre que 2.36.0 reste non vérifiée** : `setup_drt_bayesien.py`
  enregistre désormais la 2.36.0 de préférence (`drt/cmdstan_version.py`), et l'application signale
  explicitement tout repli sur une autre version.
* `sigma_min` (plancher de bruit amont, 0,002 en unités réduites) et la grille τ (10 pts/décade, une
  décade au-delà des fréquences mesurées) sont laissés aux valeurs amont : non explorés.

## 8. Reproduire

```bash
# depuis la racine du dépôt, avec cvxopt 1.3.3, cmdstanpy 1.3.0 et CmdStan 2.36.0 (CMDSTAN défini)
python drt/validation/run_matrix.py --mode optimize --out /tmp/opt.jsonl --jobs 4          # §2  (~2 min)
# §3.1, périmètre du §3.0 (a)-(c) :
python drt/validation/run_matrix.py --mode sample --out /tmp/smp.jsonl --jobs 2 --settings init_from_ridge nonneg
python drt/validation/run_matrix.py --mode sample --out /tmp/smp.jsonl --jobs 2 --settings default \
    --cases A3-60pts-1e5-1e-1 A3-60pts-bruit0.5% A4-Randles-Rct3000
python drt/validation/run_matrix.py --mode sample --out /tmp/smp.jsonl --settings nonneg+ridge \
    --cases A3-80pts-1e6-1e-1        # scénarios divergents en MAP : report.py --hmc-scope les liste
python drt/validation/report.py /tmp/smp.jsonl --hmc-scope /tmp/opt.jsonl
python drt/validation/run_matrix.py --mode sample --out /tmp/adapt.jsonl --settings nonneg+ridge \
    --cases A3-repo-71pts-1e5-1e-2 A3-60pts-bruit0.5% A3-80pts-1e6-1e-1 A4-Randles-Rct3500 \
    --chains 4 --warmup 500 --samples 500 --adapt-delta 0.99                                # §3.2
python drt/validation/run_engine_study.py --out /tmp/final.jsonl --settings nonneg+ridge  # §6 (~1 h)
python drt/validation/report.py /tmp/opt.jsonl
```
