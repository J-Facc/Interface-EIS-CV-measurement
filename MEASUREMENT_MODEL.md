# Measurement model, structure d'erreur, Kramers-Kronig et fit Orazem

> Référence de la méthode (étape 4 de la refonte). Code : `core/measurement_model.py`,
> `fits/kk_validation.py`, `fits/orazem_fit.py`, `core/regression_stats.py` ; tests :
> `tests/test_measurement_model.py`, `tests/test_orazem_fit.py`, `tests/test_validator.py`.
>
> **État de la bascule.** Le prétraitement (`core/validator.py`) utilise déjà le
> measurement model pour le verdict Kramers-Kronig. Le pipeline d'analyse
> (`core/pipeline.py`) ajuste encore l'ancien Randles à 8 paramètres
> (`fits/randles_full.py`, pondéré par `fits/error_structure.py` et sa persistance
> JSON) : il basculera sur `fits/orazem_fit.py` à l'étape 5, avec le nettoyage de ces
> modules. Les numéros d'équation des sources n'ont pas pu être vérifiés depuis
> l'environnement de développement : chaque formule est écrite en clair ci-dessous et
> dans les docstrings.

## Sources

| Réf. | Source |
|---|---|
| [A92] | Agarwal, Orazem, García-Rubio, *J. Electrochem. Soc.* 139 (1992) 1917 — measurement models I |
| [A95b] | Agarwal, Crisalle, Orazem, García-Rubio, *JES* 142 (1995) 4149 — II, contribution stochastique |
| [A95c] | Agarwal, Orazem, García-Rubio, *JES* 142 (1995) 4159 — III, cohérence Kramers-Kronig |
| [O04] | Orazem, *J. Electroanal. Chem.* 572 (2004) 317 — identification de la structure d'erreur |
| [OT] | Orazem & Tribollet, *Electrochemical Impedance Spectroscopy*, 2e éd., Wiley, 2017 |
| [S14] | Schönleber, Klotz, Ivers-Tiffée, *Electrochim. Acta* 131 (2014) 20 — Lin-KK, critère µ |
| [B95] | Boukamp, *JES* 142 (1995) 1885 — Lin-KK |
| [BW] | Bates & Watts, *Nonlinear Regression Analysis and Its Applications*, Wiley, 1988 |
| [CW] | Cook & Weisberg, *Residuals and Influence in Regression*, 1982 |
| [BBA] | Buckland, Burnham, Augustin, *Biometrics* 53 (1997) 603 ; Burnham & Anderson, *Model Selection and Multimodel Inference*, 2002 |
| [DL] | DerSimonian & Laird, *Control. Clin. Trials* 7 (1986) 177 ; Cochran, *Biometrics* 10 (1954) 101 |

## 1. Chaîne, dans l'ordre imposé

```
réplicats d'un groupe
  └─ analyze_replicates()                       core/measurement_model.py
       ├─ characterize_error_structure()        σ(ω) — ou ErrorStructureUnavailable : ARRÊT du groupe
       └─ check_kk_consistency() × (n + 1)      verdict KK du groupe  ──► AFFICHÉ
  └─ fit_replicate_group(…, analysis, …)        fits/orazem_fit.py (exige l'analyse)
       ├─ fit_spectrum() par réplicat (σ) et sur la moyenne (σ/√n)
       └─ agrégation par paramètre
```

`fit_replicate_group` prend l'analyse en argument obligatoire : le fit **ne peut
pas** précéder la structure d'erreur ni le verdict KK.

## 2. Measurement model de Voigt ([A92])

`Z(ω) = R0 + Σₖ Rₖ/(1 + jωτₖ) [+ 1/(jωC)]`, τₖ **régressés** (et non fixés sur une
grille comme Lin-KK). Éléments ajoutés un à un ; l'élément K + 1 n'est gardé que si :

1. la régression converge ;
2. **test F** des modèles emboîtés ([BW] §3.10) :
   `F = [(χ²_K − χ²_{K+1})/2] / [χ²_{K+1}/(n_obs − P_{K+1})]`, p < 0,05 ;
3. **tous** les paramètres restent significatifs : intervalle ±2σ de chaque Rₖ, τₖ
   excluant zéro — c'est **le** critère de [A92].

Pourquoi le test F plutôt que l'AIC : modèles emboîtés (cadre exact du test F) ;
valide quand σ n'est connu qu'à un facteur près (1ʳᵉ passe, pondération |Z|) ; l'AIC
équivaut à un risque ≈ 13,5 % d'ajouter un élément inutile, qui absorberait du bruit
et ferait sous-estimer σ. Détail dans la docstring de `fit_voigt`.

**Ce que faisait l'ancien code** (`fits/error_structure.py`, vérifié) : aucun ajout
successif ni critère de parcimonie. Voie par défaut : écart-type brut des Z entre
réplicats (une dérive y compte comme du bruit). Option `voigt_based` : Lin-KK à τ
fixés avec M = min(100, 2n/3) imposé.

## 3. Structure d'erreur ([A95b], [O04])

Par fréquence et composante, sur les résidus du measurement model de **chaque**
réplicat :

`σ̂(ω) = s_k[ rₖ(ω) / √(1 − hₖ(ω)) ] / c4(n)`

| Terme | Rôle | Mesuré |
|---|---|---|
| résidu du MM **de chaque réplicat** | retire la part déterministe, dérive lente entre réplicats comprise | — |
| s_k, ddof = 1 | centrage : retire le défaut d'ajustement commun | — |
| √(1 − h) (levier, [CW]) | un modèle ajusté absorbe h_ii du bruit | sans : σ sous-estimé de 5,6 à 7,4 % ; avec : +0,4 à +1,7 % (3 et 5 réplicats) |
| c4(n) | biais de s à petit n (−11 % pour n = 3) | — |

**Aucun plancher** (l'ancien `_compute_sigma` imposait 0,1 % de |Z|). Des réplicats
identiques (σ̂ ≤ 1e-9·|Z|, arrondi numérique) sont **refusés**, pas « planchérisés ».

σ̂ est régressé sur `σ = α|Zj| + β|Zr − R_sol| + γ|Z|² + δ` ([O04]), par moindres
carrés **non négatifs itérativement repondérés** (s̃ a un écart-type ∝ σ), avec
|Zj|, |Zr|, |Z|² pris sur la moyenne des MM (régresseurs sans bruit). γ est le γ/R_m
de [O04] : R_m n'a pas à être connu et le terme est toujours évalué (l'ancien code le
régressait puis l'ignorait).

**σ_r = σ_j est testée, pas imposée.** C'est le sens de l'hypothèse du measurement
model ([A95b]) — l'ancien `equal_re_im` l'interprétait à tort comme « α = β ». Un
test F compare la structure commune (4 coefficients) à deux structures (4 + 4) ;
σ_r = σ_j n'est rejetée que si p < 0,05. Mesuré : bruit conforme → retenue 94-96 %
du temps ; bruit relatif par composante des jeux synthétiques du dépôt (σ_re ∝ |Zre|,
σ_im ∝ |Zim|) → rejetée (p ≈ 1e-11), sans quoi 2 groupes sur 4 étaient déclarés à
tort non conformes KK.

Deux passes : pondération |Z| (σ connue à un facteur près), puis pondération par la
structure obtenue (σ absolue), démarrée à chaud.

**Pas de persistance.** Rien n'est écrit sur disque ni réutilisé d'une session ou
d'un groupe à l'autre (AUDIT.md ERR-2). Si la structure n'est pas caractérisable
(moins de 3 réplicats, grilles de fréquences incompatibles, valeurs non finies,
réplicats identiques, MM non ajustable), `ErrorStructureUnavailable` est levée avec
un `user_message` complet (cause, remède, « aucun fit n'a été réalisé ») : l'analyse
du groupe s'arrête (AUDIT.md ERR-1).

## 4. Kramers-Kronig par le même measurement model ([A95c])

Le Voigt vérifie KK **par construction**. Test : ajuster la partie **imaginaire**
seule, avec K et τ initiaux repris du MM complexe du même spectre, pondérée par σ ;
prédire la partie réelle à la constante R0 près (ajustée sur Re) ; comparer.

- Variance du résidu prédit : `σᵢ² − 1/Σw + gᵢᵀCgᵢ` (méthode delta, g corrigé de
  l'ajustement de R0).
- **Incertitude sur K** : prédictions des modèles à K et K + 1 éléments moyennées par
  poids d'Akaike, variance inconditionnelle de [BBA]. Sans cela : écart-type des
  résidus normalisés 1,04-1,08 et 10-13 % de rejets à tort par test ; avec : 0,94 et
  4,4 %.
- Sens Re → Im écarté : Re seule résout moins d'éléments et manque la relaxation de
  contournement haute fréquence (fausses alarmes mesurées).

## 5. Critère UNIQUE de verdict (`fits/kk_validation.kk_verdict`)

Remplace les deux critères sans source (AUDIT.md ERR-6 : résidu < 2 % par point +
paliers 10/25 %, et `fit.drt_kk_tol` = 5 %, supprimé de la configuration).

- résidu normalisé `z = r/s`, s = écart-type sous H0 ; point hors bande si |z| > 2
  (convention 2σ de [A95c], [OT]) ;
- sous H0, nombre de sorties ≈ Binomiale(n, 0,0455) ; **conforme** si ce nombre ne
  dépasse pas son quantile 95 % (80 résidus → 7 tolérés) ;
- **sans niveau de bruit : aucun verdict** (`None`), jamais un seuil de repli.

Verdict de **groupe** : n réplicats + leur moyenne testés, chacun au risque 5 %/(n + 1)
(Bonferroni).

| Calibrage mesuré (bruit d'Orazem, 100 groupes conformes) | Fausses alarmes |
|---|---|
| 3 réplicats × 40 points | 4/100 |
| 3 réplicats × 60 points | 1/100 |
| 5 réplicats × 40 points | 0/100 |

| Puissance (1 réplicat sur 3 dont Rct dérive pendant son balayage, 40 groupes) | Détecté |
|---|---|
| +5 % | 15/40 |
| +10 % | 32/40 |
| +20 % | 40/40 |

## 6. Lin-KK ([B95], [S14]) — test rapide, indicatif

Corrigé (AUDIT.md ERR-5) : `c = 0,85` est enfin utilisé, µ suit la définition de [S14]
(`µ = 1 − Σ|Rₖ<0|/Σ|Rₖ≥0|` ; l'ancien code calculait une autre quantité, ce qui
neutralisait le contrôle), pondération 1/|Z|. Lecture retenue : **début de la
dernière plage où µ < c**, car µ(M) plonge transitoirement sous c sur une grille de τ
fixés (1ᵉʳ franchissement : 3,6 % à 28 % d'erreur sur des spectres conformes). Même
ainsi, une relaxation de Debye idéale reste mal décrite : c'est pourquoi le verdict
de référence est celui du §4. Au prétraitement, sans structure d'erreur, seuls les
résidus Lin-KK sont affichés, sans verdict.

## 7. Fit Orazem sur circuit libre (`fits/orazem_fit.py`)

Entrées : `Z_func, param_names = circuit.parse_circuit(expr)`, un `ParameterSpec`
(guess, bornes, échelle facultative) **par nom de paramètre**, fourni par
l'interface (aucune heuristique), et le `target_param` désigné comme signal de
calibration. Pondération 1/σ² par composante, `absolute_sigma` toujours vrai.

| Défaut de l'ancien moteur | Correction |
|---|---|
| FIT-1 `except Exception` → guess renvoyé | échec numérique → `converged=False` + meilleur itéré + alerte ; spécification invalide → `FitSpecificationError` ; toute autre exception remonte |
| FIT-2 `inv(JᵀJ)`, `sqrt(abs(diag))` | SVD de la jacobienne équilibrée ; conditionnement κ calculé (`fit_diagnostics`) ; paramètre non identifiable → écart-type **infini** ; différence unilatérale pour un paramètre sur une borne singulière (ex. Cb = 0 : sans cela 6 fits sur 40 perdaient TOUS leurs écarts-types) ; circuit non défini des deux côtés ou SVD non convergée → tous les écarts-types à inf + alerte « incertitudes indisponibles », jamais d'exception |
| FIT-3 pas d'échelle, mono-départ | `x_scale` = ordre de grandeur de chaque paramètre ; départs multiples reproductibles, meilleur χ² gardé, alerte si un autre départ atteint un χ² équivalent (Δχ² < 1) avec une cible différente de plus de 2σ |
| FIT-4 butée à 1 % de la valeur de la borne | butée = borne à moins d'**un écart-type** (contrainte active, σ gaussien invalide) |
| FIT-5 deux formes de poids | une seule : (σ_r, σ_j) |

Solveur : `least_squares`, TRF si une borne est finie, LM sinon ; jacobienne finale
par différences centrées. Intervalle attendu de χ²ᵣ au niveau 2σ : χ²(ν)/ν si σ est
connue, **F(ν, ν_σ)** quand σ a été estimée sur ν_σ degrés de liberté (structure
d'erreur) — plus large, il compte l'incertitude sur σ elle-même.

**Par réplicat puis agrégation** : chaque réplicat (σ d'une mesure) et la moyenne
(σ/√n) sont ajustés ; sur les réplicats **convergés**, pour chaque paramètre :
moyenne, écart-type inter-réplicats s, incertitude intra-fit typique √v̄, et
incertitude de la moyenne `√(max(s², v̄)/n)` (estimateur des moments d'un effet
aléatoire, [DL]) ; Q de Cochran signale une dispersion supérieure à l'incertitude
intra-fit (dérive ou non-stationnarité). Valeur de calibration recommandée :
`target.mean ± target.sem`.

## 8. `FitResult`

`Rct`/`Rct_std` figés remplacés par `target_param`, `target_value`, `target_std`.
Retirés : `Rct_sigma`, `drt_S`, `drt_lnGamma`, `chi2_is_valid_test`. Conservés :
`chi2_reduced_ci` (la nouvelle UI doit afficher χ²ᵣ avec son intervalle attendu),
`kk_passed`/`kk_residuals` (écrits par l'ancien pipeline jusqu'à l'étape 5, désormais
`None` faute de structure d'erreur à cet endroit). Ajouté : `fit_diagnostics`.

**À afficher par l'UI (étape 5) — pas seulement stocké.** Aujourd'hui, aucun écran ne
lit `fit_diagnostics` : κ n'atteint l'utilisateur que sous forme d'alerte texte (rang
déficient, ou κ > 1/√ε ≈ 6,7·10⁷), via `warnings`. Lors de la bascule, chaque fit
affiché doit montrer à côté de ses paramètres : κ (jacobienne équilibrée), le rang
sur P, les paramètres non identifiables, ceux dérivés par différence unilatérale
(`jacobian_one_sided`), les bornes actives, et χ²ᵣ avec `chi2_reduced_ci`.
