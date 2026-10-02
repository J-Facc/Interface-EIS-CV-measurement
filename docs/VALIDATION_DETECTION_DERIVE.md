# Détection de dérive entre réplicats — calibration empirique des critères

> Question posée : remplacer le verdict de dérive affiché à l'utilisateur (`core/validator.py`,
> `_detect_drift`) par le jugement de cohérence des réplicats que porterait déjà le measurement
> model (`core/measurement_model.py`), pour qu'un seul critère de stationnarité subsiste
> (AUDIT.md ERR-4 / ERR-6).
>
> **Conclusion : AUCUN remplacement n'est fait.** Le measurement model ne contient aucun test de
> stationnarité entre réplicats. Ses trois quantités candidates restent au niveau nominal sous H0,
> mais n'ont **aucune puissance** contre une dérive entre balayages : leur taux d'alerte ne
> dépasse pas leur taux de fausse alarme, même pour une dérive de Rct de 20 % (§4). Les
> substituer à `_detect_drift` remplacerait un critère qui sonne toujours par un critère qui ne
> sonne jamais. Le code applicatif est donc inchangé ; ce document fournit les chiffres et les
> options (§6).
>
> Date : 2026-10-02 · Reproductible par `docs/validation_derive/run_drift_study.py`. Résultats
> bruts versionnés, une ligne JSON par groupe, environnement compris :
> `docs/validation_derive/results.jsonl`.

## Sources

| Réf. | Source |
|---|---|
| [A92], [A95b], [A95c], [O04], [OT] | voir `docs/MEASUREMENT_MODEL.md` (measurement model, structure d'erreur, KK) |
| [BW] | Bates & Watts, *Nonlinear Regression Analysis and Its Applications*, Wiley, 1988 (test F emboîté, linéarisation de la covariance) |
| [C54] | Cochran, *Biometrics* 10 (1954) 101 — statistique d'hétérogénéité Q |
| [W27] | Wilson, *J. Am. Stat. Assoc.* 22 (1927) 209 — intervalle de confiance d'une proportion |

## 1. Ce que le measurement model juge réellement (lecture du code)

Le measurement model contient trois quantités qui pourraient passer pour un jugement de
cohérence des réplicats. Aucune n'en est un.

| Quantité (nom exact) | Où | Loi sous H0 et seuil | Ce qu'elle teste réellement |
|---|---|---|---|
| `ErrorStructure.equality_pvalue`, `equal_re_im` | `_choose_structure` | F(4, m − 8) sur les sommes de carrés relatives des σ empiriques ; rejet si p < `alpha = 0,05` | **σ_r = σ_j** : l'égalité des variances du **bruit** sur Re et Im ([A95b]). C'est une hypothèse sur la forme du bruit, pas sur la stationnarité. |
| `VoigtModel.chi2`, `dof` (un par réplicat, `ErrorStructureEstimate.voigt_models`) | `characterize_error_structure` | χ²(dof) si σ est exacte ; aucun seuil n'est appliqué dans le code | Adéquation du Voigt **à son propre réplicat**. σ est estimée sur ces mêmes résidus, donc χ²ᵣ ≈ 1 par construction. |
| `MeasurementModelAnalysis.kk_conform` (`kk_message`) | `analyze_replicates`, `check_kk_consistency`, `fits.kk_validation.kk_verdict` | comptage des \|z\| > 2 ~ Binomiale(n, 0,0455), quantile 95 %, risque global `KK_FALSE_ALARM = 0,05`, Bonferroni sur n + 1 tests | Conformité Kramers-Kronig **de chaque spectre** et de la moyenne. Détecte une non-stationnarité **pendant** un balayage. |

**Pourquoi aucune ne peut voir une dérive ENTRE balayages.** Un Voigt est régressé **par
réplicat** : `characterize_error_structure` appelle `fit_voigt` sur chaque réplicat k. Son
docstring le revendique d'ailleurs : *« le résidu r_k(ω) = Z_k(ω) − M_k(ω) retire la part
déterministe KK-conforme, Y COMPRIS une dérive lente d'un réplicat à l'autre »*. Ce choix est
voulu : σ(ω) ne doit pas compter la dérive comme du bruit. Mais il a une conséquence directe : un
Rct qui change d'un balayage à l'autre est absorbé par M_k, et ni les résidus, ni σ, ni le test
σ_r = σ_j n'en gardent trace. Pour le verdict KK, un spectre stationnaire pendant son balayage est
KK-conforme quelle que soit sa valeur de Rct. La moyenne de spectres conformes est elle aussi
conforme, puisque les transformées KK sont linéaires (docstring d'`analyze_replicates`).

**Seuil de référence.** Le niveau 5 % est déjà celui de tout le module : `f_test_alpha = 0.05`,
`_choose_structure(alpha=0.05)`, `KK_FALSE_ALARM = 0.05`, et `q_pvalue < 0.05` dans
`fits/orazem_fit.py`. Les taux ci-dessous sont lus contre ce niveau nominal de 5 %.

**Le seul test inter-réplicats existant est hors du measurement model.** C'est le **Q de Cochran**
([C54]) de `fits/orazem_fit.aggregate_parameter` :
Q = Σ (θ̂_k − θ̄_w)²/v_k, de loi χ²(n − 1) sous H0. Il est calculé après le fit Orazem, sur les
paramètres du circuit de l'utilisateur. `fit_replicate_group` n'alerte que sur le paramètre
cible, à p < 0,05, par le message « dispersion inter-réplicats … Q de Cochran … dérive ou
non-stationnarité probable ». Il est mesuré ici à titre de comparaison.

## 2. Le critère actuel et ses consommateurs

`_detect_drift` (`core/validator.py`) calcule, à chaque fréquence, CV = écart-type / moyenne des
|résidus Im| du test KK entre réplicats. Il conclut à une dérive si CV > 0,5
(`DRIFT_CV_THRESHOLD`) sur plus de 20 % des fréquences. Appelé par `_finalize`, lui-même appelé
par `validation_from_analysis` et `validation_without_structure`. Le pipeline
(`core/pipeline.py:_analyze_group`) passe par ces deux fonctions.

| Consommateur | Usage |
|---|---|
| `ui/tabs.py` (onglet Measurement model & fit Orazem) | bandeau rouge « **Drift inter-réplicats** : Drift détecté sur X % des fréquences… » si `drift_warning` |
| `plotting/kk_plots.py` | suffixe « — drift inter-réplicats » dans le titre de la figure KK si `drift_detected` |
| journal (`logger.warning`) | même message |

**`_detect_drift` ne filtre rien en aval.** Il ne modifie jamais `all_valid`. Seul le
non-chevauchement des plages KK-valides le fait, et ce n'est pas ce critère. Il n'alimente donc
ni le verdict KK, ni `core/calibration.py`, ni les exports.

Conséquence pour le diagnostic sur données réelles : si presque tous les groupes, Probe compris,
apparaissent **« non conformes »**, ce verdict vient de `kk_conform`, c'est-à-dire du test KK du
measurement model, et non de `_detect_drift`. Le docstring de `core/calibration.py` (« la
détection de dérive qui l'alimente ») est inexact sur ce point.

Une remarque de forme : la statistique est un CV de **résidus**, centrés sur zéro. Pour du bruit
blanc, |moyenne| et écart-type sont du même ordre, donc CV ≈ 1 > 0,5 quoi qu'il arrive. Le
critère mesure la présence de bruit, pas une dérive.

## 3. Protocole

**Données.** Randles de l'Annexe A (`tests/synthetic_data.py`) : Re 200 Ω, R'e 20 Ω, Cb 1 nF,
Qdl 2 µF, α 0,9, τ_d 0,5 s, Rct 3000 Ω, R_D 900 Ω. 40 points de 1e5 à 1e-1 Hz, balayés de HF à BF.
Le temps est le rang du point : le point i du réplicat k est au rang k·N + i.

**Bruits** (gaussien, indépendant d'un point à l'autre et d'un réplicat à l'autre) :

| Nom | Forme | σ à Rct (BF) |
|---|---|---|
| `orazem_x0.5` / `x1` / `x2` | σ_r = σ_j = s·(0,004\|Zj\| + 0,004\|Zr − Re\| + 0,5) (`ORAZEM_NOISE` × s) | ≈ 0,2 / 0,4 / 0,8 % de \|Z\| |
| `relatif_0.5%` / `1%` | σ_re = p\|Zre\|, σ_im = p\|Zim\| (`noisy_arrays`, données de l'ERR-4 ; viole σ_r = σ_j) | 0,5 / 1 % par composante |

**Scénarios.**

* `stationnaire` : n réplicats du même spectre. H0, 100 groupes par bruit, n = 3, plus n = 5 au
  bruit nominal.
* `entre_balayages` : un paramètre constant pendant chaque balayage, P_k = P·(1 + a·k/(n − 1)).
  C'est une dérive **entre** réplicats (remontage, conditionnement lent entre deux mesures).
  Paramètre Rct, a ∈ {0,5 ; 1 ; 2 ; 5 ; 10 ; 20 %}, 40 groupes par amplitude. Aussi Qdl
  (paramètre **non cible**), a ∈ {2 ; 5 ; 10 ; 20 %}.
* `continue` : P(t) = P·(1 + a·t), t de 0 à 1 sur l'ensemble de la série. La dérive a lieu
  **pendant ET entre** les balayages, et chaque balayage voit une fraction a/n de la dérive
  totale. Paramètre Rct, a ∈ {1 ; 2 ; 5 ; 10 ; 20 %}.

**Chemins exécutés.** Ce sont exactement ceux de la production :
`analyze_replicates` → `validation_from_analysis` (donc `_detect_drift`) →
`fit_replicate_group` avec `FitOptions()` (8 départs, réglage du pipeline) et le circuit
`RANDLES_EXPRESSION`. Les spécifications sont celles de `tests/test_orazem_fit.py`.

**Critères relevés.** Une alerte = « dérive / incohérence signalée ».

| Colonne | Règle |
|---|---|
| `detect_drift` | `ValidationResult.drift_detected` (critère actuel) |
| `eq_rejet` | `equality_pvalue < 0,05` (σ_r = σ_j rejetée) |
| `kk_non_conforme` | `kk_conform is False` |
| `voigt_chi2` | min_k P(χ²(dof_k) > χ²_k) < 0,05/n (Bonferroni, comme le verdict KK) |
| `cochran_Rct` | Q de Cochran du paramètre cible, p < 0,05 (alerte actuelle du fit Orazem) |
| `cochran_tout_param` | min des p de Q sur les 8 paramètres < 0,05/8 |
| `explo_T_chi2`, `explo_T_F` | **exploratoire, non implémenté** : voir §5 |

Une proportion à 100 groupes a une demi-largeur d'IC 95 % ([W27]) de ±4 à ±8 points autour de
5 à 20 % ; à 40 groupes, de ±7 à ±15 points.

## 4. Résultats

### 4.1 Faux positifs — réplicats stationnaires (niveau nominal attendu : 5 %)

| Bruit | n | `detect_drift` | `eq_rejet` | `kk_non_conforme` | `voigt_chi2` | `cochran_Rct` | `cochran_tout_param` |
|---|---|---|---|---|---|---|---|
| orazem ×0,5 | 3 | **100 %** | 7 % | 1 % | 15 % | 1 % | 3 % |
| orazem ×1 | 3 | **100 %** | 5 % | 6 % | 12 % | 12 % | 21 % |
| orazem ×2 | 3 | **100 %** | 6 % | 10 % | 9 % | 23 % | 54 % |
| relatif 0,5 % | 3 | **97 %** | **100 %** | 11 % | 2 % | 2 % | 0 % |
| relatif 1 % | 3 | **100 %** | **100 %** | 11 % | 1 % | 2 % | 3 % |
| orazem ×1 | 5 | **100 %** | 4 % | 5 % | 10 % | 17 % | 28 % |

Fraction médiane de fréquences signalées par `_detect_drift` sous H0 : 62 % (orazem ×0,5), 85 %
(×1), 88 % (×2), 98 % (×1, n = 5), 36 % (relatif 0,5 %), 52 % (relatif 1 %). Toutes dépassent
le seuil de 20 % dans 97 à 100 % des groupes. C'est l'ERR-4 (72-95 % des fréquences)
reproduit ; la fraction **augmente** avec le niveau de bruit et avec le nombre de réplicats.

### 4.2 Puissance — dérive de Rct ENTRE balayages, n = 3 (40 groupes par case)

| Bruit | Amplitude totale | `detect_drift` | `eq_rejet` | `kk_non_conforme` | `voigt_chi2` | `cochran_Rct` |
|---|---|---|---|---|---|---|
| orazem ×1 | 0,5 % | 100 % | 8 % | 3 % | 5 % | 18 % |
| | 1 % | 100 % | 5 % | 10 % | 15 % | 55 % |
| | 2 % | 100 % | 3 % | 5 % | 10 % | 95 % |
| | 5 % | 100 % | 3 % | 8 % | 13 % | 100 % |
| | 10 % | 100 % | 8 % | 0 % | 13 % | 100 % |
| | 20 % | 100 % | 5 % | 5 % | 13 % | 100 % |
| orazem ×2 | 0,5 → 20 % | 100 % (toutes) | 3 à 10 % | 0 à 15 % | 5 à 20 % | 20 % → 100 % |
| relatif 0,5 % | 0,5 → 20 % | 95 à 100 % | 100 % (toutes) | 5 à 25 % | 0 à 8 % | 58 % → 100 % |

**Lecture.** Les trois quantités du measurement model (`eq_rejet`, `kk_non_conforme`,
`voigt_chi2`) gardent, à 20 % de dérive, le taux d'alerte qu'elles ont sans dérive (§4.1) : leur
puissance est **nulle**. C'est la conséquence attendue du §1, puisque chaque Voigt absorbe le Rct
de son réplicat. `_detect_drift` signale tout, avec ou sans dérive : il ne sépare pas non plus
les deux situations.

### 4.3 Puissance — dérive CONTINUE de Rct (pendant et entre les balayages), n = 3

| Bruit | Amplitude totale | `detect_drift` | `eq_rejet` | `kk_non_conforme` | `voigt_chi2` | `cochran_Rct` |
|---|---|---|---|---|---|---|
| orazem ×1 | 1 % | 100 % | 8 % | 5 % | 8 % | 23 % |
| | 2 % | 100 % | 0 % | 13 % | 13 % | 78 % |
| | 5 % | 100 % | 3 % | 15 % | 20 % | 100 % |
| | 10 % | 100 % | 3 % | **55 %** | 13 % | 100 % |
| | 20 % | 100 % | 8 % | **90 %** | 53 % | 100 % |
| relatif 0,5 % | 1 → 5 % | 95 à 98 % | 98 à 100 % | 18 à 20 % | 3 à 8 % | 85 → 100 % |
| | 10 % | 100 % | 100 % | **83 %** | 5 % | 100 % |
| | 20 % | 98 % | 100 % | **100 %** | 75 % | 100 % |

Le verdict KK ne détecte que la part de dérive **interne à chaque balayage** (a/n ≈ 3 à 7 % par
balayage pour a = 10 à 20 %). C'est cohérent avec sa calibration propre
(`MEASUREMENT_MODEL.md` §5 : +5 % pendant un balayage → 15/40, +10 % → 32/40) : il joue son
rôle, qui n'est pas celui-ci.

### 4.4 Dérive d'un paramètre NON cible (Qdl) entre balayages, orazem ×1, n = 3

| Amplitude | `detect_drift` | `eq_rejet` | `kk_non_conforme` | `voigt_chi2` | `cochran_Rct` | `cochran_tout_param` |
|---|---|---|---|---|---|---|
| 2 % | 100 % | 5 % | 0 % | 18 % | 8 % | 25 % |
| 5 % | 100 % | 0 % | 10 % | 18 % | 13 % | 18 % |
| 10 % | 100 % | 8 % | 3 % | 3 % | 20 % | 55 % |
| 20 % | 100 % | 3 % | 3 % | 13 % | 15 % | 85 % |

L'alerte de Cochran porte sur le seul paramètre cible : elle ne voit pas une dérive de la double
couche, même à 20 %.

### 4.5 Le Q de Cochran (hors measurement model) n'est pas calibré non plus

Sous H0, ses fausses alarmes croissent avec le bruit : 1 %, puis 12 %, puis 23 % (n = 3), et 17 %
à n = 5. Diagnostic, sur les 100 groupes stationnaires de chaque bruit : la dispersion réelle des
Rct ajustés vaut 0,90× (orazem ×0,5), 1,77× (×1) et 1,89× (×2) l'incertitude linéarisée du fit
(`params_std`). Les écarts normalisés atteignent |z| = 4,5. La covariance linéarisée ([BW])
sous-estime l'incertitude de Rct dès que le bruit rend sensible la corrélation Rct / R_D / τ_d du
Randles. Q hérite de cette sous-estimation. Le test n'est pas en cause : c'est v_k qui est trop
petit. La correction relève de `fits/orazem_fit.py`, hors du périmètre de cette tâche (réglages
validés à ne pas toucher). Q dépend en outre du circuit choisi par l'utilisateur, et il n'existe
qu'après le fit.

## 5. Piste exploratoire — dispersion BRUTE rapportée à σ(ω) du measurement model

> **Non implémentée.** Mesurée seulement pour savoir si un critère dérivé du measurement model
> *pourrait* exister. Ce n'est pas ce qui était demandé, et aucune décision n'est prise dessus.

Le measurement model sépare précisément ce que `_detect_drift` confond : σ(ω) est le bruit **sans**
la dérive. La dérive est l'excès de la dispersion brute des Z entre réplicats sur ce σ :

T = Σ_ω Σ_k [ (Zr_k − Z̄r)²/σ_r² + (Zj_k − Z̄j)²/σ_j² ],  ν = 2N(n − 1),  σ = structure d'erreur évaluée sur Z̄.

| Situation | T/ν moyen (é.-t.) | Alerte χ²(ν), p < 0,05 | Alerte F(ν, ν_σ) |
|---|---|---|---|
| H0 orazem ×0,5 / ×1 / ×2 (n = 3) | 0,95 / 0,97 / 0,97 (≈ 0,075 ; attendu 1 ± 0,11) | 1 % / 0 % / 0 % | 0 % |
| H0 relatif 0,5 % / 1 % | 0,82 / 0,84 | 0 % | 0 % |
| H0 orazem ×1, n = 5 | 0,97 (0,046) | 0 % | 0 % |
| Rct entre balayages, orazem ×1 : 0,5 / 1 / 2 / ≥ 5 % | — | 3 / 25 / **100** / 100 % | 0 / 3 / 98 / 100 % |
| Rct entre balayages, orazem ×2 : 1 / 2 / ≥ 5 % | — | 5 / 28 / 100 % | 0 / 13 / 100 % |
| Rct entre balayages, relatif 0,5 % : 1 / 2 / ≥ 5 % | — | 45 / 100 / 100 % | 8 / 100 / 100 % |
| Rct continue, orazem ×1 : 1 / 2 / ≥ 5 % | — | 13 / 78 / 100 % | 0 / 40 / 100 % |
| **Qdl** entre balayages, orazem ×1 : 2 / ≥ 5 % | — | **80 / 100 %** | 48 / 100 % |

Cette piste est **conservatrice** sous H0 : environ 0 % de fausses alarmes, pour un niveau
nominal de 5 %. T/ν reste sous 1, parce que σ est estimé sur des résidus que le Voigt a déjà
réduits, et que la forme régressée lisse σ. Sur bruit relatif, la forme d'Orazem décrit mal le
bruit (T/ν ≈ 0,83). La piste a pourtant une vraie puissance, qui ne dépend ni du circuit ni du
paramètre : 2 % de dérive de Rct sont détectés 40 fois sur 40 au bruit nominal, et une dérive de
Qdl de 5 % aussi. En faire un critère supposerait :

1. d'établir sa loi sous H0 compte tenu de l'estimation de σ sur les mêmes données (simulation
   ou bootstrap paramétrique), plutôt que χ² ou F nominaux, faux ici ;
2. de la valider sur des réplicats réels stationnaires (Probe) ;
3. une décision explicite d'ajouter ce test au measurement model.

C'est un chantier à part entière.

## 6. Décision et options

**Décision (tâche du 2026-10-02) : rien n'est remplacé.** Les quantités du measurement model sont
calibrées pour ce qu'elles testent : σ_r = σ_j à 4-7 % sous bruit conforme, KK à 1-11 %. Mais
elles ne testent pas la dérive entre réplicats, et leur puissance contre elle est nulle (§4.2).
Afficher l'une d'elles comme « verdict de stationnarité » promettrait à l'utilisateur une
vérification qui n'a pas lieu. Le test σ_r = σ_j le ferait même **à l'envers** : il sonne à
100 % sur des réplicats stationnaires dont le bruit est simplement relatif par composante.

`_detect_drift` reste, en l'état, un critère sans valeur diagnostique : il sonne à 97-100 % avec
ou sans dérive. Les options, à trancher par l'utilisateur :

| Option | Effet | Coût / risque |
|---|---|---|
| A. Retirer `_detect_drift` de l'affichage, sans remplaçant | supprime une alarme rouge systématique et non informative ; plus aucun verdict de dérive affiché | l'utilisateur perd une alerte, mais elle ne portait aucune information (§4) |
| B. Garder le Q de Cochran (déjà affiché dans les alertes du fit) comme seul signal | puissant sur le paramètre cible (≥ 95 % à 2 % de dérive) | 12-23 % de fausses alarmes au bruit réaliste, aveugle aux paramètres non cibles, exige d'abord de corriger l'incertitude du fit (§4.5) |
| C. Développer la statistique du §5 comme critère du measurement model | un seul verdict, issu de σ(ω), indépendant du circuit | calibration de sa loi sous H0 et validation sur données réelles à faire |
| D. Statu quo | — | deux affichages contradictoires persistent (ERR-6) |
