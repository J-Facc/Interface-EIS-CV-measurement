# DRT bayésienne — ce que calcule l'application, et comment lire le résultat

> Code : `drt/engine.py` (moteur de l'application), `drt/diagnostics.py` (gardes),
> `drt/bayes_drt2/` (bibliothèque tierce). Provenance exacte et patchs :
> [`drt/PROVENANCE.md`](../drt/PROVENANCE.md). Preuves numériques du choix des réglages :
> [`drt/VALIDATION_REGLAGES.md`](../drt/VALIDATION_REGLAGES.md) (matrice d'essais reproductible,
> scripts et résultats bruts dans `drt/validation/`).

## 1. Ce qu'est la DRT

Un spectre d'impédance mélange plusieurs processus qui ont chacun un temps caractéristique
τ (transfert de charge, diffusion, double couche…). La **distribution des temps de
relaxation** γ(τ) est une courbe qui répond à : « quelle résistance est associée à chaque
échelle de temps ? ». Un arc de transfert de charge apparaît comme un pic de γ(τ) ; l'aire sous
ce pic est la résistance correspondante. Contrairement au circuit équivalent, la DRT ne demande
pas de choisir un circuit à l'avance — c'est pourquoi l'application la calcule **à côté** du fit
de circuit, comme second avis.

L'application trace `ln(γ/γ₀)` en fonction de `ln(τ/τ₀)` (logarithme népérien, γ₀ = 1 Ω,
τ₀ = 1 s). Référence de la convention d'extraction de Rct : Bissessur, Man, Gamby, *Use of an
approach with a distribution of relaxation times for impedance analysis of a channel electrode
in microfluidics*, Phys. Rev. E **113**, 025502 (2026), DOI 10.1103/fn2s-z364.

**Rct de la DRT** : l'aire de γ(τ) sur ±3 unités de ln τ autour du pic **pénultième** (l'avant-
dernier pic, quand on parcourt les temps croissants : le dernier est la diffusion). Les pics
candidats sont cherchés dans la fenêtre de τ réellement mesurée. Si un seul pic existe
(`peak_single`) ou aucun (repli sur la résistance totale Rp, `rp_fallback`), le résultat est
**signalé** : la grandeur n'est alors plus tout à fait un arc de transfert de charge.

## 2. La bibliothèque : bayes-drt2

La DRT est reconstruite par **inversion hiérarchique bayésienne** (Huang, Papac, O'Hayre,
*Electrochimica Acta* 367 (2020) 137493) : γ(τ) est décomposée sur une base de fonctions
régulières, et les hyperparamètres (niveau de bruit, lissage) sont eux aussi estimés, par un
modèle écrit en Stan. Le code est un **clone identifié** (dépôt `jdhuang-csm/bayes-drt2`,
commit `99d5b60`, licence BSD-3) avec exactement deux patchs, tous deux justifiés dans
`drt/PROVENANCE.md` : `np.trapz → np.trapezoid` (compatibilité numpy 2) et l'exposition du
réglage NUTS `adapt_delta`. Rien d'autre n'est modifié ; les sha256 sont publiés.

Les calculs passent par CmdStan : un compilateur C++ est requis (installé automatiquement par
`launch.bat` / `python setup_drt_bayesien.py --ensure`, version 2.36.0, mingw-w64 sous Windows).
Sans lui, l'application fonctionne **sans** DRT et le dit.

## 3. Deux modes

| Mode | Méthode | Durée | Ce qu'on obtient |
|---|---|---|---|
| `optimize` (aperçu) | MAP : on cherche la **seule** courbe la plus probable (L-BFGS) | ~1 s par spectre | γ(τ) et Rct, **sans** diagnostic de convergence ni intervalle |
| `sample` (HMC / NUTS, **défaut du moteur**) | on tire des milliers de courbes plausibles | 2 à 5 min par spectre (4 cœurs) | γ(τ), Rct, **intervalle de crédibilité**, diagnostics de convergence |

La page EIS propose le mode (le pipeline d'analyse utilise `fit.drt.mode`, `optimize` par défaut
dans `config/default.yaml` pour garder l'analyse d'ensemble rapide) ; un spectre peut être
recalculé en `sample` à la demande depuis l'onglet « Courbes DRT ».

### HMC / NUTS en deux phrases

Le Monte-Carlo hamiltonien (HMC) explore l'ensemble des courbes compatibles avec les mesures
en faisant « rouler » un point dans le paysage de probabilité, ce qui est bien plus efficace
qu'un tirage au hasard ; NUTS règle automatiquement la longueur des trajectoires. On lance
**plusieurs chaînes indépendantes** (4) : si elles aboutissent toutes au même endroit, c'est
un signe que l'exploration a réussi.

## 4. Lire un intervalle de crédibilité (sans être statisticien)

Les tirages HMC forment un ensemble de courbes γ(τ) possibles, pondérées par leur plausibilité.
Pour chaque quantité (γ à un τ donné, Rct), l'application affiche :

* la **valeur** : la moyenne des tirages ;
* un **intervalle de crédibilité à 95 %** : les bornes 2,5 % et 97,5 % des tirages. Lecture
  directe : « compte tenu des données et du modèle, il y a 95 % de probabilité que la valeur
  soit dans cet intervalle ». C'est ce que beaucoup de gens croient à tort lire dans un
  intervalle de confiance classique.

Ce qu'un intervalle **ne dit pas** — limites mesurées dans `VALIDATION_REGLAGES.md` §7 :

* c'est l'incertitude **du modèle DRT sur la courbe**, pas celle de la résistance de transfert
  du circuit réel : pour un Rct vrai de 3000 Ω, l'intervalle de Rct DRT ne contient pas toujours
  3000 Ω (biais de méthode d'environ +2 % sur des spectres de type Randles, la fenêtre ±3 ln τ
  incluant un peu de diffusion) ;
* l'intervalle de **Rp** (résistance totale) ne contient pas la valeur vraie (moyenne biaisée de
  +0,6 à +1,5 % par la contrainte de positivité) : ne pas le présenter comme l'incertitude de
  Rp ;
* il ne remplace pas la **variabilité entre réplicats**, que l'application calcule séparément
  (écart inter-réplicats). Un intervalle étroit avec des réplicats très dispersés veut dire :
  chaque spectre est bien ajusté, mais les mesures ne se reproduisent pas.

## 5. Diagnostics et alertes : comment savoir si on peut croire le résultat

Le moteur ne se contente pas de rendre un chiffre : il lit les diagnostics de l'échantillonnage
et émet des **alertes** (`FitResult.warnings`, affichées dans l'onglet DRT et exportées).
`converged` est calculé à partir d'elles — jamais forcé à vrai.

| Diagnostic | En clair | Seuil d'alerte |
|---|---|---|
| **R̂** (R-hat) | les chaînes indépendantes sont-elles d'accord entre elles ? | > 1,05 |
| **ESS** (taille d'échantillon effective, centre et queues) | combien de tirages réellement indépendants ? Trop peu → intervalles imprécis | < 100 par chaîne (400 au total avec 4 chaînes) |
| **Divergences** | l'exploration a-t-elle « raté un virage » dans une région du paysage ? | au moins 1 |
| **E-BFMI** | l'énergie de l'échantillonneur est-elle bien explorée ? | < 0,3 |
| **Erreur de reconstruction** | γ(τ) redonne-t-elle bien le spectre mesuré ? | écart relatif maximal > 10 % |
| **Aire négative** (DRT non contrainte seulement) | la courbe contient-elle beaucoup de valeurs négatives non physiques ? | > 25 % de l'aire |

Conduite à tenir : une alerte de **convergence** (R̂, divergences, E-BFMI) → ne pas utiliser le
résultat tel quel (changer de graine ou refaire avec plus de tirages) ; une alerte
`ess_tail_low` seule est une alerte de **précision** des bornes de l'intervalle, pas de validité
du centre (R̂ et divergences restent bons) → relancer avec plus de tirages. Le mode `optimize`
n'a aucun de ces diagnostics : c'est un aperçu, pas un résultat à publier.

## 6. Réglages (constantes de `drt/engine.py`)

| Réglage | Valeur | Raison (détail et mesures : `VALIDATION_REGLAGES.md`) |
|---|---|---|
| mode | `sample` (HMC) | seul mode qui donne diagnostics et intervalles |
| `nonneg` | vrai | sans lui, les chaînes ne se mélangent pas sur des arcs nets, et le MAP tombait dans des optima absurdes (Rp < 0) sans qu'aucune garde ne le voie |
| `init_from_ridge` | vrai | indispensable au MAP, neutre en HMC |
| chaînes / warmup / tirages | 4 / 500 / 2000 | R̂ fiable (≥ 4 chaînes) ; 1000 tirages laissaient l'ESS sous le seuil |
| `adapt_delta` | 0,99 | 0,9 (défaut amont) produisait 3 à 24 divergences par ajustement |
| graine | 1234, explicite et enregistrée | reproductibilité : mêmes données, même graine → même courbe sur une même machine |

La graine ne fixe pas le résultat d'une machine à l'autre (compilation et arithmétique
flottante) : voir `VALIDATION_REGLAGES.md` §7 pour les deux cas du test lent qui peuvent
échouer sans qu'il s'agisse d'une régression.

## 7. Limites à garder en tête

* Pas de DRT négative : `nonneg` interdit les contributions négatives (boucles inductives,
  pseudo-inductance d'adsorption). Sur de telles données, la garde de reconstruction doit
  alerter ; non testé faute de jeu synthétique correspondant.
* Validée sur des spectres synthétiques (2 RC, Randles avec diffusion bornée), bruit 0,5 %, 40 à
  80 points, 1e6 → 1e-2 Hz. **Non vérifié** : bruit plus fort, points aberrants, spectres non
  Kramers-Kronig, plus de trois constantes de temps, Windows / autres versions de CmdStan.
* Une ondulation située dans la fenêtre mesurée peut encore être prise pour un pic (non observée
  avec le réglage retenu).
* Coût : plusieurs minutes par spectre en HMC ; l'analyse d'ensemble utilise donc `optimize` par
  défaut et le HMC se lance à la demande.
