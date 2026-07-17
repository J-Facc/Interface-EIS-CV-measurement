# Méthode de pondération unique — structure d'erreur d'Orazem

> Mémo honnête accompagnant la refonte du fit paramétrique de Randles.
> Références : Orazem & Tribollet, *Electrochemical Impedance Spectroscopy* (Wiley),
> chap. Measurement Model / Error Structure ; Agarwal, Orazem & García-Rubio,
> *J. Electrochem. Soc.* (1992-1995).

## (a) Une seule méthode, pondérée par la structure d'erreur

Le fit paramétrique du circuit de Randles n'a plus qu'**une seule** pondération.
Les anciens modes `modulus` (`1/(α_noise·|Z|)²`) et `sigma` (`1/σ²` inter-réplicats
brut) — ainsi que le sélecteur `weight_mode` et le paramètre `alpha_noise` — ont
été **supprimés** (config, code, UI, tests).

L'optimiseur est inchangé : `scipy.optimize.least_squares`, méthode TRF, bornes,
`ftol = xtol = 1e-10`. Seule change la pondération :

```
σ_i    = α·|Z_re,i| + β·|Z_im,i| + γ·(|Z_i|²/R_m) + δ        (fits/error_structure.py)
w_re,i = w_im,i = 1/σ_i²          (absolute_sigma = True, TOUJOURS)
```

Conséquences (toutes vérifiées par des tests) :

- **Covariance** = `(JᵀJ)⁻¹` **sans** rééchelonnement par `2·cost/dof`. La branche
  « modulus » (`cov · 2·cost/dof`) a été retirée : `σ_param = sqrt(diag((JᵀJ)⁻¹))`
  (`test_covariance_not_rescaled`).
- **χ²_red** est un **vrai test d'adéquation** : `FitResult.chi2_is_valid_test = True`,
  intervalle attendu `chi2_reduced_ci = [1 − 2√(2/dof), 1 + 2√(2/dof)]`, et un
  diagnostic est émis si `χ²_red` en sort (`test_chi2_reduced_around_one_with_known_sigma`,
  `test_chi2_reduced_far_from_one_when_sigma_wrong`).
- **Signe du résidu imaginaire** cohérent entre `residuals()` (`−Z.imag − Zim`) et
  le χ² post-fit (`Zim + Z_fit.imag`) : quantités opposées, carrés identiques
  (`test_imaginary_residual_sign_equivalence`).

## (b) Origine des coefficients : caractérisés sur réplicats, réutilisés depuis persistance

Les coefficients `(α, β, γ, δ)` ne sont **pas** posés a priori. Ils sont **estimés**
selon la méthode stricte d'Orazem :

1. **σ_empirique(f)** à partir de réplicats (≥ 3, `min_replicates`) :
   - voie par défaut « réplicats-directs » : écart-type inter-réplicats par
     fréquence (`core/loader.average_replicates`, `ddof=1`, **brut, sans plancher**) ;
   - option `error_structure.voigt_based: true` : un circuit de Voigt (Lin-KK,
     `fits/kk_validation.lin_kk`) est ajusté à chaque réplicat et σ = écart-type
     des **résidus** par fréquence (le Voigt absorbe la part déterministe
     KK-consistante ; le résidu isole le bruit — plus fidèle à Orazem).
2. **Régression** de σ_empirique sur la forme structurée, en **moindres carrés
   non négatifs** (`scipy.optimize.nnls`) : coefficients ≥ 0 par construction ;
   `equal_re_im` (défaut) impose `α = β` (hypothèse d'égalité des variances Re/Im
   du measurement model). `1/R_m` est absorbé dans γ à l'estimation, donc `R_m`
   n'a pas à être connu (`test_regress_coefficients_recovers_known_structure`).
3. **Persistance** : `(α, β, γ, δ, R_m)` + horodatage + identifiant de campagne
   sont écrits dans `config/error_structure.json` (chemin configurable), en
   historique JSON, écriture atomique.

**Repli (spectre sans réplicats).** On ne ré-estime pas : on recharge les
**derniers** coefficients persistés et on les utilise pour pondérer le fit. Si
**rien** n'a jamais été caractérisé **et** qu'il n'y a pas de réplicats, le fit
est **refusé** (`ErrorStructureUnavailable`, message explicite) — **aucun** repli
sur un σ = α·|Z| arbitraire (`test_refusal_without_characterization`,
`test_persistence_roundtrip_and_reuse`).

Dans le pipeline, une **passe de caractérisation en amont** (`_characterize_error_structure_upfront`)
caractérise et persiste la structure sur le premier spectre porteur de réplicats,
avant tout fit, de sorte que même les réplicats individuels (qui n'ont pas de σ
propre) partagent la même structure d'erreur.

## (c) « Caractérisé sur ce jeu » vs « réutilisé »

`FitResult.error_structure_source ∈ {"characterized_now", "reused_persisted"}`,
avec `error_structure_timestamp` et `error_structure_coeffs`. L'onglet Paramètres
(`ui/tabs.py`) affiche cette provenance :

- **caractérisé sur ce jeu** (vert) : σ mesuré sur les réplicats de ce jeu ;
- **réutilisé** (orange) : σ hérité d'une caractérisation antérieure — valable
  sous l'hypothèse assumée que l'instrument n'a pas changé depuis.

## Choix pragmatiques assumés (honnêteté)

- **Correction √N sur le datum moyenné.** σ structuré décrit le bruit d'**une**
  mesure (l'écart-type inter-réplicats estime le bruit d'un réplicat). Le pipeline
  ajuste souvent le spectre **moyen** de N réplicats, dont la variance vaut
  `σ²/N`. On pondère donc par `1/σ_i²` avec `σ_i = σ_struct/√N` (`n_replicates`),
  ce qui reste la forme `w = 1/σ_i²` mais avec `σ_i` correctement identifié par
  datum — sans quoi χ²_red vaudrait ~`1/N` sur les moyennes et le test
  d'adéquation serait faussé. Vérifié : χ²_red ≈ 1 aussi bien sur un réplicat
  unique que sur une moyenne.
- **`equal_re_im`.** En régression, `equal_re_im=true` impose `α = β` (une colonne
  `|Z_re|+|Z_im|`). Avec `false`, on estime α et β séparément mais γ et δ restent
  communs (bruit de fond partagé) : choix pragmatique, l'hypothèse standard du
  measurement model étant `σ_re = σ_im`.
- **Terme γ·|Z|²/R_m.** Sans `R_m`, `1/R_m` est absorbé dans γ à l'estimation ;
  en mode « provided » sans `R_m`, le terme est ignoré (documenté dans
  `ErrorStructure.sigma`). Garde-fous |Z|→0 et R_m→0 documentés.
- **`voigt_based`.** Implémenté et testé (`test_voigt_based_characterization_runs`)
  et activable dans le pipeline lorsque les réplicats individuels sont attachés au
  spectre moyenné (`EISSpectrum.replicates`). La voie par défaut reste
  « réplicats-directs ».
- **Plancher legacy supprimé.** `average_replicates` ne clippe plus σ à
  `0.001·|Z̄|` : la borne inférieure de σ vient de δ (bruit de fond additif estimé),
  pas d'un plancher relatif codé en dur. Un plancher **absolu** minuscule (1e-12)
  ne subsiste que pour éviter une division par zéro pathologique (δ = 0 ∧ |Z| = 0).
