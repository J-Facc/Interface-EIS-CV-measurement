# Écrire son propre circuit équivalent

Au lieu d'un circuit de Randles codé en dur, vous décrivez le circuit à ajuster
par une **expression** d'une ligne, dans une syntaxe proche de Python mais très
restreinte. Par exemple :

```text
Re + parallel(R(Rct), Q(Qdl, alpha)) + ZD_bounded(w, R_D, tau_d)
```

L'application en déduit toute seule la liste des paramètres à ajuster
(ici `Re, Rct, Qdl, alpha, R_D, tau_d`) et une fonction d'impédance `Z(ω)`.

Module : `circuit/` (`elements.py`, `parser.py`, `registry_elements.py`).
Point d'entrée : `circuit.parse_circuit(expr) -> (Z, noms_des_paramètres)`.

---

## 1. Règles d'écriture

| Vous pouvez écrire | Exemple | Signification |
|---|---|---|
| `+` | `Re + R(Rct)` | éléments **en série** (les impédances s'additionnent) |
| `parallel(a, b, …)` | `parallel(R(Rct), C(Cdl))` | éléments **en parallèle** (au moins deux branches) |
| un élément | `Q(Qdl, alpha)` | voir la liste au §2 |
| un nom seul | `Re` | un paramètre, traité comme une résistance (impédance réelle) |
| un nombre | `R(1000)`, `L(-1e-6)`, `2.5e3` | une valeur **fixée**, jamais ajustée |
| `-  *  /  **`, parenthèses | `Re + 1/(1j*w*Cdl)` | arithmétique complexe usuelle |
| `w` | `1j*w*L1` | la pulsation ω (rad/s) — nom **réservé** |
| `1j` | `1/(1j*w*C1)` | l'unité imaginaire (écrire `1j`, pas `j`) |
| retours à la ligne, `# commentaire` | | acceptés |

### Comment les paramètres sont détectés

* **Tout nom** autre que `w` et les noms d'éléments devient un paramètre à
  ajuster : dans `Re + R(Rct)`, `Re` et `Rct` sont deux paramètres.
* **Un nombre n'est jamais un paramètre** : `R(1000)` est une résistance fixe de 1 kΩ.
* **Un même nom utilisé deux fois est UN SEUL paramètre** :
  `R(Ra) + R(Ra)` n'a qu'un paramètre `Ra` (la somme vaut 2·Ra).
* Les paramètres sont listés dans l'ordre de leur première apparition.
* Le **nom est libre** : c'est la **position** dans l'élément qui fixe le sens.
  Dans `ZD_bounded(w, D, delta)`, `D` joue le rôle de `R_D` (Ω) et `delta` celui
  de `tau_d` (s), quels que soient les noms choisis.

### Arguments d'un élément

Les arguments d'un élément (`R`, `C`, `Q`…) sont **uniquement** des noms de
paramètres ou des nombres (éventuellement précédés de `-`). Pas d'expression ni
d'appel imbriqué : `R(2*Rct)` ou `R(R(1))` sont refusés. Pour combiner des
éléments, utilisez `+` et `parallel(...)`.

La pulsation `w` est ajoutée automatiquement : on écrit `R(Rct)`, pas `R(w, Rct)`.
L'écrire explicitement en **premier** argument est toléré (`ZD_bounded(w, R_D, tau_d)`) ;
ailleurs, c'est une erreur.

### Noms de paramètres valides

Lettres ASCII, chiffres et `_`, en commençant par une lettre (`Rct`, `Q_dl`,
`alpha2`) — 64 caractères au plus. Sont refusés : les noms commençant par `_`,
les noms réservés de Python (`print`, `open`, `sum`, `min`, `type`, `id`, `lambda`…),
les noms d'éléments, `w`, ainsi que `j` (écrire `1j`) et `omega` (écrire `w`),
qui cacheraient une erreur de frappe en créant un paramètre fantôme.

Seuls les caractères **ASCII** sont acceptés : écrire `alpha`, pas `α` ; `Ohm`
n'a pas sa place dans l'expression (les unités sont implicites : SI).

---

## 2. Éléments disponibles

ω est la pulsation (rad/s), j l'unité imaginaire. Les unités sont celles du SI.

| Élément | Paramètres (ordre) | Impédance | Remarque |
|---|---|---|---|
| `R(r)` | r (Ω) | Z = r | résistance |
| `C(c)` | c (F) | Z = 1 / (jωc) | capacité |
| `L(l)` | l (H) | Z = jωl | inductance (câbles, artefacts HF) |
| `Q(q, alpha)` | q (S·s^α), α (—) | Z = 1 / (q·(jω)^α) | CPE ; α = 1 ⇒ capacité pure ; phase −α·90° |
| `W(sigma)` | σ (Ω·s^−½) | Z = σ(1 − j)/√ω | Warburg semi-infini, phase −45° |
| `Wo(r, tau)` | r (Ω), τ (s) | Z = r·coth(√(jωτ)) / √(jωτ) | Warburg fini **réflectif** (frontière imperméable) : capacitif en basse fréquence |
| `Ws(r, tau)` | r (Ω), τ (s) | Z = r·tanh(√(jωτ)) / √(jωτ) | Warburg fini **transmissif** (couche de Nernst) : Z → r en basse fréquence |
| `ZD_bounded(R_D, tau_d)` | R_D (Ω), τ_d (s) | Z = R_D·tanh(√(jωτ_d)) / √(jωτ_d) | diffusion bornée, canal microfluidique (formulation Bissessur) — ex-`Z_D` de `fits/physics.py`, reprise à l'identique |
| `parallel(Z1, Z2, …)` | — | Z = 1 / (1/Z1 + 1/Z2 + …) | au moins deux branches |

> ⚠ **`ZD_bounded` et `Ws` sont la même formule** (tanh). « Bornée » désigne ici
> une couche de diffusion d'épaisseur finie, **pas** la frontière imperméable du
> Warburg réflectif `Wo` (coth). Ne pas confondre `ZD_bounded` et `Wo`.

**Sources.** Les formules sont les formes standard de M. E. Orazem & B. Tribollet,
*Electrochemical Impedance Spectroscopy*, 2e éd., Wiley, 2017 : chap. 4 (R, C, L,
association en parallèle), chap. 11 (diffusion : W, Wo, Ws), chap. 14 (CPE, avec la
notation de Brug *et al.*, J. Electroanal. Chem. 176 (1984) 275). Les numéros
d'équation n'ont pas été vérifiés contre l'ouvrage ; chaque docstring de
`circuit/elements.py` rappelle la formule exacte implémentée. `ZD_bounded`
reprend `fits/physics.py:Z_D`, dont la provenance déclarée est la « bounded-diffusion
(Bissessur) formulation for a microfluidic channel » (voir `METHODES.md` §4.1).

**Bornes par défaut** (`circuit/registry_elements.py`) : borne basse 0 pour toutes
les grandeurs (éléments passifs), 0 < α ≤ 1 pour le CPE. **Aucune borne haute
générique** n'est proposée : elle dépend du système et doit être fixée par
l'utilisateur ou estimée depuis le spectre.

---

## 3. Exemple complet annoté : le Randles complet de l'application

Le circuit historiquement codé en dur (`fits/physics.py:Z_randles_full`) :

```
Z_eq = R'e + (Rct + Z_D) / [1 + Qdl·(jω)^α·(Rct + Z_D)]
Z    = Re + Z_eq / [1 + jω·Cb·Z_eq]
```

se réécrit :

```text
Re + parallel(
    Re_prime + parallel(
        R(Rct) + ZD_bounded(R_D, tau_d),   # transfert de charge + diffusion bornée, en série
        Q(Qdl, alpha)                      # double couche (CPE), en parallèle
    ),
    C(Cb)                                  # capacité de contournement, en parallèle de Z_eq
)
```

* `Re` — résistance d'électrolyte, en série avec tout le reste ;
* `Re_prime` — résistance série interne à Z_eq ;
* `parallel(R(Rct) + ZD_bounded(R_D, tau_d), Q(Qdl, alpha))` — branche faradique
  (Rct puis diffusion) en parallèle de la double couche : c'est exactement
  `(Rct + Z_D) / [1 + Qdl·(jω)^α·(Rct + Z_D)]` ;
* `parallel(Z_eq, C(Cb))` — c'est exactement `Z_eq / [1 + jω·Cb·Z_eq]`.

Paramètres détectés (ordre d'apparition) :
`Re, Re_prime, Rct, R_D, tau_d, Qdl, alpha, Cb` — les 8 paramètres du Randles
complet. Le test `tests/test_circuit_parser.py::test_full_randles_reconstruction_matches_fits_physics`
vérifie que cette expression reproduit `Z_randles_full` à 10⁻¹² près.

Variante simplifiée : `Re + parallel(R(Rct), Q(Qdl, alpha)) + ZD_bounded(w, D, delta)`
(diffusion en série, hors de la branche faradique — ce n'est **pas** le même circuit).

Utilisation depuis Python :

```python
import numpy as np
from circuit import parse_circuit

Z, noms = parse_circuit("Re + parallel(R(Rct), C(Cdl))")
# noms == ["Re", "Rct", "Cdl"]
w = 2 * np.pi * np.logspace(-1, 5, 50)
z = Z(w, Re=100.0, Rct=1e3, Cdl=1e-6)   # tableau complexe, même forme que w
```

`Z` exige **exactement** les paramètres détectés (un oubli ou une faute de frappe
lève `CircuitEvaluationError`), chacun étant un nombre réel.

---

## 4. Messages d'erreur

| Exception | Cause |
|---|---|
| `CircuitSyntaxError` | expression mal formée : parenthèse manquante, mauvais nombre d'arguments (`Q(1)`), expression dans un argument d'élément, `j` au lieu de `1j`… |
| `UnsafeExpressionError` | construction **interdite** par la liste blanche (voir §5) |
| `CircuitEvaluationError` | appel de `Z` avec un paramètre manquant, inconnu ou non numérique |

Toutes héritent de `CircuitError` (elle-même un `ValueError`). Les messages
peuvent citer des fragments de l'expression saisie : les afficher en texte brut
dans l'interface, jamais comme du HTML.

---

## 5. Pourquoi c'est sûr — transparence

L'expression est du texte fourni par l'utilisateur : elle ne doit **jamais**
pouvoir exécuter du code arbitraire (lire un fichier, lancer une commande…).
Le module applique une **liste blanche** — tout ce qui n'est pas explicitement
autorisé est refusé — et non une liste noire, qui oublie toujours un cas.

1. **Le texte n'est jamais exécuté.** Ni `eval`, ni `exec`, ni `compile` en code
   exécutable : aucune instruction Python n'est produite à partir de votre texte.
2. **Contrôle textuel** : longueur ≤ 2000 caractères, ASCII imprimable uniquement
   (ce qui écarte les homoglyphes Unicode, les caractères invisibles et l'octet
   nul), et la séquence `__` est refusée (elle ouvre l'accès aux internes de Python).
3. **Analyse syntaxique pure** par `ast.parse(expr, mode="eval")` : le texte est
   découpé en arbre, rien n'est exécuté ; une seule expression, pas d'instruction
   (`import` est donc impossible).
4. **Liste blanche des nœuds** : l'arbre ne peut contenir que des additions,
   soustractions, multiplications, divisions, puissances, le moins unaire, des
   appels, des noms et des nombres. Tout le reste est refusé : accès aux attributs
   (`w.__class__`), indexation (`x[0]`), `lambda`, compréhensions, chaînes de
   caractères, f-strings, `:=`, comparaisons, arguments nommés, `*args`…
5. **Validation structurelle** : un appel n'est permis que si la fonction appelée
   est **directement** un nom de la liste `R, C, L, Q, W, Wo, Ws, ZD_bounded,
   parallel` — `print(...)`, `open(...)`, `exec(...)`, `getattr(...)`,
   `(lambda: 0)()` ou `R(1)(2)` sont refusés. Les noms réservés de Python ne
   peuvent pas servir de paramètres. Profondeur, taille et nombre de paramètres
   sont bornés.
6. **Évaluation fermée** : l'arbre validé est traduit en un assemblage de petites
   fonctions qui n'appellent que les fonctions de `circuit/elements.py` et les
   quatre opérations. Aucune fonction intégrée de Python n'est accessible. Les
   calculs se font en flottants numpy : `10**10**10` vaut l'infini au lieu de
   bloquer le serveur.

Les couches 2, 4 et 5 se recouvrent **volontairement** : chacune arrête à elle
seule les attaques connues. `tests/test_circuit_parser.py` vérifie le rejet de
plus de 80 attaques (dont `__import__('os').system(...)`, `exec(...)`,
`(lambda: None)()`, `[x for x in ().__class__.__bases__[0].__subclasses__()]`,
`w.__class__`, `print(...)`, `open(...)`), le rejet par chaque couche prise
isolément, l'absence d'effet de bord (fichier témoin jamais créé, `eval`/`exec`
jamais appelés) et un fuzzing aléatoire de 20 000 expressions.
