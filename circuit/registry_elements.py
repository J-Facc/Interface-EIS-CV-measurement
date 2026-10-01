"""Description des éléments disponibles dans un circuit utilisateur (pour l'UI).

``ELEMENTS`` associe à chaque nom d'élément un dictionnaire :

* ``n_params``  : nombre de paramètres physiques (sans compter ``w``) ; ``None``
  pour le combinateur variadique ``parallel`` ;
* ``params``    : rôle canonique de chaque paramètre, dans l'ordre d'appel ;
* ``units``     : unité de chaque paramètre ;
* ``formula``   : formule lisible ;
* ``description`` : phrase courte pour l'interface ;
* ``default_bounds`` : une borne ``(bas, haut)`` par paramètre ; ``None`` = non
  borné de ce côté ;
* ``bounds_note`` : justification des bornes.

Politique des bornes par défaut
-------------------------------
Seules les bornes justifiables en GÉNÉRAL sont données :

* borne basse 0 pour toute grandeur d'un élément passif (résistance, capacité,
  inductance, pré-facteur de CPE, coefficient de Warburg, temps caractéristique) :
  une valeur négative n'a pas de sens physique pour le dipôle modélisé ;
* 0 < alpha <= 1 pour le CPE : par définition la phase du CPE est comprise entre
  0° (résistance, alpha = 0) et -90° (capacité idéale, alpha = 1).

⚠ AUCUNE borne HAUTE n'est proposée (``None``) : les ordres de grandeur dépendent
entièrement du système (Ω à GΩ, pF à mF, µs à ks) et aucune valeur générique ne se
justifie. L'interface doit les faire saisir ou les estimer depuis le spectre.
Les bornes de l'ancien ``fits/randles_full.py`` (supprimé ; ex. alpha >= 0.6, Re >= 100 Ω) sont propres
à ce montage et ne sont volontairement PAS reprises ici.

La cohérence de ce registre avec les signatures de ``circuit/elements.py`` et avec
la liste blanche de ``circuit/parser.py`` est vérifiée par les tests.
"""

from __future__ import annotations

_POSITIVE = (0.0, None)
_POSITIVE_NOTE = "Borne basse 0 (élément passif) ; borne haute non justifiable en général."

ELEMENTS: dict[str, dict] = {
    "R": {
        "n_params": 1,
        "params": ["r"],
        "units": ["Ω"],
        "formula": "Z = r",
        "description": "Résistance pure.",
        "default_bounds": [_POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "C": {
        "n_params": 1,
        "params": ["c"],
        "units": ["F"],
        "formula": "Z = 1 / (j·ω·c)",
        "description": "Capacité pure.",
        "default_bounds": [_POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "L": {
        "n_params": 1,
        "params": ["l"],
        "units": ["H"],
        "formula": "Z = j·ω·l",
        "description": "Inductance pure (câblage, artefacts haute fréquence).",
        "default_bounds": [_POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "Q": {
        "n_params": 2,
        "params": ["q", "alpha"],
        "units": ["S·s^α", "—"],
        "formula": "Z = 1 / (q·(j·ω)^alpha)",
        "description": "Élément à phase constante (CPE) — capacité non idéale.",
        "default_bounds": [_POSITIVE, (0.0, 1.0)],
        "bounds_note": (
            "q : borne basse 0, pas de borne haute générique. alpha : 0 < alpha <= 1 "
            "par définition (phase entre 0° et -90°)."
        ),
    },
    "W": {
        "n_params": 1,
        "params": ["sigma"],
        "units": ["Ω·s^-1/2"],
        "formula": "Z = sigma·(1 - j)/√ω",
        "description": "Warburg semi-infini (diffusion en milieu infini, phase -45°).",
        "default_bounds": [_POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "Wo": {
        "n_params": 2,
        "params": ["r", "tau"],
        "units": ["Ω", "s"],
        "formula": "Z = r·coth(√(j·ω·tau)) / √(j·ω·tau)",
        "description": "Warburg fini réflectif (frontière imperméable) — capacitif en BF.",
        "default_bounds": [_POSITIVE, _POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "Ws": {
        "n_params": 2,
        "params": ["r", "tau"],
        "units": ["Ω", "s"],
        "formula": "Z = r·tanh(√(j·ω·tau)) / √(j·ω·tau)",
        "description": "Warburg fini transmissif (couche de Nernst) — tend vers r en BF.",
        "default_bounds": [_POSITIVE, _POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "ZD_bounded": {
        "n_params": 2,
        "params": ["R_D", "tau_d"],
        "units": ["Ω", "s"],
        "formula": "Z = R_D·tanh(√(j·ω·tau_d)) / √(j·ω·tau_d)",
        "description": (
            "Diffusion bornée, canal microfluidique (formulation Bissessur, ex-Z_D de "
            "fits/physics.py). Même formule que Ws."
        ),
        "default_bounds": [_POSITIVE, _POSITIVE],
        "bounds_note": _POSITIVE_NOTE,
    },
    "parallel": {
        "n_params": None,
        "params": [],
        "units": [],
        "formula": "Z = 1 / (1/Z1 + 1/Z2 + …)",
        "description": "Association en parallèle d'au moins deux impédances.",
        "default_bounds": [],
        "bounds_note": "Combinateur sans paramètre propre.",
    },
}
