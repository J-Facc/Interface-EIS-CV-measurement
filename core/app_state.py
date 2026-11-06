"""Clés partagées de ``st.session_state`` et leur réinitialisation (AUDIT.md §2.5, B-STATE).

Logique pure : les fonctions reçoivent l'état (``st.session_state`` ou tout mapping
mutable) en argument — aucun import Streamlit dans ``core/``.

Bug corrigé (B-STATE)
---------------------
Les pages remettaient à ``None`` des clés MORTES (``eis_session``, ``eis_validation``),
jamais lues, tandis que les clés réellement produites par l'analyse n'étaient jamais
effacées : après un rechargement de ZIP, un nouvel import ou une re-validation du
prétraitement, les pages d'analyse et d'export réaffichaient l'ANCIENNE analyse, sans
avertissement. Variante plantante (B-STATE-b) : un nouvel import ne remettait pas
``preprocessing_done`` à False — la page EIS passait sa garde puis levait ``KeyError``
sur ``experiment_clean``.

Règle : toute modification des DONNÉES (import, prétraitement) invalide TOUS les
résultats d'analyse qui en dérivent, systématiquement, via les fonctions ci-dessous.
"""

from __future__ import annotations

from typing import MutableMapping

#: Résultats d'analyse dérivés de ``experiment_clean`` — produits par les pages
#: EIS (``eis_sessions``, ``eis_validations``, ``eis_normalized``) et CV (``cv_sessions``),
#: lus par ces pages et par l'export.
#: ``eis_drt_store`` = registre des DRT par spectre (``core.drt_recompute.STORE_KEY`` ; recopié
#: ici car ce module ne doit rien importer). Il référence des ``FitResult`` des sessions :
#: le garder après un nouvel import les rattacherait à des spectres qui n'existent plus.
ANALYSIS_RESULT_KEYS = ("eis_sessions", "eis_validations", "eis_normalized", "cv_sessions",
                        "eis_drt_store")

#: Valeurs initiales des clés partagées (``app.py``). Les résultats d'analyse valent
#: None : « pas encore calculé », testé par ``not state.get(key)`` dans les pages.
SHARED_DEFAULTS = {
    "experiment": None,
    "experiment_clean": None,
    "import_validated": False,
    "preprocessing_done": False,
    "exclusions": {},
    "validation_results": None,
    "eis_config": None,
    **{k: None for k in ANALYSIS_RESULT_KEYS},
}


def init_shared_state(state: MutableMapping) -> None:
    """Crée les clés partagées absentes (n'écrase jamais une valeur existante)."""
    for key, default in SHARED_DEFAULTS.items():
        if key not in state:
            state[key] = dict(default) if isinstance(default, dict) else default


def reset_analysis_results(state: MutableMapping) -> None:
    """Invalide TOUS les résultats d'analyse (EIS et CV) — à appeler dès que
    ``experiment_clean`` change (re-validation du prétraitement)."""
    for key in ANALYSIS_RESULT_KEYS:
        state[key] = None


def reset_for_new_experiment(state: MutableMapping) -> None:
    """Nouvelle expérience (nouvel import validé ou ZIP rechargé) : invalide les
    résultats d'analyse, le verdict KK du prétraitement et le prétraitement lui-même
    (``preprocessing_done`` = False, ``experiment_clean`` retiré — corrige B-STATE-b).

    L'appelant qui reconstruit immédiatement un prétraitement (ZIP sauvegardé APRÈS
    prétraitement) repose ``experiment_clean``/``preprocessing_done`` ensuite.
    """
    reset_analysis_results(state)
    state["validation_results"] = None
    state["preprocessing_done"] = False
    state.pop("experiment_clean", None)


def preprocessing_ready(state: MutableMapping) -> bool:
    """True si une page d'analyse peut lire ``experiment_clean`` sans KeyError."""
    return bool(state.get("preprocessing_done")) and state.get("experiment_clean") is not None
