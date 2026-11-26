"""Clés partagées de ``st.session_state`` (v3) et leurs invalidations.

Logique pure : les fonctions reçoivent l'état (``st.session_state`` ou tout mapping mutable),
aucun import Streamlit — testable sans navigateur.

Flux (une page n'écrit que ses propres clés) ::

    Configuration   → experiment (Experiment validée)
    Import          → raw_files, assignment, excluded_points → measurements (nettoyées)
    Analyse         → analysis_run (calcul en cours) → results (AnalysisResults)
    Export          ← experiment, measurements, results

Règle : toute modification en amont invalide TOUT ce qui en dérive (jamais de résultat
périmé affiché). Un rechargement d'archive pose experiment + measurements + results d'un
coup (``results_source = "archive"``).
"""

from __future__ import annotations

from typing import MutableMapping

EXPERIMENT = "experiment"
RAW_FILES = "raw_files"              # {file_id: {"name", "content" (bytes), "kind"}}
ASSIGNMENT = "assignment"            # [dict] — lignes de la table d'assignation
EXCLUDED_POINTS = "excluded_points"  # {file_id: [indices de points EIS retirés]}
MEASUREMENTS = "measurements"        # [Measurement] nettoyées, ou None
CLEANING_REPORT = "cleaning_report"  # [Issue] du dernier nettoyage
ANALYSIS_RUN = "analysis_run"        # core.analysis.AnalysisRun en cours, ou None
RESULTS = "results"                  # core.results.AnalysisResults, ou None
RESULTS_SOURCE = "results_source"    # "computed" | "archive" | None
THEME = "theme_mode"                 # "light" | "dark"

DEFAULTS = {
    EXPERIMENT: None,
    RAW_FILES: {},
    ASSIGNMENT: [],
    EXCLUDED_POINTS: {},
    MEASUREMENTS: None,
    CLEANING_REPORT: [],
    ANALYSIS_RUN: None,
    RESULTS: None,
    RESULTS_SOURCE: None,
    THEME: "light",
}


def init_state(state: MutableMapping) -> None:
    """Crée les clés absentes (n'écrase jamais une valeur existante)."""
    for key, default in DEFAULTS.items():
        if key not in state:
            state[key] = type(default)() if isinstance(default, (dict, list)) else default


def cancel_run(state: MutableMapping) -> None:
    """Arrête proprement un calcul en cours (tâches non démarrées annulées)."""
    run = state.get(ANALYSIS_RUN)
    if run is not None:
        try:
            run.cancel()
        finally:
            state[ANALYSIS_RUN] = None


def invalidate_results(state: MutableMapping) -> None:
    """Les DONNÉES nettoyées ont changé : résultats d'analyse périmés."""
    cancel_run(state)
    state[RESULTS] = None
    state[RESULTS_SOURCE] = None


def invalidate_measurements(state: MutableMapping) -> None:
    """Fichiers, assignation ou configuration modifiés : le nettoyage est à refaire."""
    state[MEASUREMENTS] = None
    state[CLEANING_REPORT] = []
    invalidate_results(state)


def new_experiment(state: MutableMapping, experiment) -> None:
    """Nouvelle configuration validée : tout ce qui en dérive est invalidé (les fichiers
    déposés et leur assignation sont CONSERVÉS — l'utilisateur les corrige à la page Import)."""
    state[EXPERIMENT] = experiment
    invalidate_measurements(state)


def load_archive_state(state: MutableMapping, experiment, measurements, results) -> None:
    """Chargement d'une archive : remplace TOUT l'état d'un coup, sans recalcul."""
    cancel_run(state)
    state[EXPERIMENT] = experiment
    state[RAW_FILES] = {}
    state[ASSIGNMENT] = []
    state[EXCLUDED_POINTS] = {}
    state[MEASUREMENTS] = measurements
    state[CLEANING_REPORT] = []
    state[RESULTS] = results
    state[RESULTS_SOURCE] = "archive" if results is not None else None


def has_experiment(state: MutableMapping) -> bool:
    return state.get(EXPERIMENT) is not None


def has_measurements(state: MutableMapping) -> bool:
    return bool(state.get(MEASUREMENTS))


def has_results(state: MutableMapping) -> bool:
    return state.get(RESULTS) is not None
