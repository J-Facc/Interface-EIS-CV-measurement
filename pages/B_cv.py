"""Page B — Analyse CV seule.

Charge les courbes de voltammétrie cyclique, extrait les pics redox,
normalise par le probe, et produit une calibration par électrode via
ui.tabs.render_cv_tabs (run_cv_pipeline + plotting.cv_plots).
"""

import streamlit as st

from core.cv_loader import load_cv_file, average_cv_replicates
from core.cv_pipeline import run_cv_pipeline
from ui.tabs import render_cv_tabs


# ---------------------------------------------------------------------------
# Chargement des fichiers depuis experiment_clean
# ---------------------------------------------------------------------------

def _build_cv_assignments_electrode(experiment: dict, elec_idx: int) -> list:
    """Construit la liste cv_assignments (compatible run_cv_pipeline) pour
    une électrode donnée."""
    cv = experiment["calibration"]["cv"]
    concs = experiment["concentrations"]
    probe_dict = (experiment.get("probe") or {}).get("cv") or {}

    assignments: list = []

    probe_rep_files = probe_dict.get(f"electrode_{elec_idx}") or []
    for ri, pf in enumerate(probe_rep_files):
        if pf is None:
            continue
        content = pf.read()
        pf.seek(0)
        assignments.append({
            "content": content,
            "filename": f"probe_e{elec_idx}_r{ri+1}",
            "step": "probe",
            "concentration": 0.0,
        })

    reps_for_conc = cv.get(f"electrode_{elec_idx}", [])
    for ci, conc in enumerate(concs):
        rep_files = reps_for_conc[ci] if ci < len(reps_for_conc) else []
        for r, rf in enumerate(rep_files):
            if rf is None:
                continue
            content = rf.read()
            rf.seek(0)
            assignments.append({
                "content": content,
                "filename": f"e{elec_idx}_c{ci+1}_r{r+1}",
                "step": "hybridization",
                "concentration": conc,
            })

    return assignments


def _load_bare_cv(experiment: dict, elec_idx: int):
    """Charge et moyenne les fichiers « électrode nue » CV d'une électrode.

    Réutilise les loaders EXISTANTS (load_cv_file + average_cv_replicates) —
    aucun parsing maison, pour ne pas réintroduire le bug « temps chargé comme
    courant ». Retourne un CVScan (moyenne, UNE seule trace) ou None. Référence
    d'AFFICHAGE SEULE : jamais passée à run_cv_pipeline.
    """
    bare_dict = (experiment.get("bare") or {}).get("cv") or {}
    bare_files = bare_dict.get(f"electrode_{elec_idx}") or []

    scans = []
    for ri, bio in enumerate(bare_files):
        if bio is None:
            continue
        try:
            content = bio.read()
            bio.seek(0)
            scans.append(load_cv_file(content, f"bare_e{elec_idx}_r{ri+1}", 0.0, "bare"))
        except Exception:
            pass

    if not scans:
        return None
    return average_cv_replicates(scans) if len(scans) > 1 else scans[0]


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📈 Analyse CV — Voltammétrie cyclique")
    st.caption("Extraction des pics redox · Normalisation probe · Calibration par électrode")

    # Vérification que les données sont disponibles
    if not st.session_state.get("preprocessing_done", False):
        st.warning(
            "⚠️ Aucune donnée disponible. "
            "Importez et prétraitez vos données d'abord."
        )
        st.page_link("pages/0_import.py", label="→ Aller à l'import", icon="📂")
        st.stop()
        return

    # Récupérer les données prétraitées
    experiment = st.session_state["experiment_clean"]

    if st.button("↺ Relancer l'analyse", key="cv_rerun_btn"):
        st.session_state.pop("cv_sessions", None)
        st.rerun()

    if "cv_sessions" not in st.session_state:
        n_elec = experiment.get("n_electrodes", 2)
        cv_sessions = {}
        for e in range(1, n_elec + 1):
            cv_assignments = _build_cv_assignments_electrode(experiment, e)
            if not cv_assignments:
                continue
            cv_session = run_cv_pipeline(cv_assignments)
            # Référence « électrode nue » — attachée APRÈS l'analyse, jamais lue
            # par run_cv_pipeline (affichage seul).
            cv_session.bare_reference = _load_bare_cv(experiment, e)
            cv_sessions[e] = cv_session
        st.session_state["cv_sessions"] = cv_sessions

    render_cv_tabs(st.session_state["cv_sessions"])


if __name__ == "__main__":
    main()
