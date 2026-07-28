"""EIS CV Analyzer v3 — point d'entrée Streamlit.

Configure la navigation multipage et initialise les clés de session_state
partagées (eis_session, experiment_clean).
`st.set_page_config` est appelé une seule fois ici ; les pages ne doivent
pas le rappeler.
"""

import streamlit as st

st.set_page_config(
    page_title="EIS CV Analyzer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialisation des clés de session partagées
# ---------------------------------------------------------------------------

def _init_shared_state() -> None:
    """Crée les clés partagées si elles n'existent pas encore."""
    defaults = {
        "eis_session": None,
        "eis_config": None,
        "eis_validation": None,
        "experiment": None,
        "experiment_clean": None,
        "import_validated": False,
        "preprocessing_done": False,
        "exclusions": {},
        "validation_results": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_shared_state()

# ---------------------------------------------------------------------------
# Préparation DRT (CmdStan + Series.stan) — automatique, une fois par process
# ---------------------------------------------------------------------------
# La DRT (fits/drt_fit.py, bayes_drt2) compile des modèles Stan : sans toolchain,
# aucun mode ('optimize' ni 'sample') ne fonctionne. On prépare donc CmdStan et on
# compile Series.stan DÈS LE LANCEMENT (et non au clic de l'utilisateur), une seule
# fois, mis en cache — aucune étape manuelle requise à l'installation.

@st.cache_resource(
    show_spinner=(
        "Préparation du moteur DRT (CmdStan + Series.stan)… "
        "premier lancement uniquement, peut durer quelques minutes."
    )
)
def _bootstrap_drt() -> tuple:
    """Prépare la DRT une fois par process (idempotent). Ne bloque jamais l'app."""
    try:
        from fits import drt_fit

        if not drt_fit.bayes_available():
            return False, f"Extra DRT non installé ({drt_fit.import_error()})."
        from setup_drt_bayesien import ensure_drt_ready

        return ensure_drt_ready()
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        return False, f"Préparation DRT impossible : {exc}"


st.session_state["drt_ready"], st.session_state["drt_ready_msg"] = _bootstrap_drt()

# ---------------------------------------------------------------------------
# Navigation multipage
# ---------------------------------------------------------------------------

pg = st.navigation(
    {
        "Données": [
            st.Page("pages/0_import.py",        title="Import",        icon="📂"),
            st.Page("pages/1_pretraitement.py", title="Prétraitement", icon="🔬"),
        ],
        "Analyse": [
            st.Page("pages/A_eis.py", title="EIS seule", icon="📡"),
            st.Page("pages/B_cv.py",  title="CV seule",  icon="📈"),
        ],
        "Export": [
            st.Page("pages/E_export.py", title="Export", icon="💾"),
        ],
        "Inférence": [
            st.Page("pages/D_inference.py", title="Prédiction", icon="🎯"),
        ],
    }
)

pg.run()
