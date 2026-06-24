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
