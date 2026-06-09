"""EIS CV Analyzer v3 — point d'entrée Streamlit.

Configure la navigation multipage et initialise les clés de session_state
partagées (comparison_report, eis_session).
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
        "comparison_report": None,
        "comparison_session_data": None,
        "experiment": None,
        "experiment_clean": None,
        "import_validated": False,
        "exclusions": {},
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
        "Comparaison": [
            st.Page("pages/C_comparatif.py", title="Comparatif", icon="⚖️"),
        ],
        "Inférence": [
            st.Page("pages/D_inference.py", title="Prédiction", icon="🎯"),
        ],
    }
)

pg.run()
