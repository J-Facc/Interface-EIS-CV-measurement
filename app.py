"""EIS Analyzer v3 — point d'entrée Streamlit.

Configure la navigation multipage et initialise les clés de session_state
partagées (theme_mode, comparison_report, eis_session).
`st.set_page_config` est appelé une seule fois ici ; les pages ne doivent
pas le rappeler.
"""

import streamlit as st

st.set_page_config(
    page_title="EIS Analyzer",
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
        "theme_mode": "light",        # "light" | "dark"
        "eis_session": None,          # EISSession (page A)
        "eis_config": None,
        "eis_validation": None,
        "comparison_report": None,    # dict retourné par compute_full_report (page C)
        "comparison_session_data": None,  # session_data brut pour predict_from_session (page D)
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_shared_state()

# ---------------------------------------------------------------------------
# Toggle jour / nuit — sidebar, visible depuis toutes les pages
# ---------------------------------------------------------------------------

with st.sidebar:
    current_theme = st.session_state["theme_mode"]
    label = "🌙 Mode sombre" if current_theme == "light" else "☀️ Mode clair"
    if st.button(label, key="__theme_toggle__", use_container_width=True):
        st.session_state["theme_mode"] = "dark" if current_theme == "light" else "light"
        st.rerun()

# ---------------------------------------------------------------------------
# Navigation multipage
# ---------------------------------------------------------------------------

pg = st.navigation(
    {
        "Analyse": [
            st.Page("pages/A_eis.py", title="EIS seule",  icon="📡"),
            st.Page("pages/B_cv.py",  title="CV seule",   icon="📈"),
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
