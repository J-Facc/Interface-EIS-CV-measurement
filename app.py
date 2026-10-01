"""EIS CV Analyzer v3 — point d'entrée Streamlit.

Configure la navigation multipage et initialise les clés de session_state
partagées (core/app_state.py : experiment_clean, eis_sessions, cv_sessions…).
`st.set_page_config` est appelé une seule fois ici ; les pages ne doivent
pas le rappeler.
"""

import streamlit as st

from core.app_state import init_shared_state

st.set_page_config(
    page_title="EIS CV Analyzer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialisation des clés de session partagées
# ---------------------------------------------------------------------------

init_shared_state(st.session_state)   # crée les clés absentes, n'écrase rien

# ---------------------------------------------------------------------------
# État de la DRT (CmdStan + modèles Stan) — constaté, JAMAIS installé ici
# ---------------------------------------------------------------------------
# La DRT (drt/engine.py, drt/bayes_drt2) compile des modèles Stan : sans toolchain,
# aucun mode ('optimize' ni 'sample') ne fonctionne. L'INSTALLATION est faite par
# launch.bat (python setup_drt_bayesien.py --ensure), avant le démarrage du serveur :
# l'app se contente de ré-enregistrer le CmdStan déjà installé et de constater l'état,
# instantanément. Une installation lancée d'ici bloquerait l'interface sans délai maximal
# et, hors ligne, serait retentée à chaque démarrage du process.

@st.cache_resource(show_spinner=False)
def _drt_status() -> tuple:
    """(prête, message) — une fois par process, sans effet de bord ni réseau."""
    try:
        from drt import engine as drt_engine

        ok, why = drt_engine.library_available()
        if not ok:
            return False, f"Extra DRT non installé ({why})."
        from setup_drt_bayesien import ensure_drt_ready

        return ensure_drt_ready(allow_install=False)
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        return False, f"État de la DRT indéterminable : {exc}"


st.session_state["drt_ready"], st.session_state["drt_ready_msg"] = _drt_status()

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
    }
)

pg.run()
