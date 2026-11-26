"""EIS Analyzer v3 — point d'entrée Streamlit.

Configure la navigation multipage (Configuration → Import & nettoyage → Visualisation →
Analyse → Export), le toggle jour/nuit des figures et les clés de session_state partagées
(``ui/state.py``). ``st.set_page_config`` est appelé une seule fois ici ; les pages ne
doivent pas le rappeler.
"""

import streamlit as st

from core.app_state import init_shared_state
from core.version import APP_VERSION
from ui import nav
from ui import page_analysis, page_config, page_export, page_import, page_viz
from ui import state as S

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
S.init_state(st.session_state)

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


@st.cache_resource(show_spinner=False)
def _drt_version_warning(_ready: bool):
    """Avertissement si le CmdStan actif n'est pas la version sur laquelle les réglages de la
    DRT ont été validés (``None`` sinon). Lu APRÈS ``_drt_status()`` : c'est lui qui enregistre
    le chemin de CmdStan dans ce process. L'onglet DRT l'affiche (``ui/tabs.py``)."""
    try:
        from drt import engine as drt_engine

        return drt_engine.cmdstan_version_info()[1]
    except Exception:  # pragma: no cover - dépend de l'environnement
        return None


st.session_state["drt_version_warning"] = _drt_version_warning(st.session_state["drt_ready"])

# ---------------------------------------------------------------------------
# Sidebar : toggle jour/nuit (thème des figures)
# ---------------------------------------------------------------------------

with st.sidebar:
    night = st.toggle("🌙 Mode nuit (figures)", value=st.session_state[S.THEME] == "dark",
                      key="__theme_toggle")
    st.session_state[S.THEME] = "dark" if night else "light"
    st.caption(f"EIS Analyzer v{APP_VERSION}")

# ---------------------------------------------------------------------------
# Navigation multipage
# ---------------------------------------------------------------------------

nav.PAGES.clear()
nav.PAGES.update({
    "config": st.Page(page_config.render, title="Configuration", icon="⚙️", url_path="configuration",
                      default=True),
    "import": st.Page(page_import.render, title="Import & nettoyage", icon="📂", url_path="import-nettoyage"),
    "viz": st.Page(page_viz.render, title="Visualisation", icon="👁️", url_path="visualisation"),
    "analysis": st.Page(page_analysis.render, title="Analyse", icon="🧮", url_path="analyse"),
    "export": st.Page(page_export.render, title="Export", icon="💾", url_path="export-v3"),
    "old_import": st.Page("pages/0_import.py", title="Import (v2)", icon="📂"),
    "pretraitement": st.Page("pages/1_pretraitement.py", title="Prétraitement (v2)", icon="🔬"),
    "eis": st.Page("pages/A_eis.py", title="EIS seule", icon="📡"),
    "cv": st.Page("pages/B_cv.py", title="CV seule", icon="📈"),
    "old_export": st.Page("pages/E_export.py", title="Export (v2)", icon="💾"),
})
pg = st.navigation(list(nav.PAGES.values()))
pg.run()
