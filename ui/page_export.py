"""Page 5 — Export : archive .zip rechargeable, rapport PDF, rapport HTML interactif.

Disponible une fois TOUS les calculs terminés. Chaque fichier est généré à la demande
(``exports.archive`` / ``exports.report_pdf`` / ``exports.report_html``), puis proposé au
téléchargement. Aucun calcul : assemblage des résultats existants.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import streamlit as st

from core.config import config_to_dict, load_config
from exports.archive import archive_filename, write_archive
from exports.report_html import build_html
from exports.report_pdf import build_pdf
from ui import nav
from ui import state as S

_CACHE = "_export_cache"


def _fingerprint() -> str:
    """Change dès que l'expérience, les mesures ou les résultats changent (cache invalide)."""
    parts = [id(st.session_state.get(S.EXPERIMENT)), id(st.session_state.get(S.MEASUREMENTS)),
             id(st.session_state.get(S.RESULTS)), st.session_state.get(S.THEME)]
    res = st.session_state.get(S.RESULTS)
    if res is not None:     # un recalcul DRT modifie les résultats en place
        parts.append(sum(id(s.drt) for s in res.eis.spectra.values()))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _block(kind: str, label: str, mime: str, filename: str, build, help_text: str) -> None:
    cache = st.session_state.setdefault(_CACHE, {})
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.caption(help_text)
        key = (kind, _fingerprint())
        if st.button(f"⚙️ Générer — {label}", key=f"gen_{kind}"):
            with st.spinner("Génération…"):
                try:
                    cache[kind] = (key, build())
                except Exception as exc:            # rapport : jamais de crash de la page
                    st.error(f"Génération impossible : {exc}")
        entry = cache.get(kind)
        if entry is not None and entry[0] == key:
            st.download_button(f"📥 Télécharger ({len(entry[1]) / 1e6:.1f} Mo)", data=entry[1], file_name=filename,
                               mime=mime, key=f"dl_{kind}")


def render() -> None:
    st.title("💾 Export")
    exp = st.session_state.get(S.EXPERIMENT)
    ms = st.session_state.get(S.MEASUREMENTS)
    res = st.session_state.get(S.RESULTS)
    if st.session_state.get(S.ANALYSIS_RUN) is not None:
        st.info("Une analyse est en cours : l'export sera disponible quand tous les calculs seront terminés.")
        nav.link("analysis", "→ Analyse")
        st.stop()
    if exp is None or not ms:
        st.warning("Rien à exporter : importez et nettoyez des données (ou chargez une archive).")
        nav.link("import", "→ Import & nettoyage")
        st.stop()
    if res is None:
        st.warning("L'analyse n'a pas été lancée : l'archive et les rapports ne contiendront que les données "
                   "nettoyées.")
    stem = archive_filename(exp)[:-4]
    config = res.config if res is not None else config_to_dict(load_config())
    _block("zip", "Archive .zip rechargeable", "application/zip", archive_filename(exp),
           lambda: write_archive(exp, ms, res, config),
           "Données nettoyées, résultats (DRT, fits, CV, étalonnages), configuration et sommes de contrôle. "
           "Rechargeable depuis la page Configuration, sans recalcul ; lisible hors de l'application (JSON, CSV).")
    _block("pdf", "Rapport PDF", "application/pdf", f"{stem}_rapport.pdf", lambda: build_pdf(exp, ms, res),
           "Page de garde, CV, Nyquist/Bode, DRT, tableau des Rct, étalonnages, échecs (A4, matplotlib).")
    _block("html", "Rapport HTML interactif", "text/html", f"{stem}_rapport.html",
           lambda: build_html(exp, ms, res, st.session_state[S.THEME]).encode("utf-8"),
           "Fichier autonome (plotly.js embarqué), mêmes figures que l'application, avec sommaire.")
    st.caption(f"Session du {datetime.now():%d/%m/%Y}.")
