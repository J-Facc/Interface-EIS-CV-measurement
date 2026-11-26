"""Chargement d'une archive (.zip) depuis la page Configuration.

* archive v3 (``manifest.json``) : expérience, mesures nettoyées et TOUS les résultats
  reconstruits sans recalcul (``exports.archive.read_archive``) ;
* archive v2 (``experiment.yaml``) : configuration + fichiers bruts assignés
  (``core.legacy_import``), à nettoyer puis analyser.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from core.legacy_import import import_v2
from exports.archive import ArchiveError, is_v3_archive, read_archive
from ui import nav
from core import app_state as S


def render_archive_loader() -> None:
    up = st.file_uploader("Archive .zip (v3, ou v2 de l'ancienne application)", type=["zip"], key="arch_upload")
    if up is None:
        return
    data = up.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("_arch_digest") == digest:
        st.caption(f"Archive « {up.name} » chargée.")
        return
    if is_v3_archive(data):
        try:
            exp, ms, res = read_archive(data)
        except ArchiveError as exc:
            st.error(f"❌ Archive invalide : {exc}")
            return
        S.load_archive_state(st.session_state, exp, ms, res)
        st.session_state["_arch_digest"] = digest
        st.success(f"✅ Archive chargée : {len(ms)} mesure(s) nettoyée(s)"
                   + (", résultats d'analyse restaurés sans recalcul." if res is not None else ", sans résultats."))
        nav.link("viz", "→ Visualisation")
        if res is not None:
            nav.link("analysis", "→ Analyse (résultats)")
        return
    try:
        out = import_v2(data)
    except Exception as exc:                 # lecteur v2 historique : jamais de crash de la page
        st.error(f"❌ Archive invalide : ni v3 (manifest.json) ni v2 lisible ({exc}).")
        return
    S.load_archive_state(st.session_state, out["experiment"], None, None)
    st.session_state[S.RAW_FILES] = out["raw_files"]
    st.session_state[S.ASSIGNMENT] = [r.to_dict() for r in out["rows"]]
    st.session_state[S.EXCLUDED_POINTS] = out["excluded_points"]
    st.session_state["_arch_digest"] = digest
    for w in out["warnings"]:
        st.warning(w)
    nav.link("import", "→ Import & nettoyage")
