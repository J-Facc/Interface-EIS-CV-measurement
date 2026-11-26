"""Registre des pages (rempli par app.py) et liens entre pages."""

from __future__ import annotations

import streamlit as st

#: {nom court: st.Page} — posé par app.py à chaque exécution.
PAGES: dict = {}


def link(name: str, label: str, icon: str = None) -> None:
    """Lien vers une page enregistrée ; ne fait rien si la page n'existe pas."""
    page = PAGES.get(name)
    if page is not None:
        st.page_link(page, label=label, icon=icon)
