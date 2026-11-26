"""Page 2 — Import & nettoyage : dépôt des fichiers, assignation manuelle, nettoyage.

Aucune physique ici : détection, contrôles et nettoyage viennent de ``core/assignment.py``.
Produit ``state.MEASUREMENTS`` (réplicats conservés individuellement), seule entrée des pages
suivantes.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.assignment import (
    LEVEL_ERROR,
    AssignmentRow,
    check_assignment,
    clean_eis,
    clean_files,
    detect_kind,
    guess_rows,
    has_errors,
    measurement_summary_rows,
    step_options,
)
from core.config import config_to_dict, load_config
from core.experiment import KIND_EIS, KINDS
from core.mpr_converter import is_mpr, mpr_to_csv_bytes
from plotting.eis_plots import nyquist_points_figure
from ui import nav
from ui import state as S

_COLS = {"filename": "Fichier", "detected_kind": "Détecté", "kind": "Type", "channel": "Canal",
         "electrode": "Électrode", "step": "Étape", "replicate": "Réplicat", "exclude": "Exclure"}


def _config() -> dict:
    return config_to_dict(load_config())


def _ingest(uploaded) -> list:
    """Ajoute les fichiers déposés absents de l'état. Retourne les messages d'erreur."""
    raw = st.session_state[S.RAW_FILES]
    errors, new = [], []
    for uf in uploaded or []:
        data = uf.getvalue()
        fid = f"{uf.name}#{hashlib.sha256(data).hexdigest()[:8]}"
        if fid in raw:
            continue
        name = uf.name
        if is_mpr(name):
            try:
                data = mpr_to_csv_bytes(data)
            except Exception as exc:          # lecteur .mpr tiers : message, jamais de crash
                errors.append(f"« {name} » : conversion .mpr impossible — {exc}")
                continue
        raw[fid] = {"name": name, "content": data, "kind": detect_kind(data)}
        new.append((fid, name, raw[fid]["kind"]))
    if new:
        rows = [AssignmentRow.from_dict(d) for d in st.session_state[S.ASSIGNMENT]]
        rows += guess_rows(new, st.session_state[S.EXPERIMENT])
        st.session_state[S.ASSIGNMENT] = [r.to_dict() for r in rows]
        S.invalidate_measurements(st.session_state)
    return errors


def _assignment_editor(exp) -> list:
    rows = [AssignmentRow.from_dict(d) for d in st.session_state[S.ASSIGNMENT]]
    df = pd.DataFrame([r.to_dict() for r in rows])
    view = df[list(_COLS)].rename(columns=_COLS)
    key = "assign_" + hashlib.sha256("|".join(df["file_id"]).encode()).hexdigest()[:10]
    edited = st.data_editor(
        view, key=key, hide_index=True, width="stretch", num_rows="fixed",
        disabled=["Fichier", "Détecté"],
        column_config={
            "Type": st.column_config.SelectboxColumn(options=list(KINDS), required=True),
            "Canal": st.column_config.SelectboxColumn(options=exp.channel_ids(), required=True),
            "Électrode": st.column_config.SelectboxColumn(options=[1, 2], required=True),
            "Étape": st.column_config.SelectboxColumn(options=step_options(exp), required=True),
            "Réplicat": st.column_config.NumberColumn(min_value=1, step=1, required=True),
            "Exclure": st.column_config.CheckboxColumn(help="Ignorer ce fichier au nettoyage"),
        },
    )
    back = edited.rename(columns={v: k for k, v in _COLS.items()})
    new_rows = []
    for r, (_, e) in zip(rows, back.iterrows()):
        d = r.to_dict()
        d.update({k: (None if pd.isna(e[k]) else e[k]) for k in _COLS if k not in ("filename", "detected_kind")})
        new_rows.append(AssignmentRow.from_dict(d))
    if [r.to_dict() for r in new_rows] != [r.to_dict() for r in rows]:
        st.session_state[S.ASSIGNMENT] = [r.to_dict() for r in new_rows]
        S.invalidate_measurements(st.session_state)
    return new_rows


def _point_exclusion_editor(rows: list, config: dict) -> None:
    eis_rows = [r for r in rows if r.kind == KIND_EIS and not r.exclude]
    if not eis_rows:
        return
    with st.expander("✂️ Exclure des points d'un spectre EIS (sélection au lasso sur le Nyquist)"):
        labels = {r.file_id: r.filename for r in eis_rows}
        fid = st.selectbox("Spectre", list(labels), format_func=labels.get, key="imp_pts_file")
        excluded = st.session_state[S.EXCLUDED_POINTS].get(fid, [])
        try:
            data, _ = clean_eis(st.session_state[S.RAW_FILES][fid]["content"], labels[fid], config)
        except ValueError as exc:
            st.error(str(exc))
            return
        fig = nyquist_points_figure(data["f"], data["Zre"], data["Zim"], excluded,
                                    theme_mode=st.session_state[S.THEME])
        event = st.plotly_chart(fig, key=f"imp_pts_chart_{fid}", on_select="rerun", width="stretch")
        sel = []
        if event and getattr(event, "selection", None) and event.selection.points:
            sel = sorted({p["point_index"] for p in event.selection.points if p.get("curve_number", 0) == 0})
        c1, c2, c3 = st.columns(3)
        if c1.button(f"🗑️ Exclure {len(sel)} point(s)", disabled=not sel, key="imp_pts_excl"):
            st.session_state[S.EXCLUDED_POINTS][fid] = sorted(set(excluded) | set(sel))
            S.invalidate_measurements(st.session_state)
            st.rerun()
        if c2.button("↩️ Restaurer tous les points", disabled=not excluded, key="imp_pts_restore"):
            st.session_state[S.EXCLUDED_POINTS].pop(fid, None)
            S.invalidate_measurements(st.session_state)
            st.rerun()
        c3.caption(f"{len(excluded)} point(s) exclu(s) sur {len(data['f'])}.")


def _render_issues(issues: list) -> None:
    for i in issues:
        (st.error if i.level == LEVEL_ERROR else st.warning)(i.message)


def render() -> None:
    st.title("📂 Import & nettoyage")
    exp = st.session_state.get(S.EXPERIMENT)
    if exp is None:
        st.warning("Validez d'abord la configuration de l'expérience.")
        nav.link("config", "→ Configuration")
        st.stop()
    config = _config()

    st.subheader("1. Dépôt des fichiers")
    uploaded = st.file_uploader("Fichiers EC-Lab (EIS et CV) : .txt, .csv, .mpt, .mpr",
                                type=["txt", "csv", "mpt", "mpr"], accept_multiple_files=True,
                                key="imp_uploader")
    for msg in _ingest(uploaded):
        st.error(msg)
    raw = st.session_state[S.RAW_FILES]
    if not raw:
        st.info("Déposez les fichiers de mesure. Chaque fichier sera assigné à un canal, une électrode, "
                "une étape et un réplicat.")
        return
    c1, c2 = st.columns([3, 1])
    c1.caption(f"{len(raw)} fichier(s) déposé(s) — "
               f"{sum(1 for v in raw.values() if v['kind'] == 'EIS')} EIS, "
               f"{sum(1 for v in raw.values() if v['kind'] == 'CV')} CV.")
    if c2.button("🗑️ Retirer tous les fichiers"):
        st.session_state[S.RAW_FILES] = {}
        st.session_state[S.ASSIGNMENT] = []
        st.session_state[S.EXCLUDED_POINTS] = {}
        S.invalidate_measurements(st.session_state)
        st.rerun()

    st.subheader("2. Assignation")
    st.caption("Pré-remplissage indicatif (contenu + nom du fichier) : vérifiez et corrigez chaque ligne. "
               "Étapes : bare (électrode nue), probe (sonde), conc_i (i-ème concentration du canal).")
    rows = _assignment_editor(exp)
    issues = check_assignment(rows, exp)
    _render_issues(issues)
    _point_exclusion_editor(rows, config)

    st.subheader("3. Nettoyage")
    st.caption("EIS : séparateur auto, signe de Zim (EC-Lab), suppression des fréquences parasites "
               f"{config['fit']['n_freqs_parasites']} Hz (± {config['fit']['tol_parasites']} Hz), tri HF → BF. "
               f"CV : colonnes auto, courant en A, cycle conservé : {config['cv']['cycle']}, ordre chronologique. "
               "Les réplicats sont conservés individuellement.")
    if st.button("🧹 Nettoyer les données", type="primary", disabled=has_errors(issues)):
        contents = {fid: v["content"] for fid, v in raw.items()}
        ms, report = clean_files(rows, contents, exp, config, st.session_state[S.EXCLUDED_POINTS])
        S.invalidate_measurements(st.session_state)
        st.session_state[S.MEASUREMENTS] = ms or None
        st.session_state[S.CLEANING_REPORT] = [i for i in report if i not in issues]
        st.rerun()

    ms = st.session_state.get(S.MEASUREMENTS)
    _render_issues(st.session_state.get(S.CLEANING_REPORT) or [])
    if ms:
        st.success(f"✅ {len(ms)} mesure(s) nettoyée(s).")
        st.dataframe(pd.DataFrame(measurement_summary_rows(ms)), hide_index=True, width="stretch")
        nav.link("viz", "→ Visualisation")
