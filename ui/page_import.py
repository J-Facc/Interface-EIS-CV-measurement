"""Page 2 — Import & nettoyage : un DOSSIER par canal, assignation (vérifiable), nettoyage.

Chaque canal reçoit le dossier de ses mesures (``probe/``, ``calibration/``, ``bare/`` ;
noms ``probe_eis_e1_r1.txt``, ``eis_e1_1.00e-08M_r1.txt``… ; ``.mpr`` convertis) : l'assignation
est lue dans les noms (``core/folder_import.py``) et reste corrigeable dans le tableau. Des
fichiers isolés peuvent encore être déposés (pré-remplissage indicatif).

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
from core.experiment import KIND_EIS, KINDS, Channel, Experiment, format_concentration
from core.folder_import import EXTENSIONS, import_channel_folder, reassign_steps, same_concentrations
from core.mpr_converter import is_mpr, mpr_to_csv_bytes
from plotting.eis_plots import nyquist_points_figure
from ui import nav
from core import app_state as S

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


FOLDER_ISSUES = "folder_issues"        # {canal: [Issue]} du dernier import de dossier
FOLDER_CONCS = "folder_concentrations"  # {canal: [mol/L]} lues dans les noms
FOLDER_IGNORED = "folder_ignored"       # {canal: [(chemin, raison)]} fichiers non importés


def _ingest_folder(channel_id: int, uploaded) -> None:
    """(Ré)importe le dossier d'un canal : remplace ses fichiers et ses lignes, garde le reste."""
    files = [(uf.name, uf.getvalue()) for uf in uploaded or []]
    digest = hashlib.sha256(repr(sorted((n, hashlib.sha256(b).hexdigest()) for n, b in files)).encode()).hexdigest()
    key = f"_imp_dir_digest_{channel_id}"
    if not files or st.session_state.get(key) == digest:
        return
    res = import_channel_folder(files, st.session_state[S.EXPERIMENT], channel_id)
    prefix = f"C{channel_id}/"
    raw = {k: v for k, v in st.session_state[S.RAW_FILES].items() if not k.startswith(prefix)}
    raw.update(res.raw_files)
    rows = [d for d in st.session_state[S.ASSIGNMENT] if not str(d["file_id"]).startswith(prefix)]
    rows += [r.to_dict() for r in res.rows]
    st.session_state[S.RAW_FILES] = raw
    st.session_state[S.ASSIGNMENT] = rows
    st.session_state.setdefault(FOLDER_ISSUES, {})[channel_id] = res.issues
    st.session_state.setdefault(FOLDER_CONCS, {})[channel_id] = res.concentrations_found
    st.session_state.setdefault(FOLDER_IGNORED, {})[channel_id] = res.ignored
    st.session_state[key] = digest
    S.invalidate_measurements(st.session_state)


def _adopt_concentrations(channel_id: int, concs: list) -> None:
    """Remplace les concentrations du canal par celles lues dans les noms, puis réaffecte conc_i."""
    exp = st.session_state[S.EXPERIMENT]
    chans = [Channel(c.id, c.biomarker, list(concs) if c.id == channel_id else list(c.concentrations))
             for c in exp.channels]
    new = Experiment(biomarkers=list(exp.biomarkers), channels=chans, name=exp.name, created_at=exp.created_at)
    S.new_experiment(st.session_state, new)
    rows = reassign_steps([AssignmentRow.from_dict(d) for d in st.session_state[S.ASSIGNMENT]], new)
    st.session_state[S.ASSIGNMENT] = [r.to_dict() for r in rows]
    issues = st.session_state.get(FOLDER_ISSUES, {}).get(channel_id, [])
    st.session_state[FOLDER_ISSUES][channel_id] = [i for i in issues if "absente de la configuration" not in i.message]


def _channel_folders(exp) -> None:
    st.caption("Un dossier par canal, tel qu'il sort du poste de mesure : "
               "`probe/{eis,cv}/electrode_N/probe_eis_eN_rK.txt`, "
               "`calibration/{eis,cv}/electrode_N/eis_eN_1.00e-08M_rK.txt` (et `bare/…` facultatif). "
               "Les fichiers .txt, .csv, .mpt et .mpr (converti automatiquement) sont lus ; seuls les noms font foi. "
               "Tout autre fichier du dossier est ignoré.")
    for ch in exp.channels:
        with st.container(border=True):
            st.markdown(f"**Canal {ch.id} — {ch.biomarker}** · concentrations configurées : "
                        + ", ".join(format_concentration(c) for c in ch.concentrations))
            gen = st.session_state.get(f"_imp_dir_gen_{ch.id}", 0)      # nouvelle clé = widget vidé
            # Pas de filtre ``type`` : Streamlit marquerait en erreur chaque fichier annexe du dossier
            # (.mps, .mpl…). Tout est reçu, le tri se fait sur les noms (core/folder_import.py).
            up = st.file_uploader(f"Dossier du canal {ch.id}", accept_multiple_files="directory",
                                  key=f"imp_dir_{ch.id}_{gen}")
            _ingest_folder(ch.id, up)
            mine = [d for d in st.session_state[S.ASSIGNMENT] if str(d["file_id"]).startswith(f"C{ch.id}/")]
            ignored = st.session_state.get(FOLDER_IGNORED, {}).get(ch.id) or []
            if ignored:
                with st.expander(f"{len(ignored)} fichier(s) ignoré(s) (noms hors conventions, doublons)"):
                    st.dataframe(pd.DataFrame(ignored, columns=["Fichier", "Raison"]), hide_index=True,
                                 width="stretch")
            if not mine:
                if ignored:
                    st.warning("Aucun fichier de mesure reconnu dans ce dossier.")
                continue
            kept = [d for d in mine if not d["exclude"]]
            st.caption(f"{len(mine)} fichier(s) : {sum(d['kind'] == 'EIS' for d in kept)} EIS, "
                       f"{sum(d['kind'] == 'CV' for d in kept)} CV retenus ; électrode(s) "
                       + ", ".join(str(e) for e in sorted({d['electrode'] for d in kept if d['electrode']})) + ".")
            found = st.session_state.get(FOLDER_CONCS, {}).get(ch.id) or []
            configured = list(ch.concentrations)
            if found and not same_concentrations(found, configured):
                st.warning("Concentrations lues dans les noms : " + ", ".join(format_concentration(c) for c in found)
                           + " — différentes de la configuration du canal.")
                if st.button(f"Adopter ces concentrations pour le canal {ch.id}", key=f"imp_adopt_{ch.id}"):
                    _adopt_concentrations(ch.id, found)
                    st.rerun()
            for i in st.session_state.get(FOLDER_ISSUES, {}).get(ch.id, []):
                (st.error if i.level == LEVEL_ERROR else st.warning)(i.message)
            if st.button(f"🗑️ Retirer le dossier du canal {ch.id}", key=f"imp_rm_{ch.id}"):
                prefix = f"C{ch.id}/"
                st.session_state[S.RAW_FILES] = {k: v for k, v in st.session_state[S.RAW_FILES].items()
                                                 if not k.startswith(prefix)}
                st.session_state[S.ASSIGNMENT] = [d for d in st.session_state[S.ASSIGNMENT]
                                                  if not str(d["file_id"]).startswith(prefix)]
                st.session_state.get(FOLDER_ISSUES, {}).pop(ch.id, None)
                st.session_state.get(FOLDER_IGNORED, {}).pop(ch.id, None)
                st.session_state.pop(f"_imp_dir_digest_{ch.id}", None)
                st.session_state[f"_imp_dir_gen_{ch.id}"] = gen + 1
                S.invalidate_measurements(st.session_state)
                st.rerun()


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

    st.subheader("1. Dépôt — un dossier par canal")
    _channel_folders(exp)
    with st.expander("Fichiers isolés (hors arborescence, assignation à compléter à la main)"):
        uploaded = st.file_uploader("Fichiers EC-Lab (EIS et CV) : .txt, .csv, .mpt, .mpr",
                                    type=list(EXTENSIONS), accept_multiple_files=True, key="imp_uploader")
        for msg in _ingest(uploaded):
            st.error(msg)
    raw = st.session_state[S.RAW_FILES]
    if not raw:
        st.info("Déposez le dossier de chaque canal : l'assignation (électrode, étape, réplicat) est lue dans "
                "les noms de fichiers.")
        return
    c1, c2 = st.columns([3, 1])
    c1.caption(f"{len(raw)} fichier(s) déposé(s) — "
               f"{sum(1 for v in raw.values() if v['kind'] == 'EIS')} EIS, "
               f"{sum(1 for v in raw.values() if v['kind'] == 'CV')} CV.")
    if c2.button("🗑️ Retirer tous les fichiers"):
        for k in [k for k in st.session_state if str(k).startswith("_imp_dir_digest_")]:
            del st.session_state[k]
        for ch in exp.channels:
            st.session_state[f"_imp_dir_gen_{ch.id}"] = st.session_state.get(f"_imp_dir_gen_{ch.id}", 0) + 1
        st.session_state[FOLDER_ISSUES] = {}
        st.session_state[FOLDER_IGNORED] = {}
        st.session_state[S.RAW_FILES] = {}
        st.session_state[S.ASSIGNMENT] = []
        st.session_state[S.EXCLUDED_POINTS] = {}
        S.invalidate_measurements(st.session_state)
        st.rerun()

    st.subheader("2. Assignation")
    st.caption("Lue dans les noms pour les dossiers de canal, indicative pour les fichiers isolés : vérifiez et "
               "corrigez au besoin. "
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
