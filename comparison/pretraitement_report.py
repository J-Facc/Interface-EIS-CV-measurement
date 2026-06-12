"""Génération du rapport HTML du prétraitement.

Le HTML exporté est un miroir de pages/1_pretraitement.py :
  - Pour chaque électrode :
      * Un bloc par concentration avec Nyquist superposé + CV superposé
        (réplicats exclus grisés, points EIS supprimés retirés)
      * Tableau exclusions (statut par réplicat + nb points supprimés)
      * Panneau graphes moyens (Nyquist moyen + CV moyen)
      * Validation Kramers-Kronig (tableau résumé + résidus)
  - Résumé final : métriques globales

Appelle les mêmes fonctions de plotting que la page Streamlit.
Aucune dépendance à st.session_state — tout vient des paramètres.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


# ── Palettes (identiques à 1_pretraitement.py) ────────────────────────────────

REP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
EXCL_COLOR  = "lightgray"
_SUP_MAP    = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(conc: float) -> str:
    if conc <= 0:
        return str(conc)
    exp  = int(np.floor(np.log10(conc)))
    mant = conc / 10 ** exp
    return f"{mant:.0f}×10{str(exp).translate(_SUP_MAP)} M"


def _dp_key(e_str: str, modality: str, ci, ri: int) -> str:
    ci_str = "probe" if ci == "probe" else str(ci)
    return f"deleted_points_{e_str}_{modality}_c{ci_str}_r{ri}"


def _get_excl(exclusions: dict, e_str: str, modality: str, ci) -> list:
    return exclusions.get(e_str, {}).get(modality, {}).get(ci, [])


def _get_deleted(deleted_points: dict, e_str: str, modality: str, ci, ri: int) -> list:
    return list(deleted_points.get(_dp_key(e_str, modality, ci, ri), []))


# ── Chargement des données ────────────────────────────────────────────────────

def _load_eis(experiment: dict) -> dict:
    """Charge les spectres EIS depuis experiment, retourne {"probe": {key: [sp,...]}, "calibration": {key: [[sp,..],...]}}."""
    from core.loader import load_spectrum

    calibration = (experiment.get("calibration") or {}).get("eis") or {}
    probe_dict  = (experiment.get("probe") or {}).get("eis") or {}
    n_elec      = experiment.get("n_electrodes", 2)
    result: dict = {"probe": {}, "calibration": {}}

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        result["probe"][key] = []
        for ri, bio in enumerate(probe_dict.get(key) or []):
            if bio is None:
                continue
            try:
                bio.seek(0)
                sp = load_spectrum(bio.read(), label=f"probe_e{e}_r{ri+1}")
                bio.seek(0)
                result["probe"][key].append(sp)
            except Exception:
                result["probe"][key].append(None)

        result["calibration"][key] = []
        for ci, rep_files in enumerate(calibration.get(key) or []):
            reps = []
            for ri, bio in enumerate(rep_files or []):
                if bio is None:
                    reps.append(None)
                    continue
                try:
                    bio.seek(0)
                    sp = load_spectrum(bio.read(), label=f"e{e}_c{ci+1}_r{ri+1}")
                    bio.seek(0)
                    reps.append(sp)
                except Exception:
                    reps.append(None)
            result["calibration"][key].append(reps)

    return result


def _load_cv(experiment: dict) -> dict:
    """Charge les scans CV depuis experiment."""
    from core.cv_loader import load_cv_file

    calibration = (experiment.get("calibration") or {}).get("cv") or {}
    probe_dict  = (experiment.get("probe") or {}).get("cv") or {}
    concentrations = experiment.get("concentrations") or []
    n_elec      = experiment.get("n_electrodes", 2)
    result: dict = {"probe": {}, "calibration": {}}

    for e in range(1, n_elec + 1):
        key = f"electrode_{e}"
        result["probe"][key] = []
        for ri, bio in enumerate(probe_dict.get(key) or []):
            if bio is None:
                result["probe"][key].append(None)
                continue
            try:
                bio.seek(0)
                sc = load_cv_file(bio.read(), label=f"probe_e{e}_r{ri+1}",
                                  concentration=0.0, step="probe")
                bio.seek(0)
                result["probe"][key].append(sc)
            except Exception:
                result["probe"][key].append(None)

        result["calibration"][key] = []
        for ci, rep_files in enumerate(calibration.get(key) or []):
            conc = concentrations[ci] if ci < len(concentrations) else 0.0
            reps = []
            for ri, bio in enumerate(rep_files or []):
                if bio is None:
                    reps.append(None)
                    continue
                try:
                    bio.seek(0)
                    sc = load_cv_file(bio.read(), label=f"e{e}_c{ci+1}_r{ri+1}",
                                      concentration=conc, step="hybridization")
                    bio.seek(0)
                    reps.append(sc)
                except Exception:
                    reps.append(None)
            result["calibration"][key].append(reps)

    return result


# ── Figures standalone (mêmes que dans 1_pretraitement.py, sans st.session_state) ──

def _fig_superposed_nyquist(
    reps: list,
    e_str: str,
    ci,
    exclusions: dict,
    deleted_points: dict,
) -> go.Figure:
    """Miroir de _superposed_nyquist() de 1_pretraitement.py."""
    fig = go.Figure()
    excl_list = _get_excl(exclusions, e_str, "eis", ci)

    for ri, sp in enumerate(reps):
        if sp is None:
            continue
        excluded = excl_list[ri] if ri < len(excl_list) else False
        deleted  = _get_deleted(deleted_points, e_str, "eis", ci, ri)
        color    = EXCL_COLOR if excluded else REP_COLORS[ri % len(REP_COLORS)]
        opacity  = 0.3 if excluded else 1.0
        name     = f"Rép {ri+1}" + (" [Exclu]" if excluded else "")

        keep = np.ones(len(sp.f), dtype=bool)
        for idx in deleted:
            if 0 <= idx < len(sp.f):
                keep[idx] = False

        fig.add_trace(go.Scatter(
            x=np.asarray(sp.Zre)[keep],
            y=np.asarray(sp.Zim)[keep],
            mode="markers",
            name=name,
            marker=dict(color=color, size=6),
            opacity=opacity,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Re(Z) = %{x:.1f} Ω<br>−Im(Z) = %{y:.1f} Ω<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        height=300,
        margin=dict(t=30, b=40, l=50, r=20),
    )
    return fig


def _fig_superposed_cv(
    reps: list,
    e_str: str,
    ci,
    exclusions: dict,
) -> go.Figure:
    """Miroir de _superposed_cv() de 1_pretraitement.py."""
    fig = go.Figure()
    excl_list = _get_excl(exclusions, e_str, "cv", ci)

    for ri, sc in enumerate(reps):
        if sc is None:
            continue
        excluded = excl_list[ri] if ri < len(excl_list) else False
        color    = EXCL_COLOR if excluded else REP_COLORS[ri % len(REP_COLORS)]
        opacity  = 0.3 if excluded else 1.0
        name     = f"Rép {ri+1}" + (" [Exclu]" if excluded else "")

        fig.add_trace(go.Scatter(
            x=np.asarray(sc.E),
            y=np.asarray(sc.I) * 1e6,
            mode="lines",
            name=name,
            line=dict(color=color, dash="dot" if excluded else "solid"),
            opacity=opacity,
        ))

    fig.update_layout(
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        height=240,
        margin=dict(t=30, b=40, l=50, r=20),
    )
    return fig


def _fig_average_nyquist(groups: list) -> go.Figure:
    """Miroir de _average_nyquist() de 1_pretraitement.py.
    groups : list of (label, [active_spectra])
    """
    from core.loader import average_replicates

    fig = go.Figure()
    for ci_idx, (label, active) in enumerate(groups):
        active_nn = [sp for sp in active if sp is not None]
        if not active_nn:
            continue
        try:
            avg   = average_replicates(active_nn) if len(active_nn) > 1 else active_nn[0]
            color = REP_COLORS[ci_idx % len(REP_COLORS)]
            fig.add_trace(go.Scatter(
                x=avg.Zre, y=avg.Zim,
                mode="markers+lines",
                name=label,
                marker=dict(color=color, size=5),
                line=dict(color=color, width=1),
            ))
        except Exception:
            pass

    fig.update_layout(
        title="Nyquist moyen",
        xaxis=dict(title="Re(Z) (Ω)", rangemode="tozero"),
        yaxis=dict(title="−Im(Z) (Ω)", rangemode="tozero"),
        hovermode="closest",
        height=290,
        margin=dict(t=40, b=40, l=50, r=10),
        legend=dict(font=dict(size=10)),
    )
    return fig


def _fig_average_cv(groups: list) -> go.Figure:
    """Miroir de _average_cv() de 1_pretraitement.py.
    groups : list of (label, [active_scans])
    """
    from core.cv_loader import average_cv_replicates

    fig = go.Figure()
    for ci_idx, (label, active) in enumerate(groups):
        active_nn = [sc for sc in active if sc is not None]
        if not active_nn:
            continue
        try:
            avg   = average_cv_replicates(active_nn) if len(active_nn) > 1 else active_nn[0]
            color = REP_COLORS[ci_idx % len(REP_COLORS)]
            fig.add_trace(go.Scatter(
                x=avg.E,
                y=np.asarray(avg.I) * 1e6,
                mode="lines",
                name=label,
                line=dict(color=color),
            ))
        except Exception:
            pass

    fig.update_layout(
        title="CV moyen",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
        hovermode="closest",
        height=240,
        margin=dict(t=40, b=40, l=50, r=10),
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Helpers HTML ──────────────────────────────────────────────────────────────

_FIG_ID = 0


def _next_id() -> str:
    global _FIG_ID
    _FIG_ID += 1
    return f"fig_{_FIG_ID}"


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=_next_id())


def _exclusion_table(
    reps_eis: list,
    reps_cv: list,
    e_str: str,
    ci,
    exclusions: dict,
    deleted_points: dict,
    mode: str,
) -> str:
    """Tableau HTML : réplicat | statut EIS | points EIS supprimés | statut CV."""
    n_reps = max(
        len(reps_eis) if reps_eis else 0,
        len(reps_cv)  if reps_cv  else 0,
    )
    if n_reps == 0:
        return ""

    excl_eis = _get_excl(exclusions, e_str, "eis", ci)
    excl_cv  = _get_excl(exclusions, e_str, "cv",  ci)

    cols_header = ["<th>Réplicat</th>"]
    if mode in ("eis_only", "both"):
        cols_header += ["<th>EIS</th>", "<th>Points EIS exclus</th>"]
    if mode in ("cv_only", "both"):
        cols_header.append("<th>CV</th>")

    rows = []
    for ri in range(n_reps):
        cells = [f"<td>Rép {ri+1}</td>"]

        if mode in ("eis_only", "both"):
            excl = excl_eis[ri] if ri < len(excl_eis) else False
            cls  = "excl" if excl else "ok"
            txt  = "⚠️ Exclu" if excl else "✓ Inclus"
            cells.append(f'<td class="{cls}">{txt}</td>')
            n_del = len(_get_deleted(deleted_points, e_str, "eis", ci, ri))
            cells.append(f"<td>{'—' if excl else n_del}</td>")

        if mode in ("cv_only", "both"):
            excl = excl_cv[ri] if ri < len(excl_cv) else False
            cls  = "excl" if excl else "ok"
            txt  = "⚠️ Exclu" if excl else "✓ Inclus"
            cells.append(f'<td class="{cls}">{txt}</td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table>"
        "<thead><tr>" + "".join(cols_header) + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


def _kk_section_html(validation_results: dict, elec_idx: int) -> str:
    """Section KK pour une électrode : tableau résumé + figure résidus."""
    from plotting.kk_plots import residuals_figure, validation_summary_table

    # Filtrer les résultats pour cette électrode
    elec_vrs = {
        k: v for k, v in validation_results.items()
        if f"electrode_{elec_idx}" in k or k.startswith(f"eis_electrode_{elec_idx}")
        or k.startswith(f"probe_electrode_{elec_idx}")
    }
    if not elec_vrs:
        return ""

    parts = ["<div class='kk-section'><h3>Validation Kramers-Kronig</h3>"]

    # Tableau résumé
    try:
        fig_table = validation_summary_table(elec_vrs, theme_mode="light")
        fig_table.update_layout(height=max(100, 40 * len(elec_vrs) + 50))
        parts.append(_fig_html(fig_table))
    except Exception:
        pass

    # Résidus pour chaque spectre
    for label, vr in elec_vrs.items():
        try:
            fig_res = residuals_figure(vr, theme_mode="light")
            parts.append(f"<h4>{label}</h4>")
            parts.append(_fig_html(fig_res))
        except Exception:
            pass

    parts.append("</div>")
    return "\n".join(parts)


def _summary_html(
    experiment: dict,
    exclusions: dict,
    deleted_points: dict,
) -> str:
    """Résumé des modifications (métriques globales)."""
    mode   = experiment.get("mode", "both")
    n_elec = experiment.get("n_electrodes", 2)

    lines = ["<div class='summary'><h2>Résumé des modifications</h2><table>"]
    lines.append(
        "<thead><tr>"
        "<th>Électrode</th>"
        "<th>EIS exclus</th><th>CV exclus</th>"
        "<th>Points EIS supprimés</th>"
        "</tr></thead><tbody>"
    )

    for ei in range(1, n_elec + 1):
        e_str    = f"e{ei}"
        elec_key = f"electrode_{ei}"
        e_dict   = exclusions.get(e_str, {})

        # Compter exclusions EIS
        n_eis_excl = sum(
            1 for excl_list in e_dict.get("eis", {}).values()
            for ex in excl_list if ex
        )
        # Compter exclusions CV
        n_cv_excl = sum(
            1 for excl_list in e_dict.get("cv", {}).values()
            for ex in excl_list if ex
        )
        # Points EIS supprimés
        n_pts = sum(
            len(v) for k, v in deleted_points.items()
            if k.startswith(f"deleted_points_{e_str}_eis") and v
        )

        lines.append(
            f"<tr>"
            f"<td>Électrode {ei}</td>"
            f"<td>{n_eis_excl if mode in ('eis_only','both') else '—'}</td>"
            f"<td>{n_cv_excl if mode in ('cv_only','both') else '—'}</td>"
            f"<td>{n_pts if mode in ('eis_only','both') else '—'}</td>"
            f"</tr>"
        )

    lines.append("</tbody></table></div>")
    return "\n".join(lines)


# ── Générateur principal ──────────────────────────────────────────────────────

def generate_pretraitement_report_html(
    experiment: dict,
    exclusions: dict,
    deleted_points: dict,
    validation_results: dict = None,
    experiment_clean: dict = None,
) -> str:
    """Génère un rapport HTML autonome, miroir visuel de 1_pretraitement.py.

    Parameters
    ----------
    experiment         : dict complet de l'expérience (avec fichiers UploadedFile).
    exclusions         : st.session_state['exclusions'].
    deleted_points     : dict {dp_key: [indices]} — toutes les clés 'deleted_points_*'.
    validation_results : st.session_state.get('validation_results').
    experiment_clean   : st.session_state.get('experiment_clean') — optionnel.
    """
    global _FIG_ID
    _FIG_ID = 0  # reset pour des IDs déterministes

    name          = experiment.get("name", "Expérience") or "Expérience"
    date          = experiment.get("date", "") or ""
    mode          = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    n_elec        = experiment.get("n_electrodes", 2)

    has_eis = mode in ("eis_only", "both")
    has_cv  = mode in ("cv_only", "both")

    # ── Chargement des spectres ────────────────────────────────────────────────
    eis_spectra = _load_eis(experiment) if has_eis else {"probe": {}, "calibration": {}}
    cv_scans    = _load_cv(experiment)  if has_cv  else {"probe": {}, "calibration": {}}

    # ── En-tête ────────────────────────────────────────────────────────────────
    meta_mode = {"eis_only": "EIS seule", "cv_only": "CV seule", "both": "EIS + CV"}.get(mode, mode)
    header = (
        f"<h1>Prétraitement — {name}</h1>"
        f"<p class='meta'>"
        f"{'Date : ' + date + ' · ' if date else ''}"
        f"Mode : {meta_mode} · "
        f"{len(concentrations)} concentration(s) · "
        f"{n_elec} électrode(s)"
        f"</p><hr>"
    )

    body_parts = [header]

    # ── Section par électrode ──────────────────────────────────────────────────
    for elec_idx in range(1, n_elec + 1):
        e_str    = f"e{elec_idx}"
        elec_key = f"electrode_{elec_idx}"

        body_parts.append(f"<h2>Électrode {elec_idx}</h2>")

        # --- Groupes EIS / CV ---
        probe_eis = (eis_spectra["probe"].get(elec_key) or [])
        probe_cv  = (cv_scans["probe"].get(elec_key) or [])
        cal_eis   = eis_spectra["calibration"].get(elec_key) or []
        cal_cv    = cv_scans["calibration"].get(elec_key) or []

        def _concentration_block_html(label: str, ci, reps_eis: list, reps_cv: list) -> str:
            parts = [f"<div class='concentration-block'><h3>{label}</h3>"]
            parts.append("<div class='graphs-row'>")

            if has_eis and reps_eis:
                eis_nn = [sp for sp in reps_eis if sp is not None]
                if eis_nn:
                    fig_nyq = _fig_superposed_nyquist(reps_eis, e_str, ci, exclusions, deleted_points)
                    parts.append(
                        "<div class='graph-section'>"
                        "<h4>EIS — Nyquist superposé</h4>"
                        + _fig_html(fig_nyq)
                        + "</div>"
                    )

            if has_cv and reps_cv:
                cv_nn = [sc for sc in reps_cv if sc is not None]
                if cv_nn:
                    fig_cv = _fig_superposed_cv(reps_cv, e_str, ci, exclusions)
                    parts.append(
                        "<div class='graph-section'>"
                        "<h4>CV — I(E) superposé</h4>"
                        + _fig_html(fig_cv)
                        + "</div>"
                    )

            parts.append("</div>")  # graphs-row
            parts.append(_exclusion_table(reps_eis, reps_cv, e_str, ci, exclusions, deleted_points, mode))
            parts.append("</div>")  # concentration-block
            return "\n".join(parts)

        # Probe
        if probe_eis or probe_cv:
            body_parts.append(
                _concentration_block_html("Probe", "probe", probe_eis, probe_cv)
            )

        # Chaque concentration
        for ci, conc in enumerate(concentrations):
            reps_eis = cal_eis[ci] if ci < len(cal_eis) else []
            reps_cv  = cal_cv[ci]  if ci < len(cal_cv)  else []
            if not reps_eis and not reps_cv:
                continue
            label = f"Concentration {ci+1} — {_fmt(conc)}"
            body_parts.append(
                _concentration_block_html(label, ci, reps_eis, reps_cv)
            )

        # --- Graphes moyens ---
        body_parts.append("<div class='mean-section'>")
        body_parts.append(f"<h3>Spectres moyens — Électrode {elec_idx}</h3>")
        body_parts.append("<div class='graphs-row'>")

        if has_eis:
            groups_eis = []
            excl_probe_eis = _get_excl(exclusions, e_str, "eis", "probe")
            active_probe_eis = [
                sp for ri, sp in enumerate(probe_eis)
                if sp is not None and not (ri < len(excl_probe_eis) and excl_probe_eis[ri])
            ]
            if active_probe_eis:
                groups_eis.append(("Probe", active_probe_eis))

            for ci, conc in enumerate(concentrations):
                reps = cal_eis[ci] if ci < len(cal_eis) else []
                excl_list = _get_excl(exclusions, e_str, "eis", ci)
                active = [
                    sp for ri, sp in enumerate(reps)
                    if sp is not None and not (ri < len(excl_list) and excl_list[ri])
                ]
                if active:
                    groups_eis.append((_fmt(conc), active))

            if groups_eis:
                try:
                    fig_avg_nyq = _fig_average_nyquist(groups_eis)
                    body_parts.append(
                        "<div class='graph-section'>"
                        "<h4>Nyquist moyen</h4>"
                        + _fig_html(fig_avg_nyq)
                        + "</div>"
                    )
                except Exception:
                    pass

        if has_cv:
            groups_cv = []
            excl_probe_cv = _get_excl(exclusions, e_str, "cv", "probe")
            active_probe_cv = [
                sc for ri, sc in enumerate(probe_cv)
                if sc is not None and not (ri < len(excl_probe_cv) and excl_probe_cv[ri])
            ]
            if active_probe_cv:
                groups_cv.append(("Probe", active_probe_cv))

            for ci, conc in enumerate(concentrations):
                reps = cal_cv[ci] if ci < len(cal_cv) else []
                excl_list = _get_excl(exclusions, e_str, "cv", ci)
                active = [
                    sc for ri, sc in enumerate(reps)
                    if sc is not None and not (ri < len(excl_list) and excl_list[ri])
                ]
                if active:
                    groups_cv.append((_fmt(conc), active))

            if groups_cv:
                try:
                    fig_avg_cv = _fig_average_cv(groups_cv)
                    body_parts.append(
                        "<div class='graph-section'>"
                        "<h4>CV moyen</h4>"
                        + _fig_html(fig_avg_cv)
                        + "</div>"
                    )
                except Exception:
                    pass

        body_parts.append("</div></div>")  # graphs-row + mean-section

        # --- KK ---
        if validation_results:
            body_parts.append(_kk_section_html(validation_results, elec_idx))

    # ── Résumé final ──────────────────────────────────────────────────────────
    body_parts.append(_summary_html(experiment, exclusions, deleted_points))

    # ── Assemblage final ──────────────────────────────────────────────────────
    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport prétraitement — {name}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Lora:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root {{
      --ink:    #0f1117;
      --paper:  #f7f4ef;
      --card:   #ffffff;
      --border: #e2ddd6;
      --accent: #1a56db;
      --warn:   #c2410c;
      --ok:     #15803d;
      --muted:  #6b7280;
      --sans:   'Syne', sans-serif;
      --serif:  'Lora', Georgia, serif;
      --mono:   'JetBrains Mono', monospace;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      background: var(--paper);
      color: var(--ink);
      font-family: var(--serif);
      margin: 0;
      padding: 2rem 3rem;
      max-width: 1600px;
    }}
    h1 {{
      font-family: var(--sans);
      font-size: 2rem;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 0.25rem;
    }}
    h2 {{
      font-family: var(--sans);
      font-size: 1.3rem;
      font-weight: 700;
      border-bottom: 2px solid var(--border);
      padding-bottom: 0.4rem;
      margin-top: 2.5rem;
    }}
    h3 {{
      font-family: var(--sans);
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--ink);
      margin-top: 1.5rem;
      margin-bottom: 0.5rem;
    }}
    h4 {{
      font-family: var(--sans);
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--muted);
      margin: 0.5rem 0 0.25rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    p.meta {{
      font-family: var(--sans);
      font-size: 0.9rem;
      color: var(--muted);
      margin: 0.25rem 0 1rem;
    }}
    hr {{ border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }}

    /* Blocs */
    .concentration-block {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem 1.25rem;
      margin: 1rem 0;
    }}
    .mean-section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 6px;
      padding: 1rem 1.25rem;
      margin: 1.5rem 0;
    }}
    .kk-section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem 1.25rem;
      margin: 1.5rem 0;
    }}
    .summary {{
      background: var(--card);
      border: 1px solid var(--border);
      padding: 1.5rem;
      margin-top: 2rem;
      border-radius: 6px;
    }}

    /* Grille */
    .graphs-row {{
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      align-items: flex-start;
    }}
    .graph-section {{
      flex: 1;
      min-width: 380px;
      overflow: hidden;
    }}

    /* Tableaux */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      font-family: var(--sans);
      margin: 0.75rem 0;
    }}
    th {{
      background: var(--paper);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 0.45rem 0.6rem;
      border-bottom: 2px solid var(--border);
      text-align: left;
    }}
    td {{
      padding: 0.4rem 0.6rem;
      border-bottom: 1px solid var(--border);
    }}
    tr:hover td {{ background: #fafafa; }}
    .ok   {{ color: var(--ok);   font-weight: 600; }}
    .warn {{ color: var(--warn); font-weight: 600; }}
    .excl {{ color: var(--muted); text-decoration: line-through; }}
  </style>
</head>
<body>
{"".join(body_parts)}
</body>
</html>"""

    return full_html


def generate_pretraitement_report_pdf(
    experiment: dict,
    exclusions: dict,
    deleted_points: dict,
    validation_results: dict = None,
    experiment_clean: dict = None,
) -> bytes | None:
    """Génère un PDF via weasyprint (optionnel). Retourne None si non disponible."""
    try:
        html = generate_pretraitement_report_html(
            experiment, exclusions, deleted_points, validation_results, experiment_clean
        )
        try:
            from weasyprint import HTML
            return HTML(string=html).write_pdf()
        except ImportError:
            return None
    except Exception:
        return None
