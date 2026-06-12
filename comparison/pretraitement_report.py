"""Génération de rapports HTML/PDF du prétraitement.

Fonctions exportées :
  generate_pretraitement_report_html(experiment, exclusions, validation_results) -> str
  generate_pretraitement_report_pdf(experiment, exclusions, validation_results)  -> bytes | None
"""

from __future__ import annotations

import numpy as np


# ── Helpers internes ────────────────────────────────────────────────────────────

_SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _format_conc(conc: float) -> str:
    if conc <= 0:
        return str(conc)
    exp = int(np.floor(np.log10(conc)))
    mant = conc / 10 ** exp
    return f"{mant:.0f}×10{str(exp).translate(_SUP_MAP)} M"


def _load_spectra_for_electrode(experiment: dict, elec_idx: int) -> list:
    """Charge les spectres EIS moyennés pour une électrode.

    Retourne une liste de dicts {"label", "Zre", "Zim", "concentration"}.
    """
    from core.loader import load_spectrum, average_replicates

    key        = f"electrode_{elec_idx}"
    probe_dict = (experiment.get("probe") or {}).get("eis") or {}
    cal_dict   = (experiment.get("calibration") or {}).get("eis") or {}
    concs      = experiment.get("concentrations") or []
    result     = []

    probe_files = probe_dict.get(key) or []
    probe_specs = []
    for ri, bio in enumerate(probe_files):
        if bio is None:
            continue
        try:
            bio.seek(0)
            sp = load_spectrum(bio.read(), label=f"probe_e{elec_idx}_r{ri+1}")
            bio.seek(0)
            probe_specs.append(sp)
        except Exception:
            pass
    if probe_specs:
        avg = average_replicates(probe_specs) if len(probe_specs) > 1 else probe_specs[0]
        result.append({"label": "Probe", "Zre": avg.Zre, "Zim": avg.Zim, "concentration": 0.0})

    for ci, rep_files in enumerate(cal_dict.get(key) or []):
        conc = concs[ci] if ci < len(concs) else 0.0
        reps = []
        for ri, bio in enumerate(rep_files or []):
            if bio is None:
                continue
            try:
                bio.seek(0)
                sp = load_spectrum(bio.read(), label=f"e{elec_idx}_c{ci+1}_r{ri+1}")
                bio.seek(0)
                reps.append(sp)
            except Exception:
                pass
        if reps:
            avg = average_replicates(reps) if len(reps) > 1 else reps[0]
            result.append({
                "label": f"C{ci+1} = {_format_conc(conc)}",
                "Zre":   avg.Zre,
                "Zim":   avg.Zim,
                "concentration": conc,
            })

    return result


def _load_cv_for_electrode(experiment: dict, elec_idx: int) -> list:
    """Charge les courbes CV moyennées pour une électrode.

    Retourne une liste de dicts {"label", "E", "I", "concentration"}.
    """
    from core.cv_loader import load_cv_file, average_cv_replicates

    key        = f"electrode_{elec_idx}"
    probe_dict = (experiment.get("probe") or {}).get("cv") or {}
    cal_dict   = (experiment.get("calibration") or {}).get("cv") or {}
    concs      = experiment.get("concentrations") or []
    result     = []

    probe_files = probe_dict.get(key) or []
    probe_scans = []
    for ri, bio in enumerate(probe_files):
        if bio is None:
            continue
        try:
            bio.seek(0)
            sc = load_cv_file(bio.read(), label=f"probe_e{elec_idx}_r{ri+1}",
                              concentration=0.0, step="probe")
            bio.seek(0)
            probe_scans.append(sc)
        except Exception:
            pass
    if probe_scans:
        avg = average_cv_replicates(probe_scans) if len(probe_scans) > 1 else probe_scans[0]
        result.append({"label": "Probe", "E": avg.E, "I": avg.I, "concentration": 0.0})

    for ci, rep_files in enumerate(cal_dict.get(key) or []):
        conc = concs[ci] if ci < len(concs) else 0.0
        reps = []
        for ri, bio in enumerate(rep_files or []):
            if bio is None:
                continue
            try:
                bio.seek(0)
                sc = load_cv_file(bio.read(), label=f"e{elec_idx}_c{ci+1}_r{ri+1}",
                                  concentration=conc, step="hybridization")
                bio.seek(0)
                reps.append(sc)
            except Exception:
                pass
        if reps:
            avg = average_cv_replicates(reps) if len(reps) > 1 else reps[0]
            result.append({
                "label": f"C{ci+1} = {_format_conc(conc)}",
                "E": avg.E,
                "I": avg.I,
                "concentration": conc,
            })

    return result


def _exclusion_table_html(excl: dict, concentrations: list) -> str:
    """Génère un tableau HTML des exclusions pour une électrode."""
    rows = []
    for ci_key, excl_list in excl.get("eis", {}).items():
        if ci_key == "probe":
            label = "Probe (EIS)"
        else:
            ci = int(ci_key)
            label = f"C{ci+1} = {_format_conc(concentrations[ci])}" if ci < len(concentrations) else f"C{ci+1}"
        for ri, excluded in enumerate(excl_list):
            cls = "excluded" if excluded else "valid"
            status = "Exclu" if excluded else "OK"
            rows.append(
                f"<tr><td>{label}</td><td>Rép {ri+1}</td>"
                f'<td class="{cls}">{status}</td></tr>'
            )
    for ci_key, excl_list in excl.get("cv", {}).items():
        if ci_key == "probe":
            label = "Probe (CV)"
        else:
            ci = int(ci_key)
            label = f"C{ci+1} = {_format_conc(concentrations[ci])}" if ci < len(concentrations) else f"C{ci+1}"
        for ri, excluded in enumerate(excl_list):
            cls = "excluded" if excluded else "valid"
            status = "Exclu" if excluded else "OK"
            rows.append(
                f"<tr><td>{label}</td><td>Rép {ri+1}</td>"
                f'<td class="{cls}">{status}</td></tr>'
            )

    if not rows:
        return "<p><em>Aucune exclusion enregistrée.</em></p>"

    return (
        "<table><thead><tr>"
        "<th>Groupe</th><th>Réplicat</th><th>Statut</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _kk_table_html(validation_results: dict, elec_idx: int) -> str:
    """Génère un tableau HTML des résultats KK pour une électrode."""
    rows = []
    for key, vr in validation_results.items():
        if f"electrode_{elec_idx}" not in key and f"e{elec_idx}_" not in key:
            continue
        for rep in getattr(vr, "replicates", []):
            mu    = getattr(rep, "mu", None)
            chi2  = getattr(rep, "chi2", None)
            valid = getattr(rep, "all_valid", None)
            cls   = "valid" if valid else "warning"
            verdict = "✓ Valide" if valid else "⚠ Attention"
            rows.append(
                f"<tr>"
                f"<td>{getattr(rep, 'label', key)}</td>"
                f"<td>{f'{mu:.4f}' if mu is not None else '—'}</td>"
                f"<td>{f'{chi2:.3e}' if chi2 is not None else '—'}</td>"
                f'<td class="{cls}">{verdict}</td>'
                f"</tr>"
            )

    if not rows:
        return "<p><em>Résultats KK non disponibles.</em></p>"

    return (
        "<h4>Résultats Kramers-Kronig</h4>"
        "<table><thead><tr>"
        "<th>Spectre</th><th>µ</th><th>χ²</th><th>Verdict</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


# ── API publique ────────────────────────────────────────────────────────────────

def generate_pretraitement_report_html(
    experiment: dict,
    exclusions: dict,
    validation_results: dict = None,
) -> str:
    """Génère un rapport HTML autonome avec tous les graphiques du prétraitement.

    Contenu :
      - Métadonnées de l'expérience (nom, date, mode, concentrations)
      - Pour chaque électrode :
          * Graphe Nyquist superposé (EIS si disponible)
          * Graphe CV superposé (CV si disponible)
          * Tableau des exclusions
          * Résultats KK si disponibles
    """
    import plotly.io as pio
    from plotting.eis_plots import nyquist_figure_electrode
    from plotting.cv_plots import cv_figure_electrode

    name          = experiment.get("name", "Expérience") or "Expérience"
    date          = experiment.get("date", "")
    mode          = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    n_elec        = experiment.get("n_electrodes", 2)

    header = (
        f"<h1>Rapport de prétraitement — {name}</h1>"
        f"<p>Date : {date} | Mode : {mode} | "
        f"Concentrations : {len(concentrations)}</p>"
        "<hr>"
    )

    sections = []

    for elec_idx in range(1, n_elec + 1):
        e_str = f"e{elec_idx}"
        parts = [f"<h2>Électrode {elec_idx}</h2>"]

        if mode in ("eis_only", "both"):
            try:
                spectra = _load_spectra_for_electrode(experiment, elec_idx)
                if spectra:
                    fig = nyquist_figure_electrode(spectra, title=f"Spectres EIS — Électrode {elec_idx}")
                    parts.append("<h3>Spectres EIS</h3>")
                    parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            except Exception:
                parts.append("<p><em>Erreur lors du chargement des spectres EIS.</em></p>")

        if mode in ("cv_only", "both"):
            try:
                cv_data = _load_cv_for_electrode(experiment, elec_idx)
                if cv_data:
                    fig = cv_figure_electrode(cv_data, title=f"Courbes CV — Électrode {elec_idx}")
                    parts.append("<h3>Courbes CV</h3>")
                    parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            except Exception:
                parts.append("<p><em>Erreur lors du chargement des courbes CV.</em></p>")

        excl = exclusions.get(e_str, {})
        parts.append("<h3>Exclusions</h3>")
        parts.append(_exclusion_table_html(excl, concentrations))

        if validation_results:
            parts.append(_kk_table_html(validation_results, elec_idx))

        sections.append("".join(parts))

    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport prétraitement — {name}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    h1   {{ color: #1a56db; }}
    h2   {{ color: #374151; border-bottom: 1px solid #d1d5db; padding-bottom: 0.25rem; }}
    h3   {{ color: #4b5563; }}
    table{{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th   {{ background: #f0ede8; padding: 0.5rem; text-align: left; border-bottom: 2px solid #d1d5db; }}
    td   {{ padding: 0.5rem; border-bottom: 1px solid #e5e7eb; }}
    .valid   {{ color: #15803d; font-weight: bold; }}
    .warning {{ color: #c2410c; font-weight: bold; }}
    .excluded{{ color: #9ca3af; text-decoration: line-through; }}
    hr   {{ border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }}
  </style>
</head>
<body>
  {header}
  {''.join(sections)}
</body>
</html>"""
    return full_html


def generate_pretraitement_report_pdf(
    experiment: dict,
    exclusions: dict,
    validation_results: dict = None,
) -> bytes | None:
    """Génère un PDF du rapport prétraitement via weasyprint (optionnel).

    Retourne None si weasyprint n'est pas installé.
    """
    try:
        html_content = generate_pretraitement_report_html(
            experiment, exclusions, validation_results
        )
        try:
            from weasyprint import HTML
            return HTML(string=html_content).write_pdf()
        except ImportError:
            return None
    except Exception:
        return None
