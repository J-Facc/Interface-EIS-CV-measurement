"""Export helpers for EIS Analyzer sessions."""

from __future__ import annotations

import io
import csv
import zipfile
import yaml
import numpy as np

from core.models import EISSession
from core.calibration import compute_calibration_all, compute_cv_calibration
from core.results_table import group_rows, replicate_rows


def _as_sessions_dict(sessions) -> dict:
    """Accepte un EISSession unique ou un dict {electrode: EISSession} et
    retourne toujours un dict, pour compatibilité ascendante des exports."""
    if isinstance(sessions, dict):
        return sessions
    return {1: sessions}


_PARAM_BASE_COLUMNS = [
    "electrode", "group", "concentration", "spectrum", "kind", "model",
    "target_param", "target_value", "target_std", "chi2_reduced", "converged",
]


def _fit_rows(sessions: dict) -> list:
    """(colonnes de base, {modèle: [noms de paramètres]}) de chaque FitResult : chaque
    réplicat BRUT puis le spectre moyen, pour chaque groupe (bare, probe, concentrations)."""
    out = []
    for e, session in sorted(sessions.items()):
        for group, mean_sp, reps, _an in session.iter_groups():
            spectra = [(sp, "réplicat") for sp in reps] + [(mean_sp, "moyenne")]
            for sp, kind in spectra:
                for model_name, fit in sp.fit_results.items():
                    out.append((dict(
                        electrode=e, group=group, concentration=sp.concentration,
                        spectrum=sp.label, kind=kind, model=model_name,
                        target_param=fit.target_param, target_value=fit.target_value,
                        target_std=fit.target_std, chi2_reduced=fit.chi2_reduced,
                        converged=fit.converged,
                    ), model_name, fit))
    return out


def export_params_csv(sessions) -> bytes:
    """Paramètres de TOUS les fits (circuit Orazem et DRT), réplicats ET moyennes.

    Chaque modèle écrit SES PROPRES colonnes, nommées ``<modèle>.<paramètre>`` (et
    ``<modèle>.<paramètre>_std`` quand un écart-type existe) : une ligne ne remplit que
    les colonnes de son modèle, les autres restent vides. Corrige B-EXP (AUDIT.md
    §8.4) : l'en-tête était écrit une seule fois d'après le PREMIER modèle rencontré et
    les lignes des autres modèles y étaient alignées par POSITION (Rp de la DRT sous
    « Re », ln τ sous « Re_prime »…). L'alignement se fait désormais par NOM
    (``csv.DictWriter``) et un même nom de paramètre de deux modèles (« Rct » du
    circuit vs « Rct » DRT, grandeurs différentes) ne partage jamais une colonne.

    Accepte un EISSession unique (électrode 1) ou un dict {electrode_index: EISSession}.
    """
    rows = _fit_rows(_as_sessions_dict(sessions))
    model_columns: dict = {}
    for _base, model_name, fit in rows:
        cols = model_columns.setdefault(model_name, [])
        for k in fit.params:
            for col in (f"{model_name}.{k}",) + ((f"{model_name}.{k}_std",) if k in fit.params_std else ()):
                if col not in cols:
                    cols.append(col)
    header = _PARAM_BASE_COLUMNS + [c for cols in model_columns.values() for c in cols]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, restval="")
    writer.writeheader()
    for base, model_name, fit in rows:
        row = dict(base)
        for k, v in fit.params.items():
            row[f"{model_name}.{k}"] = v
            if k in fit.params_std:
                row[f"{model_name}.{k}_std"] = fit.params_std[k]
        writer.writerow(row)
    return buf.getvalue().encode()


def _rows_csv(rows: list) -> bytes:
    """Liste de dicts → CSV, colonnes = union dans l'ordre d'apparition."""
    header: list = []
    for r in rows:
        for k in r:
            if k not in header:
                header.append(k)
    buf = io.StringIO()
    if header:
        writer = csv.DictWriter(buf, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue().encode()


def export_replicate_results_csv(sessions) -> bytes:
    """Une ligne par spectre (réplicats bruts + moyenne) : valeur cible et incertitude
    INTRA-fit du circuit, et Rct DRT avec ses diagnostics HMC (``core/results_table.py``)."""
    return _rows_csv(replicate_rows(_as_sessions_dict(sessions)))


def export_group_results_csv(sessions) -> bytes:
    """Une ligne par groupe : statut, verdict KK, incertitude intra-fit et variabilité
    INTER-réplicats, côte à côte (``core/results_table.py``)."""
    return _rows_csv(group_rows(_as_sessions_dict(sessions)))


def _yaml_value(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_yaml_value(x) for x in v]
    if isinstance(v, (str, bool)) or v is None:
        return v
    return float(v)


def _fit_yaml(fit) -> dict:
    return {
        "target_param": str(fit.target_param),
        "target_value": float(fit.target_value),
        "target_std": float(fit.target_std),
        "chi2_reduced": float(fit.chi2_reduced),
        "converged": bool(fit.converged),
        "params": {k: _yaml_value(v) for k, v in fit.params.items()},
        "warnings": list(fit.warnings or []),
    }


def _aggregate_yaml(agg) -> dict:
    return {k: _yaml_value(getattr(agg, k)) for k in (
        "name", "n", "n_excluded", "mean", "std_between", "std_within", "sem_within", "sem",
        "q", "q_pvalue", "values", "stds")}


def export_session_yaml(session: EISSession) -> str:
    """Résumé YAML de la session : circuit, puis pour chaque groupe (bare, probe,
    concentrations) son statut, le fit de la moyenne, les fits PAR RÉPLICAT et les
    agrégats (incertitude intra-fit vs variabilité inter-réplicats)."""
    data: dict = {
        "created_at": str(session.created_at),
        "circuit": session.circuit,
        "drt_mode": session.drt_mode,
        "messages": list(session.messages),
        "load_errors": list(session.load_errors),
        "groups": [],
    }

    for group, mean_sp, reps, an in session.iter_groups():
        grp_data: dict = {
            "group": group,
            "concentration": float(mean_sp.concentration),
            "n_points": mean_sp.n_points,
            "n_replicates": len(reps),
            "status": an.status if an is not None else None,
            "message": an.message if an is not None else None,
            "fits": {m: _fit_yaml(f) for m, f in mean_sp.fit_results.items()},
            "replicates": [
                {"label": sp.label, "fits": {m: _fit_yaml(f) for m, f in sp.fit_results.items()}}
                for sp in reps
            ],
        }
        if an is not None and an.orazem is not None:
            grp_data["orazem_target"] = _aggregate_yaml(an.orazem.target)
        if an is not None and an.drt_target is not None:
            grp_data["drt_target"] = _aggregate_yaml(an.drt_target)
        if an is not None and an.drt_failures:
            grp_data["drt_failures"] = dict(an.drt_failures)
        data["groups"].append(grp_data)

    return yaml.dump(data, allow_unicode=True, sort_keys=False)


# ── Calibration EIS — export multi-électrode / multi-méthode (ajout) ─────────

def export_calibration_csv(sessions: dict, strict: bool = False) -> bytes:
    """Exporte, pour chaque électrode et méthode, conc / log10(conc) / signal
    normalisé / Rct / paramètres de régression (slope, intercept, r2, p_value, std_err)
    dans un CSV multi-colonnes unique.

    sessions: {electrode_index: EISSession} ou un EISSession unique.
    strict: exclut aussi les points au verdict KK « non conforme » (défaut : tous les points
    calculables, la colonne ``kk_non_conforme`` signale ceux à interpréter avec prudence).
    """
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "electrode", "model", "concentration_M", "log10_concentration",
        "signal_norm", "Rct_Ohm", "Rct_probe_Ohm",
        "slope", "intercept", "r2", "p_value", "std_err", "kk_non_conforme",
    ])

    # Calcul délégué à core/calibration.py : mêmes pente/ordonnée/R² que
    # plotting.eis_plots.calibration_figure (source unique).
    for e, session in sorted(sessions.items()):
        for cal in compute_calibration_all(session, strict):
            for conc, lc, sig, rct, kk in zip(cal.concentrations, cal.log_c, cal.y, cal.rcts,
                                              cal.kk_flags):
                writer.writerow([
                    e, cal.model, conc, lc, sig, rct, cal.probe_rct,
                    cal.slope, cal.intercept, cal.r2, cal.pvalue, cal.stderr, bool(kk),
                ])

    return buf.getvalue().encode()


# ── Calibration CV — export ───────────────────────────────────────────────────

def export_cv_calibration_csv_multi(cv_sessions: dict) -> bytes:
    """Exporte la calibration CV pour plusieurs électrodes : une section par
    électrode, colonnes electrode / concentration / log10(concentration) /
    delta_signal_moyen / slope / intercept / r2 / p_value / std_err."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "electrode", "concentration_M", "log10_concentration", "delta_signal_moyen",
        "slope", "intercept", "r2", "p_value", "std_err",
    ])

    for elec, session in sorted(cv_sessions.items()):
        cal = compute_cv_calibration(session)  # source unique (core/calibration.py)
        if cal is None:
            continue
        for conc, lc, sig in zip(cal.concentrations, cal.log_c, cal.signals):
            writer.writerow([
                elec, conc, lc, sig,
                cal.slope, cal.intercept, cal.r2, cal.pvalue, cal.stderr,
            ])

    return buf.getvalue().encode()


# ── DRT — export multi-électrode, multi-spectre, multi-réplicat ──────────────

def export_drt_csv(sessions: dict) -> bytes:
    """Exporte les valeurs DRT (ln_tau, ln_gamma) du modèle 'drt_bayes'.

    Le pipeline calcule la DRT de CHAQUE réplicat brut (``replicate_idx`` = indice)
    et du spectre moyen (``replicate_idx`` = 'avg'). Un spectre sans DRT (moteur
    absent, échec enregistré dans ``GroupAnalysis.drt_failures``) est simplement omis.
    La colonne ``drt_mode`` distingue 'optimize' (MAP) et 'sample' (HMC).
    """
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "electrode", "label", "concentration", "replicate_idx", "drt_mode",
        "ln_tau", "ln_gamma",
    ])

    def _write_spectrum(e, label, conc, replicate_idx, sp):
        """Écrit les lignes DRT d'un spectre s'il porte un FitResult 'drt_bayes'."""
        if sp is None:
            return
        fr = (getattr(sp, "fit_results", {}) or {}).get("drt_bayes")
        if fr is None:
            return
        tau = getattr(fr, "drt_tau", None)
        gamma = getattr(fr, "drt_gamma", None)
        if tau is None or gamma is None:
            return
        mode = getattr(fr, "drt_mode", None) or "optimize"
        ln_tau = np.log(np.asarray(tau, dtype=float) + 1e-300)
        ln_gamma = np.log(np.asarray(gamma, dtype=float) + 1e-300)
        for lt, lg in zip(ln_tau, ln_gamma):
            writer.writerow([e, label, conc, replicate_idx, mode, lt, lg])

    for e, session in sorted(sessions.items()):
        # (label, concentration, spectre moyenné, réplicats).
        spectra_by_label = []
        if session.bare is not None:
            spectra_by_label.append(("bare", 0.0, session.bare, session.bare_replicate_spectra))
        if session.probe is not None:
            spectra_by_label.append(("probe", 0.0, session.probe, session.probe_replicate_spectra))
        for grp in session.groups:
            spectra_by_label.append(
                (grp.spectrum.label, grp.concentration, grp.spectrum, grp.replicate_spectra)
            )

        for label, conc, averaged, reps in spectra_by_label:
            # Spectre moyen.
            _write_spectrum(e, label, conc, "avg", averaged)
            # Réplicats bruts : chacun porte sa propre DRT.
            for ri, sp in enumerate(reps or []):
                _write_spectrum(e, label, conc, ri, sp)

    return buf.getvalue().encode()


# ── Normalisation Nyquist (probe vs concentration) ────────────────────────────

def export_normalization_csv(normalized: dict) -> bytes:
    """Exporte la vue Nyquist normalisée construite par
    pages/A_eis.py::_build_normalized_session : {concentration: {"Zre_norm",
    "Zim_norm", ...}}. Nécessite la fréquence — reconstruite depuis la grille
    du probe si fournie sous la clé 'f' dans chaque entrée, sinon un index."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["concentration", "frequency_Hz", "Zre_norm", "Zim_norm"])

    for conc, d in sorted((normalized or {}).items()):
        Zre_norm = np.asarray(d.get("Zre_norm"))
        Zim_norm = np.asarray(d.get("Zim_norm"))
        freqs = d.get("f")
        if freqs is None:
            freqs = np.arange(len(Zre_norm))
        for f, zre, zim in zip(freqs, Zre_norm, Zim_norm):
            writer.writerow([conc, f, zre, zim])

    return buf.getvalue().encode()


# ── Reconstructions circuit (Orazem)/DRT vs mesure ─────────────────────────────────────

def export_reconstruction_csv(sessions: dict) -> bytes:
    """Exporte, pour chaque électrode/spectre MOYEN/méthode (circuit Orazem, DRT),
    les valeurs mesurées et reconstruites et l'erreur de reconstruction (RMS
    relative). Les tableaux d'un FitResult suivent l'ordre des fréquences du spectre."""
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "electrode", "label", "concentration", "model",
        "f_Hz", "Zre_mesure", "Zim_mesure", "Zre_reconstruit", "Zim_reconstruit",
        "erreur_reconstruction",
    ])

    for e, session in sorted(sessions.items()):
        spectra = []
        if session.bare is not None:
            spectra.append(("bare", 0.0, session.bare))
        if session.probe is not None:
            spectra.append(("probe", 0.0, session.probe))
        for grp in session.groups:
            spectra.append((grp.spectrum.label, grp.concentration, grp.spectrum))

        for label, conc, sp in spectra:
            for model, fr in sp.fit_results.items():
                Zfit_re = getattr(fr, "Zfit_re", None)
                Zfit_im = getattr(fr, "Zfit_im", None)
                if Zfit_re is None or Zfit_im is None:
                    continue
                err = getattr(fr, "reconstruction_error", None)
                for f, zre, zim, zfre, zfim in zip(sp.f, sp.Zre, sp.Zim, Zfit_re, Zfit_im):
                    writer.writerow([e, label, conc, model, f, zre, zim, zfre, zfim, err])

    return buf.getvalue().encode()


# ── Archive complète ───────────────────────────────────────────────────────────

def export_full_zip(
    experiment_clean: dict,
    sessions: dict,
    normalized: dict,
    cv_session: dict | None = None,
    config: dict | None = None,
) -> bytes:
    """Construit une archive ZIP en mémoire regroupant tous les exports
    disponibles. Toute donnée absente est silencieusement omise (pas de
    fichier/dossier vide créé)."""
    from core.loader import load_spectrum, average_replicates as _avg_eis_reps

    sessions = _as_sessions_dict(sessions) if sessions else {}
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # data_pretraitees/ — spectres EIS/CV bruts par électrode
        if experiment_clean:
            n_elec = experiment_clean.get("n_electrodes", 2)
            mode = experiment_clean.get("mode", "both")
            for e in range(1, n_elec + 1):
                key = f"electrode_{e}"
                if mode in ("eis_only", "both"):
                    rows = []
                    probe_files = ((experiment_clean.get("probe") or {}).get("eis") or {}).get(key) or []
                    specs = []
                    for ri, bio in enumerate(probe_files):
                        if bio is None:
                            continue
                        bio.seek(0)
                        content = bio.read()
                        bio.seek(0)
                        specs.append(load_spectrum(content, label=f"probe_e{e}_r{ri+1}"))
                    if specs:
                        avg = _avg_eis_reps(specs) if len(specs) > 1 else specs[0]
                        for f, zre, zim in zip(avg.f, avg.Zre, avg.Zim):
                            rows.append(["probe", 0.0, f, zre, zim])
                    cal = ((experiment_clean.get("calibration") or {}).get("eis") or {}).get(key) or []
                    concs = experiment_clean.get("concentrations") or []
                    for ci, rep_files in enumerate(cal):
                        conc = concs[ci] if ci < len(concs) else 0.0
                        specs = []
                        for ri, bio in enumerate(rep_files or []):
                            if bio is None:
                                continue
                            bio.seek(0)
                            content = bio.read()
                            bio.seek(0)
                            specs.append(load_spectrum(content, label=f"e{e}_c{ci+1}_r{ri+1}"))
                        if specs:
                            avg = _avg_eis_reps(specs) if len(specs) > 1 else specs[0]
                            for f, zre, zim in zip(avg.f, avg.Zre, avg.Zim):
                                rows.append([f"c{ci+1}", conc, f, zre, zim])
                    if rows:
                        sub = io.StringIO()
                        w = csv.writer(sub)
                        w.writerow(["label", "concentration", "f_Hz", "Zre_Ohm", "Zim_Ohm"])
                        w.writerows(rows)
                        zf.writestr(f"export/data_pretraitees/eis_electrode_{e}.csv", sub.getvalue())

                if mode in ("cv_only", "both"):
                    from core.cv_loader import load_cv_file, average_cv_replicates
                    rows = []
                    probe_files = ((experiment_clean.get("probe") or {}).get("cv") or {}).get(key) or []
                    scans = []
                    for ri, bio in enumerate(probe_files):
                        if bio is None:
                            continue
                        bio.seek(0)
                        content = bio.read()
                        bio.seek(0)
                        scans.append(load_cv_file(content, label=f"probe_e{e}_r{ri+1}", concentration=0.0, step="probe"))
                    if scans:
                        avg = average_cv_replicates(scans) if len(scans) > 1 else scans[0]
                        for E, I in zip(avg.E, avg.I):
                            rows.append(["probe", 0.0, E, I])
                    cal = ((experiment_clean.get("calibration") or {}).get("cv") or {}).get(key) or []
                    concs = experiment_clean.get("concentrations") or []
                    for ci, rep_files in enumerate(cal):
                        conc = concs[ci] if ci < len(concs) else 0.0
                        scans = []
                        for ri, bio in enumerate(rep_files or []):
                            if bio is None:
                                continue
                            bio.seek(0)
                            content = bio.read()
                            bio.seek(0)
                            scans.append(load_cv_file(content, label=f"e{e}_c{ci+1}_r{ri+1}", concentration=conc, step="hybridization"))
                        if scans:
                            avg = average_cv_replicates(scans) if len(scans) > 1 else scans[0]
                            for E, I in zip(avg.E, avg.I):
                                rows.append([f"c{ci+1}", conc, E, I])
                    if rows:
                        sub = io.StringIO()
                        w = csv.writer(sub)
                        w.writerow(["label", "concentration", "E_V", "I_A"])
                        w.writerows(rows)
                        zf.writestr(f"export/data_pretraitees/cv_electrode_{e}.csv", sub.getvalue())

        # drt/
        if sessions:
            drt_bytes = export_drt_csv(sessions)
            if drt_bytes.strip():
                zf.writestr("export/drt/drt_values.csv", drt_bytes)

        # normalisation/
        if normalized:
            zf.writestr("export/normalisation/nyquist_normalise.csv", export_normalization_csv(normalized))

        # fits/ — paramètres de tous les modèles (colonnes par modèle, B-EXP) et
        # résultats par réplicat / par groupe (intra-fit vs inter-réplicats).
        if sessions:
            for e, session in sorted(sessions.items()):
                one = {e: session}
                params = export_params_csv(one)
                if params.count(b"\n") > 1:
                    zf.writestr(f"export/fits/parametres_electrode_{e}.csv", params)
            zf.writestr("export/fits/resultats_par_replicat.csv", export_replicate_results_csv(sessions))
            zf.writestr("export/fits/resultats_par_groupe.csv", export_group_results_csv(sessions))

        # reconstructions/
        if sessions:
            recon_bytes = export_reconstruction_csv(sessions)
            if recon_bytes.strip():
                zf.writestr("export/reconstructions/reconstruction_values.csv", recon_bytes)

        # calibration/
        if sessions:
            cal_eis = export_calibration_csv(sessions)
            if cal_eis.strip():
                zf.writestr("export/calibration/eis_calibration.csv", cal_eis)
        if cv_session:
            cal_cv = export_cv_calibration_csv_multi(cv_session)
            if cal_cv.strip():
                zf.writestr("export/calibration/cv_calibration.csv", cal_cv)

        # session/
        if sessions:
            for e, session in sorted(sessions.items()):
                zf.writestr(
                    f"export/session/eis_session_electrode_{e}.yaml",
                    export_session_yaml(session),
                )

    return buf.getvalue()
