"""Export helpers for EIS Analyzer sessions."""

from __future__ import annotations

import io
import csv
import math
import zipfile
import yaml
import numpy as np
from scipy import stats

from core.models import EISSession
from core.cv_models import CVSession


def _as_sessions_dict(sessions) -> dict:
    """Accepte un EISSession unique ou un dict {electrode: EISSession} et
    retourne toujours un dict, pour compatibilité ascendante des exports."""
    if isinstance(sessions, dict):
        return sessions
    return {1: sessions}


def export_params_csv(sessions) -> bytes:
    """Return fitted parameters for all groups as CSV bytes.

    Accepte un EISSession unique (rétro-compatibilité) ou un dict
    {electrode_index: EISSession}.
    """
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)

    header_written = False
    for e, session in sorted(sessions.items()):
        for grp in session.groups:
            for model_name, fit in grp.fit_results.items():
                row_base = {
                    "electrode": e,
                    "concentration": grp.concentration,
                    "model": model_name,
                    "Rct": fit.Rct,
                    "Rct_std": fit.Rct_std,
                    "chi2": fit.chi2,
                    "converged": fit.converged,
                }
                row_base.update(fit.params)
                if not header_written:
                    writer.writerow(list(row_base.keys()))
                    header_written = True
                writer.writerow(list(row_base.values()))

    return buf.getvalue().encode()


def export_spectra_csv(sessions) -> bytes:
    """Return raw spectra (f, Zre, Zim) for all groups as CSV bytes.

    Accepte un EISSession unique (rétro-compatibilité) ou un dict
    {electrode_index: EISSession}.
    """
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["electrode", "label", "concentration", "f_Hz", "Zre_Ohm", "Zim_Ohm"])

    for e, session in sorted(sessions.items()):
        spectra = []
        if session.bare is not None:
            spectra.append(("bare", 0.0, session.bare))
        if session.probe is not None:
            spectra.append(("probe", 0.0, session.probe))
        for grp in session.groups:
            spectra.append((grp.spectrum.label, grp.concentration, grp.spectrum))

        for label, conc, sp in spectra:
            for f, zre, zim in zip(sp.f, sp.Zre, sp.Zim):
                writer.writerow([e, label, conc, f, zre, zim])

    return buf.getvalue().encode()


def export_figure_html(fig) -> str:
    """Return a Plotly figure as a standalone HTML string."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def export_figure_png(fig, config: dict) -> bytes:
    """Return a Plotly figure as PNG bytes.

    Requires kaleido. Raises RuntimeError if not available.
    """
    try:
        return fig.to_image(format="png", scale=2)
    except Exception as exc:
        raise RuntimeError(
            "Export PNG indisponible — installez kaleido : pip install kaleido"
        ) from exc


def export_session_yaml(session: EISSession) -> str:
    """Return a YAML summary of the session (metadata + fitted parameters)."""
    data: dict = {
        "created_at": str(session.created_at),
        "groups": [],
    }

    for grp in session.groups:
        grp_data: dict = {
            "concentration": grp.concentration,
            "n_points": grp.spectrum.n_points,
            "fits": {},
        }
        for model_name, fit in grp.fit_results.items():
            grp_data["fits"][model_name] = {
                "Rct": float(fit.Rct),
                "Rct_std": float(fit.Rct_std),
                "chi2": float(fit.chi2),
                "converged": bool(fit.converged),
                "params": {
                    k: (v.tolist() if hasattr(v, "tolist") else
                        [float(x) for x in v] if isinstance(v, (list, tuple)) else
                        v if isinstance(v, str) else
                        float(v))
                    for k, v in fit.params.items()
                    if not k.startswith("_lc_") and not k.startswith("_gcv_")
                },
            }
        data["groups"].append(grp_data)

    return yaml.dump(data, allow_unicode=True, sort_keys=False)


# ── Calibration EIS — export multi-électrode / multi-méthode (ajout) ─────────

def export_calibration_csv(sessions: dict) -> bytes:
    """Exporte, pour chaque électrode et méthode, conc / log10(conc) / signal
    normalisé / Rct / paramètres de régression (slope, intercept, r2, p_value, std_err)
    dans un CSV multi-colonnes unique.

    sessions: {electrode_index: EISSession} ou un EISSession unique.
    """
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "electrode", "model", "concentration_M", "log10_concentration",
        "signal_norm", "Rct_Ohm", "Rct_probe_Ohm",
        "slope", "intercept", "r2", "p_value", "std_err",
    ])

    for e, session in sorted(sessions.items()):
        probe_fr = getattr(session.probe, "fit_results", {}) if session.probe else {}
        if not probe_fr:
            continue
        for model, probe_fit in probe_fr.items():
            if probe_fit is None or probe_fit.Rct <= 0:
                continue
            probe_rct = probe_fit.Rct

            concs, signals, rcts = [], [], []
            for grp in session.groups:
                if grp.concentration <= 0:
                    continue
                fr = grp.fit_results.get(model)
                if fr is None or fr.Rct <= 0:
                    continue
                concs.append(grp.concentration)
                rcts.append(fr.Rct)
                signals.append(abs(probe_rct - fr.Rct) / abs(probe_rct))

            if len(concs) < 2:
                continue

            log_c = [math.log10(c) for c in concs]
            reg = stats.linregress(log_c, signals)

            for conc, lc, sig, rct in zip(concs, log_c, signals, rcts):
                writer.writerow([
                    e, model, conc, lc, sig, rct, probe_rct,
                    reg.slope, reg.intercept, reg.rvalue ** 2, reg.pvalue, reg.stderr,
                ])

    return buf.getvalue().encode()


# ── Calibration CV — export (ajout, même pattern que EIS) ─────────────────────

def export_cv_calibration_csv(cv_session: CVSession) -> bytes:
    """Exporte concentration / log10(conc) / signal normalisé moyen + régression
    OLS (slope, intercept, r2, p_value, std_err) pour la calibration CV.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "concentration_M", "log10_concentration", "signal_norm",
        "slope", "intercept", "r2", "p_value", "std_err",
    ])

    concs, signals = [], []
    for grp in cv_session.groups:
        if grp.concentration <= 0:
            continue
        mean_sig = np.nanmean(grp.delta_signal)
        if np.isfinite(mean_sig):
            concs.append(grp.concentration)
            signals.append(float(mean_sig))

    if len(concs) < 2:
        return buf.getvalue().encode()

    log_c = list(np.log10(concs))
    reg = stats.linregress(log_c, signals)

    for conc, lc, sig in zip(concs, log_c, signals):
        writer.writerow([
            conc, lc, sig,
            reg.slope, reg.intercept, reg.rvalue ** 2, reg.pvalue, reg.stderr,
        ])

    return buf.getvalue().encode()


def export_cv_calibration_csv_from_result(cv_result: dict, cv_ols: dict | None = None) -> bytes:
    """Variante de export_cv_calibration_csv pour la structure dict réellement
    produite par pages/B_cv.py (st.session_state['cv_result'] / ['cv_ols']),
    plutôt que pour le dataclass CVSession (non utilisé par la page live)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "concentration_M", "log10_concentration", "delta_I_norm_mean",
        "slope", "intercept", "r2", "p_value", "std_err",
    ])

    reg = cv_ols or {}
    for g in (cv_result or {}).get("groups", []):
        conc = g.get("concentration")
        sig = g.get("delta_I_norm_mean")
        if conc is None or conc <= 0 or sig is None or not np.isfinite(sig):
            continue
        writer.writerow([
            conc, math.log10(conc), sig,
            reg.get("slope"), reg.get("intercept"), reg.get("r2"),
            reg.get("p_value"), reg.get("stderr"),
        ])

    return buf.getvalue().encode()


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
        concs, signals = [], []
        for grp in session.groups:
            if grp.concentration <= 0:
                continue
            mean_sig = np.nanmean(grp.delta_signal)
            if np.isfinite(mean_sig):
                concs.append(grp.concentration)
                signals.append(float(mean_sig))

        if len(concs) < 2:
            continue

        log_c = list(np.log10(concs))
        reg = stats.linregress(log_c, signals)

        for conc, lc, sig in zip(concs, log_c, signals):
            writer.writerow([
                elec, conc, lc, sig,
                reg.slope, reg.intercept, reg.rvalue ** 2, reg.pvalue, reg.stderr,
            ])

    return buf.getvalue().encode()


# ── DRT — export multi-électrode, multi-spectre, multi-réplicat ──────────────

def export_drt_csv(sessions: dict) -> bytes:
    """Exporte les valeurs DRT (ln_tau, ln_gamma) du modèle 'drt_fft', pour
    chaque électrode, chaque spectre (bare/probe/groupes) et chaque réplicat
    retenu."""
    sessions = _as_sessions_dict(sessions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["electrode", "label", "concentration", "replicate_idx", "ln_tau", "ln_gamma"])

    for e, session in sorted(sessions.items()):
        spectra_by_label = []
        if session.bare is not None:
            spectra_by_label.append(("bare", 0.0, session.bare_replicate_spectra))
        if session.probe is not None:
            spectra_by_label.append(("probe", 0.0, session.probe_replicate_spectra))
        for grp in session.groups:
            spectra_by_label.append((grp.spectrum.label, grp.concentration, grp.replicate_spectra))

        for label, conc, reps in spectra_by_label:
            for ri, sp in enumerate(reps or []):
                fr = sp.fit_results.get("drt_fft")
                if fr is None:
                    continue
                tau = getattr(fr, "drt_tau", None)
                gamma = getattr(fr, "drt_gamma", None)
                if tau is None or gamma is None:
                    continue
                ln_tau = np.log(np.asarray(tau) + 1e-300)
                ln_gamma = np.log(np.asarray(gamma) + 1e-300)
                for lt, lg in zip(ln_tau, ln_gamma):
                    writer.writerow([e, label, conc, ri, lt, lg])

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


# ── Reconstructions Randles/DRT vs mesure ─────────────────────────────────────

def export_reconstruction_csv(sessions: dict) -> bytes:
    """Exporte, pour chaque électrode/spectre/méthode (randles_full, drt_fft),
    les valeurs mesurées et reconstruites et l'erreur de reconstruction."""
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
            for model in ("randles_full", "drt_fft"):
                fr = sp.fit_results.get(model)
                if fr is None:
                    continue
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
    cv_session=None,
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

        # fit_randles/
        if sessions:
            import copy
            for e, session in sorted(sessions.items()):
                filtered = copy.deepcopy(session)
                for grp in filtered.groups:
                    grp.fit_results = {k: v for k, v in grp.fit_results.items() if k == "randles_full"}
                if any(grp.fit_results for grp in filtered.groups):
                    zf.writestr(
                        f"export/fit_randles/parametres_electrode_{e}.csv",
                        export_params_csv({e: filtered}),
                    )

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
            if isinstance(cv_session, dict) and "groups" in cv_session:
                cal_cv = export_cv_calibration_csv_from_result(cv_session)
            elif isinstance(cv_session, dict):
                cal_cv = export_cv_calibration_csv_multi(cv_session)
            else:
                cal_cv = export_cv_calibration_csv(cv_session)
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
