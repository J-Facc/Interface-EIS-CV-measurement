"""Export helpers for EIS Analyzer sessions."""

from __future__ import annotations

import io
import csv
import math
import yaml
import numpy as np
from scipy import stats

from core.models import EISSession
from core.cv_models import CVSession


def export_params_csv(session: EISSession) -> bytes:
    """Return fitted parameters for all groups as CSV bytes."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    header_written = False
    for grp in session.groups:
        for model_name, fit in grp.fit_results.items():
            row_base = {
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


def export_spectra_csv(session: EISSession) -> bytes:
    """Return raw spectra (f, Zre, Zim) for all groups as CSV bytes."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["label", "concentration", "f_Hz", "Zre_Ohm", "Zim_Ohm"])

    spectra = []
    if session.bare is not None:
        spectra.append(("bare", 0.0, session.bare))
    if session.probe is not None:
        spectra.append(("probe", 0.0, session.probe))
    for grp in session.groups:
        spectra.append((grp.spectrum.label, grp.concentration, grp.spectrum))

    for label, conc, sp in spectra:
        for f, zre, zim in zip(sp.f, sp.Zre, sp.Zim):
            writer.writerow([label, conc, f, zre, zim])

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

    sessions: {electrode_index: EISSession}.
    """
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
