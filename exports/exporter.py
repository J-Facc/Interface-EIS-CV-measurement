"""Export helpers for EIS Analyzer sessions."""

from __future__ import annotations

import io
import csv
import yaml

from core.models import EISSession


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
                "params": {k: float(v) for k, v in fit.params.items()},
            }
        data["groups"].append(grp_data)

    return yaml.dump(data, allow_unicode=True, sort_keys=False)
