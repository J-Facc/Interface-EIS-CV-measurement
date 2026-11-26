"""Archive .zip rechargeable (schéma v3) — écriture et lecture, aucun import Streamlit.

Contenu (lisible hors de l'application : JSON + CSV) ::

    experience_<date>.zip
    ├── manifest.json            schema_version, app_version, expérience, table d'assignation,
    │                            config utilisée (imbriquée ET aplatie), fichiers + SHA-256
    ├── data/eis/<C>_<E>_<étape>_r<k>.csv    données EIS nettoyées (f, Zre, Zim)
    ├── data/cv/<C>_<E>_<étape>_r<k>.csv     données CV nettoyées (E, I, cycle)
    └── results/                 (si l'analyse a été faite)
        ├── analysis.json        groupes, statuts, verdicts KK, diagnostics, échecs, circuit
        ├── spectra/<clé>.csv    spectre analysé (réplicat ou moyenne) + Zfit du circuit et de la DRT
        ├── drt/<clé>.csv        tau, gamma (+ IC 95 % en mode sample, zone du pic principal)
        ├── drt_peaks.csv        pic principal et aire (Rct_DRT), Rct du moteur (contrôle)
        ├── fits_randles.csv     tous les paramètres du fit + χ²ᵣ + Rct_DRT
        ├── cv_features.csv      courants de pic, I_probe, ΔI/I
        ├── calibration_eis.csv  points moyens, écarts-types, n, régressions (sources DRT et fit)
        └── calibration_cv.csv

Les nombres sont écrits avec ``repr`` (17 chiffres significatifs) : l'aller-retour est EXACT.
:func:`read_archive` vérifie la version de schéma et chaque SHA-256, puis reconstruit
l'expérience, les mesures et TOUS les résultats sans aucun recalcul. Une archive invalide lève
``ArchiveError`` (message destiné à l'utilisateur), jamais une autre exception.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime
from typing import Optional

import numpy as np

from core.calibration import (
    SOURCE_DRT,
    SOURCE_FIT,
    VIEW_BIOMARKER,
    VIEW_ELECTRODE,
    cv_calibration,
    eis_calibration,
)
from core.experiment import KIND_EIS, Experiment, Measurement, sort_measurements
from core.report_tables import (
    calibration_level_rows,
    calibration_regression_rows,
    cv_rows,
    rct_rows,
)
from core.results import (
    AnalysisResults,
    CVFeature,
    DRTRecord,
    EISResults,
    Failure,
    FitRecord,
    GroupRecord,
    SpectrumRecord,
    _jsonable,
)
from core.version import APP_VERSION, ARCHIVE_SCHEMA_VERSION

MANIFEST = "manifest.json"
ANALYSIS_JSON = "results/analysis.json"


class ArchiveError(ValueError):
    """Archive illisible, corrompue ou d'une version de schéma inconnue."""


# ─────────────────────────────────────────────────────────────────────────────
# CSV exacts
# ─────────────────────────────────────────────────────────────────────────────

def _num(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


def _columns_csv(columns: dict) -> bytes:
    """Colonnes de même longueur (None = colonne absente) → CSV, flottants exacts."""
    cols = {k: np.asarray(v) for k, v in columns.items() if v is not None}
    n = max((len(v) for v in cols.values()), default=0)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(cols))
    for i in range(n):
        w.writerow([_num(v[i]) if i < len(v) else "" for v in cols.values()])
    return buf.getvalue().encode("utf-8")


def _rows_csv(rows: list) -> bytes:
    buf = io.StringIO()
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _num(r.get(k)) for k in keys})
    return buf.getvalue().encode("utf-8")


def _read_columns(data: bytes) -> dict:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    if not rows:
        return {}
    head, body = rows[0], rows[1:]
    out = {}
    for j, name in enumerate(head):
        vals = [r[j] for r in body if j < len(r) and r[j] != ""]
        out[name] = np.array([float(v) for v in vals], dtype=float)
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mfile(m: Measurement) -> str:
    return f"data/{m.kind.lower()}/C{m.channel_id}_E{m.electrode}_{m.step}_r{m.replicate}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Écriture
# ─────────────────────────────────────────────────────────────────────────────

def _calibration_tables(results: AnalysisResults, experiment: Experiment) -> tuple:
    eis_rows, cv_rows_ = [], []
    for src in (SOURCE_DRT, SOURCE_FIT):
        for view in (VIEW_BIOMARKER, VIEW_ELECTRODE):
            curves = eis_calibration(results, experiment, src, view)
            reg = {r["Courbe"]: r for r in calibration_regression_rows(curves)}
            for r in calibration_level_rows(curves):
                g = reg[r["Courbe"]]
                eis_rows.append({"Vue": view, **r, "Pente a": g["Pente a"], "Ordonnée b": g["Ordonnée b"],
                                 "R²": g["R²"], "p-value": g["p-value"]})
    for view in (VIEW_BIOMARKER, VIEW_ELECTRODE):
        curves = cv_calibration(results, experiment, view)
        reg = {r["Courbe"]: r for r in calibration_regression_rows(curves)}
        for r in calibration_level_rows(curves):
            g = reg[r["Courbe"]]
            cv_rows_.append({"Vue": view, **r, "Pente a": g["Pente a"], "Ordonnée b": g["Ordonnée b"],
                             "R²": g["R²"], "p-value": g["p-value"]})
    return eis_rows, cv_rows_


def write_archive(experiment: Experiment, measurements: list, results: Optional[AnalysisResults] = None,
                  config: Optional[dict] = None) -> bytes:
    """Archive .zip complète (données nettoyées + résultats s'il y en a)."""
    files: dict = {}
    for m in sort_measurements(measurements or []):
        cols = ({"f": m.data["f"], "Zre": m.data["Zre"], "Zim": m.data["Zim"]} if m.kind == KIND_EIS
                else {"E": m.data["E"], "I": m.data["I"], "cycle": m.data["cycle"]})
        files[_mfile(m)] = _columns_csv(cols)
    if results is not None:
        eis = results.eis
        for key, sp in eis.spectra.items():
            files[f"results/spectra/{key}.csv"] = _columns_csv({
                "f": sp.f, "Zre": sp.Zre, "Zim": sp.Zim,
                "Zfit_re": sp.fit.Zfit_re if sp.fit else None, "Zfit_im": sp.fit.Zfit_im if sp.fit else None,
                "Zdrt_re": sp.drt.Zfit_re if sp.drt is not None and sp.drt.Zfit_re is not None else None,
                "Zdrt_im": sp.drt.Zfit_im if sp.drt is not None and sp.drt.Zfit_im is not None else None})
            if sp.drt is not None:
                d = sp.drt
                zone = None
                if d.peak.ok:
                    zone = ((d.tau >= d.peak.tau_lo) & (d.tau <= d.peak.tau_hi)).astype(int)
                files[f"results/drt/{key}.csv"] = _columns_csv({
                    "tau": d.tau, "gamma": d.gamma, "gamma_lo": d.gamma_lo, "gamma_hi": d.gamma_hi,
                    "in_main_peak": zone})
        files["results/drt_peaks.csv"] = _rows_csv([
            {"spectre": k, "mode": s.drt.mode, "pic_trouve": s.drt.peak.ok, "tau_pic_s": s.drt.peak.tau_peak,
             "tau_min_integration_s": s.drt.peak.tau_lo, "tau_max_integration_s": s.drt.peak.tau_hi,
             "Rct_DRT_aire_pic_principal_ohm": s.drt.rct_main, "Rct_moteur_penultieme_ohm": s.drt.rct_engine,
             "Rp_ohm": s.drt.rp, "remarque": s.drt.peak.message}
            for k, s in sorted(eis.spectra.items()) if s.drt is not None])
        files["results/fits_randles.csv"] = _rows_csv(rct_rows(results))
        files["results/cv_features.csv"] = _rows_csv(cv_rows(results))
        cal_eis, cal_cv = _calibration_tables(results, experiment)
        files["results/calibration_eis.csv"] = _rows_csv(cal_eis)
        files["results/calibration_cv.csv"] = _rows_csv(cal_cv)
        analysis = {
            "computed_at": results.computed_at, "app_version": results.app_version, "config": results.config,
            "circuit": eis.circuit, "drt_mode": eis.drt_mode, "drt_messages": eis.drt_messages,
            "groups": [g.meta() for g in eis.groups.values()],
            "spectra": [s.meta() for s in eis.spectra.values()],
            "cv": [f.meta() for f in results.cv.values()],
            "failures": [f.meta() for f in results.failures],
        }
        files[ANALYSIS_JSON] = json.dumps(_jsonable(analysis), ensure_ascii=False, indent=1).encode("utf-8")

    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION, "app_version": APP_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment.to_dict(),
        "assignment": [dict(m.meta_dict(), file=_mfile(m)) for m in sort_measurements(measurements or [])],
        "config": _jsonable(config or (results.config if results else {})),
        "config_flat": _jsonable(_flatten(config or (results.config if results else {}))),
        "has_results": results is not None,
        "files": [{"path": p, "sha256": _sha(b)} for p, b in sorted(files.items())],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1))
        for path, data in sorted(files.items()):
            zf.writestr(path, data)
    return buf.getvalue()


def archive_filename(experiment: Experiment) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (experiment.name or "experience"))
    return f"{stem or 'experience'}_{datetime.now():%Y-%m-%d}.zip"


# ─────────────────────────────────────────────────────────────────────────────
# Lecture
# ─────────────────────────────────────────────────────────────────────────────

def is_v3_archive(zip_bytes: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            return MANIFEST in zf.namelist()
    except zipfile.BadZipFile:
        return False


def _spectrum_from(meta: dict, cols: dict, drt_cols: Optional[dict]) -> SpectrumRecord:
    fit = None
    if meta.get("fit"):
        fit = FitRecord.from_meta(meta["fit"], cols["Zfit_re"], cols["Zfit_im"])
    drt = None
    if meta.get("drt"):
        if drt_cols is None:
            raise ArchiveError(f"DRT de « {meta['key']} » annoncée mais absente de l'archive.")
        drt = DRTRecord.from_meta(meta["drt"], drt_cols["tau"], drt_cols["gamma"], drt_cols.get("gamma_lo"),
                                  drt_cols.get("gamma_hi"), cols.get("Zdrt_re"), cols.get("Zdrt_im"))
    conc = meta.get("concentration")
    return SpectrumRecord(key=meta["key"], group_key=meta["group_key"], channel_id=int(meta["channel_id"]),
                          electrode=int(meta["electrode"]), step=meta["step"], replicate=meta.get("replicate"),
                          concentration=None if conc is None else float(conc), f=cols["f"], Zre=cols["Zre"],
                          Zim=cols["Zim"], kk_conform=meta.get("kk_conform"), kk_message=meta.get("kk_message") or "",
                          fit=fit, fit_error=meta.get("fit_error"), drt=drt, drt_error=meta.get("drt_error"))


def read_archive(zip_bytes: bytes) -> tuple:
    """(Experiment, [Measurement], AnalysisResults ou None) — aucun recalcul.

    Raises:
        ArchiveError: ZIP illisible, manifest absent, version de schéma inconnue, fichier
            manquant ou altéré (SHA-256), contenu incohérent.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except (zipfile.BadZipFile, ValueError) as exc:
        raise ArchiveError(f"Fichier non lisible comme archive .zip ({exc}).") from exc
    with zf:
        names = set(zf.namelist())
        if MANIFEST not in names:
            raise ArchiveError("Archive sans manifest.json : ce n'est pas une archive EIS Analyzer v3.")
        try:
            manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ArchiveError(f"manifest.json illisible ({exc}).") from exc
        version = manifest.get("schema_version")
        if version != ARCHIVE_SCHEMA_VERSION:
            raise ArchiveError(f"Version de schéma d'archive inconnue : {version!r} (cette application lit la "
                               f"version {ARCHIVE_SCHEMA_VERSION}).")
        blobs = {}
        for entry in manifest.get("files") or []:
            path = entry.get("path")
            if path not in names:
                raise ArchiveError(f"Fichier manquant dans l'archive : {path}.")
            data = zf.read(path)
            if _sha(data) != entry.get("sha256"):
                raise ArchiveError(f"Somme de contrôle invalide pour {path} : l'archive a été modifiée ou "
                                   "est corrompue.")
            blobs[path] = data
    try:
        return _rebuild(manifest, blobs)
    except ArchiveError:
        raise
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise ArchiveError(f"Contenu de l'archive incohérent ({type(exc).__name__}: {exc}).") from exc


def _rebuild(manifest: dict, blobs: dict) -> tuple:
    experiment = Experiment.from_dict(manifest["experiment"])
    measurements = []
    for a in manifest.get("assignment") or []:
        cols = _read_columns(blobs[a["file"]])
        data = ({"f": cols["f"], "Zre": cols["Zre"], "Zim": cols["Zim"]} if a["kind"] == KIND_EIS
                else {"E": cols["E"], "I": cols["I"], "cycle": cols["cycle"]})
        measurements.append(Measurement(channel_id=int(a["channel_id"]), electrode=int(a["electrode"]),
                                        step=a["step"], replicate=int(a["replicate"]), kind=a["kind"],
                                        source_filename=a["source_filename"], data=data,
                                        warnings=list(a.get("warnings") or [])))
    results = None
    if manifest.get("has_results"):
        if ANALYSIS_JSON not in blobs:
            raise ArchiveError("Résultats annoncés mais results/analysis.json est absent.")
        an = json.loads(blobs[ANALYSIS_JSON].decode("utf-8"))
        eis = EISResults(circuit=an.get("circuit") or {}, drt_mode=an.get("drt_mode"),
                         drt_messages=list(an.get("drt_messages") or []))
        for g in an.get("groups") or []:
            eis.groups[g["key"]] = GroupRecord.from_meta(g)
        for s in an.get("spectra") or []:
            cols = _read_columns(blobs[f"results/spectra/{s['key']}.csv"])
            drt_path = f"results/drt/{s['key']}.csv"
            drt_cols = _read_columns(blobs[drt_path]) if drt_path in blobs else None
            eis.spectra[s["key"]] = _spectrum_from(s, cols, drt_cols)
        results = AnalysisResults(
            eis=eis, cv={f["key"]: CVFeature.from_meta(f) for f in an.get("cv") or []},
            failures=[Failure(**f) for f in an.get("failures") or []], config=an.get("config") or {},
            app_version=an.get("app_version") or "", computed_at=an.get("computed_at") or "")
    return experiment, sort_measurements(measurements), results
