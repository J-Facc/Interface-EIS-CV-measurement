"""Résultats d'analyse SÉRIALISABLES (v3) — aucun import Streamlit.

L'analyse (``core/analysis.py``) convertit, dans le processus de calcul, les objets riches du
pipeline (``GroupAnalysis``, ``FitResult``, measurement model…) en enregistrements plats :
tableaux numpy, nombres, textes. Ce sont eux — et eux seuls — que lisent la page Analyse,
l'export et l'archive : une session fraîche et une archive rechargée s'affichent donc par le
MÊME chemin, sans recalcul (``exports/archive.py``).

Clés
----
* groupe : ``C1_E2_probe`` (canal, électrode, étape) ;
* spectre : identifiant de la mesure pour un réplicat (``EIS_C1_E2_probe_r1``), ou
  ``EIS_C1_E2_probe_mean`` pour le spectre moyen du groupe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from core.drt_peak import MainPeak
from core.version import APP_VERSION

#: Pondération du fit : structure d'erreur d'Orazem (≥ 3 réplicats) ou repli proportionnel.
WEIGHTING_ORAZEM = "orazem"
WEIGHTING_PROPORTIONAL = "proportional"
WEIGHTING_LABELS = {
    WEIGHTING_ORAZEM: "structure d'erreur d'Orazem (measurement model)",
    WEIGHTING_PROPORTIONAL: "proportionnelle σ = 1 %·|Z| (repli, < 3 réplicats ou structure d'erreur "
                            "non caractérisable)",
}
#: Fraction de |Z| utilisée comme σ dans le repli proportionnel.
PROPORTIONAL_SIGMA = 0.01

GROUP_OK = "ok"
GROUP_FALLBACK = "fallback"
GROUP_FAILED = "failed"

KK_CONFORM = "conforme"
KK_NONCONFORM = "non conforme"
KK_UNDETERMINED = "indéterminé"


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _jsonable(x):
    """Copie JSON-compatible (NaN/inf → None, numpy → Python) d'une structure libre."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def _nan(x) -> float:
    """None (JSON) → NaN."""
    return float("nan") if x is None else float(x)


def group_key(channel_id: int, electrode: int, step: str) -> str:
    return f"C{int(channel_id)}_E{int(electrode)}_{step}"


def mean_key(channel_id: int, electrode: int, step: str) -> str:
    return f"EIS_C{int(channel_id)}_E{int(electrode)}_{step}_mean"


# ─────────────────────────────────────────────────────────────────────────────
# Fit du circuit et DRT d'un spectre
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FitRecord:
    """Fit Orazem (circuit utilisateur) d'UN spectre."""

    target_param: str
    params: dict
    params_std: dict
    chi2_reduced: float
    converged: bool
    weighting: str
    Zfit_re: np.ndarray
    Zfit_im: np.ndarray
    chi2_ci: Optional[list] = None
    warnings: list = field(default_factory=list)

    @property
    def target_value(self) -> float:
        return _f(self.params.get(self.target_param))

    @classmethod
    def from_fit_result(cls, fr, weighting: str) -> "FitRecord":
        ci = getattr(fr, "chi2_reduced_ci", None)
        return cls(target_param=fr.target_param,
                   params={k: _f(v) for k, v in (fr.params or {}).items()},
                   params_std={k: _f(v) for k, v in (fr.params_std or {}).items()},
                   chi2_reduced=_f(fr.chi2_reduced), converged=bool(fr.converged), weighting=weighting,
                   Zfit_re=np.asarray(fr.Zfit_re, dtype=float).copy(),
                   Zfit_im=np.asarray(fr.Zfit_im, dtype=float).copy(),
                   chi2_ci=[_f(ci[0]), _f(ci[1])] if ci is not None and weighting == WEIGHTING_ORAZEM else None,
                   warnings=[str(w) for w in fr.warnings or []])

    def meta(self) -> dict:
        return _jsonable({"target_param": self.target_param, "params": self.params,
                          "params_std": self.params_std, "chi2_reduced": self.chi2_reduced,
                          "chi2_ci": self.chi2_ci, "converged": self.converged,
                          "weighting": self.weighting, "warnings": self.warnings})

    @classmethod
    def from_meta(cls, d: dict, Zfit_re, Zfit_im) -> "FitRecord":
        return cls(target_param=d["target_param"], params={k: _nan(v) for k, v in d["params"].items()},
                   params_std={k: _nan(v) for k, v in d["params_std"].items()},
                   chi2_reduced=_nan(d.get("chi2_reduced")), converged=bool(d["converged"]),
                   weighting=d["weighting"], Zfit_re=np.asarray(Zfit_re, dtype=float),
                   Zfit_im=np.asarray(Zfit_im, dtype=float),
                   chi2_ci=[_nan(v) for v in d["chi2_ci"]] if d.get("chi2_ci") else None,
                   warnings=list(d.get("warnings") or []))


def _drt_summary(fr) -> dict:
    """Diagnostics DRT utiles à l'affichage, JSON-compatibles (sans les tirages HMC)."""
    diag = getattr(fr, "drt_diagnostics", None) or {}
    sampler = diag.get("sampler") or {}
    hmc = getattr(fr, "drt_mode", None) == "sample"
    return _jsonable({
        "rhat_max": sampler.get("rhat_max") if hmc else None,
        "divergences": sampler.get("divergences") if hmc else None,
        "ess_bulk_min": sampler.get("ess_bulk_min") if hmc else None,
        "ess_tail_min": sampler.get("ess_tail_min") if hmc else None,
        "chains": sampler.get("chains") if hmc else None,
        "Rct_ci95": diag.get("Rct_ci95"), "Rp_ci95": diag.get("Rp_ci95"),
        "alerts": [a.get("code") for a in diag.get("alerts", []) if isinstance(a, dict)],
        "cmdstan_version": diag.get("cmdstan_version"),
        "duration_s": diag.get("duration_s"), "computed_at": diag.get("computed_at"),
        "reconstruction_error_rms": getattr(fr, "reconstruction_error", None),
    })


@dataclass
class DRTRecord:
    """DRT d'UN spectre : γ(τ), pic principal (Rct_DRT) et Rct du moteur (contrôle)."""

    mode: str
    tau: np.ndarray
    gamma: np.ndarray
    peak: MainPeak
    rct_engine: float           # pic pénultième ±3 ln τ (drt.engine) — colonne de contrôle
    rct_engine_std: float
    rct_engine_source: str
    rp: float
    converged: bool
    Zfit_re: np.ndarray
    Zfit_im: np.ndarray
    gamma_lo: Optional[np.ndarray] = None
    gamma_hi: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def rct_main(self) -> float:
        """Rct par DRT retenu (aire du pic principal) ; NaN si non calculable."""
        return float(self.peak.rct) if self.peak is not None and self.peak.ok else float("nan")

    @classmethod
    def from_fit_result(cls, fr, peak: MainPeak) -> "DRTRecord":
        p = fr.params or {}
        arr = (lambda a: None if a is None else np.asarray(a, dtype=float).copy())
        return cls(mode=str(fr.drt_mode or (fr.drt_diagnostics or {}).get("mode") or ""),
                   tau=arr(fr.drt_tau), gamma=arr(fr.drt_gamma), peak=peak,
                   rct_engine=_f(fr.target_value), rct_engine_std=_f(fr.target_std),
                   rct_engine_source=str(p.get("rct_source") or ""), rp=_f(p.get("Rp")),
                   converged=bool(fr.converged), Zfit_re=arr(fr.Zfit_re), Zfit_im=arr(fr.Zfit_im),
                   gamma_lo=arr(fr.drt_gamma_lo), gamma_hi=arr(fr.drt_gamma_hi),
                   diagnostics=_drt_summary(fr), warnings=[str(w) for w in fr.warnings or []])

    def meta(self) -> dict:
        return _jsonable({"mode": self.mode, "peak": self.peak.to_dict() if self.peak else None,
                          "rct_engine": self.rct_engine, "rct_engine_std": self.rct_engine_std,
                          "rct_engine_source": self.rct_engine_source, "rp": self.rp,
                          "converged": self.converged, "diagnostics": self.diagnostics,
                          "warnings": self.warnings, "has_ci": self.gamma_lo is not None})


    @classmethod
    def from_meta(cls, d: dict, tau, gamma, gamma_lo, gamma_hi, Zfit_re, Zfit_im) -> "DRTRecord":
        return cls(mode=d["mode"], tau=np.asarray(tau, dtype=float), gamma=np.asarray(gamma, dtype=float),
                   peak=MainPeak.from_dict(d.get("peak")) or MainPeak(ok=False),
                   rct_engine=_nan(d.get("rct_engine")), rct_engine_std=_nan(d.get("rct_engine_std")),
                   rct_engine_source=d.get("rct_engine_source") or "", rp=_nan(d.get("rp")),
                   converged=bool(d.get("converged")),
                   Zfit_re=None if Zfit_re is None else np.asarray(Zfit_re, dtype=float),
                   Zfit_im=None if Zfit_im is None else np.asarray(Zfit_im, dtype=float),
                   gamma_lo=None if gamma_lo is None else np.asarray(gamma_lo, dtype=float),
                   gamma_hi=None if gamma_hi is None else np.asarray(gamma_hi, dtype=float),
                   diagnostics=dict(d.get("diagnostics") or {}), warnings=list(d.get("warnings") or []))


# ─────────────────────────────────────────────────────────────────────────────
# Spectre, groupe, résultats EIS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpectrumRecord:
    """Un spectre analysé (réplicat ou moyenne) : données, verdict KK, fit, DRT."""

    key: str
    group_key: str
    channel_id: int
    electrode: int
    step: str
    replicate: Optional[int]          # None = spectre moyen
    concentration: Optional[float]
    f: np.ndarray
    Zre: np.ndarray
    Zim: np.ndarray
    kk_conform: Optional[bool] = None
    kk_message: str = ""
    fit: Optional[FitRecord] = None
    fit_error: Optional[str] = None
    drt: Optional[DRTRecord] = None
    drt_error: Optional[str] = None

    @property
    def is_mean(self) -> bool:
        return self.replicate is None

    @property
    def label(self) -> str:
        return "moyenne" if self.is_mean else f"r{self.replicate}"

    def meta(self) -> dict:
        return _jsonable({"key": self.key, "group_key": self.group_key, "channel_id": self.channel_id,
                          "electrode": self.electrode, "step": self.step, "replicate": self.replicate,
                          "concentration": self.concentration, "kk_conform": self.kk_conform,
                          "kk_message": self.kk_message, "fit_error": self.fit_error,
                          "drt_error": self.drt_error,
                          "fit": self.fit.meta() if self.fit else None,
                          "drt": self.drt.meta() if self.drt else None})


@dataclass
class GroupRecord:
    """Un groupe de réplicats (canal, électrode, étape) : statut, pondération, verdict KK."""

    key: str
    channel_id: int
    electrode: int
    step: str
    concentration: Optional[float]
    biomarker: str
    n_replicates: int
    status: str = GROUP_OK
    weighting: Optional[str] = None
    message: str = ""
    kk_state: str = KK_UNDETERMINED
    kk_message: str = ""
    warnings: list = field(default_factory=list)
    replicate_keys: list = field(default_factory=list)
    mean_key: Optional[str] = None
    aggregate: dict = field(default_factory=dict)     # {param: {"mean","std_between","sem","n"}}

    def meta(self) -> dict:
        return _jsonable(self.__dict__)

    @classmethod
    def from_meta(cls, d: dict) -> "GroupRecord":
        d = dict(d)
        d["aggregate"] = {p: {k: _nan(v) if k != "n" else int(v or 0) for k, v in a.items()}
                          for p, a in (d.get("aggregate") or {}).items()}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Failure:
    """Un calcul en échec (n'arrête jamais les autres)."""

    key: str
    stage: str           # "fit" | "drt" | "cv" | "tâche"
    message: str

    def meta(self) -> dict:
        return dict(key=self.key, stage=self.stage, message=self.message)


@dataclass
class EISResults:
    groups: dict = field(default_factory=dict)       # {group_key: GroupRecord}
    spectra: dict = field(default_factory=dict)      # {spectrum_key: SpectrumRecord}
    circuit: dict = field(default_factory=dict)
    drt_mode: Optional[str] = None
    drt_messages: list = field(default_factory=list)

    def group_spectra(self, gkey: str, include_mean: bool = True) -> list:
        g = self.groups[gkey]
        keys = list(g.replicate_keys) + ([g.mean_key] if include_mean and g.mean_key else [])
        return [self.spectra[k] for k in keys if k in self.spectra]

    def electrodes(self) -> list:
        return sorted({(g.channel_id, g.electrode) for g in self.groups.values()})


# ─────────────────────────────────────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CVFeature:
    """Courant de pic anodique d'UNE courbe CV et son signal normalisé ΔI/I."""

    key: str                  # identifiant de la mesure
    channel_id: int
    electrode: int
    step: str
    replicate: int
    concentration: Optional[float]
    i_peak: float             # A
    e_peak: float             # V
    i_probe: float = float("nan")      # moyenne des pics « probe » de la même électrode (A)
    delta_i_norm: float = float("nan")  # (I_probe − I) / I_probe
    ok: bool = True
    message: str = ""

    def meta(self) -> dict:
        return _jsonable(self.__dict__)

    @classmethod
    def from_meta(cls, d: dict) -> "CVFeature":
        out = dict(d)
        for k in ("i_peak", "e_peak", "i_probe", "delta_i_norm"):
            out[k] = _nan(out.get(k))
        return cls(**{k: out[k] for k in cls.__dataclass_fields__ if k in out})


# ─────────────────────────────────────────────────────────────────────────────
# Résultats complets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResults:
    eis: EISResults = field(default_factory=EISResults)
    cv: dict = field(default_factory=dict)            # {measurement_id: CVFeature}
    failures: list = field(default_factory=list)      # [Failure]
    config: dict = field(default_factory=dict)        # config utilisée (snapshot)
    app_version: str = APP_VERSION
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def failure_rows(self) -> list:
        return [f.meta() for f in self.failures]
