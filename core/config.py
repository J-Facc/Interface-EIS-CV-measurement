"""Load YAML configuration and validate with Pydantic."""

from typing import Optional

import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class PhysicsSettings(BaseModel):
    T: float = 298.0
    F: float = 96485.0
    R: float = 8.314
    n: int = 1
    C0: float = 0.02
    D_FeIII: float = 7.2e-10
    D_FeII: float = 6.5e-10


class GeometrySettings(BaseModel):
    xe: float = 30e-6
    h: float = 60e-6
    d: float = 300e-6
    S_WE: float = 9e-9


class ConditionsSettings(BaseModel):
    Fv: float = 5e-10


class CircuitParameterSettings(BaseModel):
    """Guess et bornes d'UN paramètre du circuit (``None`` = non borné de ce côté)."""

    initial: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    scale: Optional[float] = None


_DEFAULT_CIRCUIT = (
    "Re + parallel(Re_prime + parallel(R(Rct) + ZD_bounded(R_D, tau_d), Q(Qdl, alpha)), C(Cb))"
)


def _default_circuit_parameters() -> dict:
    return {
        "Re": CircuitParameterSettings(initial=100.0, lower=0.0, upper=1e5),
        "Re_prime": CircuitParameterSettings(initial=50.0, lower=0.0, upper=1e5),
        "Rct": CircuitParameterSettings(initial=1500.0, lower=0.0, upper=1e9),
        "R_D": CircuitParameterSettings(initial=300.0, lower=0.0, upper=1e7),
        "tau_d": CircuitParameterSettings(initial=0.2, lower=1e-6, upper=1e4),
        "Qdl": CircuitParameterSettings(initial=5e-6, lower=0.0, upper=1e-2),
        "alpha": CircuitParameterSettings(initial=0.8, lower=0.3, upper=1.0),
        "Cb": CircuitParameterSettings(initial=3e-9, lower=0.0, upper=1e-3),
    }


class CircuitSettings(BaseModel):
    """Circuit ajusté par le fit Orazem — valeurs PAR DÉFAUT proposées par l'UI.

    La validation (expression sûre, paramètres couverts exactement, bornes cohérentes)
    est faite par ``fits.orazem_fit.compile_circuit_fit`` avant toute analyse : la
    config ne fait que transporter la saisie.
    """

    expression: str = _DEFAULT_CIRCUIT
    target_param: str = "Rct"
    parameters: dict[str, CircuitParameterSettings] = Field(
        default_factory=_default_circuit_parameters)


class DRTSettings(BaseModel):
    # DRT du pipeline (drt/engine.py), sur chaque réplicat brut ET la moyenne de chaque
    # groupe : 'optimize' (MAP, aperçu ~1 s) ou 'sample' (HMC, diagnostics de
    # convergence et intervalles, plusieurs minutes par spectre). La grille τ n'est
    # PAS configurée ici : bayes_drt2 la construit depuis les fréquences mesurées.
    enabled: bool = True
    mode: str = "optimize"

    @field_validator("mode")
    @classmethod
    def _mode_valid(cls, v: str) -> str:
        if v not in ("optimize", "sample"):
            raise ValueError(f"fit.drt.mode doit valoir 'optimize' ou 'sample' (reçu {v!r})")
        return v


class ErrorStructureSettings(BaseModel):
    """Structure d'erreur d'Orazem — caractérisée par le measurement model
    (``core/measurement_model.py``) sur les réplicats de CHAQUE groupe, à chaque
    analyse. Aucun coefficient n'est persisté ni relu (AUDIT.md ERR-2).

    - min_replicates : nombre minimal de réplicats pour caractériser (≥ 3 recommandé).
    """

    min_replicates: int = 3

    @field_validator("min_replicates")
    @classmethod
    def _min_rep_valid(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"error_structure.min_replicates doit être ≥ 2 (reçu {v})")
        return v


class FitSettings(BaseModel):
    # Pondération UNIQUE : structure d'erreur d'Orazem (core/measurement_model.py).
    error_structure: ErrorStructureSettings = Field(
        default_factory=ErrorStructureSettings
    )
    n_freqs_parasites: list = Field(default_factory=lambda: [50.0, 100.0])
    tol_parasites: float = 3.0
    max_iter: int = 10000
    n_monte_carlo: int = 1000
    circuit: CircuitSettings = Field(default_factory=CircuitSettings)
    drt: DRTSettings = Field(default_factory=DRTSettings)
    drt_wiener_W: float = 1.0e-8
    drt_n_z: int = 10000


class ExportSettings(BaseModel):
    dpi: int = 150
    fig_width: int = 1200
    fig_height: int = 800


class AcquisitionSettings(BaseModel):
    # Bornes de la plage de balayage EIS, utilisées pour reconstruire l'axe
    # fréquence des fichiers EC-Lab qui n'exportent QUE Re(Z)/-Im(Z) (pas de
    # colonne fréquence). Balayage logarithmique fixe f_max → f_min (HF→BF).
    f_max_hz: float = 1.0e6
    f_min_hz: float = 0.1


class AppSettings(BaseModel):
    physics: PhysicsSettings = Field(default_factory=PhysicsSettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    conditions: ConditionsSettings = Field(default_factory=ConditionsSettings)
    acquisition: AcquisitionSettings = Field(default_factory=AcquisitionSettings)
    fit: FitSettings = Field(default_factory=FitSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> AppSettings:
    """Load YAML config file and validate with Pydantic.

    Args:
        path: Path to YAML config file.

    Returns:
        Validated AppSettings instance.
    """
    if not path.exists():
        return AppSettings()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppSettings.model_validate(raw)


def config_to_dict(settings: AppSettings) -> dict:
    """Convert AppSettings to a plain dict for use in fits and pipeline.

    Args:
        settings: AppSettings instance.

    Returns:
        Nested dict matching the YAML structure.
    """
    return settings.model_dump()
