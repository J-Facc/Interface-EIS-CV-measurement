"""Load YAML configuration and validate with Pydantic."""

import yaml
from pathlib import Path
from pydantic import BaseModel, Field


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


class BoundsRandlesFull(BaseModel):
    # Bornes [min, max] typees list[float] : Pydantic coerce et valide, ce qui
    # neutralise le piege du resolveur float YAML 1.1 (ex. "1.0e+9" ou meme
    # "1.0e9" seraient convertis en float au lieu de rester des chaines).
    # Les cles doivent correspondre exactement a fits.randles_full._PARAM_NAMES
    # (Re, Re_prime, Cb, Rct, Qdl, alpha, R_D, tau_d) sinon les bornes YAML de
    # R_D/tau_d sont silencieusement ignorees au profit du fallback en dur.
    Re: list[float] = Field(default_factory=lambda: [100.0, 100000.0])
    Re_prime: list[float] = Field(default_factory=lambda: [1.0, 100000.0])
    Cb: list[float] = Field(default_factory=lambda: [1e-12, 1e-4])
    Rct: list[float] = Field(default_factory=lambda: [100.0, 1e9])
    Qdl: list[float] = Field(default_factory=lambda: [1e-12, 1e-4])
    alpha: list[float] = Field(default_factory=lambda: [0.6, 1.0])
    R_D: list[float] = Field(default_factory=lambda: [10.0, 1e6])
    tau_d: list[float] = Field(default_factory=lambda: [1e-4, 1e3])


class DRTSettings(BaseModel):
    n_tau: int = 50
    tau_min: float = 1e-5
    tau_max: float = 100.0
    lambda_auto: bool = True


class FitSettings(BaseModel):
    # "modulus" (pondération 1/(alpha_noise·|Z|)²) ou "sigma" (1/σ² inter-réplicats).
    weight_mode: str = "modulus"
    alpha_noise: float = 0.001
    n_freqs_parasites: list = Field(default_factory=lambda: [50.0, 100.0])
    tol_parasites: float = 3.0
    max_iter: int = 10000
    n_monte_carlo: int = 1000
    bounds_randles_full: BoundsRandlesFull = Field(default_factory=BoundsRandlesFull)
    drt: DRTSettings = Field(default_factory=DRTSettings)
    drt_wiener_W: float = 1.0e-8
    drt_n_z: int = 10000
    drt_kk_tol: float = 0.05


class ExportSettings(BaseModel):
    dpi: int = 150
    fig_width: int = 1200
    fig_height: int = 800


class AppSettings(BaseModel):
    physics: PhysicsSettings = Field(default_factory=PhysicsSettings)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    conditions: ConditionsSettings = Field(default_factory=ConditionsSettings)
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
