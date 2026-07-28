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
    # Mode DRT du plugin bayes_drt2 (fits/drt_fit.py) : 'optimize' (MAP Stan, défaut
    # lancé par le pipeline) ou 'sample' (HMC, à la demande via recompute_drt).
    # La grille τ n'est PAS configurée ici : bayes_drt2 la construit lui-même à
    # partir des fréquences mesurées (une décade au-delà de chaque borne, 10 pts/déc).
    mode: str = "optimize"


class ErrorStructureSettings(BaseModel):
    """Configuration de la structure d'erreur d'Orazem — pondération UNIQUE.

    Les COEFFICIENTS (α, β, γ, δ) ne figurent PAS ici : ils sont estimés sur
    réplicats puis persistés dans `persistence_path`. Cette section ne porte que
    les OPTIONS de la méthode.

    - equal_re_im : impose α = β (hypothèse d'égalité des variances Re/Im du
      measurement model — standard, recommandé).
    - voigt_based : estime σ empirique par l'écart-type des résidus d'un circuit
      de Voigt ajusté à chaque réplicat (plus fidèle à Orazem) ; sinon écart-type
      inter-réplicats direct.
    - R_m : résistance de mesure (Ω) ; None => terme γ·|Z|²/R_m absorbé/ignoré.
    - min_replicates : nombre minimal de réplicats pour caractériser (≥ 3 recommandé).
    - persistence_path : fichier JSON d'historique des coefficients caractérisés
      (None => config/error_structure.json).
    """

    equal_re_im: bool = True
    voigt_based: bool = False
    R_m: Optional[float] = None
    min_replicates: int = 3
    persistence_path: Optional[str] = None

    @field_validator("R_m")
    @classmethod
    def _rm_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"error_structure.R_m doit être > 0 ou null (reçu {v})")
        return v

    @field_validator("min_replicates")
    @classmethod
    def _min_rep_valid(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"error_structure.min_replicates doit être ≥ 2 (reçu {v})")
        return v


class FitSettings(BaseModel):
    # Pondération UNIQUE : structure d'erreur d'Orazem (fits/error_structure.py).
    # Plus de weight_mode ni d'alpha_noise (supprimés).
    error_structure: ErrorStructureSettings = Field(
        default_factory=ErrorStructureSettings
    )
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
