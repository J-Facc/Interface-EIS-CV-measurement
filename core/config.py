"""Load YAML configuration and validate with Pydantic."""

import os
from typing import Optional, Union

import yaml
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    #
    # Les réglages d'inversion ci-dessous reprennent À L'IDENTIQUE les défauts de
    # drt/engine.py (DEFAULT_*, justifiés dans drt/VALIDATION_REGLAGES.md) :
    # tests/test_config.py garde cette égalité. Les omettre = ces défauts.
    # Clé inconnue = ValidationError (extra='forbid') : une faute de frappe ne doit
    # pas être ignorée en silence. Ni tau_min ni tau_max n'existent : le moteur ne les lit pas.
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: str = "optimize"
    nonneg: bool = True
    init_from_ridge: bool = True
    random_seed: int = 1234
    chains: int = 4             # HMC seulement
    warmup: int = 500           # HMC seulement
    samples: int = 2000         # HMC seulement
    adapt_delta: float = 0.99   # HMC seulement
    max_iter: int = 50000       # L-BFGS ('optimize') seulement
    # Fenêtre de τ (s) où chercher le pic PRINCIPAL de γ(τ) dont l'aire donne Rct_DRT
    # (core/drt_peak.py, v3). Post-traitement de γ : n'est PAS transmise au moteur.
    tau_window: list[float] = Field(default_factory=lambda: [1.0e-5, 1.0e-1])

    @field_validator("tau_window")
    @classmethod
    def _tau_window_valid(cls, v: list) -> list:
        if len(v) != 2 or not (0 < v[0] < v[1]):
            raise ValueError(f"fit.drt.tau_window doit valoir [tau_min, tau_max] avec 0 < tau_min < tau_max "
                             f"(reçu {v!r})")
        return [float(v[0]), float(v[1])]

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
    circuit: CircuitSettings = Field(default_factory=CircuitSettings)
    drt: DRTSettings = Field(default_factory=DRTSettings)


class AcquisitionSettings(BaseModel):
    # Bornes de la plage de balayage EIS, utilisées pour reconstruire l'axe
    # fréquence des fichiers EC-Lab qui n'exportent QUE Re(Z)/-Im(Z) (pas de
    # colonne fréquence). Balayage logarithmique fixe f_max → f_min (HF→BF).
    f_max_hz: float = 1.0e6
    f_min_hz: float = 0.1


class CVSettings(BaseModel):
    """Extraction du courant de pic CV (core/cv.py).

    - peak_window_V : fenêtre de potentiel (V) où chercher le maximum de I sur la branche
      ANODIQUE (E croissant) du cycle retenu ;
    - cycle : cycle conservé au nettoyage — "last" (défaut), "first" ou un numéro (1, 2…).
    """

    model_config = ConfigDict(extra="forbid")

    peak_window_V: list[float] = Field(default_factory=lambda: [0.0, 0.25])
    cycle: Union[int, str] = "last"

    @field_validator("peak_window_V")
    @classmethod
    def _window_valid(cls, v: list) -> list:
        if len(v) != 2 or not v[0] < v[1]:
            raise ValueError(f"cv.peak_window_V doit valoir [E_min, E_max] avec E_min < E_max (reçu {v!r})")
        return [float(v[0]), float(v[1])]

    @field_validator("cycle")
    @classmethod
    def _cycle_valid(cls, v):
        if isinstance(v, str) and v not in ("last", "first"):
            raise ValueError(f"cv.cycle doit valoir 'last', 'first' ou un entier ≥ 1 (reçu {v!r})")
        if isinstance(v, int) and v < 1:
            raise ValueError(f"cv.cycle doit être ≥ 1 (reçu {v})")
        return v


class AnalysisSettings(BaseModel):
    """Calcul en arrière-plan (core/background.py).

    - max_workers : nombre de processus ; None = os.cpu_count() − 1 (minimum 1).
    """

    model_config = ConfigDict(extra="forbid")

    max_workers: Optional[int] = None


class AppSettings(BaseModel):
    acquisition: AcquisitionSettings = Field(default_factory=AcquisitionSettings)
    fit: FitSettings = Field(default_factory=FitSettings)
    cv: CVSettings = Field(default_factory=CVSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)


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


# ─────────────────────────────────────────────────────────────────────────────
# Données utilisateur persistantes (v3) — HORS du dossier de l'application
# ─────────────────────────────────────────────────────────────────────────────

#: Variable d'environnement qui force le dossier des données utilisateur (tests, usages avancés).
USER_DATA_ENV = "EIS_USER_DATA_DIR"
#: Nom du dossier extrait par launch.bat : le code y est REMPLACÉ à chaque mise à jour.
LAUNCHER_APP_DIRNAME = "eis_app"


def app_root() -> Path:
    """Dossier racine du code de l'application (celui qui contient app.py)."""
    return Path(__file__).resolve().parent.parent


def user_data_dir(root: Optional[Path] = None) -> Path:
    """Dossier des données utilisateur persistantes (``biomarkers.json``…).

    launch.bat ré-extrait le code dans ``eis_app/`` à chaque mise à jour : rien de
    persistant ne doit y vivre. Quand l'application tourne depuis ``<lanceur>/eis_app/``,
    les données vont dans ``<lanceur>/user_data/`` (à côté de launch.bat) ; sinon (clone de
    développement) dans ``<dépôt>/user_data/`` (ignoré par git).

    Args:
        root: racine du code (défaut : :func:`app_root`) — paramétrable pour les tests.
    """
    forced = os.environ.get(USER_DATA_ENV)
    if forced:
        return Path(forced)
    root = Path(root) if root is not None else app_root()
    if root.name.lower() == LAUNCHER_APP_DIRNAME:
        return root.parent / "user_data"
    return root / "user_data"
