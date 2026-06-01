"""Abstract base class for all EIS fit models."""

from abc import ABC, abstractmethod
from core.models import EISSpectrum, FitResult


class BaseFitModel(ABC):
    """Interface that every fit model must implement.

    To add a new model: create a file in fits/, subclass BaseFitModel,
    and it will be auto-discovered by FitRegistry without touching other files.
    """

    name: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def fit(self, spectrum: EISSpectrum, config: dict, weights=None) -> FitResult:
        """Fit the model to the spectrum.

        Args:
            spectrum: EIS spectrum to fit.
            config: App config dict (from config_to_dict).

        Returns:
            FitResult with parameters, fitted impedance, and diagnostics.
        """

    @abstractmethod
    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        """Estimate initial parameter values from spectrum features.

        Args:
            spectrum: EIS spectrum.
            config: App config dict.

        Returns:
            Dict {param_name: initial_value}.
        """

    @abstractmethod
    def bounds(self, config: dict) -> tuple:
        """Return lower and upper parameter bounds.

        Args:
            config: App config dict.

        Returns:
            Tuple (lower_dict, upper_dict) with same keys as initial_guess.
        """
