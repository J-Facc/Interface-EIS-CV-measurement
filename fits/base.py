"""Abstract base class for all EIS fit models."""

from abc import ABC, abstractmethod
from core.models import EISSpectrum, FitResult


class BaseFitModel(ABC):
    """Interface every fit model must implement.

    Class attributes:
        name: Short identifier used as dict key (e.g. 'randles_full').
        label: Human-readable name shown in the UI.
        description: Tooltip help text.
    """

    name: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def fit(self, spectrum: EISSpectrum, config: dict) -> FitResult:
        """Run the optimisation and return a FitResult.

        Args:
            spectrum: EIS data to fit.
            config: App config dict from AppSettings.model_dump().

        Returns:
            FitResult with fitted parameters and reconstructed impedance.
        """
        ...

    @abstractmethod
    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        """Return initial parameter guesses as {param_name: value}.

        Args:
            spectrum: EIS data (used for heuristic guessing).
            config: App config dict.

        Returns:
            Dict of parameter names to initial float values.
        """
        ...

    @abstractmethod
    def bounds(self, config: dict) -> tuple:
        """Return (lower_bounds, upper_bounds) as dicts {param_name: value}.

        Args:
            config: App config dict.

        Returns:
            Tuple (lower_dict, upper_dict).
        """
        ...
