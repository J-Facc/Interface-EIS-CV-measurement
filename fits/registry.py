"""Fit model registry: auto-discover and instantiate models by name.

To add a new model, create a file in fits/ that defines a class
decorated with @register. No other file needs modification.
"""

from fits.base import BaseFitModel
from typing import Type

_REGISTRY: dict = {}


def register(cls: Type[BaseFitModel]) -> Type[BaseFitModel]:
    """Class decorator that registers a BaseFitModel subclass.

    Args:
        cls: BaseFitModel subclass with a non-empty .name attribute.

    Returns:
        The same class (unchanged).
    """
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str) -> BaseFitModel:
    """Instantiate a registered fit model by its short name.

    Args:
        name: Model identifier (e.g. 'circular').

    Returns:
        Fresh instance of the model.

    Raises:
        KeyError: If name is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Fit model '{name}' not found. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_models() -> list:
    """Return all registered model name strings."""
    return list(_REGISTRY.keys())


def all_models() -> list:
    """Return one fresh instance of each registered model."""
    return [cls() for cls in _REGISTRY.values()]


def _auto_discover() -> None:
    """Import fit modules so their @register decorators execute."""
    import fits.circular_fit  # noqa: F401
    import fits.randles_constrained  # noqa: F401
    import fits.randles_full  # noqa: F401
    import fits.drt_tikhonov  # noqa: F401


_auto_discover()
