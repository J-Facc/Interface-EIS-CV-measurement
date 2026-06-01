"""Auto-discovery and registration of fit models from the fits/ package."""

import importlib
import pkgutil
from typing import List

import fits
from fits.base import BaseFitModel

_registry: dict = {}

from fits.drt_fit import DRTFitModel
registry.register(DRTFitModel())
def _discover() -> None:
    """Import all submodules in fits/ and register BaseFitModel subclasses."""
    if _registry:
        return

    skip = {"base", "physics", "registry"}

    for _finder, mod_name, _ispkg in pkgutil.iter_modules(fits.__path__):
        if mod_name in skip or mod_name.startswith("_"):
            continue
        full_name = f"fits.{mod_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception:
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseFitModel)
                and attr is not BaseFitModel
                and getattr(attr, "name", "")
            ):
                instance = attr()
                _registry[instance.name] = instance


def all_models() -> List[BaseFitModel]:
    """Return all registered fit model instances.

    Returns:
        List of BaseFitModel instances sorted by name.
    """
    _discover()
    return list(_registry.values())


def get_model(name: str) -> BaseFitModel:
    """Get a fit model instance by short name.

    Args:
        name: Model name string (e.g. 'randles_full').

    Returns:
        BaseFitModel instance.

    Raises:
        KeyError: If the model name is not registered.
    """
    _discover()
    if name not in _registry:
        raise KeyError(
            f"Fit model '{name}' not found. "
            f"Available: {sorted(_registry.keys())}"
        )
    return _registry[name]
