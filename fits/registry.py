"""Auto-discovery and registration of fit models from the fits/ package."""
import importlib
import pkgutil
from typing import List
import fits
from fits.base import BaseFitModel
from core.logger import get_logger

log = get_logger("registry")

_registry: dict = {}
# I6 : modèle non chargé → raison, au lieu de disparaître en silence.
# ex. {"randles_full": "No module named 'scipy'"}. Exposé via discovery_errors().
# La DRT (drt_bayes, fits/drt_fit.py, wrapper bayes_drt2) est un plugin du registre
# au même titre que les fits paramétriques : découverte auto, lancée par le pipeline.
_errors: dict = {}

def _discover() -> None:
    """Import all submodules in fits/ and register BaseFitModel subclasses.

    La DRT (fits/drt_fit.py → DRTBayesModel, wrapper bayes_drt2) est de nouveau un
    plugin du registre : elle est découverte et lancée par core/pipeline.py comme
    les autres fits (mode 'optimize' par défaut). Si l'extra DRT (cvxopt/cmdstanpy)
    est absent, son import échoue proprement et la raison est exposée via
    discovery_errors() (I6), sans masquer les autres modèles.
    """
    if _registry or _errors:
        return
    skip = {"base", "physics", "registry", "kk_validation", "weighting"}
    for _finder, mod_name, _ispkg in pkgutil.iter_modules(fits.__path__):
        if mod_name in skip or mod_name.startswith("_"):
            continue
        full_name = f"fits.{mod_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:
            # Ne pas faire disparaître le modèle en silence : logguer + mémoriser
            # la raison pour que l'UI puisse l'afficher (I6).
            _errors[mod_name] = str(exc)
            log.warning(f"Modèle '{mod_name}' non chargé : {exc}")
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
    _discover()
    return list(_registry.values())

def discovery_errors() -> dict:
    """Modules de fits/ qui n'ont pas pu être importés → raison (I6).

    Permet à l'UI d'afficher « modèle X indisponible : <raison> » au lieu de
    laisser l'utilisateur croire qu'il n'existe que N modèles.
    """
    _discover()
    return dict(_errors)

def get_model(name: str) -> BaseFitModel:
    _discover()
    if name not in _registry:
        hint = f" Modules en échec : {_errors}." if _errors else ""
        raise KeyError(
            f"Fit model '{name}' not found. "
            f"Available: {sorted(_registry.keys())}.{hint}"
        )
    return _registry[name]
