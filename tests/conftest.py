"""Fixtures partagées : isolation de la persistance de structure d'erreur.

La méthode de pondération UNIQUE (structure d'erreur d'Orazem, fits/error_structure.py)
persiste les coefficients caractérisés dans un fichier JSON. Pour que les tests
soient déterministes et n'écrivent pas dans config/error_structure.json du dépôt :

- on redirige le chemin de persistance PAR DÉFAUT vers un fichier temporaire propre
  à chaque test (monkeypatch de DEFAULT_PERSIST_PATH) ;
- on y écrit une structure d'erreur « instrument déjà caractérisé » de référence,
  de sorte que les fits de spectres SANS réplicats (cas courant des tests hérités)
  la réutilisent (source="reused_persisted") au lieu d'être refusés.

Les tests qui veulent contrôler finement la persistance (refus, réutilisation
explicite) passent leur propre `persistence_path` dans la config, ce qui court-
circuite ce défaut.
"""

import pytest

import fits.error_structure as es_mod
from fits.error_structure import ErrorStructure, persist


@pytest.fixture(autouse=True)
def isolate_error_structure(tmp_path, monkeypatch):
    """Redirige la persistance vers un tmp par test + structure de référence."""
    default_path = tmp_path / "error_structure_default.json"
    monkeypatch.setattr(es_mod, "DEFAULT_PERSIST_PATH", default_path)

    # Structure de référence « instrument caractérisé » : σ ≈ 1 % du module + fond.
    baseline = ErrorStructure(
        alpha=0.01, beta=0.01, gamma=0.0, delta=1.0, R_m=None,
        equal_re_im=True, voigt_based=False,
        timestamp="2020-01-01T00:00:00+00:00", campaign_id="baseline-fixture",
        n_replicates=5, quality=0.0, source="characterized_now",
    )
    persist(baseline, default_path)
    return default_path
