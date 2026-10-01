"""I6/I7 — les échecs silencieux deviennent visibles.

- I7 : RandlesFullModel.fit remonte des `warnings` (non convergé, résidu élevé,
  paramètre en butée) — les 3 gardes qui auraient signalé B1.
- I6 : le registre expose les modules de fits/ qui n'ont pas pu être chargés.
"""

import numpy as np

from fits.physics import Z_randles_full
from fits.randles_full import RandlesFullModel
from fits import registry
from core.models import EISSpectrum

# Pondération = structure d'erreur d'Orazem. Ces spectres synthétiques n'ont pas
# de réplicats : le fit réutilise la structure de référence persistée par la
# fixture conftest.isolate_error_structure (source="reused_persisted").
_CFG = {"fit": {"max_iter": 10000}}


def _spectrum(Rct: float) -> EISSpectrum:
    omega = np.logspace(-1, 5, 100)
    f = omega / (2 * np.pi)
    # Tous les paramètres à l'intérieur des bornes (R_D=200, tau_d=0.5) pour que
    # seul le paramètre testé puisse coller à une borne.
    Z = Z_randles_full(omega, 500.0, 50.0, 1e-9, Rct, 1e-6, 0.90, 200.0, 0.5)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(label="x", f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
                       concentration=1e-9, step="hybridization", n_points=100)


def test_clean_fit_has_no_warnings():
    # χ²ᵣ étant désormais un test d'adéquation (poids = 1/σ² de la structure
    # d'erreur), un « bon » fit doit porter un bruit COHÉRENT avec la structure
    # réutilisée (fixture : σ = 0.01·|Z_re| + 0.01·|Z_im| + 1 Ω). On injecte ce
    # bruit exact → χ²ᵣ ≈ 1 et aucune alerte (ni convergence, ni résidu, ni borne,
    # ni adéquation).
    sp = _spectrum(5000.0)
    Z = sp.Zre + 1j * sp.Zim
    sigma = 0.01 * np.abs(sp.Zre) + 0.01 * np.abs(sp.Zim) + 1.0
    rng = np.random.default_rng(0)
    sp.Zre = sp.Zre + rng.normal(0.0, sigma)
    sp.Zim = sp.Zim + rng.normal(0.0, sigma)
    fr = RandlesFullModel().fit(sp, _CFG)
    assert fr.warnings == [], fr.warnings
    lo, hi = fr.chi2_reduced_ci
    assert lo <= fr.chi2_reduced <= hi
    assert fr.reconstruction_error is not None and fr.reconstruction_error < 0.05


def test_rct_pinned_to_lower_bound_warns():
    # Vrai Rct = 50 < borne basse 100 → l'optimiseur colle Rct à 100.
    fr = RandlesFullModel().fit(_spectrum(50.0), _CFG)
    assert any("Rct" in w and "borne basse" in w for w in fr.warnings), fr.warnings


def test_high_residual_warns():
    # Bruit massif → résidu relatif élevé, ajustement médiocre.
    sp = _spectrum(5000.0)
    rng = np.random.default_rng(0)
    sp.Zre = sp.Zre + rng.normal(0, 0.5 * np.abs(sp.Zre))
    sp.Zim = sp.Zim + rng.normal(0, 0.5 * np.abs(sp.Zim))
    fr = RandlesFullModel().fit(sp, _CFG)
    assert any("résidu relatif élevé" in w for w in fr.warnings), fr.warnings


def test_registry_exposes_load_state():
    # Tous les modèles attendus se chargent dans cet environnement (cvxopt présent).
    names = {m.name for m in registry.all_models()}
    assert "randles_full" in names
    assert isinstance(registry.discovery_errors(), dict)


def test_get_model_unknown_lists_available():
    try:
        registry.get_model("modele_inexistant")
    except KeyError as e:
        assert "Available" in str(e)
    else:
        raise AssertionError("get_model aurait dû lever KeyError")
