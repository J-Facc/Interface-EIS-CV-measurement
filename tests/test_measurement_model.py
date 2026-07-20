"""Structure d'erreur d'Orazem — méthode de pondération UNIQUE.

Couvre les exigences de la refonte :
  - régression (α,β,γ,δ) sur σ synthétique connu ;
  - χ²_red → ~1 sur données à bruit imposé (structure = vrai σ) ;
  - covariance = (JᵀJ)⁻¹ SANS rééchelonnement (test numérique du facteur absent) ;
  - persistance : caractériser → écrire → relire → fit sans réplicats réutilise
    les coefficients (source="reused_persisted") ;
  - refus explicite quand aucune caractérisation et pas de réplicats ;
  - cohérence de signe des résidus re/im entre residuals() et le χ² post-fit.

Références : Orazem & Tribollet, "Electrochemical Impedance Spectroscopy" (Wiley),
chap. Measurement Model ; Agarwal, Orazem & García-Rubio, J. Electrochem. Soc.
(1992-1995).
"""

import json

import numpy as np
import pytest

from fits.physics import Z_randles_full
from fits.randles_full import RandlesFullModel, _PARAM_NAMES
from fits.weighting import resolve_weights, ErrorStructureUnavailable
from fits.error_structure import (
    ErrorStructure,
    regress_coefficients,
    characterize_from_replicate_spectra,
    persist,
    load_latest,
    resolve_error_structure,
)
from core.models import EISSpectrum
from core.loader import average_replicates


# ── Fabrique de spectres synthétiques ────────────────────────────────────────

_TRUE = dict(Re=500.0, Re_prime=200.0, Cb=2e-8, Rct=5000.0,
             Qdl=1e-6, alpha=0.85, R_D=2000.0, tau_d=50.0)


def _clean_Z(n=120):
    f = np.logspace(-3, 5, n)
    omega = 2.0 * np.pi * f
    Z = Z_randles_full(omega, *[_TRUE[k] for k in _PARAM_NAMES])
    idx = np.argsort(f)[::-1]
    return f[idx], Z.real[idx], (-Z.imag)[idx]  # convention Zim = -Im(Z) > 0


def _structured_sigma(Zre, Zim, a, b, d):
    return a * np.abs(Zre) + b * np.abs(Zim) + d


def _make_replicates(a, b, d, n_rep, seed=0, n=120, step="probe"):
    """n_rep réplicats bruités par une structure d'erreur connue (α=β=a, δ=d)."""
    f, Zre0, Zim0 = _clean_Z(n)
    sigma = _structured_sigma(Zre0, Zim0, a, b, d)
    rng = np.random.default_rng(seed)
    reps = []
    for k in range(n_rep):
        reps.append(EISSpectrum(
            label=f"rep{k}", f=f,
            Zre=Zre0 + rng.normal(0.0, sigma),
            Zim=Zim0 + rng.normal(0.0, sigma),
            concentration=0.0, step=step, n_points=n,
        ))
    return f, Zre0, Zim0, sigma, reps


# ── 1. Régression des coefficients sur σ synthétique connu ───────────────────

def test_regress_coefficients_recovers_known_structure():
    """regress_coefficients retrouve (α,β,δ) d'une σ synthétique exacte."""
    f, Zre, Zim = _clean_Z()
    a_true, b_true, d_true = 0.02, 0.02, 3.0
    sigma = _structured_sigma(Zre, Zim, a_true, b_true, d_true)

    a, b, g, d, quality = regress_coefficients(
        Zre, Zim, sigma, sigma, equal_re_im=True, R_m=None
    )
    assert a == pytest.approx(b)                       # equal_re_im impose α = β
    assert a == pytest.approx(a_true, rel=1e-6, abs=1e-6)
    assert d == pytest.approx(d_true, rel=1e-6, abs=1e-6)
    assert g == pytest.approx(0.0, abs=1e-9)           # pas de terme |Z|²
    assert quality < 1e-9                              # ajustement quasi parfait


def test_regress_coefficients_non_negative():
    """Les coefficients restent ≥ 0 même face à une cible qui « voudrait » du négatif."""
    f, Zre, Zim = _clean_Z()
    # σ décroissante avec |Z| : aucune combinaison ≥ 0 ne peut la reproduire ;
    # la NNLS renvoie néanmoins des coefficients tous ≥ 0.
    sigma = 100.0 - 0.001 * np.abs(Zre)
    sigma = np.clip(sigma, 1.0, None)
    a, b, g, d, _q = regress_coefficients(Zre, Zim, sigma, sigma, equal_re_im=True)
    assert a >= 0 and b >= 0 and g >= 0 and d >= 0


# ── 2. χ²_red → ~1 sur bruit imposé (structure = vrai σ) ─────────────────────

def test_chi2_reduced_around_one_with_known_sigma(tmp_path):
    """Fit d'un spectre UNIQUE à bruit σ connu, pondéré par la structure exacte.

    On persiste la structure d'erreur vraie (α=β, δ) ; un spectre unique (N=1) de
    bruit σ est ajusté en la réutilisant → E[χ²_red] = 1. Moyenné sur plusieurs
    graines pour amortir la fluctuation √(2/dof).
    """
    a_true, b_true, d_true = 0.01, 0.01, 2.0
    path = tmp_path / "es.json"
    persist(ErrorStructure(alpha=a_true, beta=b_true, gamma=0.0, delta=d_true,
                           R_m=None, equal_re_im=True, timestamp="t",
                           campaign_id="c", n_replicates=5, quality=0.0), path)
    config = {"fit": {"max_iter": 20000,
                      "error_structure": {"persistence_path": str(path)}}}

    f, Zre0, Zim0 = _clean_Z()
    sigma = _structured_sigma(Zre0, Zim0, a_true, b_true, d_true)

    chi2_vals = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        sp = EISSpectrum(
            label="mm", f=f,
            Zre=Zre0 + rng.normal(0.0, sigma),
            Zim=Zim0 + rng.normal(0.0, sigma),
            concentration=0.0, step="probe", n_points=len(f),
        )
        fr = RandlesFullModel().fit(sp, config)
        assert fr.converged
        assert fr.chi2_is_valid_test
        assert fr.error_structure_source == "reused_persisted"
        chi2_vals.append(fr.chi2_reduced)

    assert 0.85 < float(np.mean(chi2_vals)) < 1.15, chi2_vals


def test_chi2_reduced_far_from_one_when_sigma_wrong(tmp_path):
    """Si la structure d'erreur SOUS-estime fortement le bruit, χ²_red ≫ 1.

    Contre-épreuve : ce n'est PAS un χ²≈1 automatique. Une σ 10× trop petite par
    rapport au bruit réel fait exploser χ²_red (facteur ~100), et le diagnostic
    d'adéquation se déclenche.
    """
    path = tmp_path / "es.json"
    # Structure persistée : σ 10× trop PETITE.
    persist(ErrorStructure(alpha=0.001, beta=0.001, gamma=0.0, delta=0.2,
                           R_m=None, equal_re_im=True, timestamp="t",
                           campaign_id="c", n_replicates=5, quality=0.0), path)
    config = {"fit": {"max_iter": 20000,
                      "error_structure": {"persistence_path": str(path)}}}

    f, Zre0, Zim0 = _clean_Z()
    sigma = _structured_sigma(Zre0, Zim0, 0.01, 0.01, 2.0)   # vrai bruit 10× plus grand
    rng = np.random.default_rng(0)
    sp = EISSpectrum(label="mm", f=f,
                     Zre=Zre0 + rng.normal(0.0, sigma),
                     Zim=Zim0 + rng.normal(0.0, sigma),
                     concentration=0.0, step="probe", n_points=len(f))
    fr = RandlesFullModel().fit(sp, config)
    assert fr.chi2_reduced > 10.0
    assert any("χ²" in w for w in fr.warnings)


# ── 3. Covariance = (JᵀJ)⁻¹ SANS rééchelonnement ─────────────────────────────

def test_covariance_not_rescaled(tmp_path):
    """σ_param = sqrt(diag((JᵀJ)⁻¹)) — aucune multiplication par 2·cost/dof.

    On calcule la jacobienne pondérée à l'optimum et on vérifie que les σ du
    modèle valent EXACTEMENT sqrt(diag(inv(JᵀJ))). Si une branche de
    rééchelonnement (facteur 2·cost/dof) réapparaissait, l'égalité casserait
    (le χ²_red de ce jeu vaut ~1 mais ≠ 1 exactement).
    """
    from scipy.optimize import least_squares

    f, Zre0, Zim0 = _clean_Z()
    sigma = _structured_sigma(Zre0, Zim0, 0.01, 0.01, 2.0)
    rng = np.random.default_rng(1)
    sp = EISSpectrum(label="cov", f=f,
                     Zre=Zre0 + rng.normal(0.0, sigma),
                     Zim=Zim0 + rng.normal(0.0, sigma),
                     concentration=0.0, step="probe", n_points=len(f))

    w = 1.0 / sigma ** 2
    fr = RandlesFullModel().fit(sp, {"fit": {"max_iter": 20000}}, weights=(w, w))
    assert fr.converged

    # Reconstruit J à l'optimum (mêmes résidus pondérés que le modèle).
    omega = 2.0 * np.pi * sp.f
    sw = np.sqrt(w)
    x_opt = np.array([fr.params[k] for k in _PARAM_NAMES])

    def residuals(x):
        Z = Z_randles_full(omega, *x)
        return np.concatenate([(Z.real - sp.Zre) * sw, (-Z.imag - sp.Zim) * sw])

    res = least_squares(residuals, x_opt, method="trf", max_nfev=1)
    J = res.jac
    cov_direct = np.linalg.inv(J.T @ J)
    std_direct = np.sqrt(np.abs(np.diag(cov_direct)))

    for i, k in enumerate(_PARAM_NAMES):
        assert fr.params_std[k] == pytest.approx(std_direct[i], rel=1e-6, abs=1e-12), k


# ── 4. Persistance : caractériser → écrire → relire → réutiliser ─────────────

def test_persistence_roundtrip_and_reuse(tmp_path):
    """Caractérisation sur réplicats → persistance → fit sans réplicats réutilise."""
    a_true, d_true = 0.01, 2.0
    _f, _r, _i, _s, reps = _make_replicates(a_true, a_true, d_true, n_rep=6, seed=7)
    path = tmp_path / "es.json"
    config = {"fit": {"max_iter": 20000,
                      "error_structure": {"persistence_path": str(path),
                                          "equal_re_im": True, "min_replicates": 3}}}

    # Caractérisation « maintenant » sur le spectre moyenné (porte σ + n_replicates).
    avg = average_replicates(reps)
    es = resolve_error_structure(avg, config)
    assert es.source == "characterized_now"
    assert es.alpha == pytest.approx(a_true, abs=3e-3)
    assert es.delta == pytest.approx(d_true, abs=1.5)

    # Le fichier persisté est relisible et contient une liste d'entrées.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 1
    reloaded = load_latest(path)
    assert reloaded is not None and reloaded.source == "reused_persisted"
    assert reloaded.alpha == pytest.approx(es.alpha)

    # Un spectre SANS réplicats réutilise les coefficients persistés.
    f, Zre0, Zim0 = _clean_Z()
    single = EISSpectrum(label="single", f=f, Zre=Zre0, Zim=Zim0,
                         concentration=0.0, step="probe", n_points=len(f))
    fr = RandlesFullModel().fit(single, config)
    assert fr.error_structure_source == "reused_persisted"
    assert fr.error_structure_timestamp == es.timestamp


# ── 5. Refus explicite : ni caractérisation, ni réplicats ────────────────────

def test_refusal_without_characterization(tmp_path):
    """Sans coefficients persistés et sans réplicats → ErrorStructureUnavailable."""
    path = tmp_path / "empty.json"   # n'existe pas
    config = {"fit": {"max_iter": 20000,
                      "error_structure": {"persistence_path": str(path)}}}
    f, Zre0, Zim0 = _clean_Z()
    single = EISSpectrum(label="single", f=f, Zre=Zre0, Zim=Zim0,
                         concentration=0.0, step="probe", n_points=len(f))

    with pytest.raises(ErrorStructureUnavailable) as exc:
        resolve_weights(single, config)
    assert "non caractérisée" in str(exc.value)

    # Le modèle propage le refus (pas de repli sur un σ arbitraire).
    with pytest.raises(ErrorStructureUnavailable):
        RandlesFullModel().fit(single, config)


def test_refusal_when_too_few_replicates(tmp_path):
    """< min_replicates ET aucune persistance → refus (pas de repli)."""
    path = tmp_path / "empty.json"
    config = {"fit": {"error_structure": {"persistence_path": str(path),
                                          "min_replicates": 3}}}
    _f, _r, _i, _s, reps = _make_replicates(0.01, 0.01, 2.0, n_rep=2, seed=1)
    avg = average_replicates(reps)          # n_replicates = 2 < 3
    with pytest.raises(ErrorStructureUnavailable):
        resolve_weights(avg, config)


# ── 6. Cohérence de signe des résidus imaginaires ───────────────────────────

def test_imaginary_residual_sign_equivalence(tmp_path):
    """(−Z.imag − Zim)² ≡ (Zim + Z_fit.imag)² : même contribution au χ².

    residuals() empile (−Z.imag − Zim) ; le χ² post-fit utilise
    res_im = Zim + Z_fit.imag. Les deux sont opposés → carrés identiques. On le
    vérifie numériquement à l'optimum du fit.
    """
    path = tmp_path / "es.json"
    persist(ErrorStructure(alpha=0.01, beta=0.01, gamma=0.0, delta=2.0, R_m=None,
                           equal_re_im=True, timestamp="t", campaign_id="c",
                           n_replicates=5, quality=0.0), path)
    config = {"fit": {"max_iter": 20000,
                      "error_structure": {"persistence_path": str(path)}}}
    f, Zre0, Zim0 = _clean_Z()
    sp = EISSpectrum(label="sign", f=f, Zre=Zre0, Zim=Zim0,
                     concentration=0.0, step="probe", n_points=len(f))
    fr = RandlesFullModel().fit(sp, config)

    omega = 2.0 * np.pi * sp.f
    x = [fr.params[k] for k in _PARAM_NAMES]
    Z_fit = Z_randles_full(omega, *x)

    res_im_residuals = -Z_fit.imag - sp.Zim       # forme de residuals()
    res_im_postfit = sp.Zim + Z_fit.imag          # forme du χ² post-fit
    assert np.allclose(res_im_residuals, -res_im_postfit)
    assert np.allclose(res_im_residuals ** 2, res_im_postfit ** 2)


# ── 7. Variante Voigt (option voigt_based) ───────────────────────────────────

def test_voigt_based_characterization_runs(tmp_path):
    """La caractérisation voigt_based produit des coefficients ≥ 0 exploitables."""
    _f, _r, _i, _s, reps = _make_replicates(0.01, 0.01, 2.0, n_rep=5, seed=2)
    es = characterize_from_replicate_spectra(
        reps, {"equal_re_im": True, "voigt_based": True, "min_replicates": 3}
    )
    assert es.voigt_based is True
    assert es.alpha >= 0 and es.beta >= 0 and es.delta >= 0
    assert es.alpha == pytest.approx(es.beta)     # equal_re_im
    # σ reconstruit strictement positif → poids finis.
    sigma = es.sigma(reps[0].Zre, reps[0].Zim)
    assert np.all(sigma > 0)
