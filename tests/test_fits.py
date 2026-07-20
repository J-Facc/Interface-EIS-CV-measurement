"""Tests for fit models in fits/."""

import numpy as np
import pytest

from fits.physics import Z_randles_full
from fits.randles_full import RandlesFullModel, _PARAM_NAMES
from fits.kk_validation import kramers_kronig_check
from core.loader import load_spectrum, average_replicates
from core.models import EISSpectrum


# ── Synthetic spectrum factory ──────────────────────────────────────────────────

def _randles_spectrum(
    Re: float = 500.0,
    Rct: float = 5000.0,
    Qdl: float = 1e-6,
    alpha: float = 0.85,
    Re_prime: float = 50.0,
    Cb: float = 1e-9,
    R_D: float = 1.0,
    tau_d: float = 1.0,
    n: int = 80,
) -> EISSpectrum:
    """Generate a noiseless synthetic Randles spectrum.

    Frequency band is wide (1e-4 to 1e6 rad/s) and diffusion contribution
    negligible (R_D small) so that Im(Z) decays near zero at both ends —
    a requirement for the FFT/Wiener DRT deconvolution to resolve a single
    clean charge-transfer peak.
    """
    omega = np.logspace(-4, 6, n)
    f = omega / (2.0 * np.pi)
    Z = Z_randles_full(
        omega, Re, Re_prime, Cb, Rct, Qdl, alpha, R_D, tau_d,
    )
    # Store HF→BF (descending frequency)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        label="synthetic",
        # Convention loader/EISSpectrum : Zim = -Im(Z) > 0 (demi-cercle capacitif).
        f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
        concentration=1e-9, step="hybridization",
        n_points=n,
    )


def _zarc_spectrum(R: float = 3000.0, tau0: float = 1e-3, phi: float = 0.8,
                   n: int = 60) -> EISSpectrum:
    """Generate a synthetic ZARC (R || CPE) spectrum."""
    omega = np.logspace(-1, 5, n)
    f = omega / (2.0 * np.pi)
    Z = R / (1.0 + (1j * omega * tau0) ** phi)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        label="zarc",
        # Convention loader/EISSpectrum : Zim = -Im(Z) > 0 (demi-cercle capacitif).
        f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
        concentration=1e-9, step="hybridization",
        n_points=n,
    )


# ── Config synthétique pour la validation KK ────────────────────────────

_DRT_CONFIG = {
    "fit": {
        "drt_wiener_W": 1e-9,
        "drt_n_z": 10000,
    }
}


# ── KK validation ───────────────────────────────────────────────────────

def _rc_spectrum(R: float = 1000.0, C: float = 1e-6, n: int = 30) -> EISSpectrum:
    """30-point log-spaced synthetic R // C spectrum."""
    omega = np.logspace(0, 5, n)
    f = omega / (2.0 * np.pi)
    Z = R / (1.0 + 1j * omega * R * C)
    idx = np.argsort(f)[::-1]
    return EISSpectrum(
        # Convention loader/EISSpectrum : Zim = -Im(Z) > 0.
        label="rc", f=f[idx], Zre=Z.real[idx], Zim=-Z.imag[idx],
        concentration=1e-9, step="hybridization", n_points=n,
    )


def test_kk_validation():
    sp = _rc_spectrum()
    result = kramers_kronig_check(sp, _DRT_CONFIG)
    assert result["kk_passed"]
    assert result["max_residual"] < 0.05


# ── Bout-en-bout : loader → RandlesFullModel().fit (garde-fou du signe B1) ──

def _eclab_csv(Rct: float, n: int = 100) -> bytes:
    """Construit un CSV façon export EC-Lab : colonne '-Im(Z)/Ohm' POSITIVE
    (comme les fichiers réels), pour un spectre Randles de Rct connu."""
    omega = np.logspace(-1, 5, n)
    f = omega / (2.0 * np.pi)
    Z = Z_randles_full(omega, 500.0, 50.0, 1e-9, Rct, 1e-6, 0.90, 0.1, 0.5)
    lines = ["freq/Hz,Re(Z)/Ohm,-Im(Z)/Ohm"]
    for a, b, c in zip(f, Z.real, -Z.imag):  # -Im(Z) > 0
        lines.append(f"{a:.6e},{b:.6e},{c:.6e}")
    return ("\n".join(lines)).encode()


@pytest.mark.parametrize("Rct_true", [1000.0, 3000.0, 8000.0, 30000.0])
def test_randles_recovers_rct_end_to_end(Rct_true):
    """Chemin réel loader → fit : le Rct ajusté doit retrouver le Rct vrai à ±5 %.

    Garde-fou contre une inversion de signe du résidu imaginaire (B1) :
    load_spectrum produit Zim = -Im(Z) > 0 ; un résidu au mauvais signe fait
    diverger le fit (biais fort ou effondrement sur la borne basse).
    """
    sp = load_spectrum(_eclab_csv(Rct_true), label="e2e")
    assert np.all(sp.Zim >= 0), "le loader doit produire Zim positif"
    # Spectre unique (pas de réplicats) → pondération par la structure d'erreur
    # d'Orazem réutilisée (fixture conftest, source="reused_persisted").
    result = RandlesFullModel().fit(sp, {"fit": {"max_iter": 10000}})
    assert result.error_structure_source == "reused_persisted"
    rel_err = abs(result.Rct - Rct_true) / Rct_true
    assert rel_err < 0.05, f"Rct={result.Rct:.1f} vs {Rct_true:.1f} (rel_err={rel_err:.2%})"


# ── Écarts-types des paramètres : cohérence avec scipy.optimize.curve_fit ──

def test_randles_std_matches_curve_fit():
    """Les σ des paramètres coïncident avec curve_fit (absolute_sigma=True).

    Pondération UNIQUE (structure d'erreur d'Orazem) → poids = 1/σ² et covariance
    cov = (JᵀJ)⁻¹ SANS rééchelonnement. On injecte des poids explicites w = 1/σ²
    et on compare aux σ de curve_fit sur le même modèle empilé avec sigma = σ et
    absolute_sigma=True (pas de mise à l'échelle par χ² réduit). Ce test verrouille
    l'ABSENCE de rééchelonnement : toute réintroduction d'un facteur 2·cost/dof
    ferait diverger les σ.

    Régime bien conditionné : comparaison sur Rct/Qdl/α (demi-cercle). Re et Cb ne
    sont pas comparés (inv(JᵀJ) maison vs SVD de curve_fit divergent pour ces
    paramètres légèrement colinéaires, indépendamment du point testé).
    """
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(0)
    true = dict(Re=500.0, Re_prime=200.0, Cb=2e-8, Rct=5000.0,
                Qdl=1e-6, alpha=0.85, R_D=2000.0, tau_d=50.0)

    n = 120
    f = np.logspace(-3, 5, n)                  # 1e-3 → 1e5 Hz
    omega = 2.0 * np.pi * f
    Z = Z_randles_full(omega, *[true[k] for k in _PARAM_NAMES])

    sigma = 0.01 * np.abs(Z) + 1.0             # structure d'erreur (σ connu)
    Zre = Z.real + rng.normal(0.0, sigma)
    Zim = -Z.imag + rng.normal(0.0, sigma)     # convention Zim = -Im(Z) > 0

    idx = np.argsort(f)[::-1]                   # stockage HF→BF
    sp = EISSpectrum(
        label="cov", f=f[idx], Zre=Zre[idx], Zim=Zim[idx],
        concentration=1e-9, step="hybridization", n_points=n,
    )

    # Poids explicites w = 1/σ² (bypass de la résolution de structure d'erreur) :
    # le test isole la covariance NON rééchelonnée.
    w = 1.0 / sigma[idx] ** 2
    result = RandlesFullModel().fit(sp, {"fit": {"max_iter": 20000}}, weights=(w, w))
    assert result.converged

    # ── Référence curve_fit : sigma = σ, absolute_sigma=True (aucune mise à l'échelle) ──
    omega_sp = 2.0 * np.pi * sp.f
    sigma_stacked = np.concatenate([sigma[idx], sigma[idx]])
    ydata = np.concatenate([sp.Zre, sp.Zim])
    xdata = np.arange(ydata.size)             # abscisse factice (modèle empilé)

    def model_stacked(_x, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d):
        Zm = Z_randles_full(omega_sp, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d)
        return np.concatenate([Zm.real, -Zm.imag])

    p0 = [result.params[k] for k in _PARAM_NAMES]
    _popt, pcov = curve_fit(
        model_stacked, xdata, ydata, p0=p0,
        sigma=sigma_stacked, absolute_sigma=True, maxfev=200000,
    )
    std_scipy = dict(zip(_PARAM_NAMES, np.sqrt(np.diag(pcov))))

    for k in ("Rct", "Qdl", "alpha"):
        s_model = result.params_std[k]
        s_scipy = std_scipy[k]
        assert s_scipy > 0
        rel = abs(s_model - s_scipy) / s_scipy
        assert rel < 0.05, (
            f"σ({k}) maison={s_model:.4g} vs curve_fit={s_scipy:.4g} "
            f"(écart={rel:.1%})"
        )


def test_average_replicates_fills_sigma():
    """average_replicates renseigne σ_re/σ_im BRUTS (>1 réplicat) et None sinon.

    Le plancher relatif arbitraire (0.001·|Z̄|) a été SUPPRIMÉ : σ est l'écart-type
    inter-réplicats brut (ddof=1). n_replicates et replicates sont renseignés pour
    la caractérisation de la structure d'erreur (dont l'option voigt_based).
    """
    sp0 = _randles_spectrum(Rct=5000.0)

    # Un seul réplicat → pas de σ.
    single = average_replicates([sp0])
    assert single.sigma_re is None and single.sigma_im is None

    # Trois réplicats bruités → σ renseigné, positif, même longueur que la grille.
    rng = np.random.default_rng(3)
    reps = []
    for _ in range(3):
        sp = _randles_spectrum(Rct=5000.0)
        noise = 0.01 * np.abs(sp.Zre + 1j * sp.Zim)
        sp.Zre = sp.Zre + rng.normal(0.0, noise)
        sp.Zim = sp.Zim + rng.normal(0.0, noise)
        reps.append(sp)
    avg = average_replicates(reps)
    assert avg.sigma_re is not None and avg.sigma_im is not None
    assert len(avg.sigma_re) == len(avg.f) == len(avg.sigma_im)
    assert np.all(avg.sigma_re > 0) and np.all(avg.sigma_im > 0)
    assert avg.n_replicates == 3
    assert avg.replicates is not None and len(avg.replicates) == 3
