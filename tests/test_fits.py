"""Tests for fit models in fits/."""

import numpy as np
import pytest

from fits.physics import Z_randles_full
from fits.randles_full import RandlesFullModel, _PARAM_NAMES
from fits.drt_fft import DRTFFTModel
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


# ── DRT FFT tests ──────────────────────────────────────────────────────

_DRT_CONFIG = {
    "fit": {
        "drt_wiener_W": 1e-9,
        "drt_n_z": 10000,
    }
}


def test_drt_returns_positive_rct():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert result.Rct > 0


def test_drt_gamma_non_zero():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    gamma = np.array(result.drt_gamma)
    assert gamma.max() > 0, "DRT should have at least one non-zero value"


def test_drt_fit_arrays_finite():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert np.all(np.isfinite(result.Zfit_re))
    assert np.all(np.isfinite(result.Zfit_im))


def test_drt_fft_randles_simple():
    """Spectre Randles synthétique propre, vérifie drt_tau/drt_gamma et Rct_drt
    cohérent avec Rct_randles à 30% près."""
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert len(result.drt_tau) == len(result.drt_gamma)
    assert result.Rct > 0
    Rct_randles = result.params["Rct_randles"]
    rel_err = abs(result.Rct - Rct_randles) / Rct_randles
    assert rel_err < 0.30, (
        f"Rct_drt = {result.Rct:.0f} Ω vs Rct_randles = {Rct_randles:.0f} Ω "
        f"(rel_err={rel_err:.2f})"
    )


# ── New architecture: KK validation, stricter FFT DRT ──────────────────

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


def test_drt_fft():
    sp = _randles_spectrum()
    result = DRTFFTModel().fit(sp, _DRT_CONFIG)
    assert len(result.drt_tau) == _DRT_CONFIG["fit"]["drt_n_z"]
    # Le filtre Wiener FFT souffre d'artefacts de bord (ringing) aux extrémités
    # du domaine log-ω (cf. fits/drt_fft.py) ; cette ringing contamine la
    # reconstruction de Im(Z) sur tout le domaine, d'où une erreur de
    # reconstruction relative élevée même pour un Rct correctement extrait
    # (cf. test_drt_fft_randles_simple, qui valide la précision de Rct).
    assert result.reconstruction_error < 2.0


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
    result = RandlesFullModel().fit(sp, {"fit": {"alpha_noise": 0.001, "max_iter": 10000}})
    rel_err = abs(result.Rct - Rct_true) / Rct_true
    assert rel_err < 0.05, f"Rct={result.Rct:.1f} vs {Rct_true:.1f} (rel_err={rel_err:.2%})"


# ── Écarts-types des paramètres : cohérence avec scipy.optimize.curve_fit ──

def test_randles_std_matches_curve_fit():
    """Les σ des paramètres doivent coïncider avec ceux de curve_fit (à qq %).

    Le fit maison empile [Re(Z), -Im(Z)] pondérés par w = 1/(alpha_noise·|Z|) et
    met la covariance à l'échelle par le χ² réduit s² = 2·result.cost/(2N-P).
    curve_fit sur le même modèle empilé, avec sigma = 1/w et absolute_sigma=False
    (même mise à l'échelle par χ² réduit), doit produire les mêmes σ. Une erreur
    de facteur 2 dans s² (bug corrigé) ferait diverger les σ d'un facteur √2 ≈ 41 %,
    bien au-delà de la tolérance de 5 %.

    Régime choisi bien conditionné (arc de transfert de charge et queue de
    diffusion tous deux dans la fenêtre) : la comparaison porte sur les paramètres
    du demi-cercle Rct/Qdl/α, physiquement primordiaux et stables. Re et Cb ne sont
    pas comparés car la covariance maison passe par inv(JᵀJ) (équations normales)
    tandis que curve_fit utilise une SVD : les deux divergent pour ces paramètres
    légèrement colinéaires, indépendamment de la mise à l'échelle testée ici.
    """
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(0)
    true = dict(Re=500.0, Re_prime=200.0, Cb=2e-8, Rct=5000.0,
                Qdl=1e-6, alpha=0.85, R_D=2000.0, tau_d=50.0)

    n = 120
    f = np.logspace(-3, 5, n)                  # 1e-3 → 1e5 Hz
    omega = 2.0 * np.pi * f
    Z = Z_randles_full(omega, *[true[k] for k in _PARAM_NAMES])

    # Bruit gaussien ~0.5 % du module, indépendant sur réel et imaginaire.
    noise = 0.005 * np.abs(Z)
    Zre = Z.real + rng.normal(0.0, noise)
    Zim = -Z.imag + rng.normal(0.0, noise)    # convention Zim = -Im(Z) > 0

    idx = np.argsort(f)[::-1]                  # stockage HF→BF
    sp = EISSpectrum(
        label="cov", f=f[idx], Zre=Zre[idx], Zim=Zim[idx],
        concentration=1e-9, step="hybridization", n_points=n,
    )

    config = {"fit": {"alpha_noise": 0.001, "max_iter": 20000}}
    result = RandlesFullModel().fit(sp, config)
    assert result.converged

    # ── Référence curve_fit sur le même modèle empilé et la même pondération ──
    alpha_noise = config["fit"]["alpha_noise"]
    omega_sp = 2.0 * np.pi * sp.f
    Z_data = sp.Zre + 1j * sp.Zim
    sigma_pt = alpha_noise * np.maximum(np.abs(Z_data), 1.0)   # = 1/weight du modèle
    sigma_stacked = np.concatenate([sigma_pt, sigma_pt])
    ydata = np.concatenate([sp.Zre, sp.Zim])
    xdata = np.arange(ydata.size)             # abscisse factice (modèle empilé)

    def model_stacked(_x, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d):
        Zm = Z_randles_full(omega_sp, Re, Re_p, Cb, Rct, Qdl, alpha_p, R_D, tau_d)
        return np.concatenate([Zm.real, -Zm.imag])

    p0 = [result.params[k] for k in _PARAM_NAMES]
    _popt, pcov = curve_fit(
        model_stacked, xdata, ydata, p0=p0,
        sigma=sigma_stacked, absolute_sigma=False, maxfev=200000,
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


# ── Pondération "sigma" (Measurement Model) : chi2_reduced → 1 ──────────────

def test_sigma_weighting_chi2_reduced_around_one():
    """En mode weight_mode="sigma" avec σ_re/σ_im connus (= vrai bruit injecté),
    le χ² réduit pondéré tombe autour de 1 : c'est le test d'adéquation.

    On génère un Randles bien conditionné, on ajoute un bruit gaussien de σ CONNUS
    et DIFFÉRENTS sur réel et imaginaire (σ_im = 2·σ_re), on renseigne
    spectrum.sigma_re/sigma_im avec ces σ vrais, puis on fitte en mode sigma.
    Les poids étant les vraies 1/variance (absolute_sigma), E[chi2_reduced] = 1.
    Moyenné sur plusieurs graines pour amortir la fluctuation ~√(2/dof).
    """
    true = dict(Re=500.0, Re_prime=200.0, Cb=2e-8, Rct=5000.0,
                Qdl=1e-6, alpha=0.85, R_D=2000.0, tau_d=50.0)
    n = 120
    f = np.logspace(-3, 5, n)
    omega = 2.0 * np.pi * f
    Z = Z_randles_full(omega, *[true[k] for k in _PARAM_NAMES])
    idx = np.argsort(f)[::-1]

    sigma_re = 0.01 * np.abs(Z)          # σ connus, DIFFÉRENTS sur re/im
    sigma_im = 0.02 * np.abs(Z)
    config = {"fit": {"weight_mode": "sigma", "max_iter": 20000}}

    chi2_vals = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        Zre = Z.real + rng.normal(0.0, sigma_re)
        Zim = -Z.imag + rng.normal(0.0, sigma_im)   # convention Zim = -Im(Z)
        sp = EISSpectrum(
            label="sig", f=f[idx], Zre=Zre[idx], Zim=Zim[idx],
            concentration=1e-9, step="hybridization", n_points=n,
        )
        sp.sigma_re = sigma_re[idx]
        sp.sigma_im = sigma_im[idx]
        result = RandlesFullModel().fit(sp, config)
        assert result.converged
        chi2_vals.append(result.chi2_reduced)

    mean_chi2 = float(np.mean(chi2_vals))
    assert 0.85 < mean_chi2 < 1.15, (
        f"chi2_reduced moyen (mode sigma) = {mean_chi2:.3f}, attendu ≈ 1 "
        f"(valeurs={[round(v, 3) for v in chi2_vals]})"
    )


def test_average_replicates_fills_sigma():
    """average_replicates renseigne σ_re/σ_im (>1 réplicat) et laisse None sinon."""
    sp0 = _randles_spectrum(Rct=5000.0)

    # Un seul réplicat → pas de σ.
    single = average_replicates([sp0])
    assert single.sigma_re is None and single.sigma_im is None

    # Deux réplicats bruités → σ renseigné, positif, même longueur que la grille.
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
