"""Fit « méthode Orazem » sur circuit LIBRE (fits/orazem_fit.py) — remplace test_fits.py.

Le circuit de Randles n'est plus codé en dur : il est RECONSTRUIT par
circuit/parser.py à partir de son expression (docs/CIRCUIT_UTILISATEUR.md §3).
Garanties reconduites de l'ancien moteur (test_fits.py, ancien
test_measurement_model.py) :

  * récupération de Rct à quelques % près, du fichier EC-Lab au fit ;
  * χ²ᵣ ≈ 1 avec σ connu, χ²ᵣ ≫ 1 (et alerte) si σ est sous-estimée ;
  * covariance NON rééchelonnée (absolute_sigma), cohérente avec curve_fit ;
  * convention de signe Zim = −Im(Z) > 0 de bout en bout (bug B1) ;
  * refus explicite quand la structure d'erreur n'est pas caractérisable.

Et les correctifs FIT-1 à FIT-5 de AUDIT.md §5.4, l'agrégation par réplicat.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from circuit import parse_circuit
from core.loader import load_spectrum
from core.measurement_model import ErrorStructure, ErrorStructureUnavailable, analyze_replicates
from fits.orazem_fit import (
    FitOptions,
    FitSpecificationError,
    ParameterSpec,
    aggregate_parameter,
    fit_replicate_group,
    fit_spectrum,
    normalize_specs,
)
from fits.physics import Z_randles_full
from tests.synthetic_data import (
    ORAZEM_NOISE,
    eclab_bytes,
    orazem_noisy_arrays,
    orazem_replicates,
    orazem_sigma,
)

RANDLES = ("Re + parallel(Re_prime + parallel(R(Rct) + ZD_bounded(R_D, tau_d), "
           "Q(Qdl, alpha)), C(Cb))")
Z_FUNC, NAMES = parse_circuit(RANDLES)

# Guess et bornes « saisis par l'utilisateur » : volontairement à côté (facteurs 2 à 5).
SPECS = {
    "Re": ParameterSpec(100.0, 0.0, 1e5),
    "Re_prime": ParameterSpec(50.0, 0.0, 1e5),
    "Rct": ParameterSpec(1500.0, 0.0, 1e9),
    "R_D": ParameterSpec(300.0, 0.0, 1e7),
    "tau_d": ParameterSpec(0.2, 1e-6, 1e4),
    "Qdl": ParameterSpec(5e-6, 0.0, 1e-2),
    "alpha": ParameterSpec(0.8, 0.3, 1.0),
    "Cb": ParameterSpec(3e-9, 0.0, 1e-3),
}
_RCT = 3000.0


def _sigma(zre, zim, **noise):
    s = orazem_sigma(zre, zim, **dict(ORAZEM_NOISE, **noise))
    return s, s


def _structure(**noise):
    """Structure d'erreur « connue » (celle du générateur) — σ exact, pour isoler le fit."""
    nz = dict(ORAZEM_NOISE, **noise)
    return ErrorStructure(alpha=nz["alpha"], beta=nz["beta"], gamma=0.0, delta=nz["delta"],
                          R_sol=200.0, n_replicates=5, dof=10 ** 6)


# ═════════════════════════════════════════════════════════════════════════════
# Circuit reconstruit
# ═════════════════════════════════════════════════════════════════════════════

def test_randles_is_rebuilt_by_the_parser_not_hard_coded():
    assert NAMES == ["Re", "Re_prime", "Rct", "R_D", "tau_d", "Qdl", "alpha", "Cb"]
    w = 2 * np.pi * np.logspace(-2, 5, 30)
    p = dict(Re=500.0, Re_prime=200.0, Cb=2e-8, Rct=5000.0, Qdl=1e-6, alpha=0.85, R_D=2000.0, tau_d=50.0)
    np.testing.assert_allclose(Z_FUNC(w, **p), Z_randles_full(w, **p), rtol=1e-12)


# ═════════════════════════════════════════════════════════════════════════════
# Garanties reconduites
# ═════════════════════════════════════════════════════════════════════════════

def _eclab_csv(Rct: float, n: int = 100) -> bytes:
    """Export EC-Lab : colonne '-Im(Z)/Ohm' POSITIVE, comme les fichiers réels."""
    omega = np.logspace(-1, 5, n)
    f = omega / (2.0 * np.pi)
    Z = Z_randles_full(omega, 500.0, 50.0, 1e-9, Rct, 1e-6, 0.90, 0.3 * Rct, 0.5)
    return eclab_bytes(f, Z.real, -Z.imag)


@pytest.mark.parametrize("Rct_true", [1000.0, 3000.0, 8000.0, 30000.0])
def test_rct_recovered_end_to_end_through_the_loader(Rct_true):
    """Chemin réel loader → fit : garde-fou du signe (B1). ±5 % comme avant ; en
    pratique l'écart est bien plus petit sur ces données exactes."""
    sp = load_spectrum(_eclab_csv(Rct_true), label="e2e")
    assert np.all(sp.Zim >= 0), "le loader doit produire Zim positif"
    specs = dict(SPECS, Rct=ParameterSpec(Rct_true / 2, 0.0, 1e9), R_D=ParameterSpec(Rct_true / 10, 0.0, 1e7))
    fr = fit_spectrum(Z_FUNC, NAMES, sp.f, sp.Zre, sp.Zim, *_sigma(sp.Zre, sp.Zim), specs, "Rct")
    assert fr.converged
    assert fr.target_param == "Rct" and fr.target_value == fr.params["Rct"]
    rel_err = abs(fr.target_value - Rct_true) / Rct_true
    assert rel_err < 0.05, f"Rct={fr.target_value:.1f} vs {Rct_true:.1f} (rel_err={rel_err:.2%})"


def test_chi2_reduced_about_one_with_known_sigma():
    chi2 = []
    for seed in range(6):
        f, zre, zim, s = orazem_noisy_arrays(_RCT, seed, n_points=60)
        fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "Rct",
                          options=FitOptions(n_starts=3))
        assert fr.converged
        chi2.append(fr.chi2_reduced)
    assert 0.85 < float(np.mean(chi2)) < 1.15, chi2


def test_chi2_reduced_far_from_one_when_sigma_is_underestimated():
    """Contre-épreuve : σ 10× trop petite → χ²ᵣ ≈ 100 et alerte d'adéquation."""
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0, n_points=60)
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s / 10, s / 10, SPECS, "Rct",
                      options=FitOptions(n_starts=2))
    assert fr.chi2_reduced > 10.0
    assert any(w.startswith("χ²ᵣ =") and "sous-ajustement" in w for w in fr.warnings)


# Jeu BIEN conditionné de l'ancien test de covariance (test_fits.py) : Cb = 20 nF met
# le coude du contournement DANS la fenêtre 1 mHz – 100 kHz. Sur le jeu de l'Annexe A
# (Cb = 1 nF, coude ≈ 800 kHz), Re, R'e et Cb ne sont pas séparément identifiables
# (σ(Re) ≈ 360 Ω, Cb en butée à 0) : y comparer deux covariances n'aurait pas de sens.
_WELL = dict(Re=500.0, Re_prime=200.0, Cb=2e-8, Rct=5000.0, Qdl=1e-6, alpha=0.85, R_D=2000.0, tau_d=50.0)
_WELL_SPECS = {k: ParameterSpec(v * 1.3 if k != "alpha" else 0.8, 0.0 if k != "alpha" else 0.3,
                                np.inf if k != "alpha" else 1.0) for k, v in _WELL.items()}


def _well_conditioned(seed, n=80):
    f = np.logspace(5, -3, n)
    z = Z_FUNC(2 * np.pi * f, **_WELL)
    s = orazem_sigma(z.real, -z.imag, alpha=0.01, beta=0.01, delta=2.0, R_sol=500.0)
    rng = np.random.default_rng(seed)
    return f, z.real + rng.normal(0, s), -z.imag + rng.normal(0, s), s


def test_covariance_is_absolute_and_matches_curve_fit():
    """absolute_sigma : doubler σ double exactement les écarts-types (même optimum) —
    un rééchelonnement par χ²ᵣ les laisserait inchangés. Et accord avec curve_fit
    (absolute_sigma=True, sans bornes) sur un problème bien conditionné."""
    f, zre, zim, s = _well_conditioned(1)
    fr1 = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, _WELL_SPECS, "Rct",
                       options=FitOptions(n_starts=2))
    fr2 = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, 2 * s, 2 * s, _WELL_SPECS, "Rct",
                       options=FitOptions(n_starts=2))
    assert fr1.converged and not any("butée" in w for w in fr1.warnings)
    for k in NAMES:
        assert fr2.params[k] == pytest.approx(fr1.params[k], rel=1e-5)
        assert fr2.params_std[k] == pytest.approx(2 * fr1.params_std[k], rel=1e-3)
    assert fr2.chi2_reduced == pytest.approx(fr1.chi2_reduced / 4, rel=1e-5)

    omega = 2 * np.pi * f

    def stacked(_x, *p):
        Zm = Z_FUNC(omega, **dict(zip(NAMES, p)))
        return np.concatenate([Zm.real, -Zm.imag])

    _popt, pcov = curve_fit(stacked, np.arange(2 * f.size), np.concatenate([zre, zim]),
                            p0=[fr1.params[k] for k in NAMES], sigma=np.concatenate([s, s]),
                            absolute_sigma=True, maxfev=200000)
    ref = dict(zip(NAMES, np.sqrt(np.diag(pcov))))
    for k in NAMES:
        assert fr1.params_std[k] == pytest.approx(ref[k], rel=0.05), k


def test_imaginary_sign_convention_is_preserved():
    """Zim = −Im(Z) > 0 de bout en bout : Zfit_im positif sur l'arc capacitif, résidu
    = données − modèle dans CETTE convention, (−Im Ẑ − Zim)² ≡ (Zim + Im Ẑ)²."""
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 2)
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "Rct", options=FitOptions(n_starts=2))
    Zfit = Z_FUNC(2 * np.pi * f, **fr.params)
    np.testing.assert_allclose(fr.Zfit_im, -Zfit.imag)
    assert np.all(fr.Zfit_im[zim > 50] > 0)
    np.testing.assert_allclose(fr.residuals_im, zim - fr.Zfit_im)
    np.testing.assert_allclose((-Zfit.imag - zim) ** 2, (zim + Zfit.imag) ** 2)


def test_wrong_sign_convention_cannot_be_fitted():
    """Garde-fou B1 : un Zim au signe physique (négatif) ne peut pas être reproduit par
    un circuit passif — le fit le crie au lieu de rendre un Rct plausible."""
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 3)
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, -zim, s, s, SPECS, "Rct", options=FitOptions(n_starts=2))
    assert fr.chi2_reduced > 100 and any("résidu relatif élevé" in w for w in fr.warnings)


def test_no_fit_without_a_characterizable_error_structure():
    """ERR-1 : sans structure d'erreur, l'analyse du groupe s'ARRÊTE (exception à
    message utilisateur) avant tout fit ; aucun FitResult « vide » n'est produit."""
    reps = orazem_replicates(_RCT, n_rep=2)
    produced = []
    with pytest.raises(ErrorStructureUnavailable) as exc:
        analysis = analyze_replicates(reps, label="c1")
        produced.append(fit_replicate_group(Z_FUNC, NAMES, reps, analysis, SPECS, "Rct"))
    assert produced == []
    assert "analyse Orazem interrompue" in exc.value.user_message


# ═════════════════════════════════════════════════════════════════════════════
# Spécifications par paramètre (interface)
# ═════════════════════════════════════════════════════════════════════════════

def test_specs_accept_three_forms_and_default_scale():
    specs = normalize_specs(["a", "b", "c"], {
        "a": ParameterSpec(2.0, 0.0, 10.0),
        "b": (-3.0, -10.0, 10.0),
        "c": {"initial": 0.0, "lower": -5.0, "upper": 1e3},
    })
    assert specs["a"].resolved_scale() == 2.0
    assert specs["b"] == ParameterSpec(-3.0, -10.0, 10.0)
    assert specs["c"].resolved_scale() == 1e3                    # initial nul → plus grande borne


@pytest.mark.parametrize("specs, message", [
    ({k: v for k, v in SPECS.items() if k != "Cb"}, "sans guess/bornes : Cb"),
    (dict(SPECS, Xyz=(1.0, 0.0, 2.0)), "inconnus du circuit : Xyz"),
    (dict(SPECS, Rct=(5.0, 10.0, 20.0)), "hors des bornes"),
    (dict(SPECS, Rct=(5.0, 20.0, 10.0)), "lower < upper"),
    (dict(SPECS, Rct=(np.inf, 0.0, np.inf)), "doit être fini"),
    (dict(SPECS, Rct=(True, 0.0, 2.0)), "nombre réel"),
    (dict(SPECS, Rct={"initial": 1.0, "hi": 2.0}), "clés attendues"),
])
def test_invalid_specs_are_refused_with_a_clear_message(specs, message):
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    with pytest.raises(FitSpecificationError, match=message):
        fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, specs, "Rct")


def test_target_must_be_a_circuit_parameter_and_z_finite_at_the_guess():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    with pytest.raises(FitSpecificationError, match="paramètre cible « R_ct » absent"):
        fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "R_ct")
    with pytest.raises(FitSpecificationError, match="pas fini au guess"):
        fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, dict(SPECS, Qdl=(0.0, 0.0, 1.0)), "Rct")


# ═════════════════════════════════════════════════════════════════════════════
# FIT-1 : non convergé ≠ erreur de programmation
# ═════════════════════════════════════════════════════════════════════════════

def test_non_convergence_returns_the_best_iterate_with_an_explicit_warning():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "Rct",
                      options=FitOptions(n_starts=1, max_nfev=3))
    assert fr.converged is False
    assert any(w.startswith("ajustement NON convergé") for w in fr.warnings)
    x0 = np.array([SPECS[k].initial for k in NAMES])
    Z0 = Z_FUNC(2 * np.pi * f, **dict(zip(NAMES, x0)))
    chi2_guess = np.sum(((Z0.real - zre) / s) ** 2 + ((-Z0.imag - zim) / s) ** 2)
    assert fr.chi2_reduced * fr.fit_diagnostics["dof"] <= chi2_guess     # jamais pire que le guess


def test_a_programming_error_propagates():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    calls = {"n": 0}

    def buggy(w, **p):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyError("bug dans le circuit")
        return Z_FUNC(w, **p)

    with pytest.raises(KeyError, match="bug dans le circuit"):
        fit_spectrum(buggy, NAMES, f, zre, zim, s, s, SPECS, "Rct")


# ═════════════════════════════════════════════════════════════════════════════
# FIT-2 : covariance par SVD, conditionnement, identifiabilité
# ═════════════════════════════════════════════════════════════════════════════

def test_non_identifiable_parameters_get_infinite_std_not_a_fabricated_one():
    Z, names = parse_circuit("R(Ra) + R(Rb) + parallel(R(Rct), C(Cdl))")
    f = np.logspace(-1, 5, 40)
    z = Z(2 * np.pi * f, Ra=100.0, Rb=50.0, Rct=2000.0, Cdl=1e-6)
    s = 0.005 * np.abs(z)
    specs = {"Ra": (80.0, 0.0, 1e4), "Rb": (80.0, 0.0, 1e4), "Rct": (1000.0, 0.0, 1e6),
             "Cdl": (2e-6, 0.0, 1.0)}
    fr = fit_spectrum(Z, names, f, z.real, -z.imag, s, s, specs, "Rct", options=FitOptions(n_starts=1))
    assert fr.params_std["Ra"] == np.inf and fr.params_std["Rb"] == np.inf
    assert fr.fit_diagnostics["identifiable"] == {"Ra": False, "Rb": False, "Rct": True, "Cdl": True}
    assert fr.fit_diagnostics["condition_number"] == np.inf
    assert any("non identifiable" in w and "Ra, Rb" in w for w in fr.warnings)
    assert np.isfinite(fr.target_std) and fr.target_value == pytest.approx(2000.0, rel=1e-6)
    assert fr.params["Ra"] + fr.params["Rb"] == pytest.approx(150.0, rel=1e-6)


def test_condition_number_is_reported_on_the_equilibrated_jacobian():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "Rct", options=FitOptions(n_starts=2))
    cond = fr.fit_diagnostics["condition_number"]
    assert 1.0 < cond < 1e8            # unités (pF à côté de kΩ) éliminées par l'équilibrage
    assert fr.fit_diagnostics["rank"] == len(NAMES)


# ═════════════════════════════════════════════════════════════════════════════
# FIT-3 : x_scale et départs multiples ; choix TRF / LM
# ═════════════════════════════════════════════════════════════════════════════

def test_multistart_is_reproducible_and_keeps_the_best_chi2():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 5)
    poor = dict(SPECS, Rct=ParameterSpec(100.0, 0.0, 1e9), Qdl=ParameterSpec(1e-4, 0.0, 1e-2),
                R_D=ParameterSpec(10.0, 0.0, 1e7))
    a = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, poor, "Rct", options=FitOptions(n_starts=8, seed=3))
    b = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, poor, "Rct", options=FitOptions(n_starts=8, seed=3))
    assert a.params == b.params                                           # graine explicite
    d = a.fit_diagnostics
    assert d["n_starts"] == 8 and len(d["starts"]) == 8
    best = min(st["chi2"] for st in d["starts"] if st["converged"])
    assert d["chi2"] == pytest.approx(best, rel=1e-9)
    assert a.target_value == pytest.approx(_RCT, rel=0.03)
    assert d["x_scale"]["Qdl"] == 1e-4 and d["x_scale"]["Rct"] == 100.0  # |guess| par paramètre


def test_solver_is_trf_with_bounds_and_lm_without():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    unbounded = {k: ParameterSpec(v.initial) for k, v in SPECS.items()}
    assert fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "Rct",
                        options=FitOptions(n_starts=1)).fit_diagnostics["method"] == "trf"
    assert fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, unbounded, "Rct",
                        options=FitOptions(n_starts=1)).fit_diagnostics["method"] == "lm"


# ═════════════════════════════════════════════════════════════════════════════
# FIT-4 : paramètre en butée, relativement à son incertitude
# ═════════════════════════════════════════════════════════════════════════════

def test_a_parameter_constrained_by_its_bound_is_flagged():
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    specs = dict(SPECS, Rct=ParameterSpec(1500.0, 0.0, 0.8 * _RCT))
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, specs, "Rct", options=FitOptions(n_starts=2))
    assert fr.params["Rct"] == pytest.approx(0.8 * _RCT, rel=1e-6)
    assert any(w.startswith("Rct en butée haute") for w in fr.warnings)


def test_bound_proximity_is_judged_in_standard_deviations_not_in_bound_units():
    """L'ancien seuil (1 % de |borne|) ne voulait rien dire pour une borne 0 ou 1e-12.
    Ici : Qdl = 2 µF est à plus de 10 σ de sa borne 0 → pas d'alerte ; Cb (1 nF, coude
    hors fenêtre) est mal déterminé, son intervalle à 1σ touche 0 → alerte, car la
    contrainte est active et l'écart-type gaussien n'a plus de sens."""
    f, zre, zim, s = orazem_noisy_arrays(_RCT, 0)
    fr = fit_spectrum(Z_FUNC, NAMES, f, zre, zim, s, s, SPECS, "Rct", options=FitOptions(n_starts=2))
    assert fr.params["Qdl"] > 10 * fr.params_std["Qdl"]
    assert not any(w.startswith("Qdl en butée") for w in fr.warnings)
    assert fr.params["Cb"] < fr.params_std["Cb"]
    assert any(w.startswith("Cb en butée basse") for w in fr.warnings)


# ═════════════════════════════════════════════════════════════════════════════
# Par réplicat + agrégation
# ═════════════════════════════════════════════════════════════════════════════

def test_aggregate_parameter_formulas():
    agg = aggregate_parameter("Rct", [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
    assert agg.mean == 2.0 and agg.std_between == pytest.approx(1.0)
    assert agg.std_within == pytest.approx(0.1) and agg.sem_within == pytest.approx(np.sqrt(0.03) / 3)
    assert agg.sem == pytest.approx(np.sqrt(1.0 / 3))                  # dispersion observée
    assert agg.q == pytest.approx(200.0) and agg.q_pvalue < 1e-10     # Cochran

    tight = aggregate_parameter("Rct", [1.0, 1.01, 0.99], [0.1, 0.1, 0.1])
    assert tight.std_between == pytest.approx(0.01)
    assert tight.sem == pytest.approx(np.sqrt(0.01 / 3))              # plancher du bruit
    assert tight.q_pvalue > 0.5

    one = aggregate_parameter("Rct", [5.0], [0.2], n_excluded=2)
    assert one.n == 1 and np.isnan(one.std_between) and one.sem == pytest.approx(0.2)
    assert one.n_excluded == 2
    assert aggregate_parameter("Rct", [], []).n == 0


@pytest.fixture(scope="module")
def stationary_group():
    # Graine 600 écartée : fausse alarme KK (risque ≈ 1-4 % mesuré, voir
    # test_measurement_model) — ce test porte sur le fit, pas sur le verdict.
    reps = orazem_replicates(_RCT, n_rep=3, seed0=610)
    analysis = analyze_replicates(reps, label="c1")
    return reps, analysis, fit_replicate_group(Z_FUNC, NAMES, reps, analysis, SPECS, "Rct")


def test_every_replicate_and_the_mean_are_fitted_then_aggregated(stationary_group):
    reps, analysis, res = stationary_group
    assert analysis.kk_conform is True and res.warnings == []
    assert len(res.replicate_fits) == 3 and all(fr.converged for fr in res.replicate_fits)
    assert [fr.fit_diagnostics["label"] for fr in res.replicate_fits] == ["r0", "r1", "r2"]
    for fr in res.replicate_fits + [res.mean_fit]:
        assert fr.target_param == "Rct" and fr.target_value == pytest.approx(_RCT, rel=0.05)
        assert fr.error_structure_source == "characterized_now"
        assert fr.error_structure_coeffs["n_replicates"] == 3
    # Le spectre moyen, 3× moins bruité (σ/√3), est plus précis qu'un réplicat.
    assert res.mean_fit.target_std < min(fr.target_std for fr in res.replicate_fits)
    t = res.target
    assert t.n == 3 and t.mean == pytest.approx(np.mean([fr.target_value for fr in res.replicate_fits]))
    v_bar = np.mean([fr.target_std ** 2 for fr in res.replicate_fits])
    assert t.sem == pytest.approx(np.sqrt(max(t.std_between ** 2, v_bar) / 3))
    assert set(res.aggregate) == set(NAMES)


def test_chi2_interval_accounts_for_the_estimated_sigma(stationary_group):
    """σ estimée sur ν_σ ddl → χ²ᵣ ~ F(ν, ν_σ) : intervalle plus large que χ²(ν)/ν."""
    from scipy import stats

    _reps, _analysis, res = stationary_group
    fr = res.replicate_fits[0]
    nu = fr.fit_diagnostics["dof"]
    lo, hi = fr.chi2_reduced_ci
    assert fr.fit_diagnostics["dof_sigma"] == res.analysis.error_structure.dof
    assert hi > stats.chi2.ppf(0.97725, nu) / nu and lo < stats.chi2.ppf(0.02275, nu) / nu


def test_heterogeneous_replicates_are_flagged_and_their_spread_is_kept():
    """Rct différent d'un réplicat à l'autre (remontage, dérive ENTRE balayages) : chaque
    spectre reste KK-conforme, mais la dispersion dépasse l'incertitude intra-fit."""
    reps = [orazem_replicates(r, n_rep=1, seed0=700 + k)[0] for k, r in enumerate((2600.0, 3000.0, 3500.0))]
    for k, sp in enumerate(reps):
        sp.label = f"r{k}"
    analysis = analyze_replicates(reps, label="hetero")
    res = fit_replicate_group(Z_FUNC, NAMES, reps, analysis, SPECS, "Rct",
                              options=FitOptions(n_starts=3))
    t = res.target
    assert t.q_pvalue < 0.05 and t.std_between > 5 * t.std_within
    assert t.sem == pytest.approx(t.std_between / np.sqrt(3))
    assert any("dispersion inter-réplicats" in w and "Cochran" in w for w in res.warnings)


def test_kk_verdict_travels_with_the_fits():
    reps = orazem_replicates(_RCT, n_rep=3, seed0=800)
    drifting = orazem_replicates(_RCT, n_rep=1, seed0=805, drift=0.25)[0]
    drifting.label = "r1"
    reps[1] = drifting
    analysis = analyze_replicates(reps, label="derive")
    assert analysis.kk_conform is False
    res = fit_replicate_group(Z_FUNC, NAMES, reps, analysis, SPECS, "Rct",
                              options=FitOptions(n_starts=2))
    assert analysis.kk_message in res.warnings
    assert any("réplicat non conforme Kramers-Kronig" in w for w in res.replicate_fits[1].warnings)


def test_group_fit_refuses_replicates_that_were_not_analysed(stationary_group):
    reps, analysis, _res = stationary_group
    with pytest.raises(ValueError, match="ne sont pas ceux de l'analyse"):
        fit_replicate_group(Z_FUNC, NAMES, reps[:2], analysis, SPECS, "Rct")


def test_non_converged_replicates_are_excluded_from_the_aggregate(stationary_group):
    reps, analysis, _res = stationary_group
    res = fit_replicate_group(Z_FUNC, NAMES, reps, analysis, SPECS, "Rct",
                              options=FitOptions(n_starts=1, max_nfev=3))
    assert not any(fr.converged for fr in res.replicate_fits)
    assert res.target.n == 0 and res.target.n_excluded == 3
    assert "3 réplicat(s) non convergé(s) écarté(s) de l'agrégation." in res.warnings
