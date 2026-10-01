"""Measurement model, structure d'erreur et critère KK unique (étape 4).

Remplace l'ancien test_measurement_model.py (qui visait fits/error_structure.py et
sa persistance JSON). Couvre :

  A. le critère UNIQUE de verdict KK (fits/kk_validation.kk_verdict, ERR-6) ;
  B. Lin-KK avec le critère µ de Schönleber effectif (ERR-5) ;
  C. le measurement model de Voigt régressé et son critère de parcimonie ;
  D. la structure d'erreur : estimation sans plancher, corrections de levier et de
     petit n, test de σ_r = σ_j, AUCUNE persistance, refus explicite (ERR-1/ERR-2) ;
  E. la conformité KK par le MÊME measurement model, et son calibrage.

Données : Randles de l'Annexe A (tests/synthetic_data.py), bruit selon la structure
d'Orazem (σ_r = σ_j) sauf mention contraire. Convention de l'app : Zim = −Im(Z) > 0.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import core.measurement_model as mm
from circuit import parse_circuit
from core.measurement_model import (
    ErrorStructure,
    ErrorStructureUnavailable,
    MeasurementModelOptions,
    analyze_replicates,
    characterize_error_structure,
    check_kk_consistency,
    fit_voigt,
)
from fits.kk_validation import (
    KK_FALSE_ALARM,
    SCHONLEBER_C,
    _lin_kk_solve,
    kk_allowed_outside,
    kk_verdict,
    kramers_kronig_check,
    lin_kk,
    schonleber_mu,
)
from tests.synthetic_data import (
    N_POINTS,
    noisy_arrays,
    orazem_noisy_arrays,
    orazem_replicates,
    randles_impedance,
)

_RCT = 3000.0


def _ns(arrays, label="r"):
    """(f, Zre, Zim[, σ]) → objet réplicat minimal."""
    return SimpleNamespace(f=arrays[0], Zre=arrays[1], Zim=arrays[2], label=label)


@pytest.fixture(scope="module")
def estimate5():
    """Structure d'erreur caractérisée sur 5 réplicats (calculée une fois)."""
    return characterize_error_structure(orazem_replicates(_RCT, n_rep=5, seed0=40), label="g5")


# ═════════════════════════════════════════════════════════════════════════════
# A. Critère UNIQUE de verdict KK
# ═════════════════════════════════════════════════════════════════════════════

def test_allowed_count_is_the_binomial_quantile_of_two_sigma_exceedances():
    p0 = 2 * stats.norm.sf(2.0)
    assert p0 == pytest.approx(0.0455, abs=1e-4)
    for n in (20, 80, 120):
        k = kk_allowed_outside(n)
        assert stats.binom.sf(k, n, p0) <= KK_FALSE_ALARM < stats.binom.sf(k - 1, n, p0)
    assert kk_allowed_outside(80) == 7
    assert kk_allowed_outside(0) == 0


def test_verdict_without_noise_level_is_undetermined_not_a_threshold():
    v = kk_verdict(np.ones(10), np.ones(10), None, None)
    assert v.conform is None and v.n_residuals == 20
    assert "indéterminé" in v.message


def test_verdict_counts_two_sigma_exceedances_on_both_components():
    sd = np.ones(40)
    r = np.zeros(40)
    assert kk_verdict(r, r, sd, sd).conform is True
    bad = r.copy()
    bad[:10] = 3.0                                   # 10 résidus à 3σ (> 7 tolérés sur 80)
    v = kk_verdict(bad, r, sd, sd)
    assert v.conform is False and v.n_outside == 10 and v.n_allowed == 7
    assert "NON conforme" in v.message
    assert v.outside_re[:10].all() and not v.outside_im.any()
    assert list(np.flatnonzero(~v.inside_mask)) == list(range(10))


def test_non_finite_residual_counts_as_outside_and_bad_noise_is_rejected():
    r = np.zeros(10)
    r[0] = np.nan
    assert kk_verdict(r, None, np.ones(10), None).n_outside == 1
    with pytest.raises(ValueError, match="non fini ou ≤ 0"):
        kk_verdict(np.zeros(3), None, np.array([1.0, 0.0, 1.0]), None)
    with pytest.raises(ValueError, match="CHAQUE composante"):
        kk_verdict(np.zeros(3), np.zeros(3), np.ones(3), None)


def test_false_alarm_rate_of_the_count_test_is_at_most_alpha_under_h0():
    """z i.i.d. N(0, 1) : le test de comptage rejette au plus alpha (discrétude comprise)."""
    rng = np.random.default_rng(0)
    rejects = sum(
        not kk_verdict(rng.standard_normal(40), rng.standard_normal(40), np.ones(40), np.ones(40)).conform
        for _ in range(2000)
    )
    assert rejects / 2000 <= KK_FALSE_ALARM + 0.01


# ═════════════════════════════════════════════════════════════════════════════
# B. Lin-KK — critère µ de Schönleber EFFECTIF (ERR-5)
# ═════════════════════════════════════════════════════════════════════════════

def test_schonleber_mu_definition():
    assert schonleber_mu(np.array([1.0, 2.0, 3.0])) == 1.0
    assert schonleber_mu(np.array([4.0, -1.0])) == pytest.approx(0.75)   # 1 − 1/4
    assert schonleber_mu(np.array([-1.0, -2.0])) == float("-inf")
    assert SCHONLEBER_C == 0.85


def _asc(f, zre, zim):
    o = np.argsort(f)
    return f[o], zre[o] - 1j * zim[o]


def test_c_now_changes_the_number_of_elements():
    """ERR-5 : l'ancien wrapper ignorait c ; M ne dépendait que du nombre de points."""
    f, Z = _asc(*noisy_arrays(_RCT, 0.005, 1))
    m_default = lin_kk(f, Z).M
    m_strict = lin_kk(f, Z, c=0.3).M
    assert m_default != m_strict
    assert lin_kk(f, Z).M != 2 * N_POINTS // 3            # plus la règle 2n/3 de l'ancien code


def test_m_is_the_start_of_the_last_run_below_c_not_the_first_crossing():
    """Lecture retenue (docstring de fits/kk_validation.py) : sur le Randles de l'app,
    µ(M) plonge TRANSITOIREMENT sous c ; le 1er franchissement laisserait ~3,6 %
    d'erreur sur un spectre conforme (bruit 0,5 %), le dernier la ramène au bruit."""
    f, Z = _asc(*noisy_arrays(_RCT, 0.005, 1))
    omega = 2 * np.pi * f
    mus = [schonleber_mu(_lin_kk_solve(omega, Z, M, True)[0]) for M in range(1, len(f) + 1)]
    first = next(M for M, mu in enumerate(mus, start=1) if mu < SCHONLEBER_C)
    last_above = max(M for M, mu in enumerate(mus, start=1) if mu >= SCHONLEBER_C)

    res = lin_kk(f, Z)
    assert res.M == last_above + 1 and res.mu < SCHONLEBER_C and res.mu_criterion_met
    assert first < res.M

    def rms(M):
        _R, Zf = _lin_kk_solve(omega, Z, M, True)
        return np.sqrt(np.mean(np.abs(Z - Zf) ** 2 / np.abs(Z) ** 2))

    assert rms(first) > 0.02                    # 1er franchissement : sous-ajustement grossier
    assert rms(res.M) < 2 * 0.005               # retenu : au niveau du bruit (0,5 %)


def test_lin_kk_rejects_invalid_inputs():
    f, Z = _asc(*noisy_arrays(_RCT, 0.0))
    with pytest.raises(ValueError, match="au moins 3 points"):
        lin_kk(f[:2], Z[:2])
    with pytest.raises(ValueError, match="c doit être"):
        lin_kk(f, Z, c=1.2)
    with pytest.raises(ValueError, match="strictement croissantes"):
        lin_kk(f[::-1], Z[::-1])
    with pytest.raises(ValueError, match="fini et non nul"):
        lin_kk(f, np.where(np.arange(len(f)) == 3, 0.0, Z))


def test_kramers_kronig_check_gives_a_verdict_only_with_an_error_structure(estimate5):
    f, zre, zim, _ = orazem_noisy_arrays(_RCT, 60)
    sp = SimpleNamespace(f=f, Zre=zre, Zim=zim, n_replicates=None)
    no_sigma = kramers_kronig_check(sp)
    assert no_sigma["kk_passed"] is None and "indéterminé" in no_sigma["message"]
    with_sigma = kramers_kronig_check(sp, error_structure=estimate5.error_structure)
    assert with_sigma["kk_passed"] is True
    # Convention de l'app conservée : Z_kk_im > 0 et résidu = données − modèle.
    assert np.all(with_sigma["Z_kk_im"][zim > 1] > 0)
    np.testing.assert_allclose(with_sigma["residuals_im"], zim - with_sigma["Z_kk_im"])


# ═════════════════════════════════════════════════════════════════════════════
# C. Measurement model de Voigt régressé — critère de parcimonie
# ═════════════════════════════════════════════════════════════════════════════

def _two_rc(noise=1e-3, seed=0):
    Z, _ = parse_circuit("R0 + parallel(R(R1), C(C1)) + parallel(R(R2), C(C2))")
    f = np.logspace(-1, 5, 50)
    z = Z(2 * np.pi * f, R0=50.0, R1=500.0, C1=1e-6, R2=2000.0, C2=1e-3)
    sd = noise * np.abs(z)
    rng = np.random.default_rng(seed)
    return f, z.real + rng.normal(0, sd), -z.imag + rng.normal(0, sd), sd


def test_voigt_recovers_a_true_two_element_voigt_circuit():
    """Un vrai circuit de Voigt à 2 éléments : K = 2 et τ, R retrouvés (τ2 = 2 s est
    même au-delà de 1/ω_min = 1,6 s, là où Lin-KK à τ fixés échoue)."""
    f, zr, zj, sd = _two_rc()
    m = fit_voigt(f, zr, zj, sd, sd, absolute_sigma=True)
    assert m.n_elements == 2
    np.testing.assert_allclose(m.tau, [500.0 * 1e-6, 2000.0 * 1e-3], rtol=0.01)
    np.testing.assert_allclose(m.R, [500.0, 2000.0], rtol=0.01)
    assert m.R0 == pytest.approx(50.0, rel=0.02)
    assert 0.5 < m.chi2_reduced < 1.6


def test_parsimony_history_records_the_f_test_and_the_significance_rule():
    f, zre, zim, sd = orazem_noisy_arrays(_RCT, 3)
    m = fit_voigt(f, zre, zim, sd, sd, absolute_sigma=True)
    hist = m.history
    assert [h["K"] for h in hist] == list(range(1, len(hist) + 1))
    for h in hist[1:-1]:                                   # tous les ajouts acceptés…
        assert h["accepted"] and h["f_pvalue"] < 0.05 and h["significant"]
    last = hist[-1]                                        # …jusqu'au premier refus
    assert not last["accepted"]
    assert last.get("reason") == "non convergé" or last["f_pvalue"] >= 0.05 or not last["significant"]
    assert m.n_elements == hist[-2]["K"]
    # Paramètres du modèle retenu : tous significatifs à 2σ ([A92]).
    assert np.all(np.abs(m.R) > 2 * np.sqrt(np.diag(m.cov))[1:1 + m.n_elements])
    assert np.all(np.diff(m.tau) > 0)                      # éléments triés par τ


def test_component_fits_know_which_constants_are_identifiable():
    f, zre, zim, sd = orazem_noisy_arrays(_RCT, 4)
    m_im = fit_voigt(f, zre, zim, sd, sd, absolute_sigma=True, component="imag")
    m_re = fit_voigt(f, zre, zim, sd, sd, absolute_sigma=True, component="real")
    assert m_im.R0 is None and m_re.R0 is not None           # Im ne contient pas R0
    assert m_re.R0 == pytest.approx(220.0, rel=0.05)         # Re + R'e (Cb court-circuite au-delà)
    with pytest.raises(ValueError, match="component"):
        fit_voigt(f, zre, zim, sd, sd, absolute_sigma=True, component="both")


# ═════════════════════════════════════════════════════════════════════════════
# D. Structure d'erreur
# ═════════════════════════════════════════════════════════════════════════════

def test_error_structure_recovers_the_true_noise_without_any_floor(estimate5):
    es = estimate5.error_structure
    f, zre, zim, sigma_true = orazem_noisy_arrays(_RCT, 0)   # grille et Z identiques
    Z0 = randles_impedance(_RCT)[1]
    s_re, s_im = es.sigmas(Z0.real, -Z0.imag)
    assert es.equal_re_im is True and es.equality_pvalue >= 0.05   # σ_r = σ_j non rejetée
    np.testing.assert_allclose(s_re, s_im)
    assert np.mean(s_re / sigma_true) == pytest.approx(1.0, abs=0.10)
    assert np.all(es.to_dict()[k] >= 0 for k in ("alpha", "beta", "gamma", "delta"))
    assert es.n_replicates == 5 and es.dof > 100


def test_no_relative_floor_on_sigma():
    """AUDIT.md §2.4 : l'ancien ``_compute_sigma`` imposait σ ≥ 0,1 % de |Z|. Avec un
    instrument peu bruité (vrai σ ≈ 0,02-0,04 % de |Z|), σ estimé descend bien sous ce
    niveau et suit le vrai bruit."""
    quiet = dict(alpha=2e-4, beta=2e-4, delta=0.05)
    es = characterize_error_structure(
        orazem_replicates(_RCT, n_rep=5, seed0=500, **quiet)).error_structure
    Z0 = randles_impedance(_RCT)[1]
    s_re = es.sigmas(Z0.real, -Z0.imag)[0]
    true = orazem_noisy_arrays(_RCT, 0, **quiet)[3]
    assert np.all(s_re < 0.001 * np.abs(Z0))
    assert np.mean(s_re / true) == pytest.approx(1.0, abs=0.15)


def test_leverage_correction_removes_the_underestimation_of_sigma():
    """Un modèle ajusté absorbe une fraction h_ii du bruit : sans correction, σ est
    sous-estimé (~6-7 % mesurés) ; avec, le biais disparaît (Cook & Weisberg)."""
    ratios = {True: [], False: []}
    for seed in (70, 80, 90):
        reps = orazem_replicates(_RCT, n_rep=5, seed0=seed)
        Z0 = randles_impedance(_RCT)[1]
        true = orazem_noisy_arrays(_RCT, 0)[3]
        for lev in (True, False):
            es = characterize_error_structure(
                reps, options=MeasurementModelOptions(leverage_correction=lev)).error_structure
            ratios[lev].append(np.mean(es.sigmas(Z0.real, -Z0.imag)[0] / true))
    assert np.mean(ratios[False]) < 0.97
    assert abs(np.mean(ratios[True]) - 1.0) < 0.05
    assert np.mean(ratios[True]) > np.mean(ratios[False])


def test_c4_corrects_the_small_sample_bias_of_the_standard_deviation():
    assert mm._c4(np.array([3]))[0] == pytest.approx(0.8862, abs=1e-4)
    rng = np.random.default_rng(1)
    s = rng.standard_normal((200000, 3)).std(axis=1, ddof=1)
    assert s.mean() / mm._c4(np.array([3]))[0] == pytest.approx(1.0, abs=0.01)


def test_unequal_real_and_imaginary_noise_is_detected_not_imposed():
    """Bruit relatif par composante (σ_re ∝ |Zre|, σ_im ∝ |Zim|) : viole σ_r = σ_j.
    Le test F le rejette et deux structures sont estimées."""
    reps = [_ns(noisy_arrays(_RCT, 0.005, k), f"r{k}") for k in range(4)]
    es = characterize_error_structure(reps).error_structure
    assert es.equal_re_im is False and es.equality_pvalue < 0.05
    Z0 = randles_impedance(_RCT)[1]
    s_re, s_im = es.sigmas(Z0.real, -Z0.imag)
    hf = slice(0, 5)                                  # haute fréquence : |Zre| ≫ |Zim|
    assert np.all(s_re[hf] > 2 * s_im[hf])
    np.testing.assert_allclose(s_re[hf], 0.005 * Z0.real[hf], rtol=0.35)


@pytest.mark.parametrize("n_rep", [1, 2])
def test_refusal_with_too_few_replicates_has_a_user_message(n_rep):
    reps = orazem_replicates(_RCT, n_rep=n_rep)
    with pytest.raises(ErrorStructureUnavailable) as exc:
        characterize_error_structure(reps, label="probe e1")
    err = exc.value
    assert err.n_replicates == n_rep and err.group_label == "probe e1"
    assert err.user_message.startswith("Groupe « probe e1 » : analyse Orazem interrompue")
    assert "Aucun fit n'a été réalisé" in err.user_message
    assert "aucune pondération arbitraire" in err.user_message
    assert str(err) == err.user_message


def test_refusal_for_identical_nonfinite_or_incompatible_replicates():
    one = orazem_replicates(_RCT, n_rep=1)[0]
    with pytest.raises(ErrorStructureUnavailable, match="dispersion inter-réplicats nulle"):
        characterize_error_structure([one, one, one])
    bad = orazem_replicates(_RCT, n_rep=3)
    bad[1].Zre = bad[1].Zre.copy()
    bad[1].Zre[5] = np.nan
    with pytest.raises(ErrorStructureUnavailable, match="non finies"):
        characterize_error_structure(bad)
    other_grid = orazem_replicates(_RCT, n_rep=3)
    other_grid[2].f = other_grid[2].f * 1.07
    with pytest.raises(ErrorStructureUnavailable, match="fréquence"):
        characterize_error_structure(other_grid)


def test_no_persistence_and_no_hidden_state_between_groups(tmp_path, monkeypatch):
    """ERR-2 corrigé : rien n'est écrit sur disque ni réutilisé d'un groupe à l'autre."""
    work = tmp_path / "work"          # tmp_path contient déjà la fixture de l'ancien module
    work.mkdir()
    monkeypatch.chdir(work)
    for name in ("persist", "load_latest", "resolve_error_structure", "DEFAULT_PERSIST_PATH"):
        assert not hasattr(mm, name)
    a = orazem_replicates(_RCT, n_rep=3, seed0=100)
    b = orazem_replicates(2 * _RCT, n_rep=3, seed0=200, delta=2.0)
    alone = characterize_error_structure(b).error_structure
    characterize_error_structure(a)
    after_a = characterize_error_structure(b).error_structure
    for k in ("alpha", "beta", "gamma", "delta", "R_sol"):
        assert getattr(after_a, k) == getattr(alone, k)
    assert list(work.iterdir()) == []
    assert not (Path(mm.__file__).parent.parent / "config" / "error_structure.json").exists()


def test_error_structure_sigmas_formula():
    es = ErrorStructure(alpha=0.01, beta=0.02, gamma=1e-6, delta=0.5, R_sol=100.0,
                        n_replicates=3, dof=50)
    s_re, s_im = es.sigmas(np.array([300.0]), np.array([400.0]))
    expected = 0.01 * 400 + 0.02 * 200 + 1e-6 * 250000 + 0.5
    assert s_re[0] == pytest.approx(expected) and s_im[0] == pytest.approx(expected)
    split = ErrorStructure(alpha=0.0, beta=0.0, gamma=0.0, delta=1.0, R_sol=0.0, n_replicates=3,
                           dof=50, equal_re_im=False, im=(0.0, 0.0, 0.0, 2.0))
    assert split.sigmas(np.array([1.0]), np.array([1.0])) == (pytest.approx([1.0]), pytest.approx([2.0]))


# ═════════════════════════════════════════════════════════════════════════════
# E. Conformité KK par le MÊME measurement model
# ═════════════════════════════════════════════════════════════════════════════

def test_kk_test_reuses_the_measurement_model_of_the_error_structure(estimate5):
    es = estimate5.error_structure
    ref = estimate5.voigt_models[0]
    rep = orazem_replicates(_RCT, n_rep=5, seed0=40)[0]
    kk = check_kk_consistency(rep.f, rep.Zre, rep.Zim, es, reference=ref)
    assert kk.model_from_imag.n_elements == ref.n_elements          # même K
    assert kk.model_from_imag.component == "imag" and kk.model_from_imag.R0 is None
    assert kk.conform is True
    assert np.all(np.diff(kk.frequencies) > 0)
    np.testing.assert_allclose(kk.Zim, rep.Zim[::-1])                # convention Zim > 0 conservée
    assert np.all(kk.sd_re > kk.sigma_re * 0.99)                     # bruit + incertitude de prédiction


def test_a_spectrum_drifting_during_its_sweep_is_not_kk_conform(estimate5):
    es = estimate5.error_structure
    f, zre, zim, _ = orazem_noisy_arrays(_RCT, 77, drift=0.2)
    kk = check_kk_consistency(f, zre, zim, es)
    assert kk.conform is False
    assert kk.verdict.n_outside > kk.verdict.n_allowed


def test_group_analysis_gives_the_verdict_before_any_fit():
    reps = orazem_replicates(_RCT, n_rep=3, seed0=300)
    an = analyze_replicates(reps, label="c1")
    assert an.kk_conform is True and an.n_replicates == 3
    assert an.alpha_each == pytest.approx(an.alpha_family / 4)       # Bonferroni sur n + 1 tests
    assert an.kk_mean.n_averaged == 3 and all(k.n_averaged == 1 for k in an.kk_replicates)
    assert an.kk_message.startswith("Groupe « c1 » : conforme") and "proposé" in an.kk_message
    np.testing.assert_allclose(an.mean_Zre, np.mean([r.Zre[::-1] for r in reps], axis=0))

    drifting = orazem_replicates(_RCT, n_rep=3, seed0=300)
    drifting[2] = orazem_replicates(_RCT, n_rep=1, seed0=305, drift=0.2)[0]
    drifting[2].label = "r2"
    bad = analyze_replicates(drifting, label="c2")
    assert bad.kk_conform is False
    assert "NON conforme" in bad.kk_message and "r2" in bad.kk_message


def test_group_analysis_stops_on_an_uncharacterizable_group():
    with pytest.raises(ErrorStructureUnavailable):
        analyze_replicates(orazem_replicates(_RCT, n_rep=2), label="deux")


def test_kk_verdict_false_alarm_rate_on_conform_groups():
    """Calibrage : 12 groupes conformes (3 réplicats, 40 points) — au plus 2 rejets
    (P(X ≥ 3) ≈ 2 % pour un risque de 5 %). Mesuré hors test sur 3 × 100 groupes :
    4 %, 1 % et 0 % (voir la docstring de ``analyze_replicates``)."""
    rejects = sum(not analyze_replicates(orazem_replicates(_RCT, n_rep=3, seed0=1000 + 10 * g)).kk_conform
                  for g in range(12))
    assert rejects <= 2
