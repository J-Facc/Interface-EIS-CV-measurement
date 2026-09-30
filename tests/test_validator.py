"""Tests de CARACTÉRISATION de core/validator.py (filet de sécurité avant refonte).

Figent le comportement RÉEL du validateur KK et de la détection de dérive, défauts
compris (voir l'en-tête de tests/test_pipeline.py pour la philosophie). Défauts
figés, chacun signalé par « COMPORTEMENT ACTUEL BOGUÉ » :

  * ERR-4 (AUDIT.md) — ``_detect_drift`` : faux positifs sur des réplicats
    stationnaires ET faux négatif sur de vrais écarts entre réplicats ;
  * ERR-5 (AUDIT.md) — ``linKK`` ignore ``c`` et ``fit_type`` ; M est fixé par le
    nombre de points ;
  * trois observations SUPPLÉMENTAIRES, non listées dans l'audit : avertissement de
    non-chevauchement écrasé, pas de garde sur les spectres trop courts (1 point =
    valide), tranche « 10 % de points marginaux » ratée à cause d'un arrondi flottant.

Les fabriques de spectres viennent de tests/synthetic_data.py (style Annexe A).
"""

import logging

import numpy as np
import pytest

from core import validator as V
from core.loader import average_replicates
from core.models import EISSpectrum
from core.validator import (
    DRIFT_CV_THRESHOLD,
    INVALID_FRACTION_REJECT,
    INVALID_FRACTION_WARN,
    MU_THRESHOLD,
    RESIDUAL_THRESHOLD_PCT,
    KKResult,
    ValidationResult,
    _compute_sigma,
    _detect_drift,
    _find_valid_range,
    validate_replicate_group,
    validate_spectrum,
)
from fits.kk_validation import linKK
from tests.synthetic_data import N_POINTS, noisy_arrays, randles_impedance

_RCT = 3000.0


# ─────────────────────────────────────────────────────────────────────────────
# Outils
# ─────────────────────────────────────────────────────────────────────────────

def _clean():
    """(f, Zre, Zim) sans bruit, ordre HF → BF."""
    return noisy_arrays(_RCT, noise=0.0)


def _perturbed(n_points_hit: int, factor: float):
    """Spectre sans bruit dont ``n_points_hit`` valeurs de Re(Z) sont multipliées."""
    f, zre, zim = _clean()
    idx = np.linspace(2, N_POINTS - 3, n_points_hit).astype(int)
    zre = zre.copy()
    zre[idx] *= factor
    return f, zre, zim


def _kk(lo, hi, valid=True, res_im=None, freqs=None):
    """KKResult fabriqué à la main (résidus nuls par défaut)."""
    fr = np.logspace(-1, 5, 30) if freqs is None else freqs
    return KKResult(
        label="k", frequencies=fr, residuals_re=np.zeros(len(fr)),
        residuals_im=np.zeros(len(fr)) if res_im is None else res_im,
        mu=0.1, chi2_pseudo=0.0, f_min_valid=lo, f_max_valid=hi, is_valid=valid,
    )


def _invalid_fraction(kk):
    ok = (np.abs(kk.residuals_re) < RESIDUAL_THRESHOLD_PCT) & (np.abs(kk.residuals_im) < RESIDUAL_THRESHOLD_PCT)
    return 1.0 - ok.mean()


def _group(reps):
    """reps : liste de (f, zre, zim) → arguments de validate_replicate_group."""
    return [r[0] for r in reps], [r[1] for r in reps], [r[2] for r in reps]


# ═════════════════════════════════════════════════════════════════════════════
# validate_spectrum
# ═════════════════════════════════════════════════════════════════════════════

def test_module_thresholds_are_the_documented_defaults():
    assert MU_THRESHOLD == 0.85
    assert RESIDUAL_THRESHOLD_PCT == 2.0
    assert INVALID_FRACTION_WARN == 0.10
    assert INVALID_FRACTION_REJECT == 0.25
    assert DRIFT_CV_THRESHOLD == 0.5


def test_clean_spectrum_is_valid_with_tiny_residuals():
    f, zre, zim = _clean()
    kk = validate_spectrum(f, zre, zim, label="clean")

    assert isinstance(kk, KKResult)
    assert kk.label == "clean"
    assert kk.is_valid is True and kk.warning is None
    assert kk.mu == pytest.approx(0.056, abs=0.01)         # M=26 éléments RC pour 40 points
    assert kk.chi2_pseudo < 1e-3
    assert np.abs(kk.residuals_re).max() < 0.05           # résidus en %
    assert np.abs(kk.residuals_im).max() < 0.05
    # Sortie triée en fréquence CROISSANTE (l'entrée du pipeline est HF → BF).
    assert np.all(np.diff(kk.frequencies) > 0)
    assert len(kk.residuals_re) == len(kk.residuals_im) == N_POINTS
    assert (kk.f_min_valid, kk.f_max_valid) == (pytest.approx(0.1), pytest.approx(1e5))


def test_input_order_and_sign_of_zim_do_not_matter():
    f, zre, zim = _clean()
    ref = validate_spectrum(f, zre, zim)

    idx = np.random.default_rng(5).permutation(N_POINTS)
    shuffled = validate_spectrum(f[idx], zre[idx], zim[idx])
    negated = validate_spectrum(f, zre, -zim)        # np.abs(z_im) est appliqué

    for other in (shuffled, negated):
        assert other.mu == ref.mu
        np.testing.assert_allclose(other.residuals_re, ref.residuals_re)
        np.testing.assert_allclose(other.residuals_im, ref.residuals_im)


def test_half_percent_noise_is_still_valid_but_three_percent_is_rejected():
    f, zre, zim = noisy_arrays(_RCT, noise=0.005, seed=1)
    assert validate_spectrum(f, zre, zim).is_valid is True

    f, zre, zim = noisy_arrays(_RCT, noise=0.03, seed=2)
    kk = validate_spectrum(f, zre, zim)
    assert kk.is_valid is False
    assert "hors tolérance KK" in kk.warning and "Spectre invalide" in kk.warning
    assert _invalid_fraction(kk) >= INVALID_FRACTION_REJECT
    # La plage valide est le plus grand bloc contigu : strictement plus étroite.
    assert (kk.f_min_valid, kk.f_max_valid) != (pytest.approx(0.1), pytest.approx(1e5))
    assert 0.1 <= kk.f_min_valid <= kk.f_max_valid <= 1e5


def test_marginal_tier_is_valid_but_flagged():
    """10 % ≤ points invalides < 25 % → valide, avec un avertissement « marginaux »."""
    f, zre, zim = _perturbed(5, 1.07)                 # 6 points sur 40 hors tolérance
    kk = validate_spectrum(f, zre, zim)

    assert _invalid_fraction(kk) == pytest.approx(6 / 40)
    assert kk.is_valid is True
    assert kk.warning == "15% des points marginaux — vérifier les extrémités du spectre."


def test_reject_tier_above_a_quarter_of_invalid_points():
    f, zre, zim = _perturbed(5, 1.15)
    kk = validate_spectrum(f, zre, zim)

    assert kk.is_valid is False
    assert _invalid_fraction(kk) >= INVALID_FRACTION_REJECT
    assert "des points hors tolérance KK (seuil résidu = 2.0%). Spectre invalide." in kk.warning


def test_exactly_ten_percent_invalid_is_not_flagged_because_of_float_rounding():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    4 points invalides sur 40 = 10 % pile devraient déclencher le palier
    ``invalid_fraction >= INVALID_FRACTION_WARN``. Mais ``1.0 - valid_mask.mean()``
    vaut ``0.09999999999999998`` : le seuil n'est pas atteint, aucun avertissement.
    """
    f, zre, zim = _perturbed(4, 1.06)
    kk = validate_spectrum(f, zre, zim)

    n_invalid = round(_invalid_fraction(kk) * N_POINTS)
    assert n_invalid == 4
    assert 1.0 - (N_POINTS - n_invalid) / N_POINTS < INVALID_FRACTION_WARN   # l'arrondi en cause
    assert kk.is_valid is True
    assert kk.warning is None


def test_mu_threshold_is_checked_before_the_residual_tiers():
    f, zre, zim = _clean()
    kk = validate_spectrum(f, zre, zim, mu_threshold=0.01)
    assert kk.is_valid is False
    assert kk.warning.startswith("µ = ") and "> 0.01 : sur-ajustement probable" in kk.warning


def test_residual_threshold_is_configurable():
    f, zre, zim = noisy_arrays(_RCT, noise=0.005, seed=1)
    kk = validate_spectrum(f, zre, zim, residual_threshold_pct=0.01)
    assert kk.is_valid is False
    assert kk.warning.startswith("100% des points hors tolérance KK (seuil résidu = 0.01%)")
    # Aucun point valide → la « plage valide » est la plage complète.
    assert (kk.f_min_valid, kk.f_max_valid) == (pytest.approx(0.1), pytest.approx(1e5))


def test_a_crashing_linkk_gives_an_invalid_result_with_zero_residuals(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no convergence")

    monkeypatch.setattr("fits.kk_validation.linKK", boom)
    f, zre, zim = _clean()
    kk = validate_spectrum(f, zre, zim, label="crash")

    assert kk.is_valid is False
    assert kk.warning == "lin-KK échoué : no convergence"
    assert kk.mu == 1.0 and kk.chi2_pseudo == np.inf
    assert not kk.residuals_re.any() and not kk.residuals_im.any()
    assert (kk.f_min_valid, kk.f_max_valid) == (pytest.approx(0.1), pytest.approx(1e5))


def test_nan_in_the_data_is_reported_as_fully_invalid():
    f, zre, zim = _clean()
    zre = zre.copy()
    zre[3] = np.nan
    kk = validate_spectrum(f, zre, zim)

    assert kk.is_valid is False
    assert np.isnan(kk.chi2_pseudo)
    assert kk.warning.startswith("100% des points hors tolérance KK")


def test_empty_input_crashes_the_error_handler_itself():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    Sur un spectre vide, lin-KK échoue (``max`` d'un tableau vide) ; le bloc
    ``except`` qui doit produire un KKResult « invalide » lit alors ``f_sorted[0]``
    et lève à son tour un IndexError, qui remonte à l'appelant.
    """
    empty = np.array([])
    with pytest.raises(IndexError):
        validate_spectrum(empty, empty, empty)


def test_very_short_spectra_are_accepted_without_any_guard():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    Aucune garde sur le nombre de points : 1, 2 ou 3 points ne lèvent rien et sont
    déclarés valides (un circuit de Voigt à quelques éléments les reproduit
    forcément). Le loader, lui, refuse < 5 points en amont.
    """
    f, zre, zim = _clean()
    for n in (1, 2, 3):
        kk = validate_spectrum(f[:n], zre[:n], zim[:n])
        assert kk.is_valid is True and kk.warning is None
        assert len(kk.frequencies) == n


def test_linkk_wrapper_ignores_c_and_fit_type_and_fixes_m_from_the_point_count():
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-5, à corriger à l'étape 5.

    validate_spectrum passe ``c=0.85`` (critère de Schönleber) et ``fit_type`` à
    ``linKK``, laissant croire à une recherche itérative de M. Le wrapper les IGNORE :
    M = min(max_M, max(2, 2n//3)) est fixé par le nombre de points.
    """
    f, z = randles_impedance(_RCT)
    order = np.argsort(f)
    f_asc, z_asc = f[order], z[order]

    base = linKK(f_asc, z_asc, c=0.85, fit_type="complex")
    other = linKK(f_asc, z_asc, c=0.01, fit_type="real")     # valeurs radicalement différentes
    assert base[0] == other[0]
    for a, b in zip(base[1:], other[1:]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_points, expected_m", [(3, 2), (4, 2), (40, 26), (60, 40), (300, 100)])
def test_linkk_m_is_two_thirds_of_the_points_clamped_to_2_and_100(n_points, expected_m):
    f, z = randles_impedance(_RCT, n_points=n_points)
    order = np.argsort(f)
    assert linKK(f[order], z[order])[0] == expected_m


# ═════════════════════════════════════════════════════════════════════════════
# _find_valid_range
# ═════════════════════════════════════════════════════════════════════════════

def test_find_valid_range_returns_the_longest_contiguous_run():
    f = np.arange(1.0, 9.0)                                 # 1..8, ordre croissant (comme validate_spectrum)
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)   # runs de 2, 3 et 1 points
    assert _find_valid_range(f, mask) == (4.0, 6.0)


def test_find_valid_range_first_run_wins_a_tie():
    f = np.arange(1.0, 7.0)
    mask = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
    assert _find_valid_range(f, mask) == (1.0, 2.0)


def test_find_valid_range_single_valid_point_and_degenerate_masks():
    f = np.arange(1.0, 5.0)
    assert _find_valid_range(f, np.array([0, 0, 1, 0], dtype=bool)) == (3.0, 3.0)
    assert _find_valid_range(f, np.ones(4, dtype=bool)) == (1.0, 4.0)
    assert _find_valid_range(f, np.zeros(4, dtype=bool)) == (1.0, 4.0)   # aucun valide → plage complète


# ═════════════════════════════════════════════════════════════════════════════
# validate_replicate_group
# ═════════════════════════════════════════════════════════════════════════════

def test_empty_group_is_invalid_with_a_message():
    vr = validate_replicate_group([], [], [], label="x")
    assert isinstance(vr, ValidationResult)
    assert vr.replicates == []
    assert vr.all_valid is False
    assert vr.drift_warning == "Aucun réplicat fourni."
    assert vr.drift_detected is False
    assert vr.f_min_common == 0.0 and vr.f_max_common == np.inf
    assert vr.sigma_re is None and vr.sigma_im is None


def test_single_replicate_group_has_no_drift_check_and_no_sigma():
    f, zre, zim = _clean()
    vr = validate_replicate_group([f], [zre], [zim], label="one")

    assert [r.label for r in vr.replicates] == ["one_rep1"]
    assert vr.all_valid is True
    assert vr.drift_detected is False and vr.drift_warning is None
    assert vr.sigma_re is None and vr.sigma_im is None            # σ demande ≥ 2 réplicats
    assert (vr.f_min_common, vr.f_max_common) == (pytest.approx(0.1), pytest.approx(1e5))


def test_identical_replicates_are_valid_without_drift_and_sigma_sits_on_its_floor():
    reps = [_clean()] * 3
    vr = validate_replicate_group(*_group(reps), label="ident")

    assert vr.all_valid is True
    assert vr.drift_detected is False and vr.drift_warning is None
    assert [r.label for r in vr.replicates] == ["ident_rep1", "ident_rep2", "ident_rep3"]
    # Aucune dispersion → σ = plancher de 0,1 % du module (ordre de fréquence CROISSANT).
    _f, z = randles_impedance(_RCT)
    floor = 0.001 * np.abs(z)[::-1]
    np.testing.assert_allclose(vr.sigma_re, floor, rtol=1e-6)
    np.testing.assert_allclose(vr.sigma_im, floor, rtol=1e-6)


def test_common_range_is_the_intersection_of_replicate_ranges(monkeypatch):
    results = iter([_kk(1e-1, 1e4), _kk(1e0, 1e5), _kk(1e-2, 1e3)])
    monkeypatch.setattr(V, "validate_spectrum", lambda *a, **k: next(results))
    z = [np.ones(30)] * 3

    vr = validate_replicate_group([np.logspace(-1, 5, 30)] * 3, z, z, label="g")

    assert vr.f_min_common == 1e0          # max des bornes basses
    assert vr.f_max_common == 1e3          # min des bornes hautes
    assert vr.all_valid is True


def test_one_invalid_replicate_makes_the_group_invalid(monkeypatch):
    results = iter([_kk(1e-1, 1e5), _kk(1e-1, 1e5, valid=False)])
    monkeypatch.setattr(V, "validate_spectrum", lambda *a, **k: next(results))
    z = [np.ones(30)] * 2

    vr = validate_replicate_group([np.logspace(-1, 5, 30)] * 2, z, z, label="g")

    assert vr.all_valid is False
    assert [r.is_valid for r in vr.replicates] == [True, False]


def test_disjoint_ranges_invalidate_the_group_but_the_warning_is_overwritten(monkeypatch):
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    Quand les plages KK-valides des réplicats ne se chevauchent pas, le code pose
    ``drift_warning = "…plages KK-valides…ne se chevauchent pas…"`` puis, dès qu'il y a
    ≥ 2 réplicats, ré-affecte INCONDITIONNELLEMENT ``drift_warning`` avec le résultat de
    ``_detect_drift``. Le message de non-chevauchement est donc perdu : le groupe est
    invalide (``all_valid=False``) sans aucun message pour l'expliquer.
    """
    results = iter([_kk(1e-1, 1e1), _kk(1e3, 1e5)])
    monkeypatch.setattr(V, "validate_spectrum", lambda *a, **k: next(results))
    z = [np.ones(30)] * 2

    vr = validate_replicate_group([np.logspace(-1, 5, 30)] * 2, z, z, label="disjoint")

    assert vr.f_min_common == 1e3 and vr.f_max_common == 1e1     # bornes croisées
    assert vr.all_valid is False
    assert vr.drift_detected is False
    assert vr.drift_warning is None                              # message écrasé


def test_thresholds_are_forwarded_to_each_replicate_validation(monkeypatch):
    seen = []

    def spy(freqs, zre, zim, label="", mu_threshold=None, residual_threshold_pct=None):
        seen.append((label, mu_threshold, residual_threshold_pct))
        return _kk(1e-1, 1e5)

    monkeypatch.setattr(V, "validate_spectrum", spy)
    z = [np.ones(30)] * 2
    validate_replicate_group([np.logspace(-1, 5, 30)] * 2, z, z, label="g",
                             mu_threshold=0.3, residual_threshold_pct=1.5)

    assert seen == [("g_rep1", 0.3, 1.5), ("g_rep2", 0.3, 1.5)]


def test_drift_threshold_is_forwarded_to_the_drift_check():
    reps = [noisy_arrays(_RCT, 0.005, seed) for seed in (0, 1, 2)]
    assert validate_replicate_group(*_group(reps), label="g").drift_detected is True
    assert validate_replicate_group(*_group(reps), label="g", drift_cv_threshold=100.0).drift_detected is False


def test_mismatched_list_lengths_are_silently_truncated():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit) : ``zip`` tronque
    sans erreur — 3 fréquences mais 2 jeux de Z → 2 réplicats validés, le 3ᵉ ignoré."""
    reps = [_clean()] * 3
    f_list, zre_list, zim_list = _group(reps)
    vr = validate_replicate_group(f_list, zre_list[:2], zim_list[:2], label="g")
    assert len(vr.replicates) == 2


# ═════════════════════════════════════════════════════════════════════════════
# _detect_drift
# ═════════════════════════════════════════════════════════════════════════════

def test_stationary_replicates_are_flagged_as_drifting(caplog):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-4, à corriger à l'étape 5.

    Sur des réplicats STATIONNAIRES (même spectre, bruit gaussien indépendant de
    0,5 %), ``_detect_drift`` annonce une dérive sur > 60 % des fréquences, à chaque
    fois : le critère CV = σ/moyenne des résidus KK inter-réplicats vaut ~1 pour du
    bruit blanc, donc dépasse toujours le seuil 0,5. L'UI l'affiche en rouge.
    """
    for base_seed in (0, 100, 200, 300, 400):
        reps = [noisy_arrays(_RCT, 0.005, base_seed + k) for k in range(3)]
        with caplog.at_level(logging.WARNING, logger="core.validator"):
            vr = validate_replicate_group(*_group(reps), label="stationnaire")
        assert vr.all_valid is True                       # chaque réplicat est bien valide…
        assert vr.drift_detected is True                  # …et pourtant « dérive »
        assert vr.drift_warning.startswith("Drift détecté sur ")
        assert "(CV résidus KK > 0.5)" in vr.drift_warning
        assert "entre les 3 réplicats" in vr.drift_warning
        pct = int(vr.drift_warning.split("sur ")[1].split("%")[0])
        assert pct > 20                                    # au-delà du seuil de 20 % des fréquences
    assert any("Drift détecté" in r.getMessage() for r in caplog.records)


def test_real_differences_between_replicates_are_not_detected():
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-4 (faux négatif), à corriger à l'étape 5.

    Le critère mesure la dispersion des RÉSIDUS KK, pas celle des spectres : deux
    réplicats sans bruit dont le Rct diffère d'un facteur 2 ont chacun un résidu KK
    minuscule (sous le garde ``mean_res > 0.1``), donc aucune dérive n'est signalée.
    """
    a = noisy_arrays(_RCT, 0.0)
    b = noisy_arrays(2 * _RCT, 0.0)
    vr = validate_replicate_group(*_group([a, a, b]), label="deux_rct")

    assert vr.all_valid is True
    assert vr.drift_detected is False and vr.drift_warning is None


def test_low_noise_replicates_are_not_flagged():
    """Le garde ``mean_res > 0.1`` (en %) neutralise le CV quand les résidus sont
    minuscules : à 0,1 % de bruit, aucune dérive — le faux positif est propre au
    bruit d'au moins ~0,3 %."""
    reps = [noisy_arrays(_RCT, 0.001, k) for k in range(3)]
    vr = validate_replicate_group(*_group(reps), label="bas_bruit")
    assert vr.drift_detected is False and vr.drift_warning is None


def test_white_noise_residuals_always_exceed_the_cv_threshold():
    rng = np.random.default_rng(0)
    kks = [_kk(0, 0, res_im=rng.normal(0, 1, 30)) for _ in range(3)]
    drifted, msg = _detect_drift(kks, 0.5)
    assert drifted is True
    assert msg.startswith("Drift détecté sur ")


def test_drift_needs_strictly_more_than_twenty_percent_of_frequencies():
    res = [np.zeros(30) for _ in range(3)]
    for i in range(6):                         # 6 fréquences sur 30 = 20 % exactement
        res[0][i], res[1][i] = 2.0, -2.0
    assert _detect_drift([_kk(0, 0, res_im=r) for r in res], 0.5) == (False, None)

    res[0][6], res[1][6] = 2.0, -2.0           # 7 sur 30 = 23 % → dérive
    drifted, msg = _detect_drift([_kk(0, 0, res_im=r) for r in res], 0.5)
    assert drifted is True
    assert "sur 23% des fréquences" in msg


def test_residuals_below_a_tenth_of_a_percent_never_count_as_drift():
    kks = [_kk(0, 0, res_im=np.full(30, 0.05) * s) for s in (1.0, -1.0, 1.0)]
    assert _detect_drift(kks, 0.5) == (False, None)


def test_identical_constant_residuals_are_not_drift():
    kks = [_kk(0, 0, res_im=np.full(30, 2.0)) for _ in range(3)]
    assert _detect_drift(kks, 0.5) == (False, None)


def test_a_huge_cv_threshold_disables_the_detection():
    rng = np.random.default_rng(0)
    kks = [_kk(0, 0, res_im=rng.normal(0, 1, 30)) for _ in range(3)]
    assert _detect_drift(kks, 100.0) == (False, None)


def test_replicates_on_different_frequency_grids_are_interpolated():
    a = _kk(0, 0, freqs=np.logspace(-1, 5, 30))
    b = _kk(0, 0, freqs=np.logspace(-1, 5, 50))
    assert _detect_drift([a, b], 0.5) == (False, None)


# ═════════════════════════════════════════════════════════════════════════════
# _compute_sigma
# ═════════════════════════════════════════════════════════════════════════════

def test_sigma_is_a_sample_std_returned_in_increasing_frequency_order():
    """Valeurs calculées à la main. Entrée HF → BF (f = 100, 10, 1) ; SORTIE en ordre de
    fréquence CROISSANT (f = 1, 10, 100) — l'inverse de la convention du loader.

    Re(Z) à f=100 : (10, 12, 20) → moyenne 14, variance (16+4+36)/2 = 28, σ = √28.
    À f=10 : (20, 22, 24) → σ = 2 ; à f=1 : (30, 32, 34) → σ = 2. Im(Z) : (1,2,3),
    (2,3,4), (3,4,5) → σ = 1 partout. Les planchers (0,1 % de |Z|) sont bien plus bas.
    """
    f = np.array([100.0, 10.0, 1.0])
    zre = [np.array([10., 20., 30.]), np.array([12., 22., 32.]), np.array([20., 24., 34.])]
    zim = [np.array([1., 2., 3.]), np.array([2., 3., 4.]), np.array([3., 4., 5.])]

    sigma_re, sigma_im = _compute_sigma([f, f, f], zre, zim)

    np.testing.assert_allclose(sigma_re, [2.0, 2.0, np.sqrt(28.0)])
    np.testing.assert_allclose(sigma_im, [1.0, 1.0, 1.0])


def test_sigma_has_a_floor_of_a_tenth_of_a_percent_of_the_mean_modulus():
    """Réplicats identiques : σ = 0,001·|Z̄|. |Z| = (5, 10, 10) à f = (100, 10, 1) →
    plancher (0,01, 0,01, 0,005) en ordre croissant de fréquence."""
    f = np.array([100.0, 10.0, 1.0])
    zre = np.array([3.0, 6.0, 0.0])
    zim = np.array([4.0, 8.0, 10.0])

    sigma_re, sigma_im = _compute_sigma([f, f, f], [zre] * 3, [zim] * 3)

    np.testing.assert_allclose(sigma_re, [0.01, 0.01, 0.005])
    np.testing.assert_allclose(sigma_im, [0.01, 0.01, 0.005])


def test_sigma_interpolates_replicates_linearly_in_frequency_on_the_first_grid():
    f_a = np.array([1.0, 10.0, 100.0])
    f_b = np.array([1.0, 100.0])                       # autre grille : interpolée (linéaire en f)
    zre_a = np.array([0.0, 10.0, 99.0])
    zre_b = np.array([0.0, 99.0])                      # à f=10 : 9,0
    zim = np.ones(3)

    sigma_re, sigma_im = _compute_sigma([f_a, f_b], [zre_a, zre_b], [zim, np.ones(2)])

    # f=1 et f=100 : réplicats égaux → plancher 0,001·|Z̄| ; f=10 : std((10, 9)) = 0,7071.
    np.testing.assert_allclose(
        sigma_re, [0.001, np.sqrt(0.5), 0.001 * np.hypot(99.0, 1.0)], rtol=1e-4)
    np.testing.assert_allclose(
        sigma_im, [0.001, 0.001 * np.hypot(9.5, 1.0), 0.001 * np.hypot(99.0, 1.0)], rtol=1e-4)


def test_sigma_duplicates_the_loader_estimate_in_reverse_order_with_a_floor():
    """Doublon (AUDIT.md §2.4 : ``ValidationResult.sigma_*`` n'est jamais lu).

    ``core.loader.average_replicates`` calcule le même écart-type inter-réplicats
    (ddof=1) pour le pipeline — en ordre HF → BF et SANS plancher. Celui du validateur
    est en ordre BF → HF et AVEC plancher (0,1 % de |Z̄|) : deux estimations du même σ
    qui ne coïncident qu'aux points où le plancher ne mord pas.
    """
    arrays = [noisy_arrays(_RCT, 0.005, k) for k in range(3)]
    f_list, zre_list, zim_list = _group(arrays)
    validator_re, validator_im = _compute_sigma(f_list, zre_list, zim_list)

    spectra = [EISSpectrum(label=f"r{k}", f=f, Zre=zre, Zim=zim, concentration=0.0,
                           step="probe", n_points=len(f))
               for k, (f, zre, zim) in enumerate(arrays)]
    avg = average_replicates(spectra)
    floor = 0.001 * np.hypot(avg.Zre, avg.Zim)[::-1]           # ordre BF → HF, comme le validateur

    # σ_validateur = max(σ_loader retourné, plancher)…
    np.testing.assert_allclose(validator_re, np.maximum(avg.sigma_re[::-1], floor), rtol=1e-9)
    np.testing.assert_allclose(validator_im, np.maximum(avg.sigma_im[::-1], floor), rtol=1e-9)
    # …et le plancher mord réellement sur ce jeu (3 réplicats seulement) : le σ du loader,
    # brut, descend sous lui en au moins un point, où les deux estimations divergent.
    assert (avg.sigma_re[::-1] < floor).any()
    assert not np.allclose(validator_re, avg.sigma_re[::-1])
