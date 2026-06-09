import numpy as np
import pytest

from comparison.metrics import (
    compute_sigma_intra,
    compute_sigma_inter,
    compute_rmsep,
    compute_bias,
    compute_rpd,
    compute_ci95_bootstrap,
    compute_rmsecv_logo,
    diebold_mariano_test,
    compute_normalization_reduction,
    compute_sigma_probe,
)


# ---------------------------------------------------------------------------
# compute_sigma_intra
# ---------------------------------------------------------------------------

class TestSigmaIntra:
    def test_nominal(self):
        # std([1, 2, 3], ddof=1) = 1.0
        result = compute_sigma_intra(np.array([1.0, 2.0, 3.0]))
        assert result == pytest.approx(1.0)

    def test_identical_values(self):
        assert compute_sigma_intra(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)

    def test_single_replicate(self):
        assert compute_sigma_intra(np.array([3.0])) == 0.0

    def test_empty_array(self):
        assert compute_sigma_intra(np.array([])) == 0.0

    def test_two_replicates(self):
        # std([0, 2], ddof=1) = sqrt(2)
        result = compute_sigma_intra(np.array([0.0, 2.0]))
        assert result == pytest.approx(np.sqrt(2))


# ---------------------------------------------------------------------------
# compute_sigma_inter
# ---------------------------------------------------------------------------

class TestSigmaInter:
    def test_nominal(self):
        assert compute_sigma_inter(1.0, 3.0) == pytest.approx(2.0)

    def test_symmetry(self):
        assert compute_sigma_inter(3.0, 1.0) == pytest.approx(2.0)

    def test_identical_means(self):
        assert compute_sigma_inter(2.5, 2.5) == pytest.approx(0.0)

    def test_negative_values(self):
        # log10([c]) peut être négatif
        assert compute_sigma_inter(-10.0, -8.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# compute_rmsep
# ---------------------------------------------------------------------------

class TestRmsep:
    def test_perfect_prediction(self):
        assert compute_rmsep(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        ) == pytest.approx(0.0)

    def test_known_value(self):
        # erreurs : [1, 0, -1] → RMSE = sqrt((1+0+1)/3) = sqrt(2/3)
        result = compute_rmsep(
            np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])
        )
        assert result == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_constant_error(self):
        # erreur constante de 0.5 décade → RMSEP = 0.5
        result = compute_rmsep(
            np.array([1.5, 2.5, 3.5]), np.array([1.0, 2.0, 3.0])
        )
        assert result == pytest.approx(0.5)

    def test_empty_array(self):
        assert np.isnan(compute_rmsep(np.array([]), np.array([])))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            compute_rmsep(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# compute_bias
# ---------------------------------------------------------------------------

class TestBias:
    def test_positive_bias(self):
        # mean([0.5, 0.5]) = 0.5
        result = compute_bias(
            np.array([1.5, 2.5]), np.array([1.0, 2.0])
        )
        assert result == pytest.approx(0.5)

    def test_zero_bias(self):
        result = compute_bias(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        )
        assert result == pytest.approx(0.0)

    def test_negative_bias(self):
        result = compute_bias(
            np.array([0.5, 1.5]), np.array([1.0, 2.0])
        )
        assert result == pytest.approx(-0.5)

    def test_empty_array(self):
        assert np.isnan(compute_bias(np.array([]), np.array([])))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            compute_bias(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# compute_rpd
# ---------------------------------------------------------------------------

class TestRpd:
    def test_nominal(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmsep = 0.5
        expected = np.std(y_true, ddof=1) / rmsep
        assert compute_rpd(y_true, rmsep) == pytest.approx(expected)

    def test_rmsep_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        assert compute_rpd(y_true, 0.0) == np.inf

    def test_single_point(self):
        assert np.isnan(compute_rpd(np.array([1.0]), 0.5))

    def test_empty_array(self):
        assert np.isnan(compute_rpd(np.array([]), 0.5))

    def test_known_value(self):
        # std([0, 10], ddof=1) = sqrt(50) ≈ 7.07, rmsep=1.0 → RPD ≈ 7.07
        y_true = np.array([0.0, 10.0])
        result = compute_rpd(y_true, 1.0)
        assert result == pytest.approx(np.std(y_true, ddof=1))


# ---------------------------------------------------------------------------
# compute_ci95_bootstrap
# ---------------------------------------------------------------------------

class TestCi95Bootstrap:
    def test_nominal_interval_ordered(self):
        rng = np.random.default_rng(0)
        preds = rng.normal(loc=-12.0, scale=0.3, size=20)
        lo, hi = compute_ci95_bootstrap(preds, n_bootstrap=2000)
        assert lo <= hi

    def test_interval_contains_mean(self):
        preds = np.array([-12.0, -11.8, -12.2, -11.9, -12.1])
        lo, hi = compute_ci95_bootstrap(preds, n_bootstrap=2000)
        assert lo <= np.mean(preds) <= hi

    def test_single_replicate(self):
        lo, hi = compute_ci95_bootstrap(np.array([-10.0]))
        assert lo == pytest.approx(-10.0)
        assert hi == pytest.approx(-10.0)

    def test_empty_array(self):
        lo, hi = compute_ci95_bootstrap(np.array([]))
        assert np.isnan(lo) and np.isnan(hi)

    def test_identical_values(self):
        preds = np.array([-12.0, -12.0, -12.0])
        lo, hi = compute_ci95_bootstrap(preds)
        assert lo == pytest.approx(-12.0)
        assert hi == pytest.approx(-12.0)


# ---------------------------------------------------------------------------
# compute_rmsecv_logo
# ---------------------------------------------------------------------------

def _linear_model_fn(X_train, y_train, X_test):
    """Modèle linéaire simple pour les tests."""
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(X_train[:, 0], y_train, 1)
    return np.polyval(coeffs, X_test[:, 0])


class TestRmsecvLogo:
    def test_nominal(self):
        # Relation parfaitement linéaire : y = x → RMSECV doit être faible
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.arange(10, dtype=float)
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        result = compute_rmsecv_logo(X, y, groups, _linear_model_fn)
        assert isinstance(result, float)
        assert result < 0.5  # relation linéaire → erreur faible

    def test_single_group(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        groups = np.array([0, 0])
        assert np.isnan(compute_rmsecv_logo(X, y, groups, _linear_model_fn))

    def test_returns_float(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.arange(6, dtype=float)
        groups = np.array([0, 0, 1, 1, 2, 2])
        result = compute_rmsecv_logo(X, y, groups, _linear_model_fn)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# diebold_mariano_test
# ---------------------------------------------------------------------------

class TestDieboldMariano:
    def test_identical_errors_high_pvalue(self):
        errors = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
        stat, p = diebold_mariano_test(errors, errors)
        assert p > 0.05

    def test_identical_errors_stat_zero(self):
        errors = np.array([0.1, 0.2, 0.3])
        stat, p = diebold_mariano_test(errors, errors)
        assert stat == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_clearly_different_methods(self):
        # méthode A grosse erreur variable, méthode B petite erreur variable
        # d = e_a² - e_b² non-constant → std(d) > 0 → test valide
        rng = np.random.default_rng(0)
        errors_a = rng.uniform(3.0, 5.0, size=20)
        errors_b = rng.uniform(0.0, 0.1, size=20)
        stat, p = diebold_mariano_test(errors_a, errors_b)
        assert p < 0.05
        assert stat > 0  # d = e_a² - e_b² > 0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            diebold_mariano_test(np.array([1.0, 2.0]), np.array([1.0]))

    def test_single_point(self):
        stat, p = diebold_mariano_test(np.array([1.0]), np.array([0.5]))
        assert np.isnan(stat) and np.isnan(p)

    def test_returns_tuple(self):
        errors_a = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        errors_b = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
        result = diebold_mariano_test(errors_a, errors_b)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# compute_normalization_reduction
# ---------------------------------------------------------------------------

class TestNormalizationReduction:
    def test_nominal_equal(self):
        assert compute_normalization_reduction(1.0, 1.0) == pytest.approx(1.0)

    def test_reduction_factor_3(self):
        assert compute_normalization_reduction(3.0, 1.0) == pytest.approx(3.0)

    def test_no_effect(self):
        # sigma_inter inchangé → facteur = 1
        assert compute_normalization_reduction(0.5, 0.5) == pytest.approx(1.0)

    def test_norm_zero_raw_positive(self):
        assert compute_normalization_reduction(1.0, 0.0) == np.inf

    def test_both_zero(self):
        # Cas parfaitement homogène
        assert compute_normalization_reduction(0.0, 0.0) == pytest.approx(1.0)

    def test_norm_larger_than_raw(self):
        # Normalisation qui aggrave la variabilité → facteur < 1
        result = compute_normalization_reduction(1.0, 2.0)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_sigma_probe
# ---------------------------------------------------------------------------

class TestSigmaProbe:
    def test_identical_replicates_returns_zero(self):
        p = np.array([1.0, 2.0, 3.0])
        assert compute_sigma_probe([p, p, p]) == pytest.approx(0.0, abs=1e-10)

    def test_single_replicate_returns_zero(self):
        p = np.array([1.0, 2.0])
        assert compute_sigma_probe([p]) == 0.0

    def test_empty_list_returns_zero(self):
        assert compute_sigma_probe([]) == 0.0

    def test_known_value(self):
        # std([1,2]) = 0.7071, mean([1,2]) = 1.5 → relative = 0.4714 → ~47 %
        p1 = np.array([1.0])
        p2 = np.array([2.0])
        result = compute_sigma_probe([p1, p2])
        # std ddof=1 = 0.7071, mean = 1.5, relative = 0.4714
        expected = (0.7071 / 1.5) * 100
        assert result == pytest.approx(expected, rel=1e-3)

    def test_returns_percent(self):
        # 10 % de variation connue
        p1 = np.ones(10) * 1.0
        p2 = np.ones(10) * 1.1   # +10 %
        result = compute_sigma_probe([p1, p2])
        assert 0 < result < 20  # dans un ordre de grandeur raisonnable

    def test_near_zero_mean_handled(self):
        # Ne doit pas lever d'erreur ni retourner nan/inf
        p1 = np.array([1e-12, 1.0])
        p2 = np.array([2e-12, 2.0])
        result = compute_sigma_probe([p1, p2])
        assert np.isfinite(result)
        assert result >= 0.0
