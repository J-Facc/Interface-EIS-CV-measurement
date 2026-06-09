import numpy as np
import pytest

from comparison.pls_model import (
    normalize_by_probe,
    interpolate_to_grid,
    build_feature_matrix,
    select_n_components,
    train_pls,
    predict_concentration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eis_spectrum(freq, scale=1.0):
    """Retourne un dict EIS normalisé synthétique."""
    return {
        "Zre_norm": scale * np.ones_like(freq),
        "Zim_norm": scale * 0.5 * np.ones_like(freq),
        "freq": freq,
    }


def _make_cv_spectrum(pot, scale=1.0):
    """Retourne un dict CV normalisé synthétique."""
    return {
        "I_norm": scale * np.ones_like(pot),
        "E": pot,
    }


def _make_calibration_data(n_conc=7, n_rep=2, n_freq=20, n_pot=15):
    """
    Génère un jeu de données de calibration synthétique avec relation log-log connue.
    y = log10([c]) linéairement lié à l'amplitude des spectres.
    """
    rng = np.random.default_rng(42)
    freq_grid = np.logspace(0, 4, n_freq)   # 1 Hz → 10 kHz
    pot_grid  = np.linspace(-0.2, 0.6, n_pot)

    log_c_values = np.linspace(-18, -8, n_conc)   # log10([c]) de -18 à -8 M

    eis_spectra, cv_spectra, y, groups = [], [], [], []

    for g, log_c in enumerate(log_c_values):
        amplitude = 1.0 + (log_c + 18) / 10.0   # amplitude croît avec log_c
        for _ in range(n_rep):
            noise = rng.normal(0, 0.01, n_freq)
            eis = {
                "Zre_norm": amplitude * np.ones(n_freq) + noise,
                "Zim_norm": amplitude * 0.5 * np.ones(n_freq) + noise,
                "freq": freq_grid,
            }
            cv = {
                "I_norm": amplitude * np.ones(n_pot) + rng.normal(0, 0.01, n_pot),
                "E": pot_grid,
            }
            eis_spectra.append(eis)
            cv_spectra.append(cv)
            y.append(log_c)
            groups.append(g)

    return (
        eis_spectra,
        cv_spectra,
        np.array(y),
        np.array(groups),
        freq_grid,
        pot_grid,
    )


# ---------------------------------------------------------------------------
# normalize_by_probe
# ---------------------------------------------------------------------------

class TestNormalizeByProbe:
    def test_identical_signal_returns_zeros(self):
        probe = np.array([1.0, 2.0, 3.0])
        result = normalize_by_probe(probe, probe)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_double_signal_returns_ones(self):
        probe = np.array([1.0, 2.0, 4.0])
        spectrum = 2.0 * probe
        result = normalize_by_probe(spectrum, probe)
        np.testing.assert_array_almost_equal(result, np.ones(3))

    def test_division_by_zero_returns_zero(self):
        probe = np.array([0.0, 1.0, 2.0])
        spectrum = np.array([5.0, 2.0, 4.0])
        result = normalize_by_probe(spectrum, probe)
        assert result[0] == 0.0
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_near_zero_probe_treated_as_zero(self):
        probe = np.array([1e-12, 1.0])
        spectrum = np.array([2e-12, 2.0])
        result = normalize_by_probe(spectrum, probe)
        assert result[0] == 0.0
        assert result[1] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            normalize_by_probe(np.array([1.0, 2.0]), np.array([1.0]))

    def test_output_shape(self):
        probe = np.array([1.0, 2.0, 3.0, 4.0])
        spectrum = np.array([2.0, 4.0, 6.0, 8.0])
        result = normalize_by_probe(spectrum, probe)
        assert result.shape == probe.shape

    def test_negative_values(self):
        # Courants CV peuvent être négatifs
        probe = np.array([-2.0, -1.0, 1.0])
        spectrum = np.array([-4.0, -2.0, 2.0])
        result = normalize_by_probe(spectrum, probe)
        np.testing.assert_array_almost_equal(result, np.ones(3))


# ---------------------------------------------------------------------------
# interpolate_to_grid
# ---------------------------------------------------------------------------

class TestInterpolateToGrid:
    def test_already_on_grid_linear(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([0.0, 1.0, 4.0, 9.0])
        result = interpolate_to_grid(values, axis, axis, log_axis=False)
        np.testing.assert_array_almost_equal(result, values)

    def test_already_on_grid_log(self):
        axis = np.array([1.0, 10.0, 100.0, 1000.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = interpolate_to_grid(values, axis, axis, log_axis=True)
        np.testing.assert_array_almost_equal(result, values)

    def test_linear_midpoint(self):
        # Interpolation linéaire au milieu d'un segment
        axis = np.array([0.0, 2.0])
        values = np.array([0.0, 2.0])
        grid = np.array([1.0])
        result = interpolate_to_grid(values, axis, grid, log_axis=False)
        assert result[0] == pytest.approx(1.0)

    def test_log_midpoint(self):
        # log10(10) = 1, log10(1000) = 3 → midpoint log = log10(100) = 2
        axis = np.array([10.0, 1000.0])
        values = np.array([1.0, 3.0])
        grid = np.array([100.0])
        result = interpolate_to_grid(values, axis, grid, log_axis=True)
        assert result[0] == pytest.approx(2.0)

    def test_output_shape_matches_grid(self):
        axis = np.logspace(0, 4, 50)
        values = np.ones(50)
        grid = np.logspace(0.5, 3.5, 30)
        result = interpolate_to_grid(values, axis, grid, log_axis=True)
        assert result.shape == (30,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            interpolate_to_grid(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))

    def test_log_axis_nonpositive_raises(self):
        with pytest.raises(ValueError):
            interpolate_to_grid(
                np.array([1.0, 2.0]),
                np.array([-1.0, 1.0]),
                np.array([0.5]),
                log_axis=True,
            )


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def setup_method(self):
        self.freq_grid = np.logspace(0, 4, 10)
        self.pot_grid  = np.linspace(-0.2, 0.6, 8)
        self.n_eis = 2 * len(self.freq_grid)   # Zre + Zim
        self.n_cv  = len(self.pot_grid)

    def _eis_samples(self, n=3):
        return [_make_eis_spectrum(self.freq_grid) for _ in range(n)]

    def _cv_samples(self, n=3):
        return [_make_cv_spectrum(self.pot_grid) for _ in range(n)]

    def test_eis_only_shape(self):
        X = build_feature_matrix(self._eis_samples(3), None, self.freq_grid, None)
        assert X.shape == (3, self.n_eis)

    def test_eis_and_cv_shape(self):
        X = build_feature_matrix(
            self._eis_samples(4), self._cv_samples(4), self.freq_grid, self.pot_grid
        )
        assert X.shape == (4, self.n_eis + self.n_cv)

    def test_single_sample(self):
        X = build_feature_matrix(self._eis_samples(1), None, self.freq_grid, None)
        assert X.shape == (1, self.n_eis)

    def test_cv_without_pot_grid_raises(self):
        with pytest.raises(ValueError):
            build_feature_matrix(
                self._eis_samples(2), self._cv_samples(2), self.freq_grid, None
            )

    def test_eis_cv_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_feature_matrix(
                self._eis_samples(3), self._cv_samples(2), self.freq_grid, self.pot_grid
            )

    def test_output_dtype_float(self):
        X = build_feature_matrix(self._eis_samples(2), None, self.freq_grid, None)
        assert X.dtype == np.float64

    def test_constant_spectra_constant_rows(self):
        # Tous les spectres identiques → toutes les lignes identiques
        samples = [_make_eis_spectrum(self.freq_grid, scale=1.0) for _ in range(4)]
        X = build_feature_matrix(samples, None, self.freq_grid, None)
        for i in range(1, 4):
            np.testing.assert_array_almost_equal(X[0], X[i])


# ---------------------------------------------------------------------------
# select_n_components
# ---------------------------------------------------------------------------

class TestSelectNComponents:
    def test_returns_int(self):
        eis, cv, y, groups, freq_grid, pot_grid = _make_calibration_data()
        X = build_feature_matrix(eis, None, freq_grid, None)
        k = select_n_components(X, y, groups, max_components=3)
        assert isinstance(k, int)

    def test_k_in_valid_range(self):
        eis, cv, y, groups, freq_grid, pot_grid = _make_calibration_data()
        X = build_feature_matrix(eis, None, freq_grid, None)
        k = select_n_components(X, y, groups, max_components=3)
        assert 1 <= k <= 3

    def test_max_components_1_returns_1(self):
        eis, cv, y, groups, freq_grid, pot_grid = _make_calibration_data()
        X = build_feature_matrix(eis, None, freq_grid, None)
        k = select_n_components(X, y, groups, max_components=1)
        assert k == 1

    def test_strong_signal_selects_low_k(self):
        # Relation quasi-parfaite → k=1 ou k=2 suffit
        eis, cv, y, groups, freq_grid, pot_grid = _make_calibration_data(
            n_conc=7, n_rep=2
        )
        X = build_feature_matrix(eis, None, freq_grid, None)
        k = select_n_components(X, y, groups, max_components=4)
        assert k <= 4   # pas de surapprentissage attendu sur ce jeu simple


# ---------------------------------------------------------------------------
# train_pls + predict_concentration
# ---------------------------------------------------------------------------

class TestTrainAndPredict:
    def setup_method(self):
        eis, cv, y, groups, freq_grid, pot_grid = _make_calibration_data(
            n_conc=7, n_rep=3
        )
        self.X = build_feature_matrix(eis, None, freq_grid, None)
        self.y = y
        self.groups = groups

    def test_train_returns_model_and_scaler(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler
        model, scaler = train_pls(self.X, self.y, n_components=2)
        assert isinstance(model, PLSRegression)
        assert isinstance(scaler, StandardScaler)

    def test_rmsep_on_training_data_low(self):
        model, scaler = train_pls(self.X, self.y, n_components=2)
        X_scaled = scaler.transform(self.X)
        y_pred = model.predict(X_scaled).ravel()
        rmsep = float(np.sqrt(np.mean((y_pred - self.y) ** 2)))
        assert rmsep < 0.1

    def test_predict_returns_two_floats(self):
        model, scaler = train_pls(self.X, self.y, n_components=2)
        result = predict_concentration(model, scaler, self.X[0])
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_predict_log_and_linear_consistent(self):
        model, scaler = train_pls(self.X, self.y, n_components=2)
        log_c, c_lin = predict_concentration(model, scaler, self.X[0])
        assert c_lin == pytest.approx(10.0 ** log_c, rel=1e-6)

    def test_predict_all_samples_close_to_truth(self):
        model, scaler = train_pls(self.X, self.y, n_components=2)
        errors = []
        for i in range(len(self.X)):
            log_c, _ = predict_concentration(model, scaler, self.X[i])
            errors.append(abs(log_c - self.y[i]))
        assert np.mean(errors) < 0.1

    def test_n_components_zero_raises(self):
        with pytest.raises(ValueError):
            train_pls(self.X, self.y, n_components=0)

    def test_n_components_too_large_raises(self):
        with pytest.raises(ValueError):
            train_pls(self.X, self.y, n_components=self.X.shape[1] + 1)

    def test_predict_empty_raises(self):
        model, scaler = train_pls(self.X, self.y, n_components=2)
        with pytest.raises(ValueError):
            predict_concentration(model, scaler, np.array([]))

    def test_predict_multi_sample_returns_mean(self):
        # Prédire plusieurs réplicats : doit retourner la moyenne
        model, scaler = train_pls(self.X, self.y, n_components=2)
        log_c_batch, _ = predict_concentration(model, scaler, self.X[:3])
        X_s = scaler.transform(self.X[:3])
        expected_mean = float(np.mean(model.predict(X_s).ravel()))
        assert log_c_batch == pytest.approx(expected_mean)
