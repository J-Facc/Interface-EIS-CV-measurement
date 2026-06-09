"""Assemblage du rapport comparatif pour les 6 méthodes A1/A2/A3/B1/B2/B3.

Aucun import Streamlit. Ce module consomme session_data (dict produit par
render_data_input + pipeline de chargement), calcule toutes les métriques
et produit les figures Plotly.

Structure de session_data attendue
-----------------------------------
{
  "concentrations": [float, ...],          # calibration
  "n_electrodes": int,
  "n_replicats": int,

  # Signaux scalaires — un tableau par électrode × concentration
  # (après chargement et moyennage des réplicats)
  "eis_rct_fit":  {"e1": [float, ...], "e2": [float, ...]},   # Rct par fit Randles
  "eis_rct_drt":  {"e1": [float, ...], "e2": [float, ...]},   # Rct par DRT
  "cv_delta_I":   {"e1": [float, ...], "e2": [float, ...]},   # ΔI_pic normalisé

  # Signaux spectraux (pour PLS) — liste de np.ndarray par électrode × concentration
  "eis_spectra":  {"e1": [np.ndarray, ...], "e2": [...]},     # [Zre_norm, Zim_norm concaténés]
  "cv_spectra":   {"e1": [np.ndarray, ...], "e2": [...]},     # [I_norm]
  "freq_grid":    np.ndarray,    # fréquences communes EIS
  "pot_grid":     np.ndarray,    # potentiels communs CV

  # Probe — signaux scalaires de référence (pour sigma_inter_raw)
  "probe_rct_fit": float,
  "probe_rct_drt": float,
  "probe_delta_I": float,

  # Validation (optionnel — None si absent)
  "validation": None | {
    "concentrations": [float, ...],
    "eis_rct_fit":  {"e1": [float, ...], "e2": [...]},
    "eis_rct_drt":  {"e1": [float, ...], "e2": [...]},
    "cv_delta_I":   {"e1": [float, ...], "e2": [...]},
    "eis_spectra":  {"e1": [np.ndarray, ...], "e2": [...]},
    "cv_spectra":   {"e1": [np.ndarray, ...], "e2": [...]},
  }
}
"""

import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

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
)
from comparison.plots import (
    plot_calibration_scalar,
    plot_pls_scree,
    plot_pls_loadings,
    plot_predicted_vs_true,
    plot_variance_decomposition,
)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _safe_log10(x: float) -> float | None:
    """log10(x) ou None si x <= 0."""
    if x is not None and x > 0:
        return float(np.log10(x))
    return None


def _log10_array(arr) -> np.ndarray:
    """log10 sur un tableau, remplace les valeurs <= 0 par nan."""
    a = np.asarray(arr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, np.log10(a), np.nan)


def _ols_linregress(x: np.ndarray, y: np.ndarray):
    """linregress sécurisé — retourne None si moins de 2 points valides."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return stats.linregress(x[mask], y[mask])


def _pls_model_fn(n_components: int):
    """Retourne une model_fn compatible avec compute_rmsecv_logo."""
    def _fn(X_train, y_train, X_test):
        sc = StandardScaler()
        Xt = sc.fit_transform(X_train)
        Xv = sc.transform(X_test)
        m = PLSRegression(n_components=n_components, scale=False)
        m.fit(Xt, y_train)
        return m.predict(Xv).ravel()
    return _fn


def _build_pls_matrix(spectra_e1: list, spectra_e2: list | None) -> np.ndarray:
    """Empile les spectres EIS ou CV des deux électrodes en lignes de X."""
    rows = list(spectra_e1)
    if spectra_e2:
        rows += list(spectra_e2)
    return np.array([np.asarray(r, dtype=float) for r in rows], dtype=float)


def _groups_for_pls(n_conc: int, n_elec: int) -> np.ndarray:
    """Identifiant de groupe LOGO : même entier pour tous les réplicats d'une concentration."""
    groups_e1 = np.arange(n_conc)
    if n_elec > 1:
        return np.concatenate([groups_e1, groups_e1])
    return groups_e1


# ---------------------------------------------------------------------------
# Calcul des métriques scalaires (méthodes A)
# ---------------------------------------------------------------------------

def _metrics_scalar(
    signal_e1: np.ndarray,
    signal_e2: np.ndarray | None,
    signal_probe: float,
    concentrations: np.ndarray,
    signal_norm_e1: np.ndarray,
    signal_norm_e2: np.ndarray | None,
    val_norm_e1: np.ndarray | None,
    val_norm_e2: np.ndarray | None,
    val_concentrations: np.ndarray | None,
    method_name: str,
    theme_mode: str,
) -> dict:
    """
    Calcule toutes les métriques pour une méthode A (signal scalaire).

    signal_e1/e2     : signal brut en unité physique (Rct ou ΔI), par concentration
    signal_probe     : valeur probe de référence (pour sigma_inter_raw)
    signal_norm_e1/e2: signal normalisé par probe (pour calibration et sigma_inter_norm)
    val_norm_*       : idem pour les données de validation (None si absent)
    """
    log_c = _log10_array(concentrations)

    # --- sigma_intra : std des deux électrodes à chaque concentration ---
    if signal_norm_e2 is not None:
        per_conc = np.stack([signal_norm_e1, signal_norm_e2], axis=1)  # (n_conc, 2)
        sigma_intra = float(np.nanmean([compute_sigma_intra(row) for row in per_conc]))
    else:
        sigma_intra = 0.0

    # --- sigma_inter (brut) : variabilité avant normalisation ---
    if signal_e2 is not None and signal_probe > 0:
        raw_e1 = signal_e1 / signal_probe
        raw_e2 = signal_e2 / signal_probe
        sigma_inter_raw = float(np.nanmean([
            compute_sigma_inter(float(r1), float(r2))
            for r1, r2 in zip(raw_e1, raw_e2)
        ]))
    else:
        sigma_inter_raw = 0.0

    # --- sigma_inter (normalisé) ---
    if signal_norm_e2 is not None:
        sigma_inter_norm = float(np.nanmean([
            compute_sigma_inter(float(n1), float(n2))
            for n1, n2 in zip(signal_norm_e1, signal_norm_e2)
        ]))
    else:
        sigma_inter_norm = 0.0

    reduction_factor = compute_normalization_reduction(sigma_inter_raw, sigma_inter_norm)

    # --- RMSECV (OLS leave-one-out) ---
    norm_mean = signal_norm_e1.copy()
    if signal_norm_e2 is not None:
        norm_mean = (signal_norm_e1 + signal_norm_e2) / 2.0

    valid = np.isfinite(norm_mean) & np.isfinite(log_c)
    rmsecv = None
    if valid.sum() >= 3:
        x_cv = norm_mean[valid]
        y_cv = log_c[valid]
        n = len(x_cv)

        def _loo_fn(X_train, y_train, X_test):
            r = _ols_linregress(X_train[:, 0], y_train)
            if r is None:
                return np.full(len(X_test), np.nan)
            return r.slope * X_test[:, 0] + r.intercept

        rmsecv = compute_rmsecv_logo(
            x_cv.reshape(-1, 1), y_cv,
            groups=np.arange(n),
            model_fn=_loo_fn,
        )

    # --- Calibration figure ---
    calib_fig = plot_calibration_scalar(
        signal=norm_mean,
        concentrations=concentrations,
        method_name=method_name,
        theme_mode=theme_mode,
    )

    # --- Métriques de validation (si données présentes) ---
    rmsep = bias = rpd = ci95 = dm_tests = None

    if val_norm_e1 is not None and val_concentrations is not None:
        val_log_c = _log10_array(val_concentrations)
        val_norm_mean = val_norm_e1.copy()
        if val_norm_e2 is not None:
            val_norm_mean = (val_norm_e1 + val_norm_e2) / 2.0

        ols_fit = _ols_linregress(norm_mean[valid], log_c[valid])
        if ols_fit is not None:
            y_pred = ols_fit.slope * val_norm_mean + ols_fit.intercept
            valid_val = np.isfinite(y_pred) & np.isfinite(val_log_c)
            if valid_val.sum() >= 2:
                rmsep = compute_rmsep(y_pred[valid_val], val_log_c[valid_val])
                bias  = compute_bias(y_pred[valid_val], val_log_c[valid_val])
                rpd   = compute_rpd(val_log_c[valid_val], rmsep)
                ci95  = compute_ci95_bootstrap(y_pred[valid_val])
                dm_tests = {}   # rempli dans compute_full_report après calcul de tous les résidus

    return {
        "rmsecv":           rmsecv,
        "rmsep":            rmsep,
        "bias":             bias,
        "rpd":              rpd,
        "ci95":             ci95,
        "sigma_intra":      sigma_intra,
        "sigma_inter_raw":  sigma_inter_raw,
        "sigma_inter_norm": sigma_inter_norm,
        "reduction_factor": reduction_factor,
        "calibration_fig":  calib_fig,
        "dm_tests":         dm_tests,
        # Champs internes pour Diebold-Mariano cross-method
        "_norm_mean":       norm_mean,
        "_log_c":           log_c,
        "_val_log_c":       val_log_c if val_norm_e1 is not None else None,
        "_val_norm_mean":   val_norm_mean if val_norm_e1 is not None else None,
    }


# ---------------------------------------------------------------------------
# Calcul des métriques PLS (méthodes B)
# ---------------------------------------------------------------------------

def _metrics_pls(
    X_e1: list,
    X_e2: list | None,
    concentrations: np.ndarray,
    val_X_e1: list | None,
    val_X_e2: list | None,
    val_concentrations: np.ndarray | None,
    method_name: str,
    theme_mode: str,
    freq_grid: np.ndarray | None = None,
    pot_grid: np.ndarray | None = None,
) -> dict:
    """
    Calcule toutes les métriques pour une méthode B (spectre complet + PLS).
    """
    log_c = _log10_array(concentrations)
    n_conc = len(concentrations)
    n_elec = 2 if X_e2 else 1

    X = _build_pls_matrix(X_e1, X_e2)
    y = np.concatenate([log_c] * n_elec)
    groups = _groups_for_pls(n_conc, n_elec)

    valid_rows = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_rows]
    y = y[valid_rows]
    groups = groups[valid_rows]

    # --- select n_components via RMSECV LOGO ---
    n_unique_groups = len(np.unique(groups))
    max_k = min(4, X.shape[1], n_unique_groups - 1)
    max_k = max(1, max_k)

    best_k, best_rmsecv = 1, np.inf
    for k in range(1, max_k + 1):
        cv_err = compute_rmsecv_logo(X, y, groups, _pls_model_fn(k))
        if cv_err is not None and cv_err < best_rmsecv:
            best_rmsecv, best_k = cv_err, k

    rmsecv = best_rmsecv if np.isfinite(best_rmsecv) else None

    # --- Entraîner sur tout le jeu de calibration ---
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = PLSRegression(n_components=best_k, scale=False)
    model.fit(Xs, y)

    # --- Variance expliquée et loadings ---
    x_loadings = model.x_loadings_   # shape (n_features, n_components)
    x_scores   = model.x_scores_     # shape (n_samples, n_components)
    total_var  = np.var(Xs, axis=0).sum()
    explained_var = np.array([
        float(np.var(x_scores[:, k], axis=0) * np.sum(x_loadings[:, k] ** 2) / total_var)
        for k in range(best_k)
    ])
    explained_var = np.clip(explained_var, 0, 1)

    scree_fig = plot_pls_scree(explained_var, method_name, theme_mode)

    # Loadings figures
    loadings_figs = {}
    for comp in range(1, min(best_k + 1, 3)):
        lv = x_loadings[:, comp - 1]
        if freq_grid is not None and len(lv) >= len(freq_grid):
            # Premiers n_freq features = Zre, suivants = Zim (ou seulement Zre selon concaténation)
            n_f = len(freq_grid)
            lf = lv[:n_f]
            loadings_figs[f"eis_c{comp}"] = plot_pls_loadings(
                lf, freq_grid, "Fréquence (Hz)", component=comp, theme_mode=theme_mode
            )
        if pot_grid is not None and len(lv) > (len(freq_grid) if freq_grid is not None else 0):
            offset = len(freq_grid) * 2 if freq_grid is not None else 0
            if offset < len(lv):
                lp = lv[offset: offset + len(pot_grid)]
                if len(lp) == len(pot_grid):
                    loadings_figs[f"cv_c{comp}"] = plot_pls_loadings(
                        lp, pot_grid, "Potentiel (V)", component=comp, theme_mode=theme_mode
                    )

    # --- sigma_intra / sigma_inter ---
    y_hat = model.predict(Xs).ravel()
    per_conc_preds = [y_hat[groups == g] for g in np.unique(groups)]
    sigma_intra = float(np.nanmean([compute_sigma_intra(p) for p in per_conc_preds]))

    # Inter-électrode — uniquement si deux électrodes
    if n_elec > 1 and n_conc > 0:
        half = len(y) // 2
        preds_e1 = y_hat[:half]
        preds_e2 = y_hat[half:]
        n_pairs = min(len(preds_e1), len(preds_e2))
        sigma_inter_norm = float(np.nanmean([
            compute_sigma_inter(float(preds_e1[i]), float(preds_e2[i]))
            for i in range(n_pairs)
        ]))
        # sigma_inter_raw : sur les features avant normalisation (std des X bruts)
        Xraw = X.copy()
        raw_e1 = Xraw[:half]
        raw_e2 = Xraw[half:half + n_pairs]
        sigma_inter_raw = float(np.nanmean([
            compute_sigma_inter(float(np.mean(raw_e1[i])), float(np.mean(raw_e2[i])))
            for i in range(n_pairs)
        ]))
    else:
        sigma_inter_raw = sigma_inter_norm = 0.0

    reduction_factor = compute_normalization_reduction(sigma_inter_raw, sigma_inter_norm)

    # --- Validation ---
    rmsep = bias = rpd = ci95 = dm_tests = None
    val_log_c = val_y_pred = None

    if val_X_e1 is not None and val_concentrations is not None:
        val_log_c = _log10_array(val_concentrations)
        val_X = _build_pls_matrix(val_X_e1, val_X_e2)
        val_X_s = sc.transform(val_X)
        val_y_pred = model.predict(val_X_s).ravel()

        # Répliquer val_log_c si deux électrodes fournies
        n_val_elec = 2 if val_X_e2 else 1
        val_y_true = np.concatenate([val_log_c] * n_val_elec)
        val_y_pred_all = val_y_pred

        valid_v = np.isfinite(val_y_pred_all) & np.isfinite(val_y_true)
        if valid_v.sum() >= 2:
            rmsep = compute_rmsep(val_y_pred_all[valid_v], val_y_true[valid_v])
            bias  = compute_bias(val_y_pred_all[valid_v], val_y_true[valid_v])
            rpd   = compute_rpd(val_y_true[valid_v], rmsep)
            ci95  = compute_ci95_bootstrap(val_y_pred_all[valid_v])
            dm_tests = {}

    return {
        "rmsecv":           rmsecv,
        "rmsep":            rmsep,
        "bias":             bias,
        "rpd":              rpd,
        "ci95":             ci95,
        "sigma_intra":      sigma_intra,
        "sigma_inter_raw":  sigma_inter_raw,
        "sigma_inter_norm": sigma_inter_norm,
        "reduction_factor": reduction_factor,
        "calibration_fig":  scree_fig,
        "loadings_figs":    loadings_figs,
        "dm_tests":         dm_tests,
        "n_components":     best_k,
        # Champs internes pour Diebold-Mariano
        "_val_log_c":       val_log_c,
        "_val_y_pred":      val_y_pred,
    }


# ---------------------------------------------------------------------------
# Tests de Diebold-Mariano croisés
# ---------------------------------------------------------------------------

_DM_PAIRS = [
    ("A2", "A3"),
    ("A2", "B2"),
    ("B2", "B3"),
    ("A1", "B1"),
    ("B1", "B2"),
]


def _compute_dm_tests(methods_results: dict) -> None:
    """
    Remplit dm_tests dans chaque méthode pour les paires définies dans _DM_PAIRS.
    Modifie methods_results en place. Requiert les données de validation.
    """
    for m_a, m_b in _DM_PAIRS:
        if m_a not in methods_results or m_b not in methods_results:
            continue

        res_a = methods_results[m_a]
        res_b = methods_results[m_b]

        # Récupère les résidus de prédiction de chaque méthode
        errors_a = _get_val_errors(res_a)
        errors_b = _get_val_errors(res_b)

        if errors_a is None or errors_b is None:
            continue
        if len(errors_a) != len(errors_b) or len(errors_a) < 2:
            continue

        stat, pval = diebold_mariano_test(errors_a, errors_b)
        if res_a.get("dm_tests") is None:
            res_a["dm_tests"] = {}
        if res_b.get("dm_tests") is None:
            res_b["dm_tests"] = {}
        res_a["dm_tests"][m_b] = (stat, pval)
        res_b["dm_tests"][m_a] = (-stat, pval)  # symétrie signée


def _get_val_errors(result: dict) -> np.ndarray | None:
    """Extrait les erreurs de prédiction (y_pred - y_true) depuis les champs internes."""
    # Méthodes A
    if "_val_norm_mean" in result and result["_val_norm_mean"] is not None:
        x = result["_val_norm_mean"]
        y_true = result["_val_log_c"]
        if y_true is None:
            return None
        # recalcule la prédiction OLS sur les données de validation
        calib_x = result["_norm_mean"]
        calib_y = result["_log_c"]
        mask = np.isfinite(calib_x) & np.isfinite(calib_y)
        if mask.sum() < 2:
            return None
        r = stats.linregress(calib_x[mask], calib_y[mask])
        y_pred = r.slope * x + r.intercept
        valid = np.isfinite(y_pred) & np.isfinite(y_true)
        if valid.sum() < 2:
            return None
        return y_pred[valid] - y_true[valid]

    # Méthodes B
    if "_val_y_pred" in result and result["_val_y_pred"] is not None:
        y_pred = result["_val_y_pred"]
        y_true = result["_val_log_c"]
        if y_true is None:
            return None
        n = min(len(y_pred), len(y_true))
        errors = y_pred[:n] - y_true[:n]
        valid = np.isfinite(errors)
        return errors[valid] if valid.sum() >= 2 else None

    return None


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def compute_full_report(session_data: dict, config=None) -> dict:
    """
    Calcule toutes les métriques pour les 6 méthodes A1/A2/A3/B1/B2/B3.

    Parameters
    ----------
    session_data : dict
        Structure décrite dans le docstring de module.
    config : object | None
        Configuration de l'application (non utilisé pour l'instant,
        réservé pour paramètres physiques futurs).

    Returns
    -------
    dict
        {
          "methods": {
            "A1": { rmsecv, rmsep, bias, rpd, ci95, sigma_intra,
                    sigma_inter_raw, sigma_inter_norm, reduction_factor,
                    calibration_fig, dm_tests },
            "A2": { ... },
            "A3": { ... },
            "B1": { ... },
            "B2": { ... },
            "B3": { ... },
          },
          "figures": {
            "predicted_vs_true":      go.Figure,
            "variance_decomposition": go.Figure,
          }
        }

    Métriques non calculables sans validation (rmsep, bias, rpd, ci95, dm_tests)
    sont mises à None.
    """
    theme_mode = session_data.get("theme_mode", "light")
    concs   = np.asarray(session_data["concentrations"], dtype=float)
    val_data = session_data.get("validation")

    def _none_if_empty(arr):
        return arr if arr is not None and len(arr) > 0 else None

    def _norm(sig, probe_val):
        """Normalise un tableau par la valeur probe : (sig - probe) / probe."""
        if probe_val and probe_val > 0:
            return (sig - probe_val) / probe_val
        return sig.copy()

    # Signaux scalaires calibration
    rct_fit_e1  = np.asarray(session_data["eis_rct_fit"]["e1"], dtype=float)
    rct_fit_e2  = _none_if_empty(np.asarray(session_data["eis_rct_fit"].get("e2") or [], dtype=float))
    rct_drt_e1  = np.asarray(session_data["eis_rct_drt"]["e1"], dtype=float)
    rct_drt_e2  = _none_if_empty(np.asarray(session_data["eis_rct_drt"].get("e2") or [], dtype=float))
    cv_dI_e1    = np.asarray(session_data["cv_delta_I"]["e1"], dtype=float)
    cv_dI_e2    = _none_if_empty(np.asarray(session_data["cv_delta_I"].get("e2") or [], dtype=float))

    probe_rct_fit = float(session_data.get("probe_rct_fit") or 0)
    probe_rct_drt = float(session_data.get("probe_rct_drt") or 0)
    probe_dI      = float(session_data.get("probe_delta_I") or 0)

    norm_rct_fit_e1 = _norm(rct_fit_e1, probe_rct_fit)
    norm_rct_fit_e2 = _norm(rct_fit_e2, probe_rct_fit) if rct_fit_e2 is not None else None
    norm_rct_drt_e1 = _norm(rct_drt_e1, probe_rct_drt)
    norm_rct_drt_e2 = _norm(rct_drt_e2, probe_rct_drt) if rct_drt_e2 is not None else None
    norm_dI_e1      = _norm(cv_dI_e1, probe_dI)
    norm_dI_e2      = _norm(cv_dI_e2, probe_dI) if cv_dI_e2 is not None else None

    # Signaux validation
    val_rct_fit_e1 = val_rct_drt_e1 = val_dI_e1 = None
    val_rct_fit_e2 = val_rct_drt_e2 = val_dI_e2 = None
    val_concs = None
    val_eis_X_e1 = val_eis_X_e2 = val_cv_X_e1 = val_cv_X_e2 = None

    if val_data:
        val_concs      = np.asarray(val_data["concentrations"], dtype=float)
        val_rct_fit_e1 = np.asarray(val_data["eis_rct_fit"]["e1"], dtype=float)
        val_rct_drt_e1 = np.asarray(val_data["eis_rct_drt"]["e1"], dtype=float)
        val_dI_e1      = np.asarray(val_data["cv_delta_I"]["e1"], dtype=float)
        val_rct_fit_e1 = _norm(val_rct_fit_e1, probe_rct_fit)
        val_rct_drt_e1 = _norm(val_rct_drt_e1, probe_rct_drt)
        val_dI_e1      = _norm(val_dI_e1, probe_dI)

        val_rct_fit_e2 = _none_if_empty(_norm(np.asarray(val_data["eis_rct_fit"].get("e2") or [], dtype=float), probe_rct_fit))
        val_rct_drt_e2 = _none_if_empty(_norm(np.asarray(val_data["eis_rct_drt"].get("e2") or [], dtype=float), probe_rct_drt))
        val_dI_e2      = _none_if_empty(_norm(np.asarray(val_data["cv_delta_I"].get("e2") or [], dtype=float), probe_dI))

        val_eis_X_e1 = val_data.get("eis_spectra", {}).get("e1")
        val_eis_X_e2 = val_data.get("eis_spectra", {}).get("e2")
        val_cv_X_e1  = val_data.get("cv_spectra", {}).get("e1")
        val_cv_X_e2  = val_data.get("cv_spectra", {}).get("e2")

    freq_grid = session_data.get("freq_grid")
    pot_grid  = session_data.get("pot_grid")
    eis_X_e1  = session_data.get("eis_spectra", {}).get("e1", [])
    eis_X_e2  = session_data.get("eis_spectra", {}).get("e2")
    cv_X_e1   = session_data.get("cv_spectra", {}).get("e1", [])
    cv_X_e2   = session_data.get("cv_spectra", {}).get("e2")

    # Construire les features spectrales EIS+CV pour B3
    def _concat_eis_cv(eis_list, cv_list):
        if not eis_list or not cv_list:
            return eis_list or cv_list
        n = min(len(eis_list), len(cv_list))
        return [np.concatenate([np.asarray(eis_list[i]), np.asarray(cv_list[i])]) for i in range(n)]

    b3_X_e1 = _concat_eis_cv(eis_X_e1, cv_X_e1)
    b3_X_e2 = _concat_eis_cv(eis_X_e2, cv_X_e2) if eis_X_e2 and cv_X_e2 else None
    val_b3_X_e1 = _concat_eis_cv(val_eis_X_e1, val_cv_X_e1) if val_eis_X_e1 and val_cv_X_e1 else None
    val_b3_X_e2 = _concat_eis_cv(val_eis_X_e2, val_cv_X_e2) if val_eis_X_e2 and val_cv_X_e2 else None

    # -----------------------------------------------------------------------
    # Calcul des 6 méthodes
    # -----------------------------------------------------------------------
    methods_results: dict = {}

    methods_results["A1"] = _metrics_scalar(
        cv_dI_e1, cv_dI_e2, probe_dI, concs,
        norm_dI_e1, norm_dI_e2,
        val_dI_e1, val_dI_e2, val_concs,
        "A1", theme_mode,
    )
    methods_results["A2"] = _metrics_scalar(
        rct_fit_e1, rct_fit_e2, probe_rct_fit, concs,
        norm_rct_fit_e1, norm_rct_fit_e2,
        val_rct_fit_e1, val_rct_fit_e2, val_concs,
        "A2", theme_mode,
    )
    methods_results["A3"] = _metrics_scalar(
        rct_drt_e1, rct_drt_e2, probe_rct_drt, concs,
        norm_rct_drt_e1, norm_rct_drt_e2,
        val_rct_drt_e1, val_rct_drt_e2, val_concs,
        "A3", theme_mode,
    )

    if eis_X_e1:
        methods_results["B1"] = _metrics_pls(
            cv_X_e1, cv_X_e2, concs,
            val_cv_X_e1, val_cv_X_e2, val_concs,
            "B1", theme_mode, freq_grid=None, pot_grid=pot_grid,
        )
        methods_results["B2"] = _metrics_pls(
            eis_X_e1, eis_X_e2, concs,
            val_eis_X_e1, val_eis_X_e2, val_concs,
            "B2", theme_mode, freq_grid=freq_grid, pot_grid=None,
        )
        if b3_X_e1:
            methods_results["B3"] = _metrics_pls(
                b3_X_e1, b3_X_e2, concs,
                val_b3_X_e1, val_b3_X_e2, val_concs,
                "B3", theme_mode, freq_grid=freq_grid, pot_grid=pot_grid,
            )

    # -----------------------------------------------------------------------
    # Diebold-Mariano croisés (requiert validation)
    # -----------------------------------------------------------------------
    _compute_dm_tests(methods_results)

    # -----------------------------------------------------------------------
    # Nettoyage des champs internes avant retour
    # -----------------------------------------------------------------------
    _INTERNAL_KEYS = {"_norm_mean", "_log_c", "_val_log_c", "_val_norm_mean", "_val_y_pred"}
    public_results: dict = {}
    for m, res in methods_results.items():
        public_results[m] = {k: v for k, v in res.items() if k not in _INTERNAL_KEYS}

    # -----------------------------------------------------------------------
    # Figures globales
    # -----------------------------------------------------------------------
    # predicted_vs_true — uniquement si validation disponible
    pred_fig = _build_pred_vs_true(methods_results, theme_mode)

    # variance decomposition
    var_metrics = {
        m: {
            "sigma_intra":      res.get("sigma_intra"),
            "sigma_inter_raw":  res.get("sigma_inter_raw"),
            "sigma_inter_norm": res.get("sigma_inter_norm"),
        }
        for m, res in methods_results.items()
    }
    var_fig = plot_variance_decomposition(var_metrics, theme_mode)

    return {
        "methods": public_results,
        "figures": {
            "predicted_vs_true":      pred_fig,
            "variance_decomposition": var_fig,
        },
    }


def _build_pred_vs_true(methods_results: dict, theme_mode: str) -> go.Figure:
    """Construit la figure predicted_vs_true à partir des champs internes."""
    results_by_method = {}

    for m, res in methods_results.items():
        # Méthode A : recalcule y_pred sur la validation
        if "_val_norm_mean" in res and res["_val_norm_mean"] is not None:
            x_cal = res["_norm_mean"]
            y_cal = res["_log_c"]
            mask = np.isfinite(x_cal) & np.isfinite(y_cal)
            if mask.sum() < 2:
                continue
            r = stats.linregress(x_cal[mask], y_cal[mask])
            y_pred = r.slope * res["_val_norm_mean"] + r.intercept
            y_true = res["_val_log_c"]
            if y_true is None:
                continue
            valid = np.isfinite(y_pred) & np.isfinite(y_true)
            results_by_method[m] = {
                "y_pred": y_pred[valid],
                "y_true": y_true[valid],
                "y_std":  np.zeros(valid.sum()),
            }

        # Méthode B
        elif "_val_y_pred" in res and res["_val_y_pred"] is not None:
            y_pred = res["_val_y_pred"]
            y_true = res["_val_log_c"]
            if y_true is None:
                continue
            n = min(len(y_pred), len(y_true))
            valid = np.isfinite(y_pred[:n]) & np.isfinite(y_true[:n])
            results_by_method[m] = {
                "y_pred": y_pred[:n][valid],
                "y_true": y_true[:n][valid],
                "y_std":  np.zeros(valid.sum()),
            }

    if not results_by_method:
        fig = go.Figure()
        fig.update_layout(title="Données de validation absentes — figure non disponible")
        return fig

    return plot_predicted_vs_true(results_by_method, theme_mode)
