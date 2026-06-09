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

    scree_fig = plot_pls_scree(explained_var, method_name)

    # Loadings figures
    loadings_figs = {}
    for comp in range(1, min(best_k + 1, 3)):
        lv = x_loadings[:, comp - 1]
        if freq_grid is not None and len(lv) >= len(freq_grid):
            # Premiers n_freq features = Zre, suivants = Zim (ou seulement Zre selon concaténation)
            n_f = len(freq_grid)
            lf = lv[:n_f]
            loadings_figs[f"eis_c{comp}"] = plot_pls_loadings(
                lf, freq_grid, "Fréquence (Hz)", component=comp
            )
        if pot_grid is not None and len(lv) > (len(freq_grid) if freq_grid is not None else 0):
            offset = len(freq_grid) * 2 if freq_grid is not None else 0
            if offset < len(lv):
                lp = lv[offset: offset + len(pot_grid)]
                if len(lp) == len(pot_grid):
                    loadings_figs[f"cv_c{comp}"] = plot_pls_loadings(
                        lp, pot_grid, "Potentiel (V)", component=comp
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
        val_dI_e1, val_dI_e2, val_concs, "A1",
    )
    methods_results["A2"] = _metrics_scalar(
        rct_fit_e1, rct_fit_e2, probe_rct_fit, concs,
        norm_rct_fit_e1, norm_rct_fit_e2,
        val_rct_fit_e1, val_rct_fit_e2, val_concs, "A2",
    )
    methods_results["A3"] = _metrics_scalar(
        rct_drt_e1, rct_drt_e2, probe_rct_drt, concs,
        norm_rct_drt_e1, norm_rct_drt_e2,
        val_rct_drt_e1, val_rct_drt_e2, val_concs, "A3",
    )

    if eis_X_e1:
        methods_results["B1"] = _metrics_pls(
            cv_X_e1, cv_X_e2, concs,
            val_cv_X_e1, val_cv_X_e2, val_concs,
            "B1", freq_grid=None, pot_grid=pot_grid,
        )
        methods_results["B2"] = _metrics_pls(
            eis_X_e1, eis_X_e2, concs,
            val_eis_X_e1, val_eis_X_e2, val_concs,
            "B2", freq_grid=freq_grid, pot_grid=None,
        )
        if b3_X_e1:
            methods_results["B3"] = _metrics_pls(
                b3_X_e1, b3_X_e2, concs,
                val_b3_X_e1, val_b3_X_e2, val_concs,
                "B3", freq_grid=freq_grid, pot_grid=pot_grid,
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
    pred_fig = _build_pred_vs_true(methods_results)

    # variance decomposition
    var_metrics = {
        m: {
            "sigma_intra":      res.get("sigma_intra"),
            "sigma_inter_raw":  res.get("sigma_inter_raw"),
            "sigma_inter_norm": res.get("sigma_inter_norm"),
        }
        for m, res in methods_results.items()
    }
    var_fig = plot_variance_decomposition(var_metrics)

    return {
        "methods": public_results,
        "figures": {
            "predicted_vs_true":      pred_fig,
            "variance_decomposition": var_fig,
        },
    }


def _build_pred_vs_true(methods_results: dict) -> go.Figure:
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

    return plot_predicted_vs_true(results_by_method)


# ---------------------------------------------------------------------------
# Inférence sur une nouvelle mesure
# ---------------------------------------------------------------------------

def predict_from_session(
    session_data: dict,
    new_signals: dict,
) -> dict:
    """
    Prédit la concentration d'une mesure inconnue à partir d'une session de calibration.

    Les modèles OLS (méthodes A) et PLS (méthodes B) sont reconstruits à partir
    de session_data. La normalisation utilise le probe de la NOUVELLE mesure
    (pas le probe de calibration) pour capturer l'état actuel de la surface.

    Parameters
    ----------
    session_data : dict
        Session de calibration produite par _load_session_data() dans C_comparatif.
        Doit contenir : concentrations, eis_rct_fit, eis_rct_drt, cv_delta_I,
        eis_spectra, cv_spectra, freq_grid, pot_grid,
        probe_rct_fit, probe_rct_drt, probe_delta_I.
    new_signals : dict
        Signaux de la mesure inconnue :
        {
          "rct_fit_e1": float | None,    # Rct par fit Randles, électrode 1
          "rct_fit_e2": float | None,    # idem électrode 2
          "rct_drt_e1": float | None,    # Rct par DRT, électrode 1
          "rct_drt_e2": float | None,    # idem électrode 2
          "delta_I_e1": float | None,    # ΔI_pic CV, électrode 1
          "delta_I_e2": float | None,    # idem électrode 2
          "eis_feat_e1": np.ndarray | None,  # [Zre_norm, Zim_norm] interpolés
          "eis_feat_e2": np.ndarray | None,
          "cv_feat_e1":  np.ndarray | None,  # I_norm interpolé
          "cv_feat_e2":  np.ndarray | None,
          "probe_rct_fit": float,    # Rct probe nouvelle mesure (Randles)
          "probe_rct_drt": float,    # Rct probe nouvelle mesure (DRT)
          "probe_delta_I": float,    # I_pic probe nouvelle mesure (CV)
        }

    Returns
    -------
    dict
        {
          method_name: {
            "log10_c_e1":   float | None,   # prédiction log10([c]) électrode 1
            "log10_c_e2":   float | None,   # prédiction log10([c]) électrode 2
            "log10_c_mean": float | None,   # moyenne des deux électrodes
            "c_mean":       float | None,   # 10 ** log10_c_mean (mol/L)
            "ci_factor":    float | None,   # facteur multiplicatif IC 95 % (= 10^(1.96*sigma))
            "sigma_pred":   float | None,   # incertitude en décades (RMSEP ou RMSECV)
            "coherent":     bool,           # True si |pred_e1 - pred_e2| < 2 * sigma_inter_calib
            "delta_elec":   float | None,   # |pred_e1 - pred_e2|
          },
          ...
        }
    """
    concs    = np.asarray(session_data["concentrations"], dtype=float)
    log_c    = _log10_array(concs)

    probe_rct_fit_new = float(new_signals.get("probe_rct_fit") or 0)
    probe_rct_drt_new = float(new_signals.get("probe_rct_drt") or 0)
    probe_dI_new      = float(new_signals.get("probe_delta_I") or 0)

    def _norm_scalar(val, probe_new):
        if val is None or probe_new == 0:
            return None
        return (float(val) - probe_new) / abs(probe_new)

    # Signaux normalisés de la nouvelle mesure
    new_norm = {
        "A1_e1": _norm_scalar(new_signals.get("delta_I_e1"),  probe_dI_new),
        "A1_e2": _norm_scalar(new_signals.get("delta_I_e2"),  probe_dI_new),
        "A2_e1": _norm_scalar(new_signals.get("rct_fit_e1"),  probe_rct_fit_new),
        "A2_e2": _norm_scalar(new_signals.get("rct_fit_e2"),  probe_rct_fit_new),
        "A3_e1": _norm_scalar(new_signals.get("rct_drt_e1"),  probe_rct_drt_new),
        "A3_e2": _norm_scalar(new_signals.get("rct_drt_e2"),  probe_rct_drt_new),
    }

    # Reconstruit les OLS de calibration (méthodes A)
    def _ols_model(raw_e1_list, raw_e2_list, probe_calib):
        """Retourne (slope, intercept) OLS recalibré, ou None si pas de données."""
        if not raw_e1_list:
            return None
        raw_e1 = np.asarray(raw_e1_list, dtype=float)
        raw_e2 = np.asarray(raw_e2_list, dtype=float) if raw_e2_list else None

        def _norm_arr(arr, p):
            return (arr - p) / abs(p) if p != 0 else arr

        norm1 = _norm_arr(raw_e1, probe_calib)
        norm2 = _norm_arr(raw_e2, probe_calib) if raw_e2 is not None else None
        norm_mean = (norm1 + norm2) / 2.0 if norm2 is not None else norm1

        valid = np.isfinite(norm_mean) & np.isfinite(log_c)
        if valid.sum() < 2:
            return None
        r = stats.linregress(norm_mean[valid], log_c[valid])
        return r.slope, r.intercept

    ols_A1 = _ols_model(
        session_data["cv_delta_I"].get("e1", []),
        session_data["cv_delta_I"].get("e2"),
        session_data.get("probe_delta_I", 0),
    )
    ols_A2 = _ols_model(
        session_data["eis_rct_fit"].get("e1", []),
        session_data["eis_rct_fit"].get("e2"),
        session_data.get("probe_rct_fit", 0),
    )
    ols_A3 = _ols_model(
        session_data["eis_rct_drt"].get("e1", []),
        session_data["eis_rct_drt"].get("e2"),
        session_data.get("probe_rct_drt", 0),
    )

    def _predict_ols(ols, norm_e1, norm_e2):
        if ols is None:
            return None, None
        slope, intercept = ols
        p1 = slope * norm_e1 + intercept if norm_e1 is not None else None
        p2 = slope * norm_e2 + intercept if norm_e2 is not None else None
        return p1, p2

    # Reconstruit les modèles PLS de calibration (méthodes B)
    def _train_pls_model(X_list_e1, X_list_e2, log_c_arr, n_components=2):
        """Entraîne un PLSRegression sur les données de calibration."""
        rows = list(X_list_e1 or [])
        if X_list_e2:
            rows += list(X_list_e2)
        if not rows:
            return None, None
        X = np.array([np.asarray(r, dtype=float) for r in rows], dtype=float)
        n_elec = 2 if X_list_e2 else 1
        y = np.concatenate([log_c_arr] * n_elec)
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < max(2, n_components + 1):
            return None, None
        sc = StandardScaler()
        Xs = sc.fit_transform(X[valid])
        k = min(n_components, Xs.shape[1], valid.sum() - 1)
        model = PLSRegression(n_components=max(1, k), scale=False)
        model.fit(Xs, y[valid])
        return model, sc

    def _predict_pls(model, scaler, feat_e1, feat_e2):
        if model is None or scaler is None:
            return None, None
        def _pred(feat):
            if feat is None:
                return None
            x = np.asarray(feat, dtype=float).reshape(1, -1)
            xs = scaler.transform(x)
            return float(model.predict(xs).ravel()[0])
        return _pred(feat_e1), _pred(feat_e2)

    eis_X_e1 = session_data.get("eis_spectra", {}).get("e1", [])
    eis_X_e2 = session_data.get("eis_spectra", {}).get("e2")
    cv_X_e1  = session_data.get("cv_spectra",  {}).get("e1", [])
    cv_X_e2  = session_data.get("cv_spectra",  {}).get("e2")

    def _concat(a, b):
        if a is None or b is None:
            return None
        return np.concatenate([np.asarray(a), np.asarray(b)])

    b3_X_e1 = [_concat(e, c) for e, c in zip(eis_X_e1, cv_X_e1)] if eis_X_e1 and cv_X_e1 else []
    b3_X_e2 = [_concat(e, c) for e, c in zip(eis_X_e2 or [], cv_X_e2 or [])] if eis_X_e2 and cv_X_e2 else None

    pls_B1 = _train_pls_model(cv_X_e1,   cv_X_e2,   log_c)
    pls_B2 = _train_pls_model(eis_X_e1,  eis_X_e2,  log_c)
    pls_B3 = _train_pls_model(b3_X_e1,   b3_X_e2,   log_c)

    eis_e1_new = new_signals.get("eis_feat_e1")
    eis_e2_new = new_signals.get("eis_feat_e2")
    cv_e1_new  = new_signals.get("cv_feat_e1")
    cv_e2_new  = new_signals.get("cv_feat_e2")
    b3_e1_new  = _concat(eis_e1_new, cv_e1_new)
    b3_e2_new  = _concat(eis_e2_new, cv_e2_new)

    # ---------- Assemblage des prédictions ----------
    raw_predictions = {
        "A1": _predict_ols(ols_A1, new_norm["A1_e1"], new_norm["A1_e2"]),
        "A2": _predict_ols(ols_A2, new_norm["A2_e1"], new_norm["A2_e2"]),
        "A3": _predict_ols(ols_A3, new_norm["A3_e1"], new_norm["A3_e2"]),
        "B1": _predict_pls(pls_B1[0], pls_B1[1], cv_e1_new,  cv_e2_new),
        "B2": _predict_pls(pls_B2[0], pls_B2[1], eis_e1_new, eis_e2_new),
        "B3": _predict_pls(pls_B3[0], pls_B3[1], b3_e1_new,  b3_e2_new),
    }

    # Incertitude : RMSEP si dispo, sinon RMSECV
    # (accédé depuis session_data si un rapport a déjà été calculé)
    saved_report = session_data.get("_report")

    def _sigma(method_name):
        if saved_report:
            res = saved_report.get("methods", {}).get(method_name, {})
            v = res.get("rmsep") or res.get("rmsecv")
            if v and np.isfinite(v):
                return float(v)
        return None

    # sigma_inter de calibration pour cohérence inter-électrode
    def _sigma_inter_calib(method_name):
        if saved_report:
            res = saved_report.get("methods", {}).get(method_name, {})
            v = res.get("sigma_inter_norm") or res.get("sigma_inter_raw")
            if v and np.isfinite(v):
                return float(v)
        return None

    results = {}
    for m, (p1, p2) in raw_predictions.items():
        if p1 is None and p2 is None:
            continue

        vals = [v for v in (p1, p2) if v is not None]
        mean = float(np.mean(vals)) if vals else None
        c_lin = float(10.0 ** mean) if mean is not None else None

        sigma = _sigma(m)
        ci_factor = float(10 ** (1.96 * sigma)) if sigma is not None else None

        delta_elec = abs(p1 - p2) if (p1 is not None and p2 is not None) else None
        s_inter = _sigma_inter_calib(m)
        coherent = (delta_elec < 2 * s_inter) if (delta_elec is not None and s_inter) else True

        results[m] = {
            "log10_c_e1":   p1,
            "log10_c_e2":   p2,
            "log10_c_mean": mean,
            "c_mean":       c_lin,
            "ci_factor":    ci_factor,
            "sigma_pred":   sigma,
            "coherent":     coherent,
            "delta_elec":   delta_elec,
        }

    return results
