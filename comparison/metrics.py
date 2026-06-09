import numpy as np
from scipy.stats import t


def compute_sigma_intra(predictions: np.ndarray) -> float:
    """
    Écart-type des prédictions sur les réplicats d'une même électrode.

    Parameters
    ----------
    predictions : np.ndarray, shape (n_replicats,)
        Prédictions en log10([c]) pour les réplicats d'une même électrode
        à concentration fixée.

    Returns
    -------
    float
        Écart-type en décades. Retourne 0.0 si n_replicats < 2 ou tableau vide.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.size < 2:
        return 0.0
    return float(np.std(predictions, ddof=1))


def compute_sigma_inter(mean_e1: float, mean_e2: float) -> float:
    """
    Différence absolue entre les prédictions moyennes de deux électrodes.

    Mesure la variabilité de fonctionnalisation résiduelle entre électrodes
    exposées à la même solution (canal microfluidique multiplexé).

    Parameters
    ----------
    mean_e1 : float
        Prédiction moyenne de l'électrode 1 en log10([c]).
    mean_e2 : float
        Prédiction moyenne de l'électrode 2 en log10([c]).

    Returns
    -------
    float
        Différence absolue en décades.
    """
    return abs(float(mean_e1) - float(mean_e2))


def compute_rmsep(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Root Mean Square Error of Prediction.

    Parameters
    ----------
    y_pred : np.ndarray
        Concentrations prédites en log10([c]).
    y_true : np.ndarray
        Concentrations vraies en log10([c]).

    Returns
    -------
    float
        RMSEP en décades. Retourne np.nan si le tableau est vide.

    Raises
    ------
    ValueError
        Si y_pred et y_true n'ont pas la même taille.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_pred.size == 0:
        return np.nan
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred et y_true doivent avoir la même taille "
            f"({y_pred.shape} vs {y_true.shape})"
        )
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_bias(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Biais moyen signé entre prédictions et valeurs vraies.

    Un biais positif indique une surestimation systématique ;
    négatif une sous-estimation.

    Parameters
    ----------
    y_pred : np.ndarray
        Concentrations prédites en log10([c]).
    y_true : np.ndarray
        Concentrations vraies en log10([c]).

    Returns
    -------
    float
        Biais moyen en décades. Retourne np.nan si le tableau est vide.

    Raises
    ------
    ValueError
        Si y_pred et y_true n'ont pas la même taille.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_pred.size == 0:
        return np.nan
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred et y_true doivent avoir la même taille "
            f"({y_pred.shape} vs {y_true.shape})"
        )
    return float(np.mean(y_pred - y_true))


def compute_rpd(y_true: np.ndarray, rmsep: float) -> float:
    """
    Ratio of Performance to Deviation.

    Normalise la performance par la difficulté de la tâche (étendue de la gamme).
    Permet de comparer des méthodes évaluées sur des gammes légèrement différentes.

    Seuils d'interprétation :
      RPD < 2   → modèle non informatif
      RPD 2–3   → screening grossier uniquement
      RPD 3–5   → quantification acceptable
      RPD > 5   → modèle excellent

    Parameters
    ----------
    y_true : np.ndarray
        Concentrations vraies en log10([c]).
    rmsep : float
        RMSEP en décades (typiquement issu de compute_rmsep).

    Returns
    -------
    float
        RPD sans unité. Retourne np.inf si rmsep == 0, np.nan si y_true est vide.
    """
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size < 2:
        return np.nan
    std_y = float(np.std(y_true, ddof=1))
    if rmsep == 0.0:
        return np.inf
    return std_y / float(rmsep)


def compute_ci95_bootstrap(
    predictions: np.ndarray,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """
    Intervalle de confiance à 95 % par bootstrap sur les réplicats.

    Ré-échantillonne les prédictions avec remise et calcule la distribution
    des moyennes pour en extraire les percentiles 2.5 et 97.5.

    Parameters
    ----------
    predictions : np.ndarray
        Prédictions en log10([c]) pour les réplicats d'une électrode.
    n_bootstrap : int
        Nombre de tirages bootstrap (défaut : 1000).

    Returns
    -------
    tuple[float, float]
        (borne_inf, borne_sup) en log10([c]) à 95 %.
        Retourne (np.nan, np.nan) si predictions est vide.
        Si n_replicats == 1, retourne (predictions[0], predictions[0]).
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.size == 0:
        return (np.nan, np.nan)
    if predictions.size == 1:
        return (float(predictions[0]), float(predictions[0]))

    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        rng.choice(predictions, size=len(predictions), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return (lower, upper)


def compute_rmsecv_logo(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn,
) -> float:
    """
    RMSECV par Leave-One-Group-Out (leave-one-concentration-out).

    À chaque fold, tous les réplicats d'une même concentration sont retirés
    ensemble de l'entraînement pour éviter la fuite de données.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Matrice de features.
    y : np.ndarray, shape (n_samples,)
        Cibles en log10([c]).
    groups : np.ndarray, shape (n_samples,)
        Entier identifiant la concentration de chaque échantillon.
        Même valeur pour tous les réplicats d'une concentration donnée.
    model_fn : callable
        Fonction d'interface : model_fn(X_train, y_train, X_test) -> np.ndarray
        Doit retourner les prédictions sur X_test.

    Returns
    -------
    float
        RMSECV en décades. Retourne np.nan si moins de 2 groupes distincts.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return np.nan

    squared_errors = []
    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask
        if train_mask.sum() == 0:
            continue
        y_pred = model_fn(X[train_mask], y[train_mask], X[test_mask])
        squared_errors.extend((np.asarray(y_pred) - y[test_mask]) ** 2)

    if not squared_errors:
        return np.nan
    return float(np.sqrt(np.mean(squared_errors)))


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> tuple[float, float]:
    """
    Test de Diebold-Mariano sur erreurs au carré.

    Compare statistiquement la performance prédictive de deux méthodes
    sur les mêmes points de validation.
    H0 : les deux méthodes ont la même performance (E[d] = 0).

    La statistique de test est calculée via un t-test à un échantillon
    sur les différences de pertes d_t = e_a² - e_b².

    Parameters
    ----------
    errors_a : np.ndarray
        Erreurs de prédiction de la méthode A : y_pred_a - y_true, en décades.
    errors_b : np.ndarray
        Erreurs de prédiction de la méthode B : y_pred_b - y_true, en décades.

    Returns
    -------
    tuple[float, float]
        (statistique_DM, p_value) — test bilatéral.
        Retourne (np.nan, np.nan) si moins de 2 points ou écart-type nul.

    Raises
    ------
    ValueError
        Si errors_a et errors_b n'ont pas la même taille.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)

    if errors_a.shape != errors_b.shape:
        raise ValueError(
            f"errors_a et errors_b doivent avoir la même taille "
            f"({errors_a.shape} vs {errors_b.shape})"
        )
    n = errors_a.size
    if n < 2:
        return (np.nan, np.nan)

    d = errors_a ** 2 - errors_b ** 2
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)

    if d_std == 0.0:
        # Pertes identiques — les deux méthodes font exactement la même erreur
        return (0.0, 1.0)

    stat = d_mean / (d_std / np.sqrt(n))
    p_value = 2.0 * t.sf(abs(stat), df=n - 1)
    return (float(stat), float(p_value))


def compute_normalization_reduction(
    sigma_inter_raw: float,
    sigma_inter_norm: float,
) -> float:
    """
    Facteur de réduction de variabilité inter-électrode par la normalisation.

    Mesure l'efficacité de la normalisation par le probe à supprimer la
    variabilité de fonctionnalisation entre électrodes.

    Interprétation :
      Facteur = 1   → normalisation sans effet
      Facteur = 3   → la normalisation divise par 3 la variabilité
      Facteur = 10  → normalisation très efficace

    Parameters
    ----------
    sigma_inter_raw : float
        Variabilité inter-électrode avant normalisation (en décades).
    sigma_inter_norm : float
        Variabilité inter-électrode après normalisation (en décades).

    Returns
    -------
    float
        Ratio sigma_inter_raw / sigma_inter_norm.
        Retourne np.inf si sigma_inter_norm == 0 et sigma_inter_raw > 0.
        Retourne 1.0 si les deux sont nuls (cas parfaitement homogène).
    """
    raw = float(sigma_inter_raw)
    norm = float(sigma_inter_norm)

    if norm == 0.0:
        return np.inf if raw > 0.0 else 1.0
    return raw / norm


def compute_sigma_probe(probe_spectra: list) -> float:
    """
    Écart-type inter-réplicats du probe, moyenné sur toutes les fréquences.

    Mesure la stabilité de la fonctionnalisation pendant la session de mesure.
    Un sigma_probe élevé indique une dérive ou une inhomogénéité de surface
    qui dégradera la qualité de la normalisation.

    Parameters
    ----------
    probe_spectra : list[np.ndarray]
        Liste de 1 à N réplicats probe (spectres ou vecteurs de même shape).

    Returns
    -------
    float
        Écart-type relatif moyen en % : mean(std / |mean|) × 100.
        Retourne 0.0 si moins de 2 réplicats ou si le tableau est vide.

    Notes
    -----
    Interprétation :
        < 2 %  → fonctionnalisation stable, normalisation fiable
        2–5 %  → variabilité modérée, surveiller l'impact sur σ_inter
        > 5 %  → instabilité de surface significative
    """
    if not probe_spectra or len(probe_spectra) < 2:
        return 0.0

    stack = np.stack([np.asarray(p, dtype=float) for p in probe_spectra], axis=0)
    std_per_freq  = np.std(stack, axis=0, ddof=1)
    mean_per_freq = np.mean(stack, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        relative_std = np.where(
            np.abs(mean_per_freq) > 1e-10,
            std_per_freq / np.abs(mean_per_freq),
            0.0,
        )

    return float(np.mean(relative_std)) * 100.0
