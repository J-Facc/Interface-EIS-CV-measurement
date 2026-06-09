import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut


def normalize_by_probe(
    spectrum: np.ndarray,
    probe_spectra: list,
) -> np.ndarray:
    """
    Normalise un spectre par la moyenne des réplicats probe de référence.

    Formule : (spectrum - probe_mean) / probe_mean

    Les positions où |probe_mean| < 1e-10 valent 0.0 (évite divisions par zéro).

    Parameters
    ----------
    spectrum : np.ndarray
        Spectre à normaliser (Zre, Zim, ou courant CV).
    probe_spectra : list[np.ndarray]
        Liste de 1 à N réplicats probe. Le spectre moyen est calculé avant
        normalisation.

    Returns
    -------
    np.ndarray
        Spectre normalisé, même shape que spectrum.

    Raises
    ------
    ValueError
        Si probe_spectra est vide, ou si les shapes sont incompatibles.
    """
    spectrum = np.asarray(spectrum, dtype=float)

    if not probe_spectra:
        raise ValueError("probe_spectra ne peut pas être vide")

    probes = np.stack([np.asarray(p, dtype=float) for p in probe_spectra], axis=0)
    probe_mean = np.mean(probes, axis=0)

    if spectrum.shape != probe_mean.shape:
        raise ValueError(
            f"spectrum et probe_mean doivent avoir la même shape "
            f"({spectrum.shape} vs {probe_mean.shape})"
        )

    safe_probe = np.where(np.abs(probe_mean) < 1e-10, np.nan, probe_mean)
    result = (spectrum - probe_mean) / safe_probe
    result = np.where(np.isnan(result), 0.0, result)
    return result


def interpolate_to_grid(
    values: np.ndarray,
    axis: np.ndarray,
    grid: np.ndarray,
    log_axis: bool = False,
) -> np.ndarray:
    """
    Interpole un vecteur de valeurs sur une grille commune.

    Pour les fréquences EIS, utiliser log_axis=True : l'interpolation
    est effectuée dans l'espace log10 de l'axe, ce qui est adapté à la
    distribution logarithmique des fréquences.
    Pour les potentiels CV, utiliser log_axis=False (interpolation linéaire).

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Valeurs à interpoler (Zre, Zim, courant, etc.).
    axis : np.ndarray, shape (n,)
        Axe correspondant aux valeurs (fréquences Hz ou potentiels V).
        Doit être strictement monotone.
    grid : np.ndarray, shape (m,)
        Grille cible sur laquelle interpoler.
    log_axis : bool
        Si True, l'interpolation est effectuée sur log10(axis) et log10(grid).
        Utiliser True pour les fréquences EIS, False pour les potentiels CV.

    Returns
    -------
    np.ndarray, shape (m,)
        Valeurs interpolées sur la grille cible.

    Raises
    ------
    ValueError
        Si values et axis n'ont pas la même longueur, ou si axis contient
        des valeurs <= 0 quand log_axis=True.
    """
    values = np.asarray(values, dtype=float)
    axis = np.asarray(axis, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if values.shape != axis.shape:
        raise ValueError(
            f"values et axis doivent avoir la même longueur "
            f"({values.shape} vs {axis.shape})"
        )

    if log_axis:
        if np.any(axis <= 0) or np.any(grid <= 0):
            raise ValueError(
                "axis et grid doivent être strictement positifs pour log_axis=True"
            )
        return np.interp(np.log10(grid), np.log10(axis), values)
    else:
        return np.interp(grid, axis, values)


def build_feature_matrix(
    eis_spectra: list,
    cv_spectra: list | None,
    freq_grid: np.ndarray,
    pot_grid: np.ndarray | None,
) -> np.ndarray:
    """
    Construit la matrice de features X pour l'entraînement PLS.

    Chaque ligne correspond à un échantillon (spectre d'une électrode à une
    concentration). Les features sont construites par concaténation de :
      - EIS : [ΔZre_norm(f), ΔZim_norm(f)] interpolés sur freq_grid
      - CV  : [ΔI_norm(E)] interpolé sur pot_grid (si cv_spectra fourni)

    Chaque bloc (EIS Zre, EIS Zim, CV) est interpolé séparément avant
    concaténation. Aucune normalisation statistique n'est appliquée ici
    (elle est faite par train_pls via StandardScaler).

    Parameters
    ----------
    eis_spectra : list of dict
        Liste d'échantillons EIS normalisés. Chaque élément est un dict avec :
          - 'Zre_norm' : np.ndarray — partie réelle normalisée par probe
          - 'Zim_norm' : np.ndarray — partie imaginaire normalisée par probe
          - 'freq'     : np.ndarray — fréquences en Hz (même longueur)
    cv_spectra : list of dict or None
        Liste d'échantillons CV normalisés (même ordre que eis_spectra).
        Chaque élément est un dict avec :
          - 'I_norm' : np.ndarray — courant normalisé par probe
          - 'E'      : np.ndarray — potentiel en V (même longueur)
        Passer None pour le mode EIS seul.
    freq_grid : np.ndarray, shape (n_freq,)
        Grille de fréquences communes (Hz) pour l'interpolation EIS.
        Doit être strictement positive (interpolation log).
    pot_grid : np.ndarray or None, shape (n_pot,)
        Grille de potentiels communs (V) pour l'interpolation CV.
        Ignoré si cv_spectra is None.

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Matrice de features. n_features = 2*n_freq si cv_spectra is None,
        sinon 2*n_freq + n_pot.

    Raises
    ------
    ValueError
        Si eis_spectra et cv_spectra n'ont pas la même longueur,
        ou si cv_spectra est fourni sans pot_grid.
    """
    if cv_spectra is not None and len(eis_spectra) != len(cv_spectra):
        raise ValueError(
            f"eis_spectra et cv_spectra doivent avoir la même longueur "
            f"({len(eis_spectra)} vs {len(cv_spectra)})"
        )
    if cv_spectra is not None and pot_grid is None:
        raise ValueError("pot_grid est requis quand cv_spectra est fourni")

    freq_grid = np.asarray(freq_grid, dtype=float)
    rows = []

    for i, eis in enumerate(eis_spectra):
        zre = interpolate_to_grid(
            eis["Zre_norm"], eis["freq"], freq_grid, log_axis=True
        )
        zim = interpolate_to_grid(
            eis["Zim_norm"], eis["freq"], freq_grid, log_axis=True
        )
        features = np.concatenate([zre, zim])

        if cv_spectra is not None:
            cv = cv_spectra[i]
            pot_grid_arr = np.asarray(pot_grid, dtype=float)
            i_norm = interpolate_to_grid(
                cv["I_norm"], cv["E"], pot_grid_arr, log_axis=False
            )
            features = np.concatenate([features, i_norm])

        rows.append(features)

    return np.array(rows, dtype=float)


def select_n_components(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_components: int = 4,
) -> int:
    """
    Sélectionne le nombre optimal de composantes latentes PLS par RMSECV.

    Utilise une validation croisée Leave-One-Group-Out (leave-one-concentration-out)
    pour éviter la fuite de données entre réplicats d'une même concentration.
    Teste k de 1 à max_components et retourne le k qui minimise le RMSECV.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Matrice de features (déjà construite par build_feature_matrix).
    y : np.ndarray, shape (n_samples,)
        Cibles en log10([c]).
    groups : np.ndarray, shape (n_samples,)
        Entier identifiant la concentration de chaque échantillon.
        Même valeur pour tous les réplicats d'une même concentration.
    max_components : int
        Nombre maximum de composantes à tester (défaut : 4).
        Limité automatiquement à min(max_components, n_features, n_groups - 1).

    Returns
    -------
    int
        Nombre optimal de composantes latentes (entre 1 et max_components).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    n_groups = len(np.unique(groups))
    # PLS nécessite au moins 1 groupe en test et le reste en train
    k_max = min(max_components, X.shape[1], n_groups - 1)
    k_max = max(1, k_max)

    logo = LeaveOneGroupOut()
    best_k = 1
    best_rmsecv = np.inf

    for k in range(1, k_max + 1):
        squared_errors = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = PLSRegression(n_components=k, scale=False)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s).ravel()

            squared_errors.extend((y_pred - y_test) ** 2)

        rmsecv = float(np.sqrt(np.mean(squared_errors)))
        if rmsecv < best_rmsecv:
            best_rmsecv = rmsecv
            best_k = k

    return best_k


def train_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
) -> tuple[PLSRegression, StandardScaler]:
    """
    Entraîne un modèle PLSRegression sur l'ensemble des données de calibration.

    Le StandardScaler est ajusté exclusivement sur X_train (ici l'ensemble
    complet des données de calibration). Il doit être conservé et appliqué
    aux nouvelles données avant prédiction.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Matrice de features de calibration.
    y : np.ndarray, shape (n_samples,)
        Cibles en log10([c]).
    n_components : int
        Nombre de composantes latentes (idéalement issu de select_n_components).

    Returns
    -------
    tuple[PLSRegression, StandardScaler]
        (model, scaler) — le scaler a été fit sur X, le modèle sur X standardisé.

    Raises
    ------
    ValueError
        Si n_components < 1 ou supérieur au nombre de features / échantillons.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if n_components < 1:
        raise ValueError(f"n_components doit être >= 1, reçu {n_components}")
    if n_components > min(X.shape[0], X.shape[1]):
        raise ValueError(
            f"n_components ({n_components}) dépasse min(n_samples, n_features) "
            f"= {min(X.shape[0], X.shape[1])}"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_scaled, y)

    return model, scaler


def predict_concentration(
    model: PLSRegression,
    scaler: StandardScaler,
    X_new: np.ndarray,
) -> tuple[float, float]:
    """
    Prédit la concentration pour un ou plusieurs nouveaux spectres.

    Le scaler fourni (issu de train_pls) est appliqué avant la prédiction
    pour mettre X_new dans le même espace que les données d'entraînement.
    Si plusieurs échantillons sont fournis, retourne la moyenne des prédictions.

    Parameters
    ----------
    model : PLSRegression
        Modèle entraîné par train_pls.
    scaler : StandardScaler
        Scaler ajusté sur les données d'entraînement par train_pls.
    X_new : np.ndarray, shape (n_samples, n_features) ou (n_features,)
        Features du ou des nouveaux spectres à prédire.

    Returns
    -------
    tuple[float, float]
        (log10_c_pred, c_pred_lineaire) où c_pred_lineaire = 10 ** log10_c_pred.
        Si plusieurs échantillons, retourne la moyenne des log10([c]).

    Raises
    ------
    ValueError
        Si X_new est vide.
    """
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    if X_new.size == 0:
        raise ValueError("X_new ne peut pas être vide")

    X_scaled = scaler.transform(X_new)
    log10_c_pred = float(np.mean(model.predict(X_scaled).ravel()))
    c_lineaire = float(10.0 ** log10_c_pred)

    return (log10_c_pred, c_lineaire)
