"""Statistiques d'une régression aux moindres carrés pondérés, à partir de la jacobienne.

Partagé par le measurement model (``core/measurement_model.py``) et le fit Orazem
(``fits/orazem_fit.py``) : covariance des paramètres, nombre de conditionnement,
identifiabilité et leviers, tous obtenus par UNE décomposition en valeurs singulières.

Pourquoi pas ``np.linalg.inv(JᵀJ)`` (AUDIT.md FIT-2) : former JᵀJ ÉLÈVE AU CARRÉ le
conditionnement de J, et ``inv`` rend une matrice même quand JᵀJ est numériquement
singulière ; l'ancien code prenait ensuite ``sqrt(|diag|)``, fabriquant un écart-type
à partir d'une variance négative. Ici :

* J est d'abord ÉQUILIBRÉE par colonne (chaque colonne divisée par sa norme) : le
  conditionnement rapporté mesure alors la colinéarité des paramètres et non leurs
  unités (des pF à côté des kΩ). Van der Sluis (Numer. Math. 14 (1969) 14-23) a
  montré que cette mise à l'échelle est quasi optimale pour le conditionnement.
* Covariance = pseudo-inverse de JₛᵀJₛ par SVD, ramenée aux unités d'origine.
* Une direction de l'espace des paramètres associée à une valeur singulière
  numériquement nulle n'est PAS contrainte par les données : ``pinv`` lui
  attribuerait une variance NULLE (certitude fabriquée). Tout paramètre ayant une
  composante dans cet espace nul reçoit donc un écart-type INFINI et est marqué non
  identifiable.

Référence générale : Bates & Watts, *Nonlinear Regression Analysis and Its
Applications*, Wiley, 1988 (chap. 2, approximation linéaire : Cov(θ̂) = σ²(JᵀJ)⁻¹,
et leviers h_ii = diag(J(JᵀJ)⁻¹Jᵀ)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Seuil de composante dans l'espace nul au-delà duquel un paramètre est déclaré
#: non identifiable (norme² de sa ligne dans les vecteurs singuliers « nuls »).
_NULLSPACE_TOL = 1e-8


@dataclass
class JacobianStatistics:
    """Statistiques linéarisées à l'optimum.

    La covariance est celle des poids fournis, SANS rééchelonnement : c'est à
    l'appelant de la multiplier par un facteur de variance s'il ne connaît σ qu'à
    un facteur près.

    Attributes:
        cov: covariance (P×P), en unités des paramètres ; lignes/colonnes des
            paramètres non identifiables mises à inf.
        std: √diag(cov) (inf si non identifiable).
        condition_number: s_max/s_min de la jacobienne équilibrée par colonne
            (inf si rang déficient).
        rank: rang numérique.
        identifiable: masque booléen par paramètre.
        leverage: diag de la matrice chapeau, un par résidu (∈ [0, 1]).
    """

    cov: np.ndarray
    std: np.ndarray
    condition_number: float
    rank: int
    identifiable: np.ndarray
    leverage: np.ndarray


def jacobian_statistics(J: np.ndarray) -> JacobianStatistics:
    """Covariance (JᵀJ)⁺, conditionnement, identifiabilité et leviers par SVD.

    Args:
        J: jacobienne des résidus PONDÉRÉS (m × P), c.-à-d. ∂[(modèle − données)/σ]/∂θ.

    Returns:
        JacobianStatistics.

    Raises:
        ValueError: jacobienne non finie (à traiter par l'appelant : incertitudes
            indisponibles, et non fabriquées).
    """
    J = np.asarray(J, dtype=float)
    if not np.all(np.isfinite(J)):
        raise ValueError("jacobienne non finie : statistiques linéarisées indisponibles.")
    m, p = J.shape
    norms = np.linalg.norm(J, axis=0)
    zero_col = ~(norms > 0) | ~np.isfinite(norms)
    d = np.where(zero_col, 0.0, 1.0 / np.where(zero_col, 1.0, norms))
    Js = J * d[None, :]
    U, s, Vt = np.linalg.svd(Js, full_matrices=False)
    tol = max(m, p) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    V = Vt.T
    cov_s = (V[:, :r] / s[:r] ** 2) @ V[:, :r].T
    cov = cov_s * np.outer(d, d)

    null = V[:, r:]
    in_null = np.sum(null ** 2, axis=1) > _NULLSPACE_TOL if null.size else np.zeros(p, dtype=bool)
    identifiable = ~(in_null | zero_col)
    cov = cov.copy()
    cov[~identifiable, :] = np.inf
    cov[:, ~identifiable] = np.inf
    with np.errstate(invalid="ignore"):
        std = np.sqrt(np.where(identifiable, np.diag(cov), np.inf))
    cond = float(s[0] / s[-1]) if (s.size and s[-1] > 0 and r == p) else float("inf")
    leverage = np.sum(U[:, :r] ** 2, axis=1)
    return JacobianStatistics(
        cov=cov, std=std, condition_number=cond, rank=r,
        identifiable=identifiable, leverage=leverage,
    )
