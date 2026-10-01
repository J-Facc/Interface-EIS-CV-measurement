"""Bibliothèque d'éléments d'impédance électrochimique pour les circuits utilisateur.

Chaque fonction retourne l'impédance complexe Z(ω) (Ω) d'un élément, sous forme
d'un tableau numpy complexe de même forme que ``w``.

Convention de signature (vérifiée par ``circuit/parser.py`` et par les tests) :

* tous les ÉLÉMENTS prennent la pulsation ``w`` (rad/s, tableau numpy) en premier
  argument, puis leurs paramètres physiques dans l'ordre documenté — y compris
  ``R``, dont l'impédance ne dépend pas de ``w`` (cohérence de signature, et la
  sortie a toujours la forme de ``w``) ;
* le COMBINATEUR ``parallel(*impedances)`` ne prend PAS ``w`` : il combine des
  impédances déjà évaluées.

Dans une expression utilisateur, ``w`` est injecté automatiquement : on écrit
``R(Rct)``, le parseur appelle ``R(w, Rct)``. Écrire ``w`` explicitement en premier
argument (``ZD_bounded(w, R_D, tau_d)``) est toléré.

Références bibliographiques
---------------------------
[OT] M. E. Orazem, B. Tribollet, *Electrochemical Impedance Spectroscopy*,
     2e éd., Wiley, 2017 (ISBN 978-1-118-52739-9). Les chapitres cités ci-dessous
     sont ceux de la 2e édition. ⚠ Les numéros d'ÉQUATION n'ont pas pu être
     vérifiés contre l'ouvrage depuis l'environnement où ce module a été écrit :
     seuls les chapitres sont cités ; les formules sont les formes standard de la
     littérature, rappelées explicitement dans chaque docstring.
[BRUG] G. J. Brug, A. L. G. van den Eeden, M. Sluyters-Rehbach, J. H. Sluyters,
     « The analysis of electrode impedances complicated by the presence of a
     constant phase element », J. Electroanal. Chem. 176 (1984) 275-295.
"""

from __future__ import annotations

import numpy as np


def _omega(w) -> np.ndarray:
    """Convertit la pulsation en tableau numpy réel (rad/s)."""
    return np.asarray(w, dtype=float)


def R(w: np.ndarray, r: float) -> np.ndarray:
    """Résistance pure : Z = r.

    Source : [OT] chap. 4 « Electrical Circuits » — impédance d'une résistance,
    indépendante de la fréquence.

    Args:
        w: Pulsation (rad/s). Sert uniquement à donner sa forme à la sortie.
        r: Résistance (Ω).

    Returns:
        Tableau complexe de forme ``w.shape``, valant ``r`` partout.
    """
    omega = _omega(w)
    return np.full(omega.shape, r, dtype=complex)


def C(w: np.ndarray, c: float) -> np.ndarray:
    """Capacité pure : Z = 1 / (j·ω·c).

    Source : [OT] chap. 4 « Electrical Circuits » — impédance d'un condensateur.

    Args:
        w: Pulsation (rad/s).
        c: Capacité (F).

    Returns:
        Tableau complexe (Ω). Diverge (inf) en ω = 0.
    """
    omega = _omega(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (1j * omega * c)


def L(w: np.ndarray, l: float) -> np.ndarray:  # noqa: E741 — nom imposé par l'interface
    """Inductance pure : Z = j·ω·l.

    Source : [OT] chap. 4 « Electrical Circuits » — impédance d'une inductance.

    Args:
        w: Pulsation (rad/s).
        l: Inductance (H).

    Returns:
        Tableau complexe (Ω).
    """
    omega = _omega(w)
    return 1j * omega * l


def Q(w: np.ndarray, q: float, alpha: float) -> np.ndarray:
    """Élément à phase constante (CPE) : Z = 1 / (q·(j·ω)^alpha).

    Source : [OT] chap. 14 « Constant-Phase Elements » (chap. 13 « Time-Constant
    Dispersion » dans la 1re édition, 2008) ; forme et notation Q, α de [BRUG].
    ``(jω)^α`` est pris sur la branche principale : ``ω^α·exp(j·α·π/2)``, soit une
    phase constante de ``-α·90°``. α = 1 redonne la capacité pure ``C(w, q)``.

    Args:
        w: Pulsation (rad/s).
        q: Pré-facteur du CPE (S·s^alpha, soit F·s^(alpha-1)).
        alpha: Exposant du CPE (sans dimension, 0 < alpha <= 1).

    Returns:
        Tableau complexe (Ω). Diverge en ω = 0 pour alpha > 0.
    """
    omega = _omega(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (q * (1j * omega) ** alpha)


def W(w: np.ndarray, sigma: float) -> np.ndarray:
    """Warburg semi-infini : Z = sigma·(1 - j)/√ω  ( = sigma·√2 / √(j·ω) ).

    Source : [OT] chap. 11 « Diffusion Impedance » — diffusion dans un domaine
    semi-infini ; forme standard de Warburg (phase constante de -45°).

    Args:
        w: Pulsation (rad/s).
        sigma: Coefficient de Warburg (Ω·s^(-1/2)).

    Returns:
        Tableau complexe (Ω). Diverge en ω = 0.
    """
    omega = _omega(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return sigma * (1.0 - 1j) / np.sqrt(omega)


def Wo(w: np.ndarray, r: float, tau: float) -> np.ndarray:
    """Warburg fini RÉFLECTIF (frontière imperméable, « open ») :
    Z = r·coth(√(j·ω·tau)) / √(j·ω·tau).

    Source : [OT] chap. 11 « Diffusion Impedance » — diffusion dans une couche
    d'épaisseur finie limitée par une frontière imperméable (flux nul). À basse
    fréquence le comportement devient capacitif (Z ≈ r/3 + r/(j·ω·tau)) :
    |Z| diverge quand ω → 0 ; en ω = 0 exactement, la fonction retourne ``inf``.

    Args:
        w: Pulsation (rad/s).
        r: Résistance de diffusion (Ω).
        tau: Temps caractéristique de diffusion δ²/D (s).

    Returns:
        Tableau complexe (Ω).
    """
    omega = _omega(w)
    x = np.sqrt(1j * omega * tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = r / (np.tanh(x) * x)
    return np.where(np.abs(x) == 0.0, np.inf + 0j, z)


def Ws(w: np.ndarray, r: float, tau: float) -> np.ndarray:
    """Warburg fini TRANSMISSIF (couche de Nernst, « short ») :
    Z = r·tanh(√(j·ω·tau)) / √(j·ω·tau).

    Source : [OT] chap. 11 « Diffusion Impedance » — diffusion à travers une couche
    stagnante d'épaisseur finie, concentration imposée à la frontière. Z → r quand
    ω → 0 (limite de L'Hôpital tanh(x)/x → 1, traitée explicitement).

    Note : mathématiquement IDENTIQUE à ``ZD_bounded`` (même formule tanh(x)/x).

    Args:
        w: Pulsation (rad/s).
        r: Résistance de diffusion (Ω).
        tau: Temps caractéristique de diffusion δ²/D (s).

    Returns:
        Tableau complexe (Ω).
    """
    omega = _omega(w)
    x = np.sqrt(1j * omega * tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(x) < 1e-8, 1.0, np.tanh(x) / x)
    return r * ratio


def ZD_bounded(w: np.ndarray, R_D: float, tau_d: float) -> np.ndarray:
    """Bounded diffusion impedance (microfluidic channel, Bissessur formulation).

    Reprise À L'IDENTIQUE de ``fits/physics.py:Z_D`` (formule et traitement
    numérique), dont la provenance déclarée est : « Diffusion element follows the
    bounded-diffusion (Bissessur) formulation for a microfluidic channel »
    (ancien METHODES.md §4.1, retiré ; voir docs/ARCHITECTURE.md §6). Aucune référence plus précise (article, équation)
    n'est donnée dans le dépôt pour cette formule.

        Z_D = R_D · tanh(√(j·ω·τ_d)) / √(j·ω·τ_d)

    ⚠ Cette forme en tanh est celle du Warburg fini TRANSMISSIF (``Ws``), et non
    celle du Warburg réflectif ``Wo`` (coth) : « bornée » désigne ici une couche
    de diffusion d'épaisseur finie, pas une frontière imperméable.

    Args:
        w: Angular frequency array (rad/s).
        R_D: Diffusion resistance (Ω).
        tau_d: Characteristic diffusion time (s).

    Returns:
        Complex impedance array (Ω). Handles omega→0 by L'Hopital limit (→ R_D).
    """
    omega = _omega(w)
    x = np.sqrt(1j * omega * tau_d)
    # éviter division par zéro à basse fréquence : tanh(x)/x → 1 quand x→0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(x) < 1e-8, 1.0, np.tanh(x) / x)
    return R_D * ratio


def parallel(*impedances) -> np.ndarray:
    """Association en parallèle : Z = 1 / Σ (1 / Z_k).

    Source : [OT] chap. 4 « Electrical Circuits » — les admittances d'éléments en
    parallèle s'additionnent.

    Un court-circuit (Z_k = 0 exactement) court-circuite l'ensemble : le résultat
    vaut alors 0 en ce point (au lieu du nan que produirait 1/(inf+nan·j)).

    Args:
        *impedances: Au moins deux impédances (tableaux ou scalaires, complexes ou
            réels), diffusables (broadcast) entre elles.

    Returns:
        Tableau complexe (Ω).

    Raises:
        ValueError: moins de deux impédances fournies.
    """
    if len(impedances) < 2:
        raise ValueError("parallel() exige au moins deux impédances.")
    branches = np.broadcast_arrays(*(np.asarray(z, dtype=complex) for z in impedances))
    shorted = np.zeros(branches[0].shape, dtype=bool)
    for z in branches:
        shorted |= z == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        admittance = sum(1.0 / z for z in branches)
        total = 1.0 / admittance
    return np.where(shorted, 0.0 + 0j, total)
