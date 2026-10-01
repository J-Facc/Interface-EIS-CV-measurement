"""Fonctions électrochimiques partagées.

Le circuit de Randles codé en dur (``Z_randles_full``) a été supprimé à l'étape 5 de
la refonte : le circuit ajusté est désormais défini par l'utilisateur
(``circuit/``, ``fit.circuit`` de la config). ``Z_D`` est conservé comme référence de
l'élément ``ZD_bounded`` (diffusion bornée, formulation Bissessur pour un canal
microfluidique), vérifiée identique par ``tests/test_circuit_elements.py``.
"""

import numpy as np


def Z_D(omega: np.ndarray, R_D: float, tau_d: float) -> np.ndarray:
    """Bounded diffusion impedance (microfluidic channel, Bissessur formulation).

    Args:
        omega: Angular frequency array (rad/s).
        R_D: Diffusion resistance (Ω).
        tau_d: Characteristic diffusion time (s).

    Returns:
        Complex impedance array (Ω). Handles omega→0 by L'Hopital limit (→ R_D).
    """
    x = np.sqrt(1j * omega * tau_d)
    # éviter division par zéro à basse fréquence : tanh(x)/x → 1 quand x→0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(x) < 1e-8, 1.0, np.tanh(x) / x)
    return R_D * ratio


def Rct_bare_theory(T: float, S: float, k0: float, C0: float) -> float:
    """Theoretical bare electrode Rct from Butler-Volmer kinetics.

    Args:
        T: Temperature (K).
        S: Active electrode area (m²).
        k0: Standard rate constant (m/s).
        C0: Mediator concentration (mol/m³).

    Returns:
        Rct (Ω).
    """
    R_gas = 8.314
    F = 96485.0
    n = 1
    return (R_gas * T) / (F ** 2 * n * S * k0 * C0)


def Cdl_brug(Qdl: float, alpha: float, Re: float, Re_prime: float, Rct: float) -> float:
    """Effective double-layer capacitance via Brug formula.

    Args:
        Qdl: CPE pre-factor (S·sᵅ).
        alpha: CPE exponent (dimensionless).
        Re: Solution resistance (Ω).
        Re_prime: Series resistance (Ω).
        Rct: Charge transfer resistance (Ω).

    Returns:
        Cdl_eq (F).
    """
    return (Qdl * (1.0 / (Re + Re_prime) + 1.0 / Rct) ** (alpha - 1.0)) ** (1.0 / alpha)


def theta_EIS(Rct_bare: float, Rct_ap: float) -> float:
    """Surface coverage rate from EIS Rct ratio.

    Args:
        Rct_bare: Bare electrode Rct (Ω).
        Rct_ap: Apparent Rct after hybridisation (Ω).

    Returns:
        theta: Coverage rate (0 to 1).
    """
    return 1.0 - Rct_bare / Rct_ap
