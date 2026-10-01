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
