"""Shared electrochemical physics functions.

References:
  Poujouly 2022 (Doc 2), Deslouis (Doc 11).
"""

import numpy as np


def alpha_h(Fv: float, h: float, d: float) -> float:
    """Wall velocity gradient for rectangular channel flow.

    alpha_h = 6 * Fv / (h^2 * d)

    Args:
        Fv: Volumetric flow rate (m^3/s).
        h: Channel height (m).
        d: Channel width (m).

    Returns:
        Wall velocity gradient (s^-1).
    """
    return 6.0 * Fv / (h ** 2 * d)


def sigma_reduced(omega: float, xe: float, D: float, ah: float) -> float:
    """Dimensionless reduced frequency for convective-diffusion impedance.

    sigma = omega * (xe^2 / (D * ah^2))^(1/3)

    Args:
        omega: Angular frequency (rad/s).
        xe: Electrode half-width (m).
        D: Diffusion coefficient (m^2/s).
        ah: Wall velocity gradient alpha_h (s^-1).

    Returns:
        Reduced frequency (dimensionless).
    """
    return omega * (xe ** 2 / (D * ah ** 2)) ** (1.0 / 3.0)


def ZD_modulus_LF(sigma: float, ZD0: float) -> float:
    """Low-frequency modulus approximation of the convective-diffusion impedance.

    |ZD| = ZD0 * (1 + 0.433*sigma^2 - 0.0084*sigma^4)^(-1/2)

    Args:
        sigma: Reduced frequency.
        ZD0: DC diffusion resistance (Ω).

    Returns:
        |Z_D| at low frequency (Ω).
    """
    val = 1.0 + 0.433 * sigma ** 2 - 0.0084 * sigma ** 4
    return ZD0 * max(val, 1e-12) ** (-0.5)


def ZD_phase_LF(sigma: float) -> float:
    """Low-frequency phase of the convective-diffusion impedance (radians).

    arg(ZD) = -arctan(0.5527*sigma*(1 - 0.071*sigma^2 + 0.0023*sigma^4))

    Args:
        sigma: Reduced frequency.

    Returns:
        Phase in radians.
    """
    return -np.arctan(0.5527 * sigma * (1.0 - 0.071 * sigma ** 2 + 0.0023 * sigma ** 4))


def ZD_modulus_HF(sigma: float, ZD0: float) -> float:
    """High-frequency modulus approximation of the convective-diffusion impedance.

    |ZD| = ZD0 * (0.80755 / sigma^0.5) * (1 + 0.1768 / sigma^1.5)

    Args:
        sigma: Reduced frequency.
        ZD0: DC diffusion resistance (Ω).

    Returns:
        |Z_D| at high frequency (Ω).
    """
    return ZD0 * (0.80755 / sigma ** 0.5) * (1.0 + 0.1768 / sigma ** 1.5)


def ZD_complex(
    omega: np.ndarray,
    ZD0: float,
    xe: float,
    D: float,
    Fv: float,
    h: float,
    d: float,
) -> np.ndarray:
    """Complex convective-diffusion impedance Z_D(omega), unified LF/HF regime.

    Switches between LF polynomial and HF asymptotic at sigma = 2.0.

    Args:
        omega: Angular frequency array (rad/s).
        ZD0: DC diffusion resistance (Ω).
        xe: Electrode half-width (m).
        D: Diffusion coefficient (m^2/s).
        Fv: Volumetric flow rate (m^3/s).
        h: Channel height (m).
        d: Channel width (m).

    Returns:
        Complex impedance array Z_D (Ω).
    """
    ah = alpha_h(Fv, h, d)
    ZD = np.zeros(len(omega), dtype=complex)

    for i, w in enumerate(omega):
        if w == 0.0:
            ZD[i] = ZD0 + 0j
            continue
        sig = sigma_reduced(w, xe, D, ah)
        if sig < 2.0:
            mod = ZD_modulus_LF(sig, ZD0)
            phi = ZD_phase_LF(sig)
        else:
            mod = ZD_modulus_HF(sig, ZD0)
            phi = -np.pi / 4.0 * (1.0 - 0.5 / max(sig, 1e-6))
        ZD[i] = mod * np.exp(1j * phi)

    return ZD


def Z_randles_full(
    omega: np.ndarray,
    Re: float,
    Re_prime: float,
    Cb: float,
    Rct: float,
    Qdl: float,
    alpha: float,
    ZD0: float,
    xe: float,
    D: float,
    Fv: float,
    h: float,
    d: float,
) -> np.ndarray:
    """Full Randles circuit impedance: Re - [R'e // Cb] - [Rct // CPE] - ZD.

    Args:
        omega: Angular frequencies (rad/s).
        Re: Electrolyte resistance (Ω).
        Re_prime: Secondary electrolyte resistance (Ω).
        Cb: Bypass capacitance (F).
        Rct: Charge transfer resistance (Ω).
        Qdl: CPE coefficient (F·s^(alpha-1)).
        alpha: CPE exponent (0.5 to 1.0).
        ZD0: DC diffusion resistance (Ω).
        xe: Electrode half-width (m).
        D: Diffusion coefficient (m^2/s).
        Fv: Volumetric flow rate (m^3/s).
        h: Channel height (m).
        d: Channel width (m).

    Returns:
        Complex impedance array (Ω).
    """
    j_omega = 1j * omega
    ZD = ZD_complex(omega, ZD0, xe, D, Fv, h, d)
    Z_CPE = 1.0 / (Qdl * j_omega ** alpha)
    Z_faradaic = 1.0 / (1.0 / Rct + 1.0 / Z_CPE) + ZD
    # R'e parallel with Cb
    if Re_prime > 0 and Cb > 0:
        Z_Rprime_Cb = Re_prime / (1.0 + j_omega * Cb * Re_prime)
    else:
        Z_Rprime_Cb = np.zeros_like(omega, dtype=complex)
    return Re + Z_Rprime_Cb + Z_faradaic


def Rct_bare_theory(T: float, S: float, k0: float, C0: float) -> float:
    """Theoretical charge transfer resistance for a bare electrode.

    Rct_bare = RT / (F^2 * S * k0 * C0)

    Args:
        T: Temperature (K).
        S: Active electrode surface area (m^2).
        k0: Standard heterogeneous rate constant (m/s).
        C0: Mediator bulk concentration (mol/m^3).

    Returns:
        Rct_bare (Ω).
    """
    F_const = 96485.0
    R_const = 8.314
    return (R_const * T) / (F_const ** 2 * S * k0 * C0)


def Cdl_brug(Qdl: float, alpha: float, Re: float, Re_prime: float, Rct: float) -> float:
    """Effective double-layer capacitance via the Brug formula.

    Cdl_eq = [Qdl * (1/(Re + R'e) + 1/Rct)^(alpha - 1)]^(1/alpha)

    Args:
        Qdl: CPE coefficient (F·s^(alpha-1)).
        alpha: CPE exponent.
        Re: Electrolyte resistance (Ω).
        Re_prime: Secondary electrolyte resistance (Ω).
        Rct: Charge transfer resistance (Ω).

    Returns:
        Effective double-layer capacitance (F).
    """
    R_sum = Re + Re_prime
    return (Qdl * (1.0 / R_sum + 1.0 / Rct) ** (alpha - 1.0)) ** (1.0 / alpha)


def theta_EIS(Rct_bare: float, Rct_ap: float) -> float:
    """Surface coverage estimated from EIS charge transfer resistances.

    theta = 1 - Rct_bare / Rct_ap

    Args:
        Rct_bare: Bare electrode Rct (Ω).
        Rct_ap: Apparent Rct after surface modification (Ω).

    Returns:
        Surface coverage fraction [0, 1].
    """
    return 1.0 - Rct_bare / Rct_ap
