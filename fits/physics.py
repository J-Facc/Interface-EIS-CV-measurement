"""Shared electrochemical physics functions for EIS fit models.

All formulas follow Poujouly (2022) and Deslouis et al. for convection-diffusion
impedance in a rectangular microchannel with laminar flow.
"""

import numpy as np


def alpha_h(Fv: float, h: float, d: float) -> float:
    """Compute wall shear gradient for a rectangular microchannel.

    Args:
        Fv: Volumetric flow rate (m³/s).
        h: Channel height (m).
        d: Channel width (m).

    Returns:
        alpha_h (m⁻¹·s⁻¹).
    """
    return 6.0 * Fv / (h ** 2 * d)


def sigma_reduced(omega: np.ndarray, xe: float, D: float, ah: float) -> np.ndarray:
    """Compute dimensionless reduced frequency sigma.

    Args:
        omega: Angular frequency array (rad/s).
        xe: Electrode width (m).
        D: Diffusion coefficient (m²/s).
        ah: Wall shear gradient alpha_h (m⁻¹·s⁻¹).

    Returns:
        sigma (dimensionless).
    """
    return omega * (xe ** 2 / (D * ah ** 2)) ** (1.0 / 3.0)


def ZD_modulus_LF(sigma: np.ndarray, ZD0: float) -> np.ndarray:
    """Low-frequency modulus of convection-diffusion impedance.

    Valid for sigma < 1.

    Args:
        sigma: Reduced frequency (dimensionless).
        ZD0: DC impedance limit (Ω).

    Returns:
        |Z_D| (Ω).
    """
    denom = 1.0 + 0.433 * sigma ** 2 - 0.0084 * sigma ** 4
    denom = np.maximum(denom, 1e-12)
    return ZD0 / np.sqrt(denom)


def ZD_phase_LF(sigma: np.ndarray) -> np.ndarray:
    """Low-frequency phase of convection-diffusion impedance.

    Args:
        sigma: Reduced frequency (dimensionless).

    Returns:
        arg(Z_D) in radians (negative — capacitive behaviour).
    """
    return -np.arctan(0.5527 * sigma * (1.0 - 0.071 * sigma ** 2 + 0.0023 * sigma ** 4))


def ZD_modulus_HF(sigma: np.ndarray, ZD0: float) -> np.ndarray:
    """High-frequency modulus of convection-diffusion impedance.

    Valid for sigma >= 1 (Warburg-like regime).

    Args:
        sigma: Reduced frequency (dimensionless).
        ZD0: DC impedance limit (Ω).

    Returns:
        |Z_D| (Ω).
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
    """Complex convection-diffusion impedance Z_D(omega), unified LF/HF regime.

    Switches from LF expansion (sigma < 1) to HF Warburg-like expansion
    (sigma >= 1) to avoid divergence of the polynomial LF formula.

    Args:
        omega: Angular frequency array (rad/s).
        ZD0: DC impedance limit (Ω).
        xe: Electrode width (m).
        D: Diffusion coefficient (m²/s).
        Fv: Volumetric flow rate (m³/s).
        h: Channel height (m).
        d: Channel width (m).

    Returns:
        Complex impedance array Z_D (Ω).
    """
    ah = alpha_h(Fv, h, d)
    sigma = sigma_reduced(omega, xe, D, ah)

    Z = np.zeros(len(omega), dtype=complex)

    lf = sigma < 1.0
    hf = ~lf

    if np.any(lf):
        mod = ZD_modulus_LF(sigma[lf], ZD0)
        phase = ZD_phase_LF(sigma[lf])
        Z[lf] = mod * np.exp(1j * phase)

    if np.any(hf):
        mod = ZD_modulus_HF(sigma[hf], ZD0)
        # HF asymptote: phase → -45° (Warburg), with small correction at intermediate sigma
        phase_hf = -np.pi / 4.0 * np.ones(np.sum(hf))
        Z[hf] = mod * np.exp(1j * phase_hf)

    return Z


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
    """Full Randles circuit: Re — [R'e // Cb] — [Rct // CPE(Qdl,alpha)] — ZD.

    Args:
        omega: Angular frequency array (rad/s).
        Re: Solution resistance (Ω).
        Re_prime: Parallel branch resistance (Ω).
        Cb: Blocking capacitance (F).
        Rct: Charge transfer resistance (Ω).
        Qdl: CPE pre-factor (S·sᵅ).
        alpha: CPE exponent (0.6–1.0, dimensionless).
        ZD0: DC diffusion impedance (Ω).
        xe: Electrode width (m).
        D: Effective diffusion coefficient (m²/s).
        Fv: Volumetric flow rate (m³/s).
        h: Channel height (m).
        d: Channel width (m).

    Returns:
        Complex total impedance Z (Ω), shape = (len(omega),).
    """
    jw = 1j * omega

    # Parallel R'e // Cb branch
    Z_Cb = 1.0 / (jw * Cb)
    Z_branch1 = (Re_prime * Z_Cb) / (Re_prime + Z_Cb)

    # CPE element: 1 / (Qdl * (jω)^alpha)
    Z_CPE = 1.0 / (Qdl * (jw) ** alpha)

    # Diffusion-convection element
    Z_D = ZD_complex(omega, ZD0, xe, D, Fv, h, d)

    # Rct // CPE, then series with ZD
    Z_interface = (Rct * Z_CPE) / (Rct + Z_CPE) + Z_D

    return Re + Z_branch1 + Z_interface


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
