# -*- coding: utf-8 -*-
"""
fits/drt_fit.py — Distribution of Relaxation Times par déconvolution FFT
Méthode : Bissessur (2025)

Architecture à deux étapes
──────────────────────────
Étape 1 — Fit Randles 8 paramètres (Nelder-Mead, loss Modulus) sur les données
           expérimentales — identique à codepropretracetheoriepredictions.py.
           Produit un modèle analytique f_model(ω) sur toute la plage [z1, z2].

Étape 2 — Déconvolution FFT de Bissessur sur le signal analytique.
  Im(Z)(z) = ∫ γ(s) / (2·cosh(z−s)) ds       [Fredholm 1ère espèce]
  Déconvolution dans l'espace de Fourier :
    imZ_η  =  (π/2) · γ_η / cosh(η·π²)
    filtre(η) = cosh(η·π²) / (1 + W·cosh(η·π²)²)   [Tikhonov fréquentiel]
  Rct extrait par intégrale du pic avant-dernier (convention Bissessur T_V[-2]).

Pourquoi passer par le modèle Randles ?
  La méthode FFT suppose un signal Im(Z)(z) continu et analytique sur [-400, 420].
  Les données EC-Lab (~50–100 points) ne couvrent que ~2% de cette plage ; une
  interpolation directe génère des centaines de pics parasites. En passant par le
  modèle Randles, on dispose d'un signal analytique exact sur toute la grille.
  L'information physique reste entièrement issue des données expérimentales.

Références internes :
  - DRTparFFT.ipynb                 (déconvolution FFT sur modèle analytique)
  - alarecherchedutempsperdu.ipynb  (variation τ_d, extraction des maxima)
  - exploitationDRT.py              (application EC-Lab réelles via DRTtools)
  - codepropretracetheoriepredictions.py  (fit Randles Nelder-Mead de référence)
"""

import numpy as np
import scipy.fftpack as fftpack
import scipy.optimize as spo

from fits.base import BaseFitModel
from core.models import EISSpectrum, FitResult


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit physique
# ─────────────────────────────────────────────────────────────────────────────

def _Z_randles(omega: np.ndarray,
               R_e: float, R_e_prime: float, C_b: float,
               Q_dl: float, alpha: float, R_ct: float,
               tau_d: float, R_D: float) -> np.ndarray:
    """Impédance du circuit Randles modifié.
    Re — [ R'e // Cb ] — [ Rct // CPE(Qdl, α) ] — ZD(ω)
    Identique à la fonction f() de codepropretracetheoriepredictions.py.
    """
    Z_D = R_D * np.tanh(np.sqrt(1j * omega * tau_d)) / np.sqrt(1j * omega * tau_d)
    Z1  = R_ct + Z_D
    CPE = (np.cos(alpha * np.pi / 2) + 1j * np.sin(alpha * np.pi / 2)) * (omega ** alpha) * Q_dl
    Z2  = Z1 / (1 + CPE * Z1)
    Z3  = R_e_prime + Z2
    return (Z3 / (1 + 1j * Z3 * C_b * omega)) + R_e


# ─────────────────────────────────────────────────────────────────────────────
#  Fit Randles 8 paramètres
# ─────────────────────────────────────────────────────────────────────────────

def _loss_modulus(Z_pred: np.ndarray, Z_ref: np.ndarray) -> float:
    """Loss Modulus pondérée — identique à lossnorm() dans le code de référence."""
    return float(np.sum(
        (np.real(Z_pred  - Z_ref))**2 / (np.real(Z_ref)**2 + 1e-30)
        + (np.imag(Z_pred - Z_ref))**2 / (np.imag(Z_ref)**2 + 1e-30)
    ))


def _fit_randles_full(f_exp: np.ndarray,
                      Zre_exp: np.ndarray,
                      Zim_exp: np.ndarray) -> tuple:
    """Fit Randles 8 paramètres par Nelder-Mead — loss Modulus.

    Traduit de codepropretracetheoriepredictions.py avec estimations initiales
    robustes depuis la géométrie du diagramme de Nyquist.

    Returns:
        (R_e, R_e_prime, C_b, Q_dl, alpha, R_ct, tau_d, R_D)
    """
    omega   = 2.0 * np.pi * f_exp
    Z_exp   = Zre_exp + 1j * Zim_exp

    # ── Estimations initiales depuis le Nyquist ──────────────────────────────
    Re_init  = float(np.min(np.abs(Zre_exp)))
    Zre_span = float(np.max(Zre_exp) - Re_init)
    Rct_init = Zre_span * 0.6
    RD_init  = Zre_span * 0.4

    x0 = [Re_init,
          Re_init * 0.05,
          1e-9,
          1e-8,
          0.75,
          Rct_init,
          0.1,
          RD_init]

    bounds = [
        (Re_init * 0.5, Re_init * 2.0),  # R_e
        (1.0,           Re_init),         # R_e_prime
        (1e-11,         1e-6),            # C_b
        (1e-11,         1e-5),            # Q_dl
        (0.4,           1.0),             # alpha
        (Re_init * 0.1, Zre_span * 10),  # R_ct
        (1e-4,          10.0),            # tau_d
        (Re_init * 0.1, Zre_span * 5),   # R_D
    ]

    def objective(vec):
        try:
            Z_pred = _Z_randles(omega, *vec)
            return _loss_modulus(Z_pred, Z_exp)
        except Exception:
            return 1e30

    result = spo.minimize(
        objective,
        x0=x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 30000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )

    # Si la convergence échoue, relancer depuis le meilleur point trouvé
    if not result.success:
        result = spo.minimize(
            objective,
            x0=result.x,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )

    R_e, R_e_prime, C_b, Q_dl, alpha, R_ct, tau_d, R_D = result.x

    # Contraintes physiques minimales
    alpha = float(np.clip(alpha, 0.3, 1.0))
    R_ct  = float(np.clip(abs(R_ct), 1.0, None))
    R_D   = float(np.clip(abs(R_D),  1.0, None))

    return (R_e, R_e_prime, C_b, Q_dl, alpha, R_ct, tau_d, R_D)


# ─────────────────────────────────────────────────────────────────────────────
#  Fonctions FFT — traduites fidèlement depuis DRTparFFT.ipynb (Bissessur 2025)
# ─────────────────────────────────────────────────────────────────────────────

def _TF(x: np.ndarray, f: np.ndarray):
    """FFT discrète avec facteur de phase alterné RECT = (−1)^i."""
    RECT = np.array([1 - 2 * (i % 2) for i in range(len(x))])
    L    = fftpack.fft(f.copy()) * RECT
    L    = fftpack.fftshift(L)
    nu   = fftpack.fftshift(fftpack.fftfreq(len(x), x[1] - x[0]))
    return nu, L


def _TF_inv(x: np.ndarray, f: np.ndarray):
    """FFT inverse discrète avec facteur de phase alterné RECT = (−1)^i."""
    RECT = np.array([1 - 2 * (i % 2) for i in range(len(x))])
    L    = fftpack.ifft(f.copy()) * RECT
    L    = fftpack.fftshift(L)
    nu   = x.copy()
    nu   = fftpack.fftshift(fftpack.fftfreq(len(nu), nu[1] - nu[0]))
    return nu, L


def _local_maxima(DRT: list, l: int = 300):
    """Détecte les maxima locaux de |γ(s)| dans une fenêtre de demi-largeur l.

    Critères (Bissessur) :
      - A[i] = max sur [i−l, i+l]
      - A[i] > max_global / exp(10)

    Args:
        DRT : [S, A]  — ln(τ) et amplitudes |γ(s)|
        l   : demi-fenêtre en indices (300 dans les notebooks de Bissessur)

    Returns:
        T_V : ln(τ) des maxima, ordre croissant
        V   : amplitudes correspondantes
    """
    Tau     = DRT[0]
    A       = DRT[1]
    MAXGLOB = np.max(A) if np.max(A) > 0 else 1.0
    n       = len(A)
    V, T_V  = [], []
    for i in range(l + 1, n - l - 1):
        if np.max(A[i - l:i + l]) == A[i] and A[i] > MAXGLOB / np.exp(10):
            V.append(A[i])
            T_V.append(Tau[i])
    return np.array(T_V), np.array(V)


def _drt_from_model(R_e, R_e_prime, C_b, Q_dl, alpha, R_ct, tau_d, R_D,
                    W: float = 1e-8):
    """Calcule la DRT par déconvolution FFT sur le modèle analytique.

    Pipeline identique aux cellules 9 et 12 de DRTparFFT.ipynb :
      1. Grille z uniforme 2¹⁸ pts sur [−400, 420]
      2. Détection du débordement numérique dans tanh + padding exponentiel
      3. Déconvolution Fredholm par FFT avec filtre de régularisation W
      4. Recentrage S = S + DELTA

    Returns:
        S         : ln(τ) — abscisse DRT, shape (N_Z,)
        gamma_abs : |γ(s)| — ordonnée DRT, shape (N_Z,)
    """
    Z1_g, Z2_g = -400.0, 420.0
    N_Z        = 2**18
    Z_grid     = np.linspace(Z1_g, Z2_g, N_Z)
    DELTA      = (Z2_g + Z1_g) / 2.0

    # ── 1. Détection du point de débordement (cellule 9) ────────────────────
    IM    = np.abs(np.imag(_Z_randles(np.exp(-Z_grid) + 0j,
                                      R_e, R_e_prime, C_b,
                                      Q_dl, alpha, R_ct, tau_d, R_D)))
    IM1   = IM[1:] / (IM[:-1] + 1e-100)
    limz  = N_Z
    for z in range(N_Z - 1):
        if IM1[z] > 100:
            limz = z
            break

    omega_safe = np.exp(-Z_grid[:limz - 1])
    impart     = np.abs(np.imag(_Z_randles(omega_safe,
                                           R_e, R_e_prime, C_b,
                                           Q_dl, alpha, R_ct, tau_d, R_D)))
    Zpad  = Z_grid[limz - 1:]
    aleph = 1.0 if C_b != 0 else alpha
    impad = np.exp(-aleph * Zpad + aleph * Zpad[0] + np.log(impart[-1]))
    imZ   = np.concatenate([impart, impad])

    # ── 2. Déconvolution Fredholm par FFT (cellule 12) ──────────────────────
    ETA, imZ_eta = _TF(Z_grid, imZ)
    n_eta        = len(ETA)

    # |η| < 71 ⟹ |η·π²| < 701 < 710 = limite float64 de cosh
    COSH_LIMIT   = 71.0
    limeta1, limeta2 = 0, n_eta
    for idx in range(n_eta):
        if ETA[idx - 1] < -COSH_LIMIT and ETA[idx] >= -COSH_LIMIT:
            limeta1 = idx
        if ETA[idx - 1] < COSH_LIMIT  and ETA[idx] >= COSH_LIMIT:
            limeta2 = idx

    cosh_mid = np.cosh(ETA[limeta1:limeta2] * np.pi**2)
    conv_mid = cosh_mid / (1.0 + W * cosh_mid**2)
    conv     = np.concatenate([
        (1.0 / W) * np.ones(limeta1),
        conv_mid,
        (1.0 / W) * np.ones(n_eta - limeta2),
    ])
    CONV = (2.0 / np.pi) * imZ_eta * conv

    # ── 3. Retour dans l'espace ln(τ) ───────────────────────────────────────
    S, H_s = _TF_inv(ETA, CONV)
    S      = S + DELTA

    return S, np.abs(H_s)


# ─────────────────────────────────────────────────────────────────────────────
#  Modèle principal
# ─────────────────────────────────────────────────────────────────────────────

class DRTFitModel(BaseFitModel):
    """DRT par déconvolution FFT sur modèle Randles ajusté (Bissessur 2025).

    Étape 1 : fit Randles 8 paramètres (Nelder-Mead, loss Modulus) sur les
              données expérimentales — identique à codepropretracetheoriepredictions.py.
    Étape 2 : déconvolution FFT du signal analytique → DRT γ(s).
    Extraction Rct : intégrale de |γ(s)| sur le pic avant-dernier
                     (convention Bissessur T_V[−2]).

    Paramètre W (config.fit.W_drt, défaut 1e-8) :
      W = 1e-8  →  référence (peu lissé, pics bien résolus)
      W = 1e-6  →  plus lissé
      W = 1e-10 →  moins lissé, sensible au bruit
    """

    name         = "drt_fft"
    method       = "drt_fft"
    display_name = "DRT (FFT)"
    L_MAXIMA: int = 300   # demi-fenêtre maxima locaux (Bissessur)

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config) -> FitResult:
        """Calcule la DRT et extrait Rct.

        Args:
            spectrum : données EIS (f, Zre, Zim)
            config   : AppSettings — lit config.fit.W_drt (défaut 1e-8)

        Returns:
            FitResult avec :
              .Rct    : résistance de transfert de charge extraite de la DRT (Ω)
              .Zfit   : Z reconstruit depuis le modèle Randles ajusté
              .chi2   : erreur relative quadratique moyenne
              .params : tous les paramètres Randles + W + ln_tau_Rct
              .extras : S, H_s, tau_maxima, gamma_maxima  (pour drt_figure)
        """
        # ── 0. Données ──────────────────────────────────────────────────────
        f_exp   = np.asarray(spectrum.f,   dtype=float)
        Zre_exp = np.asarray(spectrum.Zre, dtype=float)
        Zim_exp = np.asarray(spectrum.Zim, dtype=float)
        W: float = float(getattr(getattr(config, 'fit', config), 'W_drt', 1e-8))

        # ── 1. Fit Randles 8 paramètres ─────────────────────────────────────
        (R_e, R_e_prime, C_b,
         Q_dl, alpha, R_ct,
         tau_d, R_D) = _fit_randles_full(f_exp, Zre_exp, Zim_exp)

        # ── 2. DRT FFT sur le modèle analytique ajusté ──────────────────────
        S, gamma_abs = _drt_from_model(R_e, R_e_prime, C_b,
                                       Q_dl, alpha, R_ct, tau_d, R_D,
                                       W=W)

        # ── 3. Extraction Rct — convention Bissessur T_V[−2] ────────────────
        tau_maxima, gamma_maxima = _local_maxima([S, gamma_abs], l=self.L_MAXIMA)

        Rct_drt: float
        tau_Rct: float

        if len(tau_maxima) >= 2:
            # Avant-dernier pic = transfert de charge (Bissessur T_V[-2])
            t_peak  = tau_maxima[-2]
            mask    = np.abs(S - t_peak) < 3.0
            Rct_drt = float(np.trapezoid(gamma_abs[mask], S[mask]))
            tau_Rct = float(t_peak)

        elif len(tau_maxima) == 1:
            # Un seul pic détecté — on le prend
            t_peak  = tau_maxima[0]
            mask    = np.abs(S - t_peak) < 3.0
            Rct_drt = float(np.trapezoid(gamma_abs[mask], S[mask]))
            tau_Rct = float(t_peak)

        else:
            # Aucun pic — fallback sur Rct du Randles (étape 1)
            Rct_drt = float(R_ct)
            tau_Rct = float(np.nan)

        # ── 4. Z reconstruit + χ² ───────────────────────────────────────────
        omega_exp = 2.0 * np.pi * f_exp
        Zfit = _Z_randles(omega_exp, R_e, R_e_prime, C_b,
                          Q_dl, alpha, R_ct, tau_d, R_D)

        chi2 = float(np.mean(
            (np.real(Zfit) - Zre_exp)**2 / (Zre_exp**2 + 1e-30)
            + (np.imag(Zfit) - Zim_exp)**2 / (Zim_exp**2 + 1e-30)
        ))

        # ── 5. FitResult ─────────────────────────────────────────────────────
        params = {
            "W":           W,
            "Re":          R_e,
            "Re_prime":    R_e_prime,
            "Cb":          C_b,
            "Qdl":         Q_dl,
            "alpha":       alpha,
            "Rct_randles": R_ct,
            "tau_d":       tau_d,
            "R_D":         R_D,
            "Rct_drt":     Rct_drt,
            "ln_tau_Rct":  tau_Rct,
            "ln_tau":      S,           # abscisse DRT — ln(τ)
            "gamma":       gamma_abs,   # ordonnée DRT — |γ(τ)|
            # tau_peaks en secondes (np.exp(S)) pour drt_figure qui fait np.log()
            "tau_peaks":   list(np.exp(tau_maxima)),
        }
        Zfit_re = np.real(Zfit)
        Zfit_im = np.imag(Zfit)
        return FitResult(
            model_name=self.name,
            params=params,
            params_std={k: 0.0 for k in params},
            Zfit_re=Zfit_re,
            Zfit_im=Zfit_im,
            chi2=chi2,
            residuals_re=Zre_exp - Zfit_re,
            residuals_im=Zim_exp - Zfit_im,
            Rct=Rct_drt,
            Rct_std=0.0,
            converged=True,
        )

    def predict(self, spectrum: EISSpectrum, params: dict) -> np.ndarray:
        """Reconstruit Z depuis les paramètres Randles stockés dans FitResult."""
        omega = 2.0 * np.pi * np.asarray(spectrum.f, dtype=float)
        return _Z_randles(
            omega,
            R_e       = params.get("Re",          2000.0),
            R_e_prime = params.get("Re_prime",      500.0),
            C_b       = params.get("Cb",            1e-9),
            Q_dl      = params.get("Qdl",           1e-8),
            alpha     = params.get("alpha",         0.75),
            R_ct      = params.get("Rct_randles", 5000.0),
            tau_d     = params.get("tau_d",         0.1),
            R_D       = params.get("R_D",         2000.0),
        )
