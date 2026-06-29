# -*- coding: utf-8 -*-
"""
fits/drt_fft.py — DRT FFT/Wiener sur SPECTRE IDÉAL (modèle Randles fitté)

⚠️ Ce module N'ANALYSE PAS les données expérimentales de façon indépendante :
il recalcule la DRT exacte d'un spectre idéal reconstruit depuis le modèle
Randles déjà fitté (fits/randles_full.py). C'est la méthode de la section
III.C "DRT with DFT" du papier Bissessur, Man, Gamby, Phys. Rev. E 113,
025502 (2026) (DOI: 10.1103/fn2s-z364) — "we need an exact calculation of
the DRT, for ideal EIS spectra" — utilisée pour étudier les lois d'échelle
MAD (Maxima Asymptotic Dynamics) sur des cas théoriques contrôlés, PAS pour
extraire un Rct indépendant du fit Randles à comparer à celui-ci.

Pour une DRT model-free, appliquée directement sur les données expérimentales
brutes sans hypothèse de circuit équivalent, voir fits/drt_tikhonov.py
(section III.B "DRT with DRTtools" du même papier) — c'est ce module qui
sert de DRT principale dans l'UI (calibration, reconstruction Nyquist,
comparaison de paramètres).

Méthode FFT/Wiener : Bissessur et al., PRE 2026 (éq. 2-3), traduite
fidèlement depuis le notebook de référence DRTparFFT.ipynb.

Pipeline :
  1. Fit intermédiaire du circuit Randles complet (fits/randles_full.py) pour
     obtenir Re, R'e, Cb, Rct, Qdl, α, R_D, τ_d.
  2. Grille log-ω uniforme dense (n_z points).
  3. Évaluation du modèle analytique Z_randles_full sur cette grille dense
     (pas d'interpolation des données expérimentales) — c'est le "spectre
     idéal" du papier.
  4. Déconvolution de Fredholm dans l'espace de Fourier, filtre Wiener,
     avec détection explicite des bornes spectrales utiles (limeta1/limeta2),
     appliquée sur Im(Z) du modèle analytique.
  5. Détection des maxima locaux de |γ(τ)| (fenêtre l=300, seuil max/e^10).
  6. Extraction de Rct selon la convention Bissessur (avant-dernier pic si
     ≥2 pics, intégrale trapèze sur ±3 en ln(τ)) — sert à l'étude des lois
     MAD, pas à une comparaison indépendante avec Rct_randles.
  7. Reconstruction de Z (modèle analytique + DRT) comparée aux données
     expérimentales réelles pour le χ² et l'erreur de reconstruction.
"""

import numpy as np
import scipy.fftpack as fftpack

from fits.base import BaseFitModel
from fits.physics import Z_randles_full
from fits.randles_full import RandlesFullModel
from core.models import EISSpectrum, FitResult


def _TF(x: np.ndarray, f: np.ndarray):
    RECT = np.array([1 - 2 * (i % 2) for i in range(len(x))])
    L    = fftpack.fft(f.copy()) * RECT
    L    = fftpack.fftshift(L)
    nu   = fftpack.fftshift(fftpack.fftfreq(len(x), x[1] - x[0]))
    return nu, L


def _TF_inv(x: np.ndarray, f: np.ndarray):
    RECT = np.array([1 - 2 * (i % 2) for i in range(len(x))])
    L    = fftpack.ifft(f.copy()) * RECT
    L    = fftpack.fftshift(L)
    nu   = x.copy()
    nu   = fftpack.fftshift(fftpack.fftfreq(len(nu), nu[1] - nu[0]))
    return nu, L


def _cfg_get(config, key: str, default):
    fit_cfg = config.get("fit", config) if isinstance(config, dict) else getattr(config, "fit", config)
    if isinstance(fit_cfg, dict):
        return fit_cfg.get(key, default)
    return getattr(fit_cfg, key, default)


def _local_maxima(gamma: np.ndarray, l: int = 300, threshold: float = 0.0) -> list:
    """Detect indices of local maxima of |gamma| within a sliding window.

    Args:
        gamma: |γ(τ)| array.
        l: Half-window size (indices) used to define "local".
        threshold: Minimum amplitude for a maximum to be retained.

    Returns:
        Sorted list of indices of local maxima, ascending in τ.
    """
    n = len(gamma)
    maxima = []
    for i in range(n):
        if gamma[i] <= threshold:
            continue
        lo = max(0, i - l)
        hi = min(n, i + l + 1)
        if gamma[i] >= gamma[lo:hi].max():
            maxima.append(i)
    return maxima


class DRTFFTModel(BaseFitModel):
    """DRT par déconvolution FFT directe (filtre Wiener, Bissessur et al. PRE 2026),
    appliquée sur un SPECTRE IDÉAL reconstruit depuis le modèle Randles ajusté
    (pas sur les données brutes). Outil d'étude théorique des lois MAD
    (section III.C du papier) — n'est pas une DRT indépendante du fit Randles.
    Pour la DRT model-free utilisée comme DRT principale de l'app, voir
    fits/drt_tikhonov.py (DRTTikhonovModel)."""

    name         = "drt_fft_ideal"
    method       = "drt_fft_ideal"
    display_name = "DRT FFT Wiener (spectre idéal Randles)"

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config, weights=None) -> FitResult:
        """Calcule la DRT par déconvolution FFT Wiener à partir du fit Randles
        et extrait Rct selon la convention Bissessur."""
        f_exp   = np.asarray(spectrum.f,   dtype=float)
        Zre_exp = np.asarray(spectrum.Zre, dtype=float)
        Zim_exp = np.asarray(spectrum.Zim, dtype=float)

        W   = float(_cfg_get(config, "drt_wiener_W", 1e-8))
        n_z = int(_cfg_get(config, "drt_n_z", 10000))

        # ── 1. Fit Randles intermédiaire ─────────────────────────────────────
        randles_config = config if isinstance(config, dict) else {}
        randles_result = RandlesFullModel().fit(spectrum, randles_config)
        rp = randles_result.params
        Rct_randles = float(randles_result.Rct)

        # ── 2. Grille log-ω uniforme dense ───────────────────────────────────
        omega_exp = 2.0 * np.pi * f_exp
        z_exp     = np.log(omega_exp)
        order     = np.argsort(z_exp)
        z_sorted  = z_exp[order]

        z1 = float(z_sorted[0])  - 1.0
        z2 = float(z_sorted[-1]) + 1.0
        DELTA  = (z1 + z2) / 2.0
        Z_grid = np.linspace(z1, z2, n_z)
        omega_grid = np.exp(Z_grid)

        # ── 3. Évaluation du modèle analytique sur la grille dense ──────────
        Z_model = Z_randles_full(
            omega_grid,
            rp["Re"], rp["Re_prime"], rp["Cb"], rp["Rct"],
            rp["Qdl"], rp["alpha"], rp["R_D"], rp["tau_d"],
        )
        imZ = -Z_model.imag  # convention Im(Z) positive
        reZ = Z_model.real

        # ── 4. TF + filtre Wiener (inchangé) ─────────────────────────────────
        ETA, imZ_eta = _TF(Z_grid, imZ)
        n_eta = len(ETA)

        LIM     = 31.0
        eta_lim = LIM / np.pi**2

        idx_in_range = np.where(np.abs(ETA) < eta_lim)[0]
        if idx_in_range.size > 0:
            limeta1, limeta2 = int(idx_in_range[0]), int(idx_in_range[-1])
        else:
            limeta1, limeta2 = 0, -1

        conv = np.full(n_eta, 1e-100)
        if limeta2 >= limeta1:
            sl = slice(limeta1, limeta2 + 1)
            cosh_eta = np.cosh(ETA[sl] * np.pi**2)
            conv[sl] = cosh_eta / (1.0 + W * cosh_eta**2)

        CONV = (2.0 / np.pi) * imZ_eta * conv

        # ── 5. Retour dans l'espace ln(τ) ────────────────────────────────────
        S_raw, H_s_raw = _TF_inv(ETA, CONV)
        S = S_raw + DELTA
        dS = (z2 - z1) / n_z

        gamma = np.abs(H_s_raw)
        tau   = np.exp(S)
        ln_tau = S

        # ── 6. Détection des maxima locaux et extraction de Rct ─────────────
        # Les bords du domaine ln(ω) souffrent d'artefacts numériques du filtre
        # Wiener (ringing) — on exclut une marge avant la recherche de maxima.
        margin_peaks = max(1, n_z // 20)
        peak_search_slice = slice(margin_peaks, n_z - margin_peaks)
        gamma_core = gamma[peak_search_slice]
        max_global = float(gamma_core.max()) if gamma_core.size else 0.0
        threshold = max_global / np.exp(10.0)
        maxima_idx_core = _local_maxima(gamma_core, l=300, threshold=threshold)
        maxima_idx = [i + margin_peaks for i in maxima_idx_core]

        if len(maxima_idx) >= 2:
            peak_idx = maxima_idx[-2]
        elif len(maxima_idx) == 1:
            peak_idx = maxima_idx[0]
        else:
            peak_idx = None

        if peak_idx is not None:
            center = ln_tau[peak_idx]
            mask = np.abs(ln_tau - center) <= 3.0
            if mask.sum() >= 2:
                x_win = ln_tau[mask]
                y_win = gamma[mask]
                ord_win = np.argsort(x_win)
                Rct_drt = float(np.trapezoid(y_win[ord_win], x_win[ord_win]))
            else:
                Rct_drt = float(gamma[peak_idx])
            tau_Rct = float(ln_tau[peak_idx])
        else:
            Rct_drt = Rct_randles
            tau_Rct = float("nan")

        Rct = Rct_drt

        # ── 7. Reconstruction Z (modèle + DRT) vs données expérimentales ────
        diff           = z_sorted[:, None] - S_raw[None, :]
        Zfit_im_pos    = np.real(np.sum(H_s_raw[None, :] / (2.0 * np.cosh(diff)), axis=1)) * dS
        Zfit_im_sorted = -Zfit_im_pos
        Zfit_re_sorted = np.interp(z_sorted, Z_grid, reZ)

        inv_order = np.argsort(order)
        Zfit_re = Zfit_re_sorted[inv_order]
        Zfit_im = Zfit_im_sorted[inv_order]

        residuals_re = Zre_exp - Zfit_re
        residuals_im = Zim_exp - Zfit_im

        Zmod2 = Zre_exp**2 + Zim_exp**2 + 1e-30
        chi2 = float(np.mean((residuals_re**2 + residuals_im**2) / Zmod2))
        reconstruction_error = float(np.mean(
            np.sqrt(residuals_re**2 + residuals_im**2) / np.sqrt(Zmod2)
        ))

        margin = max(1, n_z // 20)
        core_slice = slice(margin, n_z - margin)
        tau_max = float(tau[core_slice][np.argmax(gamma[core_slice])]) if n_z > 2 * margin else float("nan")

        params = {
            "Rct": Rct,
            "W": W,
            "tau_max": tau_max,
            "Rct_randles": Rct_randles,
            "tau_Rct": tau_Rct,
        }

        return FitResult(
            model_name=self.name,
            params=params,
            params_std={k: 0.0 for k in params},
            Zfit_re=Zfit_re,
            Zfit_im=Zfit_im,
            chi2=chi2,
            residuals_re=residuals_re,
            residuals_im=residuals_im,
            Rct=Rct,
            Rct_std=0.0,
            converged=randles_result.converged,
            drt_tau=tau,
            drt_gamma=gamma,
            drt_S=S,
            drt_lnGamma=np.log(np.maximum(gamma, 1e-300)),
            reconstruction_error=reconstruction_error,
        )
