# -*- coding: utf-8 -*-
"""
fits/drt_tikhonov.py — DRT model-free via pyDRTtools' RBF-discretized
Tikhonov regularization + non-negativity-constrained QP (cvxopt).

Méthode de référence : Wan, Saccoccio, Chen, Ciucci, Electrochim. Acta 184,
483 (2015) (DRTtools), citée par Bissessur, Man, Gamby, Phys. Rev. E 113,
025502 (2026) (DOI: 10.1103/fn2s-z364), section III.B "DRT with DRTtools".
λ-selection via generalized cross-validation: Maradesa, Py, Wan, Effat,
Ciucci, J. Electrochem. Soc. 170 (2023) 030502.

Ce module appelle directement le cœur de calcul de pyDRTtools
(https://github.com/ciuccislab/pyDRTtools, MIT, vendoré dans
fits/_pydrttools/ — voir THIRD_PARTY_LICENSES.md), plutôt qu'une
réimplémentation maison, pour la fidélité à l'outil de référence cité dans
la littérature. Cela remplace une précédente réimplémentation Dirac-basis
+ NNLS qui souffrait d'un bug de "peigne" de pics isolés sur données
bruitées (sous-régularisation au voisinage du bord bas de la sélection
L-curve, combinée à une grille tau plus fine que ce que les données
pouvaient résoudre).

Contrairement à fits/drt_fft.py (DRT FFT/Wiener — section III.C du papier,
appliquée à un spectre IDÉAL reconstruit depuis un fit Randles), cette
méthode est appliquée DIRECTEMENT sur les données expérimentales brutes
(spectrum.f / Zre / Zim), sans aucun fit de circuit équivalent intermédiaire.
γ(τ) est donc complètement indépendante de toute hypothèse de topologie de
circuit — c'est la méthode "model-free" du papier.

Pipeline (suit les valeurs par défaut de l'interface GUI de pyDRTtools,
cf. layout.py — pas de valeurs inventées) :
  1. Points de collocation tau = 1/f sur les fréquences expérimentales
     elles-mêmes (convention DRTtools : pas une grille synthétique dense).
  2. Discrétisation RBF (rbf_type='Gaussian') du noyau de Fredholm
     (A_re, A_im), shape factor epsilon via compute_epsilon
     (shape_control='FWHM Coefficient', coeff=0.5).
  3. Régularisation de Tikhonov sur la dérivée première de γ
     (der_used='1st order' — défaut GUI, assemble_M_1), pas la dérivée
     seconde utilisée par l'ancienne réimplémentation maison.
  4. Résistance série R0 isolée en tant que colonne dédiée de A_re
     (induct_used=0, "Fitting w/o Inductance" — défaut GUI ; pas de terme
     d'inductance, non pertinent pour ce capteur EIS).
  5. Sélection automatique de λ par validation croisée généralisée
     (cv_type='GCV', défaut de runs.py::simple_run) via une recherche
     scalaire bornée (pas basics.optimal_lambda/SLSQP, numériquement
     fragile sur ces données — voir commentaire dans fit() ; ni rGCV, qui
     sous-régularise systématiquement sur ce type de spectre — voir le
     même commentaire), adaptée à un usage headless (pas de valeur
     "custom" saisie manuellement comme le permet la GUI interactive).
  6. Résolution par QP sous contrainte de positivité (cvxopt) :
     basics.solve_gamma / quad_format_combined.
  7. Conversion des coefficients RBF x vers γ(τ) sur une grille fine
     (x_to_gamma), pour l'affichage et l'extraction de Rct.
  8. Extraction de Rct selon la convention Bissessur (avant-dernier pic
     local si ≥2 pics, intégrale trapèze de γ(τ) sur ±3 en ln(τ) autour du
     pic).
  9. Reconstruction de Z(ω) en réinjectant x dans A_re/A_im (pas de lien
     structurel avec un circuit Randles).
"""

import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import minimize_scalar

from fits.base import BaseFitModel
from fits._pydrttools import basics
from fits._pydrttools import nearest_PD
from fits._pydrttools import parameter_selection as param
from core.models import EISSpectrum, FitResult

# Défauts GUI de pyDRTtools (layout.py) — non inventés.
_RBF_TYPE = "Gaussian"
_SHAPE_CONTROL = "FWHM Coefficient"
_COEFF = 0.5
_DER_USED = "1st order"
_DATA_USED = "Combined Re-Im Data"
_N_RL = 1  # induct_used=0 ("Fitting w/o Inductance", défaut GUI) : une seule colonne hors-tau, R0.


def _cfg_get(config, key: str, default):
    fit_cfg = config.get("fit", config) if isinstance(config, dict) else getattr(config, "fit", config)
    if isinstance(fit_cfg, dict):
        drt_cfg = fit_cfg.get("drt", {})
    else:
        drt_cfg = getattr(fit_cfg, "drt", {})
    if isinstance(drt_cfg, dict) and key in drt_cfg:
        return drt_cfg[key]
    return getattr(drt_cfg, key, default) if not isinstance(drt_cfg, dict) else default


def _local_maxima(gamma: np.ndarray, l: int = 3, threshold: float = 0.0) -> list:
    """Detect indices of local maxima of gamma within a sliding window."""
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


class DRTTikhonovModel(BaseFitModel):
    """DRT model-free via pyDRTtools (RBF Tikhonov + QP sous contrainte de
    positivité), appliquée directement sur les données expérimentales
    (Wan, Saccoccio, Chen, Ciucci 2015 ; Maradesa et al. 2023 ;
    Bissessur, Man, Gamby PRE 2026, section III.B)."""

    name = "drt_tikhonov"
    label = "DRT Tikhonov + NNLS"
    method = "drt_tikhonov"
    display_name = "DRT Tikhonov + NNLS"
    description = (
        "Déconvolution model-free de la distribution des temps de relaxation "
        "via le cœur de calcul de pyDRTtools (RBF gaussienne + Tikhonov "
        "ordre 1 + QP sous contrainte de positivité), appliquée directement "
        "sur les données expérimentales (sans fit de circuit équivalent "
        "intermédiaire)."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        return {}

    def bounds(self, config: dict) -> tuple:
        return {}, {}

    def fit(self, spectrum: EISSpectrum, config, weights=None) -> FitResult:
        f_exp = np.asarray(spectrum.f, dtype=float)
        Zre_exp = np.asarray(spectrum.Zre, dtype=float)
        Zim_exp = np.asarray(spectrum.Zim, dtype=float)

        order = np.argsort(f_exp)
        freq = f_exp[order]
        Zre = Zre_exp[order]
        Zim = -Zim_exp[order]  # pyDRTtools: convention -Im(Z) < 0 pour un demi-cercle capacitif.

        # Points de collocation tau = 1/f (convention DRTtools, cf. EIS_object.__init__).
        tau = 1.0 / freq
        n_taus = tau.size

        epsilon = basics.compute_epsilon(freq, _COEFF, _RBF_TYPE, _SHAPE_CONTROL)
        A_re_rbf = basics.assemble_A_re(freq, tau, epsilon, _RBF_TYPE)
        A_im_rbf = basics.assemble_A_im(freq, tau, epsilon, _RBF_TYPE)
        M_rbf = basics.assemble_M_1(tau, epsilon, _RBF_TYPE)

        n_freqs = freq.size
        A_re = np.zeros((n_freqs, n_taus + _N_RL))
        A_re[:, _N_RL:] = A_re_rbf
        A_re[:, 0] = 1.0  # colonne R0 : ne contribue qu'à Re(Z).

        A_im = np.zeros((n_freqs, n_taus + _N_RL))
        A_im[:, _N_RL:] = A_im_rbf

        M = np.zeros((n_taus + _N_RL, n_taus + _N_RL))
        M[_N_RL:, _N_RL:] = M_rbf

        lambda_auto = bool(_cfg_get(config, "lambda_auto", True))
        reg_param_init = float(_cfg_get(config, "lambda_fixed", 1e-3))
        if lambda_auto:
            # basics.optimal_lambda's SLSQP call is numerically fragile on
            # this score (gradient computed by finite differences blows up
            # for a 1-D scalar objective spanning several orders of
            # magnitude, making SLSQP report "Inequality constraints
            # incompatible" after a single step on real-scale data) — so we
            # minimize the same score via a bounded scalar search instead,
            # over the same log-lambda bounds pyDRTtools uses (1e-7 to 1e0).
            #
            # cv_type='GCV' is the literal default of pyDRTtools' own
            # runs.py::simple_run. An earlier version of this code used
            # rGCV instead, reasoning that plain GCV is documented
            # (Maradesa, Py, Wan, Effat, Ciucci, J. Electrochem. Soc. 170
            # (2023) 030502) as occasionally unstable on sparse/noisy
            # spectra. That reasoning was backwards in practice: on this
            # integration's synthetic noisy-Randles benchmark
            # (tests/test_drt_tikhonov.py), rGCV is the one that lands
            # at/near the lambda search's lower bound (1e-7),
            # under-regularizing gamma(tau) into absorbing measurement
            # noise as spurious peaks while producing an artificially
            # near-perfect Z(omega) reconstruction — classic overfitting.
            # Plain GCV (cross-checked against mGCV, which agrees closely)
            # instead lands well inside the search interval across many
            # noise realizations, giving a smaller, more plausible peak
            # count and a residual reconstruction error consistent with
            # real EIS DRT fits. So GCV is used here, not rGCV.
            log_bounds = (np.log(1e-7), np.log(1e0))
            opt = minimize_scalar(
                param.compute_GCV,
                args=(A_re, A_im, Zre, Zim, M, _DATA_USED, 0),
                bounds=log_bounds,
                method="bounded",
            )
            lam = float(np.exp(opt.x))
        else:
            lam = reg_param_init

        # basics.solve_gamma sizes its positivity-constraint matrix to
        # A_re.shape[0] (n_freqs), which only matches len(x) when no R0
        # column is appended; with our R0 column len(x) = n_taus + N_RL, so
        # we build the constraint inline instead, exactly as the GUI's
        # simple_run (induct_used=0 branch) does.
        n_unknowns = n_taus + _N_RL
        G = matrix(-np.identity(n_unknowns))
        h = matrix(np.zeros(n_unknowns))
        # H = 2*(A_re^T A_re + A_im^T A_im + lambda*M) can fail cvxopt's
        # Cholesky-based KKT rank check even at a well-chosen lambda: the R0
        # column (constant 1, unpenalized by M) can be near-collinear with
        # the RBF columns at the low-frequency end on some noisy
        # realizations, leaving H just barely indefinite by floating-point
        # roundoff — not an under-regularization issue, so retrying with a
        # bigger lambda does not fix it (confirmed: still fails up to
        # lambda=1e21 on the seed that exposed this). pyDRTtools' own
        # parameter_selection.py already guards its internal GCV-family
        # matrices the same way, with nearest_PD (Higham 1988) snapping a
        # near-PD matrix to the closest true PD one; applying the same
        # vendored helper to H here is the same fix the upstream toolkit
        # already relies on elsewhere, not an invented workaround.
        H, c = basics.quad_format_combined(A_re, A_im, Zre, Zim, M, lam)
        if not nearest_PD.is_PD(H):
            H = nearest_PD.nearest_PD(H)
        sol = solvers.qp(matrix(H), matrix(c), G, h, options={"show_progress": False})
        x = np.array(sol["x"]).flatten()
        R0 = float(x[0])
        x_gamma = x[_N_RL:]

        # γ(τ) sur une grille fine pour l'affichage et l'extraction de Rct
        # (convention DRTtools : demi-décade de marge de part et d'autre,
        # 10 points par point expérimental — cf. EIS_object.tau_fine).
        tau_fine = np.logspace(np.log10(tau.min()) - 0.5, np.log10(tau.max()) + 0.5, 10 * n_taus)
        tau_fine = np.sort(tau_fine)
        _, gamma = basics.x_to_gamma(x_gamma, tau_fine, tau, epsilon, _RBF_TYPE)
        gamma = np.asarray(gamma).flatten()
        ln_tau = np.log(tau_fine)

        # ── Détection des maxima locaux et extraction de Rct (convention Bissessur) ──
        n_fine = len(gamma)
        margin = max(1, n_fine // 20)
        core_slice = slice(margin, n_fine - margin) if n_fine > 2 * margin else slice(0, n_fine)
        gamma_core = gamma[core_slice]
        max_global = float(gamma_core.max()) if gamma_core.size else 0.0
        threshold = max_global * 1e-3
        l_window = max(1, n_fine // 15)
        maxima_idx_core = _local_maxima(gamma_core, l=l_window, threshold=threshold)
        maxima_idx = [i + margin for i in maxima_idx_core]

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
                Rct = float(np.trapezoid(y_win[ord_win], x_win[ord_win]))
            else:
                Rct = float(gamma[peak_idx])
            tau_Rct = float(ln_tau[peak_idx])
        else:
            Rct = 0.0
            tau_Rct = float("nan")

        # ── Reconstruction de Z(omega) sur les fréquences expérimentales ────
        Zfit_re_sorted = A_re @ x
        Zfit_im_sorted = -(A_im @ x)  # retour à la convention EISSpectrum (-Im(Z) > 0).
        inv_order = np.argsort(order)
        Zfit_re = Zfit_re_sorted[inv_order]
        Zfit_im = Zfit_im_sorted[inv_order]

        residuals_re = Zre_exp - Zfit_re
        residuals_im = Zim_exp - Zfit_im
        Zmod2 = Zre_exp ** 2 + Zim_exp ** 2 + 1e-30
        chi2 = float(np.mean((residuals_re ** 2 + residuals_im ** 2) / Zmod2))
        reconstruction_error = float(np.mean(
            np.sqrt(residuals_re ** 2 + residuals_im ** 2) / np.sqrt(Zmod2)
        ))

        tau_max = float(tau_fine[core_slice][np.argmax(gamma_core)]) if gamma_core.size else float("nan")

        params = {
            "Rct": Rct,
            "R0": R0,
            "lambda": lam,
            "tau_max": tau_max,
            "tau_Rct": tau_Rct,
            "n_tau": n_fine,
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
            converged=True,
            drt_tau=tau_fine,
            drt_gamma=gamma,
            drt_S=ln_tau,
            drt_lnGamma=np.log(np.maximum(gamma, 1e-300)),
            reconstruction_error=reconstruction_error,
        )
