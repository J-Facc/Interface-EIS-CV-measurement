# -*- coding: utf-8 -*-
"""fits/drt_fit.py — Moteur DRT unique : plugin ``BaseFitModel`` autour de bayes_drt2.

La distribution des temps de relaxation γ(τ) est calculée **exclusivement** par la
bibliothèque ``bayes_drt2`` de Huang (classe :class:`Inverter`, vendorée dans
``vendor/bayes_drt2/`` — voir ``vendor/README.md`` et ``THIRD_PARTY_LICENSES.md``).
Aucune autre implémentation (Tikhonov, NNLS, L-curve, ridge…) ne subsiste.

Deux modes, un seul moteur :

* ``optimize`` — MAP Stan (L-BFGS-B). **Défaut** : lancé automatiquement par
  ``core/pipeline.py`` sur chaque spectre, comme les autres fits.
* ``sample`` — HMC bayésien (lent, intervalles de crédibilité). **Jamais**
  automatique : uniquement à la demande via ``core.pipeline.recompute_drt`` (bouton
  UI). ``ui/`` ne touche jamais directement à :class:`Inverter`.

Les deux modes compilent des modèles Stan via CmdStan : la toolchain (installée par
``setup_drt_bayesien.py`` → ``cmdstanpy.install_cmdstan(compiler=True)``) est un
prérequis strict de toute DRT — il n'y a plus de chemin sans compilation.

Conventions (guide bayes_drt2 + code vendoré, qui fait foi) :

* **Signe.** Le loader stocke ``Zim = -Im(Z) > 0`` (demi-cercle capacitif
  au-dessus de l'axe réel). bayes_drt2 attend la convention physique ``Z'' < 0`` :
  ``Z = spectrum.Zre - 1j * spectrum.Zim``.
* **Tri HF→BF** sur les **vraies fréquences lues** dans le fichier (jamais
  reconstruites par ``np.logspace`` : la position des pics en τ en dépend).
* ``inv.fit(freq, Z, mode=...)`` — fréquences et Z complexe en positionnels.
* Récupération : ``gamma = inv.predict_distribution('DRT')`` et
  ``tau = inv.distributions['DRT']['tau']`` (garde-fou ``KeyError``).

Rct (grandeur de calibration θ_EIS) : **extraction par pic (convention
Bissessur)** — pic pénultième, intégrale trapèze de γ sur ±3 en ln(τ) autour du
pic. Le pic pénultième saute délibérément la queue de diffusion BF pour n'intégrer
que l'arc de transfert de charge, ce qui garde ``Rct_drt`` cohérent avec le
``Rct`` du Randles. Le repli sur ``Rp`` (aire totale = diffusion + transfert) n'est
utilisé qu'en dernier recours et **toujours signalé** (``FitResult.warnings`` +
``params['rct_source']``) pour que la calibration ne soit jamais calculée en
silence sur une grandeur différente.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.logger import get_logger
from core.models import EISSpectrum, FitResult
from fits.base import BaseFitModel

log = get_logger("drt_fit")

DIST_NAME = "DRT"

# ── Import protégé du paquet vendoré ────────────────────────────────────────
# vendor/bayes_drt2/inversion.py importe cvxopt ET cmdstanpy au niveau module :
# si l'extra DRT n'est pas installé, l'ImportError est capturée et la DRT est
# désactivée proprement (bayes_available() → False) au lieu de casser l'app.
try:
    from vendor.bayes_drt2.inversion import Inverter  # type: ignore
    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # ImportError (cvxopt/cmdstanpy/matplotlib) ou autre
    Inverter = None  # type: ignore
    _IMPORT_ERROR = str(exc)
    log.warning(f"Moteur DRT bayes_drt2 indisponible : {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Disponibilité du moteur / de la toolchain
# ─────────────────────────────────────────────────────────────────────────────
def bayes_available() -> bool:
    """True si le paquet vendoré bayes_drt2 (et ses deps) est importable."""
    return Inverter is not None


def import_error() -> Optional[str]:
    """Raison de l'indisponibilité de la DRT, ou None si tout va bien."""
    return _IMPORT_ERROR


def cmdstan_available() -> bool:
    """True si une installation CmdStan est présente (requise pour tout fit DRT).

    N'installe ni ne compile rien : teste seulement ``cmdstanpy.cmdstan_path()``,
    qui lève si CmdStan est absent.
    """
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Extraction de Rct par pic (convention Bissessur) — grandeur de calibration
# ─────────────────────────────────────────────────────────────────────────────
def _local_maxima(gamma: np.ndarray, l: int = 3, threshold: float = 0.0) -> list:
    """Indices des maxima locaux de γ dans une fenêtre glissante de demi-largeur ``l``."""
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


def _extract_rct_peak(
    tau: np.ndarray, gamma: np.ndarray
) -> Tuple[Optional[float], float, str, str]:
    """Rct de l'arc de transfert de charge (convention Bissessur).

    Sélectionne le pic **pénultième** de γ(τ) (avant-dernier maximum local) si au
    moins deux pics sont détectés — ce pic correspond à l'arc de transfert de
    charge, la queue de diffusion BF étant écartée. Rct = ∫ γ dlnτ par trapèze sur
    ±3 en ln(τ) autour du pic.

    Returns:
        (Rct, tau_Rct, source, warning) où ``Rct`` vaut ``None`` si aucun pic n'est
        exploitable (le repli Rp est alors géré par l'appelant). ``source`` ∈
        {'peak_penultimate', 'peak_single', 'none'} ; ``warning`` non vide dès
        qu'un repli/une ambiguïté doit être signalé à l'utilisateur.
    """
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    if n == 0:
        return None, float("nan"), "none", "DRT vide : Rct indisponible."

    ln_tau = np.log(np.maximum(tau, 1e-300))

    # Fenêtre « cœur » : on écarte les bords (artefacts de grille) pour la
    # détection des maxima, mais l'intégrale reste sur toute la grille.
    margin = max(1, n // 20)
    has_core = n > 2 * margin
    core = slice(margin, n - margin) if has_core else slice(0, n)
    gamma_core = gamma[core]
    max_global = float(gamma_core.max()) if gamma_core.size else 0.0
    threshold = max_global * 1e-3
    l_window = max(1, n // 15)
    offset = margin if has_core else 0
    maxima = [i + offset for i in _local_maxima(gamma_core, l=l_window, threshold=threshold)]

    warning = ""
    if len(maxima) >= 2:
        peak_idx = maxima[-2]  # pénultième : arc de transfert de charge
        source = "peak_penultimate"
    elif len(maxima) == 1:
        peak_idx = maxima[0]
        source = "peak_single"
        warning = (
            "DRT à un seul pic : Rct extrait de ce pic unique (pas de pénultième "
            "disponible). Vérifier qu'il s'agit bien de l'arc de transfert de charge."
        )
    else:
        return None, float("nan"), "none", (
            "Aucun pic DRT détecté : extraction par pic impossible."
        )

    center = ln_tau[peak_idx]
    mask = np.abs(ln_tau - center) <= 3.0
    if mask.sum() >= 2:
        x_win = ln_tau[mask]
        y_win = gamma[mask]
        order = np.argsort(x_win)
        Rct = float(np.trapezoid(y_win[order], x_win[order]))
    else:
        Rct = float(gamma[peak_idx])
    return Rct, float(ln_tau[peak_idx]), source, warning


# ─────────────────────────────────────────────────────────────────────────────
# Cœur du fit
# ─────────────────────────────────────────────────────────────────────────────
def _mode_from_config(config) -> str:
    """Mode DRT depuis la config (``fit.drt.mode``), défaut ``'optimize'``."""
    try:
        fit_cfg = config.get("fit", {}) if isinstance(config, dict) else {}
        drt_cfg = fit_cfg.get("drt", {}) if isinstance(fit_cfg, dict) else {}
        mode = drt_cfg.get("mode")
    except AttributeError:
        mode = None
    return mode if mode in ("optimize", "sample") else "optimize"


def _reconstruction_metrics(
    spectrum: EISSpectrum, Zfit_re: np.ndarray, Zfit_im: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Résidus + misfit normalisé par le module (modèle-libre : pas de χ²ᵣ classique)."""
    Zre = np.asarray(spectrum.Zre, dtype=float)
    Zim = np.asarray(spectrum.Zim, dtype=float)
    residuals_re = Zre - Zfit_re
    residuals_im = Zim - Zfit_im
    Zmod2 = Zre ** 2 + Zim ** 2 + 1e-30
    chi2_reduced = float(np.mean((residuals_re ** 2 + residuals_im ** 2) / Zmod2))
    reconstruction_error = float(
        np.mean(np.sqrt(residuals_re ** 2 + residuals_im ** 2) / np.sqrt(Zmod2))
    )
    return chi2_reduced, reconstruction_error, residuals_re, residuals_im


def fit_drt(spectrum: EISSpectrum, mode: str = "optimize", model_name: str = "drt_bayes") -> FitResult:
    """Calcule la DRT d'un spectre avec :class:`Inverter` et assemble un ``FitResult``.

    Point d'entrée unique du calcul DRT (utilisé par :class:`DRTBayesModel.fit` et,
    indirectement, par ``core.pipeline.recompute_drt``). Aucune UI ici.
    """
    if not bayes_available():
        raise RuntimeError(f"Moteur DRT indisponible : {_IMPORT_ERROR}")
    if mode not in ("optimize", "sample"):
        raise ValueError(f"Mode DRT inconnu : {mode!r} (attendu 'optimize' ou 'sample').")

    freq = np.asarray(spectrum.f, dtype=float)
    Zre = np.asarray(spectrum.Zre, dtype=float)
    Zim = np.asarray(spectrum.Zim, dtype=float)
    # Convention de signe : loader stocke Zim = -Im(Z) > 0 → Z physique = Zre - j·Zim.
    Z = Zre - 1j * Zim

    # Tri par fréquence décroissante (HF→BF) sur les vraies fréquences.
    order = np.argsort(freq)[::-1]
    freq_sorted = freq[order]
    Z_sorted = Z[order]

    inv = Inverter()
    inv.fit(freq_sorted, Z_sorted, mode=mode)

    # Récupération γ(τ) avec garde-fou KeyError (inspection des clés dispo).
    try:
        tau = np.asarray(inv.distributions[DIST_NAME]["tau"], dtype=float)
    except KeyError as exc:
        raise KeyError(
            f"Distribution '{DIST_NAME}' absente de l'Inverter. "
            f"Clés disponibles : {list(inv.distributions.keys())}."
        ) from exc
    gamma = np.asarray(inv.predict_distribution(DIST_NAME, tau=tau), dtype=float)

    # Intervalles de crédibilité : mode 'sample' uniquement.
    gamma_lo: Optional[np.ndarray] = None
    gamma_hi: Optional[np.ndarray] = None
    if mode == "sample":
        try:
            gamma_lo = np.asarray(
                inv.predict_distribution(DIST_NAME, tau=tau, percentile=2.5), dtype=float
            )
            gamma_hi = np.asarray(
                inv.predict_distribution(DIST_NAME, tau=tau, percentile=97.5), dtype=float
            )
        except Exception as exc:  # pragma: no cover - dépend du backend Stan
            log.warning(f"Intervalles de crédibilité indisponibles : {exc}")

    # Rct par pic (Bissessur) ; repli Rp signalé si aucun pic exploitable.
    warnings_list: list = []
    Rct, tau_Rct, rct_source, rct_warning = _extract_rct_peak(tau, gamma)
    Rp = float(inv.predict_Rp())
    if Rct is None:
        Rct = Rp
        rct_source = "rp_fallback"
        warnings_list.append(
            "Rct DRT calculé par REPLI sur Rp (aire totale sous γ : diffusion + "
            "transfert de charge). Grandeur DIFFÉRENTE de l'arc de transfert — "
            "calibration θ_EIS à interpréter avec prudence."
        )
    if rct_warning:
        warnings_list.append(rct_warning)

    # Reconstruction Z aux fréquences mesurées, remise dans l'ordre d'origine.
    Zfit_sorted = np.asarray(inv.predict_Z(freq_sorted))
    inv_order = np.argsort(order)
    Zfit = Zfit_sorted[inv_order]
    Zfit_re = np.real(Zfit)
    Zfit_im = -np.imag(Zfit)  # retour convention EISSpectrum : Zim = -Im(Z) > 0.

    chi2_reduced, reconstruction_error, residuals_re, residuals_im = _reconstruction_metrics(
        spectrum, Zfit_re, Zfit_im
    )

    params = {
        "Rct": float(Rct),
        "Rp": Rp,
        "tau_Rct": tau_Rct,
        "n_tau": int(len(tau)),
        # Provenance du Rct : lisible dans l'UI et les exports pour qu'un Rct issu
        # d'un repli ne soit jamais confondu avec un Rct d'arc de transfert.
        "rct_source": rct_source,
        "drt_mode": mode,
    }
    params_std = {"Rct": 0.0, "Rp": 0.0, "tau_Rct": 0.0, "n_tau": 0.0}

    return FitResult(
        model_name=model_name,
        params=params,
        params_std=params_std,
        Zfit_re=Zfit_re,
        Zfit_im=Zfit_im,
        chi2_reduced=chi2_reduced,
        residuals_re=residuals_re,
        residuals_im=residuals_im,
        target_param="Rct",
        target_value=float(Rct),
        target_std=0.0,
        converged=True,
        drt_tau=tau,
        drt_gamma=gamma,
        drt_mode=mode,
        drt_gamma_lo=gamma_lo,
        drt_gamma_hi=gamma_hi,
        reconstruction_error=reconstruction_error,
        warnings=warnings_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plugin BaseFitModel — découvert par fits/registry.py, lancé par core/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
class DRTBayesModel(BaseFitModel):
    """DRT model-free via ``bayes_drt2`` (:class:`Inverter`).

    Inversion hiérarchique bayésienne (lignée Ciucci-Chen) : ``optimize`` (MAP,
    défaut) ou ``sample`` (HMC, à la demande). Retourne un ``FitResult`` standard
    (γ(τ) dans ``drt_tau``/``drt_gamma``, mode dans ``drt_mode``, Rct par pic).
    """

    name = "drt_bayes"
    label = "DRT bayes_drt2"
    method = "bayes_drt2 / Inverter"
    display_name = "DRT bayes_drt2 (Inverter)"
    description = (
        "Distribution des temps de relaxation γ(τ) par inversion hiérarchique "
        "bayésienne (bibliothèque bayes_drt2 de Huang, classe Inverter). Mode "
        "'optimize' (MAP Stan) par défaut ; mode 'sample' (HMC, intervalles de "
        "crédibilité) à la demande."
    )

    def initial_guess(self, spectrum: EISSpectrum, config: dict) -> dict:
        # Méthode model-free : aucun paramètre de circuit à initialiser.
        return {}

    def bounds(self, config: dict) -> tuple:
        # Aucun paramètre borné (γ(τ) ≥ 0 est géré par le modèle bayésien).
        return ({}, {})

    def fit(self, spectrum: EISSpectrum, config: dict, weights=None) -> FitResult:
        """Fit DRT (mode lu dans ``config.fit.drt.mode``, défaut 'optimize').

        ``weights`` est ignoré : bayes_drt2 estime lui-même la structure d'erreur.
        """
        return fit_drt(spectrum, mode=_mode_from_config(config), model_name=self.name)

    def predict(self, spectrum: EISSpectrum, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Impédance reconstruite (Zfit_re, Zfit_im) — convention Zim = -Im(Z) > 0."""
        fr = self.fit(spectrum, config)
        return fr.Zfit_re, fr.Zfit_im
