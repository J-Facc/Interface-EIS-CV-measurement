# -*- coding: utf-8 -*-
"""drt/engine.py — moteur DRT durci autour de ``drt.bayes_drt2.Inverter``.

Remplace, à terme (étape 5 de la refonte), ``fits/drt_fit.py`` + ``vendor/``. Aucun import
Streamlit ; ne dépend ni de ``fits/drt_fit.py`` ni de ``vendor/``.

Ce que ce module ajoute à ``Inverter`` (et pourquoi — AUDIT.md §4)
------------------------------------------------------------------
* **HMC par défaut** (``mode='sample'``, décision utilisateur : fiabilité > vitesse) ;
  ``mode='optimize'`` (MAP L-BFGS) reste disponible pour un aperçu rapide.
* **Réglages explicites**, tous des constantes documentées ci-dessous et justifiées par
  ``drt/VALIDATION_REGLAGES.md`` — aucun défaut implicite de la bibliothèque n'est subi
  (correction de DRT-1/DRT-9 : ``nonneg``, ``init_from_ridge``, graine…).
* **Diagnostics HMC lus, jamais inventés** : R-hat, ESS bulk/tail, divergences,
  saturation de profondeur d'arbre, E-BFMI (``drt.diagnostics.sampler_diagnostics``,
  API cmdstanpy ``CmdStanMCMC.summary/divergences/max_treedepths/method_variables``).
* **Gardes qualité** (``drt.diagnostics``) qui remplissent ``FitResult.warnings`` ;
  ``converged`` est CALCULÉ (plus de ``True`` codé en dur — correction de DRT-2).
* ``params['tau_Rct']`` en **secondes** (correction de DRT-4 : c'était ln τ).
* ``chi2_reduced = NaN`` : la DRT ne calcule pas de χ² pondéré ; l'erreur de
  reconstruction relative a son propre champ ``reconstruction_error_relative`` (max) et
  ``reconstruction_error`` suit la formule RMS de ``fits/randles_full.py`` (DRT-5).

Conventions (inchangées par rapport à ``fits/drt_fit.py``)
----------------------------------------------------------
* Le loader stocke ``Zim = −Im(Z) > 0`` ; bayes_drt2 attend ``Z = Zre − j·Zim``.
* Fréquences **mesurées** triées HF→BF avant l'inversion ; tout est remis dans l'ordre
  d'origine en sortie.
* ``Rct`` : convention « Bissessur » — pic **pénultième** de γ(τ), aire trapèze sur
  ±3 en ln τ ; un seul pic → ``peak_single`` (signalé) ; aucun → repli ``Rp`` (signalé).
  Seule différence avec ``fits/drt_fit.py`` : les pics candidats sont pris DANS la fenêtre
  de τ mesurée (:data:`RCT_PEAKS_IN_MEASURED_WINDOW`, justification mesurée dans
  VALIDATION_REGLAGES.md §5). AUDIT.md DRT-10 n'est traité qu'en partie : une ondulation
  située DANS la fenêtre pourrait encore être prise pour un pic.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from core.models import EISSpectrum, FitResult
from drt import diagnostics as dg

DIST_NAME = "DRT"
MODEL_NAME = "drt_bayes"

# ─────────────────────────────────────────────────────────────────────────────
# Réglages par défaut — CHAQUE valeur est justifiée dans drt/VALIDATION_REGLAGES.md
# ─────────────────────────────────────────────────────────────────────────────
#: Mode par défaut : HMC (NUTS). Décision utilisateur (fiabilité > vitesse) ; seul
#: mode fournissant des diagnostics de convergence et des intervalles de crédibilité.
DEFAULT_MODE: str = "sample"
#: DRT contrainte ≥ 0 (modèle Stan ``Series_pos``). VALIDATION_REGLAGES.md §2-3 : sans
#: elle, HMC ne mélange pas sur des arcs RC nets (profondeur d'arbre saturée, R-hat jusqu'à
#: 2,3) et le MAP tombe dans des optima à Rp < 0. Coût : biais positif de la moyenne a
#: posteriori de Rp (+0,6 à +1,5 %, §4).
DEFAULT_NONNEG: bool = True
#: Initialisation par la solution ridge hyperparamétrique. VALIDATION_REGLAGES.md §2 et
#: §3.4 : indispensable au MAP (sinon Rp +7,8 % sur un cas), neutre en HMC.
DEFAULT_INIT_FROM_RIDGE: bool = True
#: Graine Stan (optimiseur ET échantillonneur). Même valeur que le défaut amont
#: (``Inverter.fit(random_seed=1234)``), mais désormais EXPLICITE et enregistrée dans
#: chaque résultat : deux appels de mêmes données et même graine donnent le même γ(τ).
DEFAULT_RANDOM_SEED: int = 1234
#: Contrôle HMC. VALIDATION_REGLAGES.md §3.2-3.3 : le défaut amont (2 chaînes × 200/200)
#: ne permet ni un R-hat fiable ni l'ESS ≥ 100/chaîne ; 4 chaînes (Vehtari et al. 2021) ;
#: 2000 tirages/chaîne car 500 puis 1000 laissaient l'ESS sous 400 (= 100 × 4 chaînes) sur
#: des spectres de type Randles (ESS bulk/tail 356/254 à 1000 tirages).
DEFAULT_CHAINS: int = 4
DEFAULT_WARMUP: int = 500
DEFAULT_SAMPLES: int = 2000
#: adapt_delta de NUTS (cible d'acceptation de l'adaptation du pas). Amont : 0,9 codé en
#: dur → 3 à 24 divergences par ajustement sur les arcs RC nets sous nonneg ; 0,99 : aucune
#: (VALIDATION_REGLAGES.md §3.2). Exige le patch 2 de drt/PROVENANCE.md (paramètre exposé,
#: défaut amont inchangé).
DEFAULT_ADAPT_DELTA: float = 0.99
#: Itérations max de L-BFGS (mode 'optimize') — défaut amont.
DEFAULT_MAX_ITER: int = 50000

#: Recherche du pic de Rct restreinte à la fenêtre de τ MESURÉE [1/(2π f_max), 1/(2π f_min)].
#: VALIDATION_REGLAGES.md §5 : sans cette restriction, des bosses hors fenêtre (non mesurées)
#: faisaient retenir le mauvais pic (Rct faux de −47 à −64 % sur les spectres de type Randles).
RCT_PEAKS_IN_MEASURED_WINDOW: bool = True

#: Nombre minimal de fréquences accepté (garde d'entrée).
MIN_POINTS: int = 10

VALID_MODES = ("sample", "optimize")


@dataclass(frozen=True)
class DRTSettings:
    """Réglages d'une inversion — enregistrés tels quels dans ``drt_diagnostics``."""

    mode: str = DEFAULT_MODE
    nonneg: bool = DEFAULT_NONNEG
    init_from_ridge: bool = DEFAULT_INIT_FROM_RIDGE
    random_seed: int = DEFAULT_RANDOM_SEED
    chains: int = DEFAULT_CHAINS
    warmup: int = DEFAULT_WARMUP
    samples: int = DEFAULT_SAMPLES
    adapt_delta: float = DEFAULT_ADAPT_DELTA
    max_iter: int = DEFAULT_MAX_ITER

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"Mode DRT inconnu : {self.mode!r} (attendu {VALID_MODES}).")
        seed = self.random_seed
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not (0 <= int(seed) < 2 ** 32):
            raise ValueError(f"random_seed doit être un entier dans [0, 2**32) (reçu {seed!r}).")
        for name in ("chains", "warmup", "samples", "max_iter"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 1:
                raise ValueError(f"{name} doit être un entier ≥ 1 (reçu {v!r}).")
        d = self.adapt_delta
        if isinstance(d, bool) or not isinstance(d, (int, float, np.floating)) or not (0.0 < float(d) < 1.0):
            raise ValueError(f"adapt_delta doit être dans ]0, 1[ (reçu {d!r}).")


# ─────────────────────────────────────────────────────────────────────────────
# Disponibilité
# ─────────────────────────────────────────────────────────────────────────────
def _import_inverter():
    """Importe ``Inverter`` à la demande (cvxopt, cmdstanpy, matplotlib requis)."""
    from drt.bayes_drt2.inversion import Inverter  # noqa: WPS433 — import différé voulu

    return Inverter


def engine_available() -> Tuple[bool, Optional[str]]:
    """(True, None) si bayes_drt2 est importable ET CmdStan installé, sinon (False, raison)."""
    try:
        _import_inverter()
    except Exception as exc:  # noqa: BLE001
        return False, f"bayes_drt2 non importable : {exc}"
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
    except Exception as exc:  # noqa: BLE001
        return False, f"CmdStan introuvable : {exc}"
    return True, None


@contextlib.contextmanager
def _quiet_cmdstanpy(level: int = logging.ERROR):
    """Abaisse le bruit du logger cmdstanpy pendant l'inversion (AUDIT.md DRT-12).

    Ce que ce logger signale (divergences, « may have failed to converge ») est de toute
    façon relu et converti en alertes par ``drt.diagnostics``.
    """
    try:  # cmdstanpy initialise SON logger (niveau DEBUG) au premier get_logger() :
        from cmdstanpy.utils import get_logger  # l'appeler d'abord, sinon notre niveau
        lg = get_logger()                       # serait écrasé pendant l'inversion.
    except Exception:  # noqa: BLE001
        lg = logging.getLogger("cmdstanpy")
    old = lg.level
    lg.setLevel(level)
    try:
        yield
    finally:
        lg.setLevel(old)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction de Rct (convention Bissessur) — τ en SECONDES (DRT-4)
# ─────────────────────────────────────────────────────────────────────────────
def _local_maxima(gamma: np.ndarray, l: int = 3, threshold: float = 0.0) -> list:
    """Indices des maxima locaux de γ dans une fenêtre glissante de demi-largeur ``l``."""
    n = len(gamma)
    out = []
    for i in range(n):
        if gamma[i] <= threshold:
            continue
        lo, hi = max(0, i - l), min(n, i + l + 1)
        if gamma[i] >= gamma[lo:hi].max():
            out.append(i)
    return out


def measured_tau_window(freq: np.ndarray) -> Tuple[float, float]:
    """Fenêtre de τ couverte par les mesures : [1/(2π·f_max), 1/(2π·f_min)] (s)."""
    freq = np.asarray(freq, dtype=float)
    return float(1.0 / (2 * np.pi * freq.max())), float(1.0 / (2 * np.pi * freq.min()))


def rct_window(tau: np.ndarray, gamma: np.ndarray,
               tau_bounds: Optional[Tuple[float, float]] = None
               ) -> Tuple[Optional[np.ndarray], Optional[int], str, str]:
    """Choisit le pic de transfert de charge et sa fenêtre d'intégration.

    Sélection identique à ``fits/drt_fit._extract_rct_peak`` (maxima au-dessus de
    1e-3·max, bords de grille écartés, pic **pénultième**), avec UNE différence
    (constante :data:`RCT_PEAKS_IN_MEASURED_WINDOW`) : si ``tau_bounds`` est fourni, seuls
    les maxima situés DANS la fenêtre mesurée sont candidats. Mesuré
    (drt/VALIDATION_REGLAGES.md §5) : γ(τ) porte de petites bosses (≈ 0,2 % du max) hors
    de la fenêtre — donc non mesurées —, et la bosse basse fréquence devenait le « dernier »
    pic : le pénultième tombait sur la diffusion (Randles : Rct_DRT ≈ 1080 Ω pour
    Rct = 3000 Ω, TOUS réglages confondus) ou sur le 2ᵉ arc (2 RC : 80 Ω au lieu de 50 Ω).

    Séparé de l'intégration pour réutiliser la MÊME fenêtre sur chaque tirage a posteriori.

    Returns:
        ``(mask, peak_idx, source, note)`` ; ``mask`` = points à ±3 en ln τ du pic, ou
        ``None`` si aucun pic (source ``'none'``).
    """
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    if n == 0:
        return None, None, "none", "DRT vide : Rct indisponible."
    ln_tau = np.log(np.maximum(tau, 1e-300))
    margin = max(1, n // 20)
    has_core = n > 2 * margin
    core = slice(margin, n - margin) if has_core else slice(0, n)
    g_core = gamma[core]
    thr = float(g_core.max()) * 1e-3 if g_core.size else 0.0
    offset = margin if has_core else 0
    maxima = [i + offset for i in _local_maxima(g_core, l=max(1, n // 15), threshold=thr)]
    if tau_bounds is not None:
        lo, hi = tau_bounds
        maxima = [i for i in maxima if lo <= tau[i] <= hi]
    if len(maxima) >= 2:
        idx, source, note = maxima[-2], "peak_penultimate", ""
    elif len(maxima) == 1:
        idx, source = maxima[0], "peak_single"
        note = ("DRT à un seul pic : Rct extrait de ce pic unique (pas de pénultième). "
                "Vérifier qu'il s'agit bien de l'arc de transfert de charge.")
    else:
        return None, None, "none", "Aucun pic DRT détecté : extraction par pic impossible."
    mask = np.abs(ln_tau - ln_tau[idx]) <= 3.0
    return mask, idx, source, note


def _integrate(tau: np.ndarray, gamma: np.ndarray, mask: np.ndarray, peak_idx: int) -> np.ndarray:
    """∫ γ dlnτ sur ``mask`` ; ``gamma`` de forme (n,) ou (tirages, n)."""
    ln_tau = np.log(np.maximum(np.asarray(tau, dtype=float), 1e-300))
    g = np.atleast_2d(gamma)
    if mask.sum() >= 2:
        x = ln_tau[mask]
        order = np.argsort(x)
        return np.trapezoid(g[:, mask][:, order], x[order], axis=1)
    return g[:, peak_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Tirages a posteriori de γ(τ) (mode 'sample')
# ─────────────────────────────────────────────────────────────────────────────
def _posterior_gamma_draws(inv, tau: np.ndarray, gamma_mean: np.ndarray) -> Optional[np.ndarray]:
    """γ(τ) pour CHAQUE tirage HMC, ou ``None`` si la reconstruction n'est pas vérifiable.

    Reproduit ``Inverter.predict_distribution`` (branche standard) tirage par tirage :
    γ = Φ·(x·Z_scale), Φ_{ij} = φ(ln(τ_i/τ_j^base), ε). Utilise des attributs internes
    (``_sample_result``, ``_rescale_coef``) : on vérifie donc que la moyenne des tirages
    redonne EXACTEMENT (à 1e-8 près) le γ public ; sinon on renonce (pas d'incertitude
    inventée).
    """
    try:
        from drt.bayes_drt2.matrices import get_basis_func

        info = inv.distributions[DIST_NAME]
        x = np.asarray(inv._sample_result["x"], dtype=float)
        coef = inv._rescale_coef(x, info["dist_type"])
        phi = get_basis_func(inv.basis_type)
        bases = np.array([phi(np.log(tau / t_m), info["epsilon"]) for t_m in info["tau"]]).T
        draws = coef @ bases.T  # (tirages, n_tau)
    except Exception:  # noqa: BLE001
        return None
    scale = max(float(np.max(np.abs(gamma_mean))), 1e-300)
    if draws.ndim != 2 or np.max(np.abs(draws.mean(axis=0) - gamma_mean)) > 1e-8 * scale:
        return None
    return draws


# ─────────────────────────────────────────────────────────────────────────────
# Inversion
# ─────────────────────────────────────────────────────────────────────────────
def _validate_spectrum(spectrum: EISSpectrum) -> Tuple[np.ndarray, np.ndarray]:
    f = np.asarray(spectrum.f, dtype=float)
    zre = np.asarray(spectrum.Zre, dtype=float)
    zim = np.asarray(spectrum.Zim, dtype=float)
    if not (f.shape == zre.shape == zim.shape) or f.ndim != 1:
        raise ValueError("f, Zre et Zim doivent être des vecteurs 1D de même longueur.")
    if f.size < MIN_POINTS:
        raise ValueError(f"Spectre trop court pour une DRT : {f.size} points < {MIN_POINTS}.")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(zre)) and np.all(np.isfinite(zim))):
        raise ValueError("Spectre contenant des valeurs non finies (NaN/inf).")
    if np.any(f <= 0):
        raise ValueError("Fréquences ≤ 0 : spectre invalide.")
    if np.unique(f).size != f.size:
        raise ValueError("Fréquences dupliquées : la DRT exige des fréquences distinctes.")
    return f, zre - 1j * zim


def fit_drt(spectrum: EISSpectrum, *, mode: str = DEFAULT_MODE, nonneg: bool = DEFAULT_NONNEG,
            init_from_ridge: bool = DEFAULT_INIT_FROM_RIDGE, random_seed: int = DEFAULT_RANDOM_SEED,
            chains: int = DEFAULT_CHAINS, warmup: int = DEFAULT_WARMUP, samples: int = DEFAULT_SAMPLES,
            adapt_delta: float = DEFAULT_ADAPT_DELTA, max_iter: int = DEFAULT_MAX_ITER,
            model_name: str = MODEL_NAME) -> FitResult:
    """DRT d'un spectre + diagnostics + gardes qualité → ``FitResult``.

    Args:
        spectrum: spectre (convention ``Zim = −Im(Z) > 0``).
        mode: ``'sample'`` (HMC, **défaut**) ou ``'optimize'`` (MAP, aperçu rapide sans
            diagnostic de convergence ni intervalle).
        nonneg, init_from_ridge: options de ``Inverter.fit`` (voir constantes ``DEFAULT_*``).
        random_seed: graine Stan, entier dans [0, 2**32) ; même graine + mêmes données ⇒
            même résultat (enregistrée dans ``drt_diagnostics['settings']``).
        chains, warmup, samples, adapt_delta: contrôle HMC (ignorés en mode ``'optimize'``).
        max_iter: itérations L-BFGS (mode ``'optimize'``).

    Returns:
        ``FitResult`` dont ``warnings`` contient toutes les alertes (convergence, qualité,
        extraction de Rct) et ``drt_diagnostics`` le détail chiffré. ``converged`` est vrai
        ssi aucune alerte de catégorie ``'convergence'``.

    Raises:
        ValueError: réglage ou spectre invalide.
        RuntimeError: moteur indisponible, ou échec de CmdStan (erreur propagée).
    """
    settings = DRTSettings(mode=mode, nonneg=bool(nonneg), init_from_ridge=bool(init_from_ridge),
                           random_seed=random_seed,
                           chains=chains, warmup=warmup, samples=samples, adapt_delta=adapt_delta,
                           max_iter=max_iter)
    freq, Z = _validate_spectrum(spectrum)
    ok, why = engine_available()
    if not ok:
        raise RuntimeError(f"Moteur DRT indisponible — {why}")
    Inverter = _import_inverter()

    order = np.argsort(freq)[::-1]          # HF → BF sur les fréquences mesurées
    inv_order = np.argsort(order)
    f_s, Z_s = freq[order], Z[order]

    kw = dict(mode=settings.mode, nonneg=settings.nonneg, init_from_ridge=settings.init_from_ridge,
              random_seed=int(settings.random_seed))
    if settings.mode == "sample":
        kw.update(chains=int(settings.chains), warmup=int(settings.warmup), samples=int(settings.samples),
                  adapt_delta=float(settings.adapt_delta))
    else:
        kw.update(max_iter=int(settings.max_iter))

    inv = Inverter()
    with _quiet_cmdstanpy(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inv.fit(f_s, Z_s, **kw)
    library_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught
                               if not issubclass(w.category, (SyntaxWarning, DeprecationWarning))})

    try:
        tau = np.asarray(inv.distributions[DIST_NAME]["tau"], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Distribution '{DIST_NAME}' absente ; clés : {list(inv.distributions)}.") from exc
    gamma = np.asarray(inv.predict_distribution(DIST_NAME, tau=tau), dtype=float)
    Rp = float(inv.predict_Rp())
    Zfit = np.asarray(inv.predict_Z(f_s))[inv_order]
    recon = dg.reconstruction_error_relative(Z, Zfit)

    gamma_lo = gamma_hi = None
    Rp_ci = (float("nan"), float("nan"))
    sampler = None
    if settings.mode == "sample":
        gamma_lo = np.asarray(inv.predict_distribution(DIST_NAME, tau=tau, percentile=2.5), dtype=float)
        gamma_hi = np.asarray(inv.predict_distribution(DIST_NAME, tau=tau, percentile=97.5), dtype=float)
        Rp_ci = (float(inv.predict_Rp(percentile=2.5)), float(inv.predict_Rp(percentile=97.5)))
        sampler = dg.sampler_diagnostics(inv.stan_mcmc)
        conv_alerts = dg.check_sampler(sampler)
    else:
        mle_ok = bool(getattr(inv.stan_mle, "converged", False))
        conv_alerts = [] if mle_ok else [dg.Alert("optimizer_not_converged", "convergence",
                                                  "L-BFGS (CmdStan) n'a pas convergé.")]
        sampler = {"optimizer_converged": mle_ok}

    quality_alerts = dg.check_quality(rp=Rp, recon=recon, gamma=gamma, tau=tau)

    # Rct (Bissessur) — τ en secondes ; incertitude a posteriori en mode 'sample'
    notes: list = []
    tau_bounds = measured_tau_window(freq)
    mask, peak_idx, rct_source, rct_note = rct_window(
        tau, gamma, tau_bounds if RCT_PEAKS_IN_MEASURED_WINDOW else None)
    Rct_std = float("nan")   # NaN = non calculée (MAP, ou tirages illisibles) — jamais 0 inventé
    Rct_ci = (float("nan"), float("nan"))
    if mask is None:
        Rct, tau_Rct, rct_source = Rp, float("nan"), "rp_fallback"
        notes.append("Rct DRT calculé par REPLI sur Rp (aire totale sous γ : diffusion + transfert "
                     "de charge). Grandeur DIFFÉRENTE de l'arc de transfert — calibration θ_EIS à "
                     "interpréter avec prudence.")
        if rct_note:
            notes.append(rct_note)
    else:
        Rct = float(_integrate(tau, gamma, mask, peak_idx)[0])
        tau_Rct = float(tau[peak_idx])
        if rct_note:
            notes.append(rct_note)
        if settings.mode == "sample":
            draws = _posterior_gamma_draws(inv, tau, gamma)
            if draws is not None:
                rct_draws = _integrate(tau, draws, mask, peak_idx)
                Rct_std = float(np.std(rct_draws, ddof=1))
                Rct_ci = (float(np.percentile(rct_draws, 2.5)), float(np.percentile(rct_draws, 97.5)))
            else:
                notes.append("Incertitude a posteriori de Rct non calculable (tirages illisibles).")

    alerts = conv_alerts + quality_alerts
    converged = not any(a.category == "convergence" for a in alerts)
    Zfit_re, Zfit_im = np.real(Zfit), -np.imag(Zfit)
    zre, zim = np.asarray(spectrum.Zre, dtype=float), np.asarray(spectrum.Zim, dtype=float)

    diagnostics = {
        "settings": asdict(settings),
        "engine": "drt.engine / bayes_drt2 99d5b60 (drt/PROVENANCE.md)",
        "stan_model": getattr(inv, "stan_model_name", None),
        "sampler": sampler,
        "reconstruction_error_relative": recon,
        "negative_area_fraction": dg.negative_area_fraction(tau, gamma),
        "Rp_ci95": list(Rp_ci),
        "Rct_ci95": list(Rct_ci),
        "rct_source": rct_source,
        "tau_window_measured": list(tau_bounds),
        "alerts": [asdict(a) for a in alerts],
        "notes": notes,
        "library_warnings": library_warnings,
        "quality_ok": not alerts,
    }
    params = {
        "Rct": float(Rct),
        "Rp": Rp,
        "tau_Rct": tau_Rct,          # SECONDES (DRT-4)
        "n_tau": int(len(tau)),
        "rct_source": rct_source,
        "drt_mode": settings.mode,
    }
    return FitResult(
        model_name=model_name,
        params=params,
        params_std={"Rct": Rct_std},
        Zfit_re=Zfit_re,
        Zfit_im=Zfit_im,
        chi2_reduced=float("nan"),   # pas de χ² pondéré pour la DRT (DRT-5)
        residuals_re=zre - Zfit_re,
        residuals_im=zim - Zfit_im,
        target_param="Rct",
        target_value=float(Rct),
        target_std=Rct_std,
        converged=converged,
        drt_tau=tau,
        drt_gamma=gamma,
        drt_mode=settings.mode,
        drt_gamma_lo=gamma_lo,
        drt_gamma_hi=gamma_hi,
        reconstruction_error=recon["rms"],
        reconstruction_error_relative=recon["max"],
        drt_diagnostics=diagnostics,
        warnings=dg.messages(alerts) + notes,
    )
