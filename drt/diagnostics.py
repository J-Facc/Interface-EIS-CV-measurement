# -*- coding: utf-8 -*-
"""drt/diagnostics.py — gardes qualité réutilisables de la DRT bayésienne.

Séparé de ``drt/engine.py`` pour être testable sans CmdStan : toutes les fonctions
``check_*`` et les métriques sont **pures** (tableaux numpy / dict en entrée, liste
d'alertes en sortie). Seule :func:`sampler_diagnostics` lit un objet cmdstanpy.

Chaque seuil est une constante de module **documentée** — jamais un nombre en dur dans
le code — et justifiée par ``drt/VALIDATION_REGLAGES.md`` (mesures) ou par une référence.

Références
----------
* Modèle DRT hiérarchique bayésien (bayes_drt2, ``drt/bayes_drt2``) :
  J. Huang, M. Papac, R. O'Hayre (2020), « Towards robust autonomous impedance
  spectroscopy analysis: a calibrated hierarchical Bayesian approach for
  electrochemical impedance spectroscopy (EIS) inversion », *Electrochimica Acta*
  367, 137493. https://doi.org/10.1016/j.electacta.2020.137493
* R-hat rang-normalisé (split) et ESS bulk/tail — ceux que calcule ``stansummary``
  de CmdStan 2.36 et que lit :func:`sampler_diagnostics` :
  A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Bürkner (2021),
  « Rank-normalization, folding, and localization: an improved R̂ for assessing
  convergence of MCMC », *Bayesian Analysis* 16(2), 667–718.
  https://doi.org/10.1214/20-BA1221 — ESS bulk et tail ≥ ~100 **par chaîne**
  (recommandation reprise par la documentation Stan ``posterior::ess_bulk``).
* Transitions divergentes et profondeur d'arbre maximale (NUTS) :
  M. D. Hoffman, A. Gelman (2014), « The No-U-Turn Sampler », *JMLR* 15, 1593–1623 ;
  M. Betancourt (2017), « A Conceptual Introduction to Hamiltonian Monte Carlo »,
  arXiv:1701.02434.
* E-BFMI : M. Betancourt (2016), « Diagnosing Suboptimal Cotangent Disintegrations
  in Hamiltonian Monte Carlo », arXiv:1604.00695 (seuil 0,3 = celui de l'utilitaire
  ``diagnose`` de CmdStan).

Vocabulaire
-----------
Une **alerte** est un :class:`Alert` (code stable + message lisible). Les alertes de
catégorie ``"convergence"`` disent que l'algorithme (HMC ou optimiseur) n'a pas
convergé de façon démontrable ; celles de catégorie ``"qualite"`` disent que le
résultat est physiquement ou numériquement suspect même si l'algorithme a convergé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Seuils — convergence HMC
# ─────────────────────────────────────────────────────────────────────────────
#: R-hat maximal toléré (rang-normalisé, split). 1,05 = décision utilisateur ; Vehtari
#: et al. (2021) recommandent 1,01 — plus strict, à envisager avec plus de tirages.
RHAT_MAX: float = 1.05
#: Nombre de transitions divergentes post-warmup toléré : AUCUNE (une divergence signale
#: une région du postérieur que HMC n'explore pas — estimations potentiellement biaisées).
MAX_DIVERGENCES: int = 0
#: ESS (bulk ET tail) minimal PAR CHAÎNE (Vehtari et al. 2021) ; le seuil total vaut
#: ``ESS_MIN_PER_CHAIN × nombre de chaînes``.
ESS_MIN_PER_CHAIN: float = 100.0
#: E-BFMI minimal par chaîne (Betancourt 2016 ; seuil de ``diagnose`` de CmdStan).
EBFMI_MIN: float = 0.3
# Saturation de la profondeur d'arbre NUTS : RAPPORTÉE (``max_treedepth_hits``) mais PAS
# une alerte. Stan la classe comme un problème d'EFFICACITÉ, non de validité (Stan
# Development Team, « Runtime warnings and convergence problems », § Maximum treedepth),
# et ses conséquences — trajectoires tronquées, exploration lente — sont précisément ce que
# mesurent R-hat et l'ESS, qui sont gardés. Mesuré (VALIDATION_REGLAGES.md §3) : chacun
# des 32 essais saturés des §3.1-3.2 portait AUSSI une alerte R-hat ou ESS ; inversement,
# un ajustement saturé à 3996/4000 itérations peut n'avoir aucune alerte (R-hat 1,015,
# ESS 482/406, 0 divergence — §3.3) : la saturation seule ne dit rien de la validité.

# ─────────────────────────────────────────────────────────────────────────────
# Seuils — qualité du résultat (valables pour 'sample' ET 'optimize')
# ─────────────────────────────────────────────────────────────────────────────
#: Erreur de reconstruction relative MAXIMALE tolérée, max_i |Z_fit − Z|/|Z|.
#: Mesuré (drt/VALIDATION_REGLAGES.md, bruit 0,5 %) : ajustements corrects ≤ 4,0 % (moyenne
#: a posteriori sous nonneg, Randles), optima dégénérés ≥ 360 %. 10 % : ×2,5 au-dessus du
#: pire cas correct, ×36 sous l'échec. Non calé sur des bruits plus forts (limite connue).
RECONSTRUCTION_MAX_REL_MAX: float = 0.10
#: Fraction maximale de l'aire |γ| portée par des valeurs NÉGATIVES (DRT non contrainte
#: seulement ; nulle par construction avec ``nonneg=True``). Mesuré (VALIDATION_REGLAGES.md) :
#: solutions non contraintes correctes 0,2–4,2 % (ondulations), optimum dégénéré 98,9 %.
#: 25 % : ×6 au-dessus du pire cas correct, ×4 sous l'échec.
NEGATIVE_AREA_FRACTION_MAX: float = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Alertes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Alert:
    """Alerte qualité : ``code`` stable (tests, exports), ``category``, ``message`` lisible."""

    code: str
    category: str  # "convergence" | "qualite"
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Métriques pures
# ─────────────────────────────────────────────────────────────────────────────
def reconstruction_error_relative(Z: np.ndarray, Z_fit: np.ndarray) -> dict:
    """Erreur de reconstruction relative point par point, ``|Z_fit − Z| / |Z|``.

    Ce n'est **pas** un χ² : aucune pondération par une variance (correction de
    AUDIT.md DRT-5, où cette quantité était rangée dans ``chi2_reduced``).

    Returns:
        ``{"max", "mean", "rms"}`` ; ``rms = sqrt(mean(|ΔZ|²/|Z|²))`` est la formule du
        ``reconstruction_error`` de ``fits/randles_full.py`` (comparaison homogène).
    """
    Z = np.asarray(Z, dtype=complex)
    Z_fit = np.asarray(Z_fit, dtype=complex)
    if Z.shape != Z_fit.shape or Z.size == 0:
        raise ValueError("Z et Z_fit doivent être non vides et de même forme.")
    rel = np.abs(Z_fit - Z) / np.maximum(np.abs(Z), 1e-300)
    return {
        "max": float(np.max(rel)),
        "mean": float(np.mean(rel)),
        "rms": float(np.sqrt(np.mean(rel ** 2))),
    }


def negative_area_fraction(tau: np.ndarray, gamma: np.ndarray) -> float:
    """Part de l'aire ∫|γ| dlnτ portée par γ < 0 (0 si γ ≥ 0 partout).

    Une DRT est une densité de résistances ≥ 0 pour un système passif ; une aire
    négative importante trahit un optimum dégénéré (AUDIT.md DRT-1 : γ_min = −250 Ω
    pour un Rp vrai de 130 Ω).
    """
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    order = np.argsort(tau)
    x = np.log(tau[order])
    g = gamma[order]
    total = float(np.trapezoid(np.abs(g), x))
    if total <= 0.0:
        return 0.0
    return float(np.trapezoid(np.clip(-g, 0.0, None), x)) / total


def ebfmi(energy: np.ndarray) -> np.ndarray:
    """E-BFMI par chaîne (Betancourt 2016) : Σ(E_n − E_{n−1})² / Σ(E_n − Ē)².

    Args:
        energy: ``energy__`` de Stan, forme (tirages, chaînes) ou (tirages,).
    """
    e = np.asarray(energy, dtype=float)
    if e.ndim == 1:
        e = e[:, None]
    e = e.reshape(e.shape[0], -1)
    num = np.sum(np.diff(e, axis=0) ** 2, axis=0)
    den = np.sum((e - e.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des diagnostics HMC (API cmdstanpy 1.3, CmdStan 2.36)
# ─────────────────────────────────────────────────────────────────────────────
def sampler_diagnostics(fit) -> dict:
    """Diagnostics d'un ``cmdstanpy.CmdStanMCMC`` — valeurs LUES, rien d'estimé ici.

    Sources (API documentée de cmdstanpy 1.3) :

    * ``fit.divergences`` / ``fit.max_treedepths`` : comptes post-warmup par chaîne ;
    * ``fit.summary()`` : sortie de ``bin/stansummary`` de CmdStan, colonnes
      ``R_hat`` (rang-normalisé, split), ``ESS_bulk``, ``ESS_tail`` ; une ligne par
      paramètre, paramètre transformé, quantité générée, plus ``lp__`` ;
    * ``fit.method_variables()['energy__']`` → :func:`ebfmi`.

    Portée : R-hat et ESS sont pris sur **toutes** les lignes du résumé (comme
    l'utilitaire ``diagnose`` de CmdStan) ; le détail par variable est gardé dans
    ``per_variable`` pour savoir laquelle est en cause.
    """
    summ = fit.summary()
    rhat = summ["R_hat"].to_numpy(dtype=float)
    ess_b = summ["ESS_bulk"].to_numpy(dtype=float)
    ess_t = summ["ESS_tail"].to_numpy(dtype=float)
    base = np.array([str(i).split("[")[0] for i in summ.index])
    per_var = {}
    for name in dict.fromkeys(base):  # ordre de déclaration Stan
        m = base == name
        per_var[str(name)] = {
            "rhat_max": _nanmax(rhat[m]),
            "ess_bulk_min": _nanmin(ess_b[m]),
            "ess_tail_min": _nanmin(ess_t[m]),
        }
    energy = fit.method_variables()["energy__"]
    div = np.asarray(fit.divergences, dtype=int)
    mtd = np.asarray(fit.max_treedepths, dtype=int)
    worst_rhat = str(summ.index[int(np.nanargmax(rhat))]) if np.any(np.isfinite(rhat)) else None
    return {
        "chains": int(fit.chains),
        "draws_per_chain": int(fit.num_draws_sampling),
        "divergences": int(div.sum()),
        "divergences_per_chain": div.tolist(),
        "max_treedepth_hits": int(mtd.sum()),
        "rhat_max": _nanmax(rhat),
        "rhat_max_variable": worst_rhat,
        "rhat_nan_count": int(np.isnan(rhat).sum()),
        "ess_bulk_min": _nanmin(ess_b),
        "ess_tail_min": _nanmin(ess_t),
        "ebfmi_per_chain": [float(v) for v in ebfmi(np.asarray(energy, dtype=float))],
        "per_variable": per_var,
    }


def _nanmax(a: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.any(np.isfinite(a)) else None


def _nanmin(a: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    return float(np.nanmin(a)) if np.any(np.isfinite(a)) else None


# ─────────────────────────────────────────────────────────────────────────────
# Gardes
# ─────────────────────────────────────────────────────────────────────────────
def check_sampler(diag: dict, *, rhat_max: float = RHAT_MAX, max_divergences: int = MAX_DIVERGENCES,
                  ess_min_per_chain: float = ESS_MIN_PER_CHAIN, ebfmi_min: float = EBFMI_MIN) -> list:
    """Alertes de convergence HMC à partir de :func:`sampler_diagnostics`.

    Une valeur manquante (``None``) est elle-même une alerte : l'absence de diagnostic
    n'est jamais interprétée comme un succès.
    """
    alerts: list = []
    chains = int(diag.get("chains") or 0)

    rh = diag.get("rhat_max")
    if rh is None or not np.isfinite(rh):
        alerts.append(Alert("rhat_unavailable", "convergence",
                            "R-hat indisponible : convergence HMC non démontrée."))
    elif rh > rhat_max:
        alerts.append(Alert("rhat_high", "convergence",
                            f"R-hat max = {rh:.3f} > {rhat_max} ({diag.get('rhat_max_variable')}) : "
                            "chaînes HMC non mélangées, estimations non fiables."))

    nd = diag.get("divergences")
    if nd is None:
        alerts.append(Alert("divergences_unavailable", "convergence",
                            "Nombre de divergences indisponible."))
    elif nd > max_divergences:
        alerts.append(Alert("divergences", "convergence",
                            f"{nd} transition(s) divergente(s) après warmup : une région du "
                            "postérieur n'est pas explorée, biais possible."))

    need = ess_min_per_chain * max(chains, 1)
    for key, label in (("ess_bulk_min", "bulk"), ("ess_tail_min", "tail")):
        v = diag.get(key)
        if v is None or not np.isfinite(v):
            alerts.append(Alert(f"ess_{label}_unavailable", "convergence", f"ESS {label} indisponible."))
        elif v < need:
            alerts.append(Alert(f"ess_{label}_low", "convergence",
                                f"ESS {label} min = {v:.0f} < {need:.0f} ({ess_min_per_chain:.0f} × "
                                f"{chains} chaînes) : trop peu de tirages indépendants."))

    ebf = diag.get("ebfmi_per_chain") or []
    low = [v for v in ebf if not np.isfinite(v) or v < ebfmi_min]
    if not ebf or low:
        alerts.append(Alert("ebfmi_low", "convergence",
                            f"E-BFMI < {ebfmi_min} sur {len(low)} chaîne(s) : exploration "
                            "inefficace des niveaux d'énergie."))
    return alerts


def check_quality(*, rp: float, recon: dict, gamma: Optional[np.ndarray] = None,
                  tau: Optional[np.ndarray] = None,
                  reconstruction_max_rel_max: float = RECONSTRUCTION_MAX_REL_MAX,
                  negative_area_fraction_max: float = NEGATIVE_AREA_FRACTION_MAX) -> list:
    """Alertes de qualité du résultat (indépendantes de l'algorithme).

    * Rp ≤ 0 ou non fini : physiquement impossible pour un système passif
      (AUDIT.md DRT-1 : Rp = −159 Ω, −5678 Ω… acceptés en silence).
    * erreur de reconstruction max > seuil : la DRT ne reproduit pas le spectre.
    * aire négative de γ > seuil (si γ fourni) : optimum dégénéré.
    """
    alerts: list = []
    if rp is None or not np.isfinite(rp) or rp <= 0:
        alerts.append(Alert("rp_nonpositive", "qualite",
                            f"Rp = {rp} Ω ≤ 0 : résultat physiquement impossible (optimum dégénéré)."))
    rmax = recon.get("max") if recon else None
    if rmax is None or not np.isfinite(rmax):
        alerts.append(Alert("reconstruction_unavailable", "qualite",
                            "Erreur de reconstruction indisponible."))
    elif rmax > reconstruction_max_rel_max:
        alerts.append(Alert("reconstruction_error", "qualite",
                            f"Erreur de reconstruction max = {rmax * 100:.1f} % > "
                            f"{reconstruction_max_rel_max * 100:.0f} % : la DRT ne reproduit pas le spectre."))
    if gamma is not None and tau is not None:
        nf = negative_area_fraction(tau, gamma)
        if nf > negative_area_fraction_max:
            alerts.append(Alert("gamma_negative", "qualite",
                                f"{nf * 100:.0f} % de l'aire de γ(τ) est négative (> "
                                f"{negative_area_fraction_max * 100:.0f} %) : DRT non physique."))
    return alerts


def messages(alerts: Iterable[Alert]) -> list:
    """Messages lisibles (pour ``FitResult.warnings``)."""
    return [a.message for a in alerts]


def codes(alerts: Sequence[Alert]) -> list:
    return [a.code for a in alerts]
