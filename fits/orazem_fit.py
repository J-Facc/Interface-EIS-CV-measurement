"""Fit « méthode Orazem » d'un circuit équivalent LIBRE (a remplacé fits/randles_full.py, supprimé).

CNLS complexe (moindres carrés non linéaires sur Re et Im empilés), pondéré par
1/σ², σ(ω) étant la structure d'erreur stochastique caractérisée par le measurement
model sur les réplicats du groupe (``core/measurement_model.py``). Le circuit vient
de ``circuit.parse_circuit(expr) -> (Z_func, param_names)`` : aucun paramètre n'est
codé en dur, aucun guess « magique » (l'ancien Re ≥ 100 Ω, Rct ≥ 500 Ω, α = 0,85…).

Ce que l'appelant (l'interface) DOIT fournir, par nom de paramètre
-----------------------------------------------------------------
``specs`` : un dictionnaire ``{nom: spécification}`` couvrant EXACTEMENT
``param_names`` (un nom manquant ou inconnu est refusé, ``FitSpecificationError``).
Chaque spécification est l'une de :

    ParameterSpec(initial=200.0, lower=0.0, upper=1e4)      # forme canonique
    ParameterSpec(initial=1e-6, lower=0.0, upper=np.inf, scale=1e-6)
    (200.0, 0.0, 1e4)                                        # tuple (initial, lower, upper)
    {"initial": 0.9, "lower": 0.5, "upper": 1.0}             # dict (clé "scale" facultative)

* ``initial`` : valeur de départ, finie, dans [lower, upper] ;
* ``lower``/``upper`` : bornes (±inf = non borné de ce côté), lower < upper ;
* ``scale`` (facultatif) : ordre de grandeur du paramètre, > 0 ; défaut |initial|,
  sinon la plus grande borne finie non nulle, sinon 1.

Les bornes par défaut « justifiables en général » (≥ 0 pour un élément passif,
0 < α ≤ 1 pour un CPE) sont décrites dans ``circuit/registry_elements.py`` ; c'est à
l'interface de les proposer, pas à ce module de les deviner.

``target_param`` : le nom (dans ``param_names``) du paramètre qui sert de signal de
calibration (ex. « Rct ») → ``FitResult.target_value``/``target_std``.

Principes conservés de l'ancien moteur (AUDIT.md §5.2)
------------------------------------------------------
* résidu imaginaire en convention de l'app : (−Im(Z_modèle) − Zim) avec
  Zim = −Im(Z) > 0 (bug B1 historique : 25-30 % de biais sur Rct) ;
* poids w = 1/σ², ``absolute_sigma`` TOUJOURS vrai : la covariance n'est jamais
  rééchelonnée par χ²_ν, qui reste un vrai test d'adéquation modèle + bruit ;
* gardes qualité I7 (généralisées ci-dessous).

Corrections (AUDIT.md §5.4)
---------------------------
* FIT-1 : plus de ``except Exception`` qui renvoie le guess. Seuls les échecs
  NUMÉRIQUES attendus (plafond d'évaluations atteint, itéré non fini) donnent un
  résultat ``converged=False`` — avec le MEILLEUR itéré atteint et une alerte
  explicite. Une spécification invalide lève ``FitSpecificationError`` ; toute
  autre exception (erreur de programmation, circuit mal compilé…) REMONTE.
* FIT-2 : covariance par SVD de la jacobienne équilibrée
  (``fits.regression_stats``) ; conditionnement rapporté ; un paramètre non
  identifiable reçoit un écart-type INFINI (et non une valeur fabriquée). Jacobienne
  finale par différences centrées, unilatérales pour un paramètre dont un côté n'est
  pas évaluable (borne singulière) ; si le circuit n'est défini d'aucun côté, ou si la
  SVD échoue, TOUS les écarts-types valent inf avec l'alerte « incertitudes
  indisponibles » — jamais d'exception, jamais de valeur inventée.
* FIT-3 : ``x_scale`` = ordre de grandeur de chaque paramètre (des pF aux GΩ) ;
  départs multiples (guess de l'utilisateur + perturbations log-normales
  reproductibles), le meilleur χ² est gardé.
* FIT-4 : « en butée » = la borne est à moins d'UN écart-type de l'estimation
  (l'intervalle de confiance à 68 % touche la borne : la contrainte est active et
  l'écart-type gaussien n'a plus de sens), et non plus « 1 % de la valeur de la
  borne », sans signification pour une borne de 1e-12 ou de 0.
* FIT-5 : une seule forme de pondération : deux tableaux (σ_r, σ_j), toujours
  fournis tous les deux (égaux quand le measurement model a retenu σ_r = σ_j).

Par réplicat, puis agrégation
-----------------------------
``fit_replicate_group`` ajuste CHAQUE réplicat (σ d'une mesure) ET leur moyenne
(σ/√n), puis agrège chaque paramètre sur les réplicats convergés : moyenne,
écart-type inter-réplicats, incertitude intra-fit, et incertitude combinée de la
moyenne (voir ``AggregatedParameter``). Il exige l'analyse du measurement model
(structure d'erreur + verdict KK) : le fit ne peut pas précéder le verdict.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from circuit import parse_circuit       # circuit/ est une feuille : fits → circuit autorisé
from fits.result import FitResult
from fits.regression_stats import JacobianStatistics, jacobian_statistics

__all__ = [
    "ORAZEM_MODEL_NAME",
    "CircuitFit",
    "compile_circuit_fit",
    "FitSpecificationError",
    "ParameterSpec",
    "FitOptions",
    "AggregatedParameter",
    "OrazemGroupResult",
    "normalize_specs",
    "fit_spectrum",
    "fit_replicate_group",
    "aggregate_parameter",
]

#: Nom de modèle (clé de ``fit_results``) des fits Orazem produits par le pipeline.
ORAZEM_MODEL_NAME = "orazem"

#: Niveau des intervalles « 2σ » (95,45 %), convention d'Orazem & Tribollet.
_LEVEL_2SIGMA = float(1.0 - 2.0 * stats.norm.sf(2.0))
#: κ(J) au-delà duquel κ² dépasse 1/ε machine : (JᵀJ)⁻¹ à la limite de la précision.
_COND_WARN = 1.0 / math.sqrt(np.finfo(float).eps)


class FitSpecificationError(ValueError):
    """Spécification de fit invalide (paramètre manquant/inconnu, bornes, guess…).

    Erreur de SAISIE, distincte d'un échec de convergence (résultat ``converged=False``)
    et d'une erreur de programmation (qui remonte telle quelle).
    """


# ═════════════════════════════════════════════════════════════════════════════
# Spécifications par paramètre
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParameterSpec:
    """Guess initial, bornes et échelle d'UN paramètre (voir l'en-tête du module)."""

    initial: float
    lower: float = -math.inf
    upper: float = math.inf
    scale: Optional[float] = None

    def resolved_scale(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        if self.initial != 0.0:
            return abs(float(self.initial))
        finite = [abs(b) for b in (self.lower, self.upper) if math.isfinite(b) and b != 0.0]
        return max(finite) if finite else 1.0


def _as_real(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise FitSpecificationError(f"{what} doit être un nombre réel (reçu {value!r}).")
    return float(value)


def _coerce_spec(name: str, raw) -> ParameterSpec:
    if isinstance(raw, ParameterSpec):
        spec = raw
    elif isinstance(raw, Mapping):
        unknown = set(raw) - {"initial", "lower", "upper", "scale"}
        if unknown or "initial" not in raw:
            raise FitSpecificationError(
                f"« {name} » : clés attendues initial (obligatoire), lower, upper, scale "
                f"(reçu {sorted(raw)})."
            )
        spec = ParameterSpec(
            initial=_as_real(raw["initial"], f"« {name} ».initial"),
            lower=_as_real(raw.get("lower", -math.inf), f"« {name} ».lower"),
            upper=_as_real(raw.get("upper", math.inf), f"« {name} ».upper"),
            scale=None if raw.get("scale") is None else _as_real(raw["scale"], f"« {name} ».scale"),
        )
    elif isinstance(raw, (tuple, list)) and len(raw) == 3:
        spec = ParameterSpec(*(_as_real(v, f"« {name} »[{i}]") for i, v in enumerate(raw)))
    else:
        raise FitSpecificationError(
            f"« {name} » : spécification attendue ParameterSpec, (initial, lower, upper) "
            f"ou dict — reçu {type(raw).__name__}."
        )
    ini, lo, hi = (_as_real(v, f"« {name} »") for v in (spec.initial, spec.lower, spec.upper))
    if not math.isfinite(ini):
        raise FitSpecificationError(f"« {name} » : le guess initial doit être fini (reçu {ini}).")
    if math.isnan(lo) or math.isnan(hi) or not lo < hi:
        raise FitSpecificationError(f"« {name} » : bornes invalides [{lo}, {hi}] (il faut lower < upper).")
    if not lo <= ini <= hi:
        raise FitSpecificationError(f"« {name} » : guess {ini:g} hors des bornes [{lo:g}, {hi:g}].")
    if spec.scale is not None and not (math.isfinite(spec.scale) and spec.scale > 0):
        raise FitSpecificationError(f"« {name} » : scale doit être fini et > 0 (reçu {spec.scale}).")
    return ParameterSpec(ini, lo, hi, spec.scale)


def normalize_specs(param_names: Sequence[str], specs: Mapping) -> dict:
    """Valide ``specs`` contre ``param_names`` → {nom: ParameterSpec} dans l'ordre des noms.

    Raises:
        FitSpecificationError: nom manquant/inconnu, spécification invalide.
    """
    if not isinstance(specs, Mapping):
        raise FitSpecificationError("specs doit être un dictionnaire {nom de paramètre: spécification}.")
    missing = [p for p in param_names if p not in specs]
    unknown = sorted(set(specs) - set(param_names))
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"sans guess/bornes : {', '.join(missing)}")
        if unknown:
            parts.append(f"inconnus du circuit : {', '.join(unknown)}")
        raise FitSpecificationError("paramètres " + " ; ".join(parts) + ".")
    return {p: _coerce_spec(p, specs[p]) for p in param_names}


# ═════════════════════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FitOptions:
    """Réglages du solveur et des gardes.

    Attributes:
        n_starts: nombre de départs (1 = guess seul). Les départs 2..n sont des
            perturbations log-normales (paramètre de signe fixé) ou additives
            (paramètre pouvant changer de signe) du guess, graine ``seed``.
        perturbation_decades: écart-type des perturbations log-normales, en décades.
        seed: graine des perturbations (reproductibilité).
        max_nfev: plafond d'évaluations par départ (``fit.max_iter`` de la config).
        tol: ftol = xtol = gtol de ``least_squares`` (1e-10, valeur de l'ancien moteur).
        rel_residual_warn: garde I7 indépendante de σ : alerte si le résidu relatif
            RMS √mean(|Z − Ẑ|²/|Z|²) dépasse ce seuil (10 %, valeur de l'ancien moteur).
        bound_sigma: « en butée » si la borne est à moins de bound_sigma écarts-types.
    """

    n_starts: int = 8
    perturbation_decades: float = 0.5
    seed: int = 0
    max_nfev: int = 10000
    tol: float = 1e-10
    rel_residual_warn: float = 0.10
    bound_sigma: float = 1.0

    @classmethod
    def from_config(cls, config: Optional[dict], **overrides) -> "FitOptions":
        fit_cfg = (config or {}).get("fit", {}) if isinstance(config, dict) else {}
        kw = {}
        if fit_cfg.get("max_iter") is not None:
            kw["max_nfev"] = int(fit_cfg["max_iter"])
        kw.update(overrides)
        return cls(**kw)


# ═════════════════════════════════════════════════════════════════════════════
# Fit d'UN spectre
# ═════════════════════════════════════════════════════════════════════════════

def _perturbed_starts(x0, lb, ub, scale, opt: FitOptions) -> list:
    rng = np.random.default_rng(opt.seed)
    starts = [x0.copy()]
    for _ in range(max(0, opt.n_starts - 1)):
        x = x0.copy()
        for i in range(len(x0)):
            fixed_sign = (lb[i] >= 0 or ub[i] <= 0) and x0[i] != 0
            if fixed_sign:
                x[i] = x0[i] * 10.0 ** (opt.perturbation_decades * rng.standard_normal())
            else:
                x[i] = x0[i] + 0.5 * scale[i] * rng.standard_normal()
        starts.append(np.clip(x, lb, ub))
    return starts


def _central_jacobian(fun, x, lb, ub, scale):
    """Jacobienne par différences finies à l'optimum, pour la covariance.

    Différences CENTRÉES (O(h²), plus précises que la jacobienne à 2 points de la
    dernière itération du solveur), h = ε^(1/3)·max(|x|, échelle), bornes respectées.

    Si un côté n'est pas évaluable — borne atteinte, ou circuit non défini, typiquement
    une capacité à sa borne 0 où 1/(jω·0) n'est pas fini — la colonne est calculée par
    différence UNILATÉRALE du côté défini (O(h)). Sans ce repli, mesuré : 6 fits sur 40
    d'un Randles à contournement négligeable (Cb poussé à ~1e-15 F, borne 0) perdaient
    TOUS leurs écarts-types, Rct compris, à cause de cette seule colonne. Une colonne
    n'est laissée non finie que si le modèle n'est défini d'aucun côté.

    Returns:
        (J, indices des paramètres dérivés par différence unilatérale).
    """
    f0 = fun(x)
    J = np.empty((f0.size, x.size))
    one_sided = []
    for i in range(x.size):
        h = np.cbrt(np.finfo(float).eps) * max(abs(x[i]), scale[i])
        up, dn = x.copy(), x.copy()
        up[i] = min(x[i] + h, ub[i])
        dn[i] = max(x[i] - h, lb[i])
        fu, fd = fun(up), fun(dn)
        ok_up = up[i] > x[i] and bool(np.all(np.isfinite(fu)))
        ok_dn = dn[i] < x[i] and bool(np.all(np.isfinite(fd)))
        if ok_up and ok_dn:
            J[:, i] = (fu - fd) / (up[i] - dn[i])
        elif ok_up:
            J[:, i] = (fu - f0) / (up[i] - x[i])
            one_sided.append(i)
        elif ok_dn:
            J[:, i] = (f0 - fd) / (x[i] - dn[i])
            one_sided.append(i)
        else:
            J[:, i] = np.nan
    return J, one_sided


def _chi2_interval(dof: int, dof_sigma: Optional[int]) -> tuple:
    """Intervalle attendu de χ²_ν sous H0 au niveau 95,45 %.

    σ connue exactement : χ²_ν ~ χ²(ν)/ν. σ ESTIMÉE sur ν_σ degrés de liberté (structure
    d'erreur du measurement model) : χ²_ν est le rapport de deux variances estimées,
    ≈ F(ν, ν_σ) — intervalle plus large, qui compte l'incertitude sur σ elle-même.
    (Les données du fit ont servi à estimer σ : le rapport est positivement corrélé,
    l'intervalle F est donc un peu conservateur.)
    """
    lo_q, hi_q = (1.0 - _LEVEL_2SIGMA) / 2.0, (1.0 + _LEVEL_2SIGMA) / 2.0
    if dof_sigma:
        return float(stats.f.ppf(lo_q, dof, dof_sigma)), float(stats.f.ppf(hi_q, dof, dof_sigma))
    return float(stats.chi2.ppf(lo_q, dof) / dof), float(stats.chi2.ppf(hi_q, dof) / dof)


def fit_spectrum(
    Z_func: Callable[..., np.ndarray],
    param_names: Sequence[str],
    f: np.ndarray,
    Zre: np.ndarray,
    Zim: np.ndarray,
    sigma_re: np.ndarray,
    sigma_im: np.ndarray,
    specs: Mapping,
    target_param: str,
    *,
    options: Optional[FitOptions] = None,
    error_structure=None,
    model_name: str = "orazem",
    label: str = "",
) -> FitResult:
    """CNLS pondéré 1/σ² d'un circuit libre sur UN spectre.

    Args:
        Z_func, param_names: sortie de ``circuit.parse_circuit`` (Z_func(w, **params)).
        f: fréquences (Hz) > 0.
        Zre, Zim: données, convention de l'app (Zim = −Im(Z) > 0).
        sigma_re, sigma_im: écart-type (Ω) de chaque donnée, par composante —
            typiquement ``error_structure.sigmas(Zre, Zim)`` divisés par √n_moyennés.
        specs: guess et bornes par nom de paramètre (voir l'en-tête).
        target_param: paramètre désigné comme signal de calibration.
        options: FitOptions.
        error_structure: structure ayant produit ``sigma`` (provenance et ν_σ de
            l'intervalle de χ²_ν) ; None si σ est connue exactement.
        model_name, label: étiquettes du résultat.

    Returns:
        FitResult. ``converged=False`` (avec alerte) si aucun départ n'a convergé :
        les paramètres sont alors ceux du meilleur itéré atteint, JAMAIS le guess
        rendu tel quel en silence.

    Raises:
        FitSpecificationError: specs/target_param invalides, ou Z non fini au guess.
        ValueError: données ou σ invalides (formes, valeurs non finies, σ ≤ 0).
    """
    opt = options or FitOptions()
    names = list(param_names)
    if target_param not in names:
        raise FitSpecificationError(
            f"paramètre cible « {target_param} » absent du circuit (paramètres : {', '.join(names)})."
        )
    sp = normalize_specs(names, specs)
    f = np.asarray(f, dtype=float)
    Zre = np.asarray(Zre, dtype=float)
    Zim = np.asarray(Zim, dtype=float)
    sigma_re = np.broadcast_to(np.asarray(sigma_re, dtype=float), f.shape).copy()
    sigma_im = np.broadcast_to(np.asarray(sigma_im, dtype=float), f.shape).copy()
    for what, arr in (("f", f), ("Zre", Zre), ("Zim", Zim), ("sigma_re", sigma_re),
                      ("sigma_im", sigma_im)):
        if arr.shape != f.shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"fit_spectrum : « {what} » doit être fini et de même forme que f.")
    if np.any(f <= 0) or np.any(sigma_re <= 0) or np.any(sigma_im <= 0):
        raise ValueError("fit_spectrum : fréquences et σ doivent être > 0.")
    n_obs = 2 * f.size
    P = len(names)
    if n_obs <= P:
        raise FitSpecificationError(f"{P} paramètres pour {n_obs} observations : fit sous-déterminé.")

    omega = 2.0 * np.pi * f
    x0 = np.array([sp[p].initial for p in names])
    lb = np.array([sp[p].lower for p in names])
    ub = np.array([sp[p].upper for p in names])
    scale = np.array([sp[p].resolved_scale() for p in names])
    bounded = bool(np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)))
    method = "trf" if bounded else "lm"

    def model(x):
        return np.asarray(Z_func(omega, **dict(zip(names, x.tolist()))), dtype=complex)

    def fun(x):
        Zm = model(x)
        # Convention de l'app : Zim = −Im(Z) → résidu imaginaire (−Im(Z_modèle) − Zim).
        return np.concatenate([(Zm.real - Zre) / sigma_re, (-Zm.imag - Zim) / sigma_im])

    if not np.all(np.isfinite(fun(x0))):
        raise FitSpecificationError(
            "Z(ω) n'est pas fini au guess initial (division par zéro, paramètre nul…) : "
            "vérifiez les valeurs de départ."
        )

    runs = []
    for k, xs in enumerate(_perturbed_starts(x0, lb, ub, scale, opt)):
        if k > 0 and not np.all(np.isfinite(fun(xs))):
            runs.append(dict(start=k, skipped=True))
            continue
        with np.errstate(all="ignore"):
            res = least_squares(
                fun, xs, bounds=(lb, ub) if bounded else (-np.inf, np.inf), method=method,
                x_scale=scale, ftol=opt.tol, xtol=opt.tol, gtol=opt.tol,
                max_nfev=opt.max_nfev,
            )
        finite = bool(np.all(np.isfinite(res.x)) and np.isfinite(res.cost))
        runs.append(dict(start=k, skipped=False, x=res.x, cost=float(res.cost) if finite else np.inf,
                         converged=bool(res.status > 0 and finite), status=int(res.status),
                         message=str(res.message), nfev=int(res.nfev),
                         active=getattr(res, "active_mask", np.zeros(P, dtype=int))))

    done = [r for r in runs if not r["skipped"] and np.isfinite(r["cost"])]
    conv = [r for r in done if r["converged"]]
    best = min(conv or done, key=lambda r: r["cost"]) if done else None
    warnings: list = []
    if best is None:                                  # aucun itéré fini : on le DIT
        best = dict(x=x0, cost=0.5 * float(np.sum(fun(x0) ** 2)), converged=False,
                    status=0, message="aucun départ n'a produit d'itéré fini", active=np.zeros(P))
    x = np.asarray(best["x"], dtype=float)
    converged = bool(best["converged"])
    if not converged:
        warnings.append(f"ajustement NON convergé ({best['message']}) : paramètres = meilleur "
                        f"itéré atteint, à ne pas exploiter tel quel")

    # ── Statistiques à l'optimum ─────────────────────────────────────────────
    with np.errstate(all="ignore"):
        J, one_sided = _central_jacobian(fun, x, lb, ub, scale)
    st, unavailable = None, None
    undefined = [names[i] for i in range(P) if not np.all(np.isfinite(J[:, i]))]
    if undefined:
        unavailable = (f"le circuit n'est défini d'aucun côté de l'optimum pour "
                       f"{', '.join(undefined)} (jacobienne non finie)")
    else:
        try:
            st = jacobian_statistics(J)                # absolute_sigma : pas de rééchelonnement
        except np.linalg.LinAlgError:                  # échec NUMÉRIQUE, pas un bug : rapporté
            unavailable = "la décomposition en valeurs singulières de la jacobienne n'a pas convergé"
    if st is None:
        # Aucun écart-type fabriqué : tout est inf, et on le DIT. Les paramètres restent
        # ceux de l'optimum (le solveur a pu converger).
        st = JacobianStatistics(
            cov=np.full((P, P), np.inf), std=np.full(P, np.inf), condition_number=float("inf"),
            rank=0, identifiable=np.zeros(P, dtype=bool), leverage=np.full(n_obs, np.nan))
        warnings.append(f"incertitudes indisponibles : {unavailable}")
    r = fun(x)
    chi2 = float(r @ r)
    dof = n_obs - P
    chi2_red = chi2 / dof
    dof_sigma = getattr(error_structure, "dof", None) if error_structure is not None else None
    ci = _chi2_interval(dof, dof_sigma)
    if not (ci[0] <= chi2_red <= ci[1]):
        warnings.append(
            f"χ²ᵣ = {chi2_red:.3g} hors de l'intervalle attendu [{ci[0]:.2f}, {ci[1]:.2f}] — "
            + ("sous-ajustement : écarts > bruit (circuit inadapté ou σ sous-estimée)."
               if chi2_red > ci[1] else "sur-ajustement apparent : σ surestimée ou données corrélées.")
        )

    Zfit = model(x)
    res_re = Zre - Zfit.real
    res_im = Zim - (-Zfit.imag)
    rel = float(np.sqrt(np.mean((res_re ** 2 + res_im ** 2) / (Zre ** 2 + Zim ** 2))))
    if rel > opt.rel_residual_warn:
        warnings.append(f"résidu relatif élevé ({rel * 100:.0f} %) — ajustement médiocre, résultat peu fiable")

    params = {p: float(v) for p, v in zip(names, x)}
    params_std = {p: float(s) for p, s in zip(names, st.std)}
    non_id = [p for p, ok in zip(names, st.identifiable) if not ok]
    if non_id and st.rank > 0:
        warnings.append(f"paramètre(s) non identifiable(s) par les données : {', '.join(non_id)} "
                        f"(jacobienne de rang {st.rank} < {P} ; écart-type infini)")
    if st.condition_number > _COND_WARN and not non_id:
        warnings.append(f"paramètres fortement corrélés (conditionnement {st.condition_number:.2g}) : "
                        f"covariance à la limite de la précision numérique")

    active = []
    for i, p in enumerate(names):
        tol_i = max(opt.bound_sigma * st.std[i] if np.isfinite(st.std[i]) else 0.0, 1e-9 * scale[i])
        if np.isfinite(lb[i]) and x[i] - lb[i] <= tol_i:
            active.append((p, "basse", lb[i]))
        elif np.isfinite(ub[i]) and ub[i] - x[i] <= tol_i:
            active.append((p, "haute", ub[i]))
    for p, side, b in active:
        warnings.append(f"{p} en butée {side} ({b:.3g}) : borne à moins d'un écart-type — "
                        f"contrainte active, écart-type non fiable")

    target_std = params_std[target_param]
    for rr in conv:
        if rr is best:
            continue
        other = dict(zip(names, rr["x"]))[target_param]
        if (2.0 * (rr["cost"] - best["cost"]) < 1.0 and np.isfinite(target_std)
                and abs(other - params[target_param]) > 2.0 * target_std):
            warnings.append(
                f"optimum non unique : un autre départ atteint un χ² équivalent (Δχ² < 1) avec "
                f"{target_param} = {other:.4g} (retenu : {params[target_param]:.4g})"
            )
            break

    es = error_structure
    return FitResult(
        model_name=model_name,
        params=params,
        params_std=params_std,
        Zfit_re=Zfit.real,
        Zfit_im=-Zfit.imag,
        chi2_reduced=chi2_red,
        residuals_re=res_re,
        residuals_im=res_im,
        target_param=target_param,
        target_value=params[target_param],
        target_std=target_std,
        converged=converged,
        reconstruction_error=rel,
        warnings=warnings,
        error_structure_source="characterized_now" if es is not None else None,
        error_structure_timestamp=getattr(es, "characterized_at", None) if es is not None else None,
        error_structure_coeffs=es.to_dict() if es is not None and hasattr(es, "to_dict") else None,
        chi2_reduced_ci=ci,
        fit_diagnostics=dict(
            label=label, method=method, chi2=chi2, dof=dof, dof_sigma=dof_sigma,
            condition_number=st.condition_number, rank=st.rank,
            identifiable=dict(zip(names, st.identifiable.tolist())),
            x_scale=dict(zip(names, scale.tolist())), active_bounds=active,
            jacobian_one_sided=[names[i] for i in one_sided],
            n_starts=len(runs), n_converged=len(conv),
            starts=[dict(start=rr["start"], skipped=rr["skipped"],
                         chi2=(2.0 * rr["cost"]) if not rr["skipped"] else None,
                         converged=rr.get("converged"), status=rr.get("status"))
                    for rr in runs],
            solver_message=best["message"],
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Groupe de réplicats : un fit par réplicat + moyenne, puis agrégation
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AggregatedParameter:
    """Un paramètre agrégé sur les réplicats convergés d'un groupe.

    Modèle : θ̂_k = θ + b_k + e_k — b_k variabilité propre au réplicat (dérive,
    remontage…, variance τ²), e_k erreur de fit due au bruit (variance v_k = σ_k²
    intra-fit, connue). Alors E[s²_inter] = τ² + v̄ et Var(θ̄) = (τ² + v̄)/n.

    Attributes:
        mean: moyenne arithmétique θ̄ des réplicats.
        std_between: écart-type inter-réplicats s (ddof = 1) — contient bruit ET τ.
        std_within: écart-type intra-fit typique √v̄ (d'UN réplicat).
        sem_within: √(Σ v_k)/n — incertitude de θ̄ si seul le bruit comptait.
        sem: incertitude-type de θ̄, √(max(s², v̄)/n) : estimateur des moments de τ²
            tronqué à 0, τ̂² = max(0, s² − v̄) (DerSimonian & Laird, Control. Clin.
            Trials 7 (1986) 177-188). Jamais sous le plancher imposé par le bruit
            (un s² petit par hasard à n = 3), mais la dispersion observée prime dès
            qu'elle dépasse ce que le bruit explique.
        q, q_pvalue: statistique d'hétérogénéité de Cochran (Biometrics 10 (1954)
            101-129), Q = Σ (θ̂_k − θ̄_w)²/v_k ~ χ²(n − 1) sous τ = 0 ; p < 0,05 →
            les réplicats diffèrent plus que leur incertitude intra-fit ne l'explique.
        n: réplicats convergés utilisés ; n_excluded : non convergés écartés.
    """

    name: str
    n: int
    mean: float
    std_between: float
    std_within: float
    sem_within: float
    sem: float
    q: float
    q_pvalue: float
    values: list
    stds: list
    n_excluded: int = 0


def aggregate_parameter(name: str, values: Sequence[float], stds: Sequence[float],
                        n_excluded: int = 0) -> AggregatedParameter:
    """Agrège un paramètre (voir ``AggregatedParameter``)."""
    v = np.asarray(values, dtype=float)
    s = np.asarray(stds, dtype=float)
    n = int(v.size)
    nan = float("nan")
    if n == 0:
        return AggregatedParameter(name, 0, nan, nan, nan, nan, nan, nan, nan, [], [], n_excluded)
    mean = float(np.mean(v))
    var_w = s ** 2
    finite_w = bool(np.all(np.isfinite(var_w)) and np.all(var_w > 0))
    std_within = float(np.sqrt(np.mean(var_w))) if finite_w else float("inf")
    sem_within = float(np.sqrt(np.sum(var_w)) / n) if finite_w else float("inf")
    if n < 2:
        return AggregatedParameter(name, n, mean, nan, std_within, sem_within, sem_within,
                                   nan, nan, v.tolist(), s.tolist(), n_excluded)
    sb = float(np.std(v, ddof=1))
    sem = float(np.sqrt(max(sb ** 2, float(np.mean(var_w))) / n)) if finite_w else float("inf")
    if finite_w:
        wts = 1.0 / var_w
        mw = float(np.sum(wts * v) / np.sum(wts))
        q = float(np.sum(wts * (v - mw) ** 2))
        qp = float(stats.chi2.sf(q, n - 1))
    else:
        q = qp = nan
    return AggregatedParameter(name, n, mean, sb, std_within, sem_within, sem, q, qp,
                               v.tolist(), s.tolist(), n_excluded)


@dataclass
class OrazemGroupResult:
    """Résultat du fit Orazem d'un groupe de réplicats.

    Attributes:
        label, target_param, param_names: identification.
        analysis: MeasurementModelAnalysis (structure d'erreur + verdict KK) — affiché
            AVANT ce résultat.
        mean_fit: fit du spectre MOYEN (σ/√n) — pour l'affichage et le contrôle.
        replicate_fits: un FitResult par réplicat (σ d'une mesure).
        aggregate: {paramètre: AggregatedParameter} sur les réplicats convergés.
        target: agrégat du paramètre cible — valeur de calibration recommandée
            (``target.mean`` ± ``target.sem``) : elle porte la dispersion réelle entre
            réplicats, que l'incertitude intra-fit du spectre moyen ignore.
        warnings: alertes du groupe (verdict KK, réplicats écartés, hétérogénéité…).
    """

    label: str
    target_param: str
    param_names: list
    analysis: object
    mean_fit: FitResult
    replicate_fits: list
    aggregate: dict
    target: AggregatedParameter
    warnings: list = field(default_factory=list)


def fit_replicate_group(
    Z_func: Callable[..., np.ndarray],
    param_names: Sequence[str],
    replicates: Sequence,
    analysis,
    specs: Mapping,
    target_param: str,
    *,
    options: Optional[FitOptions] = None,
    model_name: str = "orazem",
) -> OrazemGroupResult:
    """Fit Orazem de chaque réplicat ET du spectre moyen, puis agrégation.

    Args:
        Z_func, param_names: sortie de ``circuit.parse_circuit``.
        replicates: les MÊMES réplicats que ceux analysés par ``analysis`` (même ordre).
        analysis: ``core.measurement_model.MeasurementModelAnalysis`` de ce groupe.
            Obligatoire : c'est lui qui porte σ (structure d'erreur) et le verdict KK,
            calculés et affichés AVANT le fit.
        specs, target_param, options, model_name: voir ``fit_spectrum``.

    Returns:
        OrazemGroupResult.

    Raises:
        FitSpecificationError: specs invalides.
        ValueError: réplicats ne correspondant pas à l'analyse.
    """
    reps = list(replicates)
    labels = [getattr(sp, "label", "") for sp in reps]
    if len(reps) != analysis.n_replicates or labels != list(analysis.replicate_labels):
        raise ValueError("fit_replicate_group : les réplicats ne sont pas ceux de l'analyse du "
                         "measurement model fournie (nombre ou ordre différents).")
    es = analysis.error_structure
    group_warnings = []
    if not analysis.kk_conform:
        group_warnings.append(analysis.kk_message)

    rep_fits = []
    for k, sp in enumerate(reps):
        zr, zj = np.asarray(sp.Zre, float), np.asarray(sp.Zim, float)
        s_re, s_im = es.sigmas(zr, zj)
        fr = fit_spectrum(Z_func, param_names, sp.f, zr, zj, s_re, s_im, specs, target_param,
                          options=options, error_structure=es, model_name=model_name,
                          label=labels[k])
        kk = analysis.kk_replicates[k]
        if kk.conform is False:
            fr.warnings.append(f"réplicat non conforme Kramers-Kronig : {kk.verdict.message}")
        rep_fits.append(fr)

    n = len(reps)
    zr_m, zj_m = analysis.mean_Zre, analysis.mean_Zim
    s_re, s_im = es.sigmas(zr_m, zj_m)
    mean_fit = fit_spectrum(Z_func, param_names, analysis.frequencies, zr_m, zj_m,
                            s_re / np.sqrt(n), s_im / np.sqrt(n), specs, target_param,
                            options=options, error_structure=es, model_name=model_name,
                            label=f"{analysis.label} (moyenne de {n})")
    if analysis.kk_mean.conform is False:
        mean_fit.warnings.append(f"spectre moyen non conforme Kramers-Kronig : "
                                 f"{analysis.kk_mean.verdict.message}")

    used = [fr for fr in rep_fits if fr.converged]
    excluded = n - len(used)
    if excluded:
        group_warnings.append(f"{excluded} réplicat(s) non convergé(s) écarté(s) de l'agrégation.")
    aggregate = {
        p: aggregate_parameter(p, [fr.params[p] for fr in used], [fr.params_std[p] for fr in used],
                               n_excluded=excluded)
        for p in param_names
    }
    target = aggregate[target_param]
    if target.n < 2:
        group_warnings.append(f"moins de 2 réplicats convergés : pas d'écart-type inter-réplicats "
                              f"pour {target_param}.")
    elif np.isfinite(target.q_pvalue) and target.q_pvalue < 0.05:
        group_warnings.append(
            f"{target_param} : dispersion inter-réplicats (s = {target.std_between:.3g}) supérieure à "
            f"l'incertitude intra-fit (√v̄ = {target.std_within:.3g} ; Q de Cochran p = "
            f"{target.q_pvalue:.2g}) — dérive ou non-stationnarité probable ; l'incertitude "
            f"retenue est la dispersion observée."
        )
    return OrazemGroupResult(
        label=analysis.label, target_param=target_param, param_names=list(param_names),
        analysis=analysis, mean_fit=mean_fit, replicate_fits=rep_fits,
        aggregate=aggregate, target=target, warnings=group_warnings,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Circuit utilisateur compilé (entrée du pipeline)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CircuitFit:
    """Circuit utilisateur compilé + spécifications VALIDÉES, prêt pour ``fit_*``.

    Construit par :func:`compile_circuit_fit` AVANT tout calcul : une saisie invalide
    (expression, guess/bornes, paramètre cible) est refusée d'emblée, pas découverte
    au milieu d'une analyse.

    Attributes:
        expression: texte du circuit (``circuit.parse_circuit``).
        Z_func, param_names: circuit compilé.
        specs: {nom: ParameterSpec} validés, dans l'ordre de ``param_names``.
        target_param: paramètre désigné comme signal de calibration.
        model_name: clé de ``fit_results`` (``"orazem"`` par défaut).
    """

    expression: str
    Z_func: Callable[..., np.ndarray]
    param_names: tuple
    specs: Mapping
    target_param: str
    model_name: str = ORAZEM_MODEL_NAME

    def to_dict(self) -> dict:
        """Description sérialisable (exports, rappel dans l'UI)."""
        return {
            "expression": self.expression,
            "target_param": self.target_param,
            "parameters": {p: {"initial": s.initial, "lower": s.lower, "upper": s.upper,
                               "scale": s.scale} for p, s in self.specs.items()},
        }


def compile_circuit_fit(expression: str, parameters: Mapping, target_param: str,
                        *, model_name: str = ORAZEM_MODEL_NAME) -> CircuitFit:
    """Compile l'expression et valide les spécifications contre ses paramètres.

    Args:
        expression: circuit (docs/CIRCUIT_UTILISATEUR.md).
        parameters: {nom: spécification} (voir l'en-tête du module) couvrant
            EXACTEMENT les paramètres du circuit.
        target_param: nom du paramètre servant de signal de calibration.

    Raises:
        circuit.CircuitError: expression invalide ou refusée (liste blanche).
        FitSpecificationError: guess/bornes invalides, paramètre cible absent.
    """
    Z_func, names = parse_circuit(expression)
    if target_param not in names:
        raise FitSpecificationError(
            f"paramètre cible « {target_param} » absent du circuit (paramètres : {', '.join(names)})."
        )
    specs = normalize_specs(names, parameters)
    return CircuitFit(expression=str(expression), Z_func=Z_func, param_names=tuple(names),
                      specs=specs, target_param=str(target_param), model_name=model_name)
