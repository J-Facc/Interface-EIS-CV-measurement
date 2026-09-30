# -*- coding: utf-8 -*-
"""
fits/kk_validation.py — Validation Kramers-Kronig : Lin-KK et critère UNIQUE de verdict.

Deux choses, et deux seulement, vivent ici :

1. ``lin_kk`` — le test Lin-KK de Boukamp (1995) avec le choix automatique du
   nombre d'éléments M de Schönleber et al. (2014) ;
2. ``kk_verdict`` — LE critère de décision « conforme / non conforme KK », seul
   critère du dépôt : Lin-KK (``kramers_kronig_check``, ``core/validator.py``)
   et le test par measurement model (``core/measurement_model.check_kk_consistency``)
   passent tous par lui.

Références
----------
[B95]  B. A. Boukamp, « A Linear Kronig-Kramers Transform Test for Immittance Data
       Validation », J. Electrochem. Soc. 142 (1995) 1885-1894.
[S14]  M. Schönleber, D. Klotz, E. Ivers-Tiffée, « A Method for Improving the
       Robustness of linear Kramers-Kronig Validity Tests », Electrochim. Acta 131
       (2014) 20-27.
[A95]  P. Agarwal, M. E. Orazem, L. H. García-Rubio, « Application of Measurement
       Models to Impedance Spectroscopy: III. Evaluation of Consistency with the
       Kramers-Kronig Relations », J. Electrochem. Soc. 142 (1995) 4159-4168.
[OT]   M. E. Orazem, B. Tribollet, *Electrochemical Impedance Spectroscopy*, 2e éd.,
       Wiley, 2017 — chapitres « Error Structure » et « Kramers-Kronig Relations ».
⚠ Les numéros d'équation de ces sources n'ont pas pu être vérifiés depuis
l'environnement où ce module a été écrit ; les formules implémentées sont rappelées
explicitement ci-dessous.

Lin-KK ([B95], [S14])
---------------------
Circuit de Voigt LINÉAIRE à constantes de temps FIXÉES :

    Ẑ(ω) = R0 + Σ_{k=1..M} R_k / (1 + jωτ_k)  [+ 1/(jωC) si add_cap]

τ_1 = 1/ω_max, τ_M = 1/ω_min, log-espacés ([S14]). Les R_k (et R0, 1/C) sont
obtenus par moindres carrés linéaires sur les parties réelle et imaginaire
empilées, chaque ligne pondérée par 1/|Z(ω_i)| (pondération « modulus » de [B95],
reprise par [S14]).

Choix de M ([S14]) : critère µ, avec c = 0,85,

    µ(M) = 1 − Σ_{R_k<0} |R_k| / Σ_{R_k≥0} |R_k|.

µ ≈ 1 tant que tous les R_k sont positifs (sous-ajustement) ; l'apparition de R_k
négatifs oscillants signe le sur-ajustement du bruit, et [S14] retient le M où µ
passe sous c. CORRECTIONS (AUDIT.md ERR-5) : l'ancien code fixait
M = min(100, 2n/3) et ignorait ``c`` (wrapper ``linKK``, supprimé), et calculait µ
comme « masse négative / masse totale » — une AUTRE quantité (≈ 0 sans R_k négatif)
que le validateur comparait pourtant au seuil 0,85 de [S14] : ce contrôle ne pouvait
pratiquement jamais se déclencher.

⚠ Lecture retenue : DERNIER franchissement, pas le premier. Mesuré sur les
spectres de cette application (grille 0,1 Hz – 100 kHz), µ(M) n'est PAS monotone :
une relaxation fine tombant ENTRE deux τ de la grille exige transitoirement des R_k
négatifs, µ plonge sous c puis remonte quand la grille s'affine. Le premier
franchissement arrête alors l'ajout bien trop tôt (bruit relatif 0,5 %) :

    spectre (40 pts)             1er franchissement        dernier franchissement
    Randles de l'app (CPE 0,9)   M = 9,  RMS 3,6 %         M = 26, RMS 0,30 %
    R // CPE (α = 0,8)           M = 4,  RMS 28 %          M = 23, RMS 0,42 %
    R // C idéal                 M = 4,  RMS 13 %          M = 13, RMS 1,5 %

Un résidu de 3,6 à 28 % sur des données PARFAITEMENT conformes donnerait un faux
verdict « non conforme ». Or l'argument de [S14] vise le sur-ajustement du bruit,
qui, lui, est DURABLE (une fois le bruit suivi, ajouter des éléments ne le « défait »
pas). On retient donc le début de la dernière plage où µ < c :
M* = 1 + max{M ≤ M_max : µ(M) ≥ c}, ce qui impose de parcourir tous les M jusqu'à
M_max = min(nb de fréquences, 100). Limite qui subsiste (dernière ligne) : une
relaxation de Debye idéale reste mal décrite par une grille de τ FIXÉS au niveau de
bruit où le sur-ajustement commence — c'est l'une des raisons pour lesquelles le
verdict de référence est celui du measurement model RÉGRESSÉ (τ libres,
``core/measurement_model.check_kk_consistency``), Lin-KK restant un test rapide.

Critère UNIQUE de verdict (AUDIT.md ERR-6)
------------------------------------------
Avant : deux critères incompatibles coexistaient — résidu relatif < 2 % par point
avec paliers 10 %/25 % (``core/validator.py``) et résidu relatif max < 5 %
(``fit.drt_kk_tol``), avec deux verdicts. Aucun de ces seuils n'avait de source.

[S14] ne donne AUCUN seuil numérique universel sur les résidus : pour des données
valides, les résidus doivent être « de l'ordre du bruit » et répartis au hasard
autour de zéro ; une tendance signe une violation (dérive, non-linéarité). « De
l'ordre du bruit » n'est décidable qu'avec le niveau de bruit : c'est la structure
d'erreur stochastique σ(ω) de l'instrument, estimée par le measurement model
([A95], ``core/measurement_model.py``). Le critère rend donc explicite la phrase
de [S14] en la confrontant à σ, à la manière de [A95] :

  * résidu normalisé  z_i = r_i / s_i, où s_i est l'écart-type du résidu sous H0
    (bruit seul : s_i = σ_i ; avec une prédiction de modèle incertaine : [A95]
    ajoute la variance de prédiction) ;
  * un point (et une composante) est HORS BANDE si |z_i| > 2 — convention « 2σ »
    (95,45 %) des intervalles de confiance de [A95] et [OT] ;
  * sous H0 (données KK-conformes, σ correct) chaque résidu sort de la bande avec
    une probabilité p0 = 2·(1 − Φ(2)) = 0,0455 ; le nombre de sorties parmi n
    résidus suit ≈ Binomiale(n, p0) (indépendance supposée) ;
  * le spectre est CONFORME si ce nombre ne dépasse pas le quantile 95 % de cette
    binomiale : risque de fausse alarme ≤ 5 % quelle que soit la taille du spectre
    (un seuil en fraction fixe, lui, dérive avec n).

Sans niveau de bruit, AUCUN verdict n'est rendu (``conform=None``, message
explicite) : pas de seuil arbitraire de repli. Un résidu non fini (NaN dans les
données) compte comme hors bande.

Remarque : les résidus d'un modèle AJUSTÉ (Lin-KK) sont un peu moins dispersés que
le bruit (fraction (1 − h_ii) de la variance, h_ii levier) : le critère est alors
légèrement conservateur (moins de fausses alarmes), jamais l'inverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from scipy import stats

# ── Constantes du critère (sources ci-dessus) ────────────────────────────────

#: Seuil µ de Schönleber et al. [S14] pour le choix de M.
SCHONLEBER_C = 0.85
#: Demi-largeur de la bande, en écarts-types (convention 2σ de [A95]/[OT]).
KK_SIGMA_LEVEL = 2.0
#: Risque de fausse alarme du test de comptage (verdict global).
KK_FALSE_ALARM = 0.05
#: Plafond de M pour Lin-KK (coût) ; le plafond effectif est aussi ≤ nb de fréquences.
LIN_KK_MAX_M = 100


# ── Lin-KK ────────────────────────────────────────────────────────────────────

class LinKKResult(NamedTuple):
    """Résultat de ``lin_kk`` (convention physique : Im(Z) < 0 pour un circuit R-C).

    Attributes:
        M: nombre d'éléments RC retenu par le critère µ.
        mu: µ de [S14] pour ce M.
        Z_fit: impédance du circuit de Voigt ajusté (complexe, physique).
        res_re, res_im: résidus ABSOLUS Z − Z_fit (Ω), par composante.
        mu_criterion_met: False si µ ≥ c jusqu'au plafond de M (sur-ajustement du
            bruit jamais atteint : spectre très peu bruité ou plafond trop bas).
    """

    M: int
    mu: float
    Z_fit: np.ndarray
    res_re: np.ndarray
    res_im: np.ndarray
    mu_criterion_met: bool


def schonleber_mu(R_k: np.ndarray) -> float:
    """µ = 1 − Σ|R_k<0| / Σ|R_k≥0| ([S14]). −inf si aucun R_k n'est positif."""
    R_k = np.asarray(R_k, dtype=float)
    pos = float(np.sum(R_k[R_k >= 0]))
    neg = float(np.sum(np.abs(R_k[R_k < 0])))
    if pos <= 0.0:
        return float("-inf")
    return 1.0 - neg / pos


def _lin_kk_solve(omega: np.ndarray, Z: np.ndarray, M: int, add_cap: bool):
    """Moindres carrés linéaires pondérés 1/|Z| pour M éléments (τ fixés)."""
    n = len(omega)
    tau = np.geomspace(1.0 / omega.max(), 1.0 / omega.min(), M)
    wt = omega[:, None] * tau[None, :]
    denom = 1.0 + wt ** 2
    A_re = np.column_stack([np.ones(n), 1.0 / denom])
    A_im = np.column_stack([np.zeros(n), -wt / denom])
    if add_cap:  # terme 1/(jωC) : partie imaginaire −1/ω, coefficient 1/C
        A_re = np.column_stack([A_re, np.zeros(n)])
        A_im = np.column_stack([A_im, -1.0 / omega])
    w = 1.0 / np.abs(Z)                         # pondération « modulus » [B95]
    A = np.vstack([A_re * w[:, None], A_im * w[:, None]])
    b = np.concatenate([Z.real * w, Z.imag * w])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    Z_fit = A_re @ x + 1j * (A_im @ x)
    return x[1:M + 1], Z_fit


def lin_kk(
    f: np.ndarray,
    Z: np.ndarray,
    c: float = SCHONLEBER_C,
    max_M: Optional[int] = None,
    add_cap: bool = True,
) -> LinKKResult:
    """Test Lin-KK avec choix de M par le critère µ de Schönleber ([B95], [S14]).

    Args:
        f: fréquences (Hz), strictement positives, en ordre CROISSANT.
        Z: impédance complexe, convention PHYSIQUE (Im(Z) < 0 pour un R-C).
        c: seuil µ de [S14] ; M = début de la dernière plage où µ < c (voir
            l'en-tête). Doit être dans ]0, 1[.
        max_M: plafond de M ; défaut ``min(nb de fréquences, LIN_KK_MAX_M)``.
        add_cap: ajoute une capacité série 1/(jωC) (comportement capacitif BF).

    Returns:
        LinKKResult.

    Raises:
        ValueError: entrées invalides (moins de 3 points, fréquences non
            strictement croissantes ou non positives, |Z| nul ou non fini, c hors
            de ]0, 1[).
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    n = len(f)
    if n < 3 or len(Z) != n:
        raise ValueError(f"Lin-KK : au moins 3 points appariés requis (reçu f={n}, Z={len(Z)}).")
    if not (0.0 < c < 1.0):
        raise ValueError(f"Lin-KK : c doit être dans ]0, 1[ (reçu {c}).")
    if np.any(~np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("Lin-KK : fréquences finies, > 0 et strictement croissantes requises.")
    if np.any(~np.isfinite(Z)) or np.any(np.abs(Z) == 0):
        raise ValueError("Lin-KK : |Z| doit être fini et non nul en tout point.")

    omega = 2.0 * np.pi * f
    cap = min(n, LIN_KK_MAX_M) if max_M is None else int(max_M)
    if cap < 1:
        raise ValueError(f"Lin-KK : max_M doit être ≥ 1 (reçu {max_M}).")

    # Dernier franchissement de c (voir l'en-tête) : M* = 1 + max{M : µ(M) ≥ c}.
    mus = []
    for M in range(1, cap + 1):
        R_k, _ = _lin_kk_solve(omega, Z, M, add_cap)
        mus.append(schonleber_mu(R_k))
    above = [m for m, mu_m in enumerate(mus, start=1) if mu_m >= c]
    M = min((above[-1] + 1) if above else 1, cap)
    R_k, Z_fit = _lin_kk_solve(omega, Z, M, add_cap)
    mu = schonleber_mu(R_k)
    met = bool(mu < c)
    return LinKKResult(
        M=M, mu=float(mu), Z_fit=Z_fit,
        res_re=Z.real - Z_fit.real, res_im=Z.imag - Z_fit.imag,
        mu_criterion_met=met,
    )


# ── Critère UNIQUE de verdict ────────────────────────────────────────────────

#: Probabilité qu'un résidu N(0, 1) sorte de ±KK_SIGMA_LEVEL.
def _p_outside(level: float) -> float:
    return float(2.0 * stats.norm.sf(level))


def kk_allowed_outside(n: int, level: float = KK_SIGMA_LEVEL,
                       alpha: float = KK_FALSE_ALARM) -> int:
    """Nombre maximal de résidus hors bande compatible avec le bruit seul.

    Quantile (1 − alpha) de Binomiale(n, p0), p0 = P(|N(0,1)| > level).
    Ex. : n = 80 résidus (40 fréquences × 2 composantes) → 7.
    """
    if n <= 0:
        return 0
    return int(stats.binom.ppf(1.0 - alpha, n, _p_outside(level)))


@dataclass
class KKVerdict:
    """Verdict Kramers-Kronig selon le critère unique du module.

    Attributes:
        conform: True/False, ou None si aucun niveau de bruit n'était disponible.
        n_residuals: nombre de résidus testés (fréquences × composantes).
        n_outside: nombre de résidus avec |r/s| > level (non finis compris).
        n_allowed: quantile binomial (1 − alpha) — seuil du verdict.
        outside_re, outside_im: masques booléens par fréquence (None sans bruit).
        z_re, z_im: résidus normalisés r/s (None sans bruit).
        level, alpha: paramètres du critère.
        message: phrase lisible pour l'utilisateur.
    """

    conform: Optional[bool]
    n_residuals: int
    n_outside: int
    n_allowed: int
    outside_re: Optional[np.ndarray] = None
    outside_im: Optional[np.ndarray] = None
    z_re: Optional[np.ndarray] = None
    z_im: Optional[np.ndarray] = None
    level: float = KK_SIGMA_LEVEL
    alpha: float = KK_FALSE_ALARM
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def inside_mask(self) -> Optional[np.ndarray]:
        """Points dont les DEUX composantes testées sont dans la bande."""
        if self.outside_re is None and self.outside_im is None:
            return None
        parts = [m for m in (self.outside_re, self.outside_im) if m is not None]
        out = np.zeros_like(parts[0], dtype=bool)
        for m in parts:
            out |= m
        return ~out


def _normalized(res, sd, name):
    res = np.asarray(res, dtype=float)
    sd = np.asarray(sd, dtype=float)
    if sd.shape != res.shape:
        raise ValueError(f"kk_verdict : écart-type {name} de forme {sd.shape} ≠ résidus {res.shape}.")
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(f"kk_verdict : écart-type {name} non fini ou ≤ 0 — niveau de bruit invalide.")
    z = res / sd
    outside = ~np.isfinite(z) | (np.abs(z) > KK_SIGMA_LEVEL)
    return z, outside


def kk_verdict(
    res_re: Optional[np.ndarray],
    res_im: Optional[np.ndarray],
    sd_re: Optional[np.ndarray],
    sd_im: Optional[np.ndarray],
    alpha: float = KK_FALSE_ALARM,
) -> KKVerdict:
    """Applique LE critère KK du dépôt (voir l'en-tête du module).

    Args:
        res_re, res_im: résidus absolus (Ω) ; l'un des deux peut être None si une
            seule composante est testée (test par prédiction de [A95]).
        sd_re, sd_im: écart-type de chaque résidu sous H0 (Ω), même forme ; None
            pour les DEUX → aucun niveau de bruit → verdict indéterminé.
        alpha: risque de fausse alarme du test de comptage.

    Returns:
        KKVerdict.
    """
    comps = [(r, s, n) for r, s, n in ((res_re, sd_re, "Re"), (res_im, sd_im, "Im")) if r is not None]
    n_res = int(sum(np.size(r) for r, _s, _n in comps))
    if all(s is None for _r, s, _n in comps):
        return KKVerdict(
            conform=None, n_residuals=n_res, n_outside=0, n_allowed=0, alpha=alpha,
            message=(
                "Verdict KK indéterminé : aucun niveau de bruit (structure d'erreur) "
                "n'est disponible pour juger si les résidus sont « de l'ordre du bruit »."
            ),
        )
    if any(s is None for _r, s, _n in comps):
        raise ValueError("kk_verdict : fournir l'écart-type de CHAQUE composante testée.")

    z = {"Re": None, "Im": None}
    out = {"Re": None, "Im": None}
    for r, s, name in comps:
        z[name], out[name] = _normalized(r, s, name)
    n_out = int(sum(int(np.sum(m)) for m in out.values() if m is not None))
    n_allowed = kk_allowed_outside(n_res, alpha=alpha)
    conform = n_out <= n_allowed
    expected = n_res * _p_outside(KK_SIGMA_LEVEL)
    msg = (
        f"{'Conforme' if conform else 'NON conforme'} Kramers-Kronig : {n_out} résidu(s) "
        f"sur {n_res} hors de ±{KK_SIGMA_LEVEL:g}σ (attendu ≈ {expected:.1f} pour du bruit "
        f"seul ; seuil {n_allowed} au risque {alpha:.0%})."
    )
    return KKVerdict(
        conform=bool(conform), n_residuals=n_res, n_outside=n_out, n_allowed=n_allowed,
        outside_re=out["Re"], outside_im=out["Im"], z_re=z["Re"], z_im=z["Im"],
        alpha=alpha, message=msg,
    )


# ── Vérification Lin-KK d'un spectre (app : Zim = −Im(Z) > 0) ────────────────

def kramers_kronig_check(spectrum, error_structure=None, n_averaged: Optional[int] = None) -> dict:
    """Lin-KK d'un ``EISSpectrum`` jugé par le critère unique.

    Args:
        spectrum: EISSpectrum (Zim en convention de l'app : −Im(Z) > 0).
        error_structure: objet muni de ``sigmas(Zre, Zim) -> (σ_r, σ_j)`` d'UNE
            mesure (``core.measurement_model.ErrorStructure``). None → verdict
            indéterminé (``kk_passed=None``).
        n_averaged: nombre de mesures moyennées dans ``spectrum`` (σ/√n) ; défaut
            ``spectrum.n_replicates`` ou 1.

    Returns:
        dict : ``kk_passed`` (True/False/None), ``residuals_re/_im`` (Ω, données −
        Lin-KK en convention de l'app, ordre du spectre), ``Z_kk_re``, ``Z_kk_im``
        (convention de l'app),
        ``max_residual`` (résidu relatif max |Z − Z_KK|/|Z|, informatif), ``mu``,
        ``M``, ``n_outside``, ``n_allowed``, ``message``.
    """
    f = np.asarray(spectrum.f, dtype=float)
    Zre = np.asarray(spectrum.Zre, dtype=float)
    Zim = np.asarray(spectrum.Zim, dtype=float)
    order = np.argsort(f)
    inv = np.argsort(order)
    Z = Zre[order] - 1j * Zim[order]      # conversion EXACTE de convention (pas d'abs)

    lk = lin_kk(f[order], Z)
    rel = np.sqrt(lk.res_re ** 2 + lk.res_im ** 2) / np.abs(Z)

    sd_re = sd_im = None
    if error_structure is not None:
        n_avg = n_averaged if n_averaged is not None else (getattr(spectrum, "n_replicates", None) or 1)
        root_n = np.sqrt(max(int(n_avg), 1))
        sd_re, sd_im = (np.asarray(s, dtype=float) / root_n
                        for s in error_structure.sigmas(Zre[order], Zim[order]))
    verdict = kk_verdict(lk.res_re, lk.res_im, sd_re, sd_im)
    return {
        "kk_passed": verdict.conform,
        "residuals_re": lk.res_re[inv],
        "residuals_im": -lk.res_im[inv],      # Zim − Z_kk_im (convention de l'app)
        "Z_kk_re": lk.Z_fit.real[inv],
        "Z_kk_im": -lk.Z_fit.imag[inv],
        "max_residual": float(np.max(rel)),
        "mu": lk.mu,
        "M": lk.M,
        "n_outside": verdict.n_outside,
        "n_allowed": verdict.n_allowed,
        "message": verdict.message,
    }
