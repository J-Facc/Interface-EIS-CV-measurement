"""Measurement model de Voigt : structure d'erreur et conformité Kramers-Kronig.

Remplace ``fits/error_structure.py`` (conservé jusqu'à la bascule du pipeline,
étape 5). Chaîne suivie, dans l'ordre de la méthode d'Orazem :

    réplicats ──► 1. measurement model (Voigt régressé) sur CHAQUE réplicat
              ──► 2. σ(ω) = dispersion INTER-RÉPLICATS des résidus du modèle
              ──► 3. régression de la structure d'erreur σ(ω) = α|Zj| + β|Zr − Rsol| + γ|Z|² + δ
                     (σ_r = σ_j TESTÉE, deux structures si elle est rejetée)
              ──► 4. conformité KK testée avec CE MÊME modèle de Voigt, pondéré par σ
              ──► (verdict affiché)  ──► 5. fit du circuit utilisateur (fits/orazem_fit.py)

Aucun état n'est conservé d'une session à l'autre : la structure d'erreur est
recalculée à chaque appel à partir des réplicats du groupe, jamais relue d'un
fichier (AUDIT.md ERR-2 — l'ancienne persistance JSON en ajout, partagée et jamais
purgée, faisait dépendre un fit de l'historique des analyses). Si elle ne peut pas
être caractérisée, ``ErrorStructureUnavailable`` est levée avec un message
destiné à l'utilisateur : l'analyse du groupe DOIT s'arrêter là (AUDIT.md ERR-1 —
l'ancien pipeline avalait l'exception et affichait « ✅ Analyse terminée » sur des
fits vides).

Références
----------
[A92]  P. Agarwal, M. E. Orazem, L. H. García-Rubio, « Measurement Models for
       Electrochemical Impedance Spectroscopy: I. Demonstration of Applicability »,
       J. Electrochem. Soc. 139 (1992) 1917-1927.
[A95b] P. Agarwal, O. D. Crisalle, M. E. Orazem, L. H. García-Rubio, « Application
       of Measurement Models to Impedance Spectroscopy: II. Determination of the
       Stochastic Contribution to the Error Structure », J. Electrochem. Soc. 142
       (1995) 4149-4158.
[A95c] P. Agarwal, M. E. Orazem, L. H. García-Rubio, « … III. Evaluation of
       Consistency with the Kramers-Kronig Relations », J. Electrochem. Soc. 142
       (1995) 4159-4168.
[O04]  M. E. Orazem, « A systematic approach toward error structure identification
       for impedance spectroscopy », J. Electroanal. Chem. 572 (2004) 317-327.
[OT]   M. E. Orazem, B. Tribollet, *Electrochemical Impedance Spectroscopy*, 2e éd.,
       Wiley, 2017 — chapitres « Measurement Models », « Error Structure »,
       « Kramers-Kronig Relations ».
[BW]   D. M. Bates, D. G. Watts, *Nonlinear Regression Analysis and Its
       Applications*, Wiley, 1988 (test F de modèles emboîtés, §3.10 ; leviers).
[CW]   R. D. Cook, S. Weisberg, *Residuals and Influence in Regression*, Chapman &
       Hall, 1982 (résidus studentisés).
[BBA]  S. T. Buckland, K. P. Burnham, N. H. Augustin, « Model selection: an integral
       part of inference », Biometrics 53 (1997) 603-618 ; K. P. Burnham, D. R.
       Anderson, *Model Selection and Multimodel Inference*, 2e éd., Springer, 2002
       (poids d'Akaike, variance inconditionnelle).
⚠ Les numéros d'ÉQUATION de ces sources n'ont pas pu être vérifiés depuis
l'environnement où ce module a été écrit : les formules sont rappelées
explicitement dans chaque docstring, les sources sont citées par article/chapitre.

Ce que faisait l'ancien code (vérifié, ``fits/error_structure.py``)
--------------------------------------------------------------------
Il n'y avait PAS d'ajustement successif d'éléments de Voigt ni de critère de
parcimonie. La voie par défaut (« réplicats-directs ») prenait l'écart-type brut
inter-réplicats de Z, sans aucun modèle — une dérive entre réplicats y était donc
comptée comme du bruit. L'option ``voigt_based`` appelait ``lin_kk`` : un Voigt
LINÉAIRE à τ fixés et M = min(100, 2n/3) éléments imposé par le nombre de points,
sans test. Autres écarts corrigés ici : forme de la structure (α|Zr| + β|Zi| au lieu
de α|Zj| + β|Zr − Rsol| de [O04]) ; ``equal_re_im`` interprété comme « α = β »
alors que l'hypothèse du measurement model est σ_r = σ_j ([A95b]) ; terme γ régressé
puis IGNORÉ par ``sigma()`` quand R_m est inconnu ; régression non pondérée
dominée par les grands |Z| ; biais de l'écart-type à petit n non corrigé.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import least_squares, nnls
from scipy.special import gammaln

from core.regression_stats import jacobian_statistics
from fits.kk_validation import KK_FALSE_ALARM, KKVerdict, kk_verdict

__all__ = [
    "ErrorStructureUnavailable",
    "MeasurementModelOptions",
    "VoigtModel",
    "ErrorStructure",
    "ErrorStructureEstimate",
    "KKConsistency",
    "MeasurementModelAnalysis",
    "fit_voigt",
    "characterize_error_structure",
    "check_kk_consistency",
    "analyze_replicates",
    "common_frequency_grid",
]


# ═════════════════════════════════════════════════════════════════════════════
# Exception : groupe non analysable
# ═════════════════════════════════════════════════════════════════════════════

class ErrorStructureUnavailable(RuntimeError):
    """La structure d'erreur ne peut pas être caractérisée pour ce groupe.

    Contrat : cette exception INTERROMPT l'analyse Orazem du groupe concerné. Elle
    ne doit jamais être convertie en « fit vide » ni en poids arbitraires :
    l'appelant (pipeline, étape 5) l'attrape POUR CE GROUPE seulement, n'y associe
    aucun FitResult et affiche ``user_message`` tel quel.

    Attributes:
        reason: cause technique courte.
        group_label: groupe concerné.
        n_replicates: nombre de réplicats fournis.
        user_message: phrase complète destinée à l'utilisateur (cause + remède).
    """

    def __init__(self, reason: str, *, group_label: str = "", n_replicates: Optional[int] = None,
                 remedy: str = ""):
        self.reason = reason
        self.group_label = group_label
        self.n_replicates = n_replicates
        self.remedy = remedy or (
            "Fournissez au moins 3 réplicats indépendants de ce groupe (mêmes conditions, "
            "mêmes fréquences) pour caractériser le bruit de mesure."
        )
        where = f"Groupe « {group_label} » : " if group_label else ""
        self.user_message = (
            f"{where}analyse Orazem interrompue — structure d'erreur non caractérisable : "
            f"{reason}. {self.remedy} Aucun fit n'a été réalisé pour ce groupe "
            f"(aucune pondération arbitraire n'est utilisée en remplacement)."
        )
        super().__init__(self.user_message)


# ═════════════════════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MeasurementModelOptions:
    """Réglages du measurement model (valeurs par défaut justifiées ci-dessous).

    Attributes:
        min_replicates: réplicats minimum pour estimer σ (écart-type à n−1 ddl :
            n = 3 est le minimum raisonnable ; lu dans ``fit.error_structure.min_replicates``).
        series_capacitance: ajoute un terme 1/(jωC) (spectres capacitifs en BF :
            électrode bloquante, Warburg réflectif). Désactivé par défaut : les
            capteurs de l'application ont une impédance finie en continu.
        max_elements: plafond du nombre d'éléments de Voigt.
        f_test_alpha: niveau du test F d'ajout d'un élément (5 %).
        significance_sigma: un paramètre est significatif si son intervalle à
            ±significance_sigma·σ exclut zéro (2 → 95,45 %, [A92]).
        passes: passes de caractérisation (1 = pondération « modulus » seule ;
            2 = une passe de raffinement pondérée par la structure obtenue, [A95b]).
        leverage_correction: corrige les résidus de leur levier (1 − h_ii) avant
            d'en prendre la dispersion ([CW]).
        freq_rtol: tolérance relative d'appariement des fréquences entre réplicats.
    """

    min_replicates: int = 3
    series_capacitance: bool = False
    max_elements: int = 20
    f_test_alpha: float = 0.05
    significance_sigma: float = 2.0
    passes: int = 2
    leverage_correction: bool = True
    freq_rtol: float = 1e-3

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "MeasurementModelOptions":
        """Lit ``fit.error_structure.min_replicates`` (seule clé partagée avec l'ancien module)."""
        fit_cfg = (config or {}).get("fit", {}) if isinstance(config, dict) else {}
        es_cfg = fit_cfg.get("error_structure", {}) or {}
        kwargs = {}
        if "min_replicates" in es_cfg and es_cfg["min_replicates"] is not None:
            kwargs["min_replicates"] = int(es_cfg["min_replicates"])
        return cls(**kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Measurement model de Voigt régressé, par ajout successif d'éléments
# ═════════════════════════════════════════════════════════════════════════════

_COMPONENTS = ("complex", "real", "imag")

#: σ(ω) ≤ _SIGMA_RESOLUTION·|Z| = dispersion non résolue (arrondi numérique) → refus.
_SIGMA_RESOLUTION = 1e-9


@dataclass
class VoigtModel:
    """Circuit de Voigt régressé ([A92]) :

        Z(ω) = R0 + Σ_{k=1..K} R_k / (1 + jωτ_k)  [+ 1/(jωC)]

    Convention PHYSIQUE (Im(Z) < 0 pour un R-C). Tout circuit de ce type satisfait
    les relations de Kramers-Kronig : c'est ce qui en fait l'outil du test KK.

    Attributes:
        R0: résistance haute fréquence (Ω) ; None si non identifiable (fit « imag »).
        R: résistances R_k (Ω).
        tau: constantes de temps τ_k (s), triées croissantes.
        c_inv: 1/C (F⁻¹) si capacité série, sinon None.
        x: vecteur des paramètres de régression [R0?, R_1..R_K, 1/C?, ln τ_1..ln τ_K].
        cov: covariance de x (déjà multipliée par ``variance_factor``).
        chi2, dof: χ² pondéré et degrés de liberté (n_obs − P).
        component: « complex », « real » ou « imag » (composantes ajustées).
        absolute_sigma: True si les écarts-types fournis sont de vraies σ.
        variance_factor: facteur appliqué à (JᵀJ)⁺ pour obtenir ``cov``.
        leverage: leviers h_ii des résidus, dans l'ordre [réels…, imaginaires…]
            restreint aux composantes ajustées.
        history: une entrée par K essayé (K, χ², p du test F, significativité,
            accepté ?) — trace du critère de parcimonie.
    """

    R0: Optional[float]
    R: np.ndarray
    tau: np.ndarray
    c_inv: Optional[float]
    x: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    component: str
    absolute_sigma: bool
    variance_factor: float
    leverage: np.ndarray
    history: list = field(default_factory=list)

    @property
    def n_elements(self) -> int:
        return int(len(self.R))

    @property
    def chi2_reduced(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        """Z(ω) complexe (convention physique) ; R0 absent compté pour 0."""
        omega = np.asarray(omega, dtype=float)
        Z = np.full(omega.shape, self.R0 or 0.0, dtype=complex)
        for Rk, tk in zip(self.R, self.tau):
            Z = Z + Rk / (1.0 + 1j * omega * tk)
        if self.c_inv is not None:
            Z = Z + self.c_inv / (1j * omega)
        return Z


class _Layout:
    """Disposition du vecteur x = [R0?, R_1..R_K, c_inv?, lnτ_1..lnτ_K]."""

    def __init__(self, K: int, with_R0: bool, with_cinv: bool):
        self.K, self.with_R0, self.with_cinv = K, with_R0, with_cinv
        self.n_lin = int(with_R0) + K + int(with_cinv)
        self.i_R = slice(int(with_R0), int(with_R0) + K)
        self.i_cinv = int(with_R0) + K if with_cinv else None
        self.i_ln = slice(self.n_lin, self.n_lin + K)
        self.p = self.n_lin + K

    def lin_basis(self, omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Base complexe (N × n_lin) : Z = B @ lin."""
        B = np.empty((omega.size, self.n_lin), dtype=complex)
        j = int(self.with_R0)
        if self.with_R0:
            B[:, 0] = 1.0
        B[:, j:j + len(tau)] = 1.0 / (1.0 + 1j * omega[:, None] * np.asarray(tau)[None, :])
        if self.with_cinv:
            B[:, -1] = 1.0 / (1j * omega)
        return B

    def model(self, x: np.ndarray, omega: np.ndarray) -> np.ndarray:
        # LM laisse ln τ libre : un pas extrême peut déborder (inf/nan). Le résultat
        # non fini est alors rejeté par le contrôle de convergence de ``refine``.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self.lin_basis(omega, np.exp(x[self.i_ln])) @ x[: self.n_lin]

    def jacobian(self, x: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """∂Z/∂x : base linéaire, puis ∂Z/∂ln τ_k = −R_k·jωτ_k / (1 + jωτ_k)²."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            tau = np.exp(x[self.i_ln])
            B = self.lin_basis(omega, tau)
            jwt = 1j * omega[:, None] * tau[None, :]
            dln = -x[self.i_R][None, :] * jwt / (1.0 + jwt) ** 2
        return np.concatenate([B, dln], axis=1)


def _stack(Zc: np.ndarray, sd_re, sd_im, component: str) -> np.ndarray:
    """Empile les composantes ajustées d'un tableau complexe (ou d'une matrice
    complexe colonne par colonne), divisées par leurs écarts-types."""
    parts = []
    if component in ("complex", "real"):
        parts.append(Zc.real / (sd_re if Zc.ndim == 1 else sd_re[:, None]))
    if component in ("complex", "imag"):
        parts.append(Zc.imag / (sd_im if Zc.ndim == 1 else sd_im[:, None]))
    return np.concatenate(parts, axis=0)


def _tau_candidates(omega: np.ndarray) -> np.ndarray:
    """Candidats pour le τ d'un nouvel élément : 4 par décade, fenêtre mesurée
    élargie d'une décade de part et d'autre (relaxations partiellement hors fenêtre)."""
    lo, hi = 0.1 / omega.max(), 10.0 / omega.min()
    n = int(np.ceil(4 * np.log10(hi / lo))) + 1
    return np.geomspace(lo, hi, n)


def _significant(model_x, std, layout: _Layout, level: float) -> bool:
    """Critère de [A92] : l'intervalle ±level·σ de chaque paramètre exclut zéro.

    Pour R_k : |R_k| > level·σ(R_k). Pour τ_k (régressé en ln τ) : τ − level·σ_τ > 0
    avec σ_τ ≈ τ·σ(ln τ), soit σ(ln τ) < 1/level. R0 et 1/C ne sont pas testés
    (asymptotes : une valeur nulle est physiquement admissible).
    """
    R = model_x[layout.i_R]
    sR = std[layout.i_R]
    sln = std[layout.i_ln]
    if not (np.all(np.isfinite(sR)) and np.all(np.isfinite(sln))):
        return False
    return bool(np.all(np.abs(R) > level * sR) and np.all(sln < 1.0 / level))


class _VoigtRegression:
    """Un problème de régression de Voigt (données, poids, composantes) et ses opérations.

    Séparé de ``fit_voigt`` pour que le test KK puisse réutiliser exactement la même
    machinerie (même initialisation, même raffinement, même covariance) à K fixé.
    """

    def __init__(self, f, Zre, Zim, sd_re, sd_im, *, absolute_sigma: bool, component: str,
                 options: MeasurementModelOptions):
        if component not in _COMPONENTS:
            raise ValueError(f"component doit être l'un de {_COMPONENTS} (reçu {component!r}).")
        f = np.asarray(f, dtype=float)
        Zre = np.asarray(Zre, dtype=float)
        Zim = np.asarray(Zim, dtype=float)
        sd_re = np.broadcast_to(np.asarray(sd_re, dtype=float), f.shape).copy()
        sd_im = np.broadcast_to(np.asarray(sd_im, dtype=float), f.shape).copy()
        for name, arr in (("f", f), ("Zre", Zre), ("Zim", Zim), ("sd_re", sd_re), ("sd_im", sd_im)):
            if arr.shape != f.shape or not np.all(np.isfinite(arr)):
                raise ValueError(f"fit_voigt : « {name} » doit être fini et de même forme que f.")
        if np.any(f <= 0) or np.any(sd_re <= 0) or np.any(sd_im <= 0):
            raise ValueError("fit_voigt : fréquences et écarts-types doivent être > 0.")
        self.opt = options
        self.component = component
        self.absolute_sigma = absolute_sigma
        self.omega = 2.0 * np.pi * f
        self.sd_re, self.sd_im = sd_re, sd_im
        self.y = _stack(Zre - 1j * Zim, sd_re, sd_im, component)   # convention physique
        self.n_obs = self.y.size
        self.with_R0 = component != "imag"
        self.with_cinv = options.series_capacitance and component != "real"
        self.ln_lo = np.log(0.01 / self.omega.max())
        self.ln_hi = np.log(100.0 / self.omega.min())
        self.candidates = _tau_candidates(self.omega)
        self.k_cap = int(min(options.max_elements,
                             (self.n_obs - int(self.with_R0) - int(self.with_cinv) - 2) // 2))
        if self.k_cap < 1:
            raise ValueError(f"fit_voigt : trop peu d'observations ({self.n_obs}) pour un élément de Voigt.")

    def layout(self, K: int) -> _Layout:
        return _Layout(K, self.with_R0, self.with_cinv)

    def linear_solve(self, tau):
        lay = self.layout(len(tau))
        A = _stack(lay.lin_basis(self.omega, tau), self.sd_re, self.sd_im, self.component)
        lin, *_ = np.linalg.lstsq(A, self.y, rcond=None)
        r = A @ lin - self.y
        return lin, float(r @ r)

    def refine(self, tau0):
        """Moindres carrés non linéaires depuis τ0 (paramètres linéaires résolus d'abord)."""
        lay = self.layout(len(tau0))
        lin0, _ = self.linear_solve(tau0)
        tau0 = np.clip(tau0, np.exp(self.ln_lo) * 1.0001, np.exp(self.ln_hi) * 0.9999)
        x0 = np.concatenate([lin0, np.log(tau0)])
        omega, y, comp, sd_re, sd_im = self.omega, self.y, self.component, self.sd_re, self.sd_im

        def fun(x):
            return _stack(lay.model(x, omega), sd_re, sd_im, comp) - y

        def jac(x):
            return _stack(lay.jacobian(x, omega), sd_re, sd_im, comp)

        def ok(r):
            return bool(r.status > 0 and np.all(np.isfinite(r.x)) and np.isfinite(r.cost)
                        and np.all((r.x[lay.i_ln] >= self.ln_lo) & (r.x[lay.i_ln] <= self.ln_hi)))

        # D'abord Levenberg-Marquardt (MINPACK, rapide) avec ln τ libre ; s'il échoue
        # ou fait sortir un τ de [0,01/ω_max, 100/ω_min], reprise en TRF borné.
        res = least_squares(fun, x0, jac=jac, method="lm", x_scale="jac",
                            ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=100 * (lay.p + 1))
        if not ok(res):
            lb = np.full(lay.p, -np.inf)
            ub = np.full(lay.p, np.inf)
            lb[lay.i_ln], ub[lay.i_ln] = self.ln_lo, self.ln_hi
            res = least_squares(fun, x0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac",
                                ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=200 * (lay.p + 1))
        return (lay, res) if ok(res) else None

    def summarize(self, lay, res):
        chi2 = 2.0 * float(res.cost)
        dof = self.n_obs - lay.p
        st = jacobian_statistics(res.jac)
        chi2_nu = chi2 / dof if dof > 0 else np.inf
        factor = max(1.0, chi2_nu) if self.absolute_sigma else chi2_nu
        return chi2, dof, st, factor

    def add_element(self, tau_fixed):
        """Meilleur modèle à len(tau_fixed)+1 éléments (nouveau τ pris sur la grille)."""
        scored = []
        for t in self.candidates:
            if tau_fixed.size and np.min(np.abs(np.log(tau_fixed / t))) < 0.05:
                continue
            _lin, chi2 = self.linear_solve(np.append(tau_fixed, t))
            scored.append((chi2, t))
        scored.sort(key=lambda s: s[0])
        # Les deux meilleurs candidats qui convergent (au plus 6 essais).
        best, n_ok = None, 0
        for _chi2, t in scored[:6]:
            out = self.refine(np.append(tau_fixed, t))
            if out is not None:
                n_ok += 1
                if best is None or out[1].cost < best[1].cost:
                    best = out
                if n_ok == 2:
                    break
        return best

    def to_model(self, lay, res, history=None) -> VoigtModel:
        chi2, dof, st, factor = self.summarize(lay, res)
        x = res.x.copy()
        order = np.argsort(x[lay.i_ln])
        # Tri des éléments par τ croissant (x et covariance permutés ensemble).
        perm = np.arange(lay.p)
        perm[lay.i_R] = np.arange(lay.p)[lay.i_R][order]
        perm[lay.i_ln] = np.arange(lay.p)[lay.i_ln][order]
        x = x[perm]
        cov = (st.cov * factor)[np.ix_(perm, perm)]
        return VoigtModel(
            R0=float(x[0]) if lay.with_R0 else None,
            R=x[lay.i_R].copy(),
            tau=np.exp(x[lay.i_ln]),
            c_inv=float(x[lay.i_cinv]) if lay.i_cinv is not None else None,
            x=x, cov=cov, chi2=chi2, dof=dof, component=self.component,
            absolute_sigma=self.absolute_sigma, variance_factor=float(factor),
            leverage=st.leverage, history=list(history or []),
        )

    def successive(self, start_tau: Optional[np.ndarray] = None) -> VoigtModel:
        """Ajout successif d'éléments (critère de ``fit_voigt``).

        ``start_tau`` (démarrage à chaud, ex. passe précédente) : le modèle à
        len(start_tau) éléments est réajusté ; s'il converge et que TOUS ses
        paramètres restent significatifs, l'ajout reprend de là (K + 1…) ; sinon on
        repart de K = 1. Les règles d'acceptation sont inchangées.
        """
        opt = self.opt
        current = None
        history = []
        if start_tau is not None and 1 <= len(start_tau) <= self.k_cap:
            warm = self.refine(np.sort(np.asarray(start_tau, dtype=float)))
            if warm is not None:
                lay, res = warm
                chi2, dof, st, factor = self.summarize(lay, res)
                if _significant(res.x, st.std * np.sqrt(factor), lay, opt.significance_sigma):
                    history.append(dict(K=lay.K, chi2=chi2, dof=dof, f_pvalue=None, significant=True,
                                        accepted=True, reason="démarrage à chaud"))
                    current = (lay, res, chi2)
        if current is None:
            first = self.add_element(np.array([]))
            if first is None:
                raise RuntimeError("measurement model : la régression à un élément de Voigt ne converge pas.")
            lay, res = first
            chi2, dof, st, factor = self.summarize(lay, res)
            history.append(dict(K=1, chi2=chi2, dof=dof, f_pvalue=None,
                                significant=_significant(res.x, st.std * np.sqrt(factor), lay,
                                                         opt.significance_sigma),
                                accepted=True))
            current = (lay, res, chi2)
        while current[0].K < self.k_cap:
            lay_k, res_k, chi2_k = current
            cand = self.add_element(np.exp(res_k.x[lay_k.i_ln]))
            K1 = lay_k.K + 1
            if cand is None:
                history.append(dict(K=K1, chi2=None, dof=None, f_pvalue=None, significant=False,
                                    accepted=False, reason="non convergé"))
                break
            lay1, res1 = cand
            chi2_1, dof_1, st1, factor1 = self.summarize(lay1, res1)
            if dof_1 <= 0 or chi2_1 >= chi2_k:
                p_f = 1.0
            else:
                F = ((chi2_k - chi2_1) / 2.0) / (chi2_1 / dof_1) if chi2_1 > 0 else np.inf
                p_f = float(stats.f.sf(F, 2, dof_1))
            sig = _significant(res1.x, st1.std * np.sqrt(factor1), lay1, opt.significance_sigma)
            accepted = (p_f < opt.f_test_alpha) and sig
            history.append(dict(K=K1, chi2=chi2_1, dof=dof_1, f_pvalue=p_f, significant=sig,
                                accepted=accepted))
            if not accepted:
                break
            current = (lay1, res1, chi2_1)
        return self.to_model(current[0], current[1], history)


def fit_voigt(
    f: np.ndarray,
    Zre: np.ndarray,
    Zim: np.ndarray,
    sd_re: np.ndarray,
    sd_im: np.ndarray,
    *,
    absolute_sigma: bool,
    component: str = "complex",
    options: Optional[MeasurementModelOptions] = None,
    start_tau: Optional[np.ndarray] = None,
) -> VoigtModel:
    """Régression du measurement model de Voigt par ajout SUCCESSIF d'éléments.

    Critère de parcimonie — choix et justification
    ----------------------------------------------
    On part de K = 1 et on essaie K + 1 ; l'élément ajouté n'est CONSERVÉ que si :

      (a) la régression converge ;
      (b) il améliore significativement l'ajustement selon le TEST F des modèles
          emboîtés ([BW] §3.10) :
              F = [(χ²_K − χ²_{K+1}) / 2] / [χ²_{K+1} / (n_obs − P_{K+1})],
          comparé à F_{0,95}(2, n_obs − P_{K+1}) (un élément = 2 paramètres R, τ) ;
      (c) TOUS les paramètres du modèle à K + 1 éléments restent significatifs :
          l'intervalle à 2σ (95,45 %) de chaque R_k et τ_k exclut zéro.

    Le premier refus arrête l'ajout ; le modèle retenu est le dernier accepté.

    (c) est LE critère d'Agarwal, Orazem & García-Rubio [A92] (repris dans [OT],
    chap. « Measurement Models ») : on ajoute des éléments tant que la régression
    converge et que les intervalles de confiance des paramètres n'incluent pas
    zéro — le nombre d'éléments est « le plus grand nombre de paramètres
    statistiquement significatifs » que le bruit laisse extraire. [A92] ne pose ni
    test F ni AIC ; (b) est ajouté ici et c'est le TEST F qui est retenu, plutôt
    que l'AIC, pour trois raisons :

      1. les modèles comparés sont EMBOÎTÉS (K ⊂ K + 1) : c'est le cadre exact du
         test F, qui contrôle le risque de retenir un élément inutile (5 %), en
         cohérence avec la philosophie « significativité statistique » de [A92] ;
         l'AIC vise la comparaison de modèles non emboîtés en perte de prédiction ;
      2. le test F porte sur un RAPPORT de χ² : il reste valide quand σ n'est connu
         qu'à un facteur près — c'est le cas de la première passe, pondérée
         « modulus » (σ ∝ |Z|) faute de structure d'erreur ; l'AIC exige la
         vraisemblance absolue, donc σ absolu ;
      3. pour 2 paramètres ajoutés, l'AIC équivaut à accepter si Δχ²/s² > 4, soit un
         risque implicite ≈ 13,5 % (χ²₂) : il retient plus volontiers un élément
         superflu. Or ici un élément superflu absorbe du BRUIT et fait SOUS-estimer
         σ à l'étape suivante : la prudence du test F est le bon côté de l'erreur.

    Initialisation d'un nouvel élément : τ essayé sur une grille (4/décade) ; pour
    chaque τ candidat, les paramètres linéaires (R0, R_k, 1/C) sont résolus
    exactement par moindres carrés (projection variable) ; les candidats sont
    affinés dans cet ordre par ``least_squares`` (Levenberg-Marquardt de MINPACK,
    jacobienne analytique ; repli en TRF borné si LM échoue ou fait sortir un τ de
    [0,01/ω_max, 100/ω_min]) jusqu'à deux convergences, le meilleur χ² est retenu.

    Covariance utilisée pour (c) : (JᵀJ)⁺ (``core.regression_stats``) multipliée par
    χ²_ν si σ n'est connu qu'à un facteur près (``absolute_sigma=False``), par
    max(1, χ²_ν) sinon (un modèle qui n'atteint pas le bruit ne doit pas voir ses
    paramètres jugés plus précis qu'ils ne sont).

    Args:
        f: fréquences (Hz) > 0.
        Zre, Zim: impédance, convention de l'APPLICATION (Zim = −Im(Z) > 0).
        sd_re, sd_im: écarts-types (Ω) des composantes (absolus ou relatifs).
        absolute_sigma: True si sd_* sont de vraies σ (et non connues à un facteur près).
        component: « complex » (Re + Im), « real » (Re seul — R0 identifiable,
            1/C non) ou « imag » (Im seul — R0 non identifiable, 1/C oui).
        options: MeasurementModelOptions.
        start_tau: démarrage à chaud (τ d'un modèle voisin, ex. passe précédente) —
            voir ``_VoigtRegression.successive``.

    Returns:
        VoigtModel.

    Raises:
        ValueError: entrées invalides (formes, valeurs non finies, σ ≤ 0).
        RuntimeError: la régression à UN élément ne converge pas.
    """
    reg = _VoigtRegression(f, Zre, Zim, sd_re, sd_im, absolute_sigma=absolute_sigma,
                           component=component, options=options or MeasurementModelOptions())
    return reg.successive(start_tau)


def _voigt_jacobian(model: VoigtModel, omega: np.ndarray) -> np.ndarray:
    """∂Z/∂x (complexe, N × P) du modèle final, dans l'ordre de ``model.x``."""
    lay = _Layout(model.n_elements, model.R0 is not None, model.c_inv is not None)
    return lay.jacobian(model.x, omega)


# ═════════════════════════════════════════════════════════════════════════════
# 2-3. Structure d'erreur
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ErrorStructure:
    """Structure d'erreur stochastique d'UNE mesure ([O04], [OT]) :

        σ(ω) = α·|Z_j| + β·|Z_r − R_sol| + γ·|Z|² + δ

    * Hypothèse du measurement model : σ_r = σ_j (égalité des variances des parties
      réelle et imaginaire, conséquence des relations de Kramers-Kronig appliquées
      au bruit et vérifiée expérimentalement, [A95b]). C'est le SENS de cette
      hypothèse — pas « α = β » comme le supposait l'ancien ``equal_re_im``. Elle
      est TESTÉE sur les données (``characterize_error_structure``) et non imposée :
      si elle est rejetée, une structure distincte est estimée pour chaque
      composante (``equal_re_im=False``, coefficients de Im dans ``im``) et
      l'écart est signalé.
    * γ est ici le coefficient de |Z|² (Ω⁻¹), c.-à-d. le γ/R_m de [O04] : R_m (résistance
      de mesure du potentiostat, qui change avec la gamme de courant) n'a pas à être
      connue, et le terme est TOUJOURS évalué (l'ancien code le régressait puis
      l'ignorait sans R_m).
    * δ (Ω) : bruit additif de fond. Aucun plancher : si la dispersion observée est
      nulle, la structure n'est pas identifiable (``ErrorStructureUnavailable``).

    Attributes:
        alpha, beta, gamma, delta: coefficients (≥ 0) de σ_r (et de σ_j si égales).
        R_sol: résistance haute fréquence moyenne des measurement models (Ω).
        n_replicates: réplicats utilisés.
        dof: degrés de liberté effectifs de l'estimation de σ (voir
            ``characterize_error_structure``) — sert à l'intervalle attendu de χ²_ν.
        label: groupe caractérisé.
        characterized_at: horodatage UTC de la caractérisation (DANS cette session).
        equal_re_im: True si σ_r = σ_j n'a pas été rejetée.
        im: (α, β, γ, δ) de σ_j si ``equal_re_im`` est False, sinon None.
        equality_pvalue: p du test F de σ_r = σ_j (structure commune vs deux structures).
    """

    alpha: float
    beta: float
    gamma: float
    delta: float
    R_sol: float
    n_replicates: int
    dof: int
    label: str = ""
    characterized_at: str = ""
    equal_re_im: bool = True
    im: Optional[tuple] = None
    equality_pvalue: float = float("nan")

    def _form(self, coefs, Zre, Zim) -> np.ndarray:
        a, b, g, d = coefs
        return a * np.abs(Zim) + b * np.abs(Zre - self.R_sol) + g * (Zre ** 2 + Zim ** 2) + d

    def sigmas(self, Zre, Zim) -> tuple:
        """(σ_r, σ_j) d'UNE mesure (Ω), en chaque point.

        Args:
            Zre, Zim: impédance (Zim en convention de l'app ; seul |Zim| intervient).
        """
        Zre = np.asarray(Zre, dtype=float)
        Zim = np.asarray(Zim, dtype=float)
        s_re = self._form((self.alpha, self.beta, self.gamma, self.delta), Zre, Zim)
        s_im = s_re.copy() if self.equal_re_im else self._form(self.im, Zre, Zim)
        return s_re, s_im

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "delta": self.delta, "R_sol": self.R_sol, "n_replicates": self.n_replicates,
            "dof": self.dof, "label": self.label, "characterized_at": self.characterized_at,
            "equal_re_im": self.equal_re_im,
            "im": None if self.im is None else list(self.im),
            "equality_pvalue": self.equality_pvalue,
            "form": "sigma = alpha*|Zj| + beta*|Zr - R_sol| + gamma*|Z|^2 + delta",
        }


@dataclass
class ErrorStructureEstimate:
    """Tout ce qui a servi à estimer la structure (transparence, affichage).

    Attributes:
        error_structure: coefficients retenus (dernière passe).
        frequencies: grille commune (Hz, croissante).
        sigma_emp_re, sigma_emp_im: σ empiriques par fréquence (après corrections),
            NaN où un point a été écarté (levier ≈ 1).
        sigma_model_re, sigma_model_im: σ_r(ω), σ_j(ω) de la structure sur la grille,
            évaluées sur le Z moyen.
        voigt_models: measurement model de chaque réplicat (dernière passe).
        passes: coefficients après chaque passe (liste de dicts).
        rel_rms: écart relatif RMS pondéré entre σ empirique et σ modèle.
    """

    error_structure: ErrorStructure
    frequencies: np.ndarray
    sigma_emp_re: np.ndarray
    sigma_emp_im: np.ndarray
    sigma_model_re: np.ndarray
    sigma_model_im: np.ndarray
    voigt_models: list
    passes: list
    rel_rms: float


def common_frequency_grid(replicates: Sequence, rtol: float = 1e-3, label: str = "") -> tuple:
    """Fréquences présentes dans TOUS les réplicats (appariement à ``rtol`` près).

    Les résidus ne sont PAS interpolés d'une grille à l'autre : interpoler du bruit
    le moyenne et ferait sous-estimer σ. On ne garde que les fréquences communes
    (cas normal : mêmes réglages d'acquisition → grilles identiques).

    Returns:
        (f_common croissant, [indices dans chaque réplicat]).

    Raises:
        ErrorStructureUnavailable: valeurs non finies ou fréquences ≤ 0, ou moins
            de la moitié des points (ou moins de 10) en commun.
    """
    for sp in replicates:
        arrays = [np.asarray(getattr(sp, a), dtype=float) for a in ("f", "Zre", "Zim")]
        if (any(not np.all(np.isfinite(a)) for a in arrays)
                or any(len(a) != len(arrays[0]) for a in arrays) or np.any(arrays[0] <= 0)):
            raise ErrorStructureUnavailable(
                f"valeurs manquantes ou non finies dans le réplicat « {getattr(sp, 'label', '?')} »",
                group_label=label, n_replicates=len(replicates),
                remedy="Corrigez ou excluez ce fichier au prétraitement.",
            )
    f_ref = np.sort(np.asarray(replicates[0].f, dtype=float))
    keep = np.ones(len(f_ref), dtype=bool)
    tol = np.log1p(rtol)
    for sp in replicates:
        lf = np.log(np.asarray(sp.f, dtype=float))
        d = np.abs(np.log(f_ref)[:, None] - lf[None, :])
        keep &= d.min(axis=1) <= tol
    f_common = f_ref[keep]
    n_min = max(10, int(np.ceil(0.5 * len(f_ref))))
    if len(f_common) < n_min:
        raise ErrorStructureUnavailable(
            f"seulement {len(f_common)} fréquence(s) communes à tous les réplicats (minimum {n_min})",
            group_label=label, n_replicates=len(replicates),
            remedy="Utilisez des réplicats acquis avec les mêmes réglages de fréquence.",
        )
    idx = []
    for sp in replicates:
        lf = np.log(np.asarray(sp.f, dtype=float))
        idx.append(np.argmin(np.abs(np.log(f_common)[:, None] - lf[None, :]), axis=1))
    return f_common, idx


def _std_over_replicates(res: np.ndarray) -> tuple:
    """Écart-type (ddof = 1) sur l'axe des réplicats, en ignorant les NaN (points
    écartés pour levier) ; NaN là où il reste moins de 2 valeurs. Rend aussi n."""
    ok = np.isfinite(res)
    n = ok.sum(axis=0)
    x = np.where(ok, res, 0.0)
    mean = x.sum(axis=0) / np.maximum(n, 1)
    ss = np.where(ok, (res - mean) ** 2, 0.0).sum(axis=0)
    sd = np.full(res.shape[1], np.nan)
    sd[n >= 2] = np.sqrt(ss[n >= 2] / (n[n >= 2] - 1))
    return sd, n


def _c4(n: np.ndarray) -> np.ndarray:
    """c4(n) = E[s]/σ pour n tirages normaux (s à ddof=1) — correction du biais de s."""
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, np.nan)
    ok = n >= 2
    out[ok] = np.sqrt(2.0 / (n[ok] - 1.0)) * np.exp(gammaln(n[ok] / 2.0) - gammaln((n[ok] - 1.0) / 2.0))
    return out


def _regress_structure(Zr, Zj, R_sol, targets):
    """NNLS itérativement repondérée de σ_emp sur [|Zj|, |Zr − R_sol|, |Z|², 1].

    ``targets`` : liste de σ empiriques (une par composante) partageant les MÊMES
    coefficients (deux cibles empilées = hypothèse σ_r = σ_j).

    s_emp a un écart-type ∝ σ (≈ σ/√(2(n−1)) pour du bruit normal) : une régression
    non pondérée est dominée par les grands σ (grands |Z|) et estime mal δ. On
    minimise donc Σ[(s_emp − σ_mod)/σ_mod]² par moindres carrés repondérés (poids
    1/σ_mod² de l'itération précédente, départ non pondéré), coefficients ≥ 0
    (amplitudes de bruit). Les cibles NaN sont écartées.

    Returns:
        (coefficients, écart relatif RMS, somme des carrés relatifs, nb de cibles).
    """
    X1 = np.column_stack([np.abs(Zj), np.abs(Zr - R_sol), Zr ** 2 + Zj ** 2, np.ones_like(Zr)])
    X = np.vstack([X1] * len(targets))
    y = np.concatenate(targets)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    col = np.linalg.norm(X, axis=0)
    col[col == 0] = 1.0
    w = np.ones_like(y)
    coef = np.zeros(4)
    for _ in range(50):
        cs, _ = nnls((X / col) * w[:, None], y * w)
        new = cs / col
        pred = X @ new
        if np.max(pred) <= 0:
            coef = new
            break
        w = 1.0 / np.maximum(pred, 1e-9 * np.max(pred))
        if np.allclose(new, coef, rtol=1e-8, atol=0.0):
            coef = new
            break
        coef = new
    pred = X @ coef
    with np.errstate(divide="ignore", invalid="ignore"):
        rss = float(np.sum(((y - pred) / pred) ** 2)) if np.all(pred > 0) else float("inf")
    return coef, float(np.sqrt(rss / max(y.size, 1))), rss, int(y.size)


def _choose_structure(Zr, Zj, R_sol, s_re, s_im, alpha: float = 0.05):
    """Structure commune (σ_r = σ_j) ou deux structures, par un test F emboîté.

    Modèle restreint : 4 coefficients communs aux deux composantes ; modèle complet :
    4 + 4. F = [(RSS_c − RSS_s)/4] / [RSS_s/(m − 8)] sur les sommes de carrés
    RELATIVES de ``_regress_structure`` (comparables d'un modèle à l'autre) ;
    σ_r = σ_j est rejetée si p < alpha. Test approché (poids itératifs, contraintes
    de positivité), suffisant pour décider s'il faut deux structures.
    """
    coef_c, rel_c, rss_c, m = _regress_structure(Zr, Zj, R_sol, [s_re, s_im])
    coef_r, _, rss_r, m_r = _regress_structure(Zr, Zj, R_sol, [s_re])
    coef_j, _, rss_j, m_j = _regress_structure(Zr, Zj, R_sol, [s_im])
    rss_s = rss_r + rss_j
    df = m_r + m_j - 8
    if df > 0 and rss_s > 0 and np.isfinite(rss_c) and rss_c > rss_s:
        F = ((rss_c - rss_s) / 4.0) / (rss_s / df)
        p = float(stats.f.sf(F, 4, df))
    else:
        p = 1.0
    if p < alpha:
        rel = float(np.sqrt(rss_s / max(m_r + m_j, 1)))
        return coef_r, tuple(float(c) for c in coef_j), False, p, rel
    return coef_c, None, True, p, rel_c


def characterize_error_structure(
    replicates: Sequence,
    *,
    options: Optional[MeasurementModelOptions] = None,
    label: str = "",
) -> ErrorStructureEstimate:
    """Caractérise la structure d'erreur à partir des résidus INTER-RÉPLICATS ([A95b]).

    Pour chaque réplicat k, un measurement model de Voigt M_k est régressé (``fit_voigt``) ;
    le résidu r_k(ω) = Z_k(ω) − M_k(ω) retire la part déterministe KK-conforme, Y
    COMPRIS une dérive lente d'un réplicat à l'autre (chaque réplicat a son propre
    modèle) — c'est ce qui distingue cette estimation de l'écart-type brut des Z,
    qui compte la dérive comme du bruit. Puis, par fréquence et par composante :

        σ̂(ω) = s_k[ r_k(ω) / √(1 − h_k(ω)) ] / c4(n)

    * s_k : écart-type sur les réplicats (ddof = 1) — le centrage retire la part de
      résidu COMMUNE à tous les réplicats (défaut d'ajustement systématique) ;
    * √(1 − h) : un modèle ajusté absorbe une fraction h_ii (levier) de la variance
      du bruit ; sans correction σ serait sous-estimé d'environ √(1 − P/2N)
      (résidus studentisés, [CW]). Les points de levier > 0,95 sont écartés ;
    * c4(n) = E[s]/σ : s est biaisé bas à petit n (−11 % pour n = 3).

    Pas de plancher (l'ancien ``_compute_sigma`` imposait 0,1 % de |Z|, AUDIT.md
    §2.4). σ̂ est ensuite régressé sur la forme de ``ErrorStructure``
    (``_regress_structure``), avec |Zj|, |Zr| et |Z|² pris sur la MOYENNE des
    measurement models (régresseurs sans bruit), R_sol = moyenne des R0 ; σ_r = σ_j
    est testée (``_choose_structure``) et n'est imposée que si le test ne la rejette
    pas. Mesuré : avec un bruit qui la viole (σ_r ∝ |Zr|, σ_j ∝ |Zj|, bruit relatif
    par composante des jeux synthétiques du dépôt), l'imposer sous-estime σ_r là
    où |Zr| ≫ |Zj| (haute fréquence) et le test KK rejette à tort 2 groupes sur 4.

    Passes ([A95b]) : la première pondère les measurement models par |Z| (σ ∝ |Z|,
    à un facteur près, faute de mieux) ; les suivantes par la structure obtenue
    (σ absolu). Arrêt quand σ(ω) varie de moins de 2 %.

    Degrés de liberté effectifs (pour l'intervalle de χ²_ν du fit Orazem) :
    ν_σ = (n − 1)·(n_obs − P̄) − q, n_obs = 2 × fréquences communes, P̄ = nombre moyen
    de paramètres des measurement models, q = coefficients régressés (4 ou 8).

    Args:
        replicates: EISSpectrum (ou objets munis de f, Zre, Zim, label), même condition.
        options: MeasurementModelOptions.
        label: nom du groupe (messages).

    Returns:
        ErrorStructureEstimate.

    Raises:
        ErrorStructureUnavailable: moins de ``min_replicates`` réplicats, grilles
            incompatibles, valeurs non finies, measurement model non ajustable, ou
            dispersion nulle (réplicats identiques : fichier dupliqué ?).
    """
    opt = options or MeasurementModelOptions()
    reps = list(replicates)
    n = len(reps)
    if n < opt.min_replicates:
        raise ErrorStructureUnavailable(
            f"{n} réplicat(s) fourni(s), {opt.min_replicates} requis au minimum "
            f"(écart-type inter-réplicats à n − 1 degrés de liberté)",
            group_label=label, n_replicates=n,
        )
    f_c, idx = common_frequency_grid(reps, opt.freq_rtol, label)
    omega = 2.0 * np.pi * f_c
    data = [(np.asarray(sp.Zre, float)[i], np.asarray(sp.Zim, float)[i]) for sp, i in zip(reps, idx)]

    es: Optional[ErrorStructure] = None
    history = []
    sigma_prev = None
    previous_models = [None] * n
    for p in range(max(1, opt.passes)):
        models, res_re, res_im = [], [], []
        for k, ((zr, zj), sp) in enumerate(zip(data, reps)):
            if es is None:
                sd = np.abs(zr - 1j * zj)            # pondération « modulus » (relative)
                absolute = False
                sd_re = sd_im = sd
            else:
                sd_re, sd_im = es.sigmas(zr, zj)
                absolute = True
            try:
                prev = previous_models[k]
                vm = fit_voigt(f_c, zr, zj, sd_re, sd_im, absolute_sigma=absolute, options=opt,
                               start_tau=None if prev is None else prev.tau)
            except RuntimeError as exc:
                raise ErrorStructureUnavailable(
                    f"le measurement model de Voigt ne s'ajuste pas au réplicat "
                    f"« {getattr(sp, 'label', '?')} » ({exc})",
                    group_label=label, n_replicates=n,
                    remedy="Vérifiez ce fichier (spectre tronqué, bruité ou non stationnaire).",
                ) from None
            Zm = vm.impedance(omega)
            rr = zr - Zm.real
            ri = -zj - Zm.imag                         # résidu sur Im physique
            if opt.leverage_correction:
                N = len(f_c)
                h_re, h_im = vm.leverage[:N], vm.leverage[N:]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rr = np.where(h_re < 0.95, rr / np.sqrt(1.0 - h_re), np.nan)
                    ri = np.where(h_im < 0.95, ri / np.sqrt(1.0 - h_im), np.nan)
            models.append(vm)
            res_re.append(rr)
            res_im.append(ri)
        res_re = np.asarray(res_re)
        res_im = np.asarray(res_im)
        s_re, n_re = _std_over_replicates(res_re)
        s_im, n_im = _std_over_replicates(res_im)
        s_re, s_im = s_re / _c4(n_re), s_im / _c4(n_im)

        Zbar = np.mean([vm.impedance(omega) for vm in models], axis=0)
        Zr, Zj = Zbar.real, -Zbar.imag
        R_sol = float(np.mean([vm.R0 for vm in models]))
        coef, coef_im, equal, p_eq, rel = _choose_structure(Zr, Zj, R_sol, s_re, s_im)
        P_mean = float(np.mean([len(vm.x) for vm in models]))
        dof = int(max(1, round((n - 1) * (2 * len(f_c) - P_mean) - (4 if equal else 8))))
        es = ErrorStructure(
            alpha=float(coef[0]), beta=float(coef[1]), gamma=float(coef[2]), delta=float(coef[3]),
            R_sol=R_sol, n_replicates=n, dof=dof, label=label,
            characterized_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            equal_re_im=equal, im=coef_im, equality_pvalue=p_eq,
        )
        sm_re, sm_im = es.sigmas(Zr, Zj)
        sigma_model = np.concatenate([sm_re, sm_im])
        modulus = np.abs(np.concatenate([Zbar, Zbar]))
        history.append(dict(pass_index=p + 1, weighting="modulus" if p == 0 else "error_structure",
                            n_elements=[vm.n_elements for vm in models], rel_rms=rel,
                            equal_re_im=equal, equality_pvalue=p_eq, im=coef_im,
                            **{k: getattr(es, k) for k in ("alpha", "beta", "gamma", "delta", "R_sol")}))
        # σ doit être RÉSOLU au-dessus de l'arrondi numérique des measurement models
        # (~1e-13 |Z|) : 1e-9 |Z| est très en deçà de la résolution de tout
        # impédancemètre (~1e-4 |Z|). Critère de REFUS, pas un plancher appliqué à σ.
        if not (np.all(np.isfinite(sigma_model)) and np.all(sigma_model > _SIGMA_RESOLUTION * modulus)):
            raise ErrorStructureUnavailable(
                "dispersion inter-réplicats nulle ou non résolue (σ(ω) ≤ 1e-9·|Z| à certaines "
                "fréquences) — des réplicats identiques (fichier dupliqué ?) ne renseignent pas le bruit",
                group_label=label, n_replicates=n,
                remedy="Vérifiez que les réplicats sont des acquisitions distinctes.",
            )
        if sigma_prev is not None and np.max(np.abs(sigma_model / sigma_prev - 1.0)) < 0.02:
            break
        sigma_prev = sigma_model
        previous_models = models

    return ErrorStructureEstimate(
        error_structure=es, frequencies=f_c, sigma_emp_re=s_re, sigma_emp_im=s_im,
        sigma_model_re=sm_re, sigma_model_im=sm_im, voigt_models=models, passes=history, rel_rms=rel,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Conformité Kramers-Kronig PAR LE MÊME measurement model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class KKConsistency:
    """Test KK d'un spectre par measurement model ([A95c]).

    Attributes:
        verdict: KKVerdict (critère unique de ``fits.kk_validation``), sur la partie
            RÉELLE prédite.
        frequencies: Hz (ordre croissant).
        residual_re: Zr_données − Zr_prédit depuis l'ajustement de la seule partie
            imaginaire (Ω).
        sd_re: écart-type de ce résidu sous H0 (bruit + incertitude de prédiction).
        sigma_re, sigma_im: σ de la donnée (bruit seul, Ω) par composante.
        model_from_imag: measurement model ajusté sur Im seule.
        n_averaged: mesures moyennées dans le spectre testé (σ/√n).
        Zre, Zim: données testées (fréquences croissantes, convention de l'app).
    """

    verdict: KKVerdict
    frequencies: np.ndarray
    residual_re: np.ndarray
    sd_re: np.ndarray
    sigma_re: np.ndarray
    sigma_im: np.ndarray
    model_from_imag: VoigtModel
    n_averaged: int
    Zre: np.ndarray
    Zim: np.ndarray
    label: str = ""

    @property
    def conform(self) -> Optional[bool]:
        return self.verdict.conform

    @property
    def residual_im_fit(self) -> np.ndarray:
        """Résidu de l'ajustement de Im (données − modèle, convention de l'app, Ω)."""
        return self.Zim + self.model_from_imag.impedance(2.0 * np.pi * self.frequencies).imag


def _predict_real_part(model: VoigtModel, omega, Zre, sd):
    """Prédit Re(Z) depuis un measurement model ajusté sur Im(Z) seule.

    Im(Z) ne contient pas R0 : la prédiction est pred = V_re(θ̂) + R0, avec R0 ajusté
    par moindres carrés pondérés sur Re (R0 = Σ w (Zre − V_re) / Σ w). Une constante
    ne peut pas masquer une violation dépendant de la fréquence.

    Variance du résidu d = Zre − pred sous H0 (bruit indépendant entre Re et Im,
    hypothèse du measurement model [A95b]) :
        Var(d_i) = σ_i² − 1/Σw + g_iᵀ C g_i,
        g_i = ∂V_re,i/∂θ − Σ_j w_j ∂V_re,j/∂θ / Σw        (méthode delta),
    C = covariance de θ issue de l'ajustement de Im ([A95c] obtient ces intervalles
    par Monte-Carlo sur la même covariance ; la méthode delta en est la limite
    linéarisée, déterministe).
    """
    dZ = _voigt_jacobian(model, omega)
    V = model.impedance(omega).real
    G = dZ.real
    w = 1.0 / sd ** 2
    S = float(np.sum(w))
    d = Zre - V
    d = d - float(np.sum(w * d)) / S
    G = G - (w @ G)[None, :] / S
    var = sd ** 2 - 1.0 / S
    if np.all(np.isfinite(model.cov)):
        var = var + np.einsum("ij,jk,ik->i", G, model.cov, G)
    else:                                     # paramètre non identifiable : aucune prédiction
        var = np.full_like(var, np.inf)
    return d, np.sqrt(var)


def check_kk_consistency(
    f: np.ndarray,
    Zre: np.ndarray,
    Zim: np.ndarray,
    error_structure: ErrorStructure,
    *,
    n_averaged: int = 1,
    reference: Optional[VoigtModel] = None,
    start_tau: Optional[np.ndarray] = None,
    alpha: float = KK_FALSE_ALARM,
    options: Optional[MeasurementModelOptions] = None,
    label: str = "",
) -> KKConsistency:
    """Conformité KK par measurement model, pondéré par la structure d'erreur ([A95c]).

    Lien explicite avec la structure d'erreur : c'est le MÊME modèle de Voigt qui a
    servi à estimer σ (même régression, même critère de parcimonie ; pour un
    réplicat, ``reference`` est littéralement son measurement model de la dernière
    passe) qui fixe ici le nombre d'éléments K, et c'est σ qui pondère le test. Le
    modèle de Voigt vérifie les relations KK PAR CONSTRUCTION : si, ajusté sur la
    seule partie imaginaire, il prédit la partie réelle au niveau du bruit, les
    données sont KK-conformes ; sinon non.

    Procédure ([A95c]) :
      1. K et τ initiaux = ceux du measurement model complexe du spectre
         (``reference``, sinon régressé ici sur Re + Im pondérés par σ) ;
      2. ajuster la partie IMAGINAIRE seule à K éléments (et à K + 1, voir 4) ;
      3. prédire la partie réelle, à la constante R0 près (``_predict_real_part``) ;
      4. écart-type de prédiction INCONDITIONNEL au choix de K (voir ci-dessous) ;
      5. verdict par le critère UNIQUE ``fits.kk_validation.kk_verdict``.

    Deux choix, mesurés sur des Randles synthétiques KK-conformes (bruit selon
    [O04], 3 réplicats, σ estimée, 90 tests, risque nominal 5 %) :

    * Im → Re seulement : procédure principale de [A95c]. Le sens Re → Im s'est
      révélé moins fiable ici : Re seule résout moins d'éléments (K = 5 au lieu de
      6-7) et l'Im prédite manque la relaxation de contournement haute fréquence,
      que Re ne voit presque pas (z ≈ 3 à 25-100 kHz, fausses alarmes).
    * Incertitude sur K : avec la seule covariance linéarisée du modèle retenu, les
      résidus normalisés ont un écart-type de 1,04-1,08 (et non 1) et le test rejette
      10-13 % des spectres conformes au lieu de 5 % — la variance de prédiction
      ignore que K a été choisi sur les mêmes données, et la troncature du Voigt
      (un Randles à CPE n'est pas une somme FINIE de R//C) biaise localement la
      prédiction jusqu'à ±0,5σ. On moyenne donc les prédictions des modèles à K et
      K + 1 éléments, pondérées par leurs poids d'Akaike w_m ∝ exp(−ΔAIC_m/2),
      AIC_m = χ²_m + 2P_m (σ connue), et on prend la variance inconditionnelle de
      Buckland, Burnham & Augustin (Biometrics 53 (1997) 603-618 ; Burnham &
      Anderson, *Model Selection and Multimodel Inference*, 2e éd., Springer, 2002,
      §4.3.2) :  s_u = Σ_m w_m √(s_m² + (p_m − p̄)²).
      Mesuré : écart-type des résidus normalisés 0,94, 4 rejets sur 90 (4,4 %) ;
      prix payé : puissance un peu moindre (dérive de Rct de 5 % pendant le
      balayage détectée 4 fois sur 15 au lieu de 5-6 ; 10 % : toujours détectée).
      Le modèle à K + 1 n'entre pas dans la moyenne s'il ne converge pas ou si sa
      variance de prédiction n'est pas numériquement valide (paire de τ quasi
      confondus aux R opposés, non identifiable).

    Args:
        f, Zre, Zim: spectre (Zim convention de l'app, −Im(Z) > 0), fréquences > 0.
        error_structure: structure d'erreur caractérisée sur ce groupe.
        n_averaged: nombre de réplicats moyennés dans ce spectre (σ/√n).
        reference: measurement model complexe de CE spectre, pondéré par cette
            structure (en général ``ErrorStructureEstimate.voigt_models[k]``).
        start_tau: sans ``reference``, démarrage à chaud de sa régression.
        alpha: risque de fausse alarme nominal du verdict.
        options: MeasurementModelOptions.
        label: libellé (messages).

    Returns:
        KKConsistency.
    """
    opt = options or MeasurementModelOptions()
    f = np.asarray(f, dtype=float)
    order = np.argsort(f)
    f = f[order]
    Zre = np.asarray(Zre, dtype=float)[order]
    Zim = np.asarray(Zim, dtype=float)[order]
    omega = 2.0 * np.pi * f
    sd_re, sd_im = error_structure.sigmas(Zre, Zim)
    root_n = np.sqrt(max(int(n_averaged), 1))
    sd_re, sd_im = sd_re / root_n, sd_im / root_n

    if reference is None:
        reference = fit_voigt(f, Zre, Zim, sd_re, sd_im, absolute_sigma=True, options=opt,
                              start_tau=start_tau)
    reg = _VoigtRegression(f, Zre, Zim, sd_re, sd_im, absolute_sigma=True, component="imag",
                           options=opt)
    fitted = reg.refine(np.sort(reference.tau))
    m_K = reg.to_model(*fitted) if fitted is not None else reg.successive()

    candidates = [m_K]
    nxt = reg.add_element(m_K.tau) if m_K.n_elements < reg.k_cap else None
    if nxt is not None:
        candidates.append(reg.to_model(*nxt))
    preds, sds, aics = [], [], []
    for m in candidates:
        with np.errstate(invalid="ignore"):
            d, sd = _predict_real_part(m, omega, Zre, sd_re)
        if m is not m_K and not np.all(np.isfinite(sd)):
            continue                                   # K + 1 numériquement invalide
        preds.append(Zre - d)
        sds.append(sd)
        aics.append(m.chi2 + 2.0 * len(m.x))
    aics = np.asarray(aics)
    w = np.exp(-(aics - aics.min()) / 2.0)
    w = w / w.sum()
    p_bar = sum(wi * p for wi, p in zip(w, preds))
    sd_u = sum(wi * np.sqrt(s ** 2 + (p - p_bar) ** 2) for wi, p, s in zip(w, preds, sds))
    d_re = Zre - p_bar

    verdict = kk_verdict(d_re, None, sd_u, None, alpha=alpha)
    verdict.extra.update(n_elements=[m.n_elements for m in candidates[: len(preds)]],
                         akaike_weights=w.tolist())
    return KKConsistency(
        verdict=verdict, frequencies=f, residual_re=d_re, sd_re=sd_u, sigma_re=sd_re, sigma_im=sd_im,
        model_from_imag=m_K, n_averaged=int(n_averaged), Zre=Zre, Zim=Zim, label=label,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration d'un groupe : structure d'erreur PUIS verdict KK
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MeasurementModelAnalysis:
    """Analyse préalable d'un groupe, à afficher AVANT de proposer le fit Orazem.

    ``fits.orazem_fit.fit_replicate_group`` exige cet objet en entrée : le fit ne
    peut pas être lancé sans que la structure d'erreur ET le verdict KK aient été
    calculés (ordre imposé par construction, pas par convention).

    Attributes:
        label: groupe.
        estimate: ErrorStructureEstimate (structure + σ empiriques + measurement models).
        frequencies: grille commune (Hz, croissante).
        mean_Zre, mean_Zim: spectre moyen sur la grille commune (convention app).
        kk_mean: test KK du spectre moyen (n_averaged = n).
        kk_replicates: test KK de chaque réplicat (n_averaged = 1).
        kk_conform: verdict du GROUPE (tous les tests conformes).
        alpha_family, alpha_each: risque global et risque par test (Bonferroni).
        kk_message: phrase de verdict pour l'utilisateur.
    """

    label: str
    estimate: ErrorStructureEstimate
    frequencies: np.ndarray
    mean_Zre: np.ndarray
    mean_Zim: np.ndarray
    replicate_labels: list
    kk_mean: KKConsistency
    kk_replicates: list
    kk_conform: bool
    alpha_family: float
    alpha_each: float
    kk_message: str

    @property
    def error_structure(self) -> ErrorStructure:
        return self.estimate.error_structure

    @property
    def n_replicates(self) -> int:
        return len(self.kk_replicates)


def analyze_replicates(
    replicates: Sequence,
    *,
    options: Optional[MeasurementModelOptions] = None,
    label: str = "",
    alpha: float = KK_FALSE_ALARM,
) -> MeasurementModelAnalysis:
    """Structure d'erreur puis verdict KK d'un groupe de réplicats.

    Verdict du groupe : les n réplicats ET leur moyenne sont testés (n + 1 tests).
    La moyenne est le test le plus sensible aux violations COMMUNES (bruit ÷ √n ;
    les transformées KK étant linéaires, la moyenne de spectres conformes est
    conforme) ; chaque réplicat, le plus sensible à une violation qui lui est
    propre (dérive pendant un balayage). Le groupe est conforme si les n + 1 tests
    le sont, chacun au risque alpha/(n + 1) (Bonferroni) : le risque de fausse
    alarme du VERDICT DE GROUPE reste ≤ alpha.

    Calibrage mesuré (Randles de l'Annexe A, bruit d'Orazem, 100 groupes conformes
    par configuration, alpha = 5 %) — fausses alarmes : 3 réplicats × 40 points
    4/100 ; 3 × 60 points 1/100 ; 5 × 40 points 0/100. Puissance (1 réplicat sur 3
    dont Rct dérive PENDANT son balayage, 40 groupes) : +5 % → 15/40, +10 % → 32/40,
    +20 % → 40/40. σ_r = σ_j retenue dans 94 à 96 groupes sur 100 (bruit conforme).

    Raises:
        ErrorStructureUnavailable: voir ``characterize_error_structure`` — l'analyse
            du groupe s'arrête ; aucun verdict ni fit n'est produit.
    """
    opt = options or MeasurementModelOptions()
    reps = list(replicates)
    est = characterize_error_structure(reps, options=opt, label=label)
    es = est.error_structure
    f_c, idx = common_frequency_grid(reps, opt.freq_rtol, label)
    zr = np.asarray([np.asarray(sp.Zre, float)[i] for sp, i in zip(reps, idx)])
    zj = np.asarray([np.asarray(sp.Zim, float)[i] for sp, i in zip(reps, idx)])
    n = len(reps)
    a_each = alpha / (n + 1)
    kk_mean = check_kk_consistency(f_c, zr.mean(axis=0), zj.mean(axis=0), es, n_averaged=n,
                                   start_tau=est.voigt_models[0].tau, alpha=a_each, options=opt,
                                   label=f"{label} (moyenne)")
    kk_reps = [
        check_kk_consistency(f_c, zr[k], zj[k], es, n_averaged=1, reference=est.voigt_models[k],
                             alpha=a_each, options=opt,
                             label=getattr(sp, "label", f"réplicat {k + 1}"))
        for k, sp in enumerate(reps)
    ]
    conform = bool(kk_mean.conform and all(k.conform for k in kk_reps))
    failed = [k.label for k in [kk_mean] + kk_reps if not k.conform]
    if conform:
        msg = (f"Groupe « {label} » : conforme Kramers-Kronig ({n} réplicats et leur moyenne, "
               f"risque global {alpha:.0%}). Le fit Orazem peut être proposé.")
    else:
        msg = (f"Groupe « {label} » : NON conforme Kramers-Kronig ({', '.join(failed)}). "
               f"Les paramètres d'un fit sur ces données seraient à interpréter avec prudence "
               f"(non-stationnarité, non-linéarité ou artefact instrumental).")
    return MeasurementModelAnalysis(
        label=label, estimate=est, frequencies=f_c, mean_Zre=zr.mean(axis=0),
        mean_Zim=zj.mean(axis=0), replicate_labels=[getattr(sp, "label", "") for sp in reps],
        kk_mean=kk_mean, kk_replicates=kk_reps, kk_conform=conform,
        alpha_family=alpha, alpha_each=a_each, kk_message=msg,
    )
