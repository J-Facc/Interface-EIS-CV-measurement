"""Jeux de données synthétiques partagés par les tests de caractérisation.

Reprend le style des scripts de l'Annexe A de AUDIT.md (spectres de type Randles
bruités, fichiers texte EC-Lab minimaux) pour que les tests du filet de sécurité
(``test_pipeline``, ``test_validator``, ``test_exporter_full``) parlent des mêmes
données que l'audit.

Ce module n'est PAS collecté par pytest (nom sans ``test_``) : il ne contient que
des fabriques, aucun test.

Convention de signe : ``Zim = -Im(Z) > 0`` (celle du loader et de ``EISSpectrum``).
"""

from __future__ import annotations

import numpy as np

from core.config import config_to_dict, load_config
from core.models import EISSpectrum, FitResult


def Z_randles_reference(omega, Re, Re_prime, Cb, Rct, Qdl, alpha, R_D, tau_d):
    """Randles complet en FORME FERMÉE — ORACLE de test, indépendant du code applicatif.

    Reprise littérale de l'ancien ``fits/physics.py:Z_randles_full`` (supprimé à
    l'étape 5 : le circuit est désormais défini par l'utilisateur). Les jeux
    synthétiques sont générés par cette formule écrite à la main, et non par
    ``circuit.parse_circuit`` : une erreur du parseur ne peut donc pas se masquer
    elle-même (``test_circuit_parser`` compare les deux à 1e-12).

        Z_D  = R_D·tanh(√(jωτ_d))/√(jωτ_d)
        Z_eq = R'e + (Rct + Z_D) / [1 + Qdl·(jω)^α·(Rct + Z_D)]
        Z    = Re + Z_eq / [1 + jω·Cb·Z_eq]
    """
    omega = np.asarray(omega, dtype=float)
    jw = 1j * omega
    x = np.sqrt(jw * tau_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        Zd = R_D * np.where(np.abs(x) < 1e-8, 1.0, np.tanh(x) / x)
    Z_eq = Re_prime + (Rct + Zd) / (1.0 + Qdl * (jw ** alpha) * (Rct + Zd))
    return Re + Z_eq / (1.0 + jw * Cb * Z_eq)


#: Expression de ce même circuit (défaut de ``fit.circuit``, docs/CIRCUIT_UTILISATEUR.md §3).
RANDLES_EXPRESSION = (
    "Re + parallel(Re_prime + parallel(R(Rct) + ZD_bounded(R_D, tau_d), Q(Qdl, alpha)), C(Cb))"
)

# 40 points : un run complet du pipeline (fits Randles de tous les réplicats
# compris) dure ~0,1 s. Avec 60 points (valeur de l'Annexe A), un réplicat bruité
# peut épuiser max_iter (9 s, converged=False) — trop lent et trop fragile pour un
# filet de sécurité. Les tests DRT réels gardent, eux, les 60 points de l'audit.
N_POINTS = 40
N_POINTS_AUDIT = 60

# Paramètres de l'Annexe A.2 (hors Rct, R_D = 0,3·Rct).
_RANDLES_FIXED = dict(Re=200.0, Re_prime=20.0, Cb=1e-9, Qdl=2e-6, alpha=0.9, tau_d=0.5)


def frequencies(n_points: int = N_POINTS) -> np.ndarray:
    """Grille HF → BF, 1e5 → 1e-1 Hz (ordre du loader)."""
    return np.logspace(5, -1, n_points)


def randles_impedance(Rct: float, n_points: int = N_POINTS) -> tuple:
    """Spectre de Randles sans bruit : (f, Z complexe, Im(Z) < 0)."""
    f = frequencies(n_points)
    Z = Z_randles_reference(2 * np.pi * f, Rct=Rct, R_D=0.3 * Rct, **_RANDLES_FIXED)
    return f, Z


def noisy_arrays(Rct: float, noise: float = 0.005, seed: int = 0,
                 n_points: int = N_POINTS) -> tuple:
    """(f, Zre, Zim) bruités (bruit gaussien relatif indépendant), Zim = -Im(Z) > 0."""
    f, Z = randles_impedance(Rct, n_points)
    rng = np.random.default_rng(seed)
    n = len(f)
    zre = Z.real * (1 + noise * rng.standard_normal(n))
    zim = (-Z.imag) * (1 + noise * rng.standard_normal(n))
    return f, zre, zim


# ── Bruit selon la structure d'erreur d'Orazem (σ_r = σ_j) ────────────────────
#
# ``noisy_arrays`` ajoute un bruit RELATIF indépendant par composante (σ_re ∝ |Zre|,
# σ_im ∝ |Zim|) : il VIOLE l'hypothèse σ_r = σ_j du measurement model (Agarwal et al.,
# JES 142 (1995) 4149). Les tests du measurement model et du fit Orazem utilisent
# plutôt ce générateur, conforme à la forme d'Orazem (JEC 572 (2004) 317) :
#     σ = α·|Zj| + β·|Zr − Re| + δ     (même σ sur Re et Im).

ORAZEM_NOISE = dict(alpha=0.004, beta=0.004, delta=0.5)


def orazem_sigma(zre, zim, alpha, beta, delta, R_sol=_RANDLES_FIXED["Re"]):
    """σ(ω) d'Orazem (même valeur sur Re et Im), convention Zim = −Im(Z) > 0."""
    return alpha * np.abs(zim) + beta * np.abs(zre - R_sol) + delta


def orazem_noisy_arrays(Rct: float, seed: int = 0, n_points: int = N_POINTS,
                        drift: float = 0.0, **noise) -> tuple:
    """(f, Zre, Zim, σ) : Randles de l'Annexe A bruité selon la structure d'Orazem.

    ``drift`` : variation relative de Rct PENDANT le balayage (HF → BF, temps ∝ rang
    du point) — spectre NON conforme Kramers-Kronig si drift ≠ 0 (non-stationnarité).
    """
    nz = dict(ORAZEM_NOISE, **noise)
    f = frequencies(n_points)
    if drift:
        Z = np.array([
            Z_randles_reference(2 * np.pi * f[i:i + 1], Rct=Rct * (1 + drift * i / (n_points - 1)),
                                R_D=0.3 * Rct, **_RANDLES_FIXED)[0]
            for i in range(n_points)
        ])
    else:
        Z = Z_randles_reference(2 * np.pi * f, Rct=Rct, R_D=0.3 * Rct, **_RANDLES_FIXED)
    zre, zim = Z.real, -Z.imag
    sigma = orazem_sigma(zre, zim, **nz)
    rng = np.random.default_rng(seed)
    return f, zre + rng.normal(0.0, sigma), zim + rng.normal(0.0, sigma), sigma


def orazem_replicates(Rct: float, n_rep: int = 3, seed0: int = 0, n_points: int = N_POINTS,
                      step: str = "probe", concentration: float = 0.0, **kw) -> list:
    """``n_rep`` EISSpectrum indépendants (bruit d'Orazem), labels ``r0``, ``r1``…"""
    out = []
    for k in range(n_rep):
        f, zre, zim, _ = orazem_noisy_arrays(Rct, seed0 + k, n_points, **kw)
        out.append(EISSpectrum(label=f"r{k}", f=f, Zre=zre, Zim=zim,
                               concentration=concentration, step=step, n_points=len(f)))
    return out


def eclab_bytes(f, zre, zim) -> bytes:
    """Sérialise en texte EC-Lab minimal (tabulations, ``-Im(Z)/Ohm`` positif)."""
    lines = ["freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm"]
    lines += [f"{a:.6g}\t{b:.6g}\t{c:.6g}" for a, b, c in zip(f, zre, zim)]
    return "\n".join(lines).encode()


def randles_file(Rct: float, noise: float = 0.005, seed: int = 0,
                 n_points: int = N_POINTS) -> bytes:
    """Fichier texte d'un spectre de Randles bruité, prêt pour ``load_spectrum``."""
    return eclab_bytes(*noisy_arrays(Rct, noise, seed, n_points))


def replicate_assignments(step: str, concentration: float, Rct: float, n_rep: int,
                          seed0: int, prefix: str, noise: float = 0.005,
                          n_points: int = N_POINTS) -> list:
    """``n_rep`` entrées de ``file_assignments`` (format de ``run_pipeline``)."""
    return [
        dict(
            content=randles_file(Rct, noise, seed0 + k, n_points),
            filename=f"{prefix}_r{k}.txt",
            step=step,
            concentration=concentration,
        )
        for k in range(n_rep)
    ]


def make_config(**drt) -> dict:
    """Config par défaut (``config/default.yaml``) ; ``drt`` surcharge ``fit.drt``
    (ex. ``make_config(enabled=False)``). Aucune structure d'erreur n'est plus
    persistée : rien n'est écrit sur disque par une analyse."""
    cfg = config_to_dict(load_config())
    cfg["fit"]["drt"].update(drt)
    return cfg


# ── Objets de modèle légers (exports, validator) ─────────────────────────────

def make_spectrum(label: str = "x", n: int = 40, concentration: float = 1e-9,
                  step: str = "hybridization") -> EISSpectrum:
    """Spectre plat minimal (Zre = Zim = 1) — le contenu n'importe pas aux exports."""
    return EISSpectrum(
        label=label, f=np.logspace(5, -1, n), Zre=np.ones(n), Zim=np.ones(n),
        concentration=concentration, step=step, n_points=n,
    )


def make_fit_result(model_name: str, params: dict, target_value: float, target_std: float = 0.0,
                    n: int = 40, **kwargs) -> FitResult:
    """FitResult minimal (tableaux de 1) avec les paramètres et le Rct voulus."""
    z = np.ones(n)
    return FitResult(
        model_name=model_name, params=params, params_std={}, Zfit_re=z, Zfit_im=z,
        chi2_reduced=1.0, residuals_re=z, residuals_im=z, target_param="Rct",
        target_value=target_value, target_std=target_std,
        converged=True, **kwargs,
    )
