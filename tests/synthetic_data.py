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
from fits.physics import Z_randles_full

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
    Z = Z_randles_full(2 * np.pi * f, Rct=Rct, R_D=0.3 * Rct, **_RANDLES_FIXED)
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


def make_config(persistence_path) -> dict:
    """Config par défaut, avec la structure d'erreur persistée dans ``persistence_path``.

    Sans cela, ``run_pipeline`` écrirait dans ``config/error_structure.json`` du dépôt.
    Passer un chemin INEXISTANT reproduit le « premier usage » (AUDIT.md ERR-1).
    """
    cfg = config_to_dict(load_config())
    cfg["fit"]["error_structure"]["persistence_path"] = str(persistence_path)
    return cfg


# ── Objets de modèle légers (exports, validator) ─────────────────────────────

def make_spectrum(label: str = "x", n: int = 40, concentration: float = 1e-9,
                  step: str = "hybridization") -> EISSpectrum:
    """Spectre plat minimal (Zre = Zim = 1) — le contenu n'importe pas aux exports."""
    return EISSpectrum(
        label=label, f=np.logspace(5, -1, n), Zre=np.ones(n), Zim=np.ones(n),
        concentration=concentration, step=step, n_points=n,
    )


def make_fit_result(model_name: str, params: dict, Rct: float, Rct_std: float = 0.0,
                    n: int = 40, **kwargs) -> FitResult:
    """FitResult minimal (tableaux de 1) avec les paramètres et le Rct voulus."""
    z = np.ones(n)
    return FitResult(
        model_name=model_name, params=params, params_std={}, Zfit_re=z, Zfit_im=z,
        chi2_reduced=1.0, residuals_re=z, residuals_im=z, Rct=Rct, Rct_std=Rct_std,
        converged=True, **kwargs,
    )
