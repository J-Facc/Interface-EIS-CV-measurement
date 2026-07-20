"""Structure d'erreur d'Orazem (Voigt measurement model) — pondération UNIQUE.

Ce module est le cœur de la méthode de pondération du fit paramétrique de Randles.
Il n'existe plus qu'UNE seule façon de pondérer le CNLS complexe : par la
structure d'erreur stochastique de l'instrument, estimée à partir de réplicats
selon la méthode d'Orazem & Tribollet.

Forme de l'écart-type stochastique par fréquence (Agarwal, Orazem & García-Rubio,
J. Electrochem. Soc. 1992-1995 ; Orazem & Tribollet, "Electrochemical Impedance
Spectroscopy", Wiley, chap. Measurement Model / Error Structure) :

    σ_i = α·|Z_re,i| + β·|Z_im,i| + γ·(|Z_i|²/R_m) + δ

Les poids du fit valent alors w_re,i = w_im,i = 1/σ_i² (absolute_sigma=True), si
bien que χ²_red devient un vrai test d'adéquation.

MÉTHODE (stricte)
    1. À partir de réplicats (≥ 3 recommandé), estimer l'écart-type stochastique
       σ_empirique(f). Deux voies :
         - « réplicats-directs » (défaut) : écart-type inter-réplicats par
           fréquence (core/loader.average_replicates, ddof=1) ;
         - « voigt_based » (option) : ajuster un circuit de Voigt (Lin-KK,
           fits/kk_validation.lin_kk) à chaque réplicat et prendre l'écart-type
           des résidus par fréquence — plus proche d'Orazem (le Voigt absorbe la
           partie déterministe reproductible, le résidu isole le bruit).
    2. Régresser σ_empirique(f) sur la forme structurée ci-dessus (moindres
       carrés NON NÉGATIFS sur les coefficients ; α = β imposé si equal_re_im).
    3. PERSISTER (α, β, γ, δ, R_m) + horodatage + identifiant de campagne.

REPLI (spectre sans réplicats)
    - On NE ré-estime PAS. On recharge les DERNIERS coefficients persistés et on
      s'en sert pour pondérer le fit (source = "reused_persisted").
    - Si rien n'a jamais été caractérisé : on REFUSE le fit
      (ErrorStructureUnavailable). Aucun repli sur un σ = α·|Z| arbitraire.

Aucune dépendance Streamlit (importable depuis core/ et fits/).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import nnls

from core.logger import get_logger

log = get_logger("error_structure")

# Chemin de persistance par défaut (surchargée par fit.error_structure.persistence_path).
DEFAULT_PERSIST_PATH = Path(__file__).parent.parent / "config" / "error_structure.json"

# Nombre de réplicats recommandé pour caractériser la structure d'erreur.
_MIN_REPLICATES_DEFAULT = 3


class ErrorStructureUnavailable(RuntimeError):
    """Levée quand aucune structure d'erreur n'est disponible (ni réplicats, ni
    coefficients persistés). Le fit paramétrique doit alors être refusé."""


@dataclass
class ErrorStructure:
    """Coefficients de la structure d'erreur d'Orazem + provenance.

    NE PAS confondre avec les paramètres de Randles : ces coefficients décrivent
    le BRUIT σ(|Z|) et servent uniquement à PONDÉRER le fit. Ce sont EUX (et non
    les paramètres de Randles) qui sont persistés et réutilisés.
    """

    alpha: float                 # coeff |Z_re|
    beta: float                  # coeff |Z_im|
    gamma: float                 # coeff |Z|²/R_m
    delta: float                 # bruit de fond additif (Ω)
    R_m: Optional[float] = None  # résistance de mesure (Ω) ; None => terme γ absorbé/ignoré
    equal_re_im: bool = True     # α = β imposé (hypothèse standard du measurement model)
    voigt_based: bool = False    # σ empirique obtenu par résidus de Voigt (sinon inter-réplicats)
    timestamp: str = ""          # horodatage ISO 8601 de la caractérisation
    campaign_id: str = ""        # identifiant de campagne (labels des réplicats)
    n_replicates: int = 0        # nombre de réplicats utilisés
    quality: float = float("nan")  # RMS relatif de la régression σ_struct vs σ_emp
    source: str = "characterized_now"  # "characterized_now" | "reused_persisted"

    def sigma(self, Zre: np.ndarray, Zim: np.ndarray) -> np.ndarray:
        """σ(f) structuré pour un spectre (convention Zim = -Im(Z) > 0).

        Garde-fous documentés :
        - R_m nul/None : le terme γ·|Z|²/R_m n'est pas identifiable sans R_m et
          est IGNORÉ (le facteur 1/R_m est de toute façon absorbé dans γ lors de
          l'estimation « fitted »). Aucune division par zéro possible.
        - |Z| → 0 : le terme en |Z|² tend vers 0 sans division dangereuse ; δ ≥ 0
          borne σ par le bas (σ ≥ δ), sans plancher relatif arbitraire.
        """
        Zre = np.asarray(Zre, dtype=float)
        Zim = np.asarray(Zim, dtype=float)
        sigma = self.alpha * np.abs(Zre) + self.beta * np.abs(Zim) + self.delta
        if self.gamma > 0 and self.R_m is not None and self.R_m > 0:
            sigma = sigma + self.gamma * (Zre ** 2 + Zim ** 2) / self.R_m
        return sigma

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorStructure":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


# ── Estimation de σ empirique à partir de réplicats ──────────────────────────

def _sigma_empirical_direct(spectra: list) -> tuple:
    """σ inter-réplicats par fréquence (ddof=1) — voie « réplicats-directs ».

    Returns (Zre_mean, Zim_mean, sigma_re, sigma_im).
    """
    f_ref = spectra[0].f
    Zre_stack, Zim_stack = [], []
    for sp in spectra:
        if len(sp.f) == len(f_ref) and np.allclose(sp.f, f_ref, rtol=0.01):
            Zre_stack.append(np.asarray(sp.Zre, dtype=float))
            Zim_stack.append(np.asarray(sp.Zim, dtype=float))
        else:  # ré-échantillonne sur la grille de référence
            idx = np.argsort(sp.f)
            Zre_stack.append(np.interp(f_ref, sp.f[idx], np.asarray(sp.Zre)[idx]))
            Zim_stack.append(np.interp(f_ref, sp.f[idx], np.asarray(sp.Zim)[idx]))
    Zre_stack = np.asarray(Zre_stack)
    Zim_stack = np.asarray(Zim_stack)
    return (
        Zre_stack.mean(axis=0),
        Zim_stack.mean(axis=0),
        Zre_stack.std(axis=0, ddof=1),
        Zim_stack.std(axis=0, ddof=1),
    )


def _sigma_empirical_voigt(spectra: list) -> tuple:
    """σ par écart-type des résidus de Voigt — voie « voigt_based » (Orazem).

    Un circuit de Voigt (Lin-KK) est ajusté à CHAQUE réplicat ; le résidu
    (Z − Z_Voigt) isole la part stochastique (le Voigt reproduit la part
    déterministe KK-consistante). σ_re/σ_im = écart-type des résidus par
    fréquence entre réplicats (ddof=1).

    Returns (Zre_mean, Zim_mean, sigma_re, sigma_im).
    """
    from fits.kk_validation import lin_kk  # import local (évite cycle au chargement)

    f_ref = np.asarray(spectra[0].f, dtype=float)
    order = np.argsort(f_ref)  # lin_kk attend f croissant
    inv = np.argsort(order)
    Zre_stack, Zim_stack, resR, resI = [], [], [], []
    for sp in spectra:
        # ré-échantillonne sur la grille de référence si nécessaire
        if len(sp.f) == len(f_ref) and np.allclose(sp.f, f_ref, rtol=0.01):
            Zre = np.asarray(sp.Zre, dtype=float)
            Zim = np.asarray(sp.Zim, dtype=float)
        else:
            idx = np.argsort(sp.f)
            Zre = np.interp(f_ref, sp.f[idx], np.asarray(sp.Zre)[idx])
            Zim = np.interp(f_ref, sp.f[idx], np.asarray(sp.Zim)[idx])
        Zre_stack.append(Zre)
        Zim_stack.append(Zim)
        # convention lin_kk : Im(Z) < 0 (dissipatif). Zim est en -Im(Z) > 0.
        Z = Zre[order] - 1j * np.abs(Zim[order])
        _M, _mu, _Zfit, rre, rim = lin_kk(f_ref[order], Z)
        resR.append(rre[inv])
        resI.append(rim[inv])
    Zre_stack = np.asarray(Zre_stack)
    Zim_stack = np.asarray(Zim_stack)
    resR = np.asarray(resR)
    resI = np.asarray(resI)
    return (
        Zre_stack.mean(axis=0),
        Zim_stack.mean(axis=0),
        resR.std(axis=0, ddof=1),
        resI.std(axis=0, ddof=1),
    )


# ── Régression des coefficients (α, β, γ, δ) ─────────────────────────────────

def regress_coefficients(Zre, Zim, sigma_re, sigma_im, equal_re_im=True, R_m=None):
    """Régresse σ_empirique(f) sur α·|Z_re| + β·|Z_im| + γ·|Z|²/R_m + δ.

    Moindres carrés NON NÉGATIFS (scipy.optimize.nnls) : les coefficients sont
    des amplitudes de bruit → contraints ≥ 0. La forme est linéaire en
    (α, β, γ_eff, δ), avec γ_eff = γ/R_m absorbant 1/R_m (R_m n'a donc pas à être
    connu pour l'estimation). Si equal_re_im, α = β (colonne |Z_re|+|Z_im| unique).

    Returns:
        (alpha, beta, gamma, delta, quality) où quality = RMS relatif
        ‖σ_struct − σ_emp‖ / ‖σ_emp‖ sur les cibles empilées.
    """
    Zre = np.asarray(Zre, dtype=float)
    Zim = np.asarray(Zim, dtype=float)
    sigma_re = np.asarray(sigma_re, dtype=float)
    sigma_im = np.asarray(sigma_im, dtype=float)
    aZre, aZim = np.abs(Zre), np.abs(Zim)
    Zmod2 = Zre ** 2 + Zim ** 2
    # γ_eff = γ/R_m : colonne |Z|² (R_m absorbé). Si R_m fourni, on le ré-extrait.
    R_m_eff = float(R_m) if (R_m is not None and R_m > 0) else 1.0
    ones = np.ones_like(aZre)

    # cible empilée [σ_re ; σ_im] : la structure d'erreur est la même sur re/im
    # sous l'hypothèse d'égalité des variances (equal_re_im).
    y = np.concatenate([sigma_re, sigma_im])

    if equal_re_im:
        # α = β : une seule colonne (|Z_re| + |Z_im|).
        col_ab = np.concatenate([aZre + aZim, aZre + aZim])
        col_g = np.concatenate([Zmod2, Zmod2]) / R_m_eff
        col_d = np.concatenate([ones, ones])
        A = np.vstack([col_ab, col_g, col_d]).T
        coeffs, _ = nnls(A, y)
        a = b = float(coeffs[0])
        g_eff = float(coeffs[1])
        d = float(coeffs[2])
    else:
        col_a = np.concatenate([aZre, np.zeros_like(aZim)])
        col_b = np.concatenate([np.zeros_like(aZre), aZim])
        # NB: sans equal_re_im, la structure re/im diffère ; on garde une seule
        # amplitude γ et δ partagées (bruit de fond commun), α/β distincts.
        col_g = np.concatenate([Zmod2, Zmod2]) / R_m_eff
        col_d = np.concatenate([ones, ones])
        A = np.vstack([col_a, col_b, col_g, col_d]).T
        coeffs, _ = nnls(A, y)
        a = float(coeffs[0])
        b = float(coeffs[1])
        g_eff = float(coeffs[2])
        d = float(coeffs[3])

    # γ retourné : si R_m fourni, γ = γ_eff·R_m ; sinon γ = γ_eff (R_m absorbé,
    # sigma() ignorera le terme faute de R_m — cohérent).
    gamma = g_eff * R_m_eff if (R_m is not None and R_m > 0) else g_eff

    sigma_pred = A @ (coeffs)
    denom = float(np.linalg.norm(y)) or 1.0
    quality = float(np.linalg.norm(sigma_pred - y) / denom)
    return a, b, gamma, d, quality


# ── Caractérisation à partir de réplicats ────────────────────────────────────

def characterize_from_replicate_spectra(spectra: list, es_cfg: dict) -> ErrorStructure:
    """Caractérise la structure d'erreur à partir d'une liste de réplicats.

    Args:
        spectra: liste d'EISSpectrum (mêmes conditions), ≥ min_replicates.
        es_cfg: sous-dict config `fit.error_structure`.

    Returns:
        ErrorStructure (source="characterized_now").

    Raises:
        ErrorStructureUnavailable: si moins de min_replicates réplicats.
    """
    equal_re_im = bool(es_cfg.get("equal_re_im", True))
    voigt_based = bool(es_cfg.get("voigt_based", False))
    R_m = es_cfg.get("R_m", None)
    min_rep = int(es_cfg.get("min_replicates", _MIN_REPLICATES_DEFAULT))

    n = len(spectra)
    if n < min_rep:
        raise ErrorStructureUnavailable(
            f"{n} réplicat(s) < {min_rep} requis pour caractériser la structure "
            f"d'erreur."
        )

    if voigt_based:
        Zre_m, Zim_m, sre, sim = _sigma_empirical_voigt(spectra)
    else:
        Zre_m, Zim_m, sre, sim = _sigma_empirical_direct(spectra)

    a, b, g, d, quality = regress_coefficients(
        Zre_m, Zim_m, sre, sim, equal_re_im=equal_re_im, R_m=R_m
    )
    return ErrorStructure(
        alpha=a, beta=b, gamma=g, delta=d, R_m=R_m,
        equal_re_im=equal_re_im, voigt_based=voigt_based,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        campaign_id="+".join(getattr(sp, "label", "?") for sp in spectra)[:200],
        n_replicates=n, quality=quality, source="characterized_now",
    )


def characterize_from_averaged(spectrum, es_cfg: dict) -> Optional[ErrorStructure]:
    """Caractérise depuis un spectre moyenné portant sigma_re/sigma_im + n_replicates.

    Voie « réplicats-directs » : réutilise les σ déjà calculés par
    core/loader.average_replicates. Si le spectre porte la liste des réplicats
    (`replicates`) ET que voigt_based est demandé, délègue à la voie Voigt.

    Returns:
        ErrorStructure, ou None si le spectre n'a pas de σ (< 2 réplicats).
    """
    voigt_based = bool(es_cfg.get("voigt_based", False))
    reps = getattr(spectrum, "replicates", None)
    if voigt_based and reps:
        return characterize_from_replicate_spectra(reps, es_cfg)

    sigma_re = getattr(spectrum, "sigma_re", None)
    sigma_im = getattr(spectrum, "sigma_im", None)
    if sigma_re is None or sigma_im is None:
        return None

    n_rep = getattr(spectrum, "n_replicates", None) or len(
        getattr(spectrum, "source_files", []) or []
    )
    min_rep = int(es_cfg.get("min_replicates", _MIN_REPLICATES_DEFAULT))
    if n_rep < min_rep:
        raise ErrorStructureUnavailable(
            f"{n_rep} réplicat(s) < {min_rep} requis pour caractériser la "
            f"structure d'erreur."
        )

    equal_re_im = bool(es_cfg.get("equal_re_im", True))
    R_m = es_cfg.get("R_m", None)
    a, b, g, d, quality = regress_coefficients(
        spectrum.Zre, spectrum.Zim, sigma_re, sigma_im,
        equal_re_im=equal_re_im, R_m=R_m,
    )
    return ErrorStructure(
        alpha=a, beta=b, gamma=g, delta=d, R_m=R_m,
        equal_re_im=equal_re_im, voigt_based=False,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        campaign_id=getattr(spectrum, "label", "?")[:200],
        n_replicates=int(n_rep), quality=quality, source="characterized_now",
    )


# ── Persistance JSON ─────────────────────────────────────────────────────────

def _persist_path(config) -> Path:
    fit_cfg = config.get("fit", {}) if isinstance(config, dict) else {}
    es_cfg = fit_cfg.get("error_structure", {}) or {}
    p = es_cfg.get("persistence_path", None)
    return Path(p) if p else DEFAULT_PERSIST_PATH


def persist(struct: ErrorStructure, path: Path) -> None:
    """Ajoute la structure caractérisée à l'historique JSON (écriture atomique).

    Le fichier contient une LISTE d'entrées (historique des campagnes) ; la plus
    récente fait foi pour le repli. Écriture via fichier temporaire + os.replace
    pour éviter la corruption en cas d'interruption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = [history]
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"[error_structure] historique illisible ({e}) — réinitialisé.")
            history = []
    entry = struct.to_dict()
    entry["source"] = "characterized_now"  # normalise ce qui est écrit
    history.append(entry)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    log.info(
        f"[error_structure] coefficients persistés ({path}) : "
        f"α={struct.alpha:.3g} β={struct.beta:.3g} γ={struct.gamma:.3g} "
        f"δ={struct.delta:.3g} campagne='{struct.campaign_id}'."
    )


def load_latest(path: Path) -> Optional[ErrorStructure]:
    """Recharge la DERNIÈRE structure persistée (source forcée à reused_persisted)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"[error_structure] lecture persistance échouée ({e}).")
        return None
    if not history:
        return None
    if isinstance(history, dict):
        history = [history]
    struct = ErrorStructure.from_dict(history[-1])
    struct.source = "reused_persisted"
    return struct


# ── Point d'entrée : caractériser MAINTENANT, sinon réutiliser, sinon refuser ─

def resolve_error_structure(spectrum, config) -> ErrorStructure:
    """Résout la structure d'erreur pour un spectre donné.

    Politique (méthode stricte d'Orazem, hybride assumé) :
      1. Le spectre porte des réplicats (σ inter-réplicats + n_replicates ≥ seuil)
         → CARACTÉRISER maintenant et PERSISTER (source="characterized_now").
      2. Sinon → RÉUTILISER les derniers coefficients persistés
         (source="reused_persisted").
      3. Sinon (jamais caractérisé, aucun réplicat) → REFUSER
         (ErrorStructureUnavailable).

    Raises:
        ErrorStructureUnavailable: cas 3.
    """
    fit_cfg = config.get("fit", {}) if isinstance(config, dict) else {}
    es_cfg = fit_cfg.get("error_structure", {}) or {}
    path = _persist_path(config)

    # Cas 1 : caractérisation sur réplicats du spectre courant.
    struct = None
    try:
        struct = characterize_from_averaged(spectrum, es_cfg)
    except ErrorStructureUnavailable:
        # Pas assez de réplicats : on tentera le repli persisté ci-dessous.
        struct = None
    if struct is not None:
        persist(struct, path)
        return struct

    # Cas 2 : repli sur coefficients persistés.
    persisted = load_latest(path)
    if persisted is not None:
        log.info(
            f"[error_structure] spectre sans réplicats → réutilisation des "
            f"coefficients persistés (caractérisés le {persisted.timestamp}, "
            f"campagne='{persisted.campaign_id}')."
        )
        return persisted

    # Cas 3 : refus explicite.
    raise ErrorStructureUnavailable(
        "structure d'erreur non caractérisée : fournir au moins une campagne de "
        "réplicats (≥ 3) pour caractériser l'instrument, ou une caractérisation "
        "persistée antérieure. Aucun repli sur un σ arbitraire n'est effectué."
    )
