# -*- coding: utf-8 -*-
"""
fits/drt_fit.py — Moteur DRT unique, wrapper de ``vendor.bayes_drt2``.

Deux méthodes, un seul paquet de calcul (bayes-drt2, Jake Huang, vendoré dans
``vendor/bayes_drt2/`` — voir ``vendor/README.md`` et ``THIRD_PARTY_LICENSES.md``) :

* :func:`drt_preview` — ``ridge_fit(freq, Z, hyper_lambda=True)`` : ridge
  hyperparamétrique, rapide, sans cmdstan. Sert d'**aperçu instantané** (affichage
  live de γ(τ) pour tous les spectres) et d'**initialiseur** du HMC.

* :func:`drt_bayes` — ``fit(freq, Z, mode='sample', init_from_ridge=True)`` : HMC,
  méthode de **référence** qui produit en plus des **intervalles de crédibilité**
  (γ_lo/γ_hi à 2.5/97.5 %, Rp_lo/Rp_hi). Nécessite ``cmdstanpy`` **et** une
  installation cmdstan.

Conventions d'impédance
-----------------------
Le loader stocke ``Zim = -Im(Z) > 0`` (demi-cercle capacitif au-dessus de l'axe
réel). bayes-drt2 attend la convention physique ``Z'' < 0`` :

    Z = spectrum.Zre - 1j * spectrum.Zim

Garde-fous (point 5 du cahier des charges)
------------------------------------------
* L'import de ``vendor.bayes_drt2`` est protégé par ``try/except ImportError`` :
  si l'extra DRT n'est pas installé (``cvxopt`` / ``cmdstanpy`` absents), la DRT
  est **désactivée proprement** (:func:`bayes_available` renvoie ``False``) au
  lieu de faire planter l'app.
* :func:`drt_bayes` teste ``cmdstanpy.cmdstan_path()`` **sans jamais compiler à la
  volée** : si cmdstan est absent, elle renvoie le résultat *ridge* assorti d'un
  statut « intervalles indisponibles : lancez setup_drt_bayesien.bat ». L'app
  reste fonctionnelle en aperçu ridge ; seul le HMC est bloqué.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import yaml

from core.logger import get_logger

log = get_logger("drt_fit")

# ── Import protégé du paquet vendoré (point 5) ──────────────────────────────
# vendor/bayes_drt2/inversion.py importe cvxopt ET cmdstanpy au niveau module :
# si l'un manque (extra DRT non installé), l'ImportError est capturée et la DRT
# est désactivée proprement, sans casser l'import de l'app.
try:
    from vendor.bayes_drt2.inversion import Inverter  # type: ignore
    _IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # ImportError (cvxopt/cmdstanpy) ou autre
    Inverter = None  # type: ignore
    _IMPORT_ERROR = str(exc)
    log.warning(f"Moteur DRT bayes-drt2 indisponible : {exc}")


DIST_NAME = "DRT"
_SETUP_HINT = "intervalles indisponibles : lancez setup_drt_bayesien.bat"

# Répertoire de persistance du cache HMC. `sessions/` est déjà .gitignore.
_CACHE_DIR = os.path.join("sessions", "drt_cache")

# Cache mémoire : {(key, engine): DRTResult}. Évite de relancer le ridge à
# chaque rerun Streamlit et sert d'index en amont du cache disque.
_MEM_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Résultat
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DRTResult:
    """Distribution des temps de relaxation γ(τ) et grandeurs dérivées.

    Attributes:
        tau: Grille des temps de relaxation τ (s).
        gamma: DRT γ(τ) — médiane a posteriori pour le HMC, solution ridge sinon.
        gamma_lo: Percentile 2.5 % de γ(τ) (HMC uniquement, sinon None).
        gamma_hi: Percentile 97.5 % de γ(τ) (HMC uniquement, sinon None).
        Rp: Résistance de polarisation (Ω) — ∫γ dlnτ.
        Rp_lo: Percentile 2.5 % de Rp (HMC uniquement, sinon None).
        Rp_hi: Percentile 97.5 % de Rp (HMC uniquement, sinon None).
        Z_fit: Impédance reconstruite (complexe, convention physique Z'' < 0),
            évaluée aux fréquences du spectre.
        score: R² de la reconstruction Z (stack Re/Im) vs mesure.
        engine: 'ridge' (aperçu) ou 'bayes' (HMC, avec intervalles).
        status: Message d'état (ex. garde-fou cmdstan). Vide si tout est nominal.
    """

    tau: np.ndarray
    gamma: np.ndarray
    gamma_lo: Optional[np.ndarray]
    gamma_hi: Optional[np.ndarray]
    Rp: float
    Rp_lo: Optional[float]
    Rp_hi: Optional[float]
    Z_fit: np.ndarray
    score: float
    engine: str
    status: str = ""

    @property
    def has_intervals(self) -> bool:
        """True si des intervalles de crédibilité sont disponibles (HMC)."""
        return self.gamma_lo is not None and self.gamma_hi is not None


# ─────────────────────────────────────────────────────────────────────────────
# Disponibilité du moteur / de cmdstan
# ─────────────────────────────────────────────────────────────────────────────
def bayes_available() -> bool:
    """True si le paquet vendoré bayes-drt2 (et ses deps) est importable."""
    return Inverter is not None


def import_error() -> Optional[str]:
    """Raison de l'indisponibilité de la DRT, ou None si tout va bien."""
    return _IMPORT_ERROR


def cmdstan_available() -> bool:
    """True si une installation cmdstan est présente (requise pour le HMC).

    N'installe ni ne compile rien : teste seulement ``cmdstanpy.cmdstan_path()``,
    qui lève si cmdstan est absent (point 5 — garde-fou).
    """
    try:
        import cmdstanpy

        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires impédance / cache
# ─────────────────────────────────────────────────────────────────────────────
def _impedance(spectrum) -> np.ndarray:
    """Z complexe en convention physique (Z'' < 0) depuis le spectre.

    Le loader stocke ``Zim = -Im(Z) > 0`` → ``Z = Zre - 1j·Zim``.
    """
    return np.asarray(spectrum.Zre, dtype=float) - 1j * np.asarray(spectrum.Zim, dtype=float)


def cache_key(spectrum) -> str:
    """Clé de cache = hash SHA-1 de (freq, Z) arrondis (point 3).

    L'arrondi (6 chiffres significatifs) rend la clé stable aux micro-variations
    de sérialisation/relecture d'une même mesure.
    """
    f = np.asarray(spectrum.f, dtype=float)
    Z = _impedance(spectrum)

    def _round(a: np.ndarray, sig: int = 6) -> np.ndarray:
        # Arrondi à `sig` chiffres significatifs, vectorisé (np.round n'accepte
        # pas un tableau de décimales).
        a = np.asarray(a, dtype=float)
        out = np.zeros_like(a)
        nz = a != 0
        mags = np.floor(np.log10(np.abs(a[nz])))
        factor = 10.0 ** (sig - 1 - mags)
        out[nz] = np.round(a[nz] * factor) / factor
        return out

    payload = np.concatenate([_round(f), _round(Z.real), _round(Z.imag)])
    return hashlib.sha1(payload.tobytes()).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(_CACHE_DIR, f"{key}.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# (Dé)sérialisation YAML — réutilise l'export YAML de session (point 3)
# ─────────────────────────────────────────────────────────────────────────────
def _to_list(a) -> Optional[list]:
    if a is None:
        return None
    return [float(x) for x in np.asarray(a, dtype=float)]


def result_to_dict(result: DRTResult) -> dict:
    """DRTResult -> dict YAML-sérialisable (numpy → listes de floats)."""
    return {
        "tau": _to_list(result.tau),
        "gamma": _to_list(result.gamma),
        "gamma_lo": _to_list(result.gamma_lo),
        "gamma_hi": _to_list(result.gamma_hi),
        "Rp": None if result.Rp is None else float(result.Rp),
        "Rp_lo": None if result.Rp_lo is None else float(result.Rp_lo),
        "Rp_hi": None if result.Rp_hi is None else float(result.Rp_hi),
        "Z_fit_re": _to_list(np.real(result.Z_fit)) if result.Z_fit is not None else None,
        "Z_fit_im": _to_list(np.imag(result.Z_fit)) if result.Z_fit is not None else None,
        "score": float(result.score),
        "engine": result.engine,
        "status": result.status,
    }


def result_from_dict(d: dict) -> DRTResult:
    """dict YAML -> DRTResult (listes → np.ndarray)."""
    def _arr(key):
        v = d.get(key)
        return None if v is None else np.asarray(v, dtype=float)

    zre = d.get("Z_fit_re")
    zim = d.get("Z_fit_im")
    z_fit = (
        None if zre is None or zim is None
        else np.asarray(zre, dtype=float) + 1j * np.asarray(zim, dtype=float)
    )
    return DRTResult(
        tau=_arr("tau"),
        gamma=_arr("gamma"),
        gamma_lo=_arr("gamma_lo"),
        gamma_hi=_arr("gamma_hi"),
        Rp=d.get("Rp"),
        Rp_lo=d.get("Rp_lo"),
        Rp_hi=d.get("Rp_hi"),
        Z_fit=z_fit,
        score=float(d.get("score", 0.0)),
        engine=d.get("engine", "bayes"),
        status=d.get("status", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Accès cache (mémoire + disque)
# ─────────────────────────────────────────────────────────────────────────────
def get_cached(spectrum, engine: str = "bayes") -> Optional[DRTResult]:
    """Résultat en cache pour ce spectre, ou None.

    Cherche d'abord en mémoire, puis (pour le HMC) sur disque dans ``sessions/``.
    Au rechargement d'une session, un HMC déjà persisté est ainsi resservi sans
    jamais être recalculé (point 3).
    """
    key = cache_key(spectrum)
    hit = _MEM_CACHE.get((key, engine))
    if hit is not None:
        return hit
    if engine == "bayes":
        path = _cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    d = yaml.safe_load(fh)
                result = result_from_dict(d)
                _MEM_CACHE[(key, engine)] = result
                return result
            except Exception as exc:
                log.warning(f"Cache DRT illisible ({path}) : {exc}")
    return None


def store_cached(spectrum, result: DRTResult) -> None:
    """Mémorise un résultat (mémoire toujours ; disque pour le HMC — point 3)."""
    key = cache_key(spectrum)
    _MEM_CACHE[(key, result.engine)] = result
    if result.engine == "bayes":
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(_cache_path(key), "w", encoding="utf-8") as fh:
                yaml.safe_dump(result_to_dict(result), fh, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            log.warning(f"Persistance cache DRT échouée : {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Cœur du calcul
# ─────────────────────────────────────────────────────────────────────────────
def _score(inv, freq: np.ndarray, Z: np.ndarray) -> float:
    """R² de la reconstruction Z (Re et Im empilés) vs mesure."""
    try:
        Z_fit = inv.predict_Z(freq)
    except Exception:
        return float("nan")
    y = np.concatenate([Z.real, Z.imag])
    yhat = np.concatenate([Z_fit.real, Z_fit.imag])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _ridge_inverter(freq: np.ndarray, Z: np.ndarray):
    """Inverter ajusté par ridge hyperparamétrique (aperçu + init HMC)."""
    inv = Inverter()
    inv.ridge_fit(freq, Z, hyper_lambda=True)
    return inv


def _result_from_ridge(inv, freq: np.ndarray, Z: np.ndarray, status: str = "") -> DRTResult:
    tau = np.asarray(inv.distributions[DIST_NAME]["tau"], dtype=float)
    gamma = np.asarray(inv.predict_distribution(DIST_NAME, tau=tau), dtype=float)
    return DRTResult(
        tau=tau,
        gamma=gamma,
        gamma_lo=None,
        gamma_hi=None,
        Rp=float(inv.predict_Rp()),
        Rp_lo=None,
        Rp_hi=None,
        Z_fit=np.asarray(inv.predict_Z(freq)),
        score=_score(inv, freq, Z),
        engine="ridge",
        status=status,
    )


def drt_preview(spectrum) -> DRTResult:
    """Aperçu DRT instantané par ridge hyperparamétrique (point 2).

    ``ridge_fit(freq, Z, hyper_lambda=True)`` — rapide, sans cmdstan. Résultat
    mis en cache mémoire (resservi tel quel aux reruns Streamlit).
    """
    if not bayes_available():
        raise RuntimeError(f"Moteur DRT indisponible : {_IMPORT_ERROR}")

    cached = _MEM_CACHE.get((cache_key(spectrum), "ridge"))
    if cached is not None:
        return cached

    freq = np.asarray(spectrum.f, dtype=float)
    Z = _impedance(spectrum)
    inv = _ridge_inverter(freq, Z)
    result = _result_from_ridge(inv, freq, Z)
    store_cached(spectrum, result)
    return result


def drt_bayes(spectrum, use_cache: bool = True) -> DRTResult:
    """DRT bayésienne (HMC) avec intervalles de crédibilité (points 2, 3, 5).

    * Un résultat HMC déjà en cache (mémoire ou ``sessions/``) est renvoyé tel
      quel — **jamais recalculé** (point 3).
    * Garde-fou cmdstan (point 5) : si cmdstan est absent, renvoie le résultat
      *ridge* + un statut explicite, **sans compiler à la volée**.
    * Sinon : ``fit(freq, Z, mode='sample', init_from_ridge=True)`` puis
      extraction de γ (médiane), γ_lo/γ_hi (2.5/97.5 %), Rp/Rp_lo/Rp_hi, Z_fit.
    """
    if not bayes_available():
        raise RuntimeError(f"Moteur DRT indisponible : {_IMPORT_ERROR}")

    if use_cache:
        cached = get_cached(spectrum, engine="bayes")
        if cached is not None:
            return cached

    freq = np.asarray(spectrum.f, dtype=float)
    Z = _impedance(spectrum)

    # Garde-fou cmdstan : pas de compilation à la volée.
    if not cmdstan_available():
        inv = _ridge_inverter(freq, Z)
        result = _result_from_ridge(inv, freq, Z, status=_SETUP_HINT)
        # On ne persiste PAS ce repli sous la clé bayes (sinon le vrai HMC
        # ultérieur serait masqué). Cache mémoire ridge uniquement.
        _MEM_CACHE[(cache_key(spectrum), "ridge")] = result
        return result

    result = _run_bayes(freq, Z)
    store_cached(spectrum, result)
    return result


def _run_bayes(freq: np.ndarray, Z: np.ndarray) -> DRTResult:
    """Exécute le HMC sur (freq, Z) et assemble le DRTResult (avec intervalles).

    Fonction au niveau module (picklable) : réutilisée par le worker du batch
    parallèle (point 4).
    """
    inv = Inverter()
    inv.fit(freq, Z, mode="sample", init_from_ridge=True)

    tau = np.asarray(inv.distributions[DIST_NAME]["tau"], dtype=float)
    gamma = np.asarray(inv.predict_distribution(DIST_NAME, tau=tau), dtype=float)
    gamma_lo = np.asarray(
        inv.predict_distribution(DIST_NAME, tau=tau, percentile=2.5), dtype=float
    )
    gamma_hi = np.asarray(
        inv.predict_distribution(DIST_NAME, tau=tau, percentile=97.5), dtype=float
    )

    return DRTResult(
        tau=tau,
        gamma=gamma,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        Rp=float(inv.predict_Rp()),
        Rp_lo=float(inv.predict_Rp(percentile=2.5)),
        Rp_hi=float(inv.predict_Rp(percentile=97.5)),
        Z_fit=np.asarray(inv.predict_Z(freq)),
        score=_score(inv, freq, Z),
        engine="bayes",
        status="",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch parallèle (point 4)
# ─────────────────────────────────────────────────────────────────────────────
def iter_session_spectra(session) -> List:
    """Tous les spectres d'une session (moyennes + réplicats), sans doublon.

    Ordre : bare, probe, puis chaque groupe (moyenne + réplicats).
    """
    seen = set()
    out: List = []

    def _add(sp):
        if sp is None:
            return
        oid = id(sp)
        if oid in seen:
            return
        seen.add(oid)
        out.append(sp)

    _add(getattr(session, "bare", None))
    _add(getattr(session, "probe", None))
    for sp in getattr(session, "bare_replicate_spectra", []) or []:
        _add(sp)
    for sp in getattr(session, "probe_replicate_spectra", []) or []:
        _add(sp)
    for grp in getattr(session, "groups", []) or []:
        _add(getattr(grp, "spectrum", None))
        for sp in getattr(grp, "replicate_spectra", []) or []:
            _add(sp)
    return out


def _bayes_worker(args):
    """Worker picklable pour ProcessPoolExecutor : (key, f, Zre, Zim) -> dict.

    Renvoie un dict sérialisable (pas le DRTResult ni l'Inverter, non nécessaires
    de l'autre côté de la frontière de processus).
    """
    key, f, zre, zim = args
    freq = np.asarray(f, dtype=float)
    Z = np.asarray(zre, dtype=float) - 1j * np.asarray(zim, dtype=float)
    result = _run_bayes(freq, Z)
    return key, result_to_dict(result)


def run_drt_bayes_batch(
    session,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: Optional[int] = None,
) -> dict:
    """Lance le HMC en parallèle sur tous les spectres non encore en cache (point 4).

    cmdstanpy exécute cmdstan en sous-processus : un ``ProcessPoolExecutor``
    (``max_workers`` = nombre de cœurs par défaut) parallélise efficacement.
    Les spectres déjà en cache (mémoire ou ``sessions/``) sont ignorés — un HMC
    n'est jamais recalculé (point 3).

    Args:
        session: EISSession.
        progress_callback: appelé après chaque spectre terminé avec
            ``(n_faits, n_total, label)`` — l'UI y branche une barre de
            progression Streamlit.
        max_workers: nombre de processus (défaut : ``os.cpu_count()``).

    Returns:
        dict {cache_key: DRTResult} des spectres calculés dans ce batch.
    """
    if not bayes_available():
        raise RuntimeError(f"Moteur DRT indisponible : {_IMPORT_ERROR}")
    if not cmdstan_available():
        raise RuntimeError(_SETUP_HINT)

    spectra = iter_session_spectra(session)

    # Ne garder que les spectres SANS cache HMC, dédupliqués par clé.
    pending = {}
    for sp in spectra:
        key = cache_key(sp)
        if key in pending:
            continue
        if get_cached(sp, engine="bayes") is not None:
            continue
        pending[key] = sp

    total = len(pending)
    computed: dict = {}
    if total == 0:
        if progress_callback is not None:
            progress_callback(0, 0, "")
        return computed

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, total))

    tasks = [
        (key, np.asarray(sp.f, dtype=float), np.asarray(sp.Zre, dtype=float),
         np.asarray(sp.Zim, dtype=float))
        for key, sp in pending.items()
    ]
    key_to_spectrum = dict(pending)

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_bayes_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            sp = key_to_spectrum[key]
            try:
                _key, payload = fut.result()
                result = result_from_dict(payload)
                store_cached(sp, result)
                computed[key] = result
            except Exception as exc:
                log.error(f"HMC échoué pour '{getattr(sp, 'label', key)}' : {exc}")
            finally:
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, getattr(sp, "label", ""))

    return computed
