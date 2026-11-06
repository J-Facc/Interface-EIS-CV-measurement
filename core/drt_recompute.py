"""Recalcul DRT à la demande — un spectre, plusieurs, ou retour à ``optimize``.

Couche MÉTIER du panneau « Recalcul » de l'onglet DRT (``ui/tabs.py``) : aucun import
Streamlit ; l'UI ne fait qu'appeler ces fonctions et afficher leurs ``RecalcOutcome``.

Par défaut tout est calculé en ``optimize`` (MAP, ~1 s par spectre : ``fit.drt.mode`` de la
config). Le ``sample`` (HMC) est un recalcul CIBLÉ, long (2 à 5 min par spectre) et parfois
instable : il passe par ici, spectre par spectre.

Règles
------
* **Jamais d'appel direct à ``Inverter``** : tout passe par ``core.pipeline.recompute_drt``,
  donc par ``drt.engine.fit_drt`` (réglages durcis, voir ``drt/VALIDATION_REGLAGES.md``).
* **Un recalcul n'écrase que SON spectre.** Un échec n'écrit rien : l'ancien résultat reste en
  place (``recompute_drt`` n'assigne qu'après le calcul réussi).
* **Une erreur sur un spectre n'en arrête pas un autre** : elle est rendue dans le
  ``RecalcOutcome`` (``ok=False``, ``error``), jamais levée. Seule une saisie invalide
  (``InvalidAnalysisInput`` : mode inconnu) remonte, car c'est un bug de l'appelant.
* **Registre à clés stables** (``store``) : un simple dict, que l'UI loge dans
  ``st.session_state[STORE_KEY]``. Il garde, par spectre, le dernier résultat de chaque mode
  (``{'optimize': FitResult, 'sample': FitResult}``), de sorte que « Revenir en optimize »
  rend le résultat MAP d'origine sans le recalculer et sans perdre le résultat HMC. Une
  entrée n'est écrasée que par un nouveau calcul du même mode sur le même spectre ; tout le
  registre est remis à zéro avec les autres résultats d'analyse
  (``core.app_state.ANALYSIS_RESULT_KEYS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, MutableMapping, Optional

from core.models import EISSession, EISSpectrum, FitResult
from core.pipeline import (
    DRT_MODEL_NAME,
    InvalidAnalysisInput,
    _iter_session_spectra,
    install_drt_result,
    recompute_drt,
)

MODE_OPTIMIZE = "optimize"
MODE_SAMPLE = "sample"

#: Clé de ``st.session_state`` du registre des résultats DRT par spectre.
STORE_KEY = "eis_drt_store"

#: Durée documentée d'un calcul HMC par spectre, en minutes (min, max) — ``config/default.yaml``,
#: ``drt/VALIDATION_REGLAGES.md``. Sert aux avertissements, pas à un calcul.
SAMPLE_MINUTES_PER_SPECTRUM = (2, 5)

#: ``on_progress(terminés, total, libellé du spectre en cours)`` ; libellé vide à la fin.
ProgressCallback = Callable[[int, int, str], None]


def drt_store_key(electrode, session_label: str) -> str:
    """Clé STABLE d'un spectre dans le registre : électrode + libellé de session.

    Le libellé de session est celui de ``core.pipeline._iter_session_spectra`` (``'probe'``,
    ``'probe#2'``, ``'1.00e-09'``, ``'1.00e-09#0'``…) : il ne dépend ni du nom de fichier ni
    de l'ordre d'affichage.
    """
    return f"e{electrode}|{session_label}"


@dataclass(frozen=True)
class DRTTarget:
    """Un spectre d'une session, identifié pour le recalcul."""

    electrode: object
    session_label: str
    session: EISSession
    spectrum: EISSpectrum

    @property
    def key(self) -> str:
        return drt_store_key(self.electrode, self.session_label)

    @property
    def display(self) -> str:
        """Libellé lisible, pour un message : électrode, rôle du spectre, nom du fichier."""
        return f"électrode {self.electrode} · {self.session_label} ({self.spectrum.label})"


@dataclass(frozen=True)
class RecalcOutcome:
    """Résultat d'UN recalcul. ``ok=False`` : ``error`` dit pourquoi, l'ancien résultat est intact."""

    key: str
    display: str
    mode: str
    ok: bool
    duration_s: Optional[float] = None
    error: Optional[str] = None
    #: True si le résultat n'a pas été recalculé (déjà dans ce mode, ou repris du registre).
    reused: bool = False


@dataclass(frozen=True)
class DRTRunInfo:
    """Mode, durée et date du calcul de la DRT ACTIVE d'un spectre (lus dans ``drt_diagnostics``)."""

    mode: Optional[str]
    duration_s: Optional[float]
    computed_at: Optional[str]


def iter_targets(sessions: dict) -> list:
    """Tous les spectres recalculables de toutes les électrodes : ``[DRTTarget]``.

    Électrodes triées ; les spectres d'un groupe ARRÊTÉ (statut non OK : aucune DRT n'y a été
    calculée, l'UI ne les propose pas) sont écartés.
    """
    out = []
    for e in sorted(sessions):
        session = sessions[e]
        stopped = {id(sp) for _lbl, mean_sp, reps, an in session.iter_groups()
                   if an is not None and not an.ok for sp in [mean_sp, *(reps or [])]}
        out += [DRTTarget(e, label, session, sp) for label, sp in _iter_session_spectra(session)
                if id(sp) not in stopped]
    return out


def target_of(electrode, session: EISSession, spectrum: EISSpectrum) -> DRTTarget:
    """``DRTTarget`` d'un objet spectre de la session (par identité).

    Raises:
        ValueError: ``spectrum`` n'appartient pas à ``session``.
    """
    for label, sp in _iter_session_spectra(session):
        if sp is spectrum:
            return DRTTarget(electrode, label, session, spectrum)
    raise ValueError(f"Spectre introuvable dans la session : {getattr(spectrum, 'label', spectrum)!r}.")


def run_info(fr: Optional[FitResult]) -> Optional[DRTRunInfo]:
    """Mode, durée et date d'un ``FitResult`` DRT ; ``None`` sans résultat.

    Une DRT sans ``duration_s`` (résultat antérieur au chronométrage) donne une durée ``None``.
    """
    if fr is None:
        return None
    diag = fr.drt_diagnostics if isinstance(fr.drt_diagnostics, dict) else {}
    duration = diag.get("duration_s")
    return DRTRunInfo(
        mode=fr.drt_mode or diag.get("mode"),
        duration_s=float(duration) if isinstance(duration, (int, float)) else None,
        computed_at=diag.get("computed_at"),
    )


def estimate_sample_minutes(n_spectra: int) -> tuple:
    """Fourchette (min, max) en minutes d'un recalcul HMC de ``n_spectra`` spectres."""
    lo, hi = SAMPLE_MINUTES_PER_SPECTRUM
    return lo * n_spectra, hi * n_spectra


def _remember(store: Optional[MutableMapping], key: str,
              previous: Optional[FitResult], current: FitResult) -> None:
    """Range dans le registre le résultat qu'on quitte et celui qu'on adopte, chacun sous son mode."""
    if store is None:
        return
    entry = store.setdefault(key, {})
    for fr in (previous, current):
        info = run_info(fr)
        if info is not None and info.mode in (MODE_OPTIMIZE, MODE_SAMPLE):
            entry[info.mode] = fr


def recalculate(target: DRTTarget, config: Optional[dict], mode: str,
                store: Optional[MutableMapping] = None) -> RecalcOutcome:
    """Recalcule la DRT d'UN spectre dans ``mode`` ; n'écrase aucun autre résultat.

    Returns:
        ``RecalcOutcome`` — ``ok=False`` si le spectre est invalide ou si CmdStan échoue
        (l'ancienne DRT du spectre reste active).

    Raises:
        InvalidAnalysisInput: ``mode`` inconnu.
    """
    previous = target.spectrum.fit_results.get(DRT_MODEL_NAME)
    try:
        fr = recompute_drt(target.session, target.spectrum, config, mode=mode)
    except InvalidAnalysisInput:
        raise
    except (ValueError, RuntimeError) as exc:      # contrat de drt.engine.fit_drt
        return RecalcOutcome(target.key, target.display, mode, ok=False, error=str(exc))
    _remember(store, target.key, previous, fr)
    info = run_info(fr)
    return RecalcOutcome(target.key, target.display, mode, ok=True,
                         duration_s=info.duration_s if info else None)


def revert_to_optimize(target: DRTTarget, config: Optional[dict],
                       store: Optional[MutableMapping] = None) -> RecalcOutcome:
    """Remet la DRT d'un spectre en ``optimize``.

    Dans l'ordre : déjà en optimize → rien à faire ; un résultat MAP est dans le registre →
    il redevient actif SANS recalcul (et le résultat HMC quitté y est conservé) ; sinon la
    DRT est recalculée en optimize (~1 s).
    """
    active = target.spectrum.fit_results.get(DRT_MODEL_NAME)
    if active is not None and run_info(active).mode == MODE_OPTIMIZE:
        return RecalcOutcome(target.key, target.display, MODE_OPTIMIZE, ok=True,
                             duration_s=run_info(active).duration_s, reused=True)
    kept = (store or {}).get(target.key, {}).get(MODE_OPTIMIZE)
    if kept is not None:
        install_drt_result(target.session, target.spectrum, kept)
        _remember(store, target.key, active, kept)
        return RecalcOutcome(target.key, target.display, MODE_OPTIMIZE, ok=True,
                             duration_s=run_info(kept).duration_s, reused=True)
    return recalculate(target, config, MODE_OPTIMIZE, store)


def _run_many(action: Callable[[DRTTarget], RecalcOutcome], targets: Iterable[DRTTarget],
              on_progress: Optional[ProgressCallback]) -> list:
    targets = list(targets)
    total = len(targets)
    outcomes = []
    for i, target in enumerate(targets):
        if on_progress is not None:
            on_progress(i, total, target.display)
        outcomes.append(action(target))
    if on_progress is not None:
        on_progress(total, total, "")
    return outcomes


def recalculate_many(targets: Iterable[DRTTarget], config: Optional[dict], mode: str,
                     store: Optional[MutableMapping] = None,
                     on_progress: Optional[ProgressCallback] = None) -> list:
    """``recalculate`` sur chaque spectre, dans l'ordre ; un échec n'interrompt pas la suite.

    Returns:
        Un ``RecalcOutcome`` par spectre, dans l'ordre de ``targets``.
    """
    return _run_many(lambda t: recalculate(t, config, mode, store), targets, on_progress)


def revert_many(targets: Iterable[DRTTarget], config: Optional[dict],
                store: Optional[MutableMapping] = None,
                on_progress: Optional[ProgressCallback] = None) -> list:
    """``revert_to_optimize`` sur chaque spectre ; un échec n'interrompt pas la suite."""
    return _run_many(lambda t: revert_to_optimize(t, config, store), targets, on_progress)
