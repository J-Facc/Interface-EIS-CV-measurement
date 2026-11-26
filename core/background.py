"""Exécution de calculs en arrière-plan (v3) — aucun import Streamlit.

``BackgroundRun`` soumet des tâches à un pool de processus (``ProcessPoolExecutor``, nombre
de workers = ``os.cpu_count() − 1``, minimum 1) et les récolte par :meth:`BackgroundRun.poll`,
appelé périodiquement par l'interface (fragment Streamlit) sans jamais la bloquer.

* Les rappels (``on_result``/``on_error``) s'exécutent dans le processus PRINCIPAL, pendant
  ``poll`` ; ils peuvent soumettre de nouvelles tâches (ex. la DRT d'un spectre moyen une fois
  son groupe ajusté).
* Une tâche en échec (exception dans le worker, worker mort…) est rapportée par ``on_error``
  avec sa cause : elle n'arrête JAMAIS les autres.
* :meth:`cancel` annule les tâches non démarrées ; une tâche en cours (ex. un échantillonnage
  CmdStan) va à son terme dans son processus mais son résultat est ignoré.

Exécuteurs : ``"process"`` (défaut, vrai parallélisme ; contexte ``spawn``, identique sous
Windows et Linux), ``"thread"`` ou ``"sync"`` (exécution immédiate, pour les tests).
"""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.logger import get_logger

log = get_logger("background")


def default_workers(max_workers: Optional[int] = None) -> int:
    """``max_workers`` s'il est donné (≥ 1), sinon ``os.cpu_count() − 1`` (minimum 1)."""
    if max_workers:
        return max(1, int(max_workers))
    return max(1, (os.cpu_count() or 2) - 1)


def _worker_init() -> None:
    """Initialisation d'un processus de calcul : ré-enregistre CmdStan (sans rien installer)."""
    try:
        from setup_drt_bayesien import ensure_drt_ready
        ensure_drt_ready(allow_install=False)
    except Exception:                    # pragma: no cover - dépend de l'environnement
        pass


class _SyncExecutor:
    """Exécuteur immédiat (tests) : même interface minimale que concurrent.futures."""

    def submit(self, fn, *args):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:     # noqa: BLE001 — relayé tel quel par le Future
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


#: Variable d'environnement qui force le type d'exécuteur (tests : "sync" ou "thread").
EXECUTOR_ENV = "EIS_EXECUTOR"


def make_executor(kind: str = "process", max_workers: Optional[int] = None):
    kind = os.environ.get(EXECUTOR_ENV) or kind
    if kind == "sync":
        return _SyncExecutor()
    n = default_workers(max_workers)
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=n)
    if kind == "process":
        return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_init)
    raise ValueError(f"exécuteur inconnu : {kind!r}")


@dataclass
class _Pending:
    key: str
    future: Any
    on_result: Callable
    on_error: Callable


class BackgroundRun:
    """Tâches en arrière-plan avec progression ``done/total`` et erreurs isolées."""

    def __init__(self, executor_kind: str = "process", max_workers: Optional[int] = None):
        self.executor_kind = executor_kind
        self.max_workers = default_workers(max_workers)
        self._executor = make_executor(executor_kind, max_workers)
        self._pending: list = []
        self.total = 0
        self.done = 0
        self.cancelled = False
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    # ── Soumission / récolte ────────────────────────────────────────────────
    def submit(self, key: str, fn: Callable, payload: Any, on_result: Callable, on_error: Callable) -> None:
        if self.cancelled:
            return
        try:
            fut = self._executor.submit(fn, payload)
        except Exception as exc:          # pool cassé (worker mort au démarrage…)
            fut = Future()
            fut.set_exception(exc)
        self._pending.append(_Pending(key, fut, on_result, on_error))

    def poll(self) -> bool:
        """Récolte les tâches terminées (rappels exécutés ici). Retourne ``finished``."""
        if self.cancelled:
            return True
        progressed = True
        while progressed:               # un rappel peut soumettre une tâche déjà terminée (sync)
            progressed = False
            for p in [p for p in self._pending if p.future.done()]:
                self._pending.remove(p)
                progressed = True
                self.done += 1
                try:
                    exc = p.future.exception()
                except BaseException as e:        # future annulée
                    exc = e
                try:
                    if exc is None:
                        p.on_result(p.future.result())
                    else:
                        msg = f"{type(exc).__name__}: {exc}"
                        log.error(f"Tâche « {p.key} » en échec : {msg}\n"
                                  + "".join(traceback.format_exception(exc)))
                        p.on_error(msg)
                except Exception as cb_exc:       # bug d'un rappel : signalé, n'arrête pas le lot
                    log.error(f"Rappel de la tâche « {p.key} » en échec :\n{traceback.format_exc()}")
                    try:
                        p.on_error(f"erreur interne : {cb_exc}")
                    except Exception:
                        pass
        if not self._pending and self.finished_at is None:
            self.finished_at = time.time()
            self._executor.shutdown(wait=False)
        return self.finished

    @property
    def finished(self) -> bool:
        return self.cancelled or not self._pending

    @property
    def running(self) -> int:
        return len(self._pending)

    def wait(self, timeout: Optional[float] = None, interval: float = 0.05) -> bool:
        """Boucle de ``poll`` jusqu'à la fin (tests, scripts). Retourne ``finished``."""
        t0 = time.time()
        while not self.poll():
            if timeout is not None and time.time() - t0 > timeout:
                return False
            time.sleep(interval)
        return True

    def cancel(self) -> None:
        """Annule les tâches non démarrées ; ignore les résultats des tâches en cours."""
        if self.cancelled:
            return
        self.cancelled = True
        for p in self._pending:
            p.future.cancel()
        self._pending.clear()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:                 # exécuteur minimal (sync)
            self._executor.shutdown(wait=False)
        self.finished_at = time.time()

    @property
    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at
