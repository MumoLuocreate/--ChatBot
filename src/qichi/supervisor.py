"""Lifecycle coordination for the local Qichi runtime.

The supervisor owns no semantic policy.  It only keeps the single inbound
Forward WebSocket consumer, the post-send memory worker and the initiative
scheduler in one cancellation domain.  Production startup can use the
worker evidence exposed here when constructing its readiness providers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import logging
from typing import Any
from uuid import uuid4

from qichi.readiness import WorkerEvidence
from qichi.transport.onebot_client import (
    OneBotConnectionError,
    OneBotTimeoutError,
    OneBotWebSocketDisconnectedError,
    OneBotWebSocketHandshakeError,
)


_LOGGER = logging.getLogger(__name__)
_RECOVERABLE_WS_ERRORS = (
    OneBotConnectionError,
    OneBotTimeoutError,
    OneBotWebSocketDisconnectedError,
    OneBotWebSocketHandshakeError,
)


class SupervisorError(RuntimeError):
    """The runtime could not keep one of its required loops alive."""


@dataclass(frozen=True, slots=True)
class LoopStatus:
    """Observable status without pretending a stopped loop is healthy."""

    worker_id: str
    recovered: bool
    alive: bool


class RuntimeSupervisor:
    """Run all production loops with one owner and one cancellation boundary.

    ``start`` waits until each loop has entered and completed its first local
    pass.  It does not perform identity checks or write a READY marker; those
    remain the responsibility of :class:`ReadinessCoordinator`.
    """

    _LOOP_NAMES = ("ws", "memory", "initiative")

    def __init__(
        self,
        application: Any,
        onebot_client: Any,
        memory_worker: Any,
        initiative_scheduler: Any,
        *,
        conversation_id: str,
        clock: Callable[[], datetime] | None = None,
        memory_poll_seconds: float = 1.0,
        initiative_poll_seconds: float = 1.0,
        startup_timeout_seconds: float = 30.0,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        close_client_on_stop: bool = True,
        worker_id_factory: Callable[[str], str] | None = None,
    ) -> None:
        if not callable(getattr(application, "handle_onebot", None)):
            raise TypeError("application must provide handle_onebot")
        if not callable(getattr(onebot_client, "event_stream", None)):
            raise TypeError("onebot_client must provide event_stream")
        if not callable(getattr(memory_worker, "run_due", None)):
            raise TypeError("memory_worker must provide run_due")
        if not callable(getattr(initiative_scheduler, "tick", None)):
            raise TypeError("initiative_scheduler must provide tick")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id must be non-empty text")
        for value, field in (
            (memory_poll_seconds, "memory_poll_seconds"),
            (initiative_poll_seconds, "initiative_poll_seconds"),
            (startup_timeout_seconds, "startup_timeout_seconds"),
            (reconnect_initial_seconds, "reconnect_initial_seconds"),
            (reconnect_max_seconds, "reconnect_max_seconds"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if reconnect_max_seconds < reconnect_initial_seconds:
            raise ValueError("reconnect_max_seconds must be >= reconnect_initial_seconds")
        if type(close_client_on_stop) is not bool:
            raise TypeError("close_client_on_stop must be bool")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if worker_id_factory is not None and not callable(worker_id_factory):
            raise TypeError("worker_id_factory must be callable")

        self.application = application
        self.onebot_client = onebot_client
        self.memory_worker = memory_worker
        self.initiative_scheduler = initiative_scheduler
        self.conversation_id = conversation_id
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.memory_poll_seconds = float(memory_poll_seconds)
        self.initiative_poll_seconds = float(initiative_poll_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.close_client_on_stop = close_client_on_stop
        self._worker_id_factory = worker_id_factory or (lambda _name: uuid4().hex)

        self._statuses = {
            name: LoopStatus(self._new_worker_id(name), False, False)
            for name in self._LOOP_NAMES
        }
        if len({status.worker_id for status in self._statuses.values()}) != len(self._statuses):
            raise ValueError("worker_id_factory must return unique worker IDs")
        self._tasks: tuple[asyncio.Task[Any], ...] = ()
        self._started = False
        self._startup_phase = True
        self._stopping = False
        self._failure: BaseException | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._failure_event = asyncio.Event()
        self._drain_lock = asyncio.Lock()
        self._drained = False
        # A production worker exposes this method; test doubles from older
        # lifecycle cards do not need a semantic gate.
        self._requires_semantic_gate = callable(
            getattr(memory_worker, "open_semantic_gate", None)
        ) or callable(getattr(initiative_scheduler, "open_semantic_gate", None))
        self._semantic_gate = not self._requires_semantic_gate

    @property
    def started(self) -> bool:
        return self._started

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set() and self._failure is None and not self._stopping

    def status(self, name: str) -> LoopStatus:
        if name not in self._statuses:
            raise KeyError(name)
        return self._statuses[name]

    def worker_evidence(self, name: str) -> WorkerEvidence:
        if name not in {"memory", "initiative"}:
            raise ValueError("worker evidence name must be memory or initiative")
        status = self._statuses[name]
        if not status.recovered or not status.alive:
            raise SupervisorError(f"{name} worker is not ready")
        return WorkerEvidence(status.worker_id, status.recovered, status.alive)

    @property
    def semantic_gate_open(self) -> bool:
        return self._semantic_gate

    def open_semantic_gate(self) -> None:
        """Open background semantic work after the READY marker is durable."""
        self._semantic_gate = True
        for component in (self.memory_worker, self.initiative_scheduler):
            opener = getattr(component, "open_semantic_gate", None)
            if callable(opener):
                opener()

    async def start(self) -> None:
        """Start all loops and wait for their first successful local pass."""
        if self._started:
            raise SupervisorError("supervisor is already started")
        if self._stopping:
            raise SupervisorError("supervisor cannot be restarted")
        self._started = True
        self._tasks = (
            asyncio.create_task(self._ws_loop(), name="qichi-forward-ws"),
            asyncio.create_task(self._memory_loop(), name="qichi-memory-worker"),
            asyncio.create_task(self._initiative_loop(), name="qichi-initiative-worker"),
        )
        try:
            await asyncio.wait_for(
                self._wait_for_ready_or_failure(), timeout=self.startup_timeout_seconds
            )
            self._startup_phase = False
        except asyncio.TimeoutError as error:
            await self.stop()
            raise SupervisorError("runtime loops did not become ready during startup") from error
        except BaseException:
            await self.stop()
            raise

    async def _wait_for_ready_or_failure(self) -> None:
        while not self._ready_event.is_set() and not self._failure_event.is_set():
            ready = asyncio.create_task(self._ready_event.wait())
            failed = asyncio.create_task(self._failure_event.wait())
            try:
                await asyncio.wait((ready, failed), return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (ready, failed):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(ready, failed, return_exceptions=True)
        if self._failure is not None:
            raise SupervisorError("runtime loop failed during startup") from self._failure

    async def run_forever(self) -> None:
        """Start, wait for shutdown, and always close child resources."""
        await self.start()
        try:
            await self.wait()
        finally:
            await self.stop()

    async def wait(self) -> None:
        """Wait for a clean WS end, a loop failure, or external stop."""
        if not self._started:
            raise SupervisorError("supervisor has not been started")
        try:
            await self._stop_event.wait()
            await self._drain_tasks()
        except asyncio.CancelledError:
            await self.stop()
            raise
        if self._failure is not None:
            raise SupervisorError("runtime loop failed") from self._failure

    async def stop(self) -> None:
        """Cancel every child and close the transport exactly once."""
        if self._stopping:
            return
        self._stopping = True
        if self._startup_phase and self._failure is None:
            self._failure = SupervisorError("supervisor stopped during startup")
            self._failure_event.set()
        self._stop_event.set()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current and not task.done():
                task.cancel()
        await self._drain_tasks()
        if self.close_client_on_stop:
            close = getattr(self.onebot_client, "close", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    if self._failure is None:
                        self._failure = error
                        self._failure_event.set()
        for name, status in self._statuses.items():
            self._statuses[name] = LoopStatus(status.worker_id, status.recovered, False)

    async def _drain_tasks(self) -> None:
        async with self._drain_lock:
            if self._drained:
                return
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._drained = True

    def _new_worker_id(self, name: str) -> str:
        value = self._worker_id_factory(name)
        if not isinstance(value, str) or not value:
            raise ValueError("worker_id_factory must return non-empty text")
        return value

    def _mark_alive(self, name: str, *, recovered: bool | None = None) -> None:
        status = self._statuses[name]
        self._statuses[name] = LoopStatus(
            status.worker_id,
            status.recovered if recovered is None else recovered,
            True,
        )
        if all(item.recovered and item.alive for item in self._statuses.values()):
            self._ready_event.set()

    def _mark_dead(self, name: str) -> None:
        status = self._statuses[name]
        self._statuses[name] = LoopStatus(status.worker_id, status.recovered, False)

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None and not isinstance(error, asyncio.CancelledError):
            self._failure = error
            self._failure_event.set()
        self._stop_event.set()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current and not task.done():
                task.cancel()

    async def _ws_loop(self) -> None:
        backoff = self.reconnect_initial_seconds
        while not self._stopping:
            consumer: asyncio.Task[Any] | None = None
            handshake: asyncio.Task[Any] | None = None
            try:
                stream = self.onebot_client.event_stream()
                if not hasattr(stream, "__aiter__"):
                    raise TypeError("event_stream must return an async iterator")
                consumer = asyncio.create_task(
                    self._consume_stream(stream), name="qichi-ws-consumer"
                )
                wait_connected = getattr(self.onebot_client, "wait_until_connected", None)
                if callable(wait_connected):
                    handshake = asyncio.create_task(
                        wait_connected(), name="qichi-ws-handshake"
                    )
                    done, _ = await asyncio.wait(
                        (consumer, handshake), return_when=asyncio.FIRST_COMPLETED
                    )
                    if handshake in done:
                        handshake.result()
                        # A real client is only live after the transport handshake.
                        self._mark_alive("ws", recovered=True)
                        await consumer
                    else:
                        await consumer
                else:
                    # Test doubles without an explicit handshake retain the old
                    # liveness contract; production OneBotClient always has one.
                    self._mark_alive("ws", recovered=True)
                    await consumer
                if not self._stopping:
                    if not self._startup_phase:
                        self._stop_event.set()
                        self._cancel_siblings()
                    else:
                        self._record_failure(
                            SupervisorError("Forward WebSocket ended during startup")
                        )
                    return
            except asyncio.CancelledError:
                raise
            except _RECOVERABLE_WS_ERRORS as error:
                if self._stopping:
                    return
                self._mark_dead("ws")
                _LOGGER.warning(
                    "Forward WebSocket disconnected; retrying (%s)", type(error).__name__
                )
                await self._wait_reconnect(backoff)
                backoff = min(self.reconnect_max_seconds, backoff * 2)
                continue
            except BaseException as error:
                self._record_failure(error)
                raise
            finally:
                for task in (consumer, handshake):
                    if task is not None and not task.done():
                        task.cancel()
                if consumer is not None or handshake is not None:
                    await asyncio.gather(
                        *(task for task in (consumer, handshake) if task is not None),
                        return_exceptions=True,
                    )
                if not self._stopping:
                    self._mark_dead("ws")

    async def _consume_stream(self, stream: Any) -> None:
        async for payload in stream:
            try:
                await self.application.handle_onebot(payload)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # A model/protocol failure for one durable event must not take
                # down the sole transport consumer. The event remains in the
                # ledger for explicit recovery and this log is intentionally
                # type-only to avoid copying private prompt or message text.
                _LOGGER.error(
                    "inbound event handler failed (%s)", type(error).__name__
                )

    async def _memory_loop(self) -> None:
        try:
            self._mark_alive("memory")
            recover = getattr(self.memory_worker, "recover_pending", None)
            if callable(recover):
                result = recover(self.conversation_id)
                if inspect.isawaitable(result):
                    await result
            if self._semantic_gate:
                await self.memory_worker.run_due(self._now())
            self._mark_alive("memory", recovered=True)
            while not self._stop_event.is_set():
                if await self._wait_interval(self.memory_poll_seconds):
                    break
                if self._semantic_gate:
                    await self.memory_worker.run_due(self._now())
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._record_failure(error)
            raise
        finally:
            self._mark_dead("memory")

    async def _initiative_loop(self) -> None:
        try:
            self._mark_alive("initiative")
            if self._semantic_gate:
                await self.initiative_scheduler.tick(self.conversation_id)
            self._mark_alive("initiative", recovered=True)
            while not self._stop_event.is_set():
                if await self._wait_interval(self.initiative_poll_seconds):
                    break
                if self._semantic_gate:
                    await self.initiative_scheduler.tick(self.conversation_id)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._record_failure(error)
            raise
        finally:
            self._mark_dead("initiative")

    async def _wait_interval(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return False
        return True

    async def _wait_reconnect(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _cancel_siblings(self) -> None:
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current and not task.done():
                task.cancel()

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return an aware datetime")
        return value.astimezone(timezone.utc)
