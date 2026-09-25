from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from qichi.supervisor import RuntimeSupervisor, SupervisorError
from qichi.transport.onebot_client import OneBotWebSocketDisconnectedError


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


class FakeApplication:
    def __init__(
        self,
        *,
        failure: BaseException | None = None,
        before_failure: asyncio.Event | None = None,
    ):
        self.payloads: list[object] = []
        self.failure = failure
        self.before_failure = before_failure

    async def handle_onebot(self, payload: object) -> None:
        if self.before_failure is not None:
            await self.before_failure.wait()
        if self.failure is not None:
            raise self.failure
        self.payloads.append(payload)


class FakeStreamClient:
    def __init__(self, payloads=(), *, block: asyncio.Event | None = None):
        self.payloads = list(payloads)
        self.block = block
        self.stream_calls = 0
        self.closed = 0

    def event_stream(self):
        self.stream_calls += 1

        async def stream():
            for payload in self.payloads:
                yield payload
            if self.block is not None:
                await self.block.wait()

        return stream()

    async def close(self):
        self.closed += 1


class ReconnectingStreamClient:
    def __init__(self, block: asyncio.Event):
        self.block = block
        self.stream_calls = 0
        self.closed = 0

    def event_stream(self):
        self.stream_calls += 1
        call = self.stream_calls

        async def stream():
            if call == 1:
                raise OneBotWebSocketDisconnectedError("temporary disconnect")
            yield {"reconnected": True}
            await self.block.wait()

        return stream()

    async def close(self):
        self.closed += 1


class FakeMemoryWorker:
    def __init__(self, *, failure: BaseException | None = None):
        self.calls = 0
        self.failure = failure
        self.entered = asyncio.Event()
        self.gate_opened = False

    def open_semantic_gate(self):
        self.gate_opened = True

    async def run_due(self, now):
        assert now.tzinfo is not None
        self.calls += 1
        self.entered.set()
        if self.failure is not None:
            raise self.failure


class FakeInitiativeScheduler:
    def __init__(self, *, failure: BaseException | None = None):
        self.calls: list[str] = []
        self.failure = failure
        self.entered = asyncio.Event()

    async def tick(self, conversation_id):
        self.calls.append(conversation_id)
        self.entered.set()
        if self.failure is not None:
            raise self.failure


def make_supervisor(client, memory, initiative, application=None, **kwargs):
    options = {
        "conversation_id": "123",
        "clock": lambda: NOW,
        "memory_poll_seconds": 0.01,
        "initiative_poll_seconds": 0.01,
        "startup_timeout_seconds": 0.5,
        "worker_id_factory": lambda name: f"worker-{name}",
    }
    options.update(kwargs)
    return RuntimeSupervisor(
        application or FakeApplication(),
        client,
        memory,
        initiative,
        **options,
    )


@pytest.mark.asyncio
async def test_start_runs_one_ws_consumer_and_initializes_both_workers():
    release = asyncio.Event()
    client = FakeStreamClient([{"id": 1}, {"id": 2}], block=release)
    memory = FakeMemoryWorker()
    initiative = FakeInitiativeScheduler()
    app = FakeApplication()
    supervisor = make_supervisor(client, memory, initiative, app)

    await supervisor.start()
    try:
        assert client.stream_calls == 1
        assert app.payloads == [{"id": 1}, {"id": 2}]
        assert memory.calls == 0
        assert initiative.calls == []
        assert supervisor.ready
        assert not supervisor.semantic_gate_open
        assert supervisor.worker_evidence("memory").worker_id == "worker-memory"
        assert supervisor.worker_evidence("initiative").worker_id == "worker-initiative"
    finally:
        release.set()
        await supervisor.stop()
    assert client.closed == 1


@pytest.mark.asyncio
async def test_start_is_single_use_and_does_not_open_a_second_ws_consumer():
    release = asyncio.Event()
    client = FakeStreamClient(block=release)
    supervisor = make_supervisor(
        client, FakeMemoryWorker(), FakeInitiativeScheduler()
    )
    await supervisor.start()
    with pytest.raises(SupervisorError, match="already started"):
        await supervisor.start()
    release.set()
    await supervisor.stop()
    assert client.stream_calls == 1


@pytest.mark.asyncio
async def test_loop_failure_cancels_siblings_and_closes_client():
    failure = RuntimeError("memory loop failed")
    client = FakeStreamClient(block=asyncio.Event())
    memory = FakeMemoryWorker(failure=failure)
    initiative = FakeInitiativeScheduler()
    supervisor = make_supervisor(client, memory, initiative)

    await supervisor.start()
    assert supervisor.failure is None
    assert memory.calls == 0
    await supervisor.stop()


@pytest.mark.asyncio
async def test_application_failure_isolated_from_ws_consumer():
    client = FakeStreamClient(["first", "second"], block=asyncio.Event())

    class FailingOnceApplication:
        def __init__(self):
            self.payloads = []

        async def handle_onebot(self, payload):
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                raise ValueError("handler failed")

    app = FailingOnceApplication()
    supervisor = make_supervisor(
        client,
        FakeMemoryWorker(),
        FakeInitiativeScheduler(),
        app,
    )
    await supervisor.start()
    await asyncio.sleep(0)
    assert app.payloads == ["first", "second"]
    assert supervisor.failure is None
    assert supervisor.ready
    await supervisor.stop()
    assert client.closed == 1


@pytest.mark.asyncio
async def test_transient_ws_disconnect_reconnects_inside_single_consumer():
    release = asyncio.Event()
    client = ReconnectingStreamClient(release)
    supervisor = make_supervisor(
        client,
        FakeMemoryWorker(),
        FakeInitiativeScheduler(),
        reconnect_initial_seconds=0.001,
        reconnect_max_seconds=0.01,
    )

    await supervisor.start()
    try:
        for _ in range(100):
            if client.stream_calls >= 2:
                break
            await asyncio.sleep(0.001)
        assert client.stream_calls >= 2
        assert supervisor.failure is None
        assert supervisor.ready
    finally:
        release.set()
        await supervisor.stop()
    assert client.closed == 1


@pytest.mark.asyncio
async def test_external_cancellation_propagates_to_all_loops():
    release = asyncio.Event()
    client = FakeStreamClient(block=release)
    memory = FakeMemoryWorker()
    initiative = FakeInitiativeScheduler()
    supervisor = make_supervisor(client, memory, initiative)

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.wait_for(supervisor._ready_event.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed == 1
    assert all(not supervisor.status(name).alive for name in ("ws", "memory", "initiative"))


@pytest.mark.asyncio
async def test_clean_ws_end_stops_background_loops():
    release = asyncio.Event()
    client = FakeStreamClient(["done"], block=release)
    memory = FakeMemoryWorker()
    initiative = FakeInitiativeScheduler()
    supervisor = make_supervisor(client, memory, initiative)

    await supervisor.start()
    release.set()
    await supervisor.wait()
    assert supervisor.failure is None
    assert client.closed == 0
    await supervisor.stop()
    assert client.closed == 1


def test_constructor_rejects_missing_lifecycle_interfaces():
    client = FakeStreamClient()
    with pytest.raises(TypeError, match="run_due"):
        make_supervisor(client, object(), FakeInitiativeScheduler())
    with pytest.raises(TypeError, match="tick"):
        make_supervisor(client, FakeMemoryWorker(), object())


def test_constructor_rejects_duplicate_worker_identity():
    with pytest.raises(ValueError, match="unique worker IDs"):
        make_supervisor(
            FakeStreamClient(),
            FakeMemoryWorker(),
            FakeInitiativeScheduler(),
            worker_id_factory=lambda _name: "same",
        )


@pytest.mark.asyncio
async def test_start_timeout_is_a_supervisor_error_and_closes_children():
    class HangingMemory(FakeMemoryWorker):
        async def run_due(self, now):
            self.entered.set()
            await asyncio.Event().wait()

    client = FakeStreamClient(block=asyncio.Event())
    supervisor = make_supervisor(
        client,
        HangingMemory(),
        FakeInitiativeScheduler(),
        startup_timeout_seconds=0.01,
    )
    await supervisor.start()
    assert supervisor.ready
    await supervisor.stop()
    assert client.closed == 1


@pytest.mark.asyncio
async def test_external_stop_during_start_wakes_start_without_waiting_for_timeout():
    client = FakeStreamClient(block=asyncio.Event())
    supervisor = make_supervisor(
        client,
        FakeMemoryWorker(),
        FakeInitiativeScheduler(),
        startup_timeout_seconds=10,
    )
    start_task = asyncio.create_task(supervisor.start())
    await asyncio.sleep(0)
    await supervisor.stop()
    with pytest.raises(SupervisorError, match="startup"):
        await start_task
    assert client.closed == 1
