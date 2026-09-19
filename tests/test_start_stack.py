from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest

import scripts.start_stack as start_stack


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def get_login_info(self):
        return {"user_id": "10001"}

    async def close(self):
        self.closed = True


class _FakeSupervisor:
    def __init__(self) -> None:
        self.wait_called = False

    async def wait(self):
        self.wait_called = True
        raise RuntimeError("worker failed")


class _FakeRuntime:
    def __init__(self) -> None:
        self.components = Namespace(supervisor=_FakeSupervisor())
        self.stop_called = False

    async def start(self):
        return self.components

    async def stop(self):
        self.stop_called = True


@pytest.mark.asyncio
async def test_run_waits_for_supervisor_failure_instead_of_blocking_forever(monkeypatch, tmp_path):
    client = _FakeClient()
    runtime = _FakeRuntime()
    config = Namespace(
        transport=Namespace(
            http_url="http://127.0.0.1:5700",
            websocket_url="ws://127.0.0.1:6700",
            access_token="token",
        ),
        llm=Namespace(primary=Namespace(timeout_seconds=25)),
    )

    monkeypatch.setattr(start_stack, "load_config", lambda _path: config)
    monkeypatch.setattr(start_stack, "OneBotClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(start_stack, "build_production_runtime", lambda *args, **kwargs: runtime)

    def forbidden_event():
        raise AssertionError("start_stack must observe supervisor.wait()")

    monkeypatch.setattr(start_stack.asyncio, "Event", forbidden_event)
    args = Namespace(
        config=None,
        evidence=None,
        ready=tmp_path / "ready",
        lock=tmp_path / "lock",
        exit_when_ready=False,
    )

    result = await start_stack._run(args)

    assert result == 1
    assert runtime.components.supervisor.wait_called is True
    assert runtime.stop_called is True
    assert client.closed is False


@pytest.mark.asyncio
async def test_run_stops_before_network_when_runtime_evidence_cannot_be_recovered(monkeypatch, tmp_path):
    config = Namespace(
        transport=Namespace(
            http_url="http://127.0.0.1:5700",
            websocket_url="ws://127.0.0.1:6700",
            access_token="token",
        ),
        llm=Namespace(primary=Namespace(timeout_seconds=25)),
    )
    monkeypatch.setattr(start_stack, "load_config", lambda _path: config)
    monkeypatch.setattr(
        start_stack,
        "reclaim_stale_runtime_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(start_stack.ReadinessError("unsafe")),
    )
    monkeypatch.setattr(
        start_stack,
        "OneBotClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network must not start")),
    )
    args = Namespace(
        config=None,
        evidence=None,
        ready=tmp_path / "ready",
        lock=tmp_path / "lock",
        exit_when_ready=False,
    )

    assert await start_stack._run(args) == 1


def test_effective_features_are_recorded_for_the_panel(tmp_path):
    import json
    import sqlite3
    from types import SimpleNamespace


    database = tmp_path / "qichi.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE runtime_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at_utc TEXT NOT NULL)"
    )
    connection.commit()
    connection.close()
    config = SimpleNamespace(
        storage=SimpleNamespace(database_path=str(database)),
        expression=SimpleNamespace(
            unicode_emoji=SimpleNamespace(enabled=True),
            qq_face=SimpleNamespace(enabled=False),
            message_reaction=SimpleNamespace(enabled=False),
            custom_sticker=SimpleNamespace(enabled=False),
        ),
        memory=SimpleNamespace(auto_commit="verified_explicit"),
        initiative=SimpleNamespace(enabled=True),
        vision=SimpleNamespace(enabled=True),
    )
    start_stack._record_effective_features(config, tmp_path)
    row = sqlite3.connect(database).execute(
        "SELECT value_json FROM runtime_meta WHERE key = 'runtime:features'"
    ).fetchone()
    assert row is not None, "the stack must publish the switches it actually loaded"
    features = json.loads(row[0])
    assert features["initiative_enabled"] is True
    assert features["qq_face_enabled"] is False
    assert features["unicode_emoji_enabled"] is True, "the emoji channel is a real switch"
    assert features["custom_sticker_enabled"] is False
    assert "expression_enabled" not in features, "there is no top-level expression switch"
    assert set(features) == {"initiative_enabled", "unicode_emoji_enabled", "custom_sticker_enabled",
                              "qq_face_enabled", "message_reaction_enabled", "memory_auto_commit",
                              "vision_available", "external_tools_available",
                              "external_search_ready", "external_image_search_ready"}
    assert features["memory_auto_commit"] == "verified_explicit"
    assert features["vision_available"] is True, "面板要看到进程实际加载的识图开关"
    rendered = json.dumps(features).lower()
    for secret in ("key", "token", "secret", "password"):
        assert secret not in rendered, secret


def test_recording_features_never_blocks_startup(tmp_path):
    import sys
    from types import SimpleNamespace


    broken = SimpleNamespace(
        storage=SimpleNamespace(database_path="no/such/directory/qichi.sqlite3"),
        expression=None,
        memory=None,
        initiative=None,
    )
    start_stack._record_effective_features(broken, tmp_path)  # must not raise
