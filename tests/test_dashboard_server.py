from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from qichi.readiness import READY_MARKER_VERSION, compute_build_id
from qichi.storage.database import Database
from qichi.storage.migrations import SCHEMA_VERSION
from qichi.dashboard import DashboardService
from scripts.dashboard_server import create_server

BUILD_ID = compute_build_id(Path(__file__).parents[1])


class _EmptyService:
    def snapshot(self):
        return {"health": {"ok": True, "reason": "ok"}}


class _RecordingService:
    def __init__(self):
        self.calls = []

    def snapshot(self, *, page, limit):
        self.calls.append((page, limit))
        return {
            "health": {"ok": True, "reason": "ok"},
            "trace_page": {"page": page, "limit": limit, "has_more": False},
        }


def _request(server, method="GET", path="/"):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}{path}"
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_get_static_and_api(tmp_path):
    static = tmp_path / "dashboard"
    static.mkdir()
    (static / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (static / "app.js").write_text("console.log(1)", encoding="utf-8")
    (static / "styles.css").write_text("body{}", encoding="utf-8")
    db = tmp_path / "db.sqlite3"
    Database(db).close()
    ready = tmp_path / "ready.json"
    ready.write_text(json.dumps({
        "schema": "qichi-ready",
        "version": READY_MARKER_VERSION,
        "instance_id": "test-instance",
        "lock_identity": "test-lock",
        "pid": 100,
        "parent_pid": 101,
        "owner_qq": "10001",
        "bot_qq": "20002",
        "database_schema_version": SCHEMA_VERSION,
        "last_recovered_sequence": -1,
        "ws_connection_id": "test-ws",
        "memory_worker_id": "test-memory",
        "initiative_worker_id": "test-initiative",
        "started_at_utc": "2026-09-03T00:00:00+00:00",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "context_window": 262144,
        "build_id": BUILD_ID,
    }), encoding="utf-8")
    server = create_server(DashboardService(db, ready), port=0, static_root=static)
    status, headers, body = _request(server, path="/")
    assert status == 200 and body == b"<h1>ok</h1>" and headers["Cache-Control"] == "no-store"
    server = create_server(DashboardService(db, ready), port=0, static_root=static)
    status, _, body = _request(server, path="/api/health")
    assert status == 200 and json.loads(body) == {"ok": True, "reason": "ok"}
    server = create_server(DashboardService(db, ready), port=0, static_root=static)
    status, _, body = _request(server, path="/api/snapshot")
    assert status == 200 and "health" in json.loads(body)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_mutating_methods_are_rejected(tmp_path, method):
    server = create_server(_EmptyService(), port=0, static_root=tmp_path)
    status, headers, _ = _request(server, method=method)
    assert status == 405 and headers["Allow"] == "GET"


def test_unknown_and_traversal_paths_are_not_served(tmp_path):
    static = tmp_path / "dashboard"
    static.mkdir()
    (static / "index.html").write_text("ok", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    server = create_server(_EmptyService(), port=0, static_root=static)
    assert _request(server, path="/missing")[0] == 404
    server = create_server(_EmptyService(), port=0, static_root=static)
    assert _request(server, path="/../secret.txt")[0] == 404


def test_bad_db_and_marker_health_fail_closed(tmp_path):
    db = tmp_path / "bad.sqlite3"
    db.write_text("not sqlite", encoding="utf-8")
    marker = tmp_path / "bad.json"
    marker.write_text("{", encoding="utf-8")
    server = create_server(DashboardService(db, marker), port=0, static_root=tmp_path)
    status, _, body = _request(server, path="/api/health")
    payload = json.loads(body)
    assert status == 200 and payload["ok"] is False and payload["reason"]


def test_server_binds_loopback(tmp_path):
    server = create_server(_EmptyService(), port=0, static_root=tmp_path)
    assert server.server_address[0] == "127.0.0.1"
    server.server_close()


@pytest.mark.parametrize(
    "query",
    [
        "page=not-an-integer",
        "page=1.5",
        "page=-1",
        "page=0",
        "page=",
        "page=1&page=2",
        "page=9223372036854775808",
        "page=9223372036854775807&limit=2",
        "limit=not-an-integer",
        "limit=-1",
        "limit=0",
        "limit=101",
        "limit=99999999999999999999999999999999999999",
    ],
)
def test_snapshot_rejects_invalid_pagination_before_calling_service(tmp_path, query):
    service = _RecordingService()
    server = create_server(service, port=0, static_root=tmp_path)

    status, _, body = _request(server, path=f"/api/snapshot?{query}")

    assert status == 400
    assert json.loads(body) == {"ok": False, "reason": "invalid query"}
    assert service.calls == []


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("", (1, 50)),
        ("?page=1&limit=1", (1, 1)),
        ("?page=2&limit=100", (2, 100)),
        ("?page=0007&limit=050", (7, 50)),
    ],
)
def test_snapshot_accepts_default_and_bounded_pagination(tmp_path, suffix, expected):
    service = _RecordingService()
    server = create_server(service, port=0, static_root=tmp_path)

    status, _, body = _request(server, path=f"/api/snapshot{suffix}")

    assert status == 200
    assert json.loads(body)["trace_page"] == {
        "page": expected[0],
        "limit": expected[1],
        "has_more": False,
    }
    assert service.calls == [expected]

class _FragmentService:
    def __init__(self):
        self.calls = []

    def fragment_detail(self, fragment_id):
        self.calls.append(("fragment", fragment_id))
        if not fragment_id:
            raise ValueError("invalid fragment id")
        return {"ok": True, "reason": "ok", "fragment": {"fragment_id": fragment_id}, "details": []}

    def recall_explanation(self, text):
        self.calls.append(("recall", text))
        if not text.strip():
            raise ValueError("invalid query text")
        return {"ok": True, "reason": "ok", "explicit_request": True, "dates": [], "by_date": [],
                "by_content": [], "newest": None, "windows": []}


def test_fragment_and_recall_routes_are_read_only_and_validate_input(tmp_path):
    static = tmp_path / "dashboard"
    static.mkdir()
    service = _FragmentService()

    server = create_server(service, port=0, static_root=static)
    status, _, body = _request(server, path="/api/fragment?id=frag-1")
    assert status == 200 and json.loads(body)["fragment"]["fragment_id"] == "frag-1"

    server = create_server(service, port=0, static_root=static)
    status, _, _ = _request(server, path="/api/fragment")
    assert status == 400, "缺少 id 必须拒绝，而不是偷偷返回全部"

    server = create_server(service, port=0, static_root=static)
    status, _, body = _request(server, path="/api/recall?q=" + urllib.parse.quote("细说九号那天"))
    assert status == 200 and json.loads(body)["explicit_request"] is True

    server = create_server(service, port=0, static_root=static)
    status, _, _ = _request(server, path="/api/recall?q=")
    assert status == 400

    assert all(call[0] in {"fragment", "recall"} for call in service.calls)

