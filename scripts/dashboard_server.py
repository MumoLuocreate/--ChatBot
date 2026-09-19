"""Local-only, read-only dashboard HTTP server."""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_ROOT = ROOT / "dashboard"
MAX_SQLITE_INTEGER = (1 << 63) - 1
MAX_PAGE_SIZE = 100


if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from qichi.dashboard import DashboardService


def _parse_positive_integer(values: list[str] | None, *, default: int, maximum: int) -> int:
    if values is None:
        return default
    if len(values) != 1:
        raise ValueError("query parameter must occur once")
    raw = values[0]
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError("query parameter must be an ASCII integer")
    normalized = raw.lstrip("0") or "0"
    upper = str(maximum)
    if len(normalized) > len(upper) or (len(normalized) == len(upper) and normalized > upper):
        raise ValueError("query parameter is too large")
    value = int(normalized)
    if value < 1:
        raise ValueError("query parameter must be positive")
    return value


def _parse_pagination(query_string: str) -> tuple[int, int]:
    query = parse_qs(query_string, keep_blank_values=True)
    page = _parse_positive_integer(query.get("page"), default=1, maximum=MAX_SQLITE_INTEGER)
    limit = _parse_positive_integer(query.get("limit"), default=50, maximum=MAX_PAGE_SIZE)
    if (page - 1) * limit > MAX_SQLITE_INTEGER:
        raise ValueError("pagination offset is too large")
    return page, limit


def _handler_factory(service: Any, static_root: Path):
    root = static_root.resolve()

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "QichiDashboard/1.0"

        def _send_json(self, payload: Mapping[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            version = payload.get("snapshot_version")
            if isinstance(version, str):
                self.send_header("ETag", f'"{version}"')
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            try:
                path.resolve().relative_to(root)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}.get(path.suffix, "application/octet-stream")
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlsplit(self.path).path)
            if path == "/api/health":
                try:
                    payload = service.health() if callable(getattr(service, "health", None)) else service.snapshot().get("health", {})
                    if not isinstance(payload, Mapping):
                        raise TypeError("health must return a mapping")
                    # Keep degraded health data readable by the local panel;
                    # the JSON ``ok`` flag carries the failure state without
                    # turning a diagnostic view into a fetch exception.
                    self._send_json(payload, HTTPStatus.OK)
                except Exception:  # service failures become fixed diagnostics, not tracebacks
                    self._send_json({"ok": False, "reason": "service unavailable"}, HTTPStatus.OK)
                return
            if path == "/api/snapshot":
                try:
                    page, limit = _parse_pagination(urlsplit(self.path).query)
                    query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    memory_page = _parse_positive_integer(query.get("memory_page"), default=1, maximum=MAX_SQLITE_INTEGER)
                    memory_limit = _parse_positive_integer(query.get("memory_limit"), default=50, maximum=MAX_PAGE_SIZE)
                    memory_status = query.get("memory_status", [None])
                    if len(memory_status) != 1 or (memory_status[0] is not None and memory_status[0] not in {"candidate", "active", "superseded", "rejected", "expired"}):
                        raise ValueError("invalid memory status")
                except (TypeError, ValueError, OverflowError):
                    self._send_json({"ok": False, "reason": "invalid query"}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    if any(key in query for key in ("memory_page", "memory_limit", "memory_status")):
                        payload = service.snapshot(page=page, limit=limit, memory_page=memory_page, memory_limit=memory_limit, memory_status=memory_status[0])
                    else:
                        payload = service.snapshot(page=page, limit=limit)
                    if not isinstance(payload, Mapping):
                        raise TypeError("snapshot must return a mapping")
                    version = payload.get("snapshot_version")
                    if isinstance(version, str) and self.headers.get("If-None-Match") == f'"{version}"':
                        self.send_response(HTTPStatus.NOT_MODIFIED)
                        self.send_header("ETag", f'"{version}"'); self.send_header("Cache-Control", "no-store"); self.end_headers(); return
                    self._send_json(payload)
                except Exception:
                    self._send_json({"ok": False, "reason": "service unavailable"}, HTTPStatus.OK)
                return
            if path == "/api/fragment":
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                try:
                    payload = service.fragment_detail(query.get("id", [""])[0])
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "reason": "invalid fragment id", "fragment": None,
                                     "details": []}, HTTPStatus.BAD_REQUEST)
                    return
                except Exception:
                    self._send_json({"ok": False, "reason": "service unavailable", "fragment": None,
                                     "details": []}, HTTPStatus.OK)
                    return
                self._send_json(payload)
                return
            if path == "/api/recall":
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                try:
                    payload = service.recall_explanation(query.get("q", [""])[0])
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "reason": "invalid query text"}, HTTPStatus.BAD_REQUEST)
                    return
                except Exception:
                    self._send_json({"ok": False, "reason": "service unavailable"}, HTTPStatus.OK)
                    return
                self._send_json(payload)
                return
            if path == "/":
                self._send_file(root / "index.html")
                return
            if path in {"/app.js", "/styles.css"}:
                self._send_file(root / path.lstrip("/"))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _method_not_allowed(self) -> None:
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_POST = _method_not_allowed
        do_PUT = _method_not_allowed
        do_PATCH = _method_not_allowed
        do_DELETE = _method_not_allowed

        def log_message(self, *_args: Any) -> None:
            return

    return DashboardHandler


def create_server(service: Any, *, port: int = 8765, static_root: str | Path = DEFAULT_DASHBOARD_ROOT) -> ThreadingHTTPServer:
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in range 0..65535")
    return ThreadingHTTPServer(("127.0.0.1", port), _handler_factory(service, Path(static_root)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local read-only Qichi dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "qichi.sqlite3")
    parser.add_argument("--ready", type=Path, default=ROOT / "runtime" / "qichi-ready.json")
    parser.add_argument("--lock", type=Path, default=ROOT / "runtime" / "qichi.lock")
    args = parser.parse_args(argv)
    service = DashboardService(
        args.db,
        args.ready,
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "context_window": 262144,
        },
        lock_path=args.lock,
    )
    server = create_server(service, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
