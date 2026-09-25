from __future__ import annotations

import json
import sqlite3
import hashlib
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

from qichi.initiative.scheduler import decode_initiative_state
from qichi.readiness import (
    PROJECT_ROOT,
    READY_MARKER_VERSION,
    LockRecord,
    ReadinessError,
    ReadyMarker,
    compute_build_id,
    marker_build_reason,
    pid_alive,
)
from qichi.storage.migrations import SCHEMA_VERSION


_TRACE_DROP = object()
_TRACE_MAX_INTEGER = (1 << 63) - 1
_TRACE_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:/"
)
# 必须覆盖 ContextBuilder 的每一个分类：投影会整张丢弃含未知键的 token 表，
# 漏一个键就等于面板上永远看不到本轮上下文构成（2026-09-11 实测）。
_TRACE_CONTEXT_CATEGORIES = frozenset({
    "role_core", "runtime_facts", "relationship_state", "memory_working_set",
    "recent_history", "memory_evidence", "memory_details", "memory_index",
    "earlier_history", "time_gaps", "direct_quotes", "current_input", "history",
    "memory_footprint",
})
_TRACE_OMITTED_CATEGORIES = frozenset({
    "relationship_state", "recent_history", "memory_evidence", "earlier_history",
    "history",
})
_TRACE_FAILURE_CATEGORIES = frozenset({
    "context:assembly", "dialogue:invalid_result", "llm:authentication",
    "llm:connection", "llm:error", "llm:model_not_found", "llm:protocol",
    "llm:rate_limit", "llm:request", "llm:server", "llm:timeout",
    "output_guard:empty", "output_guard:internal_sentinel",
    "output_guard:invalid", "output_guard:non_primary_route",
    "output_guard:not_text", "output_guard:raw_protocol_payload",
    "output_guard:system_prompt_leak", "output_guard:too_many_tokens",
    "output_guard:truncated", "response_protocol:duplicate_control_action",
    "response_protocol:empty_body", "response_protocol:empty_response",
    "response_protocol:invalid", "response_protocol:invalid_qq_marker",
    "response_protocol:marker_not_in_tail", "response_protocol:marker_not_standalone",
    "response_protocol:skip_not_allowed", "response_protocol:too_many_control_markers",
    "response_protocol:too_many_message_parts", "structural:invalid",
})
_TRACE_CANCELLATION_CATEGORIES = frozenset({
    "context_version_changed", "conversation_changed_after_generation",
    "dispatch_guard_rejected", "newer_input_after_generation", "newer_input_after_poke",
})
_TRACE_OUTBOX_STATUSES = frozenset({"pending", "dispatched", "sent", "unknown", "failed"})
# DeepSeek 官方定价（2026-09-22 抄自 https://api-docs.deepseek.com/zh-cn/quick_start/pricing）。
# 单位：元 / 百万 tokens；"空闲"是高峰价的一半。
# 高峰时段：北京时间 周一至周五（不含中国法定节假日）9:00-12:00、14:00-18:00。
# 官方另注明：模型名 deepseek-v4-flash 已下线（仍可调用、按 Flash 计价），新名是 deepseek-flash。
_LLM_PRICES: dict[str, dict[str, tuple[float, float]]] = {
    # 模型名/别名 -> {kind: (空闲价, 高峰价)}
    "deepseek-flash": {
        "hit": (0.02, 0.04), "miss": (1.0, 2.0), "output": (4.0, 8.0),
    },
    "deepseek-v4-flash": {
        "hit": (0.02, 0.04), "miss": (1.0, 2.0), "output": (4.0, 8.0),
    },
    "deepseek-v4-pro": {
        "hit": (0.15, 0.30), "miss": (4.5, 9.0), "output": (13.5, 27.0),
    },
}


def _is_peak_hour(when: datetime) -> bool:
    """北京时间工作日 9:00-12:00、14:00-18:00 为高峰，其余按空闲计价。"""

    local = when.astimezone(ZoneInfo("Asia/Shanghai"))
    if local.weekday() >= 5:
        return False
    return 9 <= local.hour < 12 or 14 <= local.hour < 18


def _price_cny(model: object, kind: str, when: datetime, tokens: int) -> float:
    """按官方单价折算金额；模型未知或 tokens 非正时返回 0（绝不猜）。"""

    if not isinstance(tokens, int) or tokens <= 0:
        return 0.0
    table = _LLM_PRICES.get(model if isinstance(model, str) else "")
    if table is None:
        return 0.0
    pair = table.get(kind)
    if pair is None:
        return 0.0
    idle, peak = pair
    rate = peak if _is_peak_hour(when) else idle
    return round(tokens / 1_000_000 * rate, 4)


_TRACE_FIELD_SCHEMAS: dict[str, tuple[object, ...]] = {
    "attempt_count": ("uint",),
    "cancellation_category": ("enum", _TRACE_CANCELLATION_CATEGORIES, False),
    "cache_hit_tokens": ("optional_uint",),
    "category_tokens": ("count_map", _TRACE_CONTEXT_CATEGORIES),
    "context_ms": ("number", False),
    "context_version": ("uint",),
    "event_kind": ("enum", frozenset({"text", "poke", "initiative"}), False),
    "expanded": ("bool",),
    "expression_key": ("identifier", 64, True),
    "expression_kind": ("enum", frozenset({"face", "reaction"}), True),
    "failure_category": ("enum", _TRACE_FAILURE_CATEGORIES, False),
    "finish_reason": ("enum", frozenset({"stop", "length", "tool_calls", "content_filter", "other"}), True),
    "first_byte_ms": ("number", True),
    "generation_ms": ("number", False),
    "has_quote_target": ("bool",),
    "input_budget_tokens": ("uint",),
    "input_tokens": ("uint",),
    # 2026-09-22 成本观测：本轮提示字符数，以及它与上一轮逐字相同的公共前缀字符数
    # （只有数字，没有正文）。与 cache_hit_tokens/input_tokens 并排看即可判定命中偏低
    # 是提示结构问题还是供应商侧的判定差异。
    "prompt_chars": ("uint",),
    "prompt_prefix_chars": ("uint",),
    "prompt_covers_previous": ("bool",),
    "memory_detail_count": ("uint",),
    "memory_detail_fragments": ("id_list", 64),
    # 2026-09-12 T1：钥匙（授权）与定位分开记。key 是冻结计划 §2.1 的那把钥匙，
    # reason 是这一轮实际选了哪条定位规则。
    # 两个**只为读旧轨迹**保留的取值（代码已不再产出）：content（词面定位，T2 之后
    # 词面不再开门）与 recent_guess（猜最近一段，T2 之前就删了）。白名单是读侧过滤，
    # 删掉它们只会让历史行少一个字段，所以留着。
    "memory_detail_indexed": ("bool",),
    "memory_detail_key": (
        "enum",
        frozenset({"date_now", "date_missing", "today", "quote", "verbatim", "date_inherited", "correction", "ambiguous", "none"}),
        False,
    ),
    "memory_detail_match_count": ("uint",),
    "memory_recall_note": ("enum", frozenset({"missing_day"}), True),
    "memory_detail_reason": (
        "enum",
        frozenset({"quote", "day", "day_missing", "content", "verbatim", "correction", "ambiguous", "recent_guess", "none"}),
        False,
    ),
    "memory_evidence_ids": ("id_list", 256),
    "memory_reasons": ("memory_reason_map",),
    "memory_scores": ("number_map",),
    "message_part_count": ("uint",),
    "model_id": ("identifier", 128, False),
    "model_tier": ("enum", frozenset({"text", "vision"}), True),
    "model_route": ("enum", frozenset({"primary"}), False),
    "omitted_counts": ("count_map", _TRACE_OMITTED_CATEGORIES),
    "omitted_ids": ("id_list", 2048),
    "outbound_event_id": ("identifier", 160, True),
    "outbox_statuses": ("status_map",),
    "output_tokens": ("uint",),
    "platform_message_id_present": ("bool",),
    "provider_latency_ms": ("number", False),
    "quote_resolution_status": ("enum", frozenset({"none", "unavailable", "resolved"}), False),
    "quoted_event_id": ("identifier", 160, True),
    "quoted_pinned_tokens": ("uint",),
    "reason": ("enum", _TRACE_FAILURE_CATEGORIES | _TRACE_CANCELLATION_CATEGORIES, False),
    "reasoning_tokens": ("optional_uint",),
    "relationship_memory_ids": ("id_list", 256),
    "reply_target_handle": ("identifier", 32, True),
    "result_type": ("enum", frozenset({"reply", "skip", "failure"}), False),
    "retrieval_candidate_ids": ("id_list", 256),
    "retrieval_degraded": ("bool",),
    "retrieval_mode": ("enum", frozenset({"fts5_trigram", "like_short_query", "like_degraded"}), False),
    "retrieval_query_source_count": ("uint",),
    "retry_count": ("uint",),
    # 联网（2026-09-14）：这一轮查了什么、查成功没有、资料多大。只报事实，不报正文。
    "tool_name": ("enum", frozenset({"web_search", "image_search"}), True),
    "tool_query": ("text", 200, True),
    "tool_ok": ("bool",),
    "tool_degraded": ("identifier", 48, True),
    "tool_elapsed_ms": ("optional_uint",),
    "tool_result_chars": ("optional_uint",),
    # 宽限窗口：这一轮挂的是上一轮那张图（张数）。
    "images_carried": ("optional_uint",),
    "selected_history_event_ids": ("id_list", 2048),
    "selected_ids": ("id_list", 2048),
    "selected_input_tokens": ("uint",),
    "selected_memory_ids": ("id_list", 256),
    "sequence": ("uint",),
    "stage": ("enum", frozenset({"received", "context", "generation", "delivery"}), False),
    "status": ("enum", frozenset({"received", "processing", "processed", "sent", "unknown", "failed", "cancelled", "skipped"}), False),
    "trigger_event_id": ("identifier", 160, False),
    "window_tokens": ("uint",),
}


def _trace_identifier(value: object, maximum_length: int, nullable: bool = False) -> object:
    if value is None:
        return None if nullable else _TRACE_DROP
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum_length
        or any(character not in _TRACE_IDENTIFIER_CHARACTERS for character in value)
    ):
        return _TRACE_DROP
    return value


def _trace_text(value: object, maximum_length: int, nullable: bool = False) -> object:
    """自由文本（检索词这种）：允许任何可见字符，但仍然有长度上限、不许控制字符。

    标识符那一档只认 ASCII，而检索词是中文——所以这一档是必需的，不是放宽。
    """

    if value is None:
        return None if nullable else _TRACE_DROP
    if not isinstance(value, str) or not value or len(value) > maximum_length:
        return _TRACE_DROP
    if any(character < " " or character == "\x7f" for character in value):
        return _TRACE_DROP
    return value


def _trace_uint(value: object, *, nullable: bool = False) -> object:
    if value is None:
        return None if nullable else _TRACE_DROP
    if type(value) is not int or not 0 <= value <= _TRACE_MAX_INTEGER:
        return _TRACE_DROP
    return value


def _trace_number(value: object, *, nullable: bool) -> object:
    if value is None:
        return None if nullable else _TRACE_DROP
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= _TRACE_MAX_INTEGER
    ):
        return _TRACE_DROP
    return value


def _trace_memory_reason(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 128:
        return False
    for part in value.split(","):
        if part in {
            "normalized_fact", "exact_quote", "exact_query_in_fact",
            "exact_query_in_quote", "no lexical hit",
        }:
            continue
        prefix = "trigram_overlap:fact="
        if not part.startswith(prefix) or ";quote=" not in part:
            return False
        fact, quote_count = part[len(prefix):].split(";quote=", 1)
        if not fact.isdecimal() or not quote_count.isdecimal():
            return False
    return True


def _trace_mapping(value: object, *, kind: str, allowed_keys: frozenset[str] | None = None) -> object:
    if not isinstance(value, dict) or len(value) > 256:
        return _TRACE_DROP
    projected: dict[str, object] = {}
    for key, item in value.items():
        if allowed_keys is not None:
            if key not in allowed_keys:
                return _TRACE_DROP
        elif _trace_identifier(key, 160) is _TRACE_DROP:
            return _TRACE_DROP
        if kind == "count":
            clean = _trace_uint(item)
        elif kind == "number":
            clean = _trace_number(item, nullable=False)
        elif kind == "memory_reason":
            clean = item if _trace_memory_reason(item) else _TRACE_DROP
        elif kind == "status":
            clean = item if isinstance(item, str) and item in _TRACE_OUTBOX_STATUSES else _TRACE_DROP
        else:
            return _TRACE_DROP
        if clean is _TRACE_DROP:
            return _TRACE_DROP
        projected[key] = clean
    return projected


def _project_trace_value(schema: tuple[object, ...], value: object) -> object:
    kind = schema[0]
    if kind == "uint":
        return _trace_uint(value)
    if kind == "optional_uint":
        return _trace_uint(value, nullable=True)
    if kind == "number":
        return _trace_number(value, nullable=bool(schema[1]))
    if kind == "bool":
        return value if type(value) is bool else _TRACE_DROP
    if kind == "enum":
        allowed, nullable = schema[1], bool(schema[2])
        if value is None:
            return None if nullable else _TRACE_DROP
        return value if isinstance(value, str) and value in allowed else _TRACE_DROP
    if kind == "identifier":
        return _trace_identifier(value, int(schema[1]), bool(schema[2]))
    if kind == "text":
        return _trace_text(value, int(schema[1]), bool(schema[2]))
    if kind == "id_list":
        if not isinstance(value, list) or len(value) > int(schema[1]):
            return _TRACE_DROP
        clean = [_trace_identifier(item, 160) for item in value]
        return clean if all(item is not _TRACE_DROP for item in clean) else _TRACE_DROP
    if kind == "count_map":
        return _trace_mapping(value, kind="count", allowed_keys=schema[1])
    if kind == "number_map":
        return _trace_mapping(value, kind="number")
    if kind == "memory_reason_map":
        return _trace_mapping(value, kind="memory_reason")
    if kind == "status_map":
        return _trace_mapping(value, kind="status")
    return _TRACE_DROP


def _marker_conflict_reason(value: Mapping[str, object]) -> str:
    """Name a database schema conflict instead of a bare "marker invalid".

    Only the two version integers are echoed; nothing else from the marker is
    projected, and every other structural problem still reports "marker
    invalid".
    """
    schema = value.get("database_schema_version")
    if type(schema) is int and schema != SCHEMA_VERSION:
        return f"database schema {schema} does not match code schema {SCHEMA_VERSION}"
    return "marker invalid"


class DashboardService:
    """Read operational state from SQLite without any write-capable handle."""

    def __init__(self, database_path: str | Path, marker_path: str | Path, runtime_info: Mapping[str, object] | None = None, *, lock_path: str | Path | None = None, pid_probe: object | None = None, clock: object | None = None, code_root: str | Path | None = None):
        self.database_path = Path(database_path)
        self.marker_path = Path(marker_path)
        self.runtime_info = dict(runtime_info or {})
        self.lock_path = Path(lock_path) if lock_path is not None else None
        self.pid_probe = pid_probe if callable(pid_probe) else self._default_pid_probe
        self.clock = clock if callable(clock) else lambda: datetime.now(timezone.utc)
        self.code_root = Path(code_root) if code_root is not None else PROJECT_ROOT

    def _memory_stall_reason(self, connection) -> str | None:
        """整合是不是卡住了（2026-09-22）。

        只在**存在终态失败的整合任务**时报警。单看「水位落后多少条」会误报：一段长时间
        连续的对话本来就只在静默 30 分钟后才整合一次，落后几百条是正常的。终态失败则一定
        是可处理的异常——真机上一条约文里的坏标点把水位钉死 8 小时，而当时**没有任何地方**
        会提醒，是用户主动问起才被发现。
        """

        try:
            row = connection.execute(
                "SELECT start_sequence, end_sequence, failure_category FROM memory_session_jobs "
                "WHERE status='failed' ORDER BY updated_at_utc LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        category = row["failure_category"] or "unknown"
        return "记忆整合卡住：有终态失败的任务（seq %s-%s，类别 %s），等重开" % (
            row["start_sequence"], row["end_sequence"], category
        )

    def snapshot(self, *, page: int = 1, limit: int = 50, memory_page: int = 1, memory_limit: int = 50, memory_status: str | None = None) -> dict[str, object]:
        page = max(1, int(page)); limit = min(100, max(1, int(limit)))
        marker, marker_reason = self._read_marker()
        result: dict[str, object] = {
            "health": {"ok": marker_reason is None, "reason": marker_reason or "ok"},
            "ready": {"ok": marker is not None, "reason": marker_reason or "ready", "marker": marker},
            "runtime": self._runtime(marker),
            "cursor": {"context_version": None, "last_processed_sequence": None, "last_user_activity_utc": None, "presence_topic_cursor": None},
            "counts": {"events": 0, "outbox": 0, "memory": 0},
            "recent_events": [], "outbox": [], "outbox_status": None, "memory_status": [],
            "memories": [], "response_audit": [], "fragments": [],
            "memory_worker": None,
            "initiative": {"raw_state": None, "enabled": None, "next_attempt_at_utc": None, "last_activity_at_utc": None},
            "traces": [], "trace_page": {"page": page, "limit": limit, "has_more": False},
            # The switches the running process actually loaded, or None when the
            # process did not record them; the panel must never guess.
            "features": None, "features_evidence": None,
            "operations": None,
        }
        ready_reason = self._ready_reason(marker, marker_reason)
        result["ready"] = {"ok": ready_reason is None, "reason": ready_reason or "ready", "marker": marker}
        projection_marker = marker if ready_reason is None else None
        result["runtime"] = self._runtime(projection_marker)
        if ready_reason is not None:
            result["health"] = {"ok": False, "reason": ready_reason}
        try:
            connection = self._connect_read_only()
        except (OSError, sqlite3.Error) as error:
            result["health"] = {"ok": False, "reason": f"database unavailable: {type(error).__name__}"}
            return result
        stall_reason = self._memory_stall_reason(connection)
        if stall_reason is not None:
            # 两条理由都要留住：marker 不健康（build 不一致等）时若被卡住理由顶掉，
            # 反而会掩盖另一条更该看见的异常。
            current = result["health"].get("reason")
            combined = stall_reason if not current or current == "ok" else f"{current}；{stall_reason}"
            result["health"] = {"ok": False, "reason": combined}
        try:
            try:
                # Keep every projection in one SQLite snapshot while the live bot writes in WAL mode.
                connection.execute("BEGIN")
                self._populate(result, connection, projection_marker, page=page, limit=limit, memory_page=memory_page, memory_limit=memory_limit, memory_status=memory_status)
            except sqlite3.Error as error:
                result["health"] = {"ok": False, "reason": f"database unavailable: {type(error).__name__}"}
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
        result["snapshot_version"] = self._snapshot_version(result)
        return result

    def health(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {"ok": snapshot["health"]["ok"], "reason": snapshot["health"]["reason"]}

    @staticmethod
    def _default_pid_probe(pid: int) -> bool:
        return pid_alive(pid)

    def _ready_reason(self, marker: dict[str, object] | None, marker_reason: str | None) -> str | None:
        if marker_reason is not None or marker is None:
            return marker_reason
        # A marker only vouches for the exact code that wrote it.  This is
        # checked before anything else, including the lock probe, because a
        # stale stack must never be reported as a healthy one.
        try:
            expected_build_id = compute_build_id(self.code_root)
        except (OSError, ReadinessError):
            return "build evidence unavailable"
        try:
            parsed = ReadyMarker.from_dict(marker)
        except (ReadinessError, TypeError):
            return "marker invalid"
        build_reason = marker_build_reason(parsed, expected_build_id=expected_build_id)
        if build_reason is not None:
            return build_reason
        if self.lock_path is None:
            return None
        try:
            lock_value = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if not isinstance(lock_value, Mapping):
                return "runtime evidence invalid"
            lock = LockRecord.from_dict(lock_value)
            for key in ("instance_id", "lock_identity", "pid", "parent_pid"):
                if getattr(lock, key) != marker.get(key):
                    return "runtime evidence invalid"
            if not self.pid_probe(int(marker["pid"])) and not self.pid_probe(int(marker["parent_pid"])):
                return "runtime evidence invalid"
            started = datetime.fromisoformat(str(marker["started_at_utc"]))
            age = (self.clock().astimezone(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
            if age < 0:
                return "runtime evidence stale"
        except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError, ReadinessError):
            return "runtime evidence invalid"
        return None

    def _connect_read_only(self) -> sqlite3.Connection:
        path = self.database_path.resolve()
        uri = "file:" + quote(path.as_posix(), safe="/:\\") + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _read_marker(self) -> tuple[dict[str, object] | None, str | None]:
        try:
            raw = self.marker_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, "marker missing"
        except (OSError, UnicodeError):
            return None, "marker invalid"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None, "marker invalid"
        if not isinstance(value, Mapping):
            return None, "marker invalid"
        try:
            marker = ReadyMarker.from_dict(value)
        except (ReadinessError, TypeError):
            return None, _marker_conflict_reason(value)
        return marker.to_dict(), None

    def _runtime(self, marker: dict[str, object] | None) -> dict[str, object]:
        fields = ("provider", "model", "context_window", "owner_qq", "bot_qq")
        return {field: marker.get(field) if marker is not None else None for field in fields}

    def _populate(self, result: dict[str, object], connection: sqlite3.Connection, marker: dict[str, object] | None, *, page: int = 1, limit: int = 50, memory_page: int = 1, memory_limit: int = 50, memory_status: str | None = None) -> None:
        owner = self._runtime(marker).get("owner_qq")
        cursor = connection.execute("SELECT context_version,last_processed_sequence,last_user_activity_utc,presence_topic_cursor FROM conversation_cursors WHERE conversation_id=? ORDER BY conversation_id LIMIT 1", (owner,)).fetchone() if owner is not None else None
        if cursor is not None:
            result["cursor"] = dict(cursor)
        outbox_status = [dict(row) for row in connection.execute(
            "SELECT o.status,COUNT(*) AS count FROM outbox o "
            "JOIN conversation_events e ON e.event_id=o.event_id "
            "WHERE e.conversation_id=? GROUP BY o.status ORDER BY o.status", (owner,)
        ).fetchall()] if owner is not None else None
        result["counts"] = {"events": connection.execute("SELECT COUNT(*) FROM conversation_events WHERE conversation_id=?", (owner,)).fetchone()[0] if owner is not None else 0, "outbox": sum(item["count"] for item in outbox_status) if outbox_status is not None else 0, "memory": connection.execute("SELECT COUNT(*) FROM memory_records m WHERE EXISTS (SELECT 1 FROM memory_evidence me JOIN conversation_events e ON e.event_id=me.event_id WHERE me.memory_id=m.memory_id AND e.conversation_id=?)", (owner,)).fetchone()[0] if owner is not None else 0}
        result["recent_events"] = self._events(connection, owner)
        result["outbox"] = self._outbox(connection, owner)
        result["outbox_status"] = outbox_status
        result["memory_status"] = [dict(row) for row in connection.execute(
            "SELECT m.status,COUNT(*) AS count FROM memory_records m WHERE EXISTS ("
            "SELECT 1 FROM memory_evidence me JOIN conversation_events e ON e.event_id=me.event_id "
            "WHERE me.memory_id=m.memory_id AND e.conversation_id=?) GROUP BY m.status ORDER BY m.status", (owner,)
        ).fetchall()] if owner is not None else []
        memories, memory_page_info = self._memories(connection, owner, page=memory_page, limit=memory_limit, status=memory_status)
        result["memories"] = memories
        result["memory_page"] = memory_page_info
        result["memory_worker"] = self._memory_worker(connection, owner)
        # marker 不可用时（build 不一致等）回落到进程记录的运行时信息，别让"算不出钱"再添一层困惑。
        marker_model = (marker.get("model") if isinstance(marker, dict) else None) or self.runtime_info.get("model")
        result["memory_usage"] = self._memory_usage_24h(connection, marker_model)
        result["chat_usage"] = self._chat_usage_24h(connection, marker_model)
        result["cost_today"] = {
            "currency": "CNY",
            "model": marker_model,
            "chat_cny": result["chat_usage"]["cost_cny"],
            "memory_cny": result["memory_usage"]["cost_cny"],
            "total_cny": round(
                result["chat_usage"]["cost_cny"] + result["memory_usage"]["cost_cny"], 3
            ),
            "note": "按 DeepSeek 官方单价折算（高峰/空闲自动判别）；模型未知时只显示 token",
        }
        result["fragments"] = self._fragments(connection, owner)
        result["operations"] = self._operations(connection, marker)
        result["response_audit"] = self._response_audit(connection, owner)
        result["initiative"] = self._initiative(connection, owner)
        result["features"], result["features_evidence"] = self._features(connection)
        traces, has_more = self._traces(connection, owner, page=page, limit=limit)
        result["traces"] = traces
        result["trace_page"] = {"page": page, "limit": limit, "has_more": has_more}

    @staticmethod
    def _snapshot_version(result: Mapping[str, object]) -> str:
        public = {key: value for key, value in result.items() if key != "snapshot_version"}
        raw = json.dumps(public, ensure_ascii=False, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    @staticmethod
    def _traces(connection: sqlite3.Connection, owner: object, *, page: int, limit: int) -> tuple[list[dict[str, object]], bool]:
        if owner is None:
            return [], False
        offset = (page - 1) * limit
        trace_rows = connection.execute(
            "SELECT trace_id, MAX(occurred_at_utc) AS latest FROM turn_trace_events WHERE conversation_id=? "
            "GROUP BY trace_id ORDER BY latest DESC, trace_id DESC LIMIT ? OFFSET ?", (owner, limit + 1, offset)
        ).fetchall()
        has_more = len(trace_rows) > limit
        trace_rows = trace_rows[:limit]
        if not trace_rows:
            return [], has_more
        trace_ids = [row["trace_id"] for row in trace_rows]
        placeholders = ",".join("?" for _ in trace_ids)
        rows = connection.execute(
            "SELECT trace_id,conversation_id,trigger_event_id,source,phase,occurred_at_utc,details_json "
            f"FROM turn_trace_events WHERE conversation_id=? AND trace_id IN ({placeholders}) "
            "ORDER BY occurred_at_utc,trace_event_id", (owner, *trace_ids)
        ).fetchall()
        grouped: dict[str, dict[str, object]] = {}
        for row in rows:
            item = grouped.setdefault(row["trace_id"], {"trace_id": row["trace_id"], "conversation_id": row["conversation_id"], "trigger_event_id": row["trigger_event_id"], "source": row["source"], "phases": []})
            try:
                details = json.loads(row["details_json"] or "{}")
                details = DashboardService._safe_trace_details(details)
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            item["phases"].append({"phase": row["phase"], "occurred_at_utc": row["occurred_at_utc"], "details": details})
        order = {trace_id: index for index, trace_id in enumerate(trace_ids)}
        values = sorted(grouped.values(), key=lambda item: order[item["trace_id"]])
        return values, has_more

    @staticmethod
    def _safe_trace_details(value: object) -> dict[str, object]:
        forbidden = {"api_key", "access_token", "authorization", "prompt", "messages", "completion", "reasoning", "chain_of_thought", "raw_payload", "text", "secret", "private_message", "full_prompt"}
        if not isinstance(value, dict):
            return {}
        if any(not isinstance(key, str) or key.casefold() in forbidden for key in value):
            return {}
        projected: dict[str, object] = {}
        for key, child in value.items():
            schema = _TRACE_FIELD_SCHEMAS.get(key)
            if schema is None:
                continue
            clean = _project_trace_value(schema, child)
            if clean is not _TRACE_DROP:
                projected[key] = clean
        return projected

    @staticmethod
    def _events(connection: sqlite3.Connection, owner: object) -> list[dict[str, object]]:
        rows = connection.execute("SELECT e.sequence,e.actor,e.direction,e.kind,e.status,e.occurred_at_utc,e.text,e.event_id,COALESCE(e.platform_message_id,m.platform_message_id) AS platform_message_id,e.reply_to_event_id FROM conversation_events e LEFT JOIN platform_message_map m ON m.event_id=e.event_id WHERE e.conversation_id=? ORDER BY e.occurred_at_utc DESC,e.sequence DESC LIMIT 50", (owner,)).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _outbox(connection: sqlite3.Connection, owner: object) -> list[dict[str, object]]:
        rows = connection.execute("SELECT o.operation_key,o.event_id,o.status,o.attempt_count,o.payload_json,o.updated_at_utc FROM outbox o JOIN conversation_events e ON e.event_id=o.event_id WHERE e.conversation_id=? ORDER BY o.updated_at_utc DESC,o.operation_key DESC LIMIT 50", (owner,)).fetchall()
        records = []
        for row in rows:
            action_kind = None
            try:
                payload = json.loads(row["payload_json"])
                if isinstance(payload, dict) and isinstance(payload.get("action_kind"), str):
                    action_kind = payload["action_kind"]
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            records.append({"operation_key": row["operation_key"], "event_id": row["event_id"], "status": row["status"], "attempt_count": row["attempt_count"], "action_kind": action_kind, "updated_at_utc": row["updated_at_utc"]})
        return records

    @staticmethod
    def _memory_usage_24h(connection: sqlite3.Connection, model: object = None) -> dict[str, object]:
        """近 24 小时记忆后台的 token 用量与折算金额（按阶段分开）。

        2026-09-22：用户问「一天不到 20 块就没了」，而**记忆后台此前完全不记账**，
        只能按窗口规模猜。这里把 worker 写进 job 账本的用量汇总出来，好回答
        「钱花在抽取还是明细补跑」——并按官方单价（见 _LLM_PRICES）折算成人民币。
        """

        empty = {"window_hours": 24, "totals": {}, "jobs": 0, "cost_cny": 0.0}
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        try:
            rows = connection.execute(
                "SELECT occurred_at_utc, details_json FROM memory_job_events WHERE occurred_at_utc >= ?",
                (since,),
            ).fetchall()
        except sqlite3.Error:
            return empty
        totals: dict[str, int] = {}
        calls = 0
        cost = 0.0
        for row in rows:
            try:
                details = json.loads(row["details_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(details, dict):
                continue
            numbers: dict[str, int] = {}
            for key, value in details.items():
                if not isinstance(key, str) or not key.endswith(("_tokens", "_calls")):
                    continue
                if not isinstance(value, str) or not value.isdigit():
                    continue
                numbers[key] = int(value)
                totals[key] = totals.get(key, 0) + numbers[key]
            if not numbers:
                continue
            calls += 1
            try:
                when = datetime.fromisoformat(row["occurred_at_utc"])
            except (TypeError, ValueError):
                when = None
            if when is not None:
                for prefix in ("extraction", "detail"):
                    hit = numbers.get(prefix + "_cache_hit_tokens", 0)
                    cost += _price_cny(model, "hit", when, hit)
                    cost += _price_cny(model, "output", when, numbers.get(prefix + "_output_tokens", 0))
                    cost += _price_cny(
                        model, "miss", when, max(0, numbers.get(prefix + "_input_tokens", 0) - hit)
                    )
        return {"window_hours": 24, "totals": totals, "jobs": calls, "cost_cny": round(cost, 3)}

    @staticmethod
    def _chat_usage_24h(connection: sqlite3.Connection, model: object = None) -> dict[str, object]:
        """近 24 小时对话路径的 token 用量与折算金额（来自 generation trace）。

        与记忆侧同一把尺子，好让「哪一侧更贵」一眼可见——2026-09-22 实测对话约 ¥1~3、
        而卡死那天的重试风暴才是大头。
        """

        empty = {"window_hours": 24, "turns": 0, "input_tokens": 0, "cache_hit_tokens": 0,
                 "output_tokens": 0, "cost_cny": 0.0}
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        try:
            rows = connection.execute(
                "SELECT occurred_at_utc, details_json FROM turn_trace_events"
                " WHERE phase='generation' AND occurred_at_utc >= ?",
                (since,),
            ).fetchall()
        except sqlite3.Error:
            return empty
        totals = {"input_tokens": 0, "cache_hit_tokens": 0, "output_tokens": 0}
        cost = 0.0
        turns = 0
        for row in rows:
            try:
                details = json.loads(row["details_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(details, dict):
                continue
            hits = details.get("cache_hit_tokens") or 0
            inputs = details.get("input_tokens") or 0
            outputs = details.get("output_tokens") or 0
            if not all(isinstance(value, int) and value >= 0 for value in (hits, inputs, outputs)):
                continue
            totals["input_tokens"] += inputs
            totals["cache_hit_tokens"] += hits
            totals["output_tokens"] += outputs
            turns += 1
            try:
                when = datetime.fromisoformat(row["occurred_at_utc"])
            except (TypeError, ValueError):
                continue
            cost += _price_cny(model, "hit", when, hits)
            cost += _price_cny(model, "miss", when, max(0, inputs - hits))
            cost += _price_cny(model, "output", when, outputs)
        return {**empty, **totals, "turns": turns, "cost_cny": round(cost, 3)}

    @staticmethod
    def _memory_worker(connection: sqlite3.Connection, owner: object) -> dict[str, object] | None:
        if owner is None:
            return None
        last_result = None
        event = connection.execute(
            "SELECT action,revision,fragment_key,start_sequence,end_sequence,"
            "failure_category,attempt_count,occurred_at_utc,details_json "
            "FROM memory_job_events WHERE conversation_id=? "
            "ORDER BY occurred_at_utc DESC,job_event_id DESC LIMIT 1",
            (owner,),
        ).fetchone()
        if event is not None:
            try:
                details = json.loads(event["details_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                details = {}
            last_result = {
                "action": event["action"],
                "revision": event["revision"],
                "fragment_key": event["fragment_key"],
                "start_sequence": event["start_sequence"],
                "end_sequence": event["end_sequence"],
                "failure_category": event["failure_category"],
                "attempt_count": event["attempt_count"],
                "occurred_at_utc": event["occurred_at_utc"],
                "details": DashboardService._safe_memory_job_details(details),
            }
        row = connection.execute(
            "SELECT conversation_id,revision,fragment_key,start_sequence,end_sequence,"
            "anchor_event_id,anchor_sequence,anchor_received_at_utc,deadline_utc,context_version,"
            "status,claim_token,claim_owner,claim_lease_until_utc,next_retry_at_utc,attempt_count,failure_category "
            "FROM memory_session_jobs WHERE conversation_id=?", (owner,)
        ).fetchone()
        if row is None:
            return {"last_result": last_result} if last_result is not None else None
        value = dict(row)
        status = value.get("status")
        if status not in {"pending", "claimed", "retry", "failed"}:
            return {"evidence": "invalid", "healthy": False}
        claims = (value.get("claim_owner"), value.get("claim_lease_until_utc"), value.get("claim_token"))
        retry = value.get("next_retry_at_utc")
        failure = value.get("failure_category")
        valid = ((status == "pending" and all(item is None for item in (*claims, retry, failure))) or
                 (status == "claimed" and all(isinstance(item, str) and item for item in claims) and retry is None and failure is None) or
                 (status == "retry" and all(item is None for item in claims) and isinstance(retry, str) and isinstance(failure, str) and retry and failure) or
                 (status == "failed" and all(item is None for item in claims) and retry is None and isinstance(failure, str) and bool(failure)))
        if not valid or not isinstance(value.get("conversation_id"), str) or int(value.get("revision", 0)) < 1:
            return {"evidence": "invalid", "healthy": False}
        return value | {"evidence": "state", "healthy": True, "last_result": last_result}

    @staticmethod
    def _safe_memory_job_details(value: object) -> dict[str, str]:
        """Project only bounded, public memory consolidation telemetry."""
        if not isinstance(value, dict):
            return {}
        allowed = {
            "candidate_count", "review_count", "dropped_candidate_count",
            "dropped_review_count", "empty_result", "parse_error_code",
            "candidate_error_codes", "review_error_codes", "provider_error",
            "outcome_kind", "outcome_reason_code",
            # 2026-09-22 成本可见性：记忆后台的 token 用量（worker 在每次任务结束时写入）。
            "extraction_input_tokens", "extraction_output_tokens",
            "extraction_cache_hit_tokens", "extraction_reasoning_tokens", "extraction_calls",
            "detail_input_tokens", "detail_output_tokens",
            "detail_cache_hit_tokens", "detail_reasoning_tokens", "detail_calls",
        }
        usage_keys = {key for key in allowed if key.endswith(("_tokens", "_calls"))}
        provider_errors = {
            "timeout", "connection", "rate_limit", "server_error", "protocol",
            "authentication", "model_not_found", "request", "unknown",
        }
        parse_codes = {
            "all_items_invalid", "candidate_duplicate", "candidate_evidence",
            "candidate_invalid", "candidate_schema", "candidate_time",
            "empty_response", "invalid_json", "item_limit", "response_schema",
            "review_duplicate", "review_evidence", "review_invalid",
            "review_schema", "review_target", "review_time",
        }
        outcome_reasons = {
            "explicit_user_preference", "historical_episode", "bilateral_bounded_agreement",
            "existing_memory_review", "candidate_proposed", "review_proposed", "nothing_new",
            "temporary_scene_or_roleplay", "ambiguous_scope", "missing_bilateral_acceptance",
            "insufficient_user_evidence", "candidate_evidence_invalid",
        }
        result: dict[str, str] = {}
        for key, item in value.items():
            if key not in allowed or not isinstance(item, str) or len(item) > 256:
                continue
            if key.endswith("_count") and (not item.isascii() or not item.isdigit() or not 0 <= int(item) <= 12):
                continue
            if key in usage_keys and (not item.isascii() or not item.isdigit() or int(item) > 10**12):
                continue
            if key == "empty_result" and item != "1":
                continue
            if key == "provider_error" and item not in provider_errors:
                continue
            if key == "parse_error_code" and item not in parse_codes:
                continue
            if key in {"candidate_error_codes", "review_error_codes"}:
                codes = item.split(",")
                if not codes or len(codes) > 12 or any(code not in parse_codes for code in codes):
                    continue
            if key == "outcome_kind" and item not in {"memory_found", "no_persistent_memory"}:
                continue
            if key == "outcome_reason_code" and item not in outcome_reasons:
                continue
            result[key] = item
        kind = result.get("outcome_kind")
        reason = result.get("outcome_reason_code")
        if kind is None or reason is None:
            result.pop("outcome_kind", None)
            result.pop("outcome_reason_code", None)
        return result

    @staticmethod
    def _memories(connection: sqlite3.Connection, owner: object, *, page: int = 1, limit: int = 50, status: str | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
        if status is not None and status not in {"candidate", "active", "superseded", "rejected", "expired"}:
            raise ValueError("invalid memory status")
        offset = (page - 1) * limit
        status_clause = " AND m.status=?" if status else ""
        status_params = (status,) if status else ()
        total_row = connection.execute(
            "SELECT COUNT(*) FROM memory_records m WHERE EXISTS (SELECT 1 FROM memory_evidence me JOIN conversation_events e ON e.event_id=me.event_id WHERE me.memory_id=m.memory_id AND e.conversation_id=?)" + (" AND m.status=?" if status else ""),
            (owner, *status_params),
        ).fetchone()
        rows = connection.execute(
            "SELECT memory_id,type,normalized_fact,modality,status,valid_from_utc,valid_until_utc,supersedes_id,certainty,importance,temporal_scope,assessment_reason_code,assessed_at_utc,privacy_class,recall_policy "
            "FROM memory_records m WHERE EXISTS (SELECT 1 FROM memory_evidence me "
            "JOIN conversation_events e ON e.event_id=me.event_id "
            "WHERE me.memory_id=m.memory_id AND e.conversation_id=?) "
            f"{status_clause} ORDER BY created_at_utc DESC,memory_id DESC LIMIT ? OFFSET ?", (owner, *status_params, limit + 1, offset)
        ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        memory_ids = [row["memory_id"] for row in rows]
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            evidence_rows = connection.execute(
                "SELECT me.memory_id,me.event_id,me.actor,me.exact_quote,me.occurred_at_utc,me.evidence_role,e.sequence "
                "FROM memory_evidence me JOIN conversation_events e ON e.event_id=me.event_id "
                f"WHERE e.conversation_id=? AND me.memory_id IN ({placeholders}) "
                "ORDER BY me.occurred_at_utc,e.sequence,me.event_id", (owner, *memory_ids)
            ).fetchall()
        else:
            evidence_rows = []
        evidence_by_memory: dict[object, list[dict[str, object]]] = {}
        for item in evidence_rows:
            evidence_by_memory.setdefault(item["memory_id"], []).append({key: item[key] for key in ("event_id", "actor", "exact_quote", "occurred_at_utc", "evidence_role", "sequence")})
        audit_by_memory: dict[str, list[dict[str, object]]] = {}
        presentation_by_memory: dict[str, list[dict[str, object]]] = {}
        if memory_ids:
            placeholders = ",".join("?" for _ in memory_ids)
            for item in connection.execute(
                "SELECT a.memory_id,a.audit_event_id,a.action,a.before_status,a.after_status,"
                "a.assessment_reason_code,a.occurred_at_utc "
                "FROM memory_audit_events a "
                "LEFT JOIN memory_session_jobs j ON j.job_id=a.session_job_id "
                f"WHERE a.memory_id IN ({placeholders}) AND ("
                "j.conversation_id=? OR (j.job_id IS NULL AND EXISTS ("
                "SELECT 1 FROM memory_evidence me JOIN conversation_events e ON e.event_id=me.event_id "
                "WHERE me.memory_id=a.memory_id AND e.conversation_id=?))"
                ") AND EXISTS ("
                "SELECT 1 FROM memory_evidence me JOIN conversation_events e ON e.event_id=me.event_id "
                "WHERE me.memory_id=a.memory_id AND e.conversation_id=?"
                ") ORDER BY a.occurred_at_utc,a.audit_event_id",
                (*memory_ids, owner, owner, owner),
            ).fetchall():
                audit_by_memory.setdefault(item["memory_id"], []).append({key: item[key] for key in ("audit_event_id", "action", "before_status", "after_status", "assessment_reason_code", "occurred_at_utc")})
            for item in connection.execute(
                "SELECT p.memory_id,p.presentation_id,p.fragment_key,p.trigger_event_id,p.context_version,p.presented_at_utc "
                "FROM memory_confirmation_presentations p "
                "JOIN conversation_events e ON e.event_id=p.trigger_event_id "
                f"WHERE p.memory_id IN ({placeholders}) AND p.conversation_id=? AND e.conversation_id=? "
                "ORDER BY p.presented_at_utc,p.presentation_id",
                (*memory_ids, owner, owner),
            ).fetchall():
                presentation_by_memory.setdefault(item["memory_id"], []).append({key: item[key] for key in ("presentation_id", "fragment_key", "trigger_event_id", "context_version", "presented_at_utc")})
        result = []
        for row in rows:
            recall = "confirmation" if row["status"] == "candidate" and row["certainty"] == "ambiguous" and row["importance"] in (2, 3) else ("always" if row["status"] == "active" and row["certainty"] in ("explicit", "confirmed") and row["importance"] == 3 and row["temporal_scope"] == "ongoing" and row["recall_policy"] == "daily_safe" else ("topic" if row["status"] == "active" and row["certainty"] in ("explicit", "confirmed") and row["importance"] >= 1 and row["temporal_scope"] != "unclassified" else "none"))
            audits = audit_by_memory.get(row["memory_id"], [])
            presentations = presentation_by_memory.get(row["memory_id"], [])
            result.append({key: row[key] for key in ("memory_id", "type", "normalized_fact", "modality", "status", "valid_from_utc", "valid_until_utc", "supersedes_id", "certainty", "importance", "temporal_scope", "assessment_reason_code", "assessed_at_utc", "privacy_class", "recall_policy")} | {"recall_scope": recall, "audit_events": audits, "confirmation_presentations": presentations} | {
                "evidence": evidence_by_memory.get(row["memory_id"], [])
            })
        total = int(total_row[0]) if total_row else 0
        return result, {"page": page, "limit": limit, "total": total, "has_more": offset + len(result) < total, "status_filter": status}

    def _operations(self, connection: sqlite3.Connection, marker: dict[str, object] | None) -> dict[str, object]:
        """运维事实：构建指纹、数据库体积、备份清单、worker 水位。

        只读列出本机文件与 runtime_meta，不触碰运行中的进程，也不读任何密钥。
        """

        try:
            code_build: str | None = compute_build_id(self.code_root)
        except (OSError, ReadinessError):
            code_build = None
        marker_build = (marker or {}).get("build_id")
        database = Path(self.database_path)
        sizes: dict[str, int | None] = {}
        for label, suffix in (("main", ""), ("wal", "-wal"), ("shm", "-shm")):
            candidate = database.with_name(database.name + suffix) if suffix else database
            try:
                sizes[label] = candidate.stat().st_size if candidate.is_file() else None
            except OSError:
                sizes[label] = None
        schema_version = None
        try:
            row = connection.execute("SELECT value_json FROM runtime_meta WHERE key='schema_version'").fetchone()
            if row is not None:
                schema_version = json.loads(row["value_json"])
        except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            schema_version = None
        watermarks: dict[str, object] = {}
        try:
            for row in connection.execute(
                "SELECT key,value_json FROM runtime_meta WHERE key LIKE 'memory_worker:%' ORDER BY key"
            ):
                key = str(row["key"]).split(":", 2)[-1]
                try:
                    watermarks[key] = json.loads(row["value_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    watermarks[key] = None
        except sqlite3.Error:
            watermarks = {}
        backups: list[dict[str, object]] = []
        backup_root = Path(self.code_root) / "_backups"
        try:
            entries = sorted(
                (item for item in backup_root.iterdir() if item.is_dir()),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )[:5]
        except OSError:
            entries = []
        for entry in entries:
            newest, size = None, 0
            try:
                for child in entry.rglob("*"):
                    if child.is_file():
                        size += child.stat().st_size
                        moment = child.stat().st_mtime
                        if newest is None or moment > newest:
                            newest = moment
            except OSError:
                continue
            backups.append({
                "name": entry.name,
                "modified_at_utc": datetime.fromtimestamp(newest, timezone.utc).isoformat() if newest else None,
                "bytes": size,
            })
        return {
            "build": {
                "code_build_id": code_build,
                "marker_build_id": marker_build if isinstance(marker_build, str) else None,
                "matches": code_build is not None and marker_build == code_build,
            },
            "database": {
                "name": database.name,
                "schema_version": schema_version,
                "bytes": sizes.get("main"),
                "wal_bytes": sizes.get("wal"),
                "shm_bytes": sizes.get("shm"),
            },
            "backups": backups,
            "watermarks": watermarks,
        }

    @staticmethod
    def _fragments(connection: sqlite3.Connection, owner: object) -> list[dict[str, object]]:
        """按时间倒序列出片段：只有事实与计数，原文按需另取。"""

        if owner is None:
            return []
        rows = connection.execute(
            "SELECT f.fragment_id,f.start_sequence,f.end_sequence,f.started_at_utc,f.ended_at_utc,f.status,"
            "COUNT(d.detail_id) AS detail_count "
            "FROM memory_fragments f LEFT JOIN memory_detail_records d ON d.fragment_id=f.fragment_id "
            "WHERE f.conversation_id=? GROUP BY f.fragment_id "
            "ORDER BY f.started_at_utc DESC,f.fragment_id DESC", (owner,)
        ).fetchall()
        privacy: dict[str, dict[str, int]] = {}
        kinds: dict[str, dict[str, int]] = {}
        for row in connection.execute(
            "SELECT d.fragment_id,d.privacy_class,COUNT(*) AS count FROM memory_detail_records d "
            "JOIN memory_fragments f ON f.fragment_id=d.fragment_id WHERE f.conversation_id=? "
            "GROUP BY d.fragment_id,d.privacy_class", (owner,)
        ):
            privacy.setdefault(row["fragment_id"], {})[row["privacy_class"]] = row["count"]
        for row in connection.execute(
            "SELECT d.fragment_id,d.detail_kind,COUNT(*) AS count FROM memory_detail_records d "
            "JOIN memory_fragments f ON f.fragment_id=d.fragment_id WHERE f.conversation_id=? "
            "GROUP BY d.fragment_id,d.detail_kind", (owner,)
        ):
            kinds.setdefault(row["fragment_id"], {})[row["detail_kind"]] = row["count"]
        return [
            {
                "fragment_id": row["fragment_id"],
                "start_sequence": row["start_sequence"],
                "end_sequence": row["end_sequence"],
                "started_at_utc": row["started_at_utc"],
                "ended_at_utc": row["ended_at_utc"],
                "status": row["status"],
                "detail_count": row["detail_count"],
                "privacy_counts": privacy.get(row["fragment_id"], {}),
                "kind_counts": kinds.get(row["fragment_id"], {}),
            }
            for row in rows
        ]

    def fragment_detail(self, fragment_id: str) -> dict[str, object]:
        """一个片段的逐条原文，按 ordinal 顺序返回。"""

        if not isinstance(fragment_id, str) or not fragment_id or len(fragment_id) > 128:
            raise ValueError("invalid fragment id")
        owner = self._owner_qq()
        if owner is None:
            return {"ok": False, "reason": "owner unknown", "fragment": None, "details": []}
        connection = self._connect_read_only()
        try:
            connection.execute("BEGIN")
            fragment = connection.execute(
                "SELECT fragment_id,start_sequence,end_sequence,started_at_utc,ended_at_utc,status,"
                "fragment_type,reality_scope,privacy_class,recall_policy,summary "
                "FROM memory_fragments WHERE fragment_id=? AND conversation_id=?", (fragment_id, owner)
            ).fetchone()
            if fragment is None:
                return {"ok": False, "reason": "fragment not found", "fragment": None, "details": []}
            rows = connection.execute(
                "SELECT ordinal,detail_id,detail_kind,actor,reality_scope,normalized_detail,exact_quote,"
                "source_event_id,occurred_at_utc,certainty,temporal_scope,status,privacy_class,recall_policy "
                "FROM memory_detail_records WHERE fragment_id=? ORDER BY ordinal,detail_id", (fragment_id,)
            ).fetchall()
            return {
                "ok": True,
                "reason": "ok",
                "fragment": dict(fragment),
                "details": [{key: row[key] for key in row.keys()} for row in rows],
            }
        except sqlite3.Error as error:
            return {"ok": False, "reason": f"database unavailable: {type(error).__name__}", "fragment": None, "details": []}
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()

    def recall_explanation(self, text: str) -> dict[str, object]:
        """对一句话跑一遍真实的定位规则，说明她会取到哪一段、为什么。

        只读复算：用的是上线代码里的同一个纯函数（日期解析、词法窗口、
        命中下限），因此面板给出的答案与她会拿到的答案同源。
        """

        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("invalid query text")
        from qichi.memory.dates import referenced_dates
        from qichi.memory.lexical import MIN_MATCH_FRAGMENTS, match_count, query_fragments

        query = text.strip()
        now = self.clock()
        local_zone = datetime.now().astimezone().tzinfo
        windows = query_fragments((query,))
        days = referenced_dates((query,), now=now, local_zone=local_zone)

        def explicit_recall_request(query_text: str) -> bool:
            """2026-09-11 去掉词表后的同一判据：这句本身点到过去某一天。"""

            return any(day < now.astimezone(local_zone).date() for day in days)
        owner = self._owner_qq()
        if owner is None:
            return {"ok": False, "reason": "owner unknown", "explicit_request": False,
                    "dates": [], "by_date": [], "by_content": [], "newest": None, "windows": list(windows)}
        connection = self._connect_read_only()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT f.fragment_id,f.start_sequence,f.started_at_utc,f.ended_at_utc,"
                "d.normalized_detail||d.exact_quote||f.summary AS searchable "
                "FROM memory_fragments f LEFT JOIN memory_detail_records d ON d.fragment_id=f.fragment_id "
                "WHERE f.conversation_id=? AND (d.status IS NULL OR d.status IN ('candidate','active')) "
                "ORDER BY f.started_at_utc DESC,f.fragment_id DESC,d.ordinal,d.detail_id", (owner,)
            ).fetchall()
        except sqlite3.Error as error:
            return {"ok": False, "reason": f"database unavailable: {type(error).__name__}",
                    "explicit_request": False, "dates": [], "by_date": [], "by_content": [],
                    "newest": None, "windows": list(windows)}
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
        order: list[str] = []
        scores: dict[str, int] = {}
        meta: dict[str, dict[str, object]] = {}
        for row in rows:
            fragment_id = row["fragment_id"]
            if fragment_id not in scores:
                scores[fragment_id] = 0
                order.append(fragment_id)
                meta[fragment_id] = {
                    "fragment_id": fragment_id,
                    "start_sequence": row["start_sequence"],
                    "started_at_utc": row["started_at_utc"],
                    "ended_at_utc": row["ended_at_utc"],
                }
            if row["searchable"]:
                score = match_count(row["searchable"], windows)
                if score > scores[fragment_id]:
                    scores[fragment_id] = score
        needed = min(MIN_MATCH_FRAGMENTS, len(windows)) if windows else 0
        by_content = [dict(meta[fid], score=scores[fid]) for fid in order if needed and scores[fid] >= needed]
        by_date = []
        for fid in order:
            try:
                moment = datetime.fromisoformat(str(meta[fid]["started_at_utc"])).astimezone(local_zone)
            except (TypeError, ValueError):
                continue
            if moment.date() in set(days):
                by_date.append(dict(meta[fid]))
        return {
            "ok": True,
            "reason": "ok",
            "explicit_request": explicit_recall_request(query),
            "dates": [day.isoformat() for day in days],
            "by_date": by_date,
            "by_content": by_content,
            "newest": dict(meta[order[0]]) if order else None,
            "windows": list(windows[:12]),
        }

    def _owner_qq(self) -> object:
        marker, _ = self._read_marker()
        return (marker or {}).get("owner_qq")

    @staticmethod
    def _response_audit(connection: sqlite3.Connection, owner: object) -> list[dict[str, object]]:
        delivery_links: dict[str, str] = {}
        for row in connection.execute(
            "SELECT trigger_event_id,details_json FROM turn_trace_events "
            "WHERE conversation_id=? AND phase='delivery'", (owner,)
        ).fetchall():
            try:
                details = json.loads(row["details_json"] or "{}")
                outbound = details.get("outbound_event_id") if isinstance(details, dict) else None
                if isinstance(outbound, str) and isinstance(row["trigger_event_id"], str):
                    delivery_links[outbound] = row["trigger_event_id"]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        events = connection.execute(
            "SELECT event_id,conversation_id,sequence,occurred_at_utc,status,text,reply_to_event_id,metadata_json "
            "FROM conversation_events WHERE direction='outbound' AND conversation_id=? ORDER BY occurred_at_utc DESC,sequence DESC LIMIT 80", (owner,)
        ).fetchall()
        prepared: list[tuple[
            sqlite3.Row,
            dict[str, object],
            dict[str, object],
            dict[str, object] | None,
            dict[str, object] | None,
        ]] = []
        referenced_memory_ids: set[str] = set()
        for event in events:
            metadata = {}
            try:
                decoded = json.loads(event["metadata_json"])
                if isinstance(decoded, dict):
                    metadata = decoded
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            group = metadata.get("delivery_group")
            safe_group = {key: group[key] for key in ("group_event_id", "part_index", "part_count") if isinstance(group, dict) and key in group}
            expression = metadata.get("expression")
            requested = expression.get("requested") if isinstance(expression, dict) else None
            safe_expression = None
            if isinstance(requested, dict):
                safe_expression = {
                    key: requested[key]
                    for key in ("kind", "key", "target_event_handle")
                    if key in requested and isinstance(requested[key], (str, int, type(None)))
                }
            generation = metadata.get("generation_metadata")
            safe_generation: dict[str, object] = {}
            if isinstance(generation, dict):
                # 2026-09-22 用户要求「把该显示出来的挂上」：working_set_memory_ids 原先不在
                # 这份白名单里，于是**每轮都注入的工作集一条都显示不出来**——他看到的是常驻的
                # 几条，而实际每轮还有几十条工作集在场。
                for key in ("source", "context_version", "relationship_memory_ids", "working_set_memory_ids", "retrieved_memory_ids", "candidate_memory_ids", "quoted_event_id", "history_event_count"):
                    value = generation.get(key)
                    if key.endswith("_ids"):
                        if (
                            isinstance(value, (list, tuple))
                            and len(value) <= 256
                            and all(
                                _trace_identifier(item, 160) is not _TRACE_DROP
                                for item in value
                            )
                        ):
                            safe_generation[key] = list(value)
                            referenced_memory_ids.update(value)
                    elif key == "source" and value in {"dialogue", "interaction", "initiative"}:
                        safe_generation[key] = value
                    elif key == "quoted_event_id" and (value is None or (isinstance(value, str) and value)):
                        safe_generation[key] = value
                    elif key in {"context_version", "history_event_count"} and type(value) is int and value >= 0:
                        safe_generation[key] = value
            prepared.append((event, metadata, safe_generation, safe_group or None, safe_expression))

        scoped_memories: dict[object, dict[str, object]] = {}
        memory_ids = sorted(referenced_memory_ids)
        for start in range(0, len(memory_ids), 500):
            chunk = memory_ids[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            memory_rows = connection.execute(
                "SELECT m.memory_id,m.type,m.normalized_fact,m.status FROM memory_records m "
                f"WHERE m.memory_id IN ({placeholders}) AND EXISTS ("
                "SELECT 1 FROM memory_evidence me JOIN conversation_events e "
                "ON e.event_id=me.event_id WHERE me.memory_id=m.memory_id AND e.conversation_id=?)",
                (*chunk, owner),
            ).fetchall()
            for row in memory_rows:
                scoped_memories[row["memory_id"]] = {
                    "memory_id": row["memory_id"],
                    "type": row["type"],
                    "normalized_fact": row["normalized_fact"],
                    "status": row["status"],
                }

        event_ids = [event["event_id"] for event in events]
        if event_ids:
            placeholders = ",".join("?" for _ in event_ids)
            actions = connection.execute(
                "SELECT o.event_id,o.payload_json,o.status FROM outbox o "
                "JOIN conversation_events e ON e.event_id=o.event_id "
                f"WHERE e.conversation_id=? AND o.event_id IN ({placeholders}) "
                "ORDER BY o.updated_at_utc DESC", (owner, *event_ids)
            ).fetchall()
        else:
            actions = []
        action_by_event: dict[object, sqlite3.Row] = {}
        for row in actions:
            action_by_event.setdefault(row["event_id"], row)
        result = []
        for event, metadata, safe_generation, safe_group, safe_expression in prepared:
            # Legacy outbound rows have no explicit trace association. Never infer a trigger from timeline adjacency.
            trigger = None
            linked_trigger_id = delivery_links.get(event["event_id"])
            if linked_trigger_id is not None:
                trigger = connection.execute(
                    "SELECT event_id,sequence,kind,direction,text,metadata_json FROM conversation_events "
                    "WHERE event_id=? AND conversation_id=? AND direction IN ('inbound','internal') LIMIT 1",
                    (linked_trigger_id, owner),
                ).fetchone()
            action = action_by_event.get(event["event_id"])
            action_kind = None
            action_status = None
            if action is not None:
                action_status = action["status"]
                try:
                    payload = json.loads(action["payload_json"])
                    action_kind = payload.get("action_kind") if isinstance(payload, dict) else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            # 2026-09-22 用户要求「把该显示出来的挂上」：工作集（每轮固定注入的那几十条）
            # 之前**没有投影**——他看到的比实际少得多（只看到常驻的 5 条，实际每轮还另有 66 条）。
            for key in ("relationship_memory_ids", "working_set_memory_ids", "retrieved_memory_ids"):
                ids = safe_generation.get(key, [])
                ref_key = key.replace("_memory_ids", "_refs")
                safe_generation[ref_key] = [scoped_memories[item] for item in ids if item in scoped_memories]
            safe_generation.setdefault("candidate_refs", [])
            if "retrieved_memory_ids" not in safe_generation and "candidate_memory_ids" in safe_generation:
                safe_generation["retrieved_memory_ids"] = list(safe_generation["candidate_memory_ids"])
            if "candidate_memory_ids" in safe_generation and "candidate_refs" not in safe_generation:
                safe_generation["candidate_refs"] = [scoped_memories[item] for item in safe_generation["candidate_memory_ids"] if item in scoped_memories]
            result.append({
                "event_id": event["event_id"], "sequence": event["sequence"], "occurred_at_utc": event["occurred_at_utc"],
                "status": event["status"], "text": event["text"], "reply_to_event_id": event["reply_to_event_id"],
                "trigger_event_id": trigger["event_id"] if trigger else None,
                "trigger_sequence": trigger["sequence"] if trigger else None,
                "trigger_kind": trigger["kind"] if trigger else None,
                "trigger_direction": trigger["direction"] if trigger else None,
                "trigger_text": trigger["text"] if trigger else None,
                "context_version": DashboardService._context_version_from_metadata(trigger["metadata_json"] if trigger else None),
                "delivery_group": safe_group, "expression_intent": safe_expression,
                "action_kind": action_kind, "action_status": action_status,
                "evidence_level": "full" if trigger else "partial",
                "reason_summary": DashboardService._reason_summary(trigger),
                **safe_generation,
            })
        return result

    @staticmethod
    def _context_version_from_metadata(raw: object) -> int | None:
        try:
            value = json.loads(raw) if isinstance(raw, str) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        context_version = value.get("context_version") if isinstance(value, dict) else None
        return context_version if type(context_version) is int and context_version >= 0 else None

    @staticmethod
    def _reason_summary(trigger: sqlite3.Row | None) -> str:
        if trigger is None:
            return "未找到可关联的可靠触发事件，仅保留发送证据。"
        if trigger["direction"] == "internal":
            return "主动消息：依据最近活动和未消费主题生成。"
        return f"基于入站事件 M{trigger['sequence']} 的可靠上下文生成。"


    @staticmethod
    def _features(connection: sqlite3.Connection) -> tuple[dict[str, object] | None, str]:
        """The switches the running process recorded at startup.

        A config file is a request; only what the process loaded is a fact, so this
        projection reads what the stack wrote and reports the absence honestly.
        """
        row = connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?", ("runtime:features",)
        ).fetchone()
        if row is None:
            return None, "无记录（该进程启动时未写入）"
        try:
            decoded = json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "记录无法解析"
        if not isinstance(decoded, dict):
            return None, "记录格式无效"
        return decoded, f"启动时记录于 {row[1]}"


    @staticmethod
    def _initiative(connection: sqlite3.Connection, owner: object) -> dict[str, object]:
        empty = {"raw_state": None, "enabled": None, "activity_anchor_utc": None, "context_version": None, "next_attempt_at_utc": None, "last_activity_at_utc": None, "quiet_hours": None, "max_unanswered": None, "unanswered": None, "paused": None, "deferred_until_utc": None, "timeline": [], "evidence": "none"}
        if owner is None:
            return empty
        row = connection.execute("SELECT value_json FROM runtime_meta WHERE key=?", (f"initiative:{owner}",)).fetchone()
        control_row = connection.execute("SELECT value_json FROM runtime_meta WHERE key=?", (f"initiative-control:{owner}",)).fetchone()
        if row is None and control_row is None:
            return empty
        control = {}
        if control_row is not None:
            try:
                decoded_control = json.loads(control_row[0])
                if isinstance(decoded_control, dict) and set(decoded_control) == {"paused", "deferred_until"}:
                    control = decoded_control
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if row is None:
            return empty | {"paused": control.get("paused"), "deferred_until_utc": control.get("deferred_until"), "evidence": "control-only"}
        try:
            state = decode_initiative_state(row[0])
        except ValueError:
            return empty | {"paused": control.get("paused"), "deferred_until_utc": control.get("deferred_until"), "evidence": "invalid"}
        status = state["status"]
        timeline = []
        if state.get("slot") and state.get("claim_id"):
            timeline.append({"category": "due", "at_utc": state["slot"], "claim_id": state["claim_id"]})
        if state.get("claimed_at") and state.get("claim_id"):
            timeline.append({"category": "claim", "at_utc": state["claimed_at"], "claim_id": state["claim_id"]})
        if status and status != "claimed" and state.get("completed_at"):
            timeline.append({"category": status, "at_utc": state["completed_at"], "claim_id": state.get("claim_id"), "failure_category": state.get("failure_category")})
        return {"raw_state": status, "enabled": None, "activity_anchor_utc": state.get("activity"), "context_version": state.get("context_version"), "next_attempt_at_utc": state.get("next_due_at"), "last_activity_at_utc": state.get("activity"), "quiet_hours": None, "max_unanswered": None, "unanswered": state.get("unanswered_attempts"), "paused": control.get("paused"), "deferred_until_utc": control.get("deferred_until"), "timeline": timeline, "evidence": "state"}
