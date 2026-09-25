import json
import sqlite3
from datetime import datetime, timezone, timedelta


FRAGMENT_COLUMNS = ("fragment_id", "conversation_id", "fragment_key", "start_sequence", "end_sequence",
                    "start_event_id", "end_event_id", "started_at_utc", "ended_at_utc", "fragment_type",
                    "reality_scope", "summary", "privacy_class", "recall_policy", "status",
                    "closed_at_utc", "created_at_utc")


def seed_fragments(path):
    """两个片段：较新的日常、较早且含成人原文。"""

    db = Database(path)
    c = db.connection
    for index, (fragment_id, started, privacy, policy) in enumerate((
        ("frag-old", "2026-08-26T04:00:00+00:00", "adult", "explicit_request_only"),
        ("frag-new", "2026-08-27T04:00:00+00:00", "ordinary", "daily_safe"),
    )):
        c.execute(
            "INSERT INTO memory_fragments (" + ",".join(FRAGMENT_COLUMNS) + ") VALUES ("
            + ",".join("?" for _ in FRAGMENT_COLUMNS) + ")",
            (fragment_id, "100", "key-" + fragment_id, 10 + index, 12 + index, "e1", "e2", started,
             started, "adult" if privacy == "adult" else "daily", "conversation", "冻结片段的完整原文索引；具体内容以事件账本为准",
             privacy, policy, "active", started, started),
        )
        quotes = ("那天中午我们说过的话", "SECRET-ADULT-LINE") if privacy == "adult" else ("今天天气不错", "晚上想吃火锅")
        for ordinal, quote in enumerate(quotes):
            c.execute(
                "INSERT INTO memory_detail_records (detail_id,fragment_id,ordinal,detail_kind,actor,reality_scope,"
                "normalized_detail,exact_quote,source_event_id,occurred_at_utc,certainty,temporal_scope,status,"
                "privacy_class,recall_policy) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fragment_id + "-d" + str(ordinal), fragment_id, ordinal, "message", "mumo", "conversation",
                 "记录一条原文。", quote, "e1", started, "explicit", "historical", "active", privacy, policy),
            )
    db.close()


def test_health_reports_a_stalled_memory_consolidation(tmp_path):
    """命中（2026-09-22）：有终态失败的整合任务时，健康条必须报出来。

    真机：一条引文里的坏标点把记忆水位钉了 8 小时，而**没有任何地方**会提醒，
    是用户主动问起才被发现。
    """

    db_path, marker_path = tmp_path / "stall.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)
    db = Database(db_path)
    db.connection.execute(
        "INSERT INTO memory_session_jobs (job_id, conversation_id, revision, fragment_key,"
        " start_sequence, end_sequence, anchor_event_id, anchor_sequence, anchor_received_at_utc,"
        " deadline_utc, context_version, status, attempt_count, failure_category,"
        " created_at_utc, updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("job-stall", "100", 1, "key", 10, 12, "e1", 12, "2026-08-27T04:00:00+00:00",
         "2026-08-27T04:30:00+00:00", 1, "failed", 4, "schema_error",
         "2026-08-27T04:00:00+00:00", "2026-08-27T04:30:00+00:00"),
    )
    db.close()

    health = DashboardService(db_path, marker_path).snapshot()["health"]

    assert health["ok"] is False
    assert "记忆整合" in health["reason"]


def test_health_ignores_a_backlog_that_is_simply_waiting(tmp_path):
    """不误判：没有终态失败就不报警——长对话本来就要静默 30 分钟才整合一次，
    落后几百条属于正常。"""

    db_path, marker_path = tmp_path / "quiet.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)

    health = DashboardService(db_path, marker_path).snapshot()["health"]

    assert "记忆整合" not in health["reason"]


def test_health_keeps_both_the_ready_reason_and_the_memory_stall(tmp_path):
    """不误判（互补）：marker 本身不健康时，卡住理由不许把原来那条顶掉——两条都要留住。"""

    db_path, marker_path = tmp_path / "both.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)
    db = Database(db_path)
    db.connection.execute(
        "INSERT INTO memory_session_jobs (job_id, conversation_id, revision, fragment_key,"
        " start_sequence, end_sequence, anchor_event_id, anchor_sequence, anchor_received_at_utc,"
        " deadline_utc, context_version, status, attempt_count, failure_category,"
        " created_at_utc, updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("job-both", "100", 1, "key", 10, 12, "e1", 12, "2026-08-27T04:00:00+00:00",
         "2026-08-27T04:30:00+00:00", 1, "failed", 4, "schema_error",
         "2026-08-27T04:00:00+00:00", "2026-08-27T04:30:00+00:00"),
    )
    db.close()
    # 让 marker 指向不存在的进程/不一致摘要，制造"另一条更该看见的异常"
    marker_path.write_text(json.dumps({
        "schema": "qichi-ready", "version": 3, "owner_qq": "100", "bot_qq": "200",
        "pid": 1, "parent_pid": 2, "instance_id": "x", "started_at_utc": "2020-01-01T00:00:00+00:00",
        "build_id": "bad", "model": "m", "provider": "p", "database_schema_version": 6,
    }), encoding="utf-8")

    health = DashboardService(db_path, marker_path).snapshot()["health"]

    assert health["ok"] is False
    assert "记忆整合" in health["reason"], "卡住理由必须在"
    assert len(health["reason"].split("；")) >= 2, "原来的理由也不许被顶掉"


def _job_event(db, *, details: dict, at: str | None = None, event_id: str = "je-1") -> None:
    db.connection.execute(
        "INSERT INTO memory_job_events (job_event_id, job_id, conversation_id, revision, fragment_key,"
        " start_sequence, end_sequence, action, failure_category, attempt_count, next_retry_at_utc,"
        " occurred_at_utc, details_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (event_id, "job-1", "100", 1, "key", 10, 12, "completed", None, 1, None,
         at or datetime.now(timezone.utc).isoformat(), json.dumps(details)),
    )


def test_memory_usage_totals_are_visible_and_bounded(tmp_path):
    """命中（2026-09-22）：面板汇总记忆后台的 token 用量，好回答「钱花在哪」。"""

    db_path, marker_path = tmp_path / "usage.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)
    db = Database(db_path)
    _job_event(db, details={
        "extraction_input_tokens": "1000", "extraction_output_tokens": "200",
        "detail_input_tokens": "2000", "detail_output_tokens": "500", "detail_calls": "3",
        "SECRET-LEAK": "x",
    })
    db.close()

    snapshot = DashboardService(db_path, marker_path).snapshot()

    assert snapshot["memory_usage"]["totals"]["extraction_input_tokens"] == 1000
    assert snapshot["memory_usage"]["totals"]["detail_input_tokens"] == 2000
    assert snapshot["memory_usage"]["jobs"] == 1
    assert "SECRET-LEAK" not in json.dumps(snapshot), "白名单外的键不许被投影"
    assert snapshot["memory_worker"]["last_result"]["details"]["detail_output_tokens"] == "500"


def test_memory_usage_absent_is_an_empty_total_not_a_guess(tmp_path):
    """不误判：没有用量记录时给空汇总，不编数字、不报错。"""

    db_path, marker_path = tmp_path / "nou.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()

    assert snapshot["memory_usage"]["totals"] == {}
    assert snapshot["memory_usage"]["jobs"] == 0


def test_prices_follow_the_official_peak_and_idle_windows():
    """命中（2026-09-22）：高峰=北京时间工作日 9:00-12:00、14:00-18:00，其余按空闲（半价）。

    价格抄自 DeepSeek 官方定价页（见 dashboard/service.py 的 _LLM_PRICES）。
    """

    from qichi.dashboard.service import _is_peak_hour, _price_cny

    monday_10 = datetime(2026, 9, 21, 2, 0, tzinfo=timezone.utc)    # 北京时间周一 10:00
    monday_20 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)   # 北京时间周一 20:00
    saturday_10 = datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)  # 北京时间周六 10:00

    assert _is_peak_hour(monday_10) is True
    assert _is_peak_hour(monday_20) is False
    assert _is_peak_hour(saturday_10) is False, "周末不算高峰"
    assert _price_cny("deepseek-flash", "miss", monday_10, 1_000_000) == 2.0
    assert _price_cny("deepseek-flash", "miss", monday_20, 1_000_000) == 1.0
    assert _price_cny("deepseek-v4-flash", "hit", monday_20, 1_000_000) == 0.02, "旧名按 Flash 计价"
    assert _price_cny("someone-else", "miss", monday_10, 1_000_000) == 0.0, "未知模型绝不猜"
    assert _price_cny("deepseek-flash", "miss", monday_10, 0) == 0.0


def test_memory_cost_is_summed_with_peak_and_idle_rates(tmp_path):
    """命中：记忆后台的用量按各自事件时刻的高峰/空闲价折算（同一把尺子）。"""

    db_path, marker_path = tmp_path / "cost.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)
    db = Database(db_path)
    # 必须落在"近 24 小时"窗口内，所以时间相对现在取；高峰/空闲由同一单价函数判定。
    from qichi.dashboard.service import _price_cny

    recent = datetime.now(timezone.utc) - timedelta(hours=2)
    older = datetime.now(timezone.utc) - timedelta(hours=4)
    _job_event(db, event_id="je-a", at=recent.isoformat(), details={
        "extraction_input_tokens": "1000000", "extraction_cache_hit_tokens": "0",
    })
    _job_event(db, event_id="je-b", at=older.isoformat(), details={
        "detail_input_tokens": "1000000", "detail_cache_hit_tokens": "0",
    })
    db.close()

    usage = DashboardService(
        db_path, marker_path, {"provider": "deepseek", "model": "deepseek-flash"}
    ).snapshot()["memory_usage"]

    assert usage["totals"]["extraction_input_tokens"] == 1_000_000
    assert usage["totals"]["detail_input_tokens"] == 1_000_000
    expected = _price_cny("deepseek-flash", "miss", recent, 1_000_000) + _price_cny(
        "deepseek-flash", "miss", older, 1_000_000
    )
    assert usage["cost_cny"] == round(expected, 3), "汇总必须等于各条事件按当时单价折算之和"


def test_cost_today_is_zero_and_never_guessed_when_nothing_ran(tmp_path):
    """不误判：没有用量时给 0，不编数字。"""

    db_path, marker_path = tmp_path / "nocost.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()

    assert snapshot["cost_today"]["currency"] == "CNY"
    assert snapshot["cost_today"]["total_cny"] == 0.0
    assert snapshot["chat_usage"]["turns"] == 0
    assert snapshot["memory_usage"]["jobs"] == 0


def test_fragment_projection_is_counts_only_and_newest_first(tmp_path):
    db_path, marker_path = tmp_path / "f.sqlite3", tmp_path / "ready"
    seed(db_path); seed_fragments(db_path); marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()

    fragments = snapshot["fragments"]
    assert [item["fragment_id"] for item in fragments] == ["frag-new", "frag-old"]
    assert fragments[0]["detail_count"] == 2
    assert fragments[1]["privacy_counts"] == {"adult": 2}
    assert fragments[0]["kind_counts"] == {"message": 2}
    assert "SECRET-ADULT-LINE" not in json.dumps(snapshot), "列表页不得带上原文"


def test_fragment_detail_returns_verbatim_lines_in_ordinal_order(tmp_path):
    db_path, marker_path = tmp_path / "d.sqlite3", tmp_path / "ready"
    seed(db_path); seed_fragments(db_path); marker(marker_path)

    service = DashboardService(db_path, marker_path)
    payload = service.fragment_detail("frag-old")

    assert payload["ok"] is True
    assert payload["fragment"]["fragment_id"] == "frag-old"
    assert [item["ordinal"] for item in payload["details"]] == [0, 1]
    assert payload["details"][1]["exact_quote"] == "SECRET-ADULT-LINE"
    assert payload["details"][1]["privacy_class"] == "adult"


def test_fragment_detail_refuses_unknown_and_malformed_ids(tmp_path):
    db_path, marker_path = tmp_path / "x.sqlite3", tmp_path / "ready"
    seed(db_path); seed_fragments(db_path); marker(marker_path)

    service = DashboardService(db_path, marker_path)

    assert service.fragment_detail("nope")["ok"] is False
    with pytest.raises(ValueError):
        service.fragment_detail("")


def test_recall_explanation_replays_the_same_rules_she_runs(tmp_path):
    db_path, marker_path = tmp_path / "r.sqlite3", tmp_path / "ready"
    seed(db_path); seed_fragments(db_path); marker(marker_path)

    service = DashboardService(db_path, marker_path)
    by_date = service.recall_explanation("细说26号那天")
    by_content = service.recall_explanation("那天中午我们说过的话")

    assert by_date["explicit_request"] is True, "词表命中必须与线上同一判据"
    assert by_date["dates"] == ["2026-08-26"]
    assert [item["fragment_id"] for item in by_date["by_date"]] == ["frag-old"]
    assert [item["fragment_id"] for item in by_content["by_content"]] == ["frag-old"]
    other = service.recall_explanation("晚上想吃火锅")
    assert [item["fragment_id"] for item in other["by_content"]] == ["frag-new"], "不得误命中另一个片段"
    assert by_content["newest"]["fragment_id"] == "frag-new"
    with pytest.raises(ValueError):
        service.recall_explanation("   ")


def test_trace_category_whitelist_covers_every_context_category():
    """投影会整张丢弃含未知键的 token 表，白名单必须与构建器同源。"""
    from qichi.dialogue.context_builder import _CATEGORY_KEYS
    from qichi.dashboard.service import _TRACE_CONTEXT_CATEGORIES

    missing = sorted(set(_CATEGORY_KEYS) - set(_TRACE_CONTEXT_CATEGORIES))
    assert missing == [], f"这些上下文字段会被静默丢弃：{missing}"

from pathlib import Path
import pytest

import qichi.dashboard.service as dashboard_service_module
from qichi.dashboard import DashboardService
from qichi.readiness import READY_MARKER_VERSION, compute_build_id
from qichi.storage.database import Database
from qichi.storage.migrations import SCHEMA_VERSION

ROOT = Path(__file__).parents[1]
BUILD_ID = compute_build_id(ROOT)


def seed(path):
    db = Database(path)
    c = db.connection
    c.execute("INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("e1", "100", 0, "inbound", "mumo", "text", "hello", "[]", None, None, "2026-08-29T00:00:00+00:00", "2026-08-29T00:00:01+00:00", "received", "{}"))
    c.execute("INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("e2", "100", 1, "outbound", "qichi", "text", "reply", "[]", "e1", "pm1", "2026-08-29T00:01:00+00:00", "2026-08-29T00:01:01+00:00", "sent", '{"delivery_group":{"group_event_id":"e2","part_index":0,"part_count":1},"expression":{"requested":{"kind":"face","key":"smile","secret":"TOP_SECRET"}}}'))
    c.execute("INSERT INTO platform_message_map VALUES (?,?,?,?)", ("pm2", "e2", "test", "2026-08-29T00:01:02+00:00"))
    c.execute("INSERT INTO conversation_cursors VALUES (?,?,?,?,?)", ("100", 2, 1, "2026-08-29T00:01:00+00:00", 4))
    c.execute("INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)", ("op1", "e2", '{"action_kind":"text","token":"TOP_SECRET"}', "sent", 1, None, "2026-08-29T00:01:00+00:00", "2026-08-29T00:01:02+00:00"))
    c.execute("INSERT INTO memory_records (memory_id,type,normalized_fact,modality,status,valid_from_utc,valid_until_utc,supersedes_id,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?)", ("m1", "preference", "likes tea", "text", "active", "2026-08-29T00:00:00+00:00", None, None, "2026-08-29T00:01:00+00:00"))
    c.execute("INSERT INTO memory_evidence (memory_id,event_id,actor,exact_quote,occurred_at_utc) VALUES (?,?,?,?,?)", ("m1", "e1", "mumo", "hello", "2026-08-29T00:00:00+00:00"))
    c.execute("INSERT INTO runtime_meta VALUES (?,?,?)", ("initiative:100", '{"status":"sent","activity":"2026-08-29T00:00:00+00:00","next_due_at":"2026-08-29T00:03:00+00:00","secret":"TOP_SECRET"}', "2026-08-29T00:02:00+00:00"))
    db.close()


def marker(path, *, build_id=BUILD_ID, version=READY_MARKER_VERSION, schema=SCHEMA_VERSION, **overrides):
    """Write a READY marker.

    The default is a *current* marker: it carries the build evidence that the
    code under test computes for this checkout.  Legacy versions are produced
    explicitly and never carry build evidence.
    """
    value = {
        "schema": "qichi-ready", "version": version,
        "instance_id": "fixture-instance", "lock_identity": "fixture-lock",
        "pid": 123, "parent_pid": 456,
        "owner_qq": "100", "bot_qq": "200",
        "database_schema_version": schema, "last_recovered_sequence": -1,
        "ws_connection_id": "fixture-ws", "memory_worker_id": "fixture-memory",
        "initiative_worker_id": "fixture-initiative",
        "started_at_utc": "2026-08-29T00:00:00+00:00",
        "provider": "deepseek", "model": "deepseek-v4-flash", "context_window": 262144,
        "build_id": build_id,
    }
    if version < READY_MARKER_VERSION:
        value.pop("build_id")
        if version == 1:
            for field in ("provider", "model", "context_window"):
                value.pop(field)
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def assert_owner_projection_unavailable(snapshot):
    assert snapshot["runtime"] == {
        "provider": None, "model": None, "context_window": None,
        "owner_qq": None, "bot_qq": None,
    }
    assert snapshot["counts"] == {"events": 0, "outbox": 0, "memory": 0}
    assert snapshot["outbox_status"] is None
    for key in ("recent_events", "outbox", "memory_status", "memories", "response_audit", "traces"):
        assert snapshot[key] == []


def test_operations_projection_reports_build_database_watermarks_and_backups(tmp_path):
    db_path, marker_path = tmp_path / "ops.sqlite3", tmp_path / "ready"
    (tmp_path / "src" / "qichi").mkdir(parents=True)
    (tmp_path / "src" / "qichi" / "app.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "0001_initial.sql").write_text("CREATE TABLE t(x);\n", encoding="utf-8")
    seed(db_path); marker(marker_path, build_id=compute_build_id(tmp_path))
    with sqlite3.connect(db_path) as raw:
        raw.execute("INSERT INTO runtime_meta VALUES (?,?,?)",
                    ("memory_worker:100:processed_sequence", "42", "2026-08-29T00:00:00+00:00"))
    backup = tmp_path / "_backups" / "deploy-1"
    backup.mkdir(parents=True)
    (backup / "qichi-before.sqlite3").write_bytes(b"x" * 32)

    service = DashboardService(db_path, marker_path, code_root=tmp_path)
    operations = service.snapshot()["operations"]

    assert operations["build"]["matches"] is True
    assert operations["build"]["marker_build_id"] == operations["build"]["code_build_id"]
    assert operations["database"]["schema_version"] == SCHEMA_VERSION
    assert operations["database"]["bytes"] and operations["database"]["bytes"] > 0
    assert operations["watermarks"] == {"processed_sequence": 42}
    assert [item["name"] for item in operations["backups"]] == ["deploy-1"]
    assert operations["backups"][0]["bytes"] >= 32
    payload = json.dumps(operations).lower()
    for leaked in ("traceback", "operationalerror", "databaseerror", "no such table"):
        assert leaked not in payload, "运维投影不得泄漏异常细节"


def test_operations_projection_never_leaks_environment_secrets(tmp_path):
    db_path, marker_path = tmp_path / "ops2.sqlite3", tmp_path / "ready"
    seed(db_path); marker(marker_path)
    (tmp_path / "src" / "qichi").mkdir(parents=True)
    (tmp_path / "src" / "qichi" / "app.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "migrations").mkdir()
    (tmp_path / "migrations" / "0001_initial.sql").write_text("CREATE TABLE t(x);\n", encoding="utf-8")

    payload = json.dumps(DashboardService(db_path, marker_path, code_root=tmp_path).snapshot(), ensure_ascii=False).lower()

    for forbidden in ("api_key", "access_token", "authorization", "bearer "):
        assert forbidden not in payload


def test_snapshot_matches_frontend_contract_and_filters_secrets(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    marker(marker_path)
    snapshot = DashboardService(db_path, marker_path, {"provider": "fake", "model": "model", "context_window": 262144}).snapshot()
    assert set(snapshot) >= {"health", "ready", "runtime", "cursor", "counts", "recent_events", "outbox", "outbox_status", "memory_status", "memories", "response_audit", "initiative", "traces", "trace_page", "snapshot_version"}
    assert snapshot["health"] == {"ok": True, "reason": "ok"}
    assert snapshot["ready"]["ok"] is True and snapshot["ready"]["marker"]["owner_qq"] == "100"
    assert "api_key" not in snapshot["ready"]["marker"]
    assert snapshot["runtime"] == {"provider": "deepseek", "model": "deepseek-v4-flash", "context_window": 262144, "owner_qq": "100", "bot_qq": "200"}
    assert snapshot["cursor"] == {"context_version": 2, "last_processed_sequence": 1, "last_user_activity_utc": "2026-08-29T00:01:00+00:00", "presence_topic_cursor": 4}
    assert snapshot["counts"] == {"events": 2, "outbox": 1, "memory": 1}
    assert set(snapshot["recent_events"][0]) == {"sequence", "actor", "direction", "kind", "status", "occurred_at_utc", "text", "event_id", "platform_message_id", "reply_to_event_id"}
    assert [item["event_id"] for item in snapshot["recent_events"]] == ["e2", "e1"]
    assert snapshot["outbox"] == [{"operation_key": "op1", "event_id": "e2", "status": "sent", "attempt_count": 1, "action_kind": "text", "updated_at_utc": "2026-08-29T00:01:02+00:00"}]
    assert snapshot["outbox_status"] == [{"status": "sent", "count": 1}]
    assert snapshot["memory_status"] == [{"status": "active", "count": 1}]
    assert snapshot["memories"][0]["evidence"] == [{"event_id": "e1", "actor": "mumo", "exact_quote": "hello", "occurred_at_utc": "2026-08-29T00:00:00+00:00", "evidence_role": "source", "sequence": 0}]
    audit = snapshot["response_audit"][0]
    assert audit["trigger_event_id"] is None and audit["evidence_level"] == "partial"
    assert audit["context_version"] is None and "可靠" in audit["reason_summary"]
    assert audit["delivery_group"] == {"group_event_id": "e2", "part_index": 0, "part_count": 1}
    assert audit["expression_intent"] == {"kind": "face", "key": "smile"}
    assert audit["action_kind"] == "text" and audit["action_status"] == "sent"
    assert snapshot["initiative"]["evidence"] == "invalid"
    assert snapshot["initiative"]["quiet_hours"] is None and snapshot["initiative"]["max_unanswered"] is None
    assert "TOP_SECRET" not in json.dumps(snapshot)


def test_memory_worker_exposes_last_public_outcome_after_job_row_is_deleted(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = sqlite3.connect(db_path)
    db.execute(
        "INSERT INTO memory_job_events (job_event_id,job_id,conversation_id,revision,fragment_key,"
        "start_sequence,end_sequence,action,failure_category,attempt_count,next_retry_at_utc,"
        "occurred_at_utc,details_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "job-event-1", "job-1", "100", 3, "fragment-1", 0, 1, "completed", None, 1, None,
            "2026-08-29T00:02:00+00:00",
            json.dumps({
                "outcome_kind": "no_persistent_memory",
                "outcome_reason_code": "temporary_scene_or_roleplay",
                "candidate_count": "0", "review_count": "0", "prompt": "must not leak",
            }),
        ),
    )
    db.commit(); db.close(); marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()
    result = snapshot["memory_worker"]["last_result"]
    assert result["action"] == "completed"
    assert result["details"] == {
        "outcome_kind": "no_persistent_memory",
        "outcome_reason_code": "temporary_scene_or_roleplay",
        "candidate_count": "0",
        "review_count": "0",
    }
    assert "prompt" not in json.dumps(snapshot)


def test_trace_projection_is_explicit_batched_and_fail_closed(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path); db = Database(db_path)
    for phase, details in (("received", {"trigger_event_id":"e1", "category_tokens":{"history":3}}), ("generation", {"selected_memory_ids":["m1"], "prompt":"private", "memory_scores":{"m1":0.8}}), ("delivery", {"model_route":"v4f", "generation_ms":12, "outbox_statuses":{"text":"sent"}})):
        db.connection.execute("INSERT INTO turn_trace_events VALUES (?,?,?,?,?,?,?,?)", (f"te-{phase}", "trace-1", "100", "e1", "dialogue", phase, "2026-08-29T00:01:00+00:00", json.dumps(details)))
    db.close(); marker(marker_path)
    snapshot = DashboardService(db_path, marker_path).snapshot(limit=1)
    assert snapshot["trace_page"] == {"page": 1, "limit": 1, "has_more": False}
    assert snapshot["traces"][0]["trace_id"] == "trace-1"
    assert len(snapshot["traces"][0]["phases"]) == 3
    phases = {item["phase"]: item["details"] for item in snapshot["traces"][0]["phases"]}
    assert phases["received"]["category_tokens"] == {"history": 3}
    assert phases["generation"] == {}
    assert snapshot["snapshot_version"]


def test_trace_projection_keeps_safe_context_and_generation_evidence(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path); db = Database(db_path)
    details = {
        "input_tokens": 120,
        "input_budget_tokens": 250000,
        "selected_history_event_ids": ["e1"],
        "selected_memory_ids": ["m1"],
        "omitted_counts": {"history": 2},
        "memory_scores": {"m1": 7},
        "memory_reasons": {
            "m1": "normalized_fact,exact_quote,trigram_overlap:fact=1;quote=3"
        },
        "quote_resolution_status": "resolved",
        "quoted_event_id": "e1",
    }
    db.connection.execute(
        "INSERT INTO turn_trace_events VALUES (?,?,?,?,?,?,?,?)",
        ("te-context", "trace-safe", "100", "e1", "dialogue", "context", "2026-08-29T00:01:00+00:00", json.dumps(details)),
    )
    db.close(); marker(marker_path)

    projected = DashboardService(db_path, marker_path).snapshot()["traces"][0]["phases"][0]["details"]

    assert projected == details


def test_trace_projection_keeps_only_explicit_bounded_field_shapes():
    legal = {
        "context_version": 7,
        "event_kind": "text",
        "has_quote_target": True,
        "context_ms": 12.5,
        "category_tokens": {"history": 3, "current_input": 2},
        "omitted_counts": {"history": 1, "memory_evidence": 0},
        "selected_history_event_ids": ["e1"],
        "selected_memory_ids": ["m1"],
        "memory_scores": {"m1": 7.0},
        "memory_reasons": {"m1": "trigram_overlap:fact=1;quote=3"},
        "quote_resolution_status": "resolved",
        "quoted_event_id": "e1",
        "result_type": "reply",
        "model_id": "deepseek-v4-flash",
        "model_route": "primary",
        "generation_ms": 23.75,
        "attempt_count": 1,
        "retry_count": 0,
        "finish_reason": "stop",
        "status": "sent",
        "outbound_event_id": "outbound-1",
        "platform_message_id_present": True,
        "expression_kind": "reaction",
        "expression_key": "heart",
        "outbox_statuses": {"operation-1": "sent"},
        "reason": "llm:timeout",
        "memory_detail_key": "verbatim",
        "memory_detail_match_count": 2,
        "memory_detail_indexed": True,
        "memory_detail_reason": "day_missing",
        "tool_name": "web_search",
        "tool_query": "示例学院 简介",
        "tool_ok": True,
        "tool_degraded": None,
        "tool_elapsed_ms": 3705,
        "tool_result_chars": 1200,
        "images_carried": 1,
    }

    assert DashboardService._safe_trace_details(legal) == legal

    unsafe = {
        "context_version": 8,
        "reason": "TOP_SECRET",
        "memory_reasons": {
            "m1": {"token": "TOP_SECRET"},
            "m2": "trigram_overlap:" + "x" * 500,
        },
        "category_tokens": {"history": {"secret": "TOP_SECRET"}},
        "outbox_statuses": {"operation-1": {"status": "sent", "token": "TOP_SECRET"}},
        "selected_memory_ids": ["m1", "private message with spaces"],
        "model_id": "m" * 500,
        "generation_ms": float("inf"),
        "finish_reason": "provider-private-body",
    }

    projected = DashboardService._safe_trace_details(unsafe)

    assert projected == {"context_version": 8}
    assert "TOP_SECRET" not in json.dumps(projected)


def test_net_tool_fields_are_projected_with_chinese_queries_but_still_fail_closed():
    """检索词是中文，标识符那一档认不了——所以专门加了一档自由文本，但仍然有上限。"""

    kept = DashboardService._safe_trace_details({
        "tool_name": "image_search",
        "tool_query": "蓝紫粉渐变纯色壁纸",
        "tool_ok": False,
        "tool_degraded": "no_results",
        "tool_elapsed_ms": 4000,
        "tool_result_chars": 0,
    })

    assert kept["tool_query"] == "蓝紫粉渐变纯色壁纸"
    assert kept["tool_degraded"] == "no_results"
    assert kept["tool_ok"] is False

    dropped = DashboardService._safe_trace_details({
        "tool_name": "rm_rf",
        "tool_query": "x" * 201,
        "tool_degraded": "with space",
        "tool_ok": "yes",
        "tool_elapsed_ms": -1,
    })

    assert dropped == {}


def test_a_control_character_in_a_query_is_dropped():
    assert DashboardService._safe_trace_details({'tool_query': 'bad\nvalue'}) == {}


def test_trace_page_boundaries_and_owner_scope(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path); db = Database(db_path)
    for index in range(3):
        db.connection.execute("INSERT INTO turn_trace_events VALUES (?,?,?,?,?,?,?,?)", (f"te-{index}", f"trace-{index}", "100", "e1", "dialogue", "received", f"2026-08-29T00:0{index}:00+00:00", "{}"))
    db.close(); marker(marker_path)
    first = DashboardService(db_path, marker_path).snapshot(page=1, limit=2)
    second = DashboardService(db_path, marker_path).snapshot(page=2, limit=2)
    empty = DashboardService(db_path, marker_path).snapshot(page=3, limit=2)
    assert len(first["traces"]) == 2 and first["trace_page"]["has_more"] is True
    assert len(second["traces"]) == 1 and second["trace_page"]["has_more"] is False
    assert empty["traces"] == []


def test_initiative_bad_json_and_other_owner_are_not_projected(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path); db = Database(db_path)
    db.connection.execute("INSERT INTO runtime_meta VALUES (?,?,?)", ("initiative:999", '{"status":"sent","owner":"999"}', "2026-08-29T00:02:00+00:00"))
    db.connection.execute("UPDATE runtime_meta SET value_json=? WHERE key=?", ("{", "initiative:100"))
    db.close(); marker(marker_path)
    initiative = DashboardService(db_path, marker_path).snapshot()["initiative"]
    assert initiative["evidence"] == "invalid"
    assert initiative["raw_state"] is None


def test_initiative_projects_production_state_with_claim_owner_identity(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    state = {
        "slot": "2026-08-29T01:00:00+00:00",
        "status": "sent",
        "owner": "0123456789abcdef0123456789abcdef",
        "claimed_at": "2026-08-29T01:00:01+00:00",
        "activity": "2026-08-29T00:00:00+00:00",
        "context_version": 2,
        "claim_id": "claim-1",
        "outbound_event_id": "outbound-1",
        "unanswered_attempts": 1,
        "completed_at": "2026-08-29T01:00:05+00:00",
        "next_due_at": "2026-08-29T02:00:05+00:00",
    }
    db = Database(db_path)
    db.connection.execute(
        "UPDATE runtime_meta SET value_json=? WHERE key=?",
        (json.dumps(state), "initiative:100"),
    )
    db.close()
    marker(marker_path)

    initiative = DashboardService(db_path, marker_path).snapshot()["initiative"]

    assert initiative["evidence"] == "state"
    assert initiative["raw_state"] == "sent"
    assert initiative["enabled"] is None
    assert initiative["activity_anchor_utc"] == state["activity"]
    assert initiative["next_attempt_at_utc"] == state["next_due_at"]
    assert initiative["unanswered"] == 1
    assert [item["category"] for item in initiative["timeline"]] == ["due", "claim", "sent"]


def test_initiative_state_is_scoped_by_runtime_meta_key_not_claim_owner(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    state = {
        "slot": "2026-08-29T01:00:00+00:00",
        "status": "sent",
        "owner": "claim-owner-for-another-conversation",
        "claimed_at": "2026-08-29T01:00:01+00:00",
        "activity": "2026-08-29T00:00:00+00:00",
        "context_version": 2,
        "claim_id": "claim-other",
        "outbound_event_id": "outbound-other",
        "unanswered_attempts": 1,
        "completed_at": "2026-08-29T01:00:05+00:00",
        "next_due_at": "2026-08-29T02:00:05+00:00",
    }
    db = Database(db_path)
    db.connection.execute("DELETE FROM runtime_meta WHERE key=?", ("initiative:100",))
    db.connection.execute(
        "INSERT INTO runtime_meta VALUES (?,?,?)",
        ("initiative:999", json.dumps(state), "2026-08-29T01:00:05+00:00"),
    )
    db.close()
    marker(marker_path)

    initiative = DashboardService(db_path, marker_path).snapshot()["initiative"]

    assert initiative["evidence"] == "none"
    assert initiative["raw_state"] is None


def test_initiative_malformed_state_with_claim_owner_stays_invalid(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    malformed = {
        "status": "sent",
        "owner": "0123456789abcdef0123456789abcdef",
        "claimed_at": "2026-08-29T01:00:01+00:00",
        "activity": "2026-08-29T00:00:00+00:00",
        "context_version": 2,
        "claim_id": "claim-1",
        "outbound_event_id": "outbound-1",
        "unanswered_attempts": 1,
        "completed_at": "2026-08-29T01:00:05+00:00",
        "next_due_at": "2026-08-29T02:00:05+00:00",
    }
    db = Database(db_path)
    db.connection.execute(
        "UPDATE runtime_meta SET value_json=? WHERE key=?",
        (json.dumps(malformed), "initiative:100"),
    )
    db.close()
    marker(marker_path)

    initiative = DashboardService(db_path, marker_path).snapshot()["initiative"]

    assert initiative["evidence"] == "invalid"
    assert initiative["raw_state"] is None
    assert initiative["timeline"] == []


def test_memory_evidence_is_scoped_to_owner_conversation(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    c = db.connection
    c.execute("INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("other", "999", 2, "inbound", "mumo", "text", "other quote", "[]", None, None, "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "received", "{}"))
    c.execute("INSERT INTO memory_evidence (memory_id,event_id,actor,exact_quote,occurred_at_utc) VALUES (?,?,?,?,?)", ("m1", "other", "mumo", "other quote", "2026-08-29T00:02:00+00:00"))
    db.close()
    marker(marker_path)
    memories = DashboardService(db_path, marker_path).snapshot()["memories"]
    assert memories[0]["evidence"] == [{"event_id": "e1", "actor": "mumo", "exact_quote": "hello", "occurred_at_utc": "2026-08-29T00:00:00+00:00", "evidence_role": "source", "sequence": 0}]


def test_memory_detail_lifecycle_rows_are_scoped_by_their_owner_relations(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    c = db.connection
    event_sql = (
        "INSERT INTO conversation_events "
        "(event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json," 
        "reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    c.execute(event_sql, ("other-trigger", "999", 0, "inbound", "mumo", "text", "other", "[]", None, None,
                          "2026-08-29T00:03:00+00:00", "2026-08-29T00:03:01+00:00", "received", "{}"))
    c.execute("INSERT INTO memory_evidence (memory_id,event_id,actor,exact_quote,occurred_at_utc) VALUES (?,?,?,?,?)",
              ("m1", "other-trigger", "mumo", "other", "2026-08-29T00:03:00+00:00"))
    job_sql = (
        "INSERT INTO memory_session_jobs "
        "(job_id,conversation_id,revision,fragment_key,start_sequence,end_sequence,anchor_event_id,anchor_sequence,"
        "anchor_received_at_utc,deadline_utc,context_version,status,claim_token,claim_owner,claim_lease_until_utc,"
        "next_retry_at_utc,attempt_count,failure_category,created_at_utc,updated_at_utc) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    for job_id, conversation_id, anchor_event_id in (("owner-job", "100", "e1"), ("other-job", "999", "other-trigger")):
        c.execute(job_sql, (job_id, conversation_id, 1, job_id, 0, 0, anchor_event_id, 0,
                             "2026-08-29T00:00:01+00:00", "2026-08-29T00:04:00+00:00", 2,
                             "failed", None, None, None, None, 1, "test", "2026-08-29T00:03:00+00:00", "2026-08-29T00:03:00+00:00"))
    audit_sql = (
        "INSERT INTO memory_audit_events "
        "(audit_event_id,memory_id,session_job_id,action,before_status,after_status,assessment_reason_code,occurred_at_utc) "
        "VALUES (?,?,?,?,?,?,?,?)"
    )
    c.execute(audit_sql, ("owner-audit", "m1", "owner-job", "assess", "candidate", "active", "explicit_user_statement", "2026-08-29T00:03:00+00:00"))
    c.execute(audit_sql, ("other-audit", "m1", "other-job", "assess", "candidate", "active", "explicit_user_statement", "2026-08-29T00:04:00+00:00"))
    presentation_sql = (
        "INSERT INTO memory_confirmation_presentations "
        "(presentation_id,conversation_id,memory_id,fragment_key,trigger_event_id,context_version,presented_at_utc) "
        "VALUES (?,?,?,?,?,?,?)"
    )
    c.execute(presentation_sql, ("owner-presentation", "100", "m1", "owner-fragment", "e1", 2, "2026-08-29T00:03:00+00:00"))
    c.execute(presentation_sql, ("other-presentation", "999", "m1", "other-fragment", "other-trigger", 2, "2026-08-29T00:04:00+00:00"))
    db.close()
    marker(marker_path)

    memory = DashboardService(db_path, marker_path).snapshot()["memories"][0]

    assert [item["audit_event_id"] for item in memory["audit_events"]] == ["owner-audit"]
    assert [item["presentation_id"] for item in memory["confirmation_presentations"]] == ["owner-presentation"]


def test_response_audit_also_projects_the_working_set(tmp_path):
    """2026-09-22 用户要求「把该显示出来的挂上」：工作集也必须投影出来。

    面板原先只显示常驻的 relationship_refs 与按话题命中的 retrieved_refs，而**每轮固定注入的
    工作集没有显示**——他看到的比实际少得多（只看到常驻 5 条，实际每轮还另有 66 条）。
    """

    db_path, marker_path = tmp_path / "ws.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    db.connection.execute(
        "INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("e3", "100", 2, "outbound", "qichi", "text", "reply with memory", "[]", "e1", "pm1",
         "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "sent",
         '{"generation_metadata":{"source":"dialogue","context_version":2,'
         '"relationship_memory_ids":["m1"],"working_set_memory_ids":["m1"],'
         '"candidate_memory_ids":[],"quoted_event_id":"e1","history_event_count":1}}'),
    )
    db.close()
    marker(marker_path)

    audit = DashboardService(db_path, marker_path).snapshot()["response_audit"][0]

    working = audit["working_set_refs"]
    assert [item["memory_id"] for item in working] == ["m1"], (
        "工作集必须投影出来，否则面板显示不出每轮真正在注入什么"
    )
    assert set(working[0]) >= {"memory_id", "type", "normalized_fact"}, "至少要有面板渲染需要的字段"
    assert audit["relationship_refs"], "常驻层照旧"


def test_response_audit_projects_scoped_memory_facts(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    db.connection.execute(
        "INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("e3", "100", 2, "outbound", "qichi", "text", "reply with memory", "[]", "e1", "pm1", "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "sent", '{"generation_metadata":{"source":"dialogue","context_version":2,"relationship_memory_ids":["m1"],"candidate_memory_ids":[],"quoted_event_id":"e1","history_event_count":1}}'),
    )
    db.close()
    marker(marker_path)
    audit = DashboardService(db_path, marker_path).snapshot()["response_audit"][0]
    assert audit["relationship_refs"] == [{"memory_id": "m1", "type": "preference", "normalized_fact": "likes tea", "status": "active"}]
    assert audit["candidate_refs"] == []


def test_memory_projection_supports_status_filter_and_pagination_meta(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    rows, meta = DashboardService._memories(db.connection, "100", page=1, limit=1, status="active")
    assert len(rows) == 1
    assert rows[0]["status"] == "active"
    assert {"certainty", "importance", "temporal_scope", "evidence"} <= set(rows[0])
    assert meta == {"page": 1, "limit": 1, "total": 1, "has_more": False, "status_filter": "active"}
    with pytest.raises(ValueError, match="invalid memory status"):
        DashboardService._memories(db.connection, "100", status="unknown")
    db.close()




def test_detail_queries_are_bounded_to_the_selected_memory_and_event_ids(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    seed(db_path)
    db = Database(db_path)
    db.connection.execute(
        "INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "e3", "100", 2, "outbound", "qichi", "text", "reply with memory", "[]",
            "e1", "pm1", "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00",
            "sent", '{"generation_metadata":{"source":"dialogue","context_version":2,"relationship_memory_ids":["m1"],"candidate_memory_ids":[],"quoted_event_id":"e1","history_event_count":1}}',
        ),
    )
    statements = []
    db.connection.set_trace_callback(statements.append)

    DashboardService._memories(db.connection, "100")
    DashboardService._response_audit(db.connection, "100")

    normalized = [" ".join(statement.lower().split()) for statement in statements]
    evidence_query = next(
        statement for statement in normalized
        if statement.startswith("select me.memory_id,me.event_id")
    )
    scoped_memory_query = next(
        statement for statement in normalized
        if statement.startswith("select m.memory_id,m.type,m.normalized_fact,m.status")
    )
    action_query = next(
        statement for statement in normalized
        if statement.startswith("select o.event_id,o.payload_json,o.status")
    )
    assert "me.memory_id in (" in evidence_query
    assert "m.memory_id in (" in scoped_memory_query
    assert "o.event_id in (" in action_query
    db.close()


def test_response_audit_rejects_unbounded_or_invalid_memory_id_lists(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    seed(db_path)
    oversized = [f"memory-{index}" for index in range(257)]
    metadata = {
        "generation_metadata": {
            "source": "dialogue",
            "context_version": 2,
            "relationship_memory_ids": oversized,
            "candidate_memory_ids": ["private message with spaces"],
            "quoted_event_id": "e1",
            "history_event_count": 1,
        }
    }
    db = Database(db_path)
    db.connection.execute(
        "INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "e3", "100", 2, "outbound", "qichi", "text", "reply", "[]", "e1", "pm1",
            "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "sent",
            json.dumps(metadata),
        ),
    )

    audit = DashboardService._response_audit(db.connection, "100")[0]

    assert audit["context_version"] == 2
    assert "relationship_memory_ids" not in audit
    assert "candidate_memory_ids" not in audit
    assert audit["relationship_refs"] == []
    assert audit["candidate_refs"] == []
    db.close()


def test_snapshot_is_read_only_and_handles_bad_inputs(tmp_path):
    path = tmp_path / "empty.sqlite3"
    db = Database(path)
    before = [tuple(row) for row in db.connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name")]
    db.close()
    missing = DashboardService(path, tmp_path / "missing.json").snapshot()
    assert missing["health"] == {"ok": False, "reason": "marker missing"}
    assert missing["counts"] == {"events": 0, "outbox": 0, "memory": 0}
    assert missing["outbox_status"] is None
    assert missing["memories"] == [] and missing["response_audit"] == []
    check = sqlite3.connect(path)
    assert before == [tuple(row) for row in check.execute("SELECT name,sql FROM sqlite_master ORDER BY name")]
    check.close()
    bad_marker = tmp_path / "bad.json"
    bad_marker.write_text("{", encoding="utf-8")
    bad = DashboardService(tmp_path / "missing.sqlite3", bad_marker).snapshot()
    assert bad["ready"] == {"ok": False, "reason": "marker invalid", "marker": None}
    assert bad["health"]["ok"] is False
    assert bad["outbox_status"] is None


def test_valid_empty_owner_has_available_zero_outbox_aggregate(tmp_path):
    db_path, marker_path = tmp_path / "empty.sqlite3", tmp_path / "ready.json"
    Database(db_path).close()
    marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()

    assert snapshot["outbox_status"] == []
    assert snapshot["counts"]["outbox"] == 0


def test_ready_requires_matching_live_lock_and_fresh_marker(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    Database(db_path).close()
    now = datetime.now(timezone.utc)
    marker(marker_path)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")
    service = DashboardService(db_path, marker_path, {"provider":"wrong", "model":"wrong"}, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now)
    snapshot = service.snapshot()
    assert snapshot["ready"]["ok"] is True
    assert snapshot["runtime"]["provider"] == "deepseek"


def test_ready_v2_projects_verified_model_profile_and_uses_core_pid_probe(tmp_path, monkeypatch):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    Database(db_path).close()
    now = datetime.now(timezone.utc)
    marker(marker_path)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")
    probed=[]
    monkeypatch.setattr(dashboard_service_module, "pid_alive", lambda pid: probed.append(pid) or pid == 123)

    snapshot = DashboardService(
        db_path,
        marker_path,
        {"provider":"wrong","model":"wrong","context_window":1},
        lock_path=lock_path,
        clock=lambda: now,
    ).snapshot()

    assert snapshot["health"] == {"ok": True, "reason": "ok"}
    assert snapshot["ready"]["ok"] is True
    assert snapshot["runtime"] == {
        "provider":"deepseek", "model":"deepseek-v4-flash", "context_window":262144,
        "owner_qq":"100", "bot_qq":"200",
    }
    assert probed == [123]


def test_ready_does_not_use_port_or_dead_pid_as_evidence(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    seed(db_path)
    now = datetime.now(timezone.utc)
    marker(marker_path)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")
    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda _pid: False, clock=lambda: now).snapshot()
    assert snapshot["ready"]["ok"] is False
    assert snapshot["ready"]["reason"] == "runtime evidence invalid"
    assert_owner_projection_unavailable(snapshot)


def test_live_runtime_does_not_expire_after_twenty_four_hours(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    Database(db_path).close()
    now = datetime.now(timezone.utc)
    started = now - timedelta(days=3)
    marker(marker_path, started_at_utc=started.isoformat())
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":started.isoformat()}), encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now).snapshot()

    assert snapshot["health"] == {"ok": True, "reason": "ok"}


def test_future_runtime_marker_is_rejected(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    seed(db_path)
    now = datetime.now(timezone.utc)
    future = now + timedelta(minutes=1)
    marker(marker_path, started_at_utc=future.isoformat())
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now).snapshot()

    assert snapshot["health"] == {"ok": False, "reason": "runtime evidence stale"}
    assert_owner_projection_unavailable(snapshot)


def test_runtime_lock_must_match_the_core_exact_schema(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    seed(db_path)
    now = datetime.now(timezone.utc)
    marker(marker_path)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat(),"unexpected":True}), encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now).snapshot()

    assert snapshot["health"] == {"ok": False, "reason": "runtime evidence invalid"}
    assert_owner_projection_unavailable(snapshot)


def test_ready_blocks_when_the_marker_was_written_by_other_code(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    seed(db_path)
    now = datetime.now(timezone.utc)
    marker(marker_path, build_id="0" * 64)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now).snapshot()

    assert snapshot["health"] == {"ok": False, "reason": "build evidence mismatch"}
    assert snapshot["ready"]["ok"] is False
    assert_owner_projection_unavailable(snapshot)


def test_ready_blocks_legacy_marker_without_build_evidence(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    seed(db_path)
    now = datetime.now(timezone.utc)
    marker(marker_path, version=2)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now).snapshot()

    assert snapshot["health"] == {"ok": False, "reason": "marker build evidence missing"}
    assert_owner_projection_unavailable(snapshot)


def test_ready_reports_database_schema_conflict_instead_of_plain_invalid(tmp_path):
    db_path, marker_path, lock_path = tmp_path / "db.sqlite3", tmp_path / "ready.json", tmp_path / "qichi.lock"
    seed(db_path)
    now = datetime.now(timezone.utc)
    marker(marker_path, version=2, schema=SCHEMA_VERSION - 1)
    lock_path.write_text(json.dumps({"schema":"qichi-lock","version":1,"instance_id":"fixture-instance","lock_identity":"fixture-lock","pid":123,"parent_pid":456,"acquired_at_utc":now.isoformat()}), encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path, lock_path=lock_path, pid_probe=lambda pid: pid == 123, clock=lambda: now).snapshot()

    assert snapshot["health"] == {"ok": False, "reason": f"database schema {SCHEMA_VERSION - 1} does not match code schema {SCHEMA_VERSION}"}
    assert_owner_projection_unavailable(snapshot)


def test_ready_compares_build_evidence_against_the_configured_code_root(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    marker(marker_path)
    other = tmp_path / "other"
    (other / "src" / "qichi").mkdir(parents=True)
    (other / "migrations").mkdir()
    (other / "src" / "qichi" / "app.py").write_text("x=1\n", encoding="utf-8")
    (other / "migrations" / "0001_initial.sql").write_text("CREATE TABLE t(x);\n", encoding="utf-8")

    same_code = DashboardService(db_path, marker_path, code_root=ROOT).snapshot()
    other_code = DashboardService(db_path, marker_path, code_root=other).snapshot()

    assert same_code["health"] == {"ok": True, "reason": "ok"}
    assert other_code["health"] == {"ok": False, "reason": "build evidence mismatch"}


def test_cross_owner_events_and_outbox_are_not_projected(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    db.connection.execute("INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("other-e", "999", 0, "inbound", "mumo", "text", "private", "[]", None, None, "2026-08-29T00:00:00+00:00", "2026-08-29T00:00:01+00:00", "received", "{}"))
    db.connection.execute("INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)", ("other-op", "other-e", '{"action_kind":"text"}', "sent", 1, None, "2026-08-29T00:00:00+00:00", "2026-08-29T00:00:01+00:00"))
    db.close(); marker(marker_path)
    snapshot = DashboardService(db_path, marker_path).snapshot()
    assert snapshot["counts"] == {"events": 2, "outbox": 1, "memory": 1}
    assert all(item["event_id"] != "other-e" for item in snapshot["recent_events"])


def test_outbox_status_uses_full_owner_history_while_details_stay_recent(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    connection = db.connection
    event_sql = (
        "INSERT INTO conversation_events "
        "(event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,"
        "reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    connection.execute(
        event_sql,
        ("owner-old-failed", "100", 2, "outbound", "qichi", "text", "old", "[]", None, None,
         "2026-08-28T23:59:00+00:00", "2026-08-28T23:59:01+00:00", "failed", "{}"),
    )
    connection.execute(
        "INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)",
        ("owner-old-failed-op", "owner-old-failed", '{"action_kind":"text"}', "failed", 1, None,
         "2026-08-28T23:59:00+00:00", "2026-08-28T23:59:01+00:00"),
    )
    for index in range(50):
        timestamp = f"2026-08-29T01:00:{index:02d}+00:00"
        event_id = f"owner-recent-{index:02d}"
        connection.execute(
            event_sql,
            (event_id, "100", index + 3, "outbound", "qichi", "text", "recent", "[]", None, None,
             timestamp, timestamp, "sent", "{}"),
        )
        connection.execute(
            "INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)",
            (f"owner-recent-op-{index:02d}", event_id, '{"action_kind":"text"}', "sent", 1, None,
             timestamp, timestamp),
        )
    connection.execute(
        event_sql,
        ("other-unknown", "999", 0, "outbound", "qichi", "text", "private", "[]", None, None,
         "2026-08-29T02:00:00+00:00", "2026-08-29T02:00:01+00:00", "sent", "{}"),
    )
    connection.execute(
        "INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)",
        ("other-unknown-op", "other-unknown", '{"action_kind":"text"}', "unknown", 1, None,
         "2026-08-29T02:00:00+00:00", "2026-08-29T02:00:01+00:00"),
    )
    db.close()
    marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()

    assert snapshot["counts"]["outbox"] == 52
    assert snapshot["outbox_status"] == [
        {"status": "failed", "count": 1},
        {"status": "sent", "count": 51},
    ]
    assert sum(item["count"] for item in snapshot["outbox_status"]) == snapshot["counts"]["outbox"]
    assert len(snapshot["outbox"]) == 50
    assert all(item["status"] == "sent" for item in snapshot["outbox"])
    assert all(item["event_id"] != "other-unknown" for item in snapshot["outbox"])


def test_snapshot_uses_one_read_transaction_during_concurrent_owner_write(tmp_path, monkeypatch):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    marker(marker_path)
    original_events = DashboardService._events
    injected = False

    def inject_write(connection, owner):
        nonlocal injected
        if not injected:
            injected = True
            writer = sqlite3.connect(db_path, isolation_level=None)
            writer.execute(
                "INSERT INTO conversation_events "
                "(event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,"
                "reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("concurrent-event", "100", 2, "outbound", "qichi", "text", "later", "[]", None, None,
                 "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "sent", "{}"),
            )
            writer.execute(
                "INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)",
                ("concurrent-op", "concurrent-event", '{"action_kind":"text"}', "sent", 1, None,
                 "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00"),
            )
            writer.close()
        return original_events(connection, owner)

    monkeypatch.setattr(DashboardService, "_events", staticmethod(inject_write))
    during = DashboardService(db_path, marker_path).snapshot()
    after = DashboardService(db_path, marker_path).snapshot()

    assert during["counts"]["outbox"] == 1
    assert during["outbox_status"] == [{"status": "sent", "count": 1}]
    assert all(item["event_id"] != "concurrent-event" for item in during["recent_events"])
    assert all(item["event_id"] != "concurrent-event" for item in during["outbox"])
    assert after["counts"]["outbox"] == 2
    assert after["outbox_status"] == [{"status": "sent", "count": 2}]


def test_memory_status_is_scoped_to_owner_conversation(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    db.connection.execute("INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("other-memory-event", "999", 2, "inbound", "mumo", "text", "other", "[]", None, None, "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "received", "{}"))
    db.connection.execute("INSERT INTO memory_records (memory_id,type,normalized_fact,modality,status,valid_from_utc,valid_until_utc,supersedes_id,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?)", ("m-other", "preference", "other fact", "text", "rejected", "2026-08-29T00:00:00+00:00", None, None, "2026-08-29T00:01:00+00:00"))
    db.connection.execute("INSERT INTO memory_evidence (memory_id,event_id,actor,exact_quote,occurred_at_utc) VALUES (?,?,?,?,?)", ("m-other", "other-memory-event", "mumo", "other", "2026-08-29T00:02:00+00:00"))
    db.close(); marker(marker_path)
    assert DashboardService(db_path, marker_path).snapshot()["memory_status"] == [{"status": "active", "count": 1}]


def test_memory_detail_audit_and_confirmation_projection_is_owner_scoped(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path)
    db = Database(db_path)
    c = db.connection
    c.execute(
        "INSERT INTO conversation_events (event_id,conversation_id,sequence,direction,actor,kind,text,message_segments_json,reply_to_event_id,reply_to_platform_message_id,occurred_at_utc,received_at_utc,status,metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("other-detail-event", "999", 2, "inbound", "mumo", "text", "other", "[]", None, None,
         "2026-08-29T00:02:00+00:00", "2026-08-29T00:02:01+00:00", "received", "{}"),
    )
    # Both rows reference the same memory id, but only one presentation belongs
    # to the runtime owner's conversation. The projection must re-check the
    # conversation through the trigger event instead of trusting memory_id.
    c.execute(
        "INSERT INTO memory_audit_events (audit_event_id,memory_id,action,before_status,after_status,assessment_reason_code,occurred_at_utc) VALUES (?,?,?,?,?,?,?)",
        ("audit-owner", "m1", "assess", "active", "active", "legacy_manual_review", "2026-08-29T00:01:10+00:00"),
    )
    c.execute(
        "INSERT INTO memory_records (memory_id,type,normalized_fact,modality,status,valid_from_utc,valid_until_utc,supersedes_id,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?)",
        ("m-other", "preference", "other fact", "text", "active", "2026-08-29T00:00:00+00:00", None, None, "2026-08-29T00:01:00+00:00"),
    )
    c.execute(
        "INSERT INTO memory_evidence (memory_id,event_id,actor,exact_quote,occurred_at_utc) VALUES (?,?,?,?,?)",
        ("m-other", "other-detail-event", "mumo", "other", "2026-08-29T00:02:00+00:00"),
    )
    c.execute(
        "INSERT INTO memory_audit_events (audit_event_id,memory_id,action,before_status,after_status,assessment_reason_code,occurred_at_utc) VALUES (?,?,?,?,?,?,?)",
        ("audit-other-memory", "m-other", "assess", "active", "active", "legacy_manual_review", "2026-08-29T00:02:10+00:00"),
    )
    c.execute(
        "INSERT INTO memory_confirmation_presentations (presentation_id,conversation_id,memory_id,fragment_key,trigger_event_id,context_version,presented_at_utc) VALUES (?,?,?,?,?,?,?)",
        ("presentation-owner", "100", "m1", "owner-fragment", "e1", 2, "2026-08-29T00:01:20+00:00"),
    )
    c.execute(
        "INSERT INTO memory_confirmation_presentations (presentation_id,conversation_id,memory_id,fragment_key,trigger_event_id,context_version,presented_at_utc) VALUES (?,?,?,?,?,?,?)",
        ("presentation-other", "999", "m1", "other-fragment", "other-detail-event", 7, "2026-08-29T00:02:20+00:00"),
    )
    db.close()
    marker(marker_path)

    snapshot = DashboardService(db_path, marker_path).snapshot()
    owner_memory = next(item for item in snapshot["memories"] if item["memory_id"] == "m1")
    assert [item["audit_event_id"] for item in owner_memory["audit_events"]] == ["audit-owner"]
    assert [item["presentation_id"] for item in owner_memory["confirmation_presentations"]] == ["presentation-owner"]
    assert all(item["memory_id"] != "m-other" for item in snapshot["memories"])


def test_snapshot_version_changes_when_any_public_projection_changes(tmp_path):
    db_path, marker_path = tmp_path / "db.sqlite3", tmp_path / "ready.json"
    seed(db_path); marker(marker_path)
    service = DashboardService(db_path, marker_path)
    before = service.snapshot()["snapshot_version"]
    db = Database(db_path)
    db.connection.execute("UPDATE outbox SET payload_json=? WHERE operation_key=?", ('{"action_kind":"face"}', "op1"))
    db.close()
    after = service.snapshot()["snapshot_version"]
    assert after != before


def test_features_projection_reports_what_the_process_recorded(tmp_path):
    import json as _json

    from qichi.storage.database import Database

    db_path = tmp_path / "features.sqlite3"
    db = Database(db_path)
    try:
        db.connection.execute(
            "INSERT INTO runtime_meta VALUES (?,?,?)",
            ("runtime:features", _json.dumps({"initiative_enabled": True, "qq_face_enabled": False}), "2026-09-11T00:00:00+00:00"),
        )
        db.connection.commit()
    finally:
        db.close()
    marker_path = tmp_path / "ready.json"
    marker_path.write_text("{}", encoding="utf-8")

    snapshot = DashboardService(db_path, marker_path).snapshot()

    assert snapshot["features"] == {"initiative_enabled": True, "qq_face_enabled": False}
    assert "启动时记录于" in snapshot["features_evidence"]


def test_features_absence_and_corruption_are_reported_not_guessed(tmp_path):
    from qichi.storage.database import Database

    marker_path = tmp_path / "ready.json"
    marker_path.write_text("{}", encoding="utf-8")

    absent_path = tmp_path / "absent.sqlite3"
    db = Database(absent_path)
    db.close()
    absent = DashboardService(absent_path, marker_path).snapshot()
    assert absent["features"] is None
    assert "无记录" in absent["features_evidence"]

    broken_path = tmp_path / "broken.sqlite3"
    db = Database(broken_path)
    db.connection.execute("INSERT INTO runtime_meta VALUES (?,?,?)", ("runtime:features", "{", "x"))
    db.connection.commit()
    db.close()
    broken = DashboardService(broken_path, marker_path).snapshot()
    assert broken["features"] is None
    assert "无法解析" in broken["features_evidence"]
    assert "api_key" not in str(broken).lower() and "access_token" not in str(broken).lower()

