# -*- coding: utf-8 -*-
"""memory_gaps.py 的判据测试：什么算「真损失」，什么不算。"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import memory_gaps  # noqa: E402

from qichi.storage.database import Database  # noqa: E402

NOW = datetime(2026, 9, 16, 1, 0, tzinfo=timezone.utc).isoformat()


def add_event(database, sequence, *, direction, actor, kind, text=None, status="sent"):
    database.connection.execute(
        "INSERT INTO conversation_events (event_id, platform_event_id, platform_message_id, conversation_id,"
        " sequence, direction, actor, kind, text, message_segments_json, reply_to_event_id,"
        " reply_to_platform_message_id, occurred_at_utc, received_at_utc, status, metadata_json, raw_payload_json)"
        " VALUES (?, NULL, NULL, 'c1', ?, ?, ?, ?, ?, '[]', NULL, NULL, ?, ?, ?, '{}', NULL)",
        (f"e{sequence}", sequence, direction, actor, kind, text, NOW, NOW, status),
    )


def add_fragment(database, start, end):
    database.connection.execute(
        "INSERT INTO memory_fragments (fragment_id, conversation_id, fragment_key, start_sequence, end_sequence,"
        " start_event_id, end_event_id, started_at_utc, ended_at_utc, fragment_type, reality_scope, summary,"
        " privacy_class, recall_policy, status, closed_at_utc, created_at_utc)"
        " VALUES (?, 'c1', ?, ?, ?, ?, ?, ?, ?, 'daily', 'conversation', 's', 'ordinary', 'daily_safe',"
        " 'candidate', NULL, ?)",
        (f"f{start}", f"k{start}", start, end, f"e{start}", f"e{end}", NOW, NOW, NOW),
    )
    database.connection.commit()


def add_detail(database, fragment_id, event_id, ordinal=0):
    database.connection.execute(
        "INSERT INTO memory_detail_records (detail_id, fragment_id, ordinal, detail_kind, actor,"
        " reality_scope, normalized_detail, exact_quote, source_event_id, occurred_at_utc, certainty,"
        " temporal_scope, status, privacy_class, recall_policy)"
        " VALUES (?, ?, ?, 'message', 'mumo', 'conversation', 'n', 'q', ?, ?, 'explicit',"
        " 'historical', 'active', 'ordinary', 'daily_safe')",
        (f"d-{fragment_id}-{ordinal}", fragment_id, ordinal, event_id, NOW),
    )
    database.connection.execute(
        "INSERT INTO memory_detail_evidence (detail_id, event_id, evidence_role, ordinal)"
        " VALUES (?, ?, 'source', 0)",
        (f"d-{fragment_id}-{ordinal}", event_id),
    )
    database.connection.commit()


def set_watermark(database, value):
    database.connection.execute(
        "INSERT INTO runtime_meta (key, value_json, updated_at_utc) VALUES (?, ?, ?)",
        (f"memory_worker:c1:processed_sequence", str(value), NOW),
    )
    database.connection.commit()


def test_a_timeline_that_stops_early_is_reported_and_only_gates_when_asked(tmp_path, capsys):
    """2026-09-17：明细没盖到片段末尾 = 那段尾巴对她就等于不存在。"""

    database = Database(tmp_path / "qichi.sqlite3")
    try:
        for sequence in range(1, 11):
            add_event(
                database, sequence, direction="outbound", actor="qichi", kind="text", text=f"m{sequence}"
            )
        add_fragment(database, 1, 10)
        add_detail(database, "f1", "e1", ordinal=0)
        add_detail(database, "f1", "e2", ordinal=1)

        assert memory_gaps.main(["--database", str(database.path)]) == 0, "默认只报告，不拉到退出码"
        output = capsys.readouterr().out
        assert "1 个片段的明细没盖到末尾" in output
        assert "明细 2 条，只盖到 2（还差 8 条没人盖）" in output

        assert memory_gaps.main(["--database", str(database.path), "--strict-details"]) == 1
    finally:
        database.close()


def test_a_tail_covered_by_another_fragment_is_not_reported(tmp_path, capsys):
    """不误判：尾巴被另一个片段补上了（补录会造成这种重叠），就不再算截断。"""

    database = Database(tmp_path / "qichi.sqlite3")
    try:
        for sequence in range(1, 11):
            add_event(
                database, sequence, direction="outbound", actor="qichi", kind="text", text=f"m{sequence}"
            )
        add_fragment(database, 1, 10)   # 明细只盖到 2
        add_detail(database, "f1", "e1", ordinal=0)
        add_detail(database, "f1", "e2", ordinal=1)
        add_fragment(database, 3, 10)   # 另一个片段把 3..10 补上
        for sequence in range(3, 11):
            add_detail(database, "f3", f"e{sequence}", ordinal=sequence)

        assert memory_gaps.main(["--database", str(database.path)]) == 0
        output = capsys.readouterr().out
        assert "每个有明细的片段都盖到了自己的末尾" in output
    finally:
        database.close()


def test_a_fragment_without_any_timeline_is_not_called_truncated(tmp_path, capsys):
    """不误判：一条明细都没有是另一条路径的事（空时间线由 job 账本说明）。"""

    database = Database(tmp_path / "qichi.sqlite3")
    try:
        for sequence in range(1, 11):
            add_event(
                database, sequence, direction="outbound", actor="qichi", kind="text", text=f"m{sequence}"
            )
        add_fragment(database, 1, 10)

        assert memory_gaps.main(["--database", str(database.path)]) == 0
        output = capsys.readouterr().out
        assert "每个有明细的片段都盖到了自己的末尾" in output
    finally:
        database.close()


def test_a_gap_above_the_worker_watermark_is_pending_not_lost(tmp_path, capsys):
    """不误判：刚聊完那一段还没轮到，不能报成真损失。"""

    database = Database(tmp_path / "qichi.sqlite3")
    try:
        add_event(database, 1, direction="outbound", actor="qichi", kind="text", text="早呀")
        add_event(database, 2, direction="inbound", actor="mumo", kind="text", text="刚聊完", status="received")
        add_event(database, 3, direction="outbound", actor="qichi", kind="text", text="嗯")
        add_fragment(database, 1, 1)
        set_watermark(database, 1)

        assert memory_gaps.main(["--database", str(database.path)]) == 0
        output = capsys.readouterr().out
        assert "还没轮到" in output
        assert "真损失" not in output
    finally:
        database.close()


def test_a_gap_holding_his_words_is_reported_as_a_loss(tmp_path, capsys):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        add_event(database, 1, direction="inbound", actor="mumo", kind="text", text="兔子，我醒了", status="received")
        add_event(database, 2, direction="outbound", actor="qichi", kind="text", text="早呀")
        add_event(database, 3, direction="outbound", actor="qichi", kind="text", text="睡得踏实吗")
        add_event(database, 4, direction="inbound", actor="mumo", kind="text", text="要去上课", status="received")
        add_event(database, 5, direction="outbound", actor="qichi", kind="text", text="先出门")
        add_fragment(database, 1, 3)

        code = memory_gaps.main(["--database", str(database.path)])

        output = capsys.readouterr().out
        assert code == 1
        assert "真损失：含他 1 条原话" in output
        assert "repair_memory_window.py --start 4 --end 5" in output
    finally:
        database.close()


def test_an_initiative_opening_without_his_words_is_not_a_loss(tmp_path, capsys):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        add_event(database, 1, direction="inbound", actor="mumo", kind="text", text="在吗", status="received")
        add_event(database, 2, direction="outbound", actor="qichi", kind="text", text="在的")
        add_event(database, 3, direction="internal", actor="platform", kind="initiative")
        add_event(database, 4, direction="outbound", actor="qichi", kind="text", text="冒个泡")
        add_fragment(database, 1, 2)

        code = memory_gaps.main(["--database", str(database.path)])

        output = capsys.readouterr().out
        assert code == 0
        assert "只有角色的主动开场" in output
        assert "没有含他原话的缺口" in output
        assert "真损失" not in output
    finally:
        database.close()


def test_events_before_the_memory_layer_are_not_counted_as_gaps(tmp_path, capsys):
    database = Database(tmp_path / "qichi.sqlite3")
    try:
        add_event(database, 1, direction="inbound", actor="mumo", kind="text", text="很久以前", status="received")
        add_event(database, 2, direction="outbound", actor="qichi", kind="text", text="嗯")
        add_event(database, 3, direction="inbound", actor="mumo", kind="text", text="今天", status="received")
        add_event(database, 4, direction="outbound", actor="qichi", kind="text", text="好")
        add_fragment(database, 3, 4)

        code = memory_gaps.main(["--database", str(database.path)])

        output = capsys.readouterr().out
        assert code == 0
        assert "记忆层起点 seq 3" in output
    finally:
        database.close()

def test_dangling_evidence_is_reported(tmp_path, capsys):
    """2026-09-17：证据指向不存在的事件会让整轮装配失败，必须有常设检查。"""

    database = Database(tmp_path / "qichi.sqlite3")
    try:
        add_event(database, 1, direction="outbound", actor="qichi", kind="text", text="在的")
        add_event(database, 2, direction="inbound", actor="mumo", kind="text", text="也在")
        add_fragment(database, 1, 2)
        add_detail(database, "f1", "e1", ordinal=0)
        database.close()
        # 用裸连接（默认不开 foreign_keys）造出悬空引用——副本回放脚本删事件时就是这样留下的：
        # 应用连接没开 FK，删除成功，证据却指向了不存在的事件。
        import sqlite3 as _sqlite3

        raw = _sqlite3.connect(database.path)
        with raw:
            raw.execute(
                "INSERT INTO memory_detail_evidence (detail_id, event_id, evidence_role, ordinal)"
                " VALUES ('d-f1-0', 'ghost-event', 'source', 1)"
            )
        raw.close()
        database = Database(database.path)

        code = memory_gaps.main(["--database", str(database.path)])

        output = capsys.readouterr().out
        assert "证据悬空" in output
        assert "ghost-e" in output, "要指出是哪个 id"
        assert code == 1, "悬空引用必须让巡检以非零退出"
    finally:
        database.close()


def test_a_clean_database_reports_no_dangling_evidence(tmp_path, capsys):
    """不误判：引用都在时不许报。"""

    database = Database(tmp_path / "qichi.sqlite3")
    try:
        add_event(database, 1, direction="outbound", actor="qichi", kind="text", text="在的")
        add_fragment(database, 1, 1)
        add_detail(database, "f1", "e1", ordinal=0)

        memory_gaps.main(["--database", str(database.path)])

        output = capsys.readouterr().out
        assert "证据悬空：没有" in output
    finally:
        database.close()

def test_longest_uncovered_run_finds_the_contiguous_hole():
    """命中：中间一整块没人盖住。"""

    from memory_gaps import longest_uncovered_run

    assert longest_uncovered_run(1, 10, {1, 2, 6, 7, 8, 9, 10}) == (3, 3)


def test_longest_uncovered_run_ignores_scattered_holes():
    """不误判：零散的单条空洞（抽取器把两条并成一条）不算缺口。"""

    from memory_gaps import longest_uncovered_run

    assert longest_uncovered_run(1, 10, {1, 3, 4, 5, 7, 8, 9, 10}) == (2, 1)
    # 全被盖住时长度是 0（起点回落成区间起点，调用方只看长度）
    assert longest_uncovered_run(1, 5, {1, 2, 3, 4, 5})[1] == 0


def test_scattered_holes_are_not_repaired(tmp_path):
    """不误判（真机踩过）：零散单条空洞是抽取器合并事件的正常结果，不该去补。"""

    import repair_all_tails

    database = Database(tmp_path / "repaired.sqlite3")
    try:
        for sequence in range(1, 13):
            add_event(database, sequence, direction="outbound", actor="qichi", kind="text", text=f"m{sequence}")
        add_fragment(database, 1, 12)
        for sequence in (1, 2, 4, 5, 7, 8, 10, 11, 12):   # 3/6/9 三条零散空洞
            add_detail(database, "f1", f"e{sequence}", ordinal=sequence)

        assert repair_all_tails.holes(database.path) == [], "零散空洞不许去补"
    finally:
        database.close()


def test_a_contiguous_hole_is_reported_once_with_its_own_range(tmp_path):
    """命中：整块缺口报一次，并给出它自己的范围（不是片段的范围）。"""

    import repair_all_tails

    database = Database(tmp_path / "hole.sqlite3")
    try:
        for sequence in range(1, 41):
            add_event(database, sequence, direction="outbound", actor="qichi", kind="text", text=f"m{sequence}")
        add_fragment(database, 1, 40)                 # 一层覆盖
        for sequence in list(range(1, 11)) + list(range(31, 41)):
            add_detail(database, "f1", f"e{sequence}", ordinal=sequence)
        add_fragment(database, 11, 30)                # 重叠的第二层，指向同一个洞
        add_detail(database, "f11", "e30", ordinal=0)

        found = repair_all_tails.holes(database.path)

        assert len(found) == 1, "同一个洞只报一次（重叠片段不许各报一遍）"
        assert (found[0]["start"], found[0]["end"]) == (11, 29)
        assert found[0]["missing"] == 19
    finally:
        database.close()
