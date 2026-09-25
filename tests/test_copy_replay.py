from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import scripts.copy_replay as copy_replay


def test_default_questions_cover_hits_and_non_misfires():
    labels = " ".join(label for label, _ in copy_replay.DEFAULT_QUESTIONS)

    assert "命中" in labels, "至少要有一条应当展开的问法"
    assert "不误判" in labels, "至少要有一条不该展开的问法"


def test_load_questions_uses_labels_and_falls_back_to_numbering():
    parser = copy_replay.build_parser()

    default = copy_replay.load_questions(parser.parse_args([]))
    assert default == copy_replay.DEFAULT_QUESTIONS

    labelled = copy_replay.load_questions(
        parser.parse_args(["--question", "一", "--question", "二", "--label", "甲"])
    )
    assert labelled == (("甲", "一"), ("问题 2", "二"))


def test_the_tool_has_no_switch_that_writes_to_the_live_database():
    destinations = {action.dest for action in copy_replay.build_parser()._actions}

    assert "apply" not in destinations and "write" not in destinations
    assert "database" in destinations, "只读源库路径仍然是参数"


def test_copy_database_opens_the_source_read_only(tmp_path, monkeypatch):
    source = tmp_path / "live.sqlite3"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE probe (value TEXT)")
    connection.execute("INSERT INTO probe VALUES ('kept')")
    connection.commit()
    connection.close()

    seen: list[str] = []
    real_connect = sqlite3.connect

    def spy(database, *args, **kwargs):
        seen.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(copy_replay.sqlite3, "connect", spy)
    target = tmp_path / "copy.sqlite3"
    copy_replay.copy_database(source, target)

    assert any("mode=ro" in item for item in seen), "源库必须以只读方式打开"
    assert target.exists()
    copied = real_connect(target)
    try:
        assert copied.execute("SELECT value FROM probe").fetchone()[0] == "kept"
    finally:
        copied.close()


def test_render_report_prints_the_decisions_and_the_answer():
    results = [{
        "label": "命中-要求细说九号",
        "question": "细说一下九号那天中午我们讲了啥",
        "reason": "day",
        "fragments": 1,
        "details": 26,
        "details_tokens": 4236,
        "index_lines": [
            "- 09-09 中午 ~ 09-09 下午 · 含成人内容 · 已结束 · 26 条原文记录 · 细节已在本轮展开（26 条已附在下方）",
            "- 09-11 晚上 · 日常 · 已结束 · 12 条原文记录 · 细节未展开",
        ],
        "detail_labels": ["[片段 09-09 中午 ~ 09-09 下午 · 含成人内容 · 26 条]"],
        "reply": "九号那段现在能摊开了",
    }]

    rendered = copy_replay.render_report(results)

    assert "判据=day 片段=1 明细=26" in rendered
    assert "已在本轮展开（26 条已附在下方）" in rendered
    assert "12 条原文记录 · 细节未展开" in rendered, "封着的行也要打出来，才看得出 pin 有没有生效"
    assert "[片段 09-09 中午 ~ 09-09 下午 · 含成人内容 · 26 条]" in rendered
    assert "九号那段现在能摊开了" in rendered


def test_context_only_mode_never_touches_the_provider():
    llm = copy_replay.NullLLM()

    assert llm.calls == []


def test_the_outbound_recorder_never_sends_anything():
    client = copy_replay.RecordingOneBot()

    assert client.sent == []
