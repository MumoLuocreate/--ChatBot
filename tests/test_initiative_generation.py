"""主动开口这一路的生成策略：超时重试一次、非超时不重试、素材窗口更宽。

2026-09-15：真机 19:25 那次主动开口 90 秒超时，她一个字都没发出来，而用户完全不知情
（主动开口失败率 14%）。这一路不占用户等待，所以允许对超时多试一次；热路径不动。
同一天还查到：她的主动消息 20% 与上一条逐字相同，而 27/27 的重复都发生在他一条都没回的时候——
素材不变就复述。所以主动轮的足迹窗口单独放宽（仍只取 ordinary + daily_safe）。
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from qichi.dialogue.llm_client import LLMProtocolError, LLMTimeoutError

from test_voice_delivery import FakeNapCat, message, make_app, voiced_single_outcome, NOW, OWNER


class FlakyEngine:
    """先按脚本抛错，再吐预排好的结果。"""

    def __init__(self, errors, outcomes):
        self.errors = list(errors)
        self.outcomes = list(outcomes)
        self.calls = 0

    async def generate(self, value):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.outcomes.pop(0)


async def _ready_app(database, napcat):
    app = make_app(database, napcat, voiced_single_outcome(version=0))
    await app.handle_onebot(message(), received_at_utc=NOW)
    cursor = database.connection.execute(
        "SELECT context_version, last_user_activity_utc FROM conversation_cursors "
        "WHERE conversation_id = ?", (OWNER,),
    ).fetchone()
    version = int(cursor["context_version"])
    activity = datetime.fromisoformat(str(cursor["last_user_activity_utc"]))
    return app, version, activity


@pytest.mark.asyncio
async def test_an_initiative_timeout_is_retried_once(tmp_path):
    """命中：第一次超时、第二次成功 → 消息照发，引擎被调两次。"""

    from qichi.storage.database import Database

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        app, version, activity = await _ready_app(database, napcat)
        engine = FlakyEngine([LLMTimeoutError("timed out")], [voiced_single_outcome(version=version)])
        app.dialogue_engine = engine

        delivered = await app.generate_initiative(OWNER, version, activity, NOW)

        assert engine.calls == 2, "超时应当再试一次"
        assert delivered is not None
        assert napcat.texts, "重试成功之后她应当真的说话了"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_a_second_timeout_still_fails_and_says_so(tmp_path):
    """不误判的反面：两次都超时仍按失败处理，并且轨迹里留下重试次数。"""

    from qichi.storage.database import Database

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        app, version, activity = await _ready_app(database, napcat)
        engine = FlakyEngine(
            [LLMTimeoutError("timed out"), LLMTimeoutError("timed out")],
            [voiced_single_outcome(version=version)],
        )
        app.dialogue_engine = engine

        with pytest.raises(LLMTimeoutError):
            await app.generate_initiative(OWNER, version, activity, NOW)

        assert engine.calls == 2, "只多试一次，不许无限重试"
        trace = database.connection.execute(
            "SELECT details_json FROM turn_trace_events WHERE phase='failure' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert trace is not None
        assert json.loads(trace["details_json"])["timeout_retries"] == 1
    finally:
        database.close()


@pytest.mark.asyncio
async def test_a_non_timeout_error_is_not_retried(tmp_path):
    """不误判：协议错误不重试——重试救不了它，只会白烧一次调用。"""

    from qichi.storage.database import Database

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        app, version, activity = await _ready_app(database, napcat)
        engine = FlakyEngine(
            [LLMProtocolError("bad json")], [voiced_single_outcome(version=version)]
        )
        app.dialogue_engine = engine

        with pytest.raises(LLMProtocolError):
            await app.generate_initiative(OWNER, version, activity, NOW)

        assert engine.calls == 1, "非超时错误不许重试"
    finally:
        database.close()


@pytest.mark.asyncio
async def test_the_initiative_turn_gets_a_wider_footprint_than_a_normal_turn(tmp_path):
    """命中：主动轮的足迹窗口更宽；不误判：普通回复轮沿用常驻值。"""

    from qichi.storage.database import Database

    database = Database(tmp_path / "qichi.sqlite3")
    napcat = FakeNapCat()
    try:
        app, version, activity = await _ready_app(database, napcat)
        seen = []
        original = app.memory_details.footprint_details

        def recorder(conversation_id, **kwargs):
            seen.append((kwargs.get("per_fragment"), kwargs.get("limit")))
            return original(conversation_id, **kwargs)

        app.memory_details.footprint_details = recorder
        app.dialogue_engine = FlakyEngine([], [voiced_single_outcome(version=version)])
        await app.generate_initiative(OWNER, version, activity, NOW)
        initiative_call = seen[-1]

        seen.clear()
        app.dialogue_engine = FlakyEngine([], [voiced_single_outcome(version=version)])
        await app.handle_onebot(message(999), received_at_utc=NOW)
        normal_call = seen[-1]

        assert initiative_call == (6, 24), "主动轮用更宽的足迹"
        assert normal_call == (4, 12), "普通回复轮逐字不变"
    finally:
        database.close()
