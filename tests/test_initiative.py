from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from qichi.app import G0Application
from qichi.domain.dialogue import DialogueResult, DialogueSkip
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.initiative import InitiativePolicy, InitiativeScheduler
from qichi.storage.database import Database
from qichi.storage.outbox_repository import OutboxRepository
from qichi.storage.event_repository import EventRepository
from qichi.transport.onebot_client import OneBotActionError, OneBotTimeoutError

from test_interactions import FakeLLM, FakeNapCat, NOW, OWNER, make_app, message


class Clock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value


def scheduler(db, app, clock, **kwargs):
    return InitiativeScheduler(db, app, InitiativePolicy(enabled=True, **kwargs), clock=clock)


def _initiative_trace_phases(db, trigger_event_id):
    rows = db.connection.execute(
        "SELECT phase, details_json FROM turn_trace_events WHERE trigger_event_id = ? ORDER BY occurred_at_utc",
        (trigger_event_id,),
    ).fetchall()
    return [(row["phase"], json.loads(row["details_json"])) for row in rows]


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(key.casefold() in forbidden or _contains_key(child, forbidden) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


@pytest.mark.asyncio
async def test_initiative_skip_trace_uses_same_context_and_one_generation(tmp_path):
    db = Database(tmp_path / "initiative-trace-skip.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    llm = FakeLLM(["普通回复", "[[qichi:skip]]"])
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "上下文锚点"), received_at_utc=NOW)
        llm.calls.clear()
        result = await scheduler(db, app, clock).tick(OWNER)
        assert isinstance(result, DialogueSkip)
        trigger_id = db.connection.execute(
            "SELECT event_id FROM conversation_events WHERE direction='internal' AND kind='initiative'"
        ).fetchone()[0]
        phases = _initiative_trace_phases(db, trigger_id)
        assert [phase for phase, _ in phases] == ["received", "context", "generation", "delivery"]
        assert phases[0][1]["event_kind"] == "initiative"
        assert phases[2][1]["attempt_count"] == 1
        assert len(llm.calls) == 1
        assert all(not _contains_key(details, {"prompt", "messages", "reasoning", "text"}) for _, details in phases)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_generation_failure_trace_is_categorized_without_content(tmp_path):
    db = Database(tmp_path / "initiative-trace-failure.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    llm = FakeLLM(["普通回复", "", ""])
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "失败上下文"), received_at_utc=NOW)
        llm.calls.clear()
        assert await scheduler(db, app, clock).tick(OWNER) is None
        trigger_id = db.connection.execute(
            "SELECT event_id FROM conversation_events WHERE direction='internal' AND kind='initiative'"
        ).fetchone()[0]
        phases = _initiative_trace_phases(db, trigger_id)
        failure = next(details for phase, details in phases if phase == "failure")
        assert failure["stage"] == "generation"
        assert "failure_category" in failure
        assert len(llm.calls) == 2
        assert not _contains_key(failure, {"prompt", "messages", "reasoning", "text"})
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_cancel_trace_records_freshness_gate_without_delivery(tmp_path):
    db = Database(tmp_path / "initiative-trace-cancel.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))

    async def advance(call_number):
        if call_number == 1:
            db.connection.execute(
                "UPDATE conversation_cursors SET context_version = context_version + 1 WHERE conversation_id = ?",
                (OWNER,),
            )

    llm = FakeLLM(["普通回复", "过时主动"], advance)
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "取消锚点"), received_at_utc=NOW)
        llm.calls.clear()
        assert await scheduler(db, app, clock).tick(OWNER) is None
        trigger_id = db.connection.execute(
            "SELECT event_id FROM conversation_events WHERE direction='internal' AND kind='initiative'"
        ).fetchone()[0]
        phases = _initiative_trace_phases(db, trigger_id)
        assert [phase for phase, _ in phases] == ["received", "context", "generation", "cancelled"]
        assert phases[-1][1]["cancellation_category"] == "conversation_changed_after_generation"
        assert len(llm.calls) == 1
    finally:
        db.close()


def test_initiative_pause_resume_and_defer_are_durable_controls(tmp_path):
    db = Database(tmp_path / "initiative-controls.sqlite3")
    instance = scheduler(db, RecordingInitiativeApp(), Clock(NOW))
    try:
        instance.pause(OWNER)
        assert instance.control(OWNER)["paused"] is True
        db.close()
        db = Database(tmp_path / "initiative-controls.sqlite3")
        reopened = scheduler(db, RecordingInitiativeApp(), Clock(NOW))
        assert reopened.control(OWNER)["paused"] is True
        reopened.resume(OWNER)
        deferred_until = NOW + timedelta(minutes=120)
        reopened.defer_once(OWNER, deferred_until)
        assert reopened.control(OWNER)["deferred_until"] == deferred_until.isoformat()
    finally:
        db.close()


def test_initiative_control_rejects_naive_or_past_defer_without_mutation(tmp_path):
    db = Database(tmp_path / "initiative-control-invalid.sqlite3")
    instance = scheduler(db, RecordingInitiativeApp(), Clock(NOW))
    try:
        before = instance.control(OWNER)
        with pytest.raises(ValueError):
            instance.defer_once(OWNER, NOW.replace(tzinfo=None))
        assert instance.control(OWNER) == before
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_defer_only_delays_next_claim_and_survives_clock_rollback(tmp_path):
    db = Database(tmp_path / "initiative-defer.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    app = RecordingInitiativeApp()
    try:
        seed_cursor(db)
        instance = scheduler(db, app, clock)
        instance.defer_once(OWNER, NOW + timedelta(minutes=120))
        assert await instance.tick(OWNER) is None
        clock.value = NOW + timedelta(minutes=119)
        assert await instance.tick(OWNER) is None
        clock.value = NOW + timedelta(minutes=120)
        assert isinstance(await instance.tick(OWNER), DialogueSkip)
        assert instance.control(OWNER)["deferred_until"] is None
        clock.value = NOW + timedelta(minutes=119)
        assert await instance.tick(OWNER) is None
    finally:
        db.close()


def test_initiative_timeline_projection_has_fixed_categories_and_no_prompt_fields(tmp_path):
    db = Database(tmp_path / "initiative-timeline.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    try:
        seed_cursor(db)
        instance = scheduler(db, RecordingInitiativeApp(), clock)
        claim = instance._claim(OWNER, clock.value)
        assert claim is not None
        EventRepository(db).insert(ConversationEvent(
            event_id=claim["claim_id"], platform_event_id=None, platform_message_id=None,
            conversation_id=OWNER, sequence=0, direction="internal", actor="platform",
            kind="initiative", text=None, message_segments=(), reply_to_event_id=None,
            reply_to_platform_message_id=None, occurred_at_utc=clock.value,
            received_at_utc=clock.value, status="received", metadata={},
        ))
        assert instance._complete_state(OWNER, claim, "failed", clock.value)
        timeline = instance.timeline(OWNER)
        assert [item["category"] for item in timeline] == ["due", "claim", "failed"]
        assert timeline[-1]["failure_category"] == "generation_failed"
        assert all("prompt" not in item for item in timeline)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_long_idle_bounds_history_at_latest_user_topic(tmp_path):
    """A delayed initiative must not replay the whole prior conversation."""
    db = Database(tmp_path / "initiative-history-boundary.sqlite3")
    first = NOW
    second = NOW + timedelta(minutes=1)
    attempt = NOW + timedelta(hours=9)
    clock = Clock(attempt)
    llm = FakeLLM(["旧话题回复", "最新话题回复", "主动联系"])
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "旧连续话题"), received_at_utc=first)
        await app.handle_onebot(
            message(2, "最新用户话题", second.timestamp()),
            received_at_utc=second,
        )

        result = await scheduler(db, app, clock).tick(OWNER)

        assert result is not None and result.status == "sent"
        prompt = "\n".join(item.content for item in llm.calls[2])
        assert "最新用户话题" in prompt
        assert "最新话题回复" in prompt
        assert "旧连续话题" not in prompt
        assert "旧话题回复" not in prompt
        trace = db.connection.execute(
            "SELECT details_json FROM turn_trace_events WHERE phase='context' "
            "AND source='initiative'"
        ).fetchone()
        details = json.loads(trace[0])
        assert details["selected_history_event_ids"]
        assert len(details["selected_history_event_ids"]) == 2
        trigger_metadata = json.loads(
            db.connection.execute(
                "SELECT metadata_json FROM conversation_events "
                "WHERE direction='internal' AND kind='initiative'"
            ).fetchone()[0]
        )
        latest_user_sequence = db.connection.execute(
            "SELECT MAX(sequence) FROM conversation_events "
            "WHERE direction='inbound' AND actor='mumo'"
        ).fetchone()[0]
        assert trigger_metadata["topic_watermark"] == latest_user_sequence
    finally:
        db.close()


def test_initiative_history_boundary_does_not_backdate_on_clock_rollback(tmp_path):
    """An initiative timestamp before the latest user event must not drop history."""
    db = Database(tmp_path / "initiative-history-rollback.sqlite3")
    events = EventRepository(db)
    def add(event_id, text, actor, direction, at):
        return events.insert(ConversationEvent(
            event_id=event_id,
            platform_event_id=f"pe-{event_id}",
            platform_message_id=f"pm-{event_id}",
            conversation_id=OWNER,
            sequence=0,
            direction=direction,
            actor=actor,
            kind="text",
            text=text,
            message_segments=(MessageSegment("text", {"text": text}),),
            reply_to_event_id=None,
            reply_to_platform_message_id=None,
            occurred_at_utc=at,
            received_at_utc=at,
            status="received" if direction == "inbound" else "sent",
            metadata={"generation_metadata": {"source": "dialogue"}}
            if direction == "outbound" else {},
        ))
    latest = add("latest-user", "最近一句", "mumo", "inbound", NOW)
    reply = add("latest-reply", "最近回复", "qichi", "outbound", NOW + timedelta(seconds=1))
    try:
        history = G0Application._session_history(
            (latest, reply), boundary_at_utc=NOW - timedelta(minutes=1)
        )
        assert history == (latest, reply)
    finally:
        db.close()


def seed_cursor(database, *, activity=NOW, context_version=1):
    database.connection.execute(
        "INSERT INTO conversation_cursors "
        "(conversation_id, context_version, last_user_activity_utc) VALUES (?, ?, ?)",
        (OWNER, context_version, activity.isoformat()),
    )


class RecordingInitiativeApp:
    def __init__(self, result=None, *, entered=None, release=None):
        self.result = result if result is not None else DialogueSkip("primary", 1)
        self.entered = entered
        self.release = release
        self.calls = []

    async def generate_initiative(self, *args):
        self.calls.append(args)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return self.result


class RecordingWorker:
    def __init__(self):
        self.calls = []

    def notify_reliable_activity(self, conversation_id):
        self.calls.append((conversation_id,))
        return True


class BlockingInitiativeNapCat(FakeNapCat):
    def __init__(self):
        super().__init__()
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def send_private_msg(self, user_id, payload):
        self.texts.append((str(user_id), payload))
        if self.block:
            self.entered.set()
            await self.release.wait()
        return {"message_id": 700 + len(self.texts)}


@pytest.mark.asyncio
async def test_initiative_disabled_and_before_due_are_noops(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock()
    app = make_app(db, FakeLLM(["不会用到"]), FakeNapCat())
    try:
        await app.handle_onebot(message(1, "你好"), received_at_utc=NOW)
        assert await InitiativeScheduler(db, app, InitiativePolicy(), clock=clock).tick(OWNER) is None
        clock.value = NOW + timedelta(minutes=59, seconds=59)
        assert await scheduler(db, app, clock).tick(OWNER) is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_second_initiative_tells_her_the_first_one_was_not_answered(tmp_path):
    """2026-09-14 用户裁定：他没回时，下一次主动开口要把这件事带进上下文。"""
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    llm = FakeLLM(["普通回复", "第一次主动", "第二次主动"])
    app = make_app(db, llm, FakeNapCat())
    try:
        await app.handle_onebot(message(1, "最近好吗"), received_at_utc=NOW)
        assert (await scheduler(db, app, clock).tick(OWNER)).status == "sent"
        clock.value = NOW + timedelta(minutes=120)
        assert (await scheduler(db, app, clock).tick(OWNER)).status == "sent"
        first = "\n".join(item.content for item in llm.calls[1])
        second = "\n".join(item.content for item in llm.calls[2])
        # 第一次开口没有「上一次」可说：计数不该把正常一问一答算成主动开口。
        assert "主动联系尝试" in first
        assert "次主动开口" not in first
        assert "本次是第 2 次主动开口" in second
        assert "之后用户没有回复" in second
        # 这句必须留在本轮事实上：稳定半要逐轮逐字节相同，否则整段前缀缓存失效。
        assert "次主动开口" not in second.split("[本轮事实]")[0]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_answered_initiative_restarts_the_count_at_one(tmp_path):
    """他回了一句之后重新计时：下一轮主动开口回到第 1 次，不再提旧账。"""
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    llm = FakeLLM(["普通回复", "第一次主动", "他回复后的回复", "新一轮主动"])
    napcat = FakeNapCat()
    app = make_app(db, llm, napcat)
    try:
        await app.handle_onebot(message(1, "最近好吗"), received_at_utc=NOW)
        assert (await scheduler(db, app, clock).tick(OWNER)).status == "sent"
        clock.value = NOW + timedelta(minutes=90)
        await app.handle_onebot(message(2, "在忙"), received_at_utc=NOW + timedelta(minutes=90))
        clock.value = NOW + timedelta(minutes=150)
        assert (await scheduler(db, app, clock).tick(OWNER)).status == "sent"
        latest = "\n".join(item.content for item in llm.calls[-1])
        assert "次主动开口" not in latest
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_due_uses_same_engine_source_and_sender(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    napcat = FakeNapCat()
    llm = FakeLLM(["普通回复", "主动说一句"])
    app = make_app(db, llm, napcat)
    try:
        await app.handle_onebot(message(1, "最近好吗"), received_at_utc=NOW)
        result = await scheduler(db, app, clock).tick(OWNER)
        assert result is not None and result.status == "sent"
        assert len(llm.calls) == 2
        prompt = "\n".join(item.content for item in llm.calls[1])
        assert "主动联系尝试" in prompt
        assert "最近好吗" in prompt and "普通回复" in prompt
        assert napcat.texts[-1][1][0]["data"]["text"] == "主动说一句"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_skip_has_no_text_event_or_outbox_and_advances_slot(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    llm = FakeLLM(["普通回复", "[[qichi:skip]]", "下一槽"])
    app = make_app(db, llm, FakeNapCat())
    try:
        await app.handle_onebot(message(1, "在吗"), received_at_utc=NOW)
        before_text = db.connection.execute("SELECT COUNT(*) FROM conversation_events WHERE kind='text'").fetchone()[0]
        assert await scheduler(db, app, clock).tick(OWNER) is not None
        after_text = db.connection.execute("SELECT COUNT(*) FROM conversation_events WHERE kind='text'").fetchone()[0]
        assert after_text == before_text and not OutboxRepository(db).outstanding()
        assert await scheduler(db, app, clock).tick(OWNER) is None
        clock.value = NOW + timedelta(minutes=120)
        assert (await scheduler(db, app, clock).tick(OWNER)).status == "sent"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_new_user_message_resets_and_cancels_stale_attempt(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def before(number):
        if number == 2:
            entered.set()
            await release.wait()

    llm = FakeLLM(["普通回复", "旧主动", "新文字"], before)
    napcat = FakeNapCat()
    app = make_app(db, llm, napcat)
    try:
        await app.handle_onebot(message(1, "旧话题"), received_at_utc=NOW)
        task = asyncio.create_task(scheduler(db, app, clock).tick(OWNER))
        await entered.wait()
        text_task = asyncio.create_task(app.handle_onebot(message(2, "我回来了"), received_at_utc=clock.value))
        await asyncio.sleep(0)
        release.set()
        assert await task is None
        await text_task
        assert napcat.texts[-1][1][0]["data"]["text"] == "新文字"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_duplicate_ticks_are_idempotent_and_state_survives_restart(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    clock = Clock(NOW + timedelta(minutes=60))
    napcat = FakeNapCat()
    llm = FakeLLM(["普通回复", "一次"])
    db = Database(path)
    app = make_app(db, llm, napcat)
    await app.handle_onebot(message(1, "你好"), received_at_utc=NOW)
    results = await asyncio.gather(scheduler(db, app, clock).tick(OWNER), scheduler(db, app, clock).tick(OWNER))
    assert sum(item is not None for item in results) == 1
    db.close()
    reopened = Database(path)
    try:
        assert await scheduler(reopened, make_app(reopened, llm, napcat), clock).tick(OWNER) is None
        assert len(napcat.texts) == 2
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_initiative_unknown_and_exception_are_terminal_for_slot(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    napcat = FakeNapCat()
    llm = FakeLLM(["普通回复", "会超时"])
    app = make_app(db, llm, napcat)
    original = app.sender.client.send_private_msg

    async def unknown(*args):
        from qichi.transport.onebot_client import OneBotTimeoutError
        raise OneBotTimeoutError("unknown")

    napcat.send_private_msg = unknown
    try:
        await app.handle_onebot(message(1, "你好"), received_at_utc=NOW)
        result = await scheduler(db, app, clock).tick(OWNER)
        assert result is not None and result.status == "unknown"
        assert await scheduler(db, app, clock).tick(OWNER) is None
        assert len(llm.calls) == 2
    finally:
        napcat.send_private_msg = original
        db.close()


@pytest.mark.asyncio
async def test_two_unanswered_sent_attempts_pause_until_new_user_activity(tmp_path):
    path = tmp_path / "unanswered-limit.sqlite3"
    db = Database(path)
    clock = Clock(NOW + timedelta(minutes=60))
    app = RecordingInitiativeApp(SimpleNamespace(status="sent"))
    try:
        seed_cursor(db)
        first = await scheduler(db, app, clock).tick(OWNER)
        assert first is not None and first.status == "sent"

        # The unanswered count must survive the same close/reopen boundary as
        # the activity anchor and initiative slot.
        db.close()
        db = Database(path)
        clock.value = NOW + timedelta(minutes=120)
        second = await scheduler(db, app, clock).tick(OWNER)
        assert second is not None and second.status == "sent"

        clock.value = NOW + timedelta(minutes=180)
        assert await scheduler(db, app, clock).tick(OWNER) is None
        assert len(app.calls) == 2
        state = json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
            ).fetchone()[0]
        )
        assert state["unanswered_attempts"] == 2

        # A fresh user activity anchor resets the unanswered sequence.
        db.connection.execute(
            "UPDATE conversation_cursors SET context_version = ?, last_user_activity_utc = ? "
            "WHERE conversation_id = ?",
            (2, (NOW + timedelta(minutes=181)).isoformat(), OWNER),
        )
        clock.value = NOW + timedelta(minutes=241)
        third = await scheduler(db, app, clock).tick(OWNER)
        assert third is not None and third.status == "sent"
        assert len(app.calls) == 3
    finally:
        db.close()


@pytest.mark.asyncio
async def test_skipped_failed_and_unknown_attempts_do_not_consume_unanswered_budget(tmp_path):
    db = Database(tmp_path / "unanswered-statuses.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    app = RecordingInitiativeApp(DialogueSkip("primary", 1))
    try:
        seed_cursor(db)
        assert isinstance(await scheduler(db, app, clock).tick(OWNER), DialogueSkip)
        state = json.loads(db.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
        ).fetchone()[0])
        assert state["unanswered_attempts"] == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_unknown_waits_for_reconciliation_across_all_later_slots(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    app = RecordingInitiativeApp(SimpleNamespace(status="unknown"))
    try:
        seed_cursor(db)
        result = await scheduler(db, app, clock).tick(OWNER)
        assert result.status == "unknown"
        assert len(app.calls) == 1

        for elapsed in (120, 180, 24 * 60):
            clock.value = NOW + timedelta(minutes=elapsed)
            assert await scheduler(db, app, clock).tick(OWNER) is None
        assert len(app.calls) == 1
        state = json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?",
                (f"initiative:{OWNER}",),
            ).fetchone()[0]
        )
        assert state["status"] == "unknown"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_long_offline_coalesces_missed_slots_and_rearms_from_attempt_time(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    offline_now = NOW + timedelta(hours=10)
    clock = Clock(offline_now)
    app = RecordingInitiativeApp()
    try:
        seed_cursor(db)
        assert isinstance(await scheduler(db, app, clock).tick(OWNER), DialogueSkip)
        assert len(app.calls) == 1

        for _ in range(5):
            assert await scheduler(db, app, clock).tick(OWNER) is None
        assert len(app.calls) == 1

        clock.value = offline_now + timedelta(minutes=59, seconds=59)
        assert await scheduler(db, app, clock).tick(OWNER) is None
        assert len(app.calls) == 1
        clock.value = offline_now + timedelta(minutes=60)
        assert isinstance(await scheduler(db, app, clock).tick(OWNER), DialogueSkip)
        assert len(app.calls) == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_crashed_claim_with_durable_trigger_is_not_generated_or_sent_again(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    clock = Clock(NOW + timedelta(minutes=60))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block_initiative(call_number):
        if call_number == 2:
            entered.set()
            await release.wait()

    llm = FakeLLM(["普通回复", "不会完成"], block_initiative)
    napcat = FakeNapCat()
    db = Database(path)
    app = make_app(db, llm, napcat)
    await app.handle_onebot(message(1, "留下锚点"), received_at_utc=NOW)
    task = asyncio.create_task(scheduler(db, app, clock).tick(OWNER))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert db.connection.execute(
        "SELECT COUNT(*) FROM conversation_events WHERE direction='internal' AND kind='initiative'"
    ).fetchone()[0] == 1
    before = (len(llm.calls), len(napcat.texts))
    db.close()

    reopened = Database(path)
    try:
        clock.value = NOW + timedelta(minutes=121)
        await scheduler(
            reopened, make_app(reopened, llm, napcat), clock
        ).tick(OWNER)
        assert (len(llm.calls), len(napcat.texts)) == before
        assert reopened.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE direction='internal' AND kind='initiative'"
        ).fetchone()[0] == 1
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_expired_orphan_trigger_is_converged_with_fixed_failure_observation(tmp_path):
    path = tmp_path / "orphan-trigger.sqlite3"
    db = Database(path)
    clock = Clock(NOW + timedelta(minutes=60))
    app = RecordingInitiativeApp()
    try:
        seed_cursor(db)
        claim = scheduler(db, app, clock)._claim(OWNER, clock.value)
        assert claim is not None
        trigger = ConversationEvent(
            event_id=claim["claim_id"], platform_event_id=None, platform_message_id=None,
            conversation_id=OWNER, sequence=0, direction="internal", actor="platform",
            kind="initiative", text=None, message_segments=(), reply_to_event_id=None,
            reply_to_platform_message_id=None, occurred_at_utc=clock.value,
            received_at_utc=clock.value, status="received", metadata={},
        )
        EventRepository(db).insert(trigger)
        clock.value = NOW + timedelta(minutes=121)
        assert await scheduler(db, app, clock).tick(OWNER) is None
        row = db.connection.execute(
            "SELECT status FROM conversation_events WHERE event_id = ?",
            (claim["claim_id"],),
        ).fetchone()
        assert row["status"] == "failed"
        state = json.loads(db.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
        ).fetchone()[0])
        assert state["failure_category"] == "recovered_orphan"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_completed_initiative_trigger_observation_is_not_failure(tmp_path):
    db = Database(tmp_path / "terminal-observation.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    app = RecordingInitiativeApp()
    try:
        seed_cursor(db)
        instance = scheduler(db, app, clock)
        claim = instance._claim(OWNER, clock.value)
        assert claim is not None
        EventRepository(db).insert(ConversationEvent(
            event_id=claim["claim_id"], platform_event_id=None, platform_message_id=None,
            conversation_id=OWNER, sequence=0, direction="internal", actor="platform",
            kind="initiative", text=None, message_segments=(), reply_to_event_id=None,
            reply_to_platform_message_id=None, occurred_at_utc=clock.value,
            received_at_utc=clock.value, status="received", metadata={},
        ))
        assert instance._complete_state(OWNER, claim, "skipped", clock.value)
        row = db.connection.execute(
            "SELECT status FROM conversation_events "
            "WHERE direction = 'internal' AND kind = 'initiative'"
        ).fetchone()
        assert row["status"] == "skipped"
        state = json.loads(db.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
        ).fetchone()[0])
        assert state.get("failure_category") is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_crashed_claim_with_dispatched_outbox_is_never_sent_again(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    clock = Clock(NOW + timedelta(minutes=60))
    napcat = BlockingInitiativeNapCat()
    llm = FakeLLM(["普通回复", "主动正文"])
    db = Database(path)
    app = make_app(db, llm, napcat)
    await app.handle_onebot(message(1, "留下锚点"), received_at_utc=NOW)
    napcat.block = True
    task = asyncio.create_task(scheduler(db, app, clock).tick(OWNER))
    await napcat.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    dispatched = db.connection.execute(
        "SELECT status FROM outbox WHERE status='dispatched'"
    ).fetchall()
    assert len(dispatched) == 1
    calls_before = len(napcat.texts)
    db.close()

    reopened = Database(path)
    try:
        clock.value = NOW + timedelta(minutes=121)
        await scheduler(
            reopened, make_app(reopened, llm, napcat), clock
        ).tick(OWNER)
        assert len(napcat.texts) == calls_before
        assert reopened.connection.execute(
            "SELECT COUNT(*) FROM outbox WHERE status='dispatched'"
        ).fetchone()[0] == 1
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_crashed_claim_without_trigger_is_recovered_once_after_lease(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    db = Database(path)
    seed_cursor(db)
    entered = asyncio.Event()
    release = asyncio.Event()
    crashed_app = RecordingInitiativeApp(entered=entered, release=release)
    clock = Clock(NOW + timedelta(minutes=60))
    task = asyncio.create_task(scheduler(db, crashed_app, clock).tick(OWNER))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    db.close()

    reopened = Database(path)
    recovered_app = RecordingInitiativeApp()
    try:
        clock.value = NOW + timedelta(minutes=119, seconds=59)
        assert await scheduler(reopened, recovered_app, clock).tick(OWNER) is None
        clock.value = NOW + timedelta(minutes=120)
        assert isinstance(
            await scheduler(reopened, recovered_app, clock).tick(OWNER), DialogueSkip
        )
        assert len(recovered_app.calls) == 1
        assert await scheduler(reopened, recovered_app, clock).tick(OWNER) is None
        assert len(recovered_app.calls) == 1
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_old_completion_cannot_overwrite_new_owner_claim_across_connections(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    first_db = Database(path)
    second_db = Database(path)
    seed_cursor(first_db)
    entered = asyncio.Event()
    release = asyncio.Event()
    old_app = RecordingInitiativeApp(entered=entered, release=release)
    old_clock = Clock(NOW + timedelta(minutes=60))
    new_clock = Clock(NOW + timedelta(minutes=120))
    try:
        old_task = asyncio.create_task(
            scheduler(first_db, old_app, old_clock).tick(OWNER)
        )
        await entered.wait()
        new_app = RecordingInitiativeApp()
        assert isinstance(
            await scheduler(second_db, new_app, new_clock).tick(OWNER), DialogueSkip
        )
        state_after_new_owner = second_db.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[:]

        release.set()
        assert isinstance(await old_task, DialogueSkip)
        assert first_db.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[:] == state_after_new_owner
    finally:
        first_db.close()
        second_db.close()


@pytest.mark.asyncio
async def test_same_scheduler_owner_recovers_expired_claim_without_durable_evidence(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    entered = asyncio.Event()
    release = asyncio.Event()
    app = RecordingInitiativeApp(entered=entered, release=release)
    clock = Clock(NOW + timedelta(minutes=60))
    same_scheduler = scheduler(db, app, clock)
    try:
        seed_cursor(db)
        task = asyncio.create_task(same_scheduler.tick(OWNER))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        app.entered = None
        app.release = None
        clock.value = NOW + timedelta(minutes=120)
        assert isinstance(await same_scheduler.tick(OWNER), DialogueSkip)
        assert len(app.calls) == 2
    finally:
        db.close()


def test_completion_is_monotonic_and_cannot_rewrite_terminal_state(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    instance = scheduler(db, RecordingInitiativeApp(), clock)
    try:
        seed_cursor(db)
        claimed = instance._claim(OWNER, clock.value)
        assert claimed is not None
        assert instance._complete_state(OWNER, claimed, "skipped", clock.value)
        terminal = db.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[:]
        assert not instance._complete_state(OWNER, claimed, "sent", clock.value + timedelta(seconds=1))
        assert db.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[:] == terminal
    finally:
        db.close()


def test_completion_clock_rollback_is_anchored_to_claim_and_next_due_is_future(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    claim_time = NOW + timedelta(minutes=60)
    instance = scheduler(db, RecordingInitiativeApp(), Clock(claim_time))
    try:
        seed_cursor(db)
        claimed = instance._claim(OWNER, claim_time)
        assert claimed is not None

        assert instance._complete_state(
            OWNER, claimed, "sent", NOW - timedelta(hours=1)
        )
        state = json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?",
                (f"initiative:{OWNER}",),
            ).fetchone()[0]
        )
        completed_at = datetime.fromisoformat(state["completed_at"])
        next_due_at = datetime.fromisoformat(state["next_due_at"])
        assert completed_at >= claim_time
        assert next_due_at > completed_at
        assert instance._claim(OWNER, claim_time) is None
    finally:
        db.close()


def _valid_persisted_state(**changes):
    state = {
        "slot": (NOW + timedelta(minutes=60)).isoformat(),
        "status": "skipped",
        "owner": "owner",
        "claim_id": "claim",
        "claimed_at": (NOW + timedelta(minutes=60)).isoformat(),
        "activity": NOW.isoformat(),
        "context_version": 1,
        "outbound_event_id": "outbound",
        "unanswered_attempts": 0,
        "completed_at": (NOW + timedelta(minutes=60)).isoformat(),
        "next_due_at": (NOW + timedelta(minutes=120)).isoformat(),
    }
    state.update(changes)
    return state


@pytest.mark.parametrize(
    "raw_state",
    [
        "{",
        "[]",
        json.dumps(_valid_persisted_state(status="unexpected")),
        json.dumps(_valid_persisted_state(slot="2026-08-28T11:00:00")),
        json.dumps(_valid_persisted_state(activity="2026-08-28T10:00:00")),
        json.dumps(_valid_persisted_state(claimed_at="2026-08-28T11:00:00")),
        json.dumps({key: value for key, value in _valid_persisted_state().items() if key != "owner"}),
        json.dumps({key: value for key, value in _valid_persisted_state().items() if key != "unanswered_attempts"}),
        json.dumps({key: value for key, value in _valid_persisted_state().items() if key != "next_due_at"}),
    ],
    ids=["invalid-json", "non-object", "unknown-status", "naive-slot", "naive-activity", "naive-claimed-at", "missing-owner", "missing-unanswered-attempts", "missing-next-due"],
)
@pytest.mark.asyncio
async def test_invalid_persisted_state_fails_closed_without_mutation(tmp_path, raw_state):
    db = Database(tmp_path / "qichi.sqlite3")
    try:
        seed_cursor(db)
        updated = NOW.isoformat()
        db.connection.execute(
            "INSERT INTO runtime_meta(key, value_json, updated_at_utc) VALUES (?, ?, ?)",
            (f"initiative:{OWNER}", raw_state, updated),
        )
        before = db.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[:]
        app = RecordingInitiativeApp()
        with pytest.raises(ValueError):
            await scheduler(db, app, Clock(NOW + timedelta(hours=2))).tick(OWNER)
        assert app.calls == []
        assert db.connection.execute(
            "SELECT value_json, updated_at_utc FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[:] == before
    finally:
        db.close()


@pytest.mark.parametrize(
    "state",
    [
        _valid_persisted_state(
            status="claimed", completed_at=(NOW + timedelta(minutes=60)).isoformat(), next_due_at=None
        ),
        _valid_persisted_state(status="claimed", completed_at=None, next_due_at=None, outbound_event_id=None),
        _valid_persisted_state(status="unknown", next_due_at=(NOW + timedelta(minutes=120)).isoformat()),
        _valid_persisted_state(status="skipped", completed_at=None),
        _valid_persisted_state(next_due_at=(NOW + timedelta(minutes=60)).isoformat()),
        _valid_persisted_state(
            activity=(NOW + timedelta(minutes=61)).isoformat(),
            slot=(NOW + timedelta(minutes=60)).isoformat(),
        ),
        _valid_persisted_state(
            completed_at=(NOW + timedelta(minutes=59)).isoformat(),
        ),
    ],
    ids=[
        "claimed-has-completion",
        "claimed-missing-outbound",
        "unknown-has-next-due",
        "terminal-missing-completion",
        "next-due-not-after-completion",
        "activity-after-slot",
        "completion-before-claim",
    ],
)
@pytest.mark.asyncio
async def test_status_specific_persisted_state_invariants_fail_closed(tmp_path, state):
    db = Database(tmp_path / "qichi.sqlite3")
    try:
        seed_cursor(db)
        raw = json.dumps(state)
        updated_at = state.get("completed_at") or state.get("claimed_at") or NOW.isoformat()
        db.connection.execute(
            "INSERT INTO runtime_meta(key, value_json, updated_at_utc) VALUES (?, ?, ?)",
            (f"initiative:{OWNER}", raw, updated_at),
        )
        with pytest.raises(ValueError):
            await scheduler(db, RecordingInitiativeApp(), Clock(NOW + timedelta(hours=3))).tick(OWNER)
        assert db.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
        ).fetchone()[0] == raw
    finally:
        db.close()


@pytest.mark.parametrize(
    "state",
    [
        _valid_persisted_state(failure_category="delivery_unknown"),
        _valid_persisted_state(failure_category=None),
    ],
    ids=["optional-category", "nullable-category"],
)
def test_decode_state_accepts_complete_required_fields_and_optional_failure_category(tmp_path, state):
    db = Database(tmp_path / "complete-state.sqlite3")
    try:
        decoded = scheduler(db, RecordingInitiativeApp(), Clock(NOW))._decode_state(json.dumps(state))
        assert decoded["context_version"] == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_valid_terminal_persisted_state_is_accepted_without_mutation(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    state = _valid_persisted_state()
    raw = json.dumps(state)
    try:
        seed_cursor(db)
        db.connection.execute(
            "INSERT INTO runtime_meta(key, value_json, updated_at_utc) VALUES (?, ?, ?)",
            (f"initiative:{OWNER}", raw, state["completed_at"]),
        )
        app = RecordingInitiativeApp()
        assert await scheduler(
            db, app, Clock(NOW + timedelta(minutes=90))
        ).tick(OWNER) is None
        assert app.calls == []
        assert db.connection.execute(
            "SELECT value_json FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[0] == raw
    finally:
        db.close()


@pytest.mark.parametrize(
    "field",
    [
        "reset_on_user_message",
        "use_same_dialogue_engine",
        "allow_model_to_skip",
        "cancel_if_conversation_changed",
    ],
)
@pytest.mark.parametrize("invalid", [False, 0, 1, "true", None])
def test_policy_requires_fixed_true_boolean_fields(field, invalid):
    with pytest.raises((TypeError, ValueError)):
        InitiativePolicy(**{field: invalid})


@pytest.mark.parametrize("invalid", [1, 59, 61, 60.0, True, "60", None])
def test_policy_requires_exact_integer_sixty_minutes(invalid):
    with pytest.raises((TypeError, ValueError)):
        InitiativePolicy(idle_attempt_minutes=invalid)


@pytest.mark.parametrize("invalid", [True, 1, 0, "false", None])
def test_policy_requires_unknown_retry_to_be_strict_false(invalid):
    with pytest.raises((TypeError, ValueError)):
        InitiativePolicy(unknown_delivery_retry=invalid)


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_policy_enabled_is_strict_boolean(invalid):
    with pytest.raises((TypeError, ValueError)):
        InitiativePolicy(enabled=invalid)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2026, 8, 28, 14, 59, tzinfo=timezone.utc), False),
        (datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 8, 28, 16, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 8, 28, 23, 59, tzinfo=timezone.utc), True),
        (datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc), False),
    ],
    ids=["2259", "2300", "midnight", "0759", "0800"],
)
def test_cross_midnight_quiet_hours_use_shanghai_boundaries(value, expected):
    policy = InitiativePolicy(quiet_hours_local="23:00-08:00")
    assert policy.is_quiet_at(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2026, 8, 29, 0, 59, tzinfo=timezone.utc), False),
        (datetime(2026, 8, 29, 1, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc), False),
    ],
    ids=["0859", "0900", "1700"],
)
def test_same_day_quiet_hours_have_the_same_half_open_boundaries(value, expected):
    policy = InitiativePolicy(quiet_hours_local="09:00-17:00")
    assert policy.is_quiet_at(value) is expected


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        "23:00",
        "23:00 - 08:00",
        "24:00-08:00",
        "23:60-08:00",
        "23:00-23:00",
        2300,
        True,
    ],
)
def test_policy_rejects_invalid_or_ambiguous_quiet_hours(invalid):
    with pytest.raises((TypeError, ValueError)):
        InitiativePolicy(quiet_hours_local=invalid)


@pytest.mark.asyncio
async def test_quiet_hours_do_not_claim_or_generate_until_window_ends(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    activity = datetime(2026, 8, 28, 13, 30, tzinfo=timezone.utc)
    clock = Clock(datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc))
    app = RecordingInitiativeApp()
    instance = scheduler(
        db,
        app,
        clock,
        quiet_hours_local="23:00-08:00",
    )
    try:
        seed_cursor(db, activity=activity)
        assert await instance.tick(OWNER) is None
        clock.value = datetime(2026, 8, 28, 23, 59, tzinfo=timezone.utc)
        assert await instance.tick(OWNER) is None
        assert app.calls == []
        assert db.connection.execute(
            "SELECT COUNT(*) FROM runtime_meta WHERE key = ?",
            (f"initiative:{OWNER}",),
        ).fetchone()[0] == 0

        clock.value = datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc)
        assert await instance.tick(OWNER) is None
        assert app.calls == []

        clock.value = datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc)
        assert isinstance(await instance.tick(OWNER), DialogueSkip)
        assert len(app.calls) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_first_due_slot_at_eight_is_allowed_without_catch_up_delay(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    activity = datetime(2026, 8, 28, 23, 0, tzinfo=timezone.utc)
    clock = Clock(datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc))
    app = RecordingInitiativeApp()
    try:
        seed_cursor(db, activity=activity)
        result = await scheduler(
            db,
            app,
            clock,
            quiet_hours_local="23:00-08:00",
        ).tick(OWNER)
        assert isinstance(result, DialogueSkip)
        assert len(app.calls) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_generation_crossing_quiet_boundary_is_cancelled_before_dispatch(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    activity = datetime(2026, 8, 28, 13, 59, tzinfo=timezone.utc)
    attempt = datetime(2026, 8, 28, 14, 59, tzinfo=timezone.utc)
    clock = Clock(attempt)
    napcat = FakeNapCat()

    async def cross_boundary(call_number):
        if call_number == 2:
            clock.value = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)

    app = make_app(
        db,
        FakeLLM(["普通回复", "不得在静默期发送"], cross_boundary),
        napcat,
        clock=clock,
    )
    try:
        await app.handle_onebot(
            message(1, "晚点聊", activity.timestamp()),
            received_at_utc=activity,
        )
        before_outbox = db.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

        result = await scheduler(
            db,
            app,
            clock,
            quiet_hours_local="23:00-08:00",
        ).tick(OWNER)

        assert result is None
        assert [segments[0]["data"]["text"] for _, segments in napcat.texts] == ["普通回复"]
        assert db.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == before_outbox
        assert db.connection.execute(
            "SELECT status FROM conversation_events WHERE kind = 'initiative'"
        ).fetchone()[0] == "cancelled"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_prompt_has_platform_trigger_bounded_history_attempt_time_and_no_reaction(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    attempt = NOW + timedelta(hours=2)
    clock = Clock(attempt)
    llm = FakeLLM(["qichi-one", "qichi-two", "initiative-result"])
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "mumo-one"), received_at_utc=NOW)
        await app.handle_onebot(
            message(2, "mumo-two", (NOW + timedelta(minutes=1)).timestamp()),
            received_at_utc=NOW + timedelta(minutes=1),
        )
        assert await scheduler(db, app, clock).tick(OWNER) is not None
        prompt = "\n".join(item.content for item in llm.calls[2])

        assert "[当前输入 | actor=platform; handle=none;" in prompt
        assert "kind=initiative]" in prompt
        assert "[历史原文 | actor=mumo; handle=M2;" in prompt
        assert "[历史原文 | actor=qichi; handle=Q3;" in prompt
        assert "mumo-one" not in prompt
        assert "qichi-one" not in prompt
        assert prompt.index("mumo-two") < prompt.index("qichi-two")
        assert "[当前输入 | actor=mumo;" not in prompt
        assert "当前时间: 2026-08-28T20:00:00+08:00 (Asia/Shanghai)" in prompt
        action_line = prompt.split("本轮可执行平台动作:", 1)[1].splitlines()[0]
        assert "reaction" not in action_line
        assert "[[qq:react:<key>]]" not in prompt
    finally:
        db.close()


@pytest.mark.asyncio
async def test_skip_persists_only_internal_trigger_without_qichi_text_or_outbox(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    app = make_app(db, FakeLLM(["普通回复", "[[qichi:skip]]"]), FakeNapCat())
    try:
        await app.handle_onebot(message(1, "在吗"), received_at_utc=NOW)
        before_qichi = db.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE actor='qichi' AND kind='text'"
        ).fetchone()[0]
        before_outbox = db.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        assert isinstance(await scheduler(db, app, clock).tick(OWNER), DialogueSkip)
        assert db.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE direction='internal' "
            "AND actor='platform' AND kind='initiative'"
        ).fetchone()[0] == 1
        assert db.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE actor='qichi' AND kind='text'"
        ).fetchone()[0] == before_qichi
        assert db.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == before_outbox
    finally:
        db.close()


@pytest.mark.parametrize(
    ("output", "send_error", "expected_status", "expected_notifications"),
    [
        ("主动正文", None, "sent", 0),
        ("[[qichi:skip]]", None, "skipped", 0),
        ("未知正文", OneBotTimeoutError("unknown"), "unknown", 0),
        ("失败正文", OneBotActionError("failed"), "failed", 0),
    ],
)
@pytest.mark.asyncio
async def test_only_sent_initiative_notifies_reliable_memory_activity(
    tmp_path, output, send_error, expected_status, expected_notifications
):
    db = Database(tmp_path / f"{expected_status}.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    worker = RecordingWorker()
    napcat = FakeNapCat()
    app = make_app(db, FakeLLM(["普通回复", output]), napcat, worker=worker)
    try:
        await app.handle_onebot(message(1, "记住上下文"), received_at_utc=NOW)
        worker.calls.clear()
        if send_error is not None:
            async def fail_send(*_args, **_kwargs):
                raise send_error
            napcat.send_private_msg = fail_send

        result = await scheduler(db, app, clock).tick(OWNER)
        actual_status = (
            "skipped" if isinstance(result, DialogueSkip) else getattr(result, "status", None)
        )
        assert actual_status == expected_status
        assert len(worker.calls) == expected_notifications
        topic_cursor = db.connection.execute(
            "SELECT presence_topic_cursor FROM conversation_cursors WHERE conversation_id = ?",
            (OWNER,),
        ).fetchone()[0]
        assert topic_cursor == (0 if expected_status == "sent" else None)
        if expected_notifications:
            assert worker.calls == [(OWNER,)]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_sent_topic_is_consumed_while_prior_qichi_text_remains_for_deduplication(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW)
    llm = FakeLLM(["普通回复", "第一次主动", "[[qichi:skip]]"])
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "只出现一次的用户话题"), received_at_utc=NOW)
        clock.value = NOW + timedelta(minutes=60)
        first = await scheduler(db, app, clock).tick(OWNER)
        assert first is not None and first.status == "sent"
        assert db.connection.execute(
            "SELECT presence_topic_cursor FROM conversation_cursors WHERE conversation_id = ?",
            (OWNER,),
        ).fetchone()[0] == 0
        trigger = db.connection.execute(
            "SELECT metadata_json FROM conversation_events WHERE direction = 'internal' "
            "AND kind = 'initiative' ORDER BY sequence LIMIT 1"
        ).fetchone()
        trigger_metadata = json.loads(trigger[0])
        assert trigger_metadata["presence_topic_cursor_before"] is None
        assert trigger_metadata["topic_watermark"] == 0

        clock.value = NOW + timedelta(minutes=120)
        assert isinstance(await scheduler(db, app, clock).tick(OWNER), DialogueSkip)
        next_prompt = "\n".join(item.content for item in llm.calls[2])
        assert "只出现一次的用户话题" not in next_prompt
        assert "第一次主动" in next_prompt
    finally:
        db.close()


@pytest.mark.asyncio
async def test_corrupt_presence_topic_cursor_fails_closed_without_active_send(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW)
    napcat = FakeNapCat()
    app = make_app(db, FakeLLM(["普通回复", "不得发送"]), napcat, clock=clock)
    try:
        await app.handle_onebot(message(1, "锚点"), received_at_utc=NOW)
        db.connection.execute(
            "UPDATE conversation_cursors SET presence_topic_cursor = 999 "
            "WHERE conversation_id = ?",
            (OWNER,),
        )
        clock.value = NOW + timedelta(minutes=60)
        assert await scheduler(db, app, clock).tick(OWNER) is None
        assert [segments[0]["data"]["text"] for _, segments in napcat.texts] == ["普通回复"]
        assert json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?",
                (f"initiative:{OWNER}",),
            ).fetchone()[0]
        )["status"] == "failed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_skip_does_not_consume_user_topic_evidence(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW)
    llm = FakeLLM(["普通回复", "[[qichi:skip]]", "第二次主动"])
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "跳过后还应可见的话题"), received_at_utc=NOW)
        clock.value = NOW + timedelta(minutes=60)
        assert isinstance(await scheduler(db, app, clock).tick(OWNER), DialogueSkip)
        assert db.connection.execute(
            "SELECT presence_topic_cursor FROM conversation_cursors WHERE conversation_id = ?",
            (OWNER,),
        ).fetchone()[0] is None

        clock.value = NOW + timedelta(minutes=120)
        second = await scheduler(db, app, clock).tick(OWNER)
        assert second is not None and second.status == "sent"
        next_prompt = "\n".join(item.content for item in llm.calls[2])
        assert "跳过后还应可见的话题" in next_prompt
    finally:
        db.close()


@pytest.mark.asyncio
async def test_user_message_committed_after_generation_check_cancels_before_dispatch(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    napcat = FakeNapCat()
    app = make_app(db, FakeLLM(["普通回复", "过时主动", "新回复"]), napcat, clock=clock)
    original_send = app.sender.send
    user_task = None
    send_calls = 0

    async def race_send(*args, **kwargs):
        nonlocal send_calls, user_task
        send_calls += 1
        if send_calls == 1:
            return await original_send(*args, **kwargs)
        user_task = asyncio.create_task(
            app.handle_onebot(message(2, "我回来了", clock.value.timestamp()), received_at_utc=clock.value)
        )
        for _ in range(100):
            version = db.connection.execute(
                "SELECT context_version FROM conversation_cursors WHERE conversation_id = ?", (OWNER,)
            ).fetchone()[0]
            if version == 2:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("user cursor did not advance before initiative dispatch")
        return await original_send(*args, **kwargs)

    app.sender.send = race_send
    try:
        await app.handle_onebot(message(1, "旧话题"), received_at_utc=NOW)
        assert await scheduler(db, app, clock).tick(OWNER) is None
        assert user_task is not None
        await user_task
        sent_texts = [segments[0]["data"]["text"] for _, segments in napcat.texts]
        assert sent_texts == ["普通回复", "新回复"]
        outbound = db.connection.execute(
            "SELECT status FROM conversation_events WHERE actor='qichi' ORDER BY sequence"
        ).fetchall()
        assert [row["status"] for row in outbound] == ["sent", "sent"]
        assert db.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_expired_real_claim_cannot_send_when_zombie_generation_resumes(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block(number):
        if number == 2:
            entered.set()
            await release.wait()

    old_clock = Clock(NOW + timedelta(minutes=60))
    napcat = FakeNapCat()
    old_db = Database(path)
    old_app = make_app(old_db, FakeLLM(["普通回复", "僵尸主动"], block), napcat, clock=old_clock)
    await old_app.handle_onebot(message(1, "锚点"), received_at_utc=NOW)
    old_task = asyncio.create_task(scheduler(old_db, old_app, old_clock).tick(OWNER))
    await entered.wait()

    recovery_db = Database(path)
    try:
        recovery_clock = Clock(NOW + timedelta(minutes=120))
        recovery_app = RecordingInitiativeApp()
        assert await scheduler(recovery_db, recovery_app, recovery_clock).tick(OWNER) is None
        release.set()
        assert await old_task is None
        assert [segments[0]["data"]["text"] for _, segments in napcat.texts] == ["普通回复"]
        assert recovery_app.calls == []
    finally:
        recovery_db.close()
        old_db.close()


@pytest.mark.asyncio
async def test_unclassified_transport_exception_converges_from_dispatched_evidence(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))
    napcat = FakeNapCat()

    async def fail_after_dispatch(*_args, **_kwargs):
        raise RuntimeError("unclassified transport failure")

    app = make_app(db, FakeLLM(["普通回复", "发送中断"]), napcat, clock=clock)
    try:
        await app.handle_onebot(message(1, "锚点"), received_at_utc=NOW)
        napcat.send_private_msg = fail_after_dispatch
        assert await scheduler(db, app, clock).tick(OWNER) is None
        state = json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
            ).fetchone()[0]
        )
        assert state["status"] == "unknown"
        assert db.connection.execute(
            "SELECT COUNT(*) FROM outbox WHERE status='dispatched'"
        ).fetchone()[0] == 1
        clock.value += timedelta(hours=24)
        assert await scheduler(db, app, clock).tick(OWNER) is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_generation_failure_without_outbox_evidence_remains_failed(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    clock = Clock(NOW + timedelta(minutes=60))

    class FailingApp:
        async def generate_initiative(self, *_args):
            raise RuntimeError("generation failed before persistence")

    try:
        seed_cursor(db)
        assert await scheduler(db, FailingApp(), clock).tick(OWNER) is None
        state = json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?", (f"initiative:{OWNER}",)
            ).fetchone()[0]
        )
        assert state["status"] == "failed"
        assert db.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_initiative_outbound_uses_send_intent_time_not_generation_start(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    attempt = NOW + timedelta(minutes=60)
    clock = Clock(attempt)

    async def advance(number):
        if number == 2:
            clock.value = attempt + timedelta(seconds=30)

    llm = FakeLLM(["普通回复", "主动正文"], advance)
    app = make_app(db, llm, FakeNapCat(), clock=clock)
    try:
        await app.handle_onebot(message(1, "锚点"), received_at_utc=NOW)
        delivered = await scheduler(db, app, clock).tick(OWNER)
        assert delivered.status == "sent"
        assert delivered.occurred_at_utc == attempt + timedelta(seconds=30)
        prompt = "\n".join(item.content for item in llm.calls[1])
        assert "当前时间: 2026-08-28T19:00:00+08:00" in prompt
    finally:
        db.close()


@pytest.mark.asyncio
async def test_generation_clock_rollback_cannot_backdate_outbound_or_retrigger(tmp_path):
    db = Database(tmp_path / "qichi.sqlite3")
    attempt = NOW + timedelta(minutes=60)
    clock = Clock(attempt)

    async def roll_back(number):
        if number == 2:
            clock.value = NOW - timedelta(hours=1)

    app = make_app(
        db,
        FakeLLM(["普通回复", "主动正文"], roll_back),
        FakeNapCat(),
        clock=clock,
    )
    try:
        await app.handle_onebot(message(1, "锚点"), received_at_utc=NOW)
        delivered = await scheduler(db, app, clock).tick(OWNER)
        assert delivered.status == "sent"
        assert delivered.occurred_at_utc == attempt

        state = json.loads(
            db.connection.execute(
                "SELECT value_json FROM runtime_meta WHERE key = ?",
                (f"initiative:{OWNER}",),
            ).fetchone()[0]
        )
        assert datetime.fromisoformat(state["completed_at"]) >= attempt
        assert datetime.fromisoformat(state["next_due_at"]) > datetime.fromisoformat(
            state["completed_at"]
        )

        clock.value = attempt
        assert await scheduler(db, app, clock).tick(OWNER) is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_same_second_cross_connection_inbound_event_and_cursor_share_one_transaction(tmp_path):
    path = tmp_path / "qichi.sqlite3"
    first_db = Database(path)
    second_db = Database(path)
    first_trace = []
    second_trace = []
    first_db.connection.set_trace_callback(first_trace.append)
    second_db.connection.set_trace_callback(second_trace.append)
    napcat = FakeNapCat()
    first_app = make_app(first_db, FakeLLM(["first"]), napcat)
    second_app = make_app(second_db, FakeLLM(["second"]), napcat)
    try:
        await asyncio.gather(
            first_app.handle_onebot(message(1, "同秒一"), received_at_utc=NOW),
            second_app.handle_onebot(message(2, "同秒二"), received_at_utc=NOW),
        )
        cursor = first_db.connection.execute(
            "SELECT context_version, last_user_activity_utc FROM conversation_cursors WHERE conversation_id = ?",
            (OWNER,),
        ).fetchone()
        assert cursor[:] == (2, NOW.isoformat())
        assert first_db.connection.execute(
            "SELECT COUNT(*) FROM conversation_events WHERE direction='inbound'"
        ).fetchone()[0] == 2

        for trace in (first_trace, second_trace):
            event_index = next(
                index for index, statement in enumerate(trace)
                if statement.startswith("INSERT INTO conversation_events")
            )
            cursor_index = next(
                index for index, statement in enumerate(trace[event_index:], event_index)
                if statement.startswith("INSERT INTO conversation_cursors")
            )
            assert not any(
                statement == "COMMIT" for statement in trace[event_index:cursor_index]
            )
    finally:
        first_db.connection.set_trace_callback(None)
        second_db.connection.set_trace_callback(None)
        first_db.close()
        second_db.close()
