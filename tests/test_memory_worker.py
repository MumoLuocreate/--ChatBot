from datetime import datetime, timedelta, timezone
import json
import pytest
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord, MemoryReview
from qichi.domain.memory_details import MemoryDetailDraft
from qichi.memory.detail_pass import MemoryDetailPass
from qichi.memory.extractor import ExtractionFailure, MemoryExtractor, MemoryExtractionResult, MemoryOutcome
from qichi.memory.worker import MemoryWorker
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository

NOW=datetime(2026,8,28,16,0,tzinfo=timezone.utc)
def event(i, at=NOW, direction='inbound', actor='mumo', status='received', source=None, kind='text', text=None):
    text=str(i) if text is None else text; meta={} if source is None else {'generation_metadata':{'source':source}}
    return ConversationEvent(str(i),None,'pm-'+str(i),'conversation-a',i,direction,actor,kind,text,(MessageSegment('text',{'text':text}),),None,None,at,at,status,meta)
class LLM:
    def __init__(self, fail=False): self.calls=[]; self.fail=fail
    async def generate(self, es):
        self.calls.append(tuple(e.event_id for e in es))
        if self.fail: raise RuntimeError('timeout')
        e=es[0]
        return json.dumps({'outcome':{'kind':'memory_found','reason_code':'explicit_user_preference'},'candidates':[{'type':'preference','normalized_fact':e.text,'modality':'explicit_statement','certainty':'explicit','importance':2,'temporal_scope':'ongoing','assessment_reason_code':'explicit_user_statement','valid_from_utc':e.occurred_at_utc.isoformat(),'valid_until_utc':None,'supersedes_id':None,'evidence':[{'event_id':e.event_id,'actor':'mumo','exact_quote':e.text,'role':'source'}]}],'reviews':[]})
@pytest.fixture
def stack(tmp_path):
    db=Database(tmp_path/'qichi.sqlite3')
    try: yield db,EventRepository(db),MemoryRepository(db)
    finally: db.close()
def worker(db,mem,llm=None): return MemoryWorker(MemoryExtractor(llm or LLM()),mem,database=db,conversation_id='conversation-a')
def test_safe_failure_details_keep_the_reason_code_but_never_its_message():
    failure = ExtractionFailure(
        "repository_error",
        "IntegrityError: UNIQUE constraint failed: memory_fragments.fragment_key",
        ("event-1",),
    )

    details = MemoryWorker._safe_failure_details(failure)

    assert details == {"reason_code": "IntegrityError"}, "只留异常类名，不留消息内容"
    assert "constraint" not in json.dumps(details)


def test_safe_failure_details_refuse_paths_and_free_text():
    failure = ExtractionFailure("worker_error", "/home/user/secret/report.txt missing", ("event-1",))

    assert MemoryWorker._safe_failure_details(failure) == {}


def test_safe_failure_details_pass_through_bounded_parse_telemetry():
    failure = ExtractionFailure(
        "parse_error",
        "fragment_type is invalid",
        ("event-1",),
        {"parse_error_code": "response_schema", "schema_field": "fragment_type", "finish_reason": "length"},
    )

    details = MemoryWorker._safe_failure_details(failure)

    assert details["parse_error_code"] == "response_schema"
    assert details["schema_field"] == "fragment_type"
    assert details["finish_reason"] == "length"
    assert "reason_code" not in details, "字段名由 schema_field 承载，reason_code 只留给异常类名"


@pytest.mark.asyncio
async def test_30_minute_hard_gate(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); assert await w.run_due(NOW+timedelta(minutes=29,seconds=59))==(); assert (await w.run_due(NOW+timedelta(minutes=30)))[0].written_memory_ids
@pytest.mark.asyncio
@pytest.mark.parametrize('count',[1,20,100])
async def test_event_count_never_flushes_early(stack,count):
    db,evs,mem=stack
    for i in range(count): evs.insert(event(i))
    w=worker(db,mem); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); assert await w.run_due(NOW+timedelta(minutes=29))==()
def test_single_job_revision_anchor(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); evs.insert(event(1,NOW+timedelta(minutes=2))); w.notify_reliable_activity('conversation-a'); r=db.connection.execute('select count(*),revision,end_sequence,anchor_event_id from memory_session_jobs').fetchone(); assert tuple(r)==(1,2,1,'1')
def test_ineligible_activity_filtered(stack):
    db,evs,mem=stack
    for i,k in enumerate((dict(direction='internal',actor='platform'),dict(direction='outbound',actor='qichi',status='pending'),dict(direction='outbound',actor='qichi',status='unknown'),dict(direction='outbound',actor='qichi',status='failed'),dict(direction='outbound',actor='qichi',source='initiative'))): evs.insert(event(i,**k))
    assert not worker(db,mem).notify_reliable_activity('conversation-a')
@pytest.mark.asyncio
async def test_gate_blocks_paid_call(stack):
    db,evs,mem=stack; evs.insert(event(0)); llm=LLM(); w=worker(db,mem,llm); w.notify_reliable_activity('conversation-a'); assert await w.run_due(NOW+timedelta(minutes=31))==() and not llm.calls
@pytest.mark.asyncio
async def test_completion_deletes_job(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); assert len(await w.run_due(NOW+timedelta(minutes=30)))==1; assert db.connection.execute('select count(*) from memory_session_jobs').fetchone()[0]==0
@pytest.mark.asyncio
async def test_failure_keeps_watermark(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem,LLM(True)); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); assert (await w.run_due(NOW+timedelta(minutes=30)))[0].failed; assert db.connection.execute("select count(*) from runtime_meta where key like '%processed_sequence'").fetchone()[0]==0
def test_restart_preserves_deadline(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); before=db.connection.execute('select deadline_utc from memory_session_jobs').fetchone()[0]; assert worker(db,mem).recover_pending('conversation-a'); assert db.connection.execute('select deadline_utc from memory_session_jobs').fetchone()[0]==before
@pytest.mark.asyncio
async def test_clock_rollback_does_not_run(stack):
    db,evs,mem=stack; evs.insert(event(0)); llm=LLM(); w=worker(db,mem,llm); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); assert await w.run_due(NOW-timedelta(hours=1))==()
def test_dialogue_outbound_reliable(stack):
    db,evs,mem=stack; evs.insert(event(0,direction='outbound',actor='qichi',status='sent',source='dialogue')); assert worker(db,mem).notify_reliable_activity('conversation-a')
def test_initiative_outbound_filtered(stack):
    db,evs,mem=stack; evs.insert(event(0,direction='outbound',actor='qichi',source='initiative')); assert not worker(db,mem).notify_reliable_activity('conversation-a')
def test_job_claim_columns_empty(stack):
    db,evs,mem=stack; evs.insert(event(0)); worker(db,mem).notify_reliable_activity('conversation-a'); r=db.connection.execute('select context_version,claim_token,claim_owner,claim_lease_until_utc from memory_session_jobs').fetchone(); assert tuple(r)==(0,None,None,None)
def test_new_activity_increments_revision(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); evs.insert(event(1)); w.notify_reliable_activity('conversation-a'); assert db.connection.execute('select revision from memory_session_jobs').fetchone()[0]==2


@pytest.mark.asyncio
async def test_later_session_does_not_reset_retry_for_frozen_earlier_fragment(stack):
    db, evs, mem = stack
    evs.insert(event(0, NOW))
    db.connection.execute(
        "INSERT INTO conversation_cursors(conversation_id,context_version,last_user_activity_utc) "
        "VALUES (?,?,?)",
        ("conversation-a", 1, NOW.isoformat()),
    )
    failing = worker(db, mem, LLM(True))
    failing.notify_reliable_activity("conversation-a")
    failing.open_semantic_gate()
    await failing.run_due(NOW + timedelta(minutes=30))
    before = db.connection.execute(
        "SELECT revision,status,attempt_count,start_sequence,end_sequence "
        "FROM memory_session_jobs"
    ).fetchone()
    assert tuple(before) == (1, "retry", 1, 0, 0)

    evs.insert(event(1, NOW + timedelta(hours=1)))
    db.connection.execute(
        "UPDATE conversation_cursors SET context_version=2,last_user_activity_utc=? "
        "WHERE conversation_id=?",
        ((NOW + timedelta(hours=1)).isoformat(), "conversation-a"),
    )
    assert not failing.notify_reliable_activity("conversation-a")
    after = db.connection.execute(
        "SELECT revision,status,attempt_count,start_sequence,end_sequence "
        "FROM memory_session_jobs"
    ).fetchone()
    assert tuple(after) == tuple(before)


@pytest.mark.asyncio
async def test_disallowed_reviews_do_not_roll_back_valid_candidates(stack):
    db, evs, mem = stack
    base = evs.insert(event(0, NOW, text="我喜欢雨声"))
    mem.create(
        MemoryRecord(
            "active-memory",
            "preference",
            "用户喜欢雨声",
            "explicit_statement",
            "active",
            NOW,
            None,
            None,
            NOW,
            (MemoryEvidence("active-memory", base.event_id, "mumo", base.text, NOW),),
            "explicit",
            2,
            "ongoing",
            "explicit_user_statement",
            NOW,
        )
    )
    qichi_event = evs.insert(
        event(
            1,
            NOW + timedelta(seconds=1),
            direction="outbound",
            actor="qichi",
            status="sent",
            source="dialogue",
            text="我也记得这件事",
        )
    )
    mem.create(
        MemoryRecord(
            "self-expression-memory",
            "self_expression",
            "角色表示自己记得这件事",
            "explicit_statement",
            "candidate",
            qichi_event.occurred_at_utc,
            None,
            None,
            qichi_event.occurred_at_utc,
            (
                MemoryEvidence(
                    "self-expression-memory",
                    qichi_event.event_id,
                    "qichi",
                    qichi_event.text,
                    qichi_event.occurred_at_utc,
                ),
            ),
            "explicit",
            1,
            "ongoing",
            "explicit_user_statement",
            qichi_event.occurred_at_utc,
        )
    )
    current = evs.insert(event(2, NOW + timedelta(seconds=2), text="我还喜欢海浪"))
    candidate_result = await MemoryExtractor(LLM()).extract((current,))
    assert candidate_result.ok
    active_support = MemoryReview(
        "active-memory",
        "support",
        "explicit",
        2,
        "ongoing",
        "explicit_user_statement",
        (
            MemoryEvidence(
                "active-memory", current.event_id, "mumo", current.text,
                current.occurred_at_utc, "source",
            ),
        ),
    )
    self_expression_support = MemoryReview(
        "self-expression-memory",
        "support",
        "explicit",
        1,
        "ongoing",
        "explicit_user_statement",
        (
            MemoryEvidence(
                "self-expression-memory", qichi_event.event_id, "qichi", qichi_event.text,
                qichi_event.occurred_at_utc, "source",
            ),
        ),
    )

    class MixedExtractor(MemoryExtractor):
        async def extract(self, events):
            return MemoryExtractionResult(
                candidate_result.candidates,
                None,
                (active_support, self_expression_support),
                {},
                MemoryOutcome("memory_found", "review_proposed"),
            )

    memory_worker = MemoryWorker(
        MixedExtractor(LLM()),
        mem,
        database=db,
        conversation_id="conversation-a",
    )
    memory_worker.notify_reliable_activity("conversation-a")
    memory_worker.open_semantic_gate()
    runs = await memory_worker.run_due(NOW + timedelta(minutes=31))
    assert runs and not runs[0].failed and runs[0].written_memory_ids
    assert mem.count() == 3
    assert db.connection.execute("SELECT COUNT(*) FROM memory_session_jobs").fetchone()[0] == 0
    details = json.loads(
        db.connection.execute(
            "SELECT details_json FROM memory_job_events WHERE action='completed' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    )
    assert details == {
        "candidate_count": "1",
        "review_count": "2",
        "dropped_review_count": "2",
        "review_error_codes": "review_target",
        "outcome_kind": "memory_found",
        "outcome_reason_code": "review_proposed",
    }


@pytest.mark.asyncio
async def test_empty_success_is_marked_without_advancing_as_failure(stack):
    db, evs, mem = stack
    evs.insert(event(0, text="只是闲聊"))

    class EmptyExtractor(MemoryExtractor):
        async def extract(self, events):
            return MemoryExtractionResult(
                (), None, (), {},
                MemoryOutcome("no_persistent_memory", "temporary_scene_or_roleplay"),
            )

    w = MemoryWorker(
        EmptyExtractor(LLM()), mem, database=db, conversation_id="conversation-a"
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    runs = await w.run_due(NOW + timedelta(minutes=30))
    assert runs and not runs[0].failed
    details = json.loads(
        db.connection.execute(
            "SELECT details_json FROM memory_job_events WHERE action='completed' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    )
    assert details == {
        "candidate_count": "0",
        "review_count": "0",
        "empty_result": "1",
        "outcome_kind": "no_persistent_memory",
        "outcome_reason_code": "temporary_scene_or_roleplay",
    }


@pytest.mark.asyncio
async def test_worker_rejects_success_without_public_outcome(stack):
    db, evs, mem = stack
    evs.insert(event(0))

    class MissingOutcomeExtractor(MemoryExtractor):
        async def extract(self, events):
            valid = await MemoryExtractor(LLM()).extract(events)
            return MemoryExtractionResult(valid.candidates, None, valid.reviews)

    w = MemoryWorker(
        MissingOutcomeExtractor(LLM()), mem, database=db, conversation_id="conversation-a"
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    runs = await w.run_due(NOW + timedelta(minutes=30))
    assert runs and runs[0].failed
    assert runs[0].failure is not None
    assert runs[0].failure.kind == "parse_error"
    assert mem.count() == 0


@pytest.mark.asyncio
async def test_nonempty_success_cannot_retain_stale_empty_marker(stack):
    db, evs, mem = stack
    evs.insert(event(0))

    class StaleMarkerExtractor(MemoryExtractor):
        async def extract(self, events):
            valid = await MemoryExtractor(LLM()).extract(events)
            return MemoryExtractionResult(
                valid.candidates,
                None,
                valid.reviews,
                {"empty_result": "1"},
                valid.outcome,
            )

    w = MemoryWorker(
        StaleMarkerExtractor(LLM()), mem, database=db, conversation_id="conversation-a"
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    await w.run_due(NOW + timedelta(minutes=30))
    details = json.loads(
        db.connection.execute(
            "SELECT details_json FROM memory_job_events WHERE action='completed' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
    )
    assert details == {
        "candidate_count": "1",
        "review_count": "0",
        "outcome_kind": "memory_found",
        "outcome_reason_code": "explicit_user_preference",
    }
@pytest.mark.asyncio
async def test_success_idempotent_after_restart(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); await w.run_due(NOW+timedelta(minutes=30)); assert mem.count()==1 and not worker(db,mem).recover_pending('conversation-a')

@pytest.mark.asyncio
async def test_received_gap_starts_remainder_without_early_call(stack):
    db,evs,mem=stack; evs.insert(event(0, NOW)); evs.insert(event(1, NOW+timedelta(minutes=31))); llm=LLM(); w=worker(db,mem,llm); w.notify_reliable_activity('conversation-a'); row=db.connection.execute('select end_sequence,anchor_sequence,deadline_utc from memory_session_jobs').fetchone(); assert tuple(row) == (0,0,(NOW+timedelta(minutes=30)).isoformat()); w.open_semantic_gate()
    await w.run_due(NOW+timedelta(minutes=30)); assert llm.calls == [('0',)]; assert db.connection.execute('select start_sequence,end_sequence from memory_session_jobs').fetchone()[:]==(1,1)

@pytest.mark.asyncio
async def test_oversized_single_event_fails_closed_and_preserves_event(stack):
    db,evs,mem=stack; evs.insert(event(0,text='x'*20)); w=worker(db,mem,LLM()); w.max_fragment_tokens=1; w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); await w.run_due(NOW+timedelta(minutes=30)); row=db.connection.execute('select status,failure_category from memory_session_jobs').fetchone(); assert tuple(row)==('failed','oversized_event'); assert db.connection.execute('select count(*) from conversation_events').fetchone()[0]==1

def test_duplicate_notify_does_not_increment_revision(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); assert w.notify_reliable_activity('conversation-a'); assert not w.notify_reliable_activity('conversation-a'); assert db.connection.execute('select revision from memory_session_jobs').fetchone()[0]==1

@pytest.mark.asyncio
async def test_unnotified_new_event_invalidates_claim(stack):
    db,evs,mem=stack; evs.insert(event(0)); llm=LLM(); w=worker(db,mem,llm); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate()
    with db.transaction() as c:
        r=c.execute('select job_id from memory_session_jobs').fetchone(); c.execute("update memory_session_jobs set status='claimed',claim_token='old',claim_owner='old',claim_lease_until_utc=? where job_id=?",((NOW+timedelta(minutes=5)).isoformat(),r[0]))
    evs.insert(event(1)); assert await w.run_due(NOW+timedelta(minutes=31)) == ()

@pytest.mark.asyncio
async def test_two_connections_claim_once(stack,tmp_path):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); other=Database(tmp_path/'qichi.sqlite3'); llm1=LLM(); llm2=LLM(); a=worker(db,MemoryRepository(db),llm1); b=worker(other,MemoryRepository(other),llm2); a.open_semantic_gate(); b.open_semantic_gate(); await __import__('asyncio').gather(a.run_due(NOW+timedelta(minutes=30)),b.run_due(NOW+timedelta(minutes=30))); assert len(llm1.calls)+len(llm2.calls)==1; other.close()

@pytest.mark.asyncio
async def test_expired_lease_recovery_replaces_token(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); db.connection.execute("update memory_session_jobs set status='claimed',claim_token='dead',claim_owner='dead',claim_lease_until_utc=?",((NOW-timedelta(seconds=1)).isoformat(),)); w.open_semantic_gate(); await w.run_due(NOW); row=db.connection.execute("select status,claim_token,claim_owner from memory_session_jobs").fetchone(); assert tuple(row)==('pending',None,None)

@pytest.mark.asyncio
async def test_network_backoff_schedule_and_final_failure(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem,LLM(True)); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate()
    # Initial due run, then 30s, 120s, and 600s retry windows.
    for minute in (30, 31, 34, 45): await w.run_due(NOW+timedelta(minutes=minute))
    assert tuple(db.connection.execute('select status,attempt_count from memory_session_jobs').fetchone())==('failed',4)


@pytest.mark.asyncio
async def test_provider_failure_details_are_persisted_without_reason_text(stack):
    db, evs, mem = stack
    evs.insert(event(0))

    class CategorizedExtractor(MemoryExtractor):
        async def extract(self, events):
            return MemoryExtractionResult(
                (),
                ExtractionFailure(
                    "timeout",
                    "LLMTimeoutError: provider generation failed secret-key",
                    tuple(item.event_id for item in events),
                    {"provider_error": "timeout", "secret": "must-not-persist"},
                ),
                (),
            )

    w = MemoryWorker(CategorizedExtractor(LLM()), mem, database=db, conversation_id="conversation-a")
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    await w.run_due(NOW + timedelta(minutes=30))
    detail = db.connection.execute(
        "select details_json from memory_job_events order by rowid desc limit 1"
    ).fetchone()[0]
    assert json.loads(detail) == {"provider_error": "timeout", "reason_code": "LLMTimeoutError"}
    assert "secret" not in detail
    assert "provider generation failed" not in detail, "只留异常类名，不留消息"


@pytest.mark.asyncio
async def test_parse_failure_code_is_persisted_without_model_text(stack):
    db, evs, mem = stack
    evs.insert(event(0))

    class ParseFailureExtractor(MemoryExtractor):
        async def extract(self, events):
            return MemoryExtractionResult(
                (),
                ExtractionFailure(
                    "parse_error",
                    "sensitive model output",
                    tuple(item.event_id for item in events),
                    {
                        "parse_error_code": "all_items_invalid",
                        "dropped_candidate_count": "1",
                        "candidate_error_codes": "candidate_evidence",
                        "secret": "must-not-persist",
                    },
                ),
                (),
            )

    w = MemoryWorker(
        ParseFailureExtractor(LLM()), mem, database=db, conversation_id="conversation-a"
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    await w.run_due(NOW + timedelta(minutes=30))
    detail = db.connection.execute(
        "select details_json from memory_job_events order by rowid desc limit 1"
    ).fetchone()[0]
    assert json.loads(detail) == {
        "candidate_error_codes": "candidate_evidence",
        "dropped_candidate_count": "1",
        "parse_error_code": "all_items_invalid",
    }
    assert "sensitive" not in detail and "secret" not in detail


@pytest.mark.asyncio
async def test_partial_parse_diagnostics_are_recorded_on_completed_job(stack):
    db, evs, mem = stack
    evs.insert(event(0))

    class PartialExtractor(MemoryExtractor):
        async def extract(self, events):
            valid = await MemoryExtractor(LLM()).extract(events)
            return MemoryExtractionResult(
                valid.candidates,
                None,
                valid.reviews,
                {
                    "dropped_candidate_count": "1",
                    "candidate_error_codes": "candidate_evidence",
                    "secret": "must-not-persist",
                },
                valid.outcome,
            )

    w = MemoryWorker(
        PartialExtractor(LLM()), mem, database=db, conversation_id="conversation-a"
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    runs = await w.run_due(NOW + timedelta(minutes=30))
    assert runs and runs[0].written_memory_ids
    detail = db.connection.execute(
        "select details_json from memory_job_events where action='completed' order by rowid desc limit 1"
    ).fetchone()[0]
    assert json.loads(detail) == {
        "candidate_count": "1",
        "review_count": "0",
        "candidate_error_codes": "candidate_evidence",
        "dropped_candidate_count": "1",
        "outcome_kind": "memory_found",
        "outcome_reason_code": "explicit_user_preference",
    }

def _parse_failing_worker(db, mem):
    """一个内容层失败（parse_error）的 worker：模拟上游崩坏时返回残缺 JSON。"""

    class ParseFailureExtractor(MemoryExtractor):
        async def extract(self, events):
            return MemoryExtractionResult(
                (),
                ExtractionFailure(
                    "parse_error",
                    "degraded provider returned partial json",
                    tuple(item.event_id for item in events),
                ),
                (),
            )

    return MemoryWorker(
        ParseFailureExtractor(LLM()), mem, database=db, conversation_id="conversation-a"
    )


def _job_row(db):
    return db.connection.execute("SELECT status,failure_category FROM memory_session_jobs").fetchone()


def _quarantine_events(db):
    return db.connection.execute(
        "SELECT COUNT(*) FROM memory_job_events WHERE action='quarantined'"
    ).fetchone()[0]


@pytest.mark.asyncio
async def test_a_content_parse_failure_is_not_quarantined_until_the_budget_is_spent(stack):
    """2026-09-15：上游崩坏时返回的残缺 JSON 就是 parse_error，一次定生死会白丢整段。

    真机事故：DeepSeek 官方崩溃那一晚，seq 6566–6600（35 条，正是用户纠正她的那一段）
    在两次尝试后被 quarantine，那一段再没进记忆层。
    """

    db, evs, mem = stack
    evs.insert(event(0))
    w = _parse_failing_worker(db, mem)
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()

    await w.run_due(NOW + timedelta(minutes=30))
    await w.run_due(NOW + timedelta(minutes=30, seconds=2))

    assert tuple(_job_row(db)) == ("failed", "parse_error")
    assert _quarantine_events(db) == 0, "冷却重试额度没用完，不许隔离"


@pytest.mark.asyncio
async def test_a_content_failure_is_quarantined_once_the_budget_runs_out(stack):
    """额度用完仍然隔离：这不是「永不放弃」，只是不再一次定生死。"""

    db, evs, mem = stack
    evs.insert(event(0))
    w = _parse_failing_worker(db, mem)
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()

    for cycle in range(MemoryWorker.CONTENT_FAILURE_REOPEN_ROUNDS + 1):
        base = NOW + timedelta(minutes=30 + cycle * 60)
        await w.run_due(base)
        await w.run_due(base + timedelta(seconds=2))
        if cycle < MemoryWorker.CONTENT_FAILURE_REOPEN_ROUNDS:
            assert _quarantine_events(db) == 0, "第 %d 次终态失败还不该隔离" % (cycle + 1)
            assert w.recover_pending("conversation-a"), "冷却之后必须能重开同一段"

    assert _quarantine_events(db) == 1


@pytest.mark.asyncio
async def test_transient_failures_do_not_spend_the_content_retry_budget(stack):
    """不误判：一次网络抖动不该把内容层失败的冷却额度吃掉。"""

    db, evs, mem = stack
    evs.insert(event(0))
    w = _parse_failing_worker(db, mem)
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    with db.transaction() as connection:
        row = connection.execute("SELECT * FROM memory_session_jobs").fetchone()
        for _ in range(2):
            MemoryWorker._append_job_event_tx(
                w, connection, row, "failed", now=NOW, failure_category="timeout"
            )

    await w.run_due(NOW + timedelta(minutes=30))
    await w.run_due(NOW + timedelta(minutes=30, seconds=2))

    assert _quarantine_events(db) == 0

class _PerEventDetailPass(MemoryDetailPass):
    """一条事件给一条明细：真机 35 条事件就是这么撞上 32 条上限的。"""

    def __init__(self):
        pass

    async def generate(self, events):
        return tuple(
            MemoryDetailDraft(
                ordinal=index, detail_kind="message", actor=item.actor,
                reality_scope="conversation", normalized_detail="n%d" % index,
                exact_quote=item.text, source_event_id=item.event_id,
                certainty="explicit", temporal_scope="historical", status="active",
                privacy_class="ordinary", recall_policy="daily_safe",
                evidence=((item.event_id, "source"),),
            )
            for index, item in enumerate(events)
        )


class _HeadOnlyDetailPass(MemoryDetailPass):
    """窗口一大就只写前半段：真机 172 条的片段就是这样只盖到第 57 条的。"""

    def __init__(self, cap: int = 12):
        self.cap = cap

    async def generate(self, events):
        keep = events if len(events) <= self.cap else events[: len(events) // 2]
        return tuple(
            MemoryDetailDraft(
                ordinal=index, detail_kind="message", actor=item.actor,
                reality_scope="conversation", normalized_detail="n%d" % index,
                exact_quote=item.text, source_event_id=item.event_id,
                certainty="explicit", temporal_scope="historical", status="active",
                privacy_class="ordinary", recall_policy="daily_safe",
                evidence=((item.event_id, "source"),),
            )
            for index, item in enumerate(keep)
        )


class _TrailingGapDetailPass(_HeadOnlyDetailPass):
    """只差末尾两条没有明细——在允许的余量之内，不该触发切分。"""

    def __init__(self):
        super().__init__(cap=10_000)

    async def generate(self, events):
        return await super().generate(events[:-2] if len(events) > 2 else events)


@pytest.mark.asyncio
async def test_a_timeline_that_stops_early_splits_instead_of_hiding_the_tail(stack):
    """2026-09-17 真机：172 条的片段只盖到第 57 条，凌晨那句话她永远翻不到。

    修之前这道保险丝烧不断——提示词叫模型自己「最多 32 条、取最重要的 32 条」，
    所以 len(drafts) > 32 永远不成立，长片段的尾巴就是静默消失的。
    """

    db, evs, mem = stack
    for index in range(20):
        evs.insert(event(index, NOW + timedelta(minutes=index)))
    w = MemoryWorker(
        MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
        detail_pass=_HeadOnlyDetailPass(cap=12),
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()

    runs = await w.run_due(NOW + timedelta(minutes=50))

    assert runs and not runs[0].failed
    assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 2
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 20, "一条明细都不许丢"
    covered = db.connection.execute("SELECT MIN(start_sequence),MAX(end_sequence) FROM memory_fragments").fetchone()
    assert tuple(covered) == (0, 19)
    detail = json.loads(db.connection.execute(
        "SELECT details_json FROM memory_job_events WHERE action='completed'"
    ).fetchone()[0])
    assert detail["split_fragment_count"] == "2"


@pytest.mark.asyncio
async def test_a_timeline_that_stops_two_events_early_is_left_alone(stack):
    """不误判：末尾两条没有明细属于允许的余量，不该为它多切一刀。"""

    db, evs, mem = stack
    for index in range(10):
        evs.insert(event(index, NOW + timedelta(minutes=index)))
    w = MemoryWorker(
        MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
        detail_pass=_TrailingGapDetailPass(),
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()

    runs = await w.run_due(NOW + timedelta(minutes=40))

    assert runs and not runs[0].failed
    assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 1
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 8
    detail = json.loads(db.connection.execute(
        "SELECT details_json FROM memory_job_events WHERE action='completed'"
    ).fetchone()[0])
    assert "split_fragment_count" not in detail


@pytest.mark.asyncio
async def test_details_beyond_the_cap_split_instead_of_losing_the_window(stack):
    """2026-09-15：35 条事件 → 35 条明细 > 每片段上限 32 → 旧行为整段落不了库。

    真机事故：seq 6566–6600（35 条）就是这样，修好之前那段记忆一条都存不进去。
    新行为：按事件对半切、各自落成片段，**原文与明细都不丢**。
    """

    db, evs, mem = stack
    for index in range(35):
        evs.insert(event(index, NOW + timedelta(minutes=index)))
    w = MemoryWorker(
        MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
        detail_pass=_PerEventDetailPass(),
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()

    # 闸门按锚点（最后一条事件）+30 分钟算，35 条事件跨了 35 分钟。
    runs = await w.run_due(NOW + timedelta(minutes=70))

    assert runs and not runs[0].failed
    assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 2
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 35, "一条明细都不许裁"
    detail = json.loads(db.connection.execute(
        "SELECT details_json FROM memory_job_events WHERE action='completed'"
    ).fetchone()[0])
    assert detail["split_fragment_count"] == "2"
    covered = db.connection.execute("SELECT MIN(start_sequence),MAX(end_sequence) FROM memory_fragments").fetchone()
    assert tuple(covered) == (0, 34)


@pytest.mark.asyncio
async def test_a_window_within_the_cap_still_lands_as_one_fragment(stack):
    """不误判：没超上限时，片段边界与切分前的行为逐字一致。"""

    db, evs, mem = stack
    for index in range(10):
        evs.insert(event(index, NOW + timedelta(minutes=index)))
    w = MemoryWorker(
        MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
        detail_pass=_PerEventDetailPass(),
    )
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    await w.run_due(NOW + timedelta(minutes=40))

    assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 1
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 10
    detail = json.loads(db.connection.execute(
        "SELECT details_json FROM memory_job_events WHERE action='completed'"
    ).fetchone()[0])
    assert "split_fragment_count" not in detail

@pytest.mark.asyncio
async def test_parse_error_short_retry_only_once(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem,LLM(True)); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate(); await w.run_due(NOW+timedelta(minutes=30)); assert db.connection.execute('select attempt_count from memory_session_jobs').fetchone()[0]==1

def test_large_session_has_stable_fragment_key(stack):
    db,evs,mem=stack
    for i in range(100): evs.insert(event(i))
    w=worker(db,mem); w.notify_reliable_activity('conversation-a'); key=db.connection.execute('select fragment_key from memory_session_jobs').fetchone()[0]; assert len(key)==64; assert not w.notify_reliable_activity('conversation-a')

@pytest.mark.asyncio
async def test_repository_failure_rolls_back_and_keeps_job(stack):
    db,evs,mem=stack; evs.insert(event(0)); w=worker(db,mem); w.notify_reliable_activity('conversation-a'); w.open_semantic_gate()
    original=mem.apply_consolidation_in_transaction
    def fail(*args,**kwargs): raise RuntimeError('repository failure')
    mem.apply_consolidation_in_transaction=fail
    runs = await w.run_due(NOW+timedelta(minutes=30))
    assert runs and runs[0].failed
    assert mem.count()==0 and db.connection.execute('select count(*) from memory_session_jobs').fetchone()[0]==1
    mem.apply_consolidation_in_transaction=original


@pytest.mark.asyncio
async def test_terminal_transient_failure_reopens_when_later_activity_arrives(stack):
    """A terminal provider failure must not permanently pin the session cursor."""
    db, evs, mem = stack
    evs.insert(event(0, NOW))
    failing = worker(db, mem, LLM(True))
    failing.notify_reliable_activity("conversation-a")
    failing.open_semantic_gate()
    for minute in (30, 31, 34, 45):
        await failing.run_due(NOW + timedelta(minutes=minute))
    failed = db.connection.execute(
        "SELECT job_id,revision,status,attempt_count,failure_category "
        "FROM memory_session_jobs"
    ).fetchone()
    assert tuple(failed)[2:] == ("failed", 4, "llm_error")

    # A later, separate session is the recovery trigger.  The old fragment is
    # kept intact and re-opened; it is not silently marked processed.
    evs.insert(event(1, NOW + timedelta(hours=1)))
    recovering = worker(db, mem, LLM())
    assert recovering.notify_reliable_activity("conversation-a")
    reopened = db.connection.execute(
        "SELECT job_id,revision,start_sequence,end_sequence,status,attempt_count,failure_category "
        "FROM memory_session_jobs"
    ).fetchone()
    assert reopened[0] == failed[0]
    assert tuple(reopened[1:]) == (2, 0, 0, "pending", 0, None)
    actions = [row[0] for row in db.connection.execute(
        "SELECT action FROM memory_job_events WHERE job_id=? ORDER BY rowid", (failed[0],)
    )]
    assert actions == ["retry_scheduled", "retry_scheduled", "retry_scheduled", "failed", "reopened"]

    recovering.open_semantic_gate()
    runs = await recovering.run_due(NOW + timedelta(hours=1))
    assert runs and runs[0].written_memory_ids
    assert db.connection.execute(
        "SELECT value_json FROM runtime_meta WHERE key=?",
        ("memory_worker:conversation-a:processed_sequence",),
    ).fetchone()[0] == "0"


@pytest.mark.asyncio
async def test_terminal_failure_is_not_reopened_without_new_reliable_activity(stack):
    db, evs, mem = stack
    evs.insert(event(0, NOW))
    failing = worker(db, mem, LLM(True))
    failing.notify_reliable_activity("conversation-a")
    failing.open_semantic_gate()
    for minute in (30, 31, 34, 45):
        await failing.run_due(NOW + timedelta(minutes=minute))
    assert not worker(db, mem, LLM()).notify_reliable_activity("conversation-a")
    row = db.connection.execute(
        "SELECT status,attempt_count,failure_category FROM memory_session_jobs"
    ).fetchone()
    assert tuple(row) == ("failed", 4, "llm_error")


@pytest.mark.asyncio
async def test_permanent_failure_quarantines_only_failed_range_and_allows_later_session(stack):
    db, evs, mem = stack
    evs.insert(event(0, NOW, text="x" * 20))
    failing = worker(db, mem, LLM())
    failing.max_fragment_tokens = 1
    failing.notify_reliable_activity("conversation-a")
    failing.open_semantic_gate()
    await failing.run_due(NOW + timedelta(minutes=30))
    failed = db.connection.execute(
        "SELECT job_id,status,failure_category FROM memory_session_jobs"
    ).fetchone()
    assert tuple(failed[1:]) == ("failed", "oversized_event")

    evs.insert(event(1, NOW + timedelta(hours=1)))
    succeeding = worker(db, mem, LLM())
    assert succeeding.notify_reliable_activity("conversation-a")
    next_job = db.connection.execute(
        "SELECT start_sequence,end_sequence,status FROM memory_session_jobs"
    ).fetchone()
    assert tuple(next_job) == (1, 1, "pending")
    quarantine = db.connection.execute(
        "SELECT value_json FROM runtime_meta WHERE key=?",
        ("memory_worker:conversation-a:quarantined_sequence",),
    ).fetchone()
    assert quarantine[0] == "0"
    succeeding.open_semantic_gate()
    runs = await succeeding.run_due(NOW + timedelta(hours=1, minutes=30))
    assert runs and runs[0].written_memory_ids
    assert db.connection.execute(
        "SELECT COUNT(*) FROM conversation_events WHERE event_id='0'"
    ).fetchone()[0] == 1


def test_recover_pending_reopens_old_transient_failure(stack):
    db, evs, mem = stack
    evs.insert(event(0, NOW))
    failing = worker(db, mem, LLM(True))
    failing.notify_reliable_activity("conversation-a")
    failing.open_semantic_gate()

    async def finish():
        for minute in (30, 31, 34, 45):
            await failing.run_due(NOW + timedelta(minutes=minute))

    import asyncio
    asyncio.run(finish())
    later = MemoryWorker(
        MemoryExtractor(LLM()),
        mem,
        database=db,
        conversation_id="conversation-a",
        clock=lambda: NOW + timedelta(hours=2),
    )
    assert later.recover_pending("conversation-a")
    row = db.connection.execute(
        "SELECT status,revision,attempt_count,failure_category FROM memory_session_jobs"
    ).fetchone()
    assert tuple(row) == ("pending", 2, 0, None)


class FakeDetailLLM:
    def __init__(self, payload, fail=False):
        self.payload, self.fail, self.calls = payload, fail, []

    async def generate(self, messages, *, thinking=None):
        self.calls.append(thinking)
        if self.fail:
            raise RuntimeError("detail pass broke")
        return LLMGeneration(self.payload, "primary", "m", 10, 20, 1.0)


def detail_payload(event_id, quote):
    return json.dumps({"details": [{
        "ordinal": 0, "detail_kind": "message", "actor": "mumo",
        "reality_scope": "conversation", "normalized_detail": "一条细节", "exact_quote": quote,
        "source_event_id": event_id, "certainty": "explicit", "temporal_scope": "historical",
        "status": "active", "privacy_class": "ordinary", "recall_policy": "daily_safe",
        "evidence": [[event_id, "source"]],
    }]}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_worker_fills_the_timeline_with_the_detail_pass(stack):
    db, evs, mem = stack
    evs.insert(event(0))
    client = FakeDetailLLM(detail_payload("0", "0"))
    w = MemoryWorker(MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
                     detail_pass=MemoryDetailPass(client))
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    await w.run_due(NOW + timedelta(minutes=30))
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 1
    assert client.calls == [{"type": "disabled"}]


@pytest.mark.asyncio
async def test_a_broken_detail_pass_still_lands_the_memory(stack):
    db, evs, mem = stack
    evs.insert(event(0))
    w = MemoryWorker(MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
                     detail_pass=MemoryDetailPass(FakeDetailLLM("", fail=True)))
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    runs = await w.run_due(NOW + timedelta(minutes=30))
    assert runs[0].written_memory_ids, "the fragment must still be written"
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 0
    assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_a_fabricated_quote_rolls_the_whole_fragment_back(stack):
    db, evs, mem = stack
    evs.insert(event(0))
    w = MemoryWorker(MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
                     detail_pass=MemoryDetailPass(FakeDetailLLM(detail_payload("0", "伪造的原话"))))
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    runs = await w.run_due(NOW + timedelta(minutes=30))
    assert runs[0].failed, "a fabricated quote must fail the transaction"
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 0
    assert db.connection.execute("SELECT COUNT(*) FROM memory_fragments").fetchone()[0] == 0


class FlakyDetailLLM:
    """Fails the whole-window call, answers each half with that half's first event."""

    def __init__(self):
        self.sizes = []

    async def generate(self, messages, *, thinking=None):
        events = json.loads(messages[1].content)["events"]
        self.sizes.append(len(events))
        if len(events) > 1:
            raise RuntimeError("whole window is too much for me")
        return LLMGeneration(
            detail_payload(events[0]["event_id"], events[0]["text"]), "primary", "m", 10, 20, 1.0
        )


def _last_job_details(db):
    return json.loads(db.connection.execute(
        "SELECT details_json FROM memory_job_events ORDER BY occurred_at_utc DESC LIMIT 1"
    ).fetchone()[0])


@pytest.mark.asyncio
async def test_a_failed_timeline_is_named_in_the_job_ledger(stack):
    """2026-09-12：时间线补跑失败过去什么都不留，于是「这一夜为什么是空的」查不出来。"""

    db, evs, mem = stack
    evs.insert(event(0))
    evs.insert(event(1))
    w = MemoryWorker(MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
                     detail_pass=MemoryDetailPass(FakeDetailLLM("", fail=True)))
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    runs = await w.run_due(NOW + timedelta(minutes=30))

    assert runs[0].written_memory_ids, "片段照旧落地"
    details = _last_job_details(db)
    assert details["detail_pass_failure"] == "detail_pass_error", "失败的种类要留下"
    assert details.get("detail_pass_split") == "1", "拆半重试过也要留下"
    assert db.connection.execute("SELECT COUNT(*) FROM memory_detail_records").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_a_broken_whole_window_call_is_retried_on_each_half(stack):
    """整段调用失败时拆成两半各跑一次，合并后仍按事件顺序重编号。"""

    db, evs, mem = stack
    evs.insert(event(0))
    evs.insert(event(1))
    client = FlakyDetailLLM()
    w = MemoryWorker(MemoryExtractor(LLM()), mem, database=db, conversation_id="conversation-a",
                     detail_pass=MemoryDetailPass(client))
    w.notify_reliable_activity("conversation-a")
    w.open_semantic_gate()
    await w.run_due(NOW + timedelta(minutes=30))

    rows = [tuple(row) for row in db.connection.execute(
        "SELECT ordinal, source_event_id FROM memory_detail_records ORDER BY ordinal")]
    assert rows == [(0, "0"), (1, "1")], "两半各自的时间线都要落地并按顺序重编号"
    details = _last_job_details(db)
    assert details["detail_pass_split"] == "1"
    assert details["detail_pass_failure"] == "detail_pass_error"


from qichi.dialogue.llm_client import LLMGeneration  # noqa: E402
from qichi.memory.detail_pass import MemoryDetailPass  # noqa: E402
