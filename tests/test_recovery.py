from __future__ import annotations
import json
import pytest
from qichi.readiness import ReadinessError,recover_database
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.domain.events import ConversationEvent,MessageSegment
from test_interactions import NOW,OWNER

def inbound(event_id="in",sequence=0,status="received"):
    return ConversationEvent(event_id,None,None,OWNER,sequence,"inbound","mumo","text","x",(MessageSegment("text",{"text":"x"}),),None,None,NOW,NOW,status,{})

def add_outbox(db,key,action,status,attempt):
    event=ConversationEvent(f"e-{key}",None,None,OWNER,0,"outbound","qichi","text","x",(MessageSegment("text",{"text":"x"}),),None,None,NOW,NOW,"pending",{})
    event=EventRepository(db).insert(event)
    payload = {"action_kind": action}
    if action == "reaction":
        payload["set"] = True
    db.connection.execute("INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?)",(key,event.event_id,json.dumps(payload),status,attempt,None,NOW.isoformat(),NOW.isoformat()))

def test_recovery_owner_cursor_and_outbox_matrix_is_read_only(tmp_path):
    db=Database(tmp_path/"db.sqlite3");db.connection.execute("INSERT INTO conversation_cursors(conversation_id,last_processed_sequence) VALUES (?,?)",(OWNER,4))
    for args in (("pt","text","pending",0),("pr","reaction","pending",0),("u","text","unknown",1),("d","poke","dispatched",1),("s","text","sent",1)):add_outbox(db,*args)
    before=db.connection.total_changes;report=recover_database(db,conversation_id=OWNER)
    assert report.last_recovered_sequence==4 and report.pending_text==("pt",) and report.pending_reaction==("pr",)
    assert report.frozen_unknown==("u",) and report.frozen_dispatched==("d",) and report.terminal==("s",) and report.network_calls==0
    assert db.connection.total_changes==before;db.close()

def test_recovery_no_cursor_is_minus_one_but_received_inbound_fails(tmp_path):
    db=Database(tmp_path/"db.sqlite3");assert recover_database(db,conversation_id=OWNER).last_recovered_sequence==-1
    EventRepository(db).insert(inbound())
    with pytest.raises(ReadinessError,match="explicit replay"):recover_database(db,conversation_id=OWNER)
    assert EventRepository(db).get("in").status=="received";db.close()

def test_recovery_rejects_inbound_beyond_cursor_and_schema_drift(tmp_path):
    db=Database(tmp_path/"db.sqlite3");EventRepository(db).insert(inbound());db.connection.execute("INSERT INTO conversation_cursors(conversation_id,last_processed_sequence) VALUES (?,?)",(OWNER,-1))
    with pytest.raises(ReadinessError):recover_database(db,conversation_id=OWNER)
    db.connection.execute("UPDATE runtime_meta SET value_json='99' WHERE key='schema_version'")
    with pytest.raises(ReadinessError,match="schema"):recover_database(db,conversation_id=OWNER);db.close()

def test_recovery_integrity_requires_exact_single_ok(tmp_path,monkeypatch):
    db=Database(tmp_path/"db.sqlite3")
    class Fake:
        def execute(self,sql,*args):
            if sql=="PRAGMA integrity_check":return self
            return db.connection.execute(sql,*args)
        def fetchall(self):return [("ok",),("ok",)]
    original=db.connection;db.connection=Fake()
    with pytest.raises(ReadinessError,match="integrity"):recover_database(db,conversation_id=OWNER)
    db.connection=original;db.close()

def test_recovery_allows_idempotent_reaction_pending_after_attempt(tmp_path):
    db=Database(tmp_path/"db.sqlite3")
    add_outbox(db,"reaction-recovered","reaction","pending",1)
    db.connection.execute("UPDATE outbox SET payload_json=? WHERE operation_key=?",(json.dumps({"action_kind":"reaction","set":True}),"reaction-recovered"))
    report=recover_database(db,conversation_id=OWNER)
    assert report.pending_reaction==("reaction-recovered",)
    db.close()


def test_recovery_accepts_a_pending_voice_row_without_wedging_startup(tmp_path):
    """2026-09-15 真机：一条没发完的语音让启动 fail closed（outbox action invalid）。

    语音的契约是永不重发，所以它的 pending 收进 frozen：既不进重发通道，也不卡启动。
    """

    db=Database(tmp_path/"db.sqlite3")
    # 用仓库自己的写入接口造行 —— 手搓的 INSERT 过不了它自己的载荷校验，那不是被测对象。
    from qichi.storage.outbox_repository import OutboxRepository
    outbox=OutboxRepository(db)
    payload={"action_kind":"record","user_id":OWNER,"message":[{"type":"record","data":{"file":"file:///x.wav"}}]}
    for key in ("voice-pending","voice-unknown"):
        event=ConversationEvent(f"e-{key}",None,None,OWNER,0,"outbound","qichi","text","x",(MessageSegment("record",{"file":"file:///x.wav"}),),None,None,NOW,NOW,"pending",{})
        event=EventRepository(db).insert(event)
        outbox.create_intent(key,event.event_id,payload,NOW)
    # 两种生产形状：还没发出去就中断的（pending/0 次），和发出去但没收到回执的（unknown）。
    # 注意 pending+已尝试 会被仓库自己的转换规则拒绝，那不是 readiness 这一层的被测对象。
    outbox.begin_dispatch("voice-unknown",NOW)
    outbox.complete_dispatch("voice-unknown","unknown",None,NOW)
    report=recover_database(db,conversation_id=OWNER)
    assert report.frozen_dispatched==("voice-pending",)
    assert report.frozen_unknown==("voice-unknown",)
    assert report.pending_text==(), "语音绝不能被当成可重发的文字"
    db.close()


def test_recovery_still_refuses_an_unknown_action_kind(tmp_path):
    """不误判：承认 record 不等于放开门，真正不认识的动作仍然 fail closed。"""

    db=Database(tmp_path/"db.sqlite3")
    add_outbox(db,"mystery","telepathy","sent",1)
    with pytest.raises(ReadinessError):   # 停在解析就 fail closed，比动作白名单更早
        recover_database(db,conversation_id=OWNER)
    db.close()
