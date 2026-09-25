from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import pytest

from qichi.config import load_config
from qichi.dialogue.model_capability import ModelCapability, ProviderCapabilityEvidence
from qichi.runtime import RuntimeAssemblyError, build_runtime
from qichi.runtime import _MemoryExtractionLLM
from qichi.dialogue.llm_client import LLMGeneration
from qichi.domain.dialogue import ModelMessage
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.domain.memory import MemoryEvidence, MemoryRecord
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.memory_repository import MemoryRepository


NOW = datetime(2026, 8, 28, 10, tzinfo=timezone.utc)
MODEL = "deepseek-v4-flash"


class Counter:
    def count_text(self, text: str) -> int:
        return len(text)


class NoNetworkOneBot:
    pass


class CapturingMemoryClient:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    async def generate(self, messages, **kwargs):
        self.messages = tuple(messages)
        self.kwargs = dict(kwargs)
        return LLMGeneration(
            '{"candidates":[],"reviews":[]}', "primary", MODEL, 1, 1, 1.0
        )


def capability(tokens: int = 262144, model: str = MODEL) -> ModelCapability:
    """能力的身份必须跟着配置里的模型走：2026-09-13 起生产主模型是 pro。"""

    evidence = ProviderCapabilityEvidence("deepseek", model, tokens, "local-probe", NOW)
    return ModelCapability(model, 1_048_576, "deepseek", evidence)


def runtime_config(config_path, complete_environment):
    return load_config(config_path, environ=complete_environment)


def test_memory_consolidation_gets_its_own_larger_output_budget():
    """合并阶段要独立预算：2026-09-11 它共用对话的 4096/25s，一次超时加一次截断就把整段会话隔离了。"""
    from qichi.runtime import (
        MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS,
        MEMORY_EXTRACTION_TIMEOUT_SECONDS,
    )

    assert MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS == 8192
    assert MEMORY_EXTRACTION_TIMEOUT_SECONDS == 90
    source = (Path(__file__).parents[1] / "src" / "qichi" / "runtime.py").read_text(encoding="utf-8")
    assert "max_output_tokens=MEMORY_EXTRACTION_MAX_OUTPUT_TOKENS" in source
    assert "timeout_seconds=MEMORY_EXTRACTION_TIMEOUT_SECONDS" in source
    assert "memory_extraction_client," in source, "合并阶段必须用自己的客户端，而不是对话客户端"


class CapturingExtractionClient:
    def __init__(self):
        self.messages = None

    async def generate(self, messages, **kwargs):
        self.messages = tuple(messages)
        return LLMGeneration('{"outcome":{"kind":"no_persistent_memory","reason_code":"nothing_new"},'
                             '"candidates":[],"reviews":[]}', "primary", MODEL, 1, 1, 1.0)


@pytest.mark.asyncio
async def test_extraction_prompt_publishes_the_fragment_contract():
    """模型必须被告知 fragment 的字段名与枚举，否则只能猜（2026-09-11 的失败根因）。"""

    client = CapturingExtractionClient()
    adapter = _MemoryExtractionLLM(client, None, "123456", local_timezone="Asia/Shanghai")
    source = ConversationEvent(
        event_id="source", platform_event_id="pe-source", platform_message_id="pm-source",
        conversation_id="123456", sequence=0, direction="inbound", actor="mumo", kind="text",
        text="我喜欢雨声", message_segments=(MessageSegment("text", {"text": "我喜欢雨声"}),),
        reply_to_event_id=None, reply_to_platform_message_id=None,
        occurred_at_utc=NOW, received_at_utc=NOW, status="received", metadata={},
    )
    await adapter.generate((source,))

    prompt = " ".join(str(message.content) for message in client.messages)
    assert "存在 candidates 或 details 时必须给出 fragment" in prompt
    assert "fragment_type 只能是 daily、intimate、adult、mixed、unknown" in prompt
    assert "recall_policy 只能是 daily_safe、topic_only、explicit_request_only" in prompt
    assert "closed 是布尔" in prompt


def test_build_runtime_wires_configured_256k_without_network(
    config_path, complete_environment, tmp_path
):
    config = runtime_config(config_path, complete_environment)
    database = Database(tmp_path / "qichi.sqlite3")
    components = build_runtime(
        config,
        project_root=config_path.parent,
        database=database,
        onebot_client=NoNetworkOneBot(),
        bot_qq="10001",
        model_capability=capability(model=config.llm.primary.model),
        token_counter=Counter(),
        face_catalog={"smile": 1},
    )

    assert components.context_builder._preferred_window_tokens == 262144
    assert components.context_builder._max_window_tokens == 262144
    assert components.context_builder._output_reserve_tokens == 6144
    assert components.llm_client._max_output_tokens == 6144
    assert components.dialogue_engine._guard._max == 1200
    assert components.application.owner_qq == "123456"
    assert components.application.bot_qq == "10001"
    assert components.application._available_face_keys == ()
    assert components.application._available_reaction_keys == ()
    assert components.application.memory_retriever.candidate_limit == 24
    assert components.application.memory_retriever.context_limit == 12
    assert components.llm_client.closed is False
    database.close()


@pytest.mark.asyncio
async def test_memory_extraction_prompt_and_payload_use_the_complete_minimal_fragment(tmp_path):
    client = CapturingMemoryClient()
    database = Database(tmp_path / "memory-prompt.sqlite3")
    event = ConversationEvent(
        event_id="m1", platform_event_id="p1", platform_message_id="q1",
        conversation_id="123456", sequence=1, direction="inbound", actor="mumo",
        kind="text", text="这一轮继续这样说", message_segments=(MessageSegment("text", {"text": "这一轮继续这样说"}),),
        reply_to_event_id=None, reply_to_platform_message_id=None,
        occurred_at_utc=NOW, received_at_utc=NOW, status="received", metadata={"private": "not-for-model"},
    )
    persisted = EventRepository(database).insert(event)
    evidence = (MemoryEvidence("active-memory", persisted.event_id, "mumo", "继续这样说", NOW),)
    active = MemoryRecord(
        "active-memory", "preference", "用户曾明确表达过一项长期偏好", "explicit_statement",
        "active", NOW, None, None, NOW, evidence,
        "explicit", 2, "ongoing", "explicit_user_statement", NOW,
    )
    MemoryRepository(database).create(active)
    candidate_evidence = (
        MemoryEvidence("candidate-memory", persisted.event_id, "mumo", "继续这样说", NOW),
    )
    candidate = MemoryRecord(
        "candidate-memory", "preference", "这句话是否代表长期偏好尚不明确", "explicit_statement",
        "candidate", NOW, None, None, NOW, candidate_evidence,
        "ambiguous", 2, "unclassified", "ambiguous_scope", NOW,
    )
    MemoryRepository(database).create(candidate)
    audit_event = EventRepository(database).insert(
        ConversationEvent(
            event_id="audit-qichi",
            platform_event_id="audit-platform",
            platform_message_id="audit-message",
            conversation_id="123456",
            sequence=2,
            direction="outbound",
            actor="qichi",
            kind="text",
            text="我先按这一轮来理解",
            message_segments=(
                MessageSegment("text", {"text": "我先按这一轮来理解"}),
            ),
            reply_to_event_id="m1",
            reply_to_platform_message_id="q1",
            occurred_at_utc=NOW,
            received_at_utc=NOW,
            status="sent",
            metadata={"generation_metadata": {"source": "dialogue"}},
        )
    )
    self_expression_evidence = (
        MemoryEvidence(
            "self-expression-memory",
            audit_event.event_id,
            "qichi",
            audit_event.text,
            NOW,
        ),
    )
    self_expression = MemoryRecord(
        "self-expression-memory",
        "self_expression",
        "这是一条只保留作审计的角色表达",
        "explicit_statement",
        "candidate",
        NOW,
        None,
        None,
        NOW,
        self_expression_evidence,
        "explicit",
        1,
        "ongoing",
        "explicit_user_statement",
        NOW,
    )
    # Bypass the public creator here because this test only needs to prove
    # that audit-only records are absent from the model's review targets.
    database.connection.execute(
        "INSERT INTO memory_records "
        "(memory_id,type,normalized_fact,modality,status,valid_from_utc,valid_until_utc,"
        "supersedes_id,created_at_utc,certainty,importance,temporal_scope,"
        "assessment_reason_code,assessed_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            self_expression.memory_id,
            self_expression.type,
            self_expression.normalized_fact,
            self_expression.modality,
            self_expression.status,
            self_expression.valid_from_utc.isoformat(),
            None,
            None,
            self_expression.created_at_utc.isoformat(),
            self_expression.certainty,
            self_expression.importance,
            self_expression.temporal_scope,
            self_expression.assessment_reason_code,
            self_expression.assessed_at_utc.isoformat(),
        ),
    )
    database.connection.execute(
        "INSERT INTO memory_evidence "
        "(memory_id,event_id,actor,exact_quote,occurred_at_utc,evidence_role) "
        "VALUES (?,?,?,?,?,?)",
        (
            self_expression.memory_id,
            audit_event.event_id,
            "qichi",
            audit_event.text,
            NOW.isoformat(),
            "source",
        ),
    )
    second = ConversationEvent(
        event_id="m2", platform_event_id="p2", platform_message_id="q2",
        conversation_id="123456", sequence=3, direction="outbound", actor="qichi",
        kind="text", text="我先按这一轮来理解", message_segments=(MessageSegment("text", {"text": "我先按这一轮来理解"}),),
        reply_to_event_id="m1", reply_to_platform_message_id="q1",
        occurred_at_utc=NOW, received_at_utc=NOW, status="sent",
        metadata={"generation_metadata": {"source": "dialogue"}},
    )
    extractor = _MemoryExtractionLLM(client, MemoryRepository(database), "123456")
    await extractor.generate((persisted, second))
    assert client.messages is not None
    prompt = client.messages[0].content
    assert "当前场景的请求、命令" in prompt
    assert "持续关系事实与跨会话短期约定" in prompt
    assert "confirmed + bounded agreement" in prompt
    assert "次日 08:00" in prompt
    assert "绝不能改写为 ongoing" in prompt
    assert "单方提议、没有独立 acceptance" in prompt
    # P5: accepting an arrangement must never be written as an obligation.
    assert "接受只证明当时接受" in prompt
    # Recorded finding (2026-09-11): marking fragment/details optional is why a
    # 182-event replay produced fragments with zero details.  Making them
    # obligatory was tried and REVERTED: the model then exceeded its 4096-token
    # output budget, timing out and returning invalid JSON, and the fragment was
    # quarantined.  The contract's caps (256 details, 2048-char quotes) cannot fit
    # that budget, so this needs a sizing or budget decision, not a wording tweak.
    # Second attempt reverted (2026-09-11): with the obligation wording the extractor
    # timed out three times, twice produced an invalid fragment_type, and the one
    # successful pass still recorded zero details.  Knob tuning is exhausted; the
    # timeline needs its own pipeline step.  See the ledger row P12-DETAIL-TIMELINE-BLOCKED.
    assert "fragment 是可选的连续片段索引" in prompt
    assert "details 是可选的有序详细时间线" in prompt
    assert "最多 32 条" in prompt  # aligned with the extractor output budget
    assert "最多 32 条" in prompt  # aligned with the extractor output budget
    assert "禁止把任何一方的接受写成" in prompt
    assert "不许反悔" in prompt
    # ...and the insertion must not have displaced the existing safeguards.
    assert "不代表当前同意、持续同意或未来许可" in prompt
    assert "不扩写行为细节" in prompt
    assert "角色扮演" in prompt
    assert "记忆范围指令" in prompt
    assert "完整冻结片段" in prompt
    assert "candidates" in prompt and "reviews" in prompt
    assert "legacy_manual_review 只供旧数据迁移，模型绝不能生成" in prompt
    assert "confirm 必须使用 confirmed" in prompt
    assert "active 目标绝不能 support" in prompt
    assert "self_expression 只作审计" in prompt
    assert "unsupported_or_transient 只能用于新 candidate" in prompt
    assert "绝不能输出第 13 条" in prompt
    assert "preference 和 episode 只能使用" in prompt
    assert "agreement 必须同时包含不同 event_id" in prompt
    assert "方向绝不能倒置" in prompt
    assert "只处理当前这一条" not in prompt
    payload = __import__("json").loads(client.messages[1].content)
    assert set(payload) == {
        "events", "evidence_event_ids", "reviewable_memories", "time_context",
    }
    assert payload["time_context"] == {
        "timezone": "Asia/Shanghai",
        "fragment_start_local": "2026-08-28T18:00:00+08:00",
        "fragment_end_local": "2026-08-28T18:00:00+08:00",
    }
    assert payload["evidence_event_ids"] == ["m1", "m2"]
    assert [item["event_id"] for item in payload["events"]] == ["m1", "m2"]
    assert {item["status"] for item in payload["reviewable_memories"]} == {"active", "candidate"}
    assert {
        item["memory_id"]: item["allowed_review_actions"]
        for item in payload["reviewable_memories"]
    } == {
        "active-memory": ["confirm", "reject", "expire"],
        "candidate-memory": ["support", "confirm", "reject", "expire"],
    }
    assert set(payload["events"][0]) == {
        "event_id", "sequence", "direction", "actor", "kind", "text",
        "occurred_at_utc", "reply_to_event_id",
    }
    assert all(set(item) == {
        "memory_id", "type", "normalized_fact", "modality", "status", "certainty",
        "importance", "temporal_scope", "assessment_reason_code", "valid_from_utc",
        "valid_until_utc", "supersedes_id", "allowed_review_actions", "evidence",
    } for item in payload["reviewable_memories"])
    assert all(
        set(evidence_item) == {
            "event_id", "actor", "exact_quote", "occurred_at_utc", "role",
        }
        for item in payload["reviewable_memories"]
        for evidence_item in item["evidence"]
    )
    serialized = client.messages[1].content
    assert "platform_message_id" not in serialized
    assert "message_segments" not in serialized
    assert "metadata" not in serialized
    assert "not-for-model" not in serialized
    assert "created_at_utc" not in serialized
    assert "assessed_at_utc" not in serialized
    assert '"memory_id":"active-memory"' not in serialized.split('"reviewable_memories"', 1)[0]
    assert client.kwargs == {
        "max_output_tokens": 8192,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.0,
        "top_p": 1.0,
    }
    database.close()


def test_memory_extraction_rejects_unknown_local_timezone():
    with pytest.raises(ValueError, match="local_timezone"):
        _MemoryExtractionLLM(CapturingMemoryClient(), local_timezone="Mars/Olympus")


@pytest.mark.asyncio
async def test_memory_extraction_rejects_conversation_mismatch_before_model_call():
    client = CapturingMemoryClient()
    fragment = ConversationEvent(
        event_id="mismatch", platform_event_id=None, platform_message_id=None,
        conversation_id="conversation-a", sequence=0, direction="inbound", actor="mumo",
        kind="text", text="原话", message_segments=(), reply_to_event_id=None,
        reply_to_platform_message_id=None, occurred_at_utc=NOW, received_at_utc=NOW,
        status="received", metadata={},
    )
    extractor = _MemoryExtractionLLM(client, conversation_id="conversation-b")
    with pytest.raises(ValueError, match="does not match"):
        await extractor.generate((fragment,))
    assert client.messages is None


@pytest.mark.asyncio
async def test_memory_extraction_projects_context_only_events_but_not_as_evidence():
    client = CapturingMemoryClient()
    first = ConversationEvent(
        event_id="evidence", platform_event_id=None, platform_message_id=None,
        conversation_id="conversation-a", sequence=1, direction="inbound", actor="mumo",
        kind="text", text="我喜欢雨声", message_segments=(), reply_to_event_id=None,
        reply_to_platform_message_id=None, occurred_at_utc=NOW, received_at_utc=NOW,
        status="received", metadata={},
    )
    context_only = ConversationEvent(
        event_id="context", platform_event_id=None, platform_message_id=None,
        conversation_id="conversation-a", sequence=2, direction="outbound", actor="qichi",
        kind="initiative", text="要听一会儿雨吗", message_segments=(), reply_to_event_id=None,
        reply_to_platform_message_id=None, occurred_at_utc=NOW, received_at_utc=NOW,
        status="sent", metadata={},
    )
    extractor = _MemoryExtractionLLM(client, conversation_id="conversation-a")
    await extractor.generate(
        (first, context_only), evidence_event_ids=frozenset({"evidence"})
    )
    payload = __import__("json").loads(client.messages[1].content)
    assert [item["event_id"] for item in payload["events"]] == ["evidence", "context"]
    assert payload["evidence_event_ids"] == ["evidence"]
    assert "context-only" in client.messages[0].content


def test_build_runtime_rejects_unverified_or_insufficient_context_before_wiring(
    config_path, complete_environment, tmp_path
):
    config = runtime_config(config_path, complete_environment)
    database = Database(tmp_path / "qichi.sqlite3")
    unknown = ModelCapability(config.llm.primary.model, 1_048_576, "deepseek", None)
    with pytest.raises(RuntimeAssemblyError, match="context window"):
        build_runtime(
            config,
            project_root=config_path.parent,
            database=database,
            onebot_client=NoNetworkOneBot(),
            bot_qq="10001",
            model_capability=unknown,
            token_counter=Counter(),
            face_catalog={"smile": 1},
        )
    database.close()


def test_build_runtime_requires_catalog_when_qq_face_enabled(
    config_path, complete_environment, tmp_path
):
    config = runtime_config(config_path, complete_environment)
    config = replace(
        config,
        expression=replace(
            config.expression,
            qq_face=replace(config.expression.qq_face, enabled=True),
        ),
    )
    database = Database(tmp_path / "qichi.sqlite3")
    with pytest.raises(RuntimeAssemblyError, match="face catalog"):
        build_runtime(
            config,
            project_root=config_path.parent,
            database=database,
            onebot_client=NoNetworkOneBot(),
            bot_qq="10001",
            model_capability=capability(model=config.llm.primary.model),
            token_counter=Counter(),
        )
    database.close()


def test_build_runtime_loads_verified_face_catalog_when_enabled(
    config_path, complete_environment, tmp_path
):
    config = runtime_config(config_path, complete_environment)
    config = replace(
        config,
        expression=replace(
            config.expression,
            qq_face=replace(
                config.expression.qq_face,
                enabled=True,
                runtime_catalog="data/qq-expression-catalog.example.json",
            ),
        ),
    )
    database = Database(tmp_path / "qichi.sqlite3")
    components = build_runtime(
        config,
        project_root=config_path.parent,
        database=database,
        onebot_client=NoNetworkOneBot(),
        bot_qq="10001",
        model_capability=capability(model=config.llm.primary.model),
        token_counter=Counter(),
    )
    assert components.application.face_catalog["shy"] == 6
    database.close()


def test_build_runtime_rejects_model_identity_mismatch(
    config_path, complete_environment, tmp_path
):
    config = runtime_config(config_path, complete_environment)
    database = Database(tmp_path / "qichi.sqlite3")
    other = ModelCapability(
        "other-model",
        1_048_576,
        "deepseek",
        ProviderCapabilityEvidence("deepseek", "other-model", 131072, "local-probe", NOW),
    )
    with pytest.raises(RuntimeAssemblyError, match="model"):
        build_runtime(
            config,
            project_root=config_path.parent,
            database=database,
            onebot_client=NoNetworkOneBot(),
            bot_qq="10001",
            model_capability=other,
            token_counter=Counter(),
            face_catalog={"smile": 1},
        )
    database.close()


def test_build_runtime_rejects_invalid_enabled_face_catalog(
    config_path, complete_environment, tmp_path
):
    config = runtime_config(config_path, complete_environment)
    config = replace(
        config,
        expression=replace(
            config.expression,
            qq_face=replace(config.expression.qq_face, enabled=True),
        ),
    )
    database = Database(tmp_path / "qichi.sqlite3")
    with pytest.raises(ValueError, match="decimal ID"):
        build_runtime(
            config,
            project_root=config_path.parent,
            database=database,
            onebot_client=NoNetworkOneBot(),
            bot_qq="10001",
            model_capability=capability(model=config.llm.primary.model),
            token_counter=Counter(),
            face_catalog={"smile": "not-an-id"},
        )
    database.close()
