from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qichi.domain.dialogue import DialogueResult, ExpressionIntent
from qichi.domain.events import ConversationEvent, MessageSegment
from qichi.storage.database import Database
from qichi.storage.event_repository import EventRepository
from qichi.storage.outbox_repository import OutboxRepository

from .onebot_client import (
    OneBotActionError,
    OneBotConnectionError,
    OneBotHTTPClientError,
    OneBotHTTPRedirectError,
    OneBotHTTPServerError,
    OneBotProtocolError,
    OneBotTimeoutError,
)

_OBSERVATION_KEYS = frozenset({
    "source",
    "context_version",
    "relationship_memory_ids",
    "working_set_memory_ids",
    "retrieved_memory_ids",
    "candidate_memory_ids",  # read-only compatibility for pre-T58 traces
    "quoted_event_id",
    "history_event_count",
})
_FAILURE_NOTICE_TEXT = "消息已收到，但本次处理未完成，请稍后重试。"

def _safe_generation_metadata(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) - _OBSERVATION_KEYS:
        raise ValueError("generation_metadata contains unsupported fields")
    result = dict(value)
    if result.get("source") not in {"dialogue", "interaction", "initiative"}:
        raise ValueError("generation_metadata source is invalid")
    if type(result.get("context_version")) is not int or result["context_version"] < 0:
        raise ValueError("generation_metadata context_version is invalid")
    for key in (
        "relationship_memory_ids",
        "working_set_memory_ids",
        "retrieved_memory_ids",
        "candidate_memory_ids",
    ):
        ids = result.get(key, ())
        if not isinstance(ids, (tuple, list)) or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("generation_metadata memory IDs are invalid")
        result[key] = tuple(ids)
    if (
        "retrieved_memory_ids" in result
        and "candidate_memory_ids" in result
        and result["retrieved_memory_ids"] != result["candidate_memory_ids"]
    ):
        raise ValueError("generation_metadata memory ID aliases disagree")
    if result.get("quoted_event_id") is not None and (not isinstance(result["quoted_event_id"], str) or not result["quoted_event_id"]):
        raise ValueError("generation_metadata quote ID is invalid")
    if type(result.get("history_event_count")) is not int or result["history_event_count"] < 0:
        raise ValueError("generation_metadata history count is invalid")
    return result


class Sender:
    def __init__(
        self,
        database: Database,
        onebot_client: Any,
        *,
        face_catalog: Mapping[str, int | str],
        reaction_catalog: Mapping[str, int | str],
    ):
        self.database = database
        self.client = onebot_client
        self.events = EventRepository(database)
        self.outbox = OutboxRepository(database)
        self.face_catalog = dict(face_catalog)
        self.reaction_catalog = dict(reaction_catalog)

    @staticmethod
    def text_operation_key(event_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"qichi:outbox:text:{event_id}"))

    @staticmethod
    def reaction_operation_key(event_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"qichi:outbox:reaction:{event_id}"))

    @staticmethod
    def record_operation_key(event_id: str) -> str:
        """语音段自己的幂等键。

        绝不与 text 共用：payload 形状不同，create_intent 会直接判成
        "operation_key immutable intent identity conflict"（2026-09-14，
        见 doc/TTS-实施计划-20260914.md §3.5）。
        """

        return str(uuid5(NAMESPACE_URL, f"qichi:outbox:record:{event_id}"))

    @staticmethod
    def poke_operation_key(event_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"qichi:outbox:poke:{event_id}"))

    async def send_poke(
        self,
        event: ConversationEvent,
        *,
        owner_qq: int | str,
        occurred_at_utc: datetime,
    ) -> str:
        normalized_owner = self._decimal_id(owner_qq)
        if normalized_owner is None or event.conversation_id != normalized_owner:
            raise ValueError("poke event must belong to owner conversation")
        if event.kind != "poke" or event.direction != "inbound" or event.actor != "mumo" or event.status != "received":
            raise ValueError("event must be an inbound poke")
        operation_key = self.poke_operation_key(event.event_id)
        payload = {"action_kind": "poke", "user_id": normalized_owner}
        record = self.outbox.create_intent(operation_key, event.event_id, payload, occurred_at_utc)
        if record.status != "pending":
            return record.status
        try:
            self.outbox.begin_dispatch(operation_key, occurred_at_utc)
        except ValueError:
            return self.outbox.get(operation_key).status
        try:
            response = await self.client.send_poke(normalized_owner)
        except (OneBotTimeoutError, OneBotConnectionError, OneBotHTTPServerError, OneBotProtocolError) as error:
            self.outbox.complete_dispatch(operation_key, "unknown", {"error": type(error).__name__}, occurred_at_utc)
            return "unknown"
        except (OneBotActionError, OneBotHTTPClientError, OneBotHTTPRedirectError) as error:
            self.outbox.complete_dispatch(operation_key, "failed", {"error": type(error).__name__}, occurred_at_utc)
            return "failed"
        # A poke has nothing to return: NapCat answers status ok with no payload at
        # all, and the client has already validated that envelope.  A missing
        # payload therefore means success, while a malformed one stays unknown.
        if response is not None and not isinstance(response, Mapping):
            self.outbox.complete_dispatch(operation_key, "unknown", {"error": "invalid_poke_response"}, occurred_at_utc)
            return "unknown"
        self.outbox.complete_dispatch(operation_key, "sent", dict(response) if response else {}, occurred_at_utc)
        return "sent"

    async def send_failure_notice(
        self,
        source_event: ConversationEvent,
        *,
        owner_qq: int | str,
        occurred_at_utc: datetime,
        reason: str,
    ) -> str:
        """Deliver an idempotent transport status after a failed inbound turn.

        This is a platform notice, not a generated Qichi reply. It is stored
        as an internal event so it cannot become conversational history or a
        future style exemplar, while its outbox record prevents duplicate
        notices after consumer retries.
        """
        normalized_owner = self._decimal_id(owner_qq)
        if normalized_owner is None or source_event.conversation_id != normalized_owner:
            raise ValueError("failure notice conversation must match owner_qq")
        if source_event.direction != "inbound":
            raise ValueError("failure notice source must be inbound")
        if not isinstance(reason, str) or re.fullmatch(r"[a-z0-9:_-]{1,64}", reason) is None:
            raise ValueError("failure notice reason is invalid")
        if not isinstance(occurred_at_utc, datetime) or occurred_at_utc.tzinfo is None or occurred_at_utc.utcoffset() is None:
            raise ValueError("occurred_at_utc must be timezone-aware")
        timestamp = occurred_at_utc.astimezone(timezone.utc)
        event_id = str(uuid5(NAMESPACE_URL, f"qichi:failure-notice:{source_event.event_id}"))
        operation_key = str(uuid5(NAMESPACE_URL, f"qichi:outbox:failure-notice:{source_event.event_id}"))
        notice = ConversationEvent(
            event_id=event_id,
            platform_event_id=None,
            platform_message_id=None,
            conversation_id=normalized_owner,
            sequence=0,
            direction="internal",
            actor="platform",
            kind="processing_failure_notice",
            text=_FAILURE_NOTICE_TEXT,
            message_segments=(MessageSegment("text", {"text": _FAILURE_NOTICE_TEXT}),),
            reply_to_event_id=None,
            reply_to_platform_message_id=None,
            occurred_at_utc=timestamp,
            received_at_utc=timestamp,
            status="pending",
            metadata={"source_event_id": source_event.event_id, "reason": reason},
        )
        try:
            existing = self.events.get(event_id)
        except KeyError:
            self.events.insert(notice)
        else:
            if (
                existing.direction != "internal"
                or existing.actor != "platform"
                or existing.kind != notice.kind
                or existing.conversation_id != notice.conversation_id
                or existing.text != notice.text
                or existing.metadata != notice.metadata
            ):
                raise ValueError("failure notice identity conflict")
        payload = {
            "action_kind": "text",
            "user_id": normalized_owner,
            "message": [segment.to_dict() for segment in notice.message_segments],
        }
        return await self._dispatch_text(operation_key, self.events.get(event_id), payload, timestamp)

    async def send(
        self,
        result: DialogueResult,
        *,
        event_id: str,
        conversation_id: str,
        owner_qq: int | str,
        occurred_at_utc: datetime,
        dispatch_guard: Callable[[Any], bool] | None = None,
        sent_commit_hook: Callable[[Any], None] | None = None,
        generation_metadata: Mapping[str, object] | None = None,
    ) -> ConversationEvent | None:
        normalized_owner = self._decimal_id(owner_qq)
        if normalized_owner is None or conversation_id != normalized_owner:
            raise ValueError("conversation_id must match owner_qq")
        if sent_commit_hook is not None and not callable(sent_commit_hook):
            raise TypeError("sent_commit_hook must be callable")
        if sent_commit_hook is not None and dispatch_guard is None:
            raise ValueError("sent_commit_hook requires guarded dispatch")
        parts = result.message_parts
        if not isinstance(parts, tuple) or not parts:
            raise ValueError("DialogueResult must contain message_parts")
        delivered: ConversationEvent | None = None
        part_count = len(parts)
        for index, part in enumerate(parts):
            part_result = replace(
                result,
                text=part,
                reply_target=result.reply_target if index == 0 else None,
                expression_intent=(
                    result.expression_intent if index == part_count - 1 else None
                ),
                message_parts=(part,),
                # 逐段发送时，段号只在**原来那一轮**里成立（语音段已由上层摘走）。
                # 不重置会让带 [[qq:voice:N]] 的结果在这里直接越界炸掉。
                voice_part_index=None,
            )
            part_event_id = self.message_part_event_id(event_id, index, part_count)
            delivered = await self._send_one(
                part_result,
                event_id=part_event_id,
                conversation_id=conversation_id,
                owner_qq=normalized_owner,
                occurred_at_utc=occurred_at_utc,
                dispatch_guard=dispatch_guard,
                sent_commit_hook=(
                    sent_commit_hook if index == 0 else None
                ),
                delivery_group=(event_id, index, part_count),
                generation_metadata=generation_metadata,
            )
            if delivered is None or delivered.status != "sent":
                return delivered
        return delivered

    @staticmethod
    def voice_part_event_id(group_event_id: str, part_index: int) -> str:
        """语音段自己的事件 id。

        不能用 message_part_event_id：文字组被摘掉一段之后只剩一段，"最后一段沿用组事件 id"
        的约定会同时命中两边 —— 语音和文字抢同一个 event_id（2026-09-14 端到端回放抓到的 bug）。
        """

        if not isinstance(group_event_id, str) or not group_event_id:
            raise ValueError("group_event_id must be non-empty text")
        if type(part_index) is not int or part_index < 0:
            raise ValueError("part index is invalid")
        return str(uuid5(NAMESPACE_URL, f"qichi:voice-part:{group_event_id}:{part_index}"))

    @staticmethod
    def message_part_event_id(group_event_id: str, index: int, count: int) -> str:
        if not isinstance(group_event_id, str) or not group_event_id:
            raise ValueError("group_event_id must be non-empty text")
        if type(index) is not int or type(count) is not int or count < 1 or not 0 <= index < count:
            raise ValueError("message part position is invalid")
        if index == count - 1:
            return group_event_id
        return str(
            uuid5(
                NAMESPACE_URL,
                f"qichi:outbound-part:{group_event_id}:{index}:{count}",
            )
        )

    async def _send_one(
        self,
        result: DialogueResult,
        *,
        event_id: str,
        conversation_id: str,
        owner_qq: str,
        occurred_at_utc: datetime,
        dispatch_guard: Callable[[Any], bool] | None,
        sent_commit_hook: Callable[[Any], None] | None,
        delivery_group: tuple[str, int, int],
        generation_metadata: Mapping[str, object] | None,
    ) -> ConversationEvent | None:
        reply_target = self._reply_target(conversation_id, result.reply_target)
        segments = self._message_segments(result, reply_target)
        candidate = self._build_event(
            event_id,
            conversation_id,
            result,
            reply_target,
            segments,
            occurred_at_utc,
            delivery_group,
            generation_metadata,
        )
        text_key = self.text_operation_key(event_id)
        if dispatch_guard is not None:
            prepared = self._prepare_guarded_text(
                text_key, candidate, result, owner_qq, occurred_at_utc, dispatch_guard
            )
            if prepared is None:
                return None
            event, text_payload, text_status, dispatch_authorized = prepared
            if dispatch_authorized:
                text_status = await self._dispatch_started_text(
                    text_key,
                    event,
                    text_payload,
                    occurred_at_utc,
                    sent_commit_hook=sent_commit_hook,
                )
            elif text_status == "dispatched":
                raise RuntimeError("guarded text dispatch is awaiting reconciliation")
            if text_status == "sent":
                await self._dispatch_stored_reaction(event, occurred_at_utc)
            return self.events.get(event_id)

        try:
            event = self.events.get(event_id)
        except KeyError:
            event = self.events.insert(candidate)
        else:
            self._validate_existing_event(event, conversation_id, result)
        text_payload = {
            "action_kind": "text",
            "user_id": owner_qq,
            "message": [segment.to_dict() for segment in event.message_segments],
        }
        text_status = await self._dispatch_text(
            text_key,
            event,
            text_payload,
            occurred_at_utc,
            sent_commit_hook=sent_commit_hook,
        )
        if text_status == "sent":
            await self._dispatch_stored_reaction(event, occurred_at_utc)
        return self.events.get(event_id)

    def _build_event(
        self,
        event_id: str,
        conversation_id: str,
        result: DialogueResult,
        reply_target: ConversationEvent | None,
        segments: tuple[MessageSegment, ...],
        occurred_at_utc: datetime,
        delivery_group: tuple[str, int, int],
        generation_metadata: Mapping[str, object] | None = None,
    ) -> ConversationEvent:
        face_key = (
            result.expression_intent.key
            if result.expression_intent is not None
            and result.expression_intent.kind == "face"
            and any(segment.type == "face" for segment in segments)
            else None
        )
        reaction_details = self._reaction_details(conversation_id, result.expression_intent)
        metadata = {
            "delivery_group": {
                "group_event_id": delivery_group[0],
                "part_index": delivery_group[1],
                "part_count": delivery_group[2],
            },
            "expression": {
                "requested_reply_target": result.reply_target,
                "requested": result.expression_intent.to_dict() if result.expression_intent else None,
                "face_segment_key": face_key,
                "reaction": (
                    {"message_id": reaction_details[0], "emoji_id": reaction_details[1]}
                    if reaction_details is not None
                    else None
                ),
            }
        }
        if generation_metadata is not None:
            metadata["generation_metadata"] = _safe_generation_metadata(generation_metadata)
        event = ConversationEvent(
            event_id=event_id,
            platform_event_id=None,
            platform_message_id=None,
            conversation_id=conversation_id,
            sequence=0,
            direction="outbound",
            actor="qichi",
            kind="text",
            text=result.text,
            message_segments=segments,
            reply_to_event_id=reply_target.event_id if reply_target else None,
            reply_to_platform_message_id=self._platform_message_id(reply_target) if reply_target else None,
            occurred_at_utc=occurred_at_utc,
            received_at_utc=occurred_at_utc,
            status="pending",
            metadata=metadata,
        )
        return event

    def _validate_existing_event(
        self, existing: ConversationEvent, conversation_id: str, result: DialogueResult
    ) -> None:
        expression = existing.metadata.get("expression")
        if (
            existing.direction != "outbound"
            or existing.actor != "qichi"
            or existing.kind != "text"
            or existing.conversation_id != conversation_id
            or existing.text != result.text
            or not isinstance(expression, Mapping)
            or expression.get("requested_reply_target") != result.reply_target
            or expression.get("requested")
            != (result.expression_intent.to_dict() if result.expression_intent else None)
        ):
            raise ValueError("outbound event identity conflict")
        return existing

    def _message_segments(
        self, result: DialogueResult, reply_target: ConversationEvent | None
    ) -> tuple[MessageSegment, ...]:
        segments: list[MessageSegment] = []
        if reply_target is not None:
            platform_message_id = self._platform_message_id(reply_target)
            if platform_message_id is not None:
                segments.append(MessageSegment("reply", {"id": platform_message_id}))
        segments.append(MessageSegment("text", {"text": result.text}))
        intent = result.expression_intent
        if intent is not None and intent.kind == "face":
            face_id = self._decimal_id(self.face_catalog.get(intent.key))
            if face_id is not None:
                segments.append(MessageSegment("face", {"id": face_id}))
        return tuple(segments)

    def _reply_target(self, conversation_id: str, handle: str | None) -> ConversationEvent | None:
        if handle is None:
            return None
        try:
            target = self.events.resolve_handle(conversation_id, handle)
        except (KeyError, ValueError):
            return None
        return target if self._platform_message_id(target) is not None else None

    def _platform_message_id(self, event: ConversationEvent | None) -> str | None:
        if event is None:
            return None
        platform_message_id = self._decimal_id(event.platform_message_id)
        if platform_message_id is not None:
            return platform_message_id
        row = self.database.connection.execute(
            "SELECT platform_message_id FROM platform_message_map WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        return self._decimal_id(row["platform_message_id"]) if row is not None else None

    async def dispatch_record(
        self,
        *,
        event: ConversationEvent,
        payload: Mapping[str, object],
        occurred_at_utc: datetime,
    ) -> str:
        """派发一段语音。

        语音不幂等：超时/断连/5xx 一律收成 unknown，且永不自动重试 ——
        重复一条语音比少一条更糟（recoverable_pending 全仓无人调用，
        dispatched 卡住的记录不会重发，这正是语音需要的失败方向）。
        """

        if payload.get("action_kind") != "record":
            raise ValueError("voice payload must carry action_kind=record")
        return await self._dispatch_text(
            self.record_operation_key(event.event_id), event, payload, occurred_at_utc
        )

    async def dispatch_text(
        self,
        *,
        event: ConversationEvent,
        payload: Mapping[str, object],
        occurred_at_utc: datetime,
    ) -> str:
        """按该事件自己的文字幂等键发出（合成失败时的退路、以及被剥下来的颜文字）。"""

        if payload.get("action_kind") != "text":
            raise ValueError("text payload must carry action_kind=text")
        return await self._dispatch_text(
            self.text_operation_key(event.event_id), event, payload, occurred_at_utc
        )

    async def _dispatch_text(
        self,
        operation_key: str,
        event: ConversationEvent,
        payload: Mapping[str, object],
        timestamp: datetime,
        *,
        sent_commit_hook: Callable[[Any], None] | None = None,
    ) -> str:
        record = self.outbox.create_intent(operation_key, event.event_id, payload, timestamp)
        if record.status != "pending":
            return record.status
        try:
            self.outbox.begin_dispatch(operation_key, timestamp)
        except ValueError:
            return self.outbox.get(operation_key).status
        return await self._dispatch_started_text(
            operation_key,
            event,
            payload,
            timestamp,
            sent_commit_hook=sent_commit_hook,
        )

    async def _dispatch_started_text(
        self,
        operation_key: str,
        event: ConversationEvent,
        payload: Mapping[str, object],
        timestamp: datetime,
        *,
        sent_commit_hook: Callable[[Any], None] | None = None,
    ) -> str:
        try:
            response = await self.client.send_private_msg(payload["user_id"], payload["message"])
        except (OneBotTimeoutError, OneBotConnectionError, OneBotHTTPServerError, OneBotProtocolError) as error:
            self._complete_text_terminal(operation_key, event.event_id, "unknown", {"error": type(error).__name__}, timestamp)
            return "unknown"
        except (OneBotActionError, OneBotHTTPClientError, OneBotHTTPRedirectError) as error:
            self._complete_text_terminal(operation_key, event.event_id, "failed", {"error": type(error).__name__}, timestamp)
            return "failed"
        try:
            platform_message_id = self._message_id(response)
        except ValueError:
            self._complete_text_terminal(operation_key, event.event_id, "unknown", {"error": "invalid_send_response"}, timestamp)
            return "unknown"
        self._complete_text_success(
            operation_key,
            event.event_id,
            platform_message_id,
            response,
            timestamp,
            sent_commit_hook=sent_commit_hook,
        )
        return "sent"

    def _prepare_guarded_text(
        self,
        operation_key: str,
        candidate: ConversationEvent,
        result: DialogueResult,
        owner_qq: str,
        timestamp: datetime,
        dispatch_guard: Callable[[Any], bool],
    ) -> tuple[ConversationEvent, Mapping[str, object], str, bool] | None:
        with self.database.transaction() as connection:
            allowed = dispatch_guard(connection)
            if type(allowed) is not bool:
                raise TypeError("dispatch_guard must return a bool")
            if not allowed:
                return None

            existing = connection.execute(
                "SELECT event_id FROM conversation_events WHERE event_id = ?",
                (candidate.event_id,),
            ).fetchone()
            if existing is None:
                event, created = self.events.insert_in_transaction(connection, candidate)
                if not created:
                    raise ValueError("guarded outbound event was unexpectedly deduplicated")
            else:
                event = self.events.get(candidate.event_id)
                self._validate_existing_event(event, candidate.conversation_id, result)

            payload = {
                "action_kind": "text",
                "user_id": owner_qq,
                "message": [segment.to_dict() for segment in event.message_segments],
            }
            payload_json = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            row = connection.execute(
                "SELECT event_id, payload_json, status FROM outbox WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO outbox (operation_key, event_id, payload_json, status, "
                    "attempt_count, response_json, created_at_utc, updated_at_utc) "
                    "VALUES (?, ?, ?, 'pending', 0, NULL, ?, ?)",
                    (
                        operation_key,
                        event.event_id,
                        payload_json,
                        timestamp.isoformat(),
                        timestamp.isoformat(),
                    ),
                )
                status = "pending"
            else:
                if row["event_id"] != event.event_id or row["payload_json"] != payload_json:
                    raise ValueError("operation_key immutable intent identity conflict")
                status = row["status"]
            dispatch_authorized = False
            if status == "pending":
                cursor = connection.execute(
                    "UPDATE outbox SET status = 'dispatched', "
                    "attempt_count = attempt_count + 1, updated_at_utc = ? "
                    "WHERE operation_key = ? AND status = 'pending'",
                    (timestamp.isoformat(), operation_key),
                )
                if cursor.rowcount != 1:
                    raise ValueError("cannot begin guarded dispatch")
                status = "dispatched"
                dispatch_authorized = True
            return event, payload, status, dispatch_authorized

    async def _dispatch_stored_reaction(self, event: ConversationEvent, timestamp: datetime) -> None:
        expression = event.metadata.get("expression")
        if not isinstance(expression, Mapping) or not isinstance(expression.get("reaction"), Mapping):
            return
        reaction = expression["reaction"]
        platform_message_id = self._decimal_id(reaction.get("message_id"))
        emoji_id = self._decimal_id(reaction.get("emoji_id"))
        if platform_message_id is None or emoji_id is None:
            return
        operation_key = self.reaction_operation_key(event.event_id)
        payload = {
            "action_kind": "reaction",
            "message_id": platform_message_id,
            "emoji_id": str(emoji_id),
            "set": True,
        }
        record = self.outbox.create_intent(operation_key, event.event_id, payload, timestamp)
        if record.status != "pending":
            return
        try:
            self.outbox.begin_dispatch(operation_key, timestamp)
        except ValueError:
            return
        try:
            response = await self.client.set_msg_emoji_like(platform_message_id, emoji_id, set=True)
        except (OneBotTimeoutError, OneBotConnectionError, OneBotHTTPServerError, OneBotProtocolError) as error:
            self.outbox.complete_dispatch(operation_key, "unknown", {"error": type(error).__name__}, timestamp)
        except (OneBotActionError, OneBotHTTPClientError, OneBotHTTPRedirectError) as error:
            self.outbox.complete_dispatch(operation_key, "failed", {"error": type(error).__name__}, timestamp)
        else:
            if not isinstance(response, Mapping):
                self.outbox.complete_dispatch(operation_key, "unknown", {"error": "invalid_reaction_response"}, timestamp)
            else:
                self.outbox.complete_dispatch(operation_key, "sent", response, timestamp)

    def _complete_text_success(
        self,
        operation_key: str,
        event_id: str,
        platform_message_id: str,
        response: object,
        timestamp: datetime,
        *,
        sent_commit_hook: Callable[[Any], None] | None = None,
    ) -> None:
        collision = False
        with self.database.transaction() as connection:
            mapping = connection.execute(
                "SELECT event_id FROM platform_message_map WHERE platform_message_id = ?",
                (platform_message_id,),
            ).fetchone()
            reverse_mapping = connection.execute(
                "SELECT platform_message_id FROM platform_message_map WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if (
                (mapping is not None and mapping["event_id"] != event_id)
                or (
                    reverse_mapping is not None
                    and reverse_mapping["platform_message_id"] != platform_message_id
                )
            ):
                collision = True
                collision_outbox = connection.execute(
                    "UPDATE outbox SET status = ?, response_json = ?, updated_at_utc = ? "
                    "WHERE operation_key = ? AND status = ?",
                    ("unknown", json.dumps({"error": "platform_message_map_collision"}), timestamp.isoformat(), operation_key, "dispatched"),
                )
                collision_event = connection.execute(
                    "UPDATE conversation_events SET status = ? WHERE event_id = ?",
                    ("unknown", event_id),
                )
                if collision_outbox.rowcount != 1 or collision_event.rowcount != 1:
                    raise ValueError("cannot fail closed after platform message mapping collision")
            else:
                if mapping is None:
                    connection.execute(
                        "INSERT INTO platform_message_map (platform_message_id, event_id, source, created_at_utc) "
                        "VALUES (?, ?, ?, ?)",
                        (platform_message_id, event_id, "sender", timestamp.isoformat()),
                    )
                cursor = connection.execute(
                    "UPDATE outbox SET status = ?, response_json = ?, updated_at_utc = ? "
                    "WHERE operation_key = ? AND status = ?",
                    ("sent", json.dumps(response, ensure_ascii=False, separators=(",", ":")), timestamp.isoformat(), operation_key, "dispatched"),
                )
                if cursor.rowcount != 1:
                    raise ValueError("cannot complete text dispatch")
                event_cursor = connection.execute(
                    "UPDATE conversation_events SET status = ? WHERE event_id = ?",
                    ("sent", event_id),
                )
                if event_cursor.rowcount != 1:
                    raise KeyError(event_id)
                if sent_commit_hook is not None:
                    sent_commit_hook(connection)
        if collision:
            raise ValueError("platform message mapping collision")

    @staticmethod
    def _message_id(response: object) -> str:
        if not isinstance(response, Mapping):
            raise ValueError("send response must be a mapping")
        message_id = response.get("message_id")
        decimal = Sender._decimal_id(message_id)
        if decimal is not None:
            return decimal
        raise ValueError("send response message_id is invalid")

    def _complete_text_terminal(
        self, operation_key: str, event_id: str, status: str, response: object, timestamp: datetime
    ) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE outbox SET status = ?, response_json = ?, updated_at_utc = ? "
                "WHERE operation_key = ? AND status = ?",
                (status, json.dumps(response, ensure_ascii=False, separators=(",", ":")), timestamp.isoformat(), operation_key, "dispatched"),
            )
            if cursor.rowcount != 1:
                raise ValueError("cannot complete text dispatch")
            event_cursor = connection.execute(
                "UPDATE conversation_events SET status = ? WHERE event_id = ?",
                (status, event_id),
            )
            if event_cursor.rowcount != 1:
                raise KeyError(event_id)

    def _reaction_details(
        self, conversation_id: str, intent: ExpressionIntent | None
    ) -> tuple[str, str] | None:
        if intent is None or intent.kind != "reaction" or intent.target_event_handle is None:
            return None
        target = self._reply_target(conversation_id, intent.target_event_handle)
        platform_message_id = self._platform_message_id(target)
        emoji_id = self._decimal_id(self.reaction_catalog.get(intent.key))
        if platform_message_id is None or emoji_id is None:
            return None
        return platform_message_id, emoji_id

    @staticmethod
    def _decimal_id(value: object) -> str | None:
        if type(value) is int and value >= 0:
            return str(value)
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            return value
        return None
