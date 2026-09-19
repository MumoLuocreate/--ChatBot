"""对话可调用的外部工具：声明 + 执行。

分工（2026-09-13 用户裁定后定稿）：
  * 代码只负责**声明**（名字、参数、能查什么、不能查什么）与**执行**（校验、限额、超时、标注来源）；
  * **要不要查**是模型的事——它自己决定调用哪个工具、query 写什么；
  * **能不能查**是策略的事，由调用方按轮次形状决定给不给工具：
      新图到达轮不给（她必须先问用户）、宽限轮给、纯文字轮给、主动消息轮不给。

工具结果永远渲染成「外部、不可信、带来源与时间戳」的资料块，绝不写进记忆。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from qichi.dialogue.llm_client import ToolCall
from qichi.net.image_search import SerpApiLensClient, WebImageSearchError, render_image_block
from qichi.net.search import TavilySearchClient, WebSearchError, render_external_block

import logging

_LOGGER = logging.getLogger(__name__)

WEB_SEARCH_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "查一个你不知道、但网上查得到的事实。用户明确要你查、或者你确实不知道时才能用。"
            "图里认不出的东西不要用这个工具——先问用户要不要你去查。"
            "不要把用户的原话、隐私或成人内容放进 query。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查的那件事，尽量短而具体（不超过 200 字）"},
            },
            "required": ["query"],
        },
    },
}

IMAGE_SEARCH_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "image_search",
        "description": (
            "用用户刚发的那张图去搜它是什么（角色、牌子、出处、同款）。不需要参数。"
            "只有用户已经同意你去查的时候才能用；看到图的那一轮不要用——先问。"
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def tool_plan(
    *, enabled: bool, source: str, fresh_images: int, carried_images: int
) -> tuple[bool, bool]:
    """这一轮给不给工具、要不要带图搜工具（纯函数，好测）。

    规则（2026-09-13 用户裁定）：
      * 新图到达轮**不给任何工具**——她必须先问用户要不要去查；
      * 宽限轮（图来自本地存档）给工具，带上图搜；
      * 纯文字轮给工具；
      * 主动消息轮不给（不主动查）。
    """

    if not enabled or source == "initiative":
        return (False, False)
    if fresh_images:
        return (False, False)
    return (True, bool(carried_images))


@dataclass(frozen=True)
class ToolRunResult:
    name: str
    query: str | None
    block: str
    ok: bool
    degraded_reason: str | None
    elapsed_ms: int


class SearchToolRunner:
    def __init__(
        self,
        text_client: TavilySearchClient | None = None,
        image_client: SerpApiLensClient | None = None,
    ) -> None:
        self._text = text_client
        self._image = image_client

    def declares(self, *, with_image: bool) -> tuple[Mapping[str, Any], ...]:
        """这一轮声明哪些工具。没有客户端就不声明（开关关着时什么都不变）。"""

        declared: list[Mapping[str, Any]] = []
        if self._text is not None:
            declared.append(WEB_SEARCH_TOOL)
        if with_image and self._image is not None:
            declared.append(IMAGE_SEARCH_TOOL)
        return tuple(declared)

    async def run(
        self,
        call: ToolCall,
        *,
        image_path: Path | None,
        now: datetime,
    ) -> ToolRunResult:
        """执行一次工具调用，永远返回一个可以喂给模型的资料块（失败也在块里说清）。"""

        arguments = self._arguments(call)
        if arguments is None:
            return ToolRunResult(call.name, None, self._failure(call.name, "bad_arguments"), False,
                                 "bad_arguments", 0)
        query = arguments.get("query")
        query = query.strip() if isinstance(query, str) and query.strip() else None
        if call.name == "web_search":
            if self._text is None:
                return ToolRunResult(call.name, query, self._failure(call.name, "unavailable"), False,
                                     "unavailable", 0)
            if query is None:
                return ToolRunResult(call.name, None, self._failure(call.name, "missing_query"), False,
                                     "missing_query", 0)
            try:
                outcome = await self._text.search(query)
            except WebSearchError:
                # 调用方式错误（查询词超长/为空、客户端已关）：**绝不能因此弄丢整轮回复**。
                return ToolRunResult(call.name, query, self._failure(call.name, "bad_query"), False,
                                     "bad_query", 0)
            except Exception as error:  # 边界：工具层永不向上抛，否则她这一轮就没有回复
                _LOGGER.error("search tool failed (%s)", type(error).__name__)
                return ToolRunResult(call.name, query, self._failure(call.name, "tool_error"), False,
                                     "tool_error", 0)
            return ToolRunResult(
                call.name, outcome.query, render_external_block(outcome, now=now), outcome.ok,
                outcome.degraded_reason, outcome.elapsed_ms,
            )
        if call.name == "image_search":
            if self._image is None or image_path is None:
                return ToolRunResult(call.name, query, self._failure(call.name, "no_image"), False,
                                     "no_image", 0)
            try:
                outcome = await self._image.search_image(image_path)
            except WebImageSearchError:
                # 存档里的图没了/类型不支持：同样是调用方式问题，照样交回一个诚实的失败块。
                return ToolRunResult(call.name, query, self._failure(call.name, "bad_image"), False,
                                     "bad_image", 0)
            except Exception as error:  # 边界：同上
                _LOGGER.error("image search tool failed (%s)", type(error).__name__)
                return ToolRunResult(call.name, query, self._failure(call.name, "tool_error"), False,
                                     "tool_error", 0)
            return ToolRunResult(
                call.name, query, render_image_block(outcome, now=now), outcome.ok,
                outcome.degraded_reason, outcome.elapsed_ms,
            )
        return ToolRunResult(call.name, query, self._failure(call.name, "unknown_tool"), False,
                             "unknown_tool", 0)

    @staticmethod
    def _arguments(call: ToolCall) -> dict[str, Any] | None:
        try:
            parsed = json.loads(call.arguments or "{}")
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _failure(name: str, reason: str) -> str:
        return (
            "[工具调用失败 | 不是本机事实，也不是查到的内容]\n"
            f"工具：{name}\n原因：{reason}\n"
            "不要用猜测填补，直接说没查到或者换个说法问用户。"
        )
