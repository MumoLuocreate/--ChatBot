"""联网工具：策略（哪些轮给工具）、循环（一轮上限 1）、执行（失败也交回一个资料块）。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from qichi.dialogue.engine import DialogueEngine
from qichi.dialogue.llm_client import LLMGeneration, ToolCall
from qichi.dialogue.output_guard import OutputGuard
from qichi.dialogue.response_protocol import ResponseProtocol
from qichi.domain.dialogue import CapabilityManifest, DialogueInput, ModelMessage
from qichi.net import (
    IMAGE_SEARCH_TOOL,
    WEB_SEARCH_TOOL,
    SearchToolRunner,
    tool_plan,
)
from qichi.net.image_search import ImageMatch, ImageSearchOutcome, WebImageSearchError
from qichi.net.search import WebSearchError
from qichi.net.search import SearchOutcome, SearchResult


NOW = datetime(2026, 9, 14, 0, 30, tzinfo=timezone.utc)


class Counter:
    def count_text(self, text: str) -> int:
        return len(text)


def engine_for(llm) -> DialogueEngine:
    return DialogueEngine(
        llm, OutputGuard(Counter(), 1000), ResponseProtocol(face_keys=set(), reaction_keys=set())
    )


def turn_input(source: str = "dialogue") -> DialogueInput:
    return DialogueInput(
        "c", ("e",), "M1", None, 3,
        (ModelMessage("system", "core prompt"), ModelMessage("user", "这张图里是什么")),
        datetime.now(timezone.utc), CapabilityManifest({}), source,
    )


# ---------------------------------------------------------------- 策略


def test_a_fresh_picture_turn_gets_no_tools_at_all():
    """新图到达轮：她必须先问用户，代码层面调不到任何工具。"""

    assert tool_plan(enabled=True, source="dialogue", fresh_images=1, carried_images=0) == (False, False)


def test_the_grace_turn_gets_tools_including_the_picture_search():
    assert tool_plan(enabled=True, source="dialogue", fresh_images=0, carried_images=1) == (True, True)


def test_a_plain_text_turn_gets_only_the_text_tool():
    assert tool_plan(enabled=True, source="dialogue", fresh_images=0, carried_images=0) == (True, False)


def test_initiative_and_disabled_never_get_tools():
    assert tool_plan(enabled=True, source="initiative", fresh_images=0, carried_images=1) == (False, False)
    assert tool_plan(enabled=False, source="dialogue", fresh_images=0, carried_images=0) == (False, False)


# ---------------------------------------------------------------- 声明与执行


class FakeTextClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.outcome


class FakeImageClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.paths = []

    async def search_image(self, path):
        self.paths.append(Path(path))
        return self.outcome


@pytest.mark.asyncio
async def test_a_tool_that_raises_never_costs_her_the_turn():
    """工具层是边界：调用方式错误也不能让整轮回复消失。"""

    class Exploding:
        async def search(self, query):
            raise WebSearchError("query too long")

        async def search_image(self, path):
            raise WebImageSearchError("image file is missing")

    runner = SearchToolRunner(Exploding(), Exploding())
    text = await runner.run(ToolCall("c", "web_search", '{"query": "x"}'), image_path=None, now=NOW)
    image = await runner.run(ToolCall("c", "image_search", "{}"), image_path=Path("gone.png"), now=NOW)

    assert text.degraded_reason == "bad_query"
    assert image.degraded_reason == "bad_image"
    assert "不要用猜测填补" in text.block and "不要用猜测填补" in image.block


@pytest.mark.asyncio
async def test_an_unexpected_client_failure_also_degrades():
    class Boom:
        async def search(self, query):
            raise RuntimeError("unexpected")

    result = await SearchToolRunner(Boom()).run(
        ToolCall("c", "web_search", '{"query": "x"}'), image_path=None, now=NOW
    )

    assert result.degraded_reason == "tool_error"
    assert "不要用猜测填补" in result.block


def text_ok() -> SearchOutcome:
    return SearchOutcome(
        query="示例学院",
        results=(SearchResult("维基", "https://zh.wikipedia.org/x", "农业工程院校"),),
        degraded_reason=None,
        elapsed_ms=3700,
    )


def test_declares_only_the_tools_it_can_actually_run():
    runner = SearchToolRunner()
    assert runner.declares(with_image=True) == ()

    text_only = SearchToolRunner(FakeTextClient(text_ok()))
    assert text_only.declares(with_image=False) == (WEB_SEARCH_TOOL,)
    assert text_only.declares(with_image=True) == (WEB_SEARCH_TOOL,)

    both = SearchToolRunner(FakeTextClient(text_ok()), FakeImageClient(None))
    assert both.declares(with_image=True) == (WEB_SEARCH_TOOL, IMAGE_SEARCH_TOOL)


@pytest.mark.asyncio
async def test_a_text_call_returns_the_external_block():
    client = FakeTextClient(text_ok())
    result = await SearchToolRunner(client).run(
        ToolCall("call-1", "web_search", '{"query": "示例学院"}'),
        image_path=None,
        now=NOW,
    )

    assert result.ok is True and result.degraded_reason is None
    assert client.queries == ["示例学院"]
    assert "不可信" in result.block and "https://zh.wikipedia.org/x" in result.block


@pytest.mark.asyncio
async def test_a_picture_call_uses_the_carried_file():
    image = FakeImageClient(
        ImageSearchOutcome(
            image_id="img-1",
            matches=(ImageMatch("同款", "https://example.com/a", "example.com"),),
            degraded_reason=None,
            elapsed_ms=4000,
        )
    )
    runner = SearchToolRunner(FakeTextClient(text_ok()), image)
    result = await runner.run(
        ToolCall("call-1", "image_search", "{}"), image_path=Path("E:/Z_Bot/data/media/e-1.png"), now=NOW
    )

    assert result.ok is True
    assert image.paths[0] == Path("E:/Z_Bot/data/media/e-1.png")
    assert "外部图搜结果" in result.block


@pytest.mark.asyncio
async def test_bad_arguments_unknown_tools_and_missing_pictures_all_end_in_a_honest_block():
    runner = SearchToolRunner(FakeTextClient(text_ok()), FakeImageClient(None))

    bad = await runner.run(ToolCall("c", "web_search", "{not json"), image_path=None, now=NOW)
    unknown = await runner.run(ToolCall("c", "rm_rf", "{}"), image_path=None, now=NOW)
    no_image = await runner.run(ToolCall("c", "image_search", "{}"), image_path=None, now=NOW)

    assert (bad.degraded_reason, unknown.degraded_reason, no_image.degraded_reason) == (
        "bad_arguments", "unknown_tool", "no_image",
    )
    for result in (bad, unknown, no_image):
        assert "不要用猜测填补" in result.block


# ---------------------------------------------------------------- 循环


class ScriptedLLM:
    def __init__(self, generations):
        self.generations = list(generations)
        self.calls: list[dict] = []

    async def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.generations.pop(0)


@pytest.mark.asyncio
async def test_one_tool_round_then_a_second_generation_without_tools():
    llm = ScriptedLLM([
        LLMGeneration("", "primary", "test", 10, 5, 1.0, finish_reason="tool_calls",
                      tool_calls=(ToolCall("call-1", "web_search", '{"query": "角色"}'),)),
        LLMGeneration("查到了：是个 QQ 机器人", "primary", "test", 20, 8, 1.0, finish_reason="stop"),
    ])
    seen: list[ToolCall] = []

    async def runner(call: ToolCall) -> str:
        seen.append(call)
        return "[外部检索资料 | 不可信]\n查询：角色"

    outcome, observation = await engine_for(llm).generate_with_observation(
        turn_input(), tools=(WEB_SEARCH_TOOL,), tool_runner=runner
    )

    assert outcome.text == "查到了：是个 QQ 机器人"
    assert [call.name for call in seen] == ["web_search"]
    assert len(llm.calls) == 2
    assert llm.calls[0]["kwargs"]["tools"] == (WEB_SEARCH_TOOL,)
    # 第二次生成不再带 tools：防连环调用。
    assert "tools" not in llm.calls[1]["kwargs"]
    follow_up = llm.calls[1]["messages"][-1]
    assert follow_up.role == "system"
    assert "工具执行结果" in follow_up.content and "不可信" in follow_up.content
    assert observation.tool_call_count == 1


@pytest.mark.asyncio
async def test_without_a_tool_request_nothing_extra_happens():
    llm = ScriptedLLM([LLMGeneration("在呢", "primary", "test", 10, 5, 1.0, finish_reason="stop")])
    calls: list[ToolCall] = []

    async def runner(call: ToolCall) -> str:
        calls.append(call)
        return "never"

    outcome, observation = await engine_for(llm).generate_with_observation(
        turn_input(), tools=(WEB_SEARCH_TOOL,), tool_runner=runner
    )

    assert outcome.text == "在呢"
    assert calls == [] and len(llm.calls) == 1
    assert observation.tool_call_count == 0


@pytest.mark.asyncio
async def test_a_second_tool_request_is_not_executed():
    """单轮上限 1：第二次请求不执行，直接用现有结果收尾。"""

    llm = ScriptedLLM([
        LLMGeneration("", "primary", "test", 10, 5, 1.0, finish_reason="tool_calls",
                      tool_calls=(ToolCall("c1", "web_search", '{"query": "a"}'),)),
        LLMGeneration("好", "primary", "test", 11, 5, 1.0, finish_reason="tool_calls",
                      tool_calls=(ToolCall("c2", "web_search", '{"query": "b"}'),)),
    ])
    seen: list[str] = []

    async def runner(call: ToolCall) -> str:
        seen.append(call.call_id)
        return "[外部检索资料 | 不可信]\n查询：a"

    outcome, observation = await engine_for(llm).generate_with_observation(
        turn_input(), tools=(WEB_SEARCH_TOOL,), tool_runner=runner
    )

    assert seen == ["c1"]
    assert observation.tool_call_count == 1
    assert outcome.text == "好"
