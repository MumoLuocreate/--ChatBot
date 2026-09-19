"""补录脚本的失败描述：那次它把真正的错误吞掉了（2026-09-17）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qichi.memory.extractor import ExtractionFailure  # noqa: E402

import repair_memory_window as repair  # noqa: E402


def test_a_failure_without_a_message_attribute_still_describes_itself():
    """回归：ExtractionFailure 只有 .reason，没有 .message——旧代码在这里崩过。"""

    failure = ExtractionFailure(
        "llm_error", "LLMRequestError: provider generation failed", ("e1",),
        {'provider_error': 'request'},
    )

    described = repair.describe_failure(failure)

    assert "llm_error" in described
    assert "provider generation failed" in described
    assert "provider_error" in described, "诊断字段要一起露出来，否则查不出真因"


def test_a_failure_without_details_is_still_readable():
    failure = ExtractionFailure("input_error", "duplicate event IDs in fragment", ("e1",))

    described = repair.describe_failure(failure)

    assert described == "input_error / duplicate event IDs in fragment"