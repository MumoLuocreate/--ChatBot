from __future__ import annotations

from pathlib import Path
from dataclasses import FrozenInstanceError

import pytest

from qichi.config import ConfigError, load_config


def test_vision_switch_and_limits_match_the_shipped_configuration(
    config_path: Path, complete_environment: dict[str, str]
):
    """开关由用户 2026-09-11 裁定打开；上限与保留策略同为用户裁定。"""

    config = load_config(config_path, environ=complete_environment)

    assert config.vision.enabled is True
    assert config.vision.max_images_per_turn == 2
    assert config.vision.max_image_bytes == 5_242_880
    assert config.vision.download_timeout_seconds == 10
    assert config.vision.keep_original_images is True
    assert config.vision.max_total_media_bytes == 200 * 1024 * 1024
    assert config.vision.max_media_age_days == 90


def test_load_example_config_from_environment(config_path: Path, complete_environment: dict[str, str]):
    config = load_config(config_path, environ=complete_environment)

    assert config.app.timezone == "Asia/Shanghai"
    assert config.app.owner_qq_env == "QICHI_OWNER_QQ"
    assert config.storage.database_path == Path("data/qichi.sqlite3")
    assert config.app.owner_qq == "123456"
    assert config.llm.api_key == "test-deepseek-key"
    assert config.llm.provider == "deepseek"
    assert config.llm.base_url == "https://api.deepseek.com/v1"
    # 2026-09-16 起示例配置的主模型是 flash（用户裁定「全换 f」），思考跟随供应商默认。
    assert config.llm.primary.model == "deepseek-v4-flash"
    assert config.llm.primary.thinking == "default"
    assert config.llm.api_key_env == "DEEPSEEK_API_KEY"
    assert config.llm.primary.temperature == 0.5
    assert config.llm.primary.top_p == 0.92
    assert config.llm.primary.max_output_tokens == 6144
    assert config.llm.primary.max_visible_output_tokens == 1200
    # 出厂值 2026-09-14 从 40 提到 90（主动开口开思考后实测生成 23~56 秒）。
    assert config.llm.primary.timeout_seconds == 90
    assert config.llm.primary.required_min_context_tokens == 262144
    assert not hasattr(config.llm, "fallback")
    assert not hasattr(config.llm, "circuit_breaker")
    assert config.llm.normal_reply_calls == 1
    assert config.transport.adapter == "napcat_onebot11"
    assert config.transport.websocket_url_env == "NAPCAT_WS_URL"
    assert config.transport.http_url_env == "NAPCAT_HTTP_URL"
    assert config.transport.access_token_env == "NAPCAT_ACCESS_TOKEN"
    assert config.transport.persist_before_process is True
    assert config.transport.quote_context is True
    assert config.transport.quote_resolution.local_first is True
    assert config.transport.quote_resolution.fetch_missing_with_get_msg is True
    assert config.transport.quote_resolution.persist_resolved_snapshot is True
    assert config.transport.quote_resolution.include_inbound_and_outbound_links is True
    assert config.transport.quote_resolution.unresolved_must_be_explicit is True
    assert config.transport.local_chat_history.enabled is True
    assert config.transport.local_chat_history.preserve_original_text is True
    assert config.transport.local_chat_history.preserve_message_segments is True
    assert config.dialogue.context_window_preferred_tokens == 262144
    assert config.dialogue.context_window_max_tokens == 262144
    assert config.dialogue.output_reserve_tokens == 6144
    assert config.dialogue.recent_history_budget_tokens == 16384
    assert config.dialogue.require_provider_context_support is True
    assert config.dialogue.context_strategy == "raw_history_first"
    assert config.dialogue.adaptive_expansion is False
    assert config.dialogue.pin_current_input is True
    assert config.dialogue.pin_quoted_target is True
    assert config.dialogue.pin_active_relationship_state is True
    assert config.dialogue.merge_window_ms == 700
    assert config.dialogue.merge_window_max_ms == 2200
    assert config.dialogue.per_conversation_serial is True
    assert config.dialogue.natural_pause_ms == 0
    assert not hasattr(config.dialogue, "semantic_reviewer")
    assert not hasattr(config.dialogue, "state_classifier")
    assert not hasattr(config.dialogue, "phrase_pool")
    assert not hasattr(config.dialogue, "response_self_check_json")
    assert config.dialogue.structural_retry_limit == 1
    assert config.dialogue.llm_decides_reply_target is True
    assert config.dialogue.auto_quote_current_message is True
    assert config.transport.quote_resolution.fetch_missing_with_get_msg is True
    assert config.transport.local_chat_history.retention_days is None
    assert config.persona.prompt_mode == "thin"
    assert config.persona.system_prompt_file == Path("doc/运行时角色核心.example.md")
    assert config.persona.reference_files == (
        Path("doc/人设-角色.md"),
        Path("doc/用户画像.md"),
        Path("doc/角色外貌设定.md"),
    )
    assert config.persona.inject_reference_files_verbatim is False
    assert config.expression.unicode_emoji.enabled is True
    assert config.expression.unicode_emoji.selection == "model_native"
    assert config.expression.qq_face.enabled is False
    assert config.expression.qq_face.selection == "same_generation_intent"
    assert config.expression.qq_face.runtime_catalog == "data/qq-expression-catalog.json"
    assert config.expression.message_reaction.enabled is False
    assert config.expression.message_reaction.selection == "same_generation_intent"
    assert config.expression.message_reaction.idempotent_set is True
    assert config.expression.custom_sticker.enabled is False
    assert config.expression.custom_sticker.selection is None
    assert config.expression.intent_protocol == "standalone_control_line"
    assert config.expression.max_native_actions_per_turn == 1
    assert config.expression.random_frequency is False
    assert config.expression.fixed_turn_interval is False
    assert config.expression.replace_text_reply is False
    assert config.expression.persist_action_metadata is True
    assert config.memory.write_candidates_async is True
    assert config.memory.require_source_quote is True
    assert config.memory.require_source_message_id is True
    assert config.memory.require_source_actor is True
    assert config.memory.include_evidence_in_context is True
    assert config.memory.auto_commit == "verified_explicit"
    assert config.memory.always_include_types == ("agreement", "correction")
    assert config.memory.retrieval_candidate_top_k == 24
    assert config.memory.episodic_top_k == 12
    assert config.memory.lifecycle_states == ("candidate", "active", "superseded", "rejected", "expired")
    assert config.memory.correction_precedence is True
    assert config.memory.ambiguity_stays_candidate is True
    assert config.memory.extraction_mode == "async_session_consolidation_v2"
    assert config.memory.extraction_idle_minutes == 30
    assert config.memory.extractor_must_reference_event_ids is True
    assert config.memory.uncertain_as_fact is False
    assert config.memory.generated_text_as_user_evidence is False
    assert config.memory.recent_history_from_raw_events is True
    assert config.boundary.inject_dynamic_capability_manifest is True
    assert config.boundary.label_all_fact_sources is True
    assert config.boundary.distinguish_past_from_current is True
    assert config.boundary.distinguish_user_fact_from_assistant_history is True
    assert config.boundary.unavailable_capabilities_are_explicit is True
    assert config.boundary.llm_owns_semantic_expression is True
    assert not hasattr(config.boundary, "semantic_post_rewrite")
    assert not hasattr(config.boundary, "keyword_body_filters")
    assert not hasattr(config.boundary, "online_reviewer")
    assert config.decision_authority.llm == (
        "reply_content",
        "tone_and_length",
        "relationship_expression",
        "memory_relevance",
        "optional_quote_target",
        "emoji_and_qq_expression",
        "initiative_topic_or_skip",
    )
    assert config.decision_authority.code == (
        "exact_message_identity",
        "quote_resolution",
        "fact_provenance",
        "current_time",
        "available_platform_capabilities",
        "persistence_order",
        "idempotency_and_delivery_state",
        "privacy_and_owner_scope",
    )
    assert config.initiative.enabled is True
    # 2026-09-19 体验优化：实测 allow_model_to_skip 空转（115 次尝试 0 次跳过），
    # 「两条就不发了」完全是这个上限造成的。间隔与日上限是 InitiativePolicy 冻结的
    # （60 分钟、必须保持未设置），只有上限这一项可以放。
    assert config.initiative.idle_attempt_minutes == 60
    assert config.initiative.max_unanswered_attempts == 5
    assert config.initiative.reset_on_user_message is True
    assert config.initiative.use_same_dialogue_engine is True
    assert config.initiative.allow_model_to_skip is True
    assert config.decision_authority.llm[-1] == "initiative_topic_or_skip"
    assert config.initiative.daily_send_limit is None
    assert config.initiative.quiet_hours_local == "23:00-08:00"
    assert config.initiative.cancel_if_conversation_changed is True
    assert config.initiative.unknown_delivery_retry is False
    assert config.output_guard.reject_empty is True
    assert config.output_guard.reject_internal_prompt_leak is True
    assert config.output_guard.reject_raw_protocol_payload is True
    assert not hasattr(config.output_guard, "semantic_keyword_filters")
    assert not hasattr(config.output_guard, "rewrite_sentences_in_place")
    assert config.observability.redact_secrets is True
    assert config.observability.log_level == "INFO"
    assert config.observability.log_context_sources is True
    assert config.observability.log_model_route is True
    assert config.observability.log_latency is True
    assert not hasattr(config.observability, "online_style_scoring")


def test_missing_environment_variable_is_rejected(config_path: Path, complete_environment: dict[str, str]):
    del complete_environment["DEEPSEEK_API_KEY"]

    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        load_config(config_path, environ=complete_environment)


def test_expression_protocol_legacy_alias_is_normalized(
    config_path: Path,
    complete_environment: dict[str, str],
    tmp_path: Path,
):
    source = config_path.read_text(encoding="utf-8")
    legacy = source.replace(
        "intent_protocol: standalone_control_line",
        "intent_protocol: trailing_control_token",
    )
    path = tmp_path / "legacy-protocol.yaml"
    path.write_text(legacy, encoding="utf-8")

    config = load_config(path, environ=complete_environment)

    assert config.expression.intent_protocol == "standalone_control_line"


@pytest.mark.parametrize("value", ["json", "", "standalone_control_token"])
def test_unknown_expression_protocol_is_rejected(
    config_path: Path,
    complete_environment: dict[str, str],
    tmp_path: Path,
    value: str,
):
    source = config_path.read_text(encoding="utf-8")
    configured = source.replace(
        "intent_protocol: standalone_control_line",
        f"intent_protocol: {value!r}",
    )
    path = tmp_path / "invalid-protocol.yaml"
    path.write_text(configured, encoding="utf-8")

    with pytest.raises(ConfigError, match="expression.intent_protocol"):
        load_config(path, environ=complete_environment)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_url", "https://api.siliconflow.cn/v1", "base_url"),
        ("api_key_env", "SILICONFLOW_API_KEY", "api_key_env"),
    ],
)
def test_provider_endpoint_and_key_environment_must_be_same_source(
    config_path: Path,
    complete_environment: dict[str, str],
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
):
    source = config_path.read_text(encoding="utf-8")
    anchor = "  base_url: https://api.deepseek.com/v1" if field == "base_url" else "  api_key_env: DEEPSEEK_API_KEY"
    configured = source.replace(anchor, f"  {field}: {replacement}")
    path = tmp_path / "config.yaml"
    path.write_text(configured, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path, environ=complete_environment)


def test_missing_owner_environment_fails_closed(config_path: Path, complete_environment: dict[str, str]):
    del complete_environment["QICHI_OWNER_QQ"]

    with pytest.raises(ConfigError, match="QICHI_OWNER_QQ"):
        load_config(config_path, environ=complete_environment)


def test_owner_private_only_must_be_strict_true(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace("owner_private_only: true", "owner_private_only: false"), encoding="utf-8")

    with pytest.raises(ConfigError, match="owner_private_only"):
        load_config(path, environ=complete_environment)


def test_unknown_runtime_field_is_rejected(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source + "\nunknown_runtime_block: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown_runtime_block"):
        load_config(path, environ=complete_environment)


def test_preferred_window_must_not_exceed_max(config_path: Path, complete_environment: dict[str, str], tmp_path: Path):
    source = config_path.read_text(encoding="utf-8")
    broken = source.replace("context_window_preferred_tokens: 262144", "context_window_preferred_tokens: 262145")
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="preferred.*max"):
        load_config(path, environ=complete_environment)


def test_output_reserve_must_leave_room_in_preferred_window(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    broken = source.replace("output_reserve_tokens: 6144", "output_reserve_tokens: 262144")
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="output reserve"):
        load_config(path, environ=complete_environment)


def test_output_reserve_must_cover_provider_completion_budget(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    broken = source.replace("output_reserve_tokens: 6144", "output_reserve_tokens: 6143")
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="cover.*max_output_tokens"):
        load_config(path, environ=complete_environment)


def test_visible_output_limit_must_fit_provider_completion_budget(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    broken = source.replace(
        "max_visible_output_tokens: 1200", "max_visible_output_tokens: 6145"
    )
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="max_visible_output_tokens.*max_output_tokens"):
        load_config(path, environ=complete_environment)


def test_model_minimum_context_must_cover_preferred_window(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    broken = source.replace("required_min_context_tokens: 262144", "required_min_context_tokens: 65536")
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="required_min_context_tokens"):
        load_config(path, environ=complete_environment)


def test_memory_auto_commit_policy_is_explicitly_bounded(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace("auto_commit: verified_explicit", "auto_commit: unsafe_guess"), encoding="utf-8")

    with pytest.raises(ConfigError, match="memory.auto_commit"):
        load_config(path, environ=complete_environment)


def test_fallback_model_configuration_is_rejected(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    configured = source.replace(
        "  normal_reply_calls: 1",
        "  fallback:\n    enabled: false\n    model: deepseek-ai/DeepSeek-V3.2\n  normal_reply_calls: 1",
    )
    path = tmp_path / "config.yaml"
    path.write_text(configured, encoding="utf-8")

    with pytest.raises(ConfigError, match=r"unknown configuration field\(s\).*fallback"):
        load_config(path, environ=complete_environment)


@pytest.mark.parametrize(
    ("anchor", "legacy_field"),
    [
        ("  structural_retry_limit: 1", "semantic_reviewer"),
        ("  llm_owns_semantic_expression: true", "online_reviewer"),
        ("  reject_raw_protocol_payload: true", "semantic_keyword_filters"),
        ("  log_latency: true", "online_style_scoring"),
    ],
)
def test_removed_semantic_mechanism_flags_are_rejected(
    config_path: Path,
    complete_environment: dict[str, str],
    tmp_path: Path,
    anchor: str,
    legacy_field: str,
):
    source = config_path.read_text(encoding="utf-8")
    indent = anchor[: len(anchor) - len(anchor.lstrip())]
    configured = source.replace(anchor, f"{indent}{legacy_field}: true\n{anchor}")
    path = tmp_path / "config.yaml"
    path.write_text(configured, encoding="utf-8")

    with pytest.raises(ConfigError, match=legacy_field):
        load_config(path, environ=complete_environment)


def test_only_verified_v4_identifiers_can_be_configured(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    """2026-09-12 起 pro 也是已验证标识（分流用），但没验证过的一律拒绝。"""

    shipped = load_config(config_path, environ=complete_environment)
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        source.replace(f"    model: {shipped.llm.primary.model}", "    model: deepseek-v3.2", 1),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="verified DeepSeek V4 identifier"):
        load_config(path, environ=complete_environment)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("temperature: 0.5", 'temperature: "hot"', "temperature"),
        ("temperature: 0.5", "temperature: -0.1", "temperature"),
        ("top_p: 0.92", "top_p: 0", "top_p"),
        ("top_p: 0.92", "top_p: 1.1", "top_p"),
        ("max_output_tokens: 6144", 'max_output_tokens: "6144"', "max_output_tokens"),
        ("max_visible_output_tokens: 1200", 'max_visible_output_tokens: "1200"', "max_visible_output_tokens"),
        # 锚点跟着出厂值走（2026-09-14：40 -> 90）。
        ("timeout_seconds: 90", "timeout_seconds: 0", "timeout_seconds"),
        ("timeout_seconds: 90", 'timeout_seconds: "90"', "timeout_seconds"),
    ],
)
def test_invalid_model_numeric_values_are_rejected(
    config_path: Path,
    complete_environment: dict[str, str],
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path, environ=complete_environment)


def test_invalid_boolean_type_is_rejected(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace("persist_before_process: true", 'persist_before_process: "true"'), encoding="utf-8")

    with pytest.raises(ConfigError, match="persist_before_process"):
        load_config(path, environ=complete_environment)


def test_inline_secret_is_rejected(config_path: Path, complete_environment: dict[str, str], tmp_path: Path):
    source = config_path.read_text(encoding="utf-8")
    broken = source.replace("api_key_env: DEEPSEEK_API_KEY", "api_key: hard-coded-secret")
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="secret.*environment"):
        load_config(path, environ=complete_environment)


def test_repr_redacts_secret(config_path: Path, complete_environment: dict[str, str]):
    config = load_config(config_path, environ=complete_environment)
    rendered = repr(config)

    assert "test-deepseek-key" not in rendered
    assert "test-napcat-token" not in rendered
    assert "123456" not in rendered
    assert "REDACTED" in rendered


def test_all_settings_are_frozen(config_path: Path, complete_environment: dict[str, str]):
    config = load_config(config_path, environ=complete_environment)

    with pytest.raises(FrozenInstanceError):
        config.dialogue.context_window_max_tokens = 1


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_temperature_is_rejected(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path, value: str
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace("temperature: 0.5", f"temperature: {value}"), encoding="utf-8")

    with pytest.raises(ConfigError, match="temperature"):
        load_config(path, environ=complete_environment)


def test_merge_window_order_is_rejected(
    config_path: Path, complete_environment: dict[str, str], tmp_path: Path
):
    source = config_path.read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace("merge_window_ms: 700", "merge_window_ms: 2300"), encoding="utf-8")

    with pytest.raises(ConfigError, match="merge_window_ms"):
        load_config(path, environ=complete_environment)

# --- 语音配置（TTS P1-5，见 doc/TTS-实施计划-20260914.md §3.6）---

SHIPPED_CONFIG = Path(__file__).parents[1] / "config.example.yaml"


def voice_config(tmp_path, replacements: "dict[str, str]") -> Path:
    text = SHIPPED_CONFIG.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert text.count(old) == 1, "anchor must be unique: %r" % old
        text = text.replace(old, new, 1)
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def toggle_voice(source: str, enabled: bool) -> str:
    """只动 voice 段的 enabled。

    2026-09-14：出厂配置上线后 enabled 就是 true（回滚靠改回 false），所以测试不能再假设
    它的初值，也不能用整串替换 —— net、image_search 那些段里也有 enabled。
    """

    lines = source.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("voice:"))
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith((" ", "\t")):
            break
        if line.startswith("  enabled:"):
            lines[index] = "  enabled: %s\n" % ("true" if enabled else "false")
            return "".join(lines)
    raise AssertionError("voice section has no enabled line")


def with_voice(tmp_path, enabled: bool, extra=None):
    replacements = dict(extra or {})
    path = voice_config(tmp_path, replacements)
    path.write_text(toggle_voice(path.read_text(encoding="utf-8"), enabled), encoding="utf-8")
    return path


def enable_voice(tmp_path, extra=None):
    return with_voice(tmp_path, True, extra)


def disable_voice(tmp_path, extra=None):
    return with_voice(tmp_path, False, extra)


def test_voice_defaults_to_disabled_and_needs_no_api_key(tmp_path, complete_environment):
    path = disable_voice(tmp_path)
    settings = load_config(path, environ=complete_environment).voice

    assert settings.enabled is False
    assert settings.api_key == "", "关着的时候不该因为少一个 key 而起不来"
    assert settings.model and settings.voice_id, "冻结的音色身份应当写在配置里"
    assert settings.max_chars_per_utterance == 120
    assert settings.cache_dir == Path("data/voice")
    assert settings.cache_max_age_days == 30


def test_enabling_voice_requires_the_key(tmp_path, complete_environment):
    path = enable_voice(tmp_path)
    without = {name: value for name, value in complete_environment.items() if name != "DASHSCOPE_API_KEY"}

    with pytest.raises(ConfigError, match="DASHSCOPE_API_KEY"):
        load_config(path, environ=without)

    env = {**complete_environment, "DASHSCOPE_API_KEY": "secret"}
    settings = load_config(path, environ=env).voice
    assert settings.enabled is True and settings.api_key == "secret"


def test_voice_refuses_an_unverified_endpoint_or_key_name(tmp_path, complete_environment):
    env = {**complete_environment, "DASHSCOPE_API_KEY": "secret"}
    for old, new, match in (
        ("base_url: https://dashscope.aliyuncs.com/api/v1", "base_url: https://example.invalid/v1", "base_url"),
        ("api_key_env: DASHSCOPE_API_KEY", "api_key_env: SOMETHING_ELSE", "api_key_env"),
    ):
        path = voice_config(tmp_path, {old: new})
        with pytest.raises(ConfigError, match=match):
            load_config(path, environ=env)


def test_voice_refuses_an_inline_api_key(tmp_path, complete_environment):
    path = voice_config(tmp_path, {
        "  api_key_env: DASHSCOPE_API_KEY": "  api_key_env: DASHSCOPE_API_KEY\n  api_key: sk-inline",
    })

    with pytest.raises(ConfigError, match="secret"):
        load_config(path, environ=complete_environment)


def test_voice_enabled_without_a_frozen_voice_is_refused(tmp_path, complete_environment):
    path = enable_voice(tmp_path, {
        "  voice_id: qwen-tts-vd-qichi_cast2-voice-20260914182133631-9160": "  voice_id: \"\"",
    })
    env = {**complete_environment, "DASHSCOPE_API_KEY": "secret"}

    with pytest.raises(ConfigError, match="voice_id"):
        load_config(path, environ=env)


def test_config_repr_never_leaks_the_voice_key(tmp_path, complete_environment):
    path = enable_voice(tmp_path)
    env = {**complete_environment, "DASHSCOPE_API_KEY": "sk-voice-secret"}

    assert "sk-voice-secret" not in repr(load_config(path, environ=env))

