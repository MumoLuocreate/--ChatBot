from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or unsafe."""


_CANONICAL_INTENT_PROTOCOL = "standalone_control_line"
_INTENT_PROTOCOL_ALIASES = {
    _CANONICAL_INTENT_PROTOCOL: _CANONICAL_INTENT_PROTOCOL,
    "trailing_control_token": _CANONICAL_INTENT_PROTOCOL,
}


@dataclass(frozen=True)
class AppSettings:
    timezone: str
    owner_qq_env: str
    owner_qq: str = field(repr=False)


@dataclass(frozen=True)
class QuoteResolutionSettings:
    local_first: bool
    fetch_missing_with_get_msg: bool
    persist_resolved_snapshot: bool
    include_inbound_and_outbound_links: bool
    unresolved_must_be_explicit: bool


@dataclass(frozen=True)
class LocalChatHistorySettings:
    enabled: bool
    preserve_original_text: bool
    preserve_message_segments: bool
    retention_days: int | None


@dataclass(frozen=True)
class TransportSettings:
    adapter: str
    websocket_url_env: str
    http_url_env: str
    access_token_env: str
    websocket_url: str = field(repr=False)
    http_url: str = field(repr=False)
    access_token: str = field(repr=False)
    owner_private_only: bool
    persist_before_process: bool
    quote_context: bool
    quote_resolution: QuoteResolutionSettings
    local_chat_history: LocalChatHistorySettings


@dataclass(frozen=True)
class StorageSettings:
    database_path: Path


@dataclass(frozen=True)
class PrimaryModelSettings:
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    max_visible_output_tokens: int
    timeout_seconds: int
    required_min_context_tokens: int
    # 2026-09-13：对话是否让模型思考。default = 不传该参数（交给供应商）；
    # enabled / disabled 显式发送。切 pro 时用 disabled：实测思考让一轮从 2.8 秒变成
    # 27 秒，但关掉之后她会抄自己上一轮，所以必须与「复读统计提示」一起用。
    thinking: str = "default"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key_env: str
    api_key: str = field(repr=False)
    primary: PrimaryModelSettings
    normal_reply_calls: int
    # 2026-09-12：分流。primary 是说话用的模型；vision_model 只在这一轮带图时改用它
    # （pro 不支持视觉，flash 支持）；None 表示不分流，行为与之前完全一致。
    vision_model: str | None = None
    # 后台调用（记忆抽取、明细/时间线 pass）用的模型。缺省与 primary 相同；
    # 用户要求「pro 只用于对话」，所以切 pro 时这里显式写 flash，避免额外开销。
    background_model: str = ""


@dataclass(frozen=True)
class PersonaSettings:
    system_prompt_file: Path
    reference_files: tuple[Path, ...]
    inject_reference_files_verbatim: bool
    prompt_mode: str


@dataclass(frozen=True)
class DialogueSettings:
    context_window_preferred_tokens: int
    context_window_max_tokens: int
    output_reserve_tokens: int
    recent_history_budget_tokens: int
    require_provider_context_support: bool
    context_strategy: str
    adaptive_expansion: bool
    pin_current_input: bool
    pin_quoted_target: bool
    pin_active_relationship_state: bool
    merge_window_ms: int
    merge_window_max_ms: int
    per_conversation_serial: bool
    natural_pause_ms: int
    structural_retry_limit: int
    llm_decides_reply_target: bool
    auto_quote_current_message: bool


@dataclass(frozen=True)
class ExpressionChannelSettings:
    enabled: bool
    selection: str | None
    runtime_catalog: str | None = None
    idempotent_set: bool | None = None


@dataclass(frozen=True)
class ExpressionSettings:
    unicode_emoji: ExpressionChannelSettings
    qq_face: ExpressionChannelSettings
    message_reaction: ExpressionChannelSettings
    custom_sticker: ExpressionChannelSettings
    intent_protocol: str
    max_native_actions_per_turn: int
    random_frequency: bool
    fixed_turn_interval: bool
    replace_text_reply: bool
    persist_action_metadata: bool


@dataclass(frozen=True)
class MemorySettings:
    write_candidates_async: bool
    require_source_quote: bool
    require_source_message_id: bool
    require_source_actor: bool
    include_evidence_in_context: bool
    auto_commit: str
    always_include_types: tuple[str, ...]
    retrieval_candidate_top_k: int
    episodic_top_k: int
    lifecycle_states: tuple[str, ...]
    correction_precedence: bool
    ambiguity_stays_candidate: bool
    extraction_mode: str
    extraction_idle_minutes: int
    extractor_must_reference_event_ids: bool
    uncertain_as_fact: bool
    generated_text_as_user_evidence: bool
    recent_history_from_raw_events: bool


@dataclass(frozen=True)
class BoundarySettings:
    inject_dynamic_capability_manifest: bool
    label_all_fact_sources: bool
    distinguish_past_from_current: bool
    distinguish_user_fact_from_assistant_history: bool
    unavailable_capabilities_are_explicit: bool
    llm_owns_semantic_expression: bool


@dataclass(frozen=True)
class DecisionAuthoritySettings:
    llm: tuple[str, ...]
    code: tuple[str, ...]


@dataclass(frozen=True)
class InitiativeSettings:
    enabled: bool
    idle_attempt_minutes: int
    max_unanswered_attempts: int
    reset_on_user_message: bool
    use_same_dialogue_engine: bool
    allow_model_to_skip: bool
    daily_send_limit: int | None
    quiet_hours_local: str | None
    cancel_if_conversation_changed: bool
    unknown_delivery_retry: bool
    # 2026-09-14 用户裁定：主动开口这一路显式要思考。理由有实测支撑——关思考的 pro
    # 不做「打出来还是说出来」这种元决策（0/11 次选用语音），开思考 4/4 次会主动发声；
    # 而这一路不占用户等回复的时间。热路径仍然照 llm.primary.thinking。
    thinking: str


@dataclass(frozen=True)
class VisionSettings:
    """Inbound image handling.  Disabled by default; this switch is the whole gate.

    Limits live here rather than in code so the boundary is auditable, and the
    retention numbers are the ones the user approved (200 MB / 90 days).
    """

    enabled: bool
    max_images_per_turn: int
    max_image_bytes: int
    download_timeout_seconds: int
    keep_original_images: bool
    max_total_media_bytes: int
    max_media_age_days: int


@dataclass(frozen=True)
class NetSettings:
    """外部检索。默认关闭，这个开关就是全部的门。

    2026-09-13 用户裁定：后端用 Tavily；不主动查（主动消息里不声明工具）；
    带图轮不查——先把「能不能去查」问过用户；识图“稳定两周”的前置由用户豁免。
    """

    enabled: bool
    base_url: str
    api_key_env: str
    timeout_seconds: int
    max_results: int
    max_chars: int
    search_depth: str
    # 以图搜图（甲）：另一家供应商、另一个 key。图只经过这一家。
    image_api_key_env: str
    image_max_matches: int
    # 宽限窗口：一张新图到达后，之后多少轮内仍可从本地存档重挂（供「先问再查」）。
    image_carry_turns: int


@dataclass(frozen=True)
class VoiceSettings:
    """角色的嗓子。默认关闭，这个开关就是全部的门。

    2026-09-14 用户裁定 1a：走阿里云百炼「声音设计」，音色从零造、不克隆。
    关掉 enabled 即回到纯文字行为，与今天逐字一致 —— 这就是回滚方式。
    """

    enabled: bool
    base_url: str
    api_key_env: str
    api_key: str = field(repr=False)
    # 设计一次就冻结下来的音色身份；运行时不再重新设计（设计是掷骰子，见计划 §2.17）。
    model: str
    voice_id: str
    max_chars_per_utterance: int
    synthesize_timeout_seconds: int
    cache_dir: Path
    cache_max_total_bytes: int
    cache_max_age_days: int


@dataclass(frozen=True)
class OutputGuardSettings:
    reject_empty: bool
    reject_internal_prompt_leak: bool
    reject_raw_protocol_payload: bool


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    redact_secrets: bool
    log_context_sources: bool
    log_model_route: bool
    log_latency: bool


@dataclass(frozen=True, repr=False)
class Config:
    app: AppSettings
    transport: TransportSettings
    storage: StorageSettings
    llm: LLMSettings
    persona: PersonaSettings
    dialogue: DialogueSettings
    expression: ExpressionSettings
    memory: MemorySettings
    boundary: BoundarySettings
    decision_authority: DecisionAuthoritySettings
    initiative: InitiativeSettings
    output_guard: OutputGuardSettings
    observability: ObservabilitySettings
    vision: VisionSettings
    net: NetSettings
    voice: VoiceSettings

    def __repr__(self) -> str:
        return (
            "Config(app={!r}, transport={!r}, storage={!r}, llm={!r}, persona={!r}, "
            "dialogue={!r}, expression={!r}, memory={!r}, boundary={!r}, "
            "decision_authority={!r}, initiative={!r}, output_guard={!r}, observability={!r}, "
            "vision={!r}, net={!r}, voice={!r}, secrets='REDACTED')"
        ).format(
            self.app,
            self.transport,
            self.storage,
            self.llm,
            self.persona,
            self.dialogue,
            self.expression,
            self.memory,
            self.boundary,
            self.decision_authority,
            self.initiative,
            self.output_guard,
            self.observability,
            self.vision,
            self.net,
            self.voice,
        )


_ROOT_KEYS = {
    "app",
    "transport",
    "storage",
    "llm",
    "persona",
    "dialogue",
    "expression",
    "memory",
    "boundary",
    "decision_authority",
    "initiative",
    "vision",
    "output_guard",
    "observability",
    "net",
    "voice",
}
_SENSITIVE_KEYS = {"api_key", "access_token", "token", "secret", "owner_qq"}


def _reject_inline_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in _SENSITIVE_KEYS and child not in (None, ""):
                raise ConfigError(f"secret values must come from environment: {path}.{key}")
            _reject_inline_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_inline_secrets(child, f"{path}[{index}]")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _keys(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise ConfigError(f"unknown configuration field(s) in {name}: {', '.join(unknown)}")


def _text(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _optional_text(mapping: Mapping[str, Any], key: str, name: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name}.{key} must be a non-empty string or null")
    return value.strip()


def _bool(mapping: Mapping[str, Any], key: str, name: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise ConfigError(f"{name}.{key} must be a boolean")
    return value


def _integer(
    mapping: Mapping[str, Any],
    key: str,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ConfigError(f"{name}.{key} must be an integer")
    if value < minimum:
        raise ConfigError(f"{name}.{key} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}.{key} must be at most {maximum}")
    return value


def _nullable_integer(mapping: Mapping[str, Any], key: str, name: str, *, minimum: int = 0) -> int | None:
    if mapping.get(key) is None:
        return None
    return _integer(mapping, key, name, minimum=minimum)


def _number(
    mapping: Mapping[str, Any],
    key: str,
    name: str,
    *,
    minimum: float,
    maximum: float,
    minimum_exclusive: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name}.{key} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigError(f"{name}.{key} must be finite")
    below_minimum = converted <= minimum if minimum_exclusive else converted < minimum
    if below_minimum:
        raise ConfigError(f"{name}.{key} is below its allowed range")
    if converted > maximum:
        raise ConfigError(f"{name}.{key} is above its allowed range")
    return converted


def _strings(mapping: Mapping[str, Any], key: str, name: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError(f"{name}.{key} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _nullable_string(mapping: Mapping[str, Any], key: str, name: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{name}.{key} must be a string or null")
    return value.strip()


def _env_name(mapping: Mapping[str, Any], key: str, name: str) -> str:
    return _text(mapping, key, name)


def _from_env(environ: Mapping[str, str], variable: str) -> str:
    value = environ.get(variable)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"required environment variable is missing: {variable}")
    return value.strip()


def _parse_app(raw: Mapping[str, Any], environ: Mapping[str, str]) -> AppSettings:
    _keys(raw, {"timezone", "owner_qq_env"}, "app")
    owner_env = _env_name(raw, "owner_qq_env", "app")
    owner_qq = _from_env(environ, owner_env)
    if not owner_qq.isdecimal():
        raise ConfigError("owner QQ environment variable must contain only digits")
    return AppSettings(timezone=_text(raw, "timezone", "app"), owner_qq_env=owner_env, owner_qq=owner_qq)


def _parse_transport(raw: Mapping[str, Any], environ: Mapping[str, str]) -> TransportSettings:
    _keys(
        raw,
        {
            "adapter",
            "websocket_url_env",
            "http_url_env",
            "access_token_env",
            "owner_private_only",
            "persist_before_process",
            "quote_context",
            "quote_resolution",
            "local_chat_history",
        },
        "transport",
    )
    owner_private_only = _bool(raw, "owner_private_only", "transport")
    if not owner_private_only:
        raise ConfigError("transport.owner_private_only must be true")
    quote_raw = _mapping(raw.get("quote_resolution"), "transport.quote_resolution")
    _keys(
        quote_raw,
        {
            "local_first",
            "fetch_missing_with_get_msg",
            "persist_resolved_snapshot",
            "include_inbound_and_outbound_links",
            "unresolved_must_be_explicit",
        },
        "transport.quote_resolution",
    )
    history_raw = _mapping(raw.get("local_chat_history"), "transport.local_chat_history")
    _keys(
        history_raw,
        {"enabled", "preserve_original_text", "preserve_message_segments", "retention_days"},
        "transport.local_chat_history",
    )
    websocket_url_env = _env_name(raw, "websocket_url_env", "transport")
    http_url_env = _env_name(raw, "http_url_env", "transport")
    access_token_env = _env_name(raw, "access_token_env", "transport")
    return TransportSettings(
        adapter=_text(raw, "adapter", "transport"),
        websocket_url_env=websocket_url_env,
        http_url_env=http_url_env,
        access_token_env=access_token_env,
        websocket_url=_from_env(environ, websocket_url_env),
        http_url=_from_env(environ, http_url_env),
        access_token=_from_env(environ, access_token_env),
        owner_private_only=owner_private_only,
        persist_before_process=_bool(raw, "persist_before_process", "transport"),
        quote_context=_bool(raw, "quote_context", "transport"),
        quote_resolution=QuoteResolutionSettings(
            local_first=_bool(quote_raw, "local_first", "transport.quote_resolution"),
            fetch_missing_with_get_msg=_bool(quote_raw, "fetch_missing_with_get_msg", "transport.quote_resolution"),
            persist_resolved_snapshot=_bool(quote_raw, "persist_resolved_snapshot", "transport.quote_resolution"),
            include_inbound_and_outbound_links=_bool(
                quote_raw, "include_inbound_and_outbound_links", "transport.quote_resolution"
            ),
            unresolved_must_be_explicit=_bool(quote_raw, "unresolved_must_be_explicit", "transport.quote_resolution"),
        ),
        local_chat_history=LocalChatHistorySettings(
            enabled=_bool(history_raw, "enabled", "transport.local_chat_history"),
            preserve_original_text=_bool(history_raw, "preserve_original_text", "transport.local_chat_history"),
            preserve_message_segments=_bool(history_raw, "preserve_message_segments", "transport.local_chat_history"),
            retention_days=_nullable_integer(history_raw, "retention_days", "transport.local_chat_history", minimum=1),
        ),
    )


def _parse_primary(raw: Mapping[str, Any]) -> PrimaryModelSettings:
    _keys(
        raw,
        {
            "model",
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_visible_output_tokens",
            "timeout_seconds",
            "required_min_context_tokens",
            "thinking",
        },
        "llm.primary",
    )
    model = _text(raw, "model", "llm.primary")
    if model not in {"deepseek-ai/DeepSeek-V4-Flash", "deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ConfigError("llm.primary.model must be a verified DeepSeek V4 identifier")
    max_output_tokens = _integer(raw, "max_output_tokens", "llm.primary", minimum=1)
    max_visible_output_tokens = _integer(
        raw, "max_visible_output_tokens", "llm.primary", minimum=1
    )
    if max_visible_output_tokens > max_output_tokens:
        raise ConfigError(
            "llm.primary.max_visible_output_tokens must not exceed max_output_tokens"
        )
    thinking = _optional_text(raw, "thinking", "llm.primary") or "default"
    if thinking not in {"default", "enabled", "disabled"}:
        raise ConfigError("llm.primary.thinking must be default, enabled or disabled")
    return PrimaryModelSettings(
        model=model,
        temperature=_number(raw, "temperature", "llm.primary", minimum=0.0, maximum=2.0),
        top_p=_number(raw, "top_p", "llm.primary", minimum=0.0, maximum=1.0, minimum_exclusive=True),
        max_output_tokens=max_output_tokens,
        max_visible_output_tokens=max_visible_output_tokens,
        timeout_seconds=_integer(raw, "timeout_seconds", "llm.primary", minimum=1),
        required_min_context_tokens=_integer(raw, "required_min_context_tokens", "llm.primary", minimum=1),
        thinking=thinking,
    )


def _parse_llm(raw: Mapping[str, Any], environ: Mapping[str, str], preferred: int) -> LLMSettings:
    _keys(
        raw,
        {
            "provider", "base_url", "api_key_env", "primary", "normal_reply_calls",
            "vision_model", "background_model",
        },
        "llm",
    )
    provider = _text(raw, "provider", "llm").casefold()
    if provider not in {"deepseek", "siliconflow"}:
        raise ConfigError("llm.provider must be deepseek or siliconflow")
    primary = _parse_primary(_mapping(raw.get("primary"), "llm.primary"))
    allowed_models = (
        frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})
        if provider == "deepseek"
        else frozenset({"deepseek-ai/DeepSeek-V4-Flash"})
    )
    if primary.model not in allowed_models:
        raise ConfigError("llm.primary.model is not a verified identifier for this provider")
    vision_model = _optional_text(raw, "vision_model", "llm")
    if vision_model is not None:
        if provider != "deepseek":
            raise ConfigError("llm.vision_model is only supported for the deepseek provider")
        if vision_model not in allowed_models:
            raise ConfigError("llm.vision_model is not a verified identifier for this provider")
    background_model = _optional_text(raw, "background_model", "llm") or primary.model
    if background_model not in allowed_models:
        raise ConfigError("llm.background_model is not a verified identifier for this provider")
    if primary.required_min_context_tokens < preferred:
        raise ConfigError("primary.required_min_context_tokens must cover preferred context window")
    base_url = _text(raw, "base_url", "llm").rstrip("/")
    api_key_env = _env_name(raw, "api_key_env", "llm")
    expected_base_url = {
        "deepseek": "https://api.deepseek.com/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
    }[provider]
    expected_key_env = {
        "deepseek": "DEEPSEEK_API_KEY",
        "siliconflow": "SILICONFLOW_API_KEY",
    }[provider]
    if base_url != expected_base_url:
        raise ConfigError(f"llm.base_url must be {expected_base_url} for provider {provider}")
    if api_key_env != expected_key_env:
        raise ConfigError(f"llm.api_key_env must be {expected_key_env} for provider {provider}")
    return LLMSettings(
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=_from_env(environ, api_key_env),
        primary=primary,
        normal_reply_calls=_integer(raw, "normal_reply_calls", "llm", minimum=1, maximum=1),
        vision_model=vision_model,
        background_model=background_model,
    )


def _parse_persona(raw: Mapping[str, Any]) -> PersonaSettings:
    _keys(raw, {"system_prompt_file", "reference_files", "inject_reference_files_verbatim", "prompt_mode"}, "persona")
    references = _strings(raw, "reference_files", "persona")
    return PersonaSettings(
        system_prompt_file=Path(_text(raw, "system_prompt_file", "persona")),
        reference_files=tuple(Path(item) for item in references),
        inject_reference_files_verbatim=_bool(raw, "inject_reference_files_verbatim", "persona"),
        prompt_mode=_text(raw, "prompt_mode", "persona"),
    )


def _parse_dialogue(raw: Mapping[str, Any]) -> DialogueSettings:
    _keys(
        raw,
        {
            "context_window_preferred_tokens",
            "context_window_max_tokens",
            "output_reserve_tokens",
            "recent_history_budget_tokens",
            "require_provider_context_support",
            "context_strategy",
            "adaptive_expansion",
            "pin_current_input",
            "pin_quoted_target",
            "pin_active_relationship_state",
            "merge_window_ms",
            "merge_window_max_ms",
            "per_conversation_serial",
            "natural_pause_ms",
            "structural_retry_limit",
            "llm_decides_reply_target",
            "auto_quote_current_message",
        },
        "dialogue",
    )
    preferred = _integer(raw, "context_window_preferred_tokens", "dialogue", minimum=1)
    maximum = _integer(raw, "context_window_max_tokens", "dialogue", minimum=1, maximum=524288)
    reserve = _integer(raw, "output_reserve_tokens", "dialogue", minimum=1)
    recent_history_budget = _integer(raw, "recent_history_budget_tokens", "dialogue", minimum=0)
    if preferred > maximum:
        raise ConfigError("preferred context window must not exceed max")
    if reserve >= preferred:
        raise ConfigError("output reserve must leave room in preferred context window")
    merge_window_ms = _integer(raw, "merge_window_ms", "dialogue", minimum=0)
    merge_window_max_ms = _integer(raw, "merge_window_max_ms", "dialogue", minimum=0)
    if merge_window_ms > merge_window_max_ms:
        raise ConfigError("dialogue.merge_window_ms must not exceed merge_window_max_ms")
    return DialogueSettings(
        context_window_preferred_tokens=preferred,
        context_window_max_tokens=maximum,
        output_reserve_tokens=reserve,
        recent_history_budget_tokens=recent_history_budget,
        require_provider_context_support=_bool(raw, "require_provider_context_support", "dialogue"),
        context_strategy=_text(raw, "context_strategy", "dialogue"),
        adaptive_expansion=_bool(raw, "adaptive_expansion", "dialogue"),
        pin_current_input=_bool(raw, "pin_current_input", "dialogue"),
        pin_quoted_target=_bool(raw, "pin_quoted_target", "dialogue"),
        pin_active_relationship_state=_bool(raw, "pin_active_relationship_state", "dialogue"),
        merge_window_ms=merge_window_ms,
        merge_window_max_ms=merge_window_max_ms,
        per_conversation_serial=_bool(raw, "per_conversation_serial", "dialogue"),
        natural_pause_ms=_integer(raw, "natural_pause_ms", "dialogue", minimum=0),
        structural_retry_limit=_integer(raw, "structural_retry_limit", "dialogue", minimum=0),
        llm_decides_reply_target=_bool(raw, "llm_decides_reply_target", "dialogue"),
        auto_quote_current_message=_bool(raw, "auto_quote_current_message", "dialogue"),
    )


def _parse_expression_channel(
    raw: Mapping[str, Any],
    name: str,
    *,
    catalog: bool = False,
    idempotent: bool = False,
    selection_required: bool = True,
) -> ExpressionChannelSettings:
    allowed = {"enabled", "selection"}
    if not selection_required:
        allowed.remove("selection")
    if catalog:
        allowed.add("runtime_catalog")
    if idempotent:
        allowed.add("idempotent_set")
    _keys(raw, allowed, name)
    return ExpressionChannelSettings(
        enabled=_bool(raw, "enabled", name),
        selection=_text(raw, "selection", name) if selection_required else None,
        runtime_catalog=_optional_text(raw, "runtime_catalog", name) if catalog else None,
        idempotent_set=_bool(raw, "idempotent_set", name) if idempotent else None,
    )


def _parse_expression(raw: Mapping[str, Any]) -> ExpressionSettings:
    _keys(
        raw,
        {
            "unicode_emoji",
            "qq_face",
            "message_reaction",
            "custom_sticker",
            "intent_protocol",
            "max_native_actions_per_turn",
            "random_frequency",
            "fixed_turn_interval",
            "replace_text_reply",
            "persist_action_metadata",
        },
        "expression",
    )
    intent_protocol = _text(raw, "intent_protocol", "expression")
    try:
        intent_protocol = _INTENT_PROTOCOL_ALIASES[intent_protocol]
    except KeyError as exc:
        accepted = ", ".join(sorted(_INTENT_PROTOCOL_ALIASES))
        raise ConfigError(
            f"expression.intent_protocol must be one of: {accepted}"
        ) from exc
    return ExpressionSettings(
        unicode_emoji=_parse_expression_channel(_mapping(raw.get("unicode_emoji"), "expression.unicode_emoji"), "expression.unicode_emoji"),
        qq_face=_parse_expression_channel(
            _mapping(raw.get("qq_face"), "expression.qq_face"), "expression.qq_face", catalog=True
        ),
        message_reaction=_parse_expression_channel(
            _mapping(raw.get("message_reaction"), "expression.message_reaction"),
            "expression.message_reaction",
            idempotent=True,
        ),
        custom_sticker=_parse_expression_channel(
            _mapping(raw.get("custom_sticker"), "expression.custom_sticker"),
            "expression.custom_sticker",
            selection_required=False,
        ),
        intent_protocol=intent_protocol,
        max_native_actions_per_turn=_integer(raw, "max_native_actions_per_turn", "expression", minimum=1, maximum=1),
        random_frequency=_bool(raw, "random_frequency", "expression"),
        fixed_turn_interval=_bool(raw, "fixed_turn_interval", "expression"),
        replace_text_reply=_bool(raw, "replace_text_reply", "expression"),
        persist_action_metadata=_bool(raw, "persist_action_metadata", "expression"),
    )


def _parse_memory(raw: Mapping[str, Any]) -> MemorySettings:
    _keys(
        raw,
        {
            "write_candidates_async",
            "require_source_quote",
            "require_source_message_id",
            "require_source_actor",
            "include_evidence_in_context",
            "auto_commit",
            "always_include_types",
            "retrieval_candidate_top_k",
            "episodic_top_k",
            "lifecycle_states",
            "correction_precedence",
            "ambiguity_stays_candidate",
            "extraction_mode",
            "extraction_idle_minutes",
            "extractor_must_reference_event_ids",
            "uncertain_as_fact",
            "generated_text_as_user_evidence",
            "recent_history_from_raw_events",
        },
        "memory",
    )
    auto_commit = _text(raw, "auto_commit", "memory")
    if auto_commit not in {"explicit_only", "verified_explicit"}:
        raise ConfigError("memory.auto_commit must be explicit_only or verified_explicit")
    extraction_mode = _text(raw, "extraction_mode", "memory")
    if extraction_mode != "async_session_consolidation_v2":
        raise ConfigError(
            "memory.extraction_mode must be async_session_consolidation_v2"
        )
    extraction_idle_minutes = _integer(
        raw, "extraction_idle_minutes", "memory", minimum=1
    )
    if extraction_idle_minutes != 30:
        raise ConfigError("memory.extraction_idle_minutes is fixed at 30")
    return MemorySettings(
        write_candidates_async=_bool(raw, "write_candidates_async", "memory"),
        require_source_quote=_bool(raw, "require_source_quote", "memory"),
        require_source_message_id=_bool(raw, "require_source_message_id", "memory"),
        require_source_actor=_bool(raw, "require_source_actor", "memory"),
        include_evidence_in_context=_bool(raw, "include_evidence_in_context", "memory"),
        auto_commit=auto_commit,
        always_include_types=_strings(raw, "always_include_types", "memory"),
        retrieval_candidate_top_k=_integer(raw, "retrieval_candidate_top_k", "memory", minimum=1),
        episodic_top_k=_integer(raw, "episodic_top_k", "memory", minimum=1),
        lifecycle_states=_strings(raw, "lifecycle_states", "memory"),
        correction_precedence=_bool(raw, "correction_precedence", "memory"),
        ambiguity_stays_candidate=_bool(raw, "ambiguity_stays_candidate", "memory"),
        extraction_mode=extraction_mode,
        extraction_idle_minutes=extraction_idle_minutes,
        extractor_must_reference_event_ids=_bool(raw, "extractor_must_reference_event_ids", "memory"),
        uncertain_as_fact=_bool(raw, "uncertain_as_fact", "memory"),
        generated_text_as_user_evidence=_bool(raw, "generated_text_as_user_evidence", "memory"),
        recent_history_from_raw_events=_bool(raw, "recent_history_from_raw_events", "memory"),
    )


def _parse_boundary(raw: Mapping[str, Any]) -> BoundarySettings:
    fields = {
        "inject_dynamic_capability_manifest",
        "label_all_fact_sources",
        "distinguish_past_from_current",
        "distinguish_user_fact_from_assistant_history",
        "unavailable_capabilities_are_explicit",
        "llm_owns_semantic_expression",
    }
    _keys(raw, fields, "boundary")
    return BoundarySettings(**{field_name: _bool(raw, field_name, "boundary") for field_name in fields})


def _parse_decision_authority(raw: Mapping[str, Any]) -> DecisionAuthoritySettings:
    _keys(raw, {"llm", "code"}, "decision_authority")
    return DecisionAuthoritySettings(
        llm=_strings(raw, "llm", "decision_authority"),
        code=_strings(raw, "code", "decision_authority"),
    )


def _parse_initiative(raw: Mapping[str, Any]) -> InitiativeSettings:
    _keys(
        raw,
        {
            "enabled",
            "idle_attempt_minutes",
            "max_unanswered_attempts",
            "reset_on_user_message",
            "use_same_dialogue_engine",
            "allow_model_to_skip",
            "daily_send_limit",
            "quiet_hours_local",
            "cancel_if_conversation_changed",
            "unknown_delivery_retry",
            "thinking",
        },
        "initiative",
    )
    thinking = _optional_text(raw, "thinking", "initiative") or "disabled"
    if thinking not in {"default", "enabled", "disabled"}:
        raise ConfigError("initiative.thinking must be default, enabled or disabled")
    return InitiativeSettings(
        enabled=_bool(raw, "enabled", "initiative"),
        idle_attempt_minutes=_integer(raw, "idle_attempt_minutes", "initiative", minimum=1),
        max_unanswered_attempts=_integer(raw, "max_unanswered_attempts", "initiative", minimum=1),
        reset_on_user_message=_bool(raw, "reset_on_user_message", "initiative"),
        use_same_dialogue_engine=_bool(raw, "use_same_dialogue_engine", "initiative"),
        allow_model_to_skip=_bool(raw, "allow_model_to_skip", "initiative"),
        daily_send_limit=_nullable_integer(raw, "daily_send_limit", "initiative", minimum=1),
        quiet_hours_local=_nullable_string(raw, "quiet_hours_local", "initiative"),
        cancel_if_conversation_changed=_bool(raw, "cancel_if_conversation_changed", "initiative"),
        unknown_delivery_retry=_bool(raw, "unknown_delivery_retry", "initiative"),
        thinking=thinking,
    )


def _parse_vision(raw: Mapping[str, Any]) -> VisionSettings:
    _keys(
        raw,
        {
            "enabled",
            "max_images_per_turn",
            "max_image_bytes",
            "download_timeout_seconds",
            "keep_original_images",
            "max_total_media_bytes",
            "max_media_age_days",
        },
        "vision",
    )
    return VisionSettings(
        enabled=_bool(raw, "enabled", "vision"),
        max_images_per_turn=_integer(raw, "max_images_per_turn", "vision", minimum=1),
        max_image_bytes=_integer(raw, "max_image_bytes", "vision", minimum=1),
        download_timeout_seconds=_integer(raw, "download_timeout_seconds", "vision", minimum=1),
        keep_original_images=_bool(raw, "keep_original_images", "vision"),
        max_total_media_bytes=_integer(raw, "max_total_media_bytes", "vision", minimum=1),
        max_media_age_days=_integer(raw, "max_media_age_days", "vision", minimum=1),
    )


def _parse_net(raw: Mapping[str, Any]) -> NetSettings:
    _keys(
        raw,
        {
            "enabled",
            "base_url",
            "api_key_env",
            "timeout_seconds",
            "max_results",
            "max_chars",
            "search_depth",
            "image_api_key_env",
            "image_max_matches",
            "image_carry_turns",
        },
        "net",
    )
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ConfigError("net.enabled must be a boolean")
    depth = _optional_text(raw, "search_depth", "net") or "basic"
    if depth not in {"basic", "advanced", "fast"}:
        raise ConfigError("net.search_depth must be basic, advanced or fast")
    return NetSettings(
        enabled=enabled,
        base_url=_optional_text(raw, "base_url", "net") or "https://api.tavily.com",
        api_key_env=_optional_text(raw, "api_key_env", "net") or "TAVILY_API_KEY",
        timeout_seconds=_defaulted_integer(raw, "timeout_seconds", "net", 8, minimum=1, maximum=30),
        max_results=_defaulted_integer(raw, "max_results", "net", 3, minimum=1, maximum=5),
        max_chars=_defaulted_integer(raw, "max_chars", "net", 1200, minimum=200, maximum=4000),
        search_depth=depth,
        image_api_key_env=_optional_text(raw, "image_api_key_env", "net") or "SERPAPI_API_KEY",
        image_max_matches=_defaulted_integer(raw, "image_max_matches", "net", 5, minimum=1, maximum=10),
        image_carry_turns=_defaulted_integer(raw, "image_carry_turns", "net", 6, minimum=1, maximum=20),
    )


def _defaulted_integer(
    mapping: Mapping[str, Any], key: str, name: str, default: int, *, minimum: int, maximum: int
) -> int:
    if mapping.get(key) is None:
        return default
    return _integer(mapping, key, name, minimum=minimum, maximum=maximum)


_VOICE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
_VOICE_KEY_ENV = "DASHSCOPE_API_KEY"


def _parse_voice(raw: Mapping[str, Any], environ: Mapping[str, str]) -> VoiceSettings:
    _keys(
        raw,
        {
            "enabled",
            "base_url",
            "api_key_env",
            "model",
            "voice_id",
            "max_chars_per_utterance",
            "synthesize_timeout_seconds",
            "cache_dir",
            "cache_max_total_bytes",
            "cache_max_age_days",
        },
        "voice",
    )
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ConfigError("voice.enabled must be a boolean")
    base_url = (_optional_text(raw, "base_url", "voice") or _VOICE_BASE_URL).rstrip("/")
    if base_url != _VOICE_BASE_URL:
        raise ConfigError("voice.base_url must be " + _VOICE_BASE_URL)
    api_key_env = _optional_text(raw, "api_key_env", "voice") or _VOICE_KEY_ENV
    if api_key_env != _VOICE_KEY_ENV:
        raise ConfigError("voice.api_key_env must be " + _VOICE_KEY_ENV)
    model = _optional_text(raw, "model", "voice") or ""
    voice_id = _optional_text(raw, "voice_id", "voice") or ""
    if enabled and (not model or not voice_id):
        # fail closed：说得出能力就必须做得出来，缺音色身份宁可拒绝启动。
        raise ConfigError("voice.model and voice.voice_id are required when voice is enabled")
    return VoiceSettings(
        enabled=enabled,
        base_url=base_url,
        api_key_env=api_key_env,
        # 只有开启时才要求密钥：关着的时候不该因为少一个 key 而起不来。
        api_key=_from_env(environ, api_key_env) if enabled else "",
        model=model,
        voice_id=voice_id,
        max_chars_per_utterance=_defaulted_integer(
            raw, "max_chars_per_utterance", "voice", 120, minimum=10, maximum=400
        ),
        synthesize_timeout_seconds=_defaulted_integer(
            raw, "synthesize_timeout_seconds", "voice", 12, minimum=1, maximum=60
        ),
        cache_dir=Path(_optional_text(raw, "cache_dir", "voice") or "data/voice"),
        cache_max_total_bytes=_defaulted_integer(
            raw, "cache_max_total_bytes", "voice", 209715200, minimum=1048576, maximum=10737418240
        ),
        cache_max_age_days=_defaulted_integer(
            raw, "cache_max_age_days", "voice", 30, minimum=1, maximum=3650
        ),
    )


def _parse_output_guard(raw: Mapping[str, Any]) -> OutputGuardSettings:
    fields = {
        "reject_empty",
        "reject_internal_prompt_leak",
        "reject_raw_protocol_payload",
    }
    _keys(raw, fields, "output_guard")
    return OutputGuardSettings(**{field_name: _bool(raw, field_name, "output_guard") for field_name in fields})


def _parse_observability(raw: Mapping[str, Any]) -> ObservabilitySettings:
    _keys(
        raw,
        {"log_level", "redact_secrets", "log_context_sources", "log_model_route", "log_latency"},
        "observability",
    )
    return ObservabilitySettings(
        log_level=_text(raw, "log_level", "observability"),
        redact_secrets=_bool(raw, "redact_secrets", "observability"),
        log_context_sources=_bool(raw, "log_context_sources", "observability"),
        log_model_route=_bool(raw, "log_model_route", "observability"),
        log_latency=_bool(raw, "log_latency", "observability"),
    )


def load_config(path: str | Path, *, environ: Mapping[str, str] | None = None) -> Config:
    """Load every runtime setting and resolve secrets exclusively from ``environ``."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read configuration: {source}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML configuration: {source}") from exc
    root = _mapping(raw, "config")
    _reject_inline_secrets(root)
    _keys(root, _ROOT_KEYS, "config")
    env = os.environ if environ is None else environ

    dialogue = _parse_dialogue(_mapping(root.get("dialogue"), "dialogue"))
    llm = _parse_llm(_mapping(root.get("llm"), "llm"), env, dialogue.context_window_preferred_tokens)
    if dialogue.output_reserve_tokens < llm.primary.max_output_tokens:
        raise ConfigError(
            "dialogue.output_reserve_tokens must cover llm.primary.max_output_tokens"
        )
    storage_raw = _mapping(root.get("storage"), "storage")
    _keys(storage_raw, {"database_path"}, "storage")
    database_path = (
        Path(_text(storage_raw, "database_path", "storage"))
        if "database_path" in storage_raw
        else Path("data/qichi.sqlite3")
    )

    return Config(
        app=_parse_app(_mapping(root.get("app"), "app"), env),
        transport=_parse_transport(_mapping(root.get("transport"), "transport"), env),
        storage=StorageSettings(database_path=database_path),
        llm=llm,
        persona=_parse_persona(_mapping(root.get("persona"), "persona")),
        dialogue=dialogue,
        expression=_parse_expression(_mapping(root.get("expression"), "expression")),
        memory=_parse_memory(_mapping(root.get("memory"), "memory")),
        boundary=_parse_boundary(_mapping(root.get("boundary"), "boundary")),
        decision_authority=_parse_decision_authority(_mapping(root.get("decision_authority"), "decision_authority")),
        initiative=_parse_initiative(_mapping(root.get("initiative"), "initiative")),
        output_guard=_parse_output_guard(_mapping(root.get("output_guard"), "output_guard")),
        observability=_parse_observability(_mapping(root.get("observability"), "observability")),
        vision=_parse_vision(_mapping(root.get("vision"), "vision")),
        net=_parse_net(_mapping(root.get("net") or {}, "net")),
        voice=_parse_voice(_mapping(root.get("voice") or {}, "voice"), env),
    )
