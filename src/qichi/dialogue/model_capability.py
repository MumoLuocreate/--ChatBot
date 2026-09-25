"""Verified model and provider context-window capability boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .token_counter import HashedTokenizerCounter


class ModelCapabilityError(ValueError):
    """A requested context budget is invalid or not supported."""


class ProviderCapabilityUnverifiedError(ModelCapabilityError):
    """The provider has not supplied a verified context-window capacity."""


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an int")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    try:
        valid = len(bytes.fromhex(value)) == 32
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")


@dataclass(frozen=True, slots=True)
class ProviderCapabilityEvidence:
    """A time-bound, exact-provider observation of one model's total window."""

    provider: str
    model_id: str
    context_tokens: int
    source: str
    retrieved_at: datetime

    def __post_init__(self) -> None:
        for field in ("provider", "model_id", "source"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
        _positive_int(self.context_tokens, "context_tokens")
        if not isinstance(self.retrieved_at, datetime) or self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")


def load_provider_capability_evidence(
    path: str | Path,
    *,
    provider: str,
    model_id: str,
) -> ProviderCapabilityEvidence:
    """Load an exact, human-auditable provider capability snapshot.

    The snapshot is deliberately separate from the model manifest: architecture
    limits never stand in for what the provider actually advertises.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCapabilityError("provider capability evidence is unavailable") from error
    if not isinstance(raw, Mapping):
        raise ModelCapabilityError("provider capability evidence must be an object")
    required = {
        "schema",
        "provider",
        "model_id",
        "context_tokens",
        "source",
        "retrieved_at_utc",
        "observation",
    }
    if set(raw) != required or raw.get("schema") != "qichi.provider-capability/v1":
        raise ModelCapabilityError("provider capability evidence schema is invalid")
    if raw.get("provider") != provider or raw.get("model_id") != model_id:
        raise ModelCapabilityError("provider capability evidence identity mismatch")
    observation = raw.get("observation")
    if not isinstance(observation, str) or model_id not in observation:
        raise ModelCapabilityError("provider capability evidence observation is incomplete")
    retrieved = raw.get("retrieved_at_utc")
    if not isinstance(retrieved, str):
        raise ModelCapabilityError("provider capability evidence timestamp is invalid")
    try:
        retrieved_at = datetime.fromisoformat(retrieved)
    except ValueError as error:
        raise ModelCapabilityError("provider capability evidence timestamp is invalid") from error
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ModelCapabilityError("provider capability evidence timestamp must be timezone-aware")
    context_tokens = raw.get("context_tokens")
    if type(context_tokens) is not int or context_tokens <= 0:
        raise ModelCapabilityError("provider capability evidence context is invalid")
    source = raw.get("source")
    if not isinstance(source, str) or not source.startswith(("https://", "http://")):
        raise ModelCapabilityError("provider capability evidence source is invalid")
    return ProviderCapabilityEvidence(
        provider=provider,
        model_id=model_id,
        context_tokens=context_tokens,
        source=source,
        retrieved_at=retrieved_at,
    )


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Architecture and evidence-backed provider limits for one exact model."""

    model_id: str
    model_context_tokens: int
    provider: str = "SiliconFlow"
    provider_evidence: ProviderCapabilityEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a nonempty string")
        _positive_int(self.model_context_tokens, "model_context_tokens")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a nonempty string")
        if self.provider_evidence is not None:
            if not isinstance(self.provider_evidence, ProviderCapabilityEvidence):
                raise TypeError("provider_evidence must be ProviderCapabilityEvidence or None")
            if self.provider_evidence.model_id != self.model_id:
                raise ModelCapabilityError("provider evidence model ID does not match capability")
            if self.provider_evidence.provider != self.provider:
                raise ModelCapabilityError("provider evidence provider does not match capability")

    @property
    def provider_context_tokens(self) -> int | None:
        return None if self.provider_evidence is None else self.provider_evidence.context_tokens

    @property
    def effective_context_tokens(self) -> int | None:
        if self.provider_evidence is None:
            return None
        return min(self.model_context_tokens, self.provider_evidence.context_tokens)

    def supports_context(self, window_tokens: int) -> bool | None:
        """Return unsupported, unknown, or evidence-backed support for a total window."""
        _positive_int(window_tokens, "window_tokens")
        if window_tokens > self.model_context_tokens:
            return False
        if self.provider_evidence is None:
            return None
        return window_tokens <= self.provider_evidence.context_tokens

    def assert_supports(self, required_input: int, *, output_reserve: int = 0) -> None:
        """Raise unless input plus reserved output fit one verified total window."""
        _positive_int(required_input, "required_input")
        if isinstance(output_reserve, bool) or not isinstance(output_reserve, int):
            raise TypeError("output_reserve must be an int")
        if output_reserve < 0:
            raise ValueError("output_reserve must not be negative")
        total_required = required_input + output_reserve
        if total_required > self.model_context_tokens:
            raise ModelCapabilityError("requested input and output reserve exceed model context capacity")
        if self.provider_evidence is None:
            raise ProviderCapabilityUnverifiedError("provider context capacity is UNVERIFIED")
        if total_required > self.provider_evidence.context_tokens:
            raise ModelCapabilityError("requested input and output reserve exceed provider context capacity")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Architecture and tokenizer facts pinned to one exact model identifier."""

    model_id: str
    model_context_tokens: int
    tokenizer_revision: str
    tokenizer_url: str
    tokenizer_sha256: str
    config_url: str
    config_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a nonempty string")
        _positive_int(self.model_context_tokens, "model_context_tokens")
        for field in ("tokenizer_revision", "tokenizer_url", "config_url"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")
        _sha256(self.tokenizer_sha256, "tokenizer_sha256")
        _sha256(self.config_sha256, "config_sha256")

    def capability_for(
        self,
        model_id: str,
        *,
        provider: str = "SiliconFlow",
        provider_evidence: ProviderCapabilityEvidence | None = None,
    ) -> ModelCapability:
        if model_id != self.model_id:
            raise ModelCapabilityError("model ID does not match the verified manifest")
        if provider_evidence is not None:
            if not isinstance(provider_evidence, ProviderCapabilityEvidence):
                raise TypeError("provider_evidence must be ProviderCapabilityEvidence or None")
            if provider_evidence.model_id != self.model_id:
                raise ModelCapabilityError("provider evidence model ID does not match the verified manifest")
            if provider_evidence.provider != provider:
                raise ModelCapabilityError("provider evidence provider does not match capability")
        return ModelCapability(self.model_id, self.model_context_tokens, provider, provider_evidence)

    def load_token_counter(self, artifact_path: str | Path, *, add_special_tokens: bool = False) -> HashedTokenizerCounter:
        """Load a local artifact only after checking this manifest's pinned hash."""
        return HashedTokenizerCounter(artifact_path, self.tokenizer_sha256, add_special_tokens=add_special_tokens)


MODEL_MANIFESTS: Mapping[str, ModelManifest] = MappingProxyType(
    {
        # DeepSeek's official API exposes the same V4 Flash model under the
        # short identifier below.  The pinned local tokenizer/config facts
        # are shared with the verified V4 Flash artifact used by providers.
        "deepseek-v4-flash": ModelManifest(
            "deepseek-v4-flash", 1_048_576,
            "60d8d70770c6776ff598c94bb586a859a38244f1",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json",
            "8F9F37CA37FDC4F5FD36D5CF4D3B0E8392EDB4E894FD10CC0D70B4957C8633CF",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/config.json",
            "B628E63398A645ABC711D92207F8737DD8140F7A4EF1E0A5B3616019E0DDD818",
        ),
        # 2026-09-12：pro 的 tokenizer.json 与 flash 逐字节相同（两边 SHA256 都是
        # 8F9F37CA…，从 HF 官方仓库 b5968e91 下回来核对过），所以共用同一份本地
        # artifact 计数是准的；config.json 是 pro 自己的，单独钉住。
        "deepseek-v4-pro": ModelManifest(
            "deepseek-v4-pro", 1_048_576,
            "b5968e9190ef611bbf34a7229255be88a0e937c1",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json",
            "8F9F37CA37FDC4F5FD36D5CF4D3B0E8392EDB4E894FD10CC0D70B4957C8633CF",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json",
            "5FE4568DAEE51C208CB8A79538EAEDA090AE011ADE1DEE2C386AA95F569C810E",
        ),
        "deepseek-ai/DeepSeek-V4-Flash": ModelManifest(
            "deepseek-ai/DeepSeek-V4-Flash", 1_048_576,
            "60d8d70770c6776ff598c94bb586a859a38244f1",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json",
            "8F9F37CA37FDC4F5FD36D5CF4D3B0E8392EDB4E894FD10CC0D70B4957C8633CF",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/config.json",
            "B628E63398A645ABC711D92207F8737DD8140F7A4EF1E0A5B3616019E0DDD818",
        ),
        "deepseek-ai/DeepSeek-V3.2": ModelManifest(
            "deepseek-ai/DeepSeek-V3.2", 163_840,
            "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
            "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/tokenizer.json",
            "CD050BE35CAE877F8F0AA847F45AA87E23835A56CA32B29B28545597852784E5",
            "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json",
            "C7FA8B191E9936D8E6A57D864BAAB82B792FAE16A116416CDD3A75BA76BC5AF1",
        ),
    }
)


def model_capability_for(
    model_id: str,
    *,
    provider: str = "SiliconFlow",
    provider_evidence: ProviderCapabilityEvidence | None = None,
) -> ModelCapability:
    """Build a capability from an exact manifest and optional provider evidence."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a str")
    try:
        manifest = MODEL_MANIFESTS[model_id]
    except KeyError as exc:
        raise ModelCapabilityError("no verified manifest for model ID") from exc
    return manifest.capability_for(model_id, provider=provider, provider_evidence=provider_evidence)
