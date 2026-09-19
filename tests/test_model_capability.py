from __future__ import annotations

from datetime import UTC, datetime

import pytest

from qichi.dialogue.model_capability import (
    MODEL_MANIFESTS,
    ModelCapability,
    ModelCapabilityError,
    ProviderCapabilityEvidence,
    ProviderCapabilityUnverifiedError,
    model_capability_for,
)


def _evidence(model_id: str, context_tokens: int) -> ProviderCapabilityEvidence:
    return ProviderCapabilityEvidence(
        "SiliconFlow", model_id, context_tokens,
        "https://provider.example/models/verified-observation", datetime(2026, 8, 27, tzinfo=UTC),
    )


def test_effective_capacity_requires_a_verified_provider_cap():
    capability = ModelCapability("test/model", 300_000)
    assert capability.effective_context_tokens is None
    assert capability.supports_context(262_144) is None
    with pytest.raises(ProviderCapabilityUnverifiedError, match="UNVERIFIED"):
        capability.assert_supports(1)


def test_effective_capacity_is_the_lower_of_model_and_provider():
    capability = ModelCapability("test/model", 300_000, provider_evidence=_evidence("test/model", 262_144))
    assert capability.effective_context_tokens == 262_144
    capability.assert_supports(258_048, output_reserve=4_096)
    with pytest.raises(ModelCapabilityError):
        capability.assert_supports(258_049, output_reserve=4_096)


@pytest.mark.parametrize("required,reserve", [(True, 0), (0, 0), (-1, 0), (1, True), (1, -1)])
def test_required_input_and_reserve_are_strictly_validated(required, reserve):
    capability = ModelCapability("test/model", 10, provider_evidence=_evidence("test/model", 10))
    with pytest.raises((TypeError, ValueError)):
        capability.assert_supports(required, output_reserve=reserve)


def test_output_reserve_is_part_of_the_total_window():
    capability = ModelCapability("test/model", 262_144, provider_evidence=_evidence("test/model", 262_144))
    with pytest.raises(ModelCapabilityError):
        capability.assert_supports(262_144, output_reserve=1)
    capability.assert_supports(262_144, output_reserve=0)
    with pytest.raises(ModelCapabilityError):
        capability.assert_supports(1, output_reserve=262_144)


def test_context_support_is_explicitly_false_unknown_or_evidence_backed():
    v32 = model_capability_for("deepseek-ai/DeepSeek-V3.2")
    v4_unknown = model_capability_for("deepseek-ai/DeepSeek-V4-Flash")
    v4_256k = model_capability_for("deepseek-ai/DeepSeek-V4-Flash", provider_evidence=_evidence("deepseek-ai/DeepSeek-V4-Flash", 262_144))
    v4_512k = model_capability_for("deepseek-ai/DeepSeek-V4-Flash", provider_evidence=_evidence("deepseek-ai/DeepSeek-V4-Flash", 524_288))

    assert v32.supports_context(262_144) is False
    assert v32.supports_context(524_288) is False
    assert v4_unknown.supports_context(262_144) is None
    assert v4_unknown.supports_context(524_288) is None
    assert v4_256k.supports_context(262_144) is True
    assert v4_256k.supports_context(524_288) is False
    assert v4_512k.supports_context(262_144) is True
    assert v4_512k.supports_context(524_288) is True


def test_known_architecture_shortfall_precedes_provider_unknown():
    with pytest.raises(ModelCapabilityError, match="model context"):
        model_capability_for("deepseek-ai/DeepSeek-V3.2").assert_supports(262_144)


def test_provider_evidence_is_frozen_exact_and_time_bound():
    with pytest.raises(ValueError, match="timezone-aware"):
        ProviderCapabilityEvidence("SiliconFlow", "test/model", 1, "source", datetime(2026, 8, 27))
    evidence = _evidence("deepseek-ai/DeepSeek-V3.2", 524_288)
    with pytest.raises(ModelCapabilityError, match="evidence model ID"):
        model_capability_for("deepseek-ai/DeepSeek-V4-Flash", provider_evidence=evidence)
    with pytest.raises(TypeError):
        model_capability_for("deepseek-ai/DeepSeek-V4-Flash", provider_evidence=262_144)  # type: ignore[arg-type]
    wrong_provider = ProviderCapabilityEvidence(
        "OtherProvider", "deepseek-ai/DeepSeek-V4-Flash", 524_288,
        "https://other.example/models", datetime(2026, 8, 27, tzinfo=UTC),
    )
    with pytest.raises(ModelCapabilityError, match="evidence provider"):
        model_capability_for("deepseek-ai/DeepSeek-V4-Flash", provider_evidence=wrong_provider)


def test_manifests_bind_tokenizers_and_model_config_sources():
    manifest = MODEL_MANIFESTS["deepseek-ai/DeepSeek-V4-Flash"]
    assert manifest.tokenizer_revision in manifest.tokenizer_url
    assert manifest.tokenizer_revision in manifest.config_url
    assert manifest.config_sha256 == "B628E63398A645ABC711D92207F8737DD8140F7A4EF1E0A5B3616019E0DDD818"
    with pytest.raises(ModelCapabilityError):
        manifest.capability_for("deepseek-ai/DeepSeek-V3.2")
    with pytest.raises(ModelCapabilityError):
        model_capability_for("deepseek-ai/DeepSeek-V4-Flash ")
