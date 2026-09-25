from __future__ import annotations

from hashlib import sha256

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from qichi.dialogue.model_capability import ModelManifest
from qichi.dialogue.token_counter import HashedTokenizerCounter, TokenizerArtifactError


def _write_tokenizer(tmp_path, *, with_special_tokens: bool = False):
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4, "你": 5, "好": 6, "🙂": 7}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    if with_special_tokens:
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )
    artifact = tmp_path / "tokenizer.json"
    tokenizer.save(str(artifact))
    return artifact, sha256(artifact.read_bytes()).hexdigest()


def test_counts_tokens_with_verified_local_artifact(tmp_path):
    artifact, digest = _write_tokenizer(tmp_path)
    counter = HashedTokenizerCounter(artifact, digest)

    assert counter.count_text("") == 0
    assert counter.count_text("hello world") == 2
    assert counter.count_text("你 好 🙂") == 3


def test_special_token_policy_is_explicit(tmp_path):
    artifact, digest = _write_tokenizer(tmp_path, with_special_tokens=True)

    without_specials = HashedTokenizerCounter(artifact, digest, add_special_tokens=False)
    with_specials = HashedTokenizerCounter(artifact, digest, add_special_tokens=True)

    assert without_specials.count_text("hello") == 1
    assert with_specials.count_text("hello") == 3


def test_manifest_binds_tokenizer_hash_for_loading(tmp_path):
    artifact, digest = _write_tokenizer(tmp_path)
    manifest = ModelManifest(
        "test/model", 16, "test-revision", "https://example.test/tokenizer.json", digest,
        "https://example.test/config.json", "1" * 64,
    )
    assert manifest.load_token_counter(str(artifact)).count_text("hello") == 1


def test_missing_mismatched_or_unloadable_artifact_fails_closed(tmp_path):
    artifact, digest = _write_tokenizer(tmp_path)

    with pytest.raises(TokenizerArtifactError):
        HashedTokenizerCounter(tmp_path / "missing.json", digest)
    with pytest.raises(TokenizerArtifactError):
        HashedTokenizerCounter(artifact, "0" * 64)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not a tokenizer", encoding="utf-8")
    malformed_digest = sha256(malformed.read_bytes()).hexdigest()
    with pytest.raises(TokenizerArtifactError):
        HashedTokenizerCounter(malformed, malformed_digest)


def test_rejects_invalid_arguments_and_keeps_repr_non_sensitive(tmp_path):
    artifact, digest = _write_tokenizer(tmp_path)
    counter = HashedTokenizerCounter(artifact, digest)

    with pytest.raises(TypeError):
        counter.count_text(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HashedTokenizerCounter(artifact, "not-a-digest")
    with pytest.raises(TypeError):
        HashedTokenizerCounter(artifact, digest, add_special_tokens=1)  # type: ignore[arg-type]

    representation = repr(counter)
    assert str(artifact) not in representation
    assert digest not in representation
