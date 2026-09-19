"""Exact token counting from hash-verified local tokenizer artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer


class TokenCounter(Protocol):
    """Counts text using the tokenizer for the selected model."""

    def count_text(self, text: str) -> int:
        """Return the exact nonnegative token count for *text*."""


class TokenizerArtifactError(ValueError):
    """The local tokenizer artifact could not be verified and loaded."""


class HashedTokenizerCounter:
    """A token counter whose tokenizer is verified before it is deserialized."""

    def __init__(
        self,
        artifact_path: str | Path,
        expected_sha256: str,
        *,
        add_special_tokens: bool = False,
    ) -> None:
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("expected_sha256 must be a SHA-256 hexadecimal digest")
        try:
            expected_digest = bytes.fromhex(expected_sha256)
        except ValueError as exc:
            raise ValueError("expected_sha256 must be a SHA-256 hexadecimal digest") from exc
        if len(expected_digest) != 32:
            raise ValueError("expected_sha256 must be a SHA-256 hexadecimal digest")
        if type(add_special_tokens) is not bool:
            raise TypeError("add_special_tokens must be a bool")

        try:
            artifact = Path(artifact_path)
        except TypeError as exc:
            raise TypeError("artifact_path must be a local path") from exc
        if not artifact.is_file():
            raise TokenizerArtifactError("tokenizer artifact is unavailable")

        digest = self._hash_file(artifact)
        if digest.lower() != expected_sha256.lower():
            raise TokenizerArtifactError("tokenizer artifact hash mismatch")
        try:
            tokenizer = Tokenizer.from_file(str(artifact))
        except Exception as exc:
            raise TokenizerArtifactError("tokenizer artifact could not be loaded") from exc

        self._tokenizer = tokenizer
        self._add_special_tokens = add_special_tokens

    @staticmethod
    def _hash_file(artifact: Path) -> str:
        digest = sha256()
        try:
            with artifact.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise TokenizerArtifactError("tokenizer artifact is unavailable") from exc
        return digest.hexdigest()

    def count_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        count = len(self._tokenizer.encode(text, add_special_tokens=self._add_special_tokens).ids)
        if count < 0:  # Defensive: tokenizers must never return a negative sequence length.
            raise RuntimeError("tokenizer returned an invalid token count")
        return count

    def __repr__(self) -> str:
        return f"HashedTokenizerCounter(add_special_tokens={self._add_special_tokens!r})"
