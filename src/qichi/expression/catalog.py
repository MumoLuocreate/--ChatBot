"""Verified, read-only mapping of semantic QQ face names to QSid values."""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


class ExpressionCatalogError(ValueError):
    """The catalog is missing, malformed, or does not satisfy its schema."""


_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SCHEMA = "qichi.qq-expression-catalog/v1"
_SOURCE_COMMIT = "0c4c371eee209bf6449647efdac5ab7649e2b9ef"
_MAX_QSID = 2**31 - 1


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExpressionCatalogError(f"{label} must be an object")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ExpressionCatalogError(f"{label} contains unknown fields")


def _q_sid(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ExpressionCatalogError(f"{label} must be a decimal QSid")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value and value.isascii() and value.isdecimal():
        result = int(value, 10)
    else:
        raise ExpressionCatalogError(f"{label} must be a decimal QSid")
    if result < 0 or result > _MAX_QSID:
        raise ExpressionCatalogError(f"{label} is out of range")
    return result


def _validate_mapping(value: Any, label: str, *, allow_empty: bool) -> dict[str, int]:
    faces = _object(value, label)
    if not faces and not allow_empty:
        raise ExpressionCatalogError(f"{label} must not be empty")
    result: dict[str, int] = {}
    ids: set[int] = set()
    for name, q_sid in faces.items():
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ExpressionCatalogError(f"invalid semantic {label[:-1]} name")
        parsed = _q_sid(q_sid, f"{label}.{name}")
        if parsed in ids:
            raise ExpressionCatalogError("duplicate platform face ID")
        result[name] = parsed
        ids.add(parsed)
    return result


@dataclass(frozen=True, slots=True)
class QQExpressionCatalog:
    """Immutable semantic-name to QQ QSid catalog."""

    schema: str
    version: int
    source: str
    source_commit: str
    source_hash: str
    _faces: Mapping[str, int]
    _reactions: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ExpressionCatalogError("unsupported catalog schema")
        if self.version != 1:
            raise ExpressionCatalogError("unsupported catalog version")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ExpressionCatalogError("source must be a non-empty string")
        if self.source_commit != _SOURCE_COMMIT:
            raise ExpressionCatalogError("source commit is not the verified official commit")
        if not isinstance(self.source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", self.source_hash) is None:
            raise ExpressionCatalogError("source_hash must be a lowercase SHA-256")
        object.__setattr__(self, "_faces", MappingProxyType(_validate_mapping(self._faces, "faces", allow_empty=False)))
        object.__setattr__(self, "_reactions", MappingProxyType(_validate_mapping(self._reactions, "reactions", allow_empty=True)))

    @property
    def faces(self) -> Mapping[str, int]:
        return self._faces

    @property
    def reactions(self) -> Mapping[str, int]:
        return self._reactions

    def resolve_reaction(self, semantic_name: str) -> int | None:
        if not isinstance(semantic_name, str):
            raise TypeError("semantic_name must be a string")
        return self._reactions.get(semantic_name)

    def resolve(self, semantic_name: str) -> int | None:
        if not isinstance(semantic_name, str):
            raise TypeError("semantic_name must be a string")
        return self._faces.get(semantic_name)

    def get(self, semantic_name: str) -> int | None:
        return self.resolve(semantic_name)

    @classmethod
    def from_data(cls, value: Any) -> "QQExpressionCatalog":
        root = _object(value, "catalog")
        _strict_keys(root, {"schema", "version", "source", "source_commit", "source_hash", "faces", "reactions"}, "catalog")
        if root.get("schema") != _SCHEMA:
            raise ExpressionCatalogError("unsupported catalog schema")
        if root.get("version") != 1:
            raise ExpressionCatalogError("unsupported catalog version")
        source = root.get("source")
        if not isinstance(source, str) or not source:
            raise ExpressionCatalogError("source must be a non-empty string")
        source_commit = root.get("source_commit")
        source_hash = root.get("source_hash")
        if source_commit != _SOURCE_COMMIT:
            raise ExpressionCatalogError("source commit is not the verified official commit")
        canonical = dict(root)
        canonical.pop("source_hash", None)
        expected_hash = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if source_hash != expected_hash:
            raise ExpressionCatalogError("catalog source hash does not match content")
        return cls(_SCHEMA, 1, source, source_commit, source_hash, root.get("faces"), root.get("reactions"))

    @classmethod
    def load(cls, path: str | Path) -> "QQExpressionCatalog":
        try:
            catalog_path = Path(path)
            with catalog_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExpressionCatalogError("catalog could not be loaded") from None
        return cls.from_data(value)


ExpressionCatalog = QQExpressionCatalog


def load_qq_expression_catalog(path: str | Path) -> QQExpressionCatalog:
    return QQExpressionCatalog.load(path)
