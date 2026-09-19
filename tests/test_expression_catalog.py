from __future__ import annotations

import hashlib
import json

import pytest

from qichi.expression.catalog import (
    ExpressionCatalogError,
    QQExpressionCatalog,
    load_qq_expression_catalog,
)


SOURCE = "https://github.com/NapNeko/NapCatQQ/tree/0c4c371eee209bf6449647efdac5ab7649e2b9ef"
COMMIT = "0c4c371eee209bf6449647efdac5ab7649e2b9ef"
FACES = {
    "smile": 14, "laugh": 182, "shy": 6, "smug": 4, "annoyed": 22,
    "confused": 32, "sleepy": 25, "sad": 15, "heart": 66,
}


def data(**overrides):
    value = {"schema": "qichi.qq-expression-catalog/v1", "version": 1, "source": SOURCE,
             "source_commit": COMMIT, "source_hash": "0" * 64, "faces": FACES.copy(), "reactions": {}}
    value.update(overrides)
    canonical = dict(value)
    canonical.pop("source_hash", None)
    value["source_hash"] = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def test_loads_example_and_resolves_known_or_unknown_names():
    catalog = load_qq_expression_catalog("data/qq-expression-catalog.example.json")
    assert dict(catalog.faces) == FACES
    assert catalog.resolve("shy") == 6
    assert catalog.resolve("missing") is None
    assert catalog.reactions == {}
    assert catalog.resolve_reaction("heart") is None


def test_catalog_provenance_hash_is_strict_and_unknown_reaction_is_noop():
    value = data()
    value["source_hash"] = "0" * 64
    with pytest.raises(ExpressionCatalogError, match="hash"):
        QQExpressionCatalog.from_data(value)
    assert QQExpressionCatalog.from_data(data()).resolve_reaction("unknown") is None


@pytest.mark.parametrize("field", ["version", "source_commit", "reactions"])
def test_required_provenance_or_channel_fields_cannot_be_missing(field):
    value = data()
    value.pop(field)
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(value)


def test_result_is_immutable():
    catalog = QQExpressionCatalog.from_data(data())
    with pytest.raises(TypeError):
        catalog.faces["new"] = 1
    with pytest.raises(AttributeError):
        catalog.faces.clear()  # type: ignore[attr-defined]


@pytest.mark.parametrize("faces", [
    {},
    {"Bad Name": 6},
    {"shy": True},
    {"shy": -1},
    {"shy": 2**31},
    {"shy": 6, "other": 6},
])
def test_direct_constructor_enforces_face_invariants(faces):
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog("qichi.qq-expression-catalog/v1", 1, SOURCE, COMMIT, "0" * 64, faces, {})


def test_direct_constructor_enforces_source_and_remains_immutable():
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog("qichi.qq-expression-catalog/v1", 1, "   ", COMMIT, "0" * 64, {"shy": 6}, {})
    catalog = QQExpressionCatalog("qichi.qq-expression-catalog/v1", 1, SOURCE, COMMIT, "0" * 64, {"shy": 6}, {})
    assert catalog.resolve("shy") == 6
    with pytest.raises(TypeError):
        catalog.faces["other"] = 7


@pytest.mark.parametrize("value", [None, [], "text", {"schema": "x"}])
def test_non_object_or_invalid_schema_is_rejected(value):
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(value)


def test_unknown_root_and_face_fields_are_rejected():
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(data(extra=True))
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(data(faces={"shy": {"id": 6}}))


def test_empty_or_missing_catalog_directory_is_rejected(tmp_path):
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.load(tmp_path / "missing.json")
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(data(faces={}))


@pytest.mark.parametrize("name", ["", "Shy", "shy name", "中文", "shy/"])
def test_semantic_names_are_strict(name):
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(data(faces={name: 6}))


@pytest.mark.parametrize("value", ["6.0", "+6", "-1", "６", True, -1, 2**31])
def test_ids_must_be_decimal_and_in_range(value):
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(data(faces={"shy": value}))


def test_duplicate_platform_ids_are_rejected():
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.from_data(data(faces={"shy": 6, "other": 6}))


def test_json_parser_is_used_and_load_errors_are_safe(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data()), encoding="utf-8")
    assert QQExpressionCatalog.load(path).resolve("heart") == 66
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ExpressionCatalogError):
        QQExpressionCatalog.load(path)
