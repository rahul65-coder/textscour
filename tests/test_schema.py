"""
Unit tests for textscour.schema: Text extractor compilation, ID resolution, domain extraction.
"""

from __future__ import annotations

import pytest

from textscour.config import TextSpec
from textscour.schema import (
    compile_extractor,
    extract_domain,
    extract_id,
)


def test_extractor_field_mode():
    """Verify field extractor retrieves named field."""
    spec = TextSpec(mode="field", field="article_content")
    extractor = compile_extractor(spec)

    doc = {"article_content": "Detailed article text here.", "title": "My Title"}
    assert extractor(doc) == "Detailed article text here."

    doc_missing = {"title": "No article content"}
    assert extractor(doc_missing) is None


def test_extractor_path_mode():
    """Verify path extractor traverses nested dictionaries and lists."""
    spec = TextSpec(mode="path", field="meta.payload.0.body")
    extractor = compile_extractor(spec)

    doc = {
        "meta": {
            "payload": [
                {"body": "Nested text inside list"}
            ]
        }
    }
    assert extractor(doc) == "Nested text inside list"

    # Out of bounds index or missing key returns None
    assert extractor({"meta": {"payload": []}}) is None
    assert extractor({"meta": None}) is None


def test_extractor_concat_mode():
    """Verify concat extractor joins specified fields with separator."""
    spec = TextSpec(mode="concat", fields=["title", "abstract", "body"], sep=" --- ")
    extractor = compile_extractor(spec)

    doc = {
        "title": "Quantum Computing",
        "abstract": "An overview.",
        "body": "Full experimental details.",
    }
    expected = "Quantum Computing --- An overview. --- Full experimental details."
    assert extractor(doc) == expected

    # Skips missing or empty fields
    doc_partial = {"title": "Header", "body": "Paragraph"}
    assert extractor(doc_partial) == "Header --- Paragraph"


def test_extractor_template_mode():
    """Verify template extractor formats string variables."""
    spec = TextSpec(mode="template", template="# {title}\n\n{content}")
    extractor = compile_extractor(spec)

    doc = {"title": "Chapter 1", "content": "It was the best of times."}
    assert extractor(doc) == "# Chapter 1\n\nIt was the best of times."


def test_extractor_messages_mode():
    """Verify messages extractor converts chat messages to readable text."""
    spec = TextSpec(mode="messages", field="chat", sep="\n", role_filter=["user", "assistant"])
    extractor = compile_extractor(spec)

    doc = {
        "chat": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
    }
    extracted = extractor(doc)
    assert extracted is not None
    assert "system" not in extracted  # Filtered out
    assert "user: What is 2+2?" in extracted
    assert "assistant: 2+2 equals 4." in extracted


def test_extractor_auto_mode():
    """Verify auto extractor discovers common text field names."""
    spec = TextSpec(mode="auto")
    extractor = compile_extractor(spec)

    assert extractor({"text": "Found in text field"}) == "Found in text field"
    assert extractor({"content": "Found in content field"}) == "Found in content field"
    assert extractor({"body": "Found in body field"}) == "Found in body field"
    assert extractor({"custom_col": "Longest text snippet in dictionary"}) == "Longest text snippet in dictionary"


def test_extract_id():
    """Verify document ID resolution with field priority and fallback."""
    doc = {"uid": "custom-1234", "text": "sample"}
    assert extract_id(doc, id_field="uid", default_id="fallback") == "custom-1234"

    doc_auto = {"id": "auto-5678", "text": "sample"}
    assert extract_id(doc_auto, id_field=None, default_id="fallback") == "auto-5678"

    doc_none = {"text": "no id"}
    assert extract_id(doc_none, id_field=None, default_id="fallback-99") == "fallback-99"


def test_extract_domain():
    """Verify domain extraction including URL host parsing."""
    doc = {"domain": "tech.news.com"}
    assert extract_domain(doc, domain_field=None) == "tech.news.com"

    doc_url = {"url": "https://sub.example.org/path/to/page"}
    assert extract_domain(doc_url, domain_field=None) == "sub.example.org"

    doc_default = {"text": "hello"}
    assert extract_domain(doc_default, domain_field=None, default_domain="default_dom") == "default_dom"


def test_extractor_never_returns_id_as_text():
    """Verify auto and nested extractors never return document ID as text."""
    spec_auto = TextSpec(mode="auto")
    extractor_auto = compile_extractor(spec_auto)

    # Document with only ID fields must return None (quarantined as E-SCHEMA-001)
    doc_id_only = {"id": "doc_12345", "uid": "uid_6789", "uuid": "uuid_0000"}
    assert extractor_auto(doc_id_only) is None

    # Document with ID and nested body must return the nested body, NEVER the ID
    doc_nested = {
        "id": "doc_12345",
        "data": {
            "article": {
                "body": "This is real nested article content.",
                "tags": ["science"],
            }
        },
        "domain": "nested_path",
    }
    extracted = extractor_auto(doc_nested)
    assert extracted == "This is real nested article content."
    assert "doc_12345" not in extracted

    # Path mode targeting an ID field must return None
    spec_path_id = TextSpec(mode="path", field="meta.id")
    extractor_path_id = compile_extractor(spec_path_id)
    assert extractor_path_id({"meta": {"id": "nested_id_999"}}) is None


def test_extractor_path_missing_or_empty_returns_none():
    """Verify path extractor returns None on missing or empty string fields."""
    spec = TextSpec(mode="path", field="payload.text")
    extractor = compile_extractor(spec)

    assert extractor({"payload": {"text": "   "}}) is None
    assert extractor({"payload": {"text": ""}}) is None
    assert extractor({"payload": {}}) is None
    assert extractor({"other": "value"}) is None
    assert extractor({"payload": None}) is None

