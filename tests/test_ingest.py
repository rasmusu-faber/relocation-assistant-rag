"""Tests for document loading / front-matter parsing.

``parse_front_matter`` is pure Python, so these run without the ML stack.
"""
from __future__ import annotations

from app.rag.ingest import parse_front_matter


def test_no_front_matter_returns_text_unchanged():
    text = "# Title\n\nSome body text."
    meta, body = parse_front_matter(text)
    assert meta == {}
    assert body == text


def test_parses_keys_and_strips_block_from_body():
    text = (
        "---\n"
        "source_name: Your Europe — Bank accounts in the EU\n"
        "source_url: https://europa.eu/youreurope/example\n"
        "---\n"
        "\n"
        "# Opening a bank account\n"
    )
    meta, body = parse_front_matter(text)
    assert meta["source_name"] == "Your Europe — Bank accounts in the EU"
    assert meta["source_url"] == "https://europa.eu/youreurope/example"
    # The front matter must not leak into the body (and thus into a chunk).
    assert body.startswith("# Opening a bank account")
    assert "source_url" not in body


def test_url_containing_colons_is_kept_intact():
    """Values are split on the first colon only, so URLs survive."""
    text = "---\nsource_url: https://example.org/a:b?x=1\n---\n\nBody"
    meta, _ = parse_front_matter(text)
    assert meta["source_url"] == "https://example.org/a:b?x=1"


def test_unterminated_front_matter_is_treated_as_body():
    text = "---\nsource_url: https://example.org\n\n# Title"
    meta, body = parse_front_matter(text)
    assert meta == {}
    assert body == text
