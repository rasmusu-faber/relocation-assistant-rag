"""Unit tests for the chunking utility (no heavy dependencies needed)."""
from __future__ import annotations

import pytest

from app.rag.chunking import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello world", chunk_size=100, overlap=20) == ["hello world"]


def test_empty_text_returns_empty_list():
    assert chunk_text("   ", chunk_size=100, overlap=20) == []


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_cover_all_words():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    joined = " ".join(chunks)
    for i in (0, 150, 299):
        assert f"w{i}" in joined


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
