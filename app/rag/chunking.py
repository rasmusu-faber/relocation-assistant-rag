"""Pure-Python text chunking (no heavy dependencies, easily unit-tested)."""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into overlapping, word-aware chunks.

    Chunks are at most ``chunk_size`` characters; consecutive chunks share
    ``overlap`` characters of context. Splitting happens on whitespace so words
    are not cut in half.

    Args:
        text: The input text.
        chunk_size: Maximum chunk length in characters.
        overlap: Number of characters of overlap between consecutive chunks.

    Returns:
        A list of non-empty text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Prefer to break on the last whitespace within the window.
        if end < n:
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
