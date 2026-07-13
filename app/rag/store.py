"""Shared embedding model and Chroma vector-store access.

Heavy dependencies (sentence-transformers, chromadb) are imported lazily so the
rest of the package can be imported, tested, and type-checked without them.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Return a cached SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into vectors."""
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_collection() -> Any:
    """Return (creating if needed) the persistent Chroma collection."""
    import chromadb

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
