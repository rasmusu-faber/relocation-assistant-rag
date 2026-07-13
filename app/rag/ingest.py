"""Ingest documents: load -> chunk -> embed -> store in Chroma.

Run as a script:  python -m app.rag.ingest
"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.chunking import chunk_text


def load_documents(data_dir: str) -> list[tuple[str, str]]:
    """Load ``.md`` and ``.txt`` files from ``data_dir``.

    Returns a list of (filename, text) tuples.
    """
    path = Path(data_dir)
    docs: list[tuple[str, str]] = []
    for file in sorted(path.glob("**/*")):
        if file.suffix.lower() in {".md", ".txt"} and file.is_file():
            docs.append((file.name, file.read_text(encoding="utf-8")))
    return docs


def ingest(data_dir: str | None = None) -> dict[str, int | str]:
    """Ingest all documents in ``data_dir`` into the vector store.

    Returns a small summary dict: documents, chunks, collection.
    """
    from app.rag.store import embed, get_collection

    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    collection = get_collection()

    documents = load_documents(data_dir)
    if not documents:
        print(f"No documents found in '{data_dir}'.")
        return {"documents": 0, "chunks": 0, "collection": settings.collection_name}

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str]] = []
    for name, text in documents:
        for i, chunk in enumerate(
            chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        ):
            ids.append(f"{name}::{i}")
            texts.append(chunk)
            metadatas.append({"document": name, "chunk": str(i)})

    # Upsert keeps re-ingestion idempotent.
    collection.upsert(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metadatas)

    summary = {
        "documents": len(documents),
        "chunks": len(texts),
        "collection": settings.collection_name,
    }
    print(f"Ingested {summary['documents']} document(s), {summary['chunks']} chunk(s).")
    return summary


if __name__ == "__main__":
    ingest()
