"""Ingest documents: load -> chunk -> embed -> store in Chroma.

Run as a script:  python -m app.rag.ingest
"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.chunking import chunk_text


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split optional ``---`` front matter from the document body.

    Front matter is a small block of ``key: value`` lines used to attach
    provenance to a document, e.g.::

        ---
        source_name: Your Europe — Your health insurance cover
        source_url: https://europa.eu/...
        ---

    Kept deliberately simple (no YAML dependency) since only flat string values
    are needed. Returns ``({}, text)`` when there is no front matter.
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            meta: dict[str, str] = {}
            for line in lines[1:i]:
                key, sep, value = line.partition(":")
                if sep and key.strip():
                    meta[key.strip()] = value.strip()
            return meta, "\n".join(lines[i + 1 :]).lstrip("\n")

    # Unterminated front matter: treat the whole file as body.
    return {}, text


def load_documents(data_dir: str) -> list[tuple[str, str, dict[str, str]]]:
    """Load ``.md`` and ``.txt`` files from ``data_dir``.

    Returns a list of (filename, body_text, front_matter) tuples. The front
    matter is stripped from the body so it never ends up inside a chunk.
    """
    path = Path(data_dir)
    docs: list[tuple[str, str, dict[str, str]]] = []
    for file in sorted(path.glob("**/*")):
        if file.suffix.lower() in {".md", ".txt"} and file.is_file():
            meta, body = parse_front_matter(file.read_text(encoding="utf-8"))
            docs.append((file.name, body, meta))
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
    for name, text, front_matter in documents:
        for i, chunk in enumerate(
            chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        ):
            ids.append(f"{name}::{i}")
            texts.append(chunk)
            # Provenance travels with the chunk so answers can cite the original
            # source, not just the local filename. Chroma metadata must be
            # scalars, so missing values become empty strings.
            metadatas.append(
                {
                    "document": name,
                    "chunk": str(i),
                    "source_url": front_matter.get("source_url", ""),
                    "source_name": front_matter.get("source_name", ""),
                }
            )

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
