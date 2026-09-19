"""Vector retrieval over the Chroma collection."""
from __future__ import annotations

from app.config import get_settings
from app.models import Source
from app.observability import observe


@observe(name="retrieve", as_type="retriever")
def retrieve(question: str, top_k: int | None = None) -> list[Source]:
    """Return the ``top_k`` most relevant passages for ``question``."""
    from app.rag.store import embed, get_collection

    settings = get_settings()
    k = top_k or settings.top_k
    collection = get_collection()

    result = collection.query(
        query_embeddings=embed([question]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    sources: list[Source] = []
    for text, meta, dist in zip(docs, metas, dists, strict=True):
        # Cosine distance -> similarity score in [0, 1] (higher is better).
        sources.append(
            Source(
                document=str(meta.get("document", "unknown")),
                snippet=text,
                score=round(1.0 - float(dist), 4),
                # Empty strings (no provenance recorded) become None.
                source_name=str(meta.get("source_name") or "") or None,
                source_url=str(meta.get("source_url") or "") or None,
            )
        )
    return sources
