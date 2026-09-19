"""Cross-encoder re-ranking of retrieved passages.

The bi-encoder retriever (see ``retriever.py``) scores the question and each
passage *independently* and compares them by cosine similarity — fast, but coarse:
it never looks at a query and a passage together. A **cross-encoder** scores the
``(question, passage)`` pair jointly and ranks far more accurately, but it is too
expensive to run over a whole corpus. The standard remedy is two stages: let the
retriever fetch a wider candidate set cheaply, then re-order those candidates with
the cross-encoder and keep the best ``top_k``.

This module is that re-ranking capability in isolation: a pure, side-effect-free
function that only *re-orders* the ``Source`` objects it is given. Widening the
retriever pool and gating this behind a config flag in the pipeline is a separate
step.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.models import Source
from app.observability import observe


@lru_cache(maxsize=1)
def get_reranker() -> Any:
    """Return a cached CrossEncoder model (lazy import: heavy dependency)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


@observe(name="rerank")
def rerank(
    question: str, sources: list[Source], top_k: int | None = None
) -> list[Source]:
    """Re-order ``sources`` by cross-encoder relevance; return the best ``top_k``.

    Only the ordering changes: each ``Source`` is returned unmodified, so its
    ``score`` still carries the retriever's similarity (kept for provenance rather
    than overwritten with the cross-encoder's un-normalised relevance logit).
    Returns at most ``top_k`` sources, or all of them re-ordered when ``top_k`` is
    ``None``. An empty input returns ``[]`` without loading the model.
    """
    if not sources:
        return []

    model = get_reranker()
    scores = model.predict([(question, s.snippet) for s in sources])

    # Sort by cross-encoder score, descending. Key on the score alone so equal
    # scores never fall back to comparing (unorderable) Source objects; Python's
    # stable sort then preserves the retriever's original order on ties.
    ranked = [
        s
        for _, s in sorted(
            zip(scores, sources, strict=True), key=lambda p: p[0], reverse=True
        )
    ]

    k = top_k if top_k is not None else len(ranked)
    return ranked[:k]
