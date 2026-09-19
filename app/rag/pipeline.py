"""End-to-end RAG pipeline: retrieve relevant passages, then generate an answer."""
from __future__ import annotations

from app.config import get_settings
from app.models import ChatResponse, Source
from app.observability import observe
from app.rag.generator import generate
from app.rag.retriever import retrieve


def _retrieve_sources(question: str, k: int) -> list[Source]:
    """Retrieve the top-``k`` passages, optionally re-ranked.

    When re-ranking is enabled, retrieve a wider candidate pool with the
    bi-encoder and re-order it with the cross-encoder down to ``k`` — this lets a
    passage the bi-encoder ranked lower still reach the top. Otherwise return the
    bi-encoder's own top-``k``.
    """
    settings = get_settings()
    if not settings.rerank_enabled:
        return retrieve(question, top_k=k)

    from app.rag.reranker import rerank  # lazy: heavy dependency, only when enabled

    pool = retrieve(question, top_k=max(settings.rerank_candidates, k))
    return rerank(question, pool, top_k=k)


@observe(name="rag_answer")
def answer_question(question: str, top_k: int | None = None) -> ChatResponse:
    """Run retrieval + generation and return an answer with its sources."""
    k = top_k or get_settings().top_k
    sources = _retrieve_sources(question, k)
    if not sources:
        return ChatResponse(
            answer="I couldn't find anything relevant in the knowledge base. "
            "Try ingesting documents first or rephrasing your question.",
            sources=[],
        )
    context_blocks = [s.snippet for s in sources]
    answer = generate(question, context_blocks)
    return ChatResponse(answer=answer, sources=sources)
