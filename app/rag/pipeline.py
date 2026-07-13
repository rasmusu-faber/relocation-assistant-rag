"""End-to-end RAG pipeline: retrieve relevant passages, then generate an answer."""
from __future__ import annotations

from app.models import ChatResponse
from app.rag.generator import generate
from app.rag.retriever import retrieve


def answer_question(question: str, top_k: int | None = None) -> ChatResponse:
    """Run retrieval + generation and return an answer with its sources."""
    sources = retrieve(question, top_k=top_k)
    if not sources:
        return ChatResponse(
            answer="I couldn't find anything relevant in the knowledge base. "
            "Try ingesting documents first or rephrasing your question.",
            sources=[],
        )
    context_blocks = [s.snippet for s in sources]
    answer = generate(question, context_blocks)
    return ChatResponse(answer=answer, sources=sources)
