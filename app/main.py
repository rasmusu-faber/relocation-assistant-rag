"""FastAPI application exposing the RAG assistant."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app import __version__
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
)

app = FastAPI(
    title="Relocation Assistant (RAG)",
    version=__version__,
    description="RAG assistant with source citations for relocating to Poland.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check; also reports the active provider and collection."""
    settings = get_settings()
    return HealthResponse(
        llm_provider=settings.llm_provider,
        collection=settings.collection_name,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint() -> IngestResponse:
    """(Re-)ingest the documents in the configured data directory."""
    from app.rag.ingest import ingest

    summary = ingest()
    return IngestResponse(**summary)  # type: ignore[arg-type]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question, grounded in retrieved documents."""
    from app.rag.pipeline import answer_question

    try:
        return answer_question(req.question, top_k=req.top_k)
    except Exception as exc:  # noqa: BLE001 - surface a clean API error
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc
