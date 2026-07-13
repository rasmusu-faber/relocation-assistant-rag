"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    top_k: int | None = Field(None, ge=1, le=20, description="Override number of passages to retrieve.")


class Source(BaseModel):
    """A retrieved passage that supported the answer."""

    document: str = Field(..., description="Source document name.")
    snippet: str = Field(..., description="The retrieved text chunk.")
    score: float = Field(..., description="Similarity score (higher = more relevant).")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    collection: str


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_provider: str
    collection: str
