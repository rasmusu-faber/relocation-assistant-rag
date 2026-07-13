"""API tests that don't require the heavy ML stack to be installed at import time.

The /health route is pure; /chat is tested with retrieval + generation mocked,
so the test suite stays fast and runs in CI without a model server.
"""
from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.models import ChatResponse, Source  # noqa: E402

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body


def test_chat_uses_pipeline(monkeypatch):
    """/chat should return the pipeline's answer and sources."""
    fake = ChatResponse(
        answer="You apply at the municipal office.",
        sources=[Source(document="pesel.md", snippet="Go to the municipal office.", score=0.9)],
    )

    # Patch the symbol where main.py imports it (lazy import inside the handler).
    import app.rag.pipeline as pipeline

    monkeypatch.setattr(pipeline, "answer_question", lambda *a, **k: fake)

    resp = client.post("/chat", json={"question": "How do I get a PESEL?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake.answer
    assert body["sources"][0]["document"] == "pesel.md"
