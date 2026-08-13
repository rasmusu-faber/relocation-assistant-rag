"""Tests for the pipeline's retrieve/rerank wiring (no heavy deps loaded).

Retrieval, generation and re-ranking are all monkeypatched, so these assert the
control flow (does re-ranking widen the pool and re-order?) without a model.
"""
from __future__ import annotations

from types import SimpleNamespace

import app.rag.pipeline as pipeline
from app.models import Source


def _settings(**over):
    base = dict(top_k=4, rerank_enabled=False, rerank_candidates=20)
    base.update(over)
    return SimpleNamespace(**base)


def _sources(*docs: str) -> list[Source]:
    return [Source(document=d, snippet=d, score=0.5) for d in docs]


def test_pipeline_without_rerank_uses_retriever_top_k(monkeypatch):
    monkeypatch.setattr(pipeline, "get_settings", lambda: _settings(rerank_enabled=False))
    calls: dict = {}

    def fake_retrieve(q, top_k=None):
        calls["top_k"] = top_k
        return _sources("a.md", "b.md")

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "generate", lambda q, ctx: "ans")

    resp = pipeline.answer_question("q")
    assert calls["top_k"] == 4  # retrieves top_k directly, no widening
    assert [s.document for s in resp.sources] == ["a.md", "b.md"]


def test_pipeline_with_rerank_widens_pool_and_reorders(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "get_settings",
        lambda: _settings(rerank_enabled=True, rerank_candidates=10),
    )
    calls: dict = {}

    def fake_retrieve(q, top_k=None):
        calls["top_k"] = top_k
        return _sources("a.md", "b.md", "c.md")

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)

    import app.rag.reranker as reranker

    def fake_rerank(q, sources, top_k=None):
        calls["reranked"] = True
        return list(reversed(sources))[:top_k]

    monkeypatch.setattr(reranker, "rerank", fake_rerank)
    monkeypatch.setattr(pipeline, "generate", lambda q, ctx: "ans")

    resp = pipeline.answer_question("q", top_k=2)
    assert calls["top_k"] == 10  # widened to the candidate pool (>= k)
    assert calls.get("reranked") is True
    assert [s.document for s in resp.sources] == ["c.md", "b.md"]  # reordered, cut to 2
