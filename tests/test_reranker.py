"""Unit tests for the cross-encoder re-ranking logic.

The real CrossEncoder is never loaded: ``get_reranker`` is monkeypatched with a
stub whose scores we control, so the ordering/truncation logic is tested in
isolation and the suite stays fast (no model download).
"""
from __future__ import annotations

import app.rag.reranker as reranker
from app.models import Source


def _src(doc: str, snippet: str = "x", score: float = 0.5) -> Source:
    return Source(document=doc, snippet=snippet, score=score)


class _StubModel:
    """Returns pre-set scores, one per (question, passage) pair, in order."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == len(self._scores)
        return self._scores


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    sources = [_src("a"), _src("b"), _src("c")]  # retriever order a, b, c
    # Cross-encoder prefers c > a > b.
    monkeypatch.setattr(reranker, "get_reranker", lambda: _StubModel([0.1, 0.0, 0.9]))
    out = reranker.rerank("q", sources)
    assert [s.document for s in out] == ["c", "a", "b"]


def test_rerank_respects_top_k(monkeypatch):
    sources = [_src("a"), _src("b"), _src("c")]
    monkeypatch.setattr(reranker, "get_reranker", lambda: _StubModel([0.1, 0.0, 0.9]))
    out = reranker.rerank("q", sources, top_k=2)
    assert [s.document for s in out] == ["c", "a"]


def test_rerank_preserves_retriever_score(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker", lambda: _StubModel([0.9]))
    out = reranker.rerank("q", [_src("a", score=0.42)])
    assert out[0].score == 0.42  # retriever similarity left untouched
    assert out[0].document == "a"


def test_rerank_ties_keep_retriever_order(monkeypatch):
    sources = [_src("a"), _src("b"), _src("c")]
    monkeypatch.setattr(reranker, "get_reranker", lambda: _StubModel([0.5, 0.5, 0.5]))
    out = reranker.rerank("q", sources)
    assert [s.document for s in out] == ["a", "b", "c"]


def test_rerank_empty_input_does_not_load_model(monkeypatch):
    def _boom():
        raise AssertionError("model must not be loaded for empty input")

    monkeypatch.setattr(reranker, "get_reranker", _boom)
    assert reranker.rerank("q", []) == []
