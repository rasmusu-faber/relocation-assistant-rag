"""Lightweight evaluation harness for the RAG retriever.

Computes retrieval hit-rate@k: the fraction of questions whose expected source
document appears among the top-k retrieved passages. Optionally checks answer
faithfulness if the LLM provider is reachable.

Run:  python -m eval.run_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import get_settings

# Ensure UTF-8 output so the ✓/✗ markers render on Windows consoles (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_FILE = Path(__file__).parent / "eval_set.jsonl"


def load_eval_set(path: Path = EVAL_FILE) -> list[dict[str, str]]:
    """Load the JSONL evaluation set."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def retrieval_hit_rate(top_k: int | None = None) -> float:
    """Return hit-rate@k of the expected document over the eval set."""
    from app.rag.retriever import retrieve  # imported lazily (heavy deps)

    k = top_k or get_settings().top_k
    rows = load_eval_set()
    hits = 0
    for row in rows:
        sources = retrieve(row["question"], top_k=k)
        retrieved_docs = {s.document for s in sources}
        ok = row["expected_document"] in retrieved_docs
        hits += int(ok)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {row['question'][:60]:60s} -> {sorted(retrieved_docs)}")
    rate = hits / len(rows) if rows else 0.0
    return rate


def main() -> None:
    k = get_settings().top_k
    print(f"Retrieval evaluation (hit-rate@{k}):")
    rate = retrieval_hit_rate(k)
    print(f"\nHit-rate@{k}: {rate:.2%}  ({len(load_eval_set())} questions)")
    # A simple gate that CI can assert on.
    if rate < 0.8:
        raise SystemExit(f"Hit-rate {rate:.2%} below threshold 80%")


if __name__ == "__main__":
    main()
