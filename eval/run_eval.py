"""Lightweight evaluation harness for the RAG retriever.

Computes retrieval metrics over an eval set of (question, expected-source) pairs:

* **hit-rate@k** — fraction of questions whose expected document appears among the
  top-k retrieved passages (this is the CI quality gate).
* **hit-rate@1** — stricter: the expected document is the very first passage.
* **MRR@k** — mean reciprocal rank of the first correct passage (rewards ranking
  the right document higher, not just having it somewhere in the top-k).

Retrieval-only, so it runs deterministically in CI without an LLM or API key.
(Answer-faithfulness scoring is planned but not implemented yet.)

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
GATE_THRESHOLD = 0.8


def load_eval_set(path: Path = EVAL_FILE) -> list[dict[str, str]]:
    """Load the JSONL evaluation set."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def first_hit_rank(retrieved_docs: list[str], expected: str) -> int | None:
    """1-indexed rank of the first ``expected`` document, or ``None`` if absent.

    ``retrieved_docs`` is the list of source document names in retrieval order
    (one entry per retrieved passage; the same document may repeat).
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc == expected:
            return rank
    return None


def evaluate(top_k: int | None = None) -> dict[str, float | int]:
    """Evaluate the retriever over the eval set and return metric values.

    Returns a dict with ``hit_rate``, ``hit_rate_at_1``, ``mrr``, ``k`` and ``n``.
    Also prints a per-question trace.
    """
    from app.rag.retriever import retrieve  # imported lazily (heavy deps)

    k = top_k or get_settings().top_k
    rows = load_eval_set()

    hits = 0
    hits_at_1 = 0
    reciprocal_ranks = 0.0
    for row in rows:
        sources = retrieve(row["question"], top_k=k)
        retrieved_docs = [s.document for s in sources]
        rank = first_hit_rank(retrieved_docs, row["expected_document"])

        hits += int(rank is not None)
        hits_at_1 += int(rank == 1)
        reciprocal_ranks += (1.0 / rank) if rank else 0.0

        flag = "✓" if rank is not None else "✗"
        pos = f"@{rank}" if rank is not None else "miss"
        print(f"  {flag} {row['question'][:56]:56s} [{pos:>4}] -> {sorted(set(retrieved_docs))}")

    n = len(rows) or 1
    return {
        "hit_rate": hits / n,
        "hit_rate_at_1": hits_at_1 / n,
        "mrr": reciprocal_ranks / n,
        "k": k,
        "n": len(rows),
    }


def retrieval_hit_rate(top_k: int | None = None) -> float:
    """Backward-compatible helper: return hit-rate@k over the eval set."""
    return float(evaluate(top_k)["hit_rate"])


def main() -> None:
    k = get_settings().top_k
    print(f"Retrieval evaluation (top-k = {k}):")
    m = evaluate(k)
    print(
        f"\nhit-rate@{k}: {m['hit_rate']:.2%}   "
        f"hit-rate@1: {m['hit_rate_at_1']:.2%}   "
        f"MRR@{k}: {m['mrr']:.3f}   ({m['n']} questions)"
    )
    # The CI gate asserts only on hit-rate@k; the other metrics are informational.
    if m["hit_rate"] < GATE_THRESHOLD:
        raise SystemExit(
            f"Hit-rate {m['hit_rate']:.2%} below threshold {GATE_THRESHOLD:.0%}"
        )


if __name__ == "__main__":
    main()
