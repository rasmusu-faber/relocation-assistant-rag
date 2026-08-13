"""Measure what cross-encoder re-ranking buys, per eval set.

Runs each eval set twice — plain bi-encoder retrieval vs. retrieve-then-rerank —
and reports hit@1 / hit@k / MRR side by side.

Re-ranking cannot add a document the retriever's candidate pool missed, so on a
corpus of topically-distinct documents (the *easy* set) the metrics barely move.
Its job is to push the right document to rank 1 when several look similar, so the
payoff shows up on the *hard* set of confusable sibling documents — exactly where
hit@1 and MRR have room to rise.

Run:  python -m eval.rerank_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.config import get_settings
from app.models import Source
from eval.run_eval import first_hit_rank, load_eval_set

# Ensure UTF-8 output so any non-ASCII renders on Windows consoles (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_SETS = {
    "easy (distinct)": Path(__file__).parent / "eval_set.jsonl",
    "hard (siblings)": Path(__file__).parent / "eval_set_hard.jsonl",
}

RESULTS_FILE = Path(__file__).parent / "rerank_results.md"


def _sources(question: str, k: int, candidates: int, use_rerank: bool) -> list[Source]:
    from app.rag.retriever import retrieve

    if not use_rerank:
        return retrieve(question, top_k=k)

    from app.rag.reranker import rerank

    pool = retrieve(question, top_k=max(candidates, k))
    return rerank(question, pool, top_k=k)


def _metrics(
    rows: list[dict[str, str]], k: int, candidates: int, use_rerank: bool
) -> tuple[float, float, float]:
    """Return (hit@k, hit@1, MRR@k) over ``rows``."""
    hits = h1 = 0
    reciprocal = 0.0
    for row in rows:
        docs = [s.document for s in _sources(row["question"], k, candidates, use_rerank)]
        rank = first_hit_rank(docs, row["expected_document"])
        hits += int(rank is not None)
        h1 += int(rank == 1)
        reciprocal += (1.0 / rank) if rank else 0.0
    n = len(rows) or 1
    return hits / n, h1 / n, reciprocal / n


def main() -> None:
    s = get_settings()
    k, cand = s.top_k, s.rerank_candidates
    print(f"Re-ranking eval  (k={k}, candidate pool={cand}, model={s.reranker_model})\n")
    header = f"{'eval set':18s} {'variant':10s} {'hit@'+str(k):>6s} {'hit@1':>6s} {'MRR':>6s}"
    print(header)
    print("-" * len(header))

    md = [
        f"| Eval set | Variant | hit@{k} | hit@1 | MRR@{k} |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, path in EVAL_SETS.items():
        rows = load_eval_set(path)
        for variant, use_rerank in (("baseline", False), ("+ rerank", True)):
            hk, h1, mrr = _metrics(rows, k, cand, use_rerank)
            print(f"{name:18s} {variant:10s} {hk:>5.0%} {h1:>6.0%} {mrr:>6.3f}")
            md.append(f"| {name} ({len(rows)} q) | {variant} | {hk:.0%} | {h1:.0%} | {mrr:.3f} |")
        print()

    note = (
        f"\n_Candidate pool = {cand}, top-k = {k}, cross-encoder = `{s.reranker_model}`. "
        f"Re-ranking only re-orders the retriever's pool, so hit@{k} is unchanged; "
        f"hit@1 / MRR show whether the right document is pulled to the top._"
    )
    RESULTS_FILE.write_text("\n".join(md) + "\n" + note + "\n", encoding="utf-8")
    print(f"Written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
