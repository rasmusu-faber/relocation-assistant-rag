"""Answer-groundedness (faithfulness) proxy for the RAG pipeline.

Retrieval metrics (see ``run_eval.py``) ask *"did we fetch the right document?"*.
Groundedness asks the complementary question: *"does the generated answer stay
within what those fetched passages actually say — or does the model add claims of
its own?"* For a citation-based assistant that is the failure mode that matters:
an answer can look trustworthy (sources attached) yet contain a hallucinated
detail that appears in none of them.

This is a **lightweight, offline proxy**, deliberately consistent with the rest of
the eval harness:

1. Generate an answer for each question (needs a configured LLM provider).
2. Split the answer into sentences.
3. Embed each sentence and each retrieved passage with the *same* embedding model
   used for retrieval, and take the best cosine similarity of the sentence against
   any passage. A sentence counts as **supported** when that similarity clears a
   threshold.
4. Groundedness of an answer = fraction of supported sentences; report the mean
   across questions.

**What it does not do:** it measures *semantic overlap*, not logical entailment. A
sentence that contradicts a passage while reusing its vocabulary can still score
as supported. It is a cheap, deterministic signal — a screen for ungrounded
answers, not a substitute for an NLI / LLM-as-judge faithfulness model.

Because it calls the LLM, it runs locally rather than as the API-key-free CI gate.

Run:  python -m eval.groundedness            # all questions
      python -m eval.groundedness --limit 3  # quick sample
"""
from __future__ import annotations

import argparse
import re
import sys

from eval.run_eval import load_eval_set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Sentence counts as grounded when its best cosine similarity to any retrieved
# passage clears this. Tuned for MiniLM-style normalized embeddings; expose via
# --threshold to recalibrate for a different embedding model.
SUPPORT_THRESHOLD = 0.40

# Minimum words for a fragment to count as a scorable claim. Filters out list
# enumeration leftovers ("1.", "2)") and stray tokens that would otherwise be
# scored as ungrounded and distort the ratio.
MIN_CLAIM_WORDS = 3

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# Leading list marker at the start of a line: "1.", "2)", "a.", "-", "*", "•".
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[a-zA-Z][.)]|[-*•])\s+")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentence-like claims (a heuristic, no NLP deps).

    Per line: strip a leading list marker, split on sentence punctuation, and keep
    only fragments of at least ``MIN_CLAIM_WORDS`` words. This discards numbering
    artifacts ("1.", "2.") so they are not mistaken for ungrounded claims.
    """
    sentences: list[str] = []
    for line in text.strip().splitlines():
        line = _LIST_MARKER.sub("", line.strip()).strip()
        if not line:
            continue
        for part in _SENTENCE_BOUNDARY.split(line):
            part = part.strip()
            if len(part.split()) >= MIN_CLAIM_WORDS and any(ch.isalnum() for ch in part):
                sentences.append(part)
    return sentences


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product; inputs are L2-normalized embeddings, so this is cosine."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def score_answer(
    answer: str, contexts: list[str], threshold: float = SUPPORT_THRESHOLD
) -> list[tuple[str, float, bool]]:
    """Return ``(sentence, best_similarity, is_supported)`` for each sentence."""
    from app.rag.store import embed

    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return [(s, 0.0, False) for s in sentences]

    context_vecs = embed(contexts)
    sentence_vecs = embed(sentences)
    results: list[tuple[str, float, bool]] = []
    for sentence, vec in zip(sentences, sentence_vecs, strict=True):
        best = max(_cosine(vec, cv) for cv in context_vecs)
        results.append((sentence, best, best >= threshold))
    return results


def evaluate(
    threshold: float = SUPPORT_THRESHOLD, limit: int | None = None
) -> dict[str, float | int]:
    """Generate answers, score their groundedness, and print a per-question trace."""
    from app.rag.generator import generate
    from app.rag.retriever import retrieve

    rows = load_eval_set()
    if limit is not None:
        rows = rows[:limit]

    ratios: list[float] = []
    fully_grounded = 0
    errors = 0
    for row in rows:
        question = row["question"]
        print(f"\nQ: {question}")
        try:
            contexts = [s.snippet for s in retrieve(question)]
            answer = generate(question, contexts)
        except Exception as exc:  # noqa: BLE001 - a flaky LLM call must not abort the whole run
            errors += 1
            print(f"   ! generation failed, skipped: {exc}")
            continue

        results = score_answer(answer, contexts, threshold)
        supported = sum(1 for _, _, ok in results if ok)
        total = len(results)
        ratio = supported / total if total else 1.0
        ratios.append(ratio)
        fully_grounded += int(supported == total and total > 0)

        print(f"   grounded {supported}/{total} sentences ({ratio:.0%})")
        for sentence, best, ok in results:
            flag = "✓" if ok else "✗"
            print(f"   {flag} [{best:.2f}] {sentence[:88]}")

    n = len(ratios) or 1
    return {
        "mean_groundedness": sum(ratios) / n,
        "fully_grounded_rate": fully_grounded / n,
        "n": len(ratios),
        "errors": errors,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG answer-groundedness proxy.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--threshold", type=float, default=SUPPORT_THRESHOLD)
    args = parser.parse_args()

    print(f"Answer-groundedness proxy (support threshold = {args.threshold}):")
    m = evaluate(threshold=args.threshold, limit=args.limit)
    skipped = f", {m['errors']} skipped" if m["errors"] else ""
    print(
        f"\nMean groundedness: {m['mean_groundedness']:.1%}   "
        f"fully-grounded answers: {m['fully_grounded_rate']:.1%}   "
        f"({m['n']} questions{skipped})"
    )


if __name__ == "__main__":
    main()
