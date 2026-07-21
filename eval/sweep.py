"""Retrieval configuration sweep: compare chunk sizes x embedding models.

For every (embedding model, chunk size) combination this re-chunks and re-embeds
the corpus into an **in-memory** Chroma collection (the persistent ``.chroma`` store
is never touched), runs the eval questions, and reports:

* **hit@1**  — expected document is the top-ranked passage
* **hit@k**  — expected document appears in the top-k passages
* **MRR@k**  — mean reciprocal rank of the first correct passage

Output is a Markdown table (printed and written to ``eval/sweep_results.md``) that
can be pasted into the README. Run locally — it downloads the second embedding
model (~130 MB) on first use and is intentionally kept out of the CI gate.

Run:  python -m eval.sweep
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.rag.chunking import chunk_text
from app.rag.ingest import load_documents
from eval.run_eval import first_hit_rank, load_eval_set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Sweep grid -------------------------------------------------------------
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]
CHUNK_SIZES = [400, 800, 1200]
OVERLAP_RATIO = 0.15  # overlap scales with chunk size for a fair comparison
TOP_K = 4

RESULTS_FILE = Path(__file__).parent / "sweep_results.md"


@lru_cache(maxsize=None)
def _load_model(name: str) -> Any:
    """Load (and cache) a SentenceTransformer by name."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _embed(model: Any, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()


def _evaluate_config(model_name: str, chunk_size: int) -> dict[str, float]:
    """Build an in-memory index for one config and return its metrics."""
    import chromadb

    model = _load_model(model_name)
    overlap = round(chunk_size * OVERLAP_RATIO)

    # Chunk the corpus.
    ids: list[str] = []
    texts: list[str] = []
    docs_meta: list[str] = []
    for name, body, _front_matter in load_documents(get_settings().data_dir):
        for i, chunk in enumerate(chunk_text(body, chunk_size, overlap)):
            ids.append(f"{name}::{i}")
            texts.append(chunk)
            docs_meta.append(name)

    # Fresh in-memory collection (cosine space, like production). EphemeralClient
    # is process-wide, so drop any collection left over from a previous config.
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("sweep")
    except Exception:
        pass
    collection = client.create_collection(
        name="sweep", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=_embed(model, texts),
        metadatas=[{"document": d} for d in docs_meta],
    )

    rows = load_eval_set()
    hits = hits_at_1 = 0
    reciprocal_ranks = 0.0
    for row in rows:
        result = collection.query(
            query_embeddings=_embed(model, [row["question"]]),
            n_results=TOP_K,
            include=["metadatas"],
        )
        retrieved_docs = [m.get("document", "") for m in result["metadatas"][0]]
        rank = first_hit_rank(retrieved_docs, row["expected_document"])
        hits += int(rank is not None)
        hits_at_1 += int(rank == 1)
        reciprocal_ranks += (1.0 / rank) if rank else 0.0

    n = len(rows) or 1
    return {
        "hit_at_1": hits_at_1 / n,
        "hit_at_k": hits / n,
        "mrr": reciprocal_ranks / n,
        "chunks": len(texts),
    }


def _short_model(name: str) -> str:
    return name.split("/")[-1]


def run_sweep() -> str:
    """Run the full grid and return a Markdown results table."""
    header = (
        f"| Embedding model | Chunk size | hit@1 | hit@{TOP_K} | MRR@{TOP_K} | Chunks |\n"
        f"| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    print(f"Sweep: {len(EMBEDDING_MODELS)} models x {len(CHUNK_SIZES)} chunk sizes "
          f"(top-k = {TOP_K})\n")
    for model_name in EMBEDDING_MODELS:
        for chunk_size in CHUNK_SIZES:
            m = _evaluate_config(model_name, chunk_size)
            row = (
                f"| `{_short_model(model_name)}` | {chunk_size} | "
                f"{m['hit_at_1']:.0%} | {m['hit_at_k']:.0%} | {m['mrr']:.3f} | "
                f"{int(m['chunks'])} |"
            )
            lines.append(row)
            print(f"  {_short_model(model_name):24s} size={chunk_size:<5d} "
                  f"hit@1={m['hit_at_1']:.0%}  hit@{TOP_K}={m['hit_at_k']:.0%}  "
                  f"MRR={m['mrr']:.3f}")
    return "\n".join(lines)


def main() -> None:
    table = run_sweep()
    n = len(load_eval_set())
    note = (
        f"\n_{n} eval questions; overlap = {int(OVERLAP_RATIO * 100)}% of chunk size; "
        f"embeddings L2-normalized, cosine space; models compared out-of-the-box "
        f"(no query-side instructions)._"
    )
    RESULTS_FILE.write_text(table + "\n" + note + "\n", encoding="utf-8")
    print("\n" + table + note)
    print(f"\nWritten to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
