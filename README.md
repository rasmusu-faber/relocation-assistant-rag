# Relocation Assistant — RAG with Source Citations & Evaluation

A retrieval-augmented generation (RAG) assistant that answers questions about
relocating to Poland as an EU citizen (PESEL, residence registration, health
insurance, …) **grounded in official documents** and returns the **source
passages** behind every answer.

> The domain is swappable — point `data/` at any document corpus and re-ingest.

What makes this more than a "chat with your PDF" demo: it ships with a small
**evaluation harness** that measures retrieval quality and answer faithfulness,
runnable locally and in CI. (Built by someone whose M.Sc. thesis was LLM
evaluation — so evaluation is treated as a first-class concern, not an
afterthought.)

## Architecture

```
                ┌─────────────┐      ┌──────────────┐
  documents ──▶ │   ingest    │ ──▶  │  vector DB   │
  (data/)       │ chunk+embed │      │  (Chroma)    │
                └─────────────┘      └──────┬───────┘
                                            │ top-k
  user ──▶ Streamlit ──▶ FastAPI /chat ──▶ retriever ──▶ generator (LLM) ──▶ answer + sources
                                                          │
                                             Ollama (local) │ OpenAI-compatible API
```

- **Backend:** FastAPI (`/health`, `/ingest`, `/chat`)
- **Embeddings:** `sentence-transformers` (default `all-MiniLM-L6-v2`)
- **Vector store:** Chroma (local, persistent)
- **LLM:** pluggable provider — Ollama (local) or any OpenAI-compatible endpoint
- **Frontend:** Streamlit
- **Quality:** pytest + GitHub Actions CI; evaluation harness in `eval/`
- **Packaging:** Dockerfile + docker-compose

## Quickstart (local)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env        # edit if you want a different model/provider

# 3. (Option A) Run with Ollama locally
ollama pull llama3.1:8b     # or any model you have
# (Option B) point LLM_PROVIDER=openai at an OpenAI-compatible endpoint in .env

# 4. Ingest the sample documents
python -m app.rag.ingest

# 5. Run the API + UI
uvicorn app.main:app --reload          # http://localhost:8000/docs
streamlit run frontend/streamlit_app.py # http://localhost:8501
```

## Run with Docker

```bash
docker compose up --build
# API → http://localhost:8000 , UI → http://localhost:8501
```

## Evaluation

```bash
python -m eval.run_eval      # prints retrieval hit-rate and (optional) faithfulness
```

The evaluation set lives in `eval/eval_set.jsonl` (question / expected-source pairs).
Extend it as you add documents.

## Project structure

```
relocation-assistant-rag/
├── app/
│   ├── main.py            # FastAPI app & routes
│   ├── config.py          # settings from environment
│   ├── models.py          # request/response schemas
│   └── rag/
│       ├── ingest.py      # load → chunk → embed → store
│       ├── retriever.py   # vector search
│       ├── generator.py   # LLM provider abstraction (Ollama / OpenAI-compatible)
│       └── pipeline.py    # retrieve + generate → answer with sources
├── frontend/streamlit_app.py
├── eval/{run_eval.py, eval_set.jsonl}
├── data/                  # sample source documents (.md/.txt)
├── tests/                 # pytest
├── .github/workflows/ci.yml
├── Dockerfile, docker-compose.yml
├── requirements.txt, .env.example, .gitignore
```

## Roadmap

- [ ] Weekend 1 (MVP): ingest, cited answers, FastAPI + Streamlit in Docker
- [ ] Weekend 2: eval harness in CI, public deployment (HF Spaces / Render), agentic clarify-question step

## Disclaimer

Educational portfolio project. Answers are model-generated and **not** legal
advice; always verify with the official source linked in each answer.

## License

MIT — see [LICENSE](LICENSE).
