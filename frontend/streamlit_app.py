"""Streamlit UI for the Relocation Assistant.

Runs in two modes:

- **HTTP mode** (default): calls the FastAPI backend over HTTP. Used by
  ``docker compose`` and for local two-service development.
- **In-process mode** (``RAG_IN_PROCESS=1``): imports the RAG pipeline directly,
  so a single process serves the whole demo — no separate API server needed.
  This is what the Hugging Face Space uses (one container, port 7860).
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable so `import app.…` works regardless of how the app
# is launched (e.g. `streamlit run frontend/streamlit_app.py` on Streamlit
# Community Cloud, which otherwise only puts frontend/ on the path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# On Streamlit Community Cloud, config is provided via st.secrets. Mirror those
# into the environment so the pydantic-settings config (which reads env vars)
# picks them up. Real env vars (local/.env) take precedence.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001 - no secrets configured (e.g. local dev)
    pass

# Mode selection:
# - If RAG_IN_PROCESS is set explicitly, honour it.
# - Otherwise default to in-process, unless an API_URL is provided (then use HTTP).
API_URL = os.getenv("API_URL")
_force = os.getenv("RAG_IN_PROCESS")
if _force is not None:
    IN_PROCESS = _force.lower() in {"1", "true", "yes"}
else:
    IN_PROCESS = API_URL is None
API_URL = API_URL or "http://localhost:8000"

st.set_page_config(page_title="Relocation Assistant", page_icon="🧭")
st.title("🧭 Relocation Assistant")
st.caption("RAG over official documents · every answer shows its sources.")


@st.cache_resource(show_spinner="Building the knowledge base…")
def _ensure_ingested() -> dict:
    """Ingest the documents once per container start (in-process mode).

    The Hugging Face Space filesystem is ephemeral, so the vector store is
    rebuilt on each cold start. ``st.cache_resource`` guarantees this runs only
    once per process, not on every interaction.
    """
    from app.rag.ingest import ingest

    return ingest()


def _ask_in_process(question: str) -> dict:
    """Answer via a direct pipeline call (no HTTP)."""
    from app.rag.pipeline import answer_question

    resp = answer_question(question)
    return {"answer": resp.answer, "sources": [s.model_dump() for s in resp.sources]}


def _ask_http(question: str) -> dict:
    """Answer via the FastAPI backend over HTTP."""
    import httpx

    resp = httpx.post(f"{API_URL}/chat", json={"question": question}, timeout=120)
    resp.raise_for_status()
    return resp.json()


# In-process mode: make sure the knowledge base exists before the first question.
if IN_PROCESS:
    _ensure_ingested()

with st.sidebar:
    st.header("About")
    st.write(
        "Ask about relocating to Poland (PESEL, residence registration, "
        "health insurance, taxes, ZUS, …). Answers are grounded in the ingested "
        "documents and every answer shows its sources."
    )
    st.caption(f"Mode: {'in-process' if IN_PROCESS else 'HTTP API'}")
    # Re-ingest is only meaningful in HTTP mode; in-process ingests on startup.
    if not IN_PROCESS and st.button("Re-ingest documents"):
        import httpx

        try:
            r = httpx.post(f"{API_URL}/ingest", timeout=120)
            r.raise_for_status()
            st.success(f"Ingested: {r.json()}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ingest failed: {exc}")

question = st.text_input("Your question", placeholder="How do I get a PESEL number?")

if st.button("Ask") and question:
    with st.spinner("Thinking…"):
        try:
            data = _ask_in_process(question) if IN_PROCESS else _ask_http(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
        else:
            st.markdown("### Answer")
            st.write(data["answer"])
            st.markdown("### Sources")
            for s in data["sources"]:
                with st.expander(f"{s['document']}  ·  score {s['score']}"):
                    st.write(s["snippet"])
