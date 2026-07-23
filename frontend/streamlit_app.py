"""Streamlit UI for the Relocation Assistant.

Runs in two modes:

- **In-process mode** (default when no ``API_URL`` is set): imports the RAG
  pipeline directly, so a single process serves the whole demo — no separate API
  server needed. This is what the public Streamlit Community Cloud demo uses.
- **HTTP mode** (when ``API_URL`` is set): calls the FastAPI backend over HTTP.
  Used by ``docker compose`` and for local two-service development.
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

REPO_URL = "https://github.com/rasmusfaber-ai/relocation-assistant-rag"

EXAMPLE_QUESTIONS = [
    "How do I get a PESEL number?",
    "Who registers me with ZUS when I start a job?",
    "What do I need to open a bank account?",
]

st.set_page_config(
    page_title="Relocation Assistant",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"About": f"RAG assistant with source citations. Code: {REPO_URL}"},
)

# --- Presentation -----------------------------------------------------------
# Hide the default Streamlit chrome (top gradient bar, toolbar, hamburger menu,
# footer) so the demo reads as a product rather than a framework template. The
# header element itself is kept (transparent) so the sidebar toggle still works.
st.markdown(
    """
    <style>
      [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none; }
      #MainMenu, footer { visibility: hidden; }
      [data-testid="stHeader"] { background: transparent; }

      .block-container { padding-top: 2.5rem; max-width: 46rem; }

      .ra-title {
        font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0;
      }
      .ra-subtitle {
        color: #64748b; font-size: 0.98rem; margin: 0 0 0.25rem 0;
      }
      .ra-rule {
        border: none; border-top: 1px solid #e2e8f0; margin: 1.25rem 0 1.5rem 0;
      }
      .ra-footer {
        color: #94a3b8; font-size: 0.82rem; line-height: 1.5;
        border-top: 1px solid #e2e8f0; margin-top: 2.5rem; padding-top: 1rem;
      }
      .ra-footer a { color: #64748b; }
      /* Retrieved passages (st.text): verbatim, wrapped, keep line structure. */
      [data-testid="stExpander"] [data-testid="stText"] {
        white-space: pre-wrap; word-break: break-word;
        font-family: inherit; font-size: 0.9rem; color: #334155; line-height: 1.6;
        background: transparent; border: none; padding: 0; margin: 0;
      }
      /* Example-question buttons: quiet, pill-like. */
      div[data-testid="column"] .stButton > button {
        border-radius: 999px; border: 1px solid #e2e8f0; background: #f8fafc;
        color: #475569; font-size: 0.82rem; font-weight: 500; padding: 0.3rem 0.8rem;
      }
      div[data-testid="column"] .stButton > button:hover {
        border-color: #2563eb; color: #2563eb; background: #ffffff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="ra-title">🧭 Relocation Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="ra-subtitle">Answers about relocating to Poland — grounded in a '
    "curated knowledge base, with the source passages behind every answer.</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="ra-rule">', unsafe_allow_html=True)


# --- Backend ----------------------------------------------------------------
@st.cache_resource(show_spinner="Building the knowledge base…")
def _ensure_ingested() -> dict:
    """Ingest the documents once per process start (in-process mode).

    The hosted filesystem is ephemeral, so the vector store is rebuilt on each
    cold start. ``st.cache_resource`` guarantees this runs only once per process,
    not on every interaction.
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
    st.subheader("About")
    st.write(
        "Ask about relocating to Poland (PESEL, residence registration, health "
        "insurance, taxes, ZUS, …). Every answer is grounded in the ingested "
        "documents and shows the passages it used."
    )
    st.caption(f"Mode: {'in-process' if IN_PROCESS else 'HTTP API'}")
    st.link_button("View source on GitHub", REPO_URL, use_container_width=True)
    # Re-ingest is only meaningful in HTTP mode; in-process ingests on startup.
    if not IN_PROCESS and st.button("Re-ingest documents", use_container_width=True):
        import httpx

        try:
            r = httpx.post(f"{API_URL}/ingest", timeout=120)
            r.raise_for_status()
            st.success(f"Ingested: {r.json()}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ingest failed: {exc}")


# --- Input ------------------------------------------------------------------
st.session_state.setdefault("question", "")

st.caption("Try one of these:")
for col, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS, strict=True):
    # Clicking a chip fills the input; the rerun applies it before the widget
    # below is instantiated, which is why we set state and rerun here.
    if col.button(example, use_container_width=True):
        st.session_state.question = example
        st.rerun()

with st.form("ask", border=False):
    st.text_input(
        "Your question",
        key="question",
        placeholder="e.g. How do I get a PESEL number and what documents do I need?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

# --- Answer -----------------------------------------------------------------
question = st.session_state.question

if submitted and question.strip():
    with st.spinner("Retrieving sources and composing an answer…"):
        try:
            data = _ask_in_process(question) if IN_PROCESS else _ask_http(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
        else:
            with st.container(border=True):
                st.markdown("**Answer**")
                st.write(data["answer"])

            sources = data["sources"]
            if sources:
                st.markdown(f"**Sources** · {len(sources)} passages")
                for s in sources:
                    title = s.get("source_name") or s["document"]
                    with st.expander(f"{title}  —  relevance {s['score']:.2f}"):
                        # st.text (not st.write/markdown) renders the passage
                        # verbatim. It is the raw text the model saw, so a chunk
                        # starting with "# Heading" must not become a huge title.
                        st.text(s["snippet"])
                        if s.get("source_url"):
                            st.markdown(f"[Open the official source ↗]({s['source_url']})")
                        st.caption(f"Knowledge-base file: `{s['document']}`")
elif submitted:
    st.warning("Please enter a question first.")

st.markdown(
    '<div class="ra-footer">Answers are model-generated from a small curated '
    "knowledge base of summaries compiled from official sources — <strong>not legal "
    'advice</strong>. Always verify with the official source. '
    f'<a href="{REPO_URL}">Source code</a></div>',
    unsafe_allow_html=True,
)
