"""Minimal Streamlit UI for the Relocation Assistant."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Relocation Assistant", page_icon="🧭")
st.title("🧭 Relocation Assistant")
st.caption("RAG over official documents · every answer shows its sources.")

with st.sidebar:
    st.header("About")
    st.write(
        "Ask about relocating to Poland (PESEL, residence registration, "
        "health insurance, …). Answers are grounded in the ingested documents."
    )
    if st.button("Re-ingest documents"):
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
            resp = httpx.post(f"{API_URL}/chat", json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
        else:
            st.markdown("### Answer")
            st.write(data["answer"])
            st.markdown("### Sources")
            for s in data["sources"]:
                with st.expander(f"{s['document']}  ·  score {s['score']}"):
                    st.write(s["snippet"])
