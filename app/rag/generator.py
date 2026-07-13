"""LLM provider abstraction: Ollama (local) or any OpenAI-compatible endpoint."""
from __future__ import annotations

import httpx

from app.config import get_settings

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about relocating to Poland. "
    "Answer ONLY using the provided context. If the context does not contain the "
    "answer, say so honestly. Be concise and cite the document names you used."
)


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """Assemble the user prompt from the question and retrieved context."""
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_blocks))
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (grounded in the context above):"
    )


def generate(question: str, context_blocks: list[str], timeout: float = 60.0) -> str:
    """Generate an answer with the configured LLM provider."""
    settings = get_settings()
    prompt = build_prompt(question, context_blocks)

    if settings.llm_provider == "ollama":
        return _generate_ollama(prompt, settings, timeout)
    if settings.llm_provider == "openai":
        return _generate_openai(prompt, settings, timeout)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def _generate_ollama(prompt: str, settings, timeout: float) -> str:
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _generate_openai(prompt: str, settings, timeout: float) -> str:
    resp = httpx.post(
        f"{settings.openai_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
