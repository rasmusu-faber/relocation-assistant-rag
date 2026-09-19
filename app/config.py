"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object. Override any field via the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider
    llm_provider: str = "ollama"  # "ollama" | "openai"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # OpenAI-compatible
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Re-ranking (cross-encoder). When enabled, the pipeline retrieves a wider
    # candidate pool and re-orders it with the cross-encoder down to top_k.
    rerank_enabled: bool = False
    rerank_candidates: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval / chunking
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Langfuse tracing (optional; active only when both keys are set)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Storage
    chroma_dir: str = ".chroma"
    collection_name: str = "relocation"
    data_dir: str = "data"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
