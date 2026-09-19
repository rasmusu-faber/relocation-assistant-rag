FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt requirements-tracing.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional Langfuse tracing: docker build --build-arg INSTALL_TRACING=true .
ARG INSTALL_TRACING=false
RUN if [ "$INSTALL_TRACING" = "true" ]; then pip install --no-cache-dir -r requirements-tracing.txt; fi

COPY . .

# Writable caches for the embedding-model download and the vector store.
# (Hugging Face Space filesystems are ephemeral and can be restrictive.)
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache \
    CHROMA_DIR=/app/.chroma
RUN mkdir -p /app/.cache /app/.chroma && chmod -R 777 /app/.cache /app/.chroma

# Hugging Face Spaces serve on port 7860.
EXPOSE 7860

# Default: the public demo — a single Streamlit process that calls the RAG
# pipeline in-process (no separate API server). Ingest first because the Space
# filesystem is rebuilt on each cold start, then serve on 7860.
#
# The FastAPI "production API" is still available:
#   docker run ... uvicorn app.main:app --host 0.0.0.0 --port 8000
# and docker-compose runs the API + UI as two services (see docker-compose.yml).
CMD ["sh", "-c", "python -m app.rag.ingest && RAG_IN_PROCESS=1 streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 7860"]
