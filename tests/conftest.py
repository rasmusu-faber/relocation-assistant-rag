from app.observability import disable_tracing

# Keep tests independent of any Langfuse keys in a developer's local .env.
disable_tracing()
