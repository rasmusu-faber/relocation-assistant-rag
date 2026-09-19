"""Evaluation harness. Importing it switches Langfuse tracing off (eval runs are noise)."""
from app.observability import disable_tracing

disable_tracing()
