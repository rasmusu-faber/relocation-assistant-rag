"""Optional Langfuse tracing: a no-op unless configured, so nothing else depends on it."""
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_disabled = False
_client_ready = False


def disable_tracing() -> None:
    """Switch tracing off for this process (used by the eval harness and tests)."""
    global _disabled
    _disabled = True


def tracing_active() -> bool:
    """True when Langfuse keys are configured and tracing was not disabled."""
    s = get_settings()
    return not _disabled and bool(s.langfuse_public_key and s.langfuse_secret_key)


def _client() -> Any:
    """Initialise the Langfuse client once from our settings (which read .env)."""
    global _client_ready
    from langfuse import Langfuse, get_client

    if not _client_ready:
        s = get_settings()
        Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        _client_ready = True
    return get_client()


def observe(name: str | None = None, as_type: Any = None) -> Callable:
    """Trace a function as a Langfuse span; calls it untouched when tracing is off."""

    def decorator(fn: Callable) -> Callable:
        traced: Callable | None = None

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced
            if not tracing_active():
                return fn(*args, **kwargs)
            if traced is None:
                from langfuse import observe as lf_observe

                _client()
                traced = lf_observe(name=name or fn.__name__, as_type=as_type)(fn)
            return traced(*args, **kwargs)

        return wrapper

    return decorator


def record_generation(**fields: Any) -> None:
    """Attach model / prompt / output to the current generation span."""
    if not tracing_active():
        return
    try:
        _client().update_current_generation(**fields)
    except Exception:  # noqa: BLE001 - tracing must never break a request
        logger.warning("Could not record Langfuse generation", exc_info=True)


def flush() -> None:
    """Send any buffered traces (call on shutdown)."""
    if not tracing_active():
        return
    try:
        _client().flush()
    except Exception:  # noqa: BLE001
        logger.warning("Langfuse flush failed", exc_info=True)
