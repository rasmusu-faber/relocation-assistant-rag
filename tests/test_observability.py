"""Tests for the optional Langfuse tracing wrapper (no Langfuse install needed)."""
from __future__ import annotations

from types import SimpleNamespace

import app.observability as obs


def _settings(public: str = "", secret: str = ""):
    return SimpleNamespace(
        langfuse_public_key=public, langfuse_secret_key=secret, langfuse_host="h"
    )


def test_inactive_without_keys(monkeypatch):
    monkeypatch.setattr(obs, "_disabled", False)
    monkeypatch.setattr(obs, "get_settings", lambda: _settings())
    assert not obs.tracing_active()


def test_active_with_both_keys(monkeypatch):
    monkeypatch.setattr(obs, "_disabled", False)
    monkeypatch.setattr(obs, "get_settings", lambda: _settings("pk", "sk"))
    assert obs.tracing_active()


def test_disable_tracing_overrides_keys(monkeypatch):
    monkeypatch.setattr(obs, "_disabled", False)
    monkeypatch.setattr(obs, "get_settings", lambda: _settings("pk", "sk"))
    obs.disable_tracing()
    assert not obs.tracing_active()


def test_observe_is_a_passthrough_when_inactive(monkeypatch):
    monkeypatch.setattr(obs, "_disabled", True)

    @obs.observe(name="add")
    def add(a: int, b: int = 0) -> int:
        return a + b

    assert add(2, b=3) == 5
    assert add.__name__ == "add"
    obs.record_generation(model="m")  # no-op, must not raise
    obs.flush()
