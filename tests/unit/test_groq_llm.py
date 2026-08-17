"""Tests for the Groq LLM client.

Mirrors tests/unit/test_llm.py in shape. Covers:

* Quota / rate-limit (HTTP 429) surfaces as LLMQuotaExhaustedError
* Transient errors propagate (not retried at the application layer)
* Empty / non-text chunks don't break streaming
* Unconfigured client raises a clear RuntimeError
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from backend.config.config import LLMConfig
from backend.llm.gemini_client import LLMError, LLMQuotaExhaustedError
from backend.llm.groq_client import (
    GroqLLMClient,
    _is_quota_error,
)


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible client (stream + quota + transient scenarios)
# ---------------------------------------------------------------------------


@dataclass
class _Choice:
    delta: Any = None
    text: str = ""


@dataclass
class _Chunk:
    choices: list[_Choice] = field(default_factory=list)


class _FakeQuotaError(Exception):
    """Mimics openai.RateLimitError (status_code=429)."""

    def __init__(self, msg: str = "Rate limit reached", retry_after: float | None = 2.5):
        super().__init__(msg)
        self.status_code = 429
        self.retry_after = retry_after


class _FakeOpenAIClient:
    """Drop-in replacement for `openai.OpenAI` used by GroqLLMClient."""

    call_count = 0

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.call_count += 1
            stream = kwargs.get("stream", False)
            if not stream:
                raise NotImplementedError("non-streaming path tested separately")
            mode = self._outer.mode
            if mode == "quota":
                raise _FakeQuotaError()
            if mode == "transient":
                raise ConnectionError("transient network blip")
            if mode == "ok":
                return self._outer._ok_stream()
            if mode == "empty_first_chunk":
                # First chunk has no text content (role-only), then real text.
                return self._outer._empty_first_chunk_stream()
            raise RuntimeError("unknown mode")

    class _Chat:
        def __init__(self, outer):
            self.completions = _FakeOpenAIClient._Completions(outer)

    def __init__(self, **_):
        self.mode = "ok"
        self.call_count = 0
        self.chat = _FakeOpenAIClient._Chat(self)

    def _ok_stream(self):
        for piece in ["Hello", ", ", "world", "!"]:
            yield _Chunk(choices=[_Choice(delta=type("D", (), {"content": piece})())])

    def _empty_first_chunk_stream(self):
        # First chunk sets role but has None content (real OpenAI behaviour).
        none_delta = type("D", (), {"content": None})()
        yield _Chunk(choices=[_Choice(delta=none_delta)])
        for piece in ["ok", " ", "stream"]:
            yield _Chunk(choices=[_Choice(delta=type("D", (), {"content": piece})())])


def _attach_fake_client(monkeypatch, mode: str = "ok") -> GroqLLMClient:
    """Build a GroqLLMClient whose underlying openai.OpenAI is the fake."""
    cfg = LLMConfig(api_key_env="DUMMY_GROQ_KEY")
    monkeypatch.setenv("DUMMY_GROQ_KEY", "dummy-groq-key")
    import openai as openai_mod

    fake = _FakeOpenAIClient()
    fake.mode = mode
    monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: fake)
    client = GroqLLMClient(cfg)
    # Sanity: client constructor should have used the fake.
    assert client._client is fake, "fake OpenAI client not wired up"
    return client


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_unconfigured_client_raises_runtime_error() -> None:
    """No api_key in env -> client stays unconfigured -> raises on use."""
    cfg = LLMConfig(api_key_env="DEFINITELY_NOT_SET_GROQ")
    monkeypatch_holder = None
    import os

    os.environ.pop("DEFINITELY_NOT_SET_GROQ", None)
    client = GroqLLMClient(cfg)
    with pytest.raises(RuntimeError, match="not configured"):
        list(client.generate_stream("hi"))


def test_client_uses_GROQ_API_KEY_fallback(monkeypatch) -> None:
    """If cfg.api_key_env is unset, GROQ_API_KEY should be honoured."""
    cfg = LLMConfig(api_key_env="")  # empty -> fallback path
    monkeypatch.setenv("GROQ_API_KEY", "fallback-key")
    # OpenAI constructor needs to be importable; patch it to a no-op
    # recorder so we don't actually call the network.
    import openai as openai_mod

    seen: dict[str, Any] = {}

    def _spy(**kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(openai_mod, "OpenAI", _spy)
    GroqLLMClient(cfg)
    assert seen.get("api_key") == "fallback-key"
    assert seen.get("base_url", "").endswith("/openai/v1")


# ---------------------------------------------------------------------------
# Streaming: success path
# ---------------------------------------------------------------------------


def test_generate_stream_yields_text_deltas(monkeypatch) -> None:
    client = _attach_fake_client(monkeypatch, mode="ok")
    pieces = list(client.generate_stream("hi"))
    assert pieces == ["Hello", ", ", "world", "!"]


def test_generate_collects_full_text_and_records_ttft(monkeypatch) -> None:
    client = _attach_fake_client(monkeypatch, mode="ok")
    result = client.generate("hi")
    assert result.text == "Hello, world!"
    assert result.ttft_ms > 0
    assert result.total_ms >= result.ttft_ms


def test_empty_first_chunk_does_not_blow_up(monkeypatch) -> None:
    """A role-only first chunk (content=None) must not crash the stream."""
    client = _attach_fake_client(monkeypatch, mode="empty_first_chunk")
    pieces = list(client.generate_stream("hi"))
    assert "".join(pieces) == "ok stream"


# ---------------------------------------------------------------------------
# Quota errors
# ---------------------------------------------------------------------------


def test_quota_error_translates_to_LLMQuotaExhaustedError(monkeypatch) -> None:
    client = _attach_fake_client(monkeypatch, mode="quota")
    t0 = time.perf_counter()
    with pytest.raises(LLMQuotaExhaustedError) as excinfo:
        list(client.generate_stream("hi"))
    elapsed = time.perf_counter() - t0
    assert excinfo.value.retry_after_seconds == pytest.approx(2.5)
    # Fail-fast: exactly one HTTP-level call.
    assert client._client.call_count == 1
    # And it must be fast (no retry sleep).
    assert elapsed < 2.0, f"quota raise took {elapsed:.2f}s -- too slow"


def test_quota_error_in_generate_path(monkeypatch) -> None:
    """generate() (non-streaming) must also translate quota errors."""
    client = _attach_fake_client(monkeypatch, mode="quota")
    with pytest.raises(LLMQuotaExhaustedError):
        client.generate("hi")


def test_quota_detection_helper() -> None:
    """_is_quota_error picks up 429 by code, message, or both."""

    @dataclass
    class _E1:
        status_code: int = 429

    @dataclass
    class _E2:
        pass

    assert _is_quota_error(_E1()) is True
    assert _is_quota_error(Exception("HTTP 429 Too Many Requests")) is True
    assert _is_quota_error(Exception("rate limit reached")) is True
    assert _is_quota_error(_E2()) is False
    assert _is_quota_error(Exception("regular network timeout")) is False


# ---------------------------------------------------------------------------
# Transient errors
# ---------------------------------------------------------------------------


def test_transient_error_propagates_quota_translation_skipped(monkeypatch) -> None:
    """A non-quota error must propagate unchanged (no false quota translation)."""
    client = _attach_fake_client(monkeypatch, mode="transient")
    with pytest.raises(ConnectionError):
        list(client.generate_stream("hi"))
