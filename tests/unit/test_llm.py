"""Tests for src.llm -- LLMClient interface, FakeLLMClient, GeminiLLMClient.

We don't hit the real Gemini API here; the live client is exercised by
``scripts/live_smoke.py``. These tests pin the contract that both
pipelines rely on.
"""

from __future__ import annotations

import time

import pytest

from backend.config.config import LLMConfig
from backend.llm.gemini_client import (
    FakeLLMClient,
    GeminiLLMClient,
    LLMClient,
    LLMError,
    LLMQuotaExhaustedError,
    LLMResult,
    _is_quota_error,
)


def test_fake_llm_client_generate_echoes_user() -> None:
    client = FakeLLMClient(response_prefix="ECHO:")
    result = client.generate(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hello world"},
        ]
    )
    assert isinstance(result, LLMResult)
    assert result.text == "ECHO: hello world"


def test_fake_llm_client_accepts_string_prompt() -> None:
    client = FakeLLMClient()
    result = client.generate("plain string prompt")
    assert result.text.startswith("ANSWER:")


def test_fake_llm_client_records_calls() -> None:
    client = FakeLLMClient()
    client.generate([{"role": "user", "content": "q1"}])
    client.generate("q2")
    assert len(client.calls) == 2


def test_fake_llm_client_streaming_yields_pieces() -> None:
    client = FakeLLMClient(response_prefix="X")
    pieces = list(client.generate_stream("hello"))
    assert "".join(pieces).endswith("hello")
    assert pieces[0].startswith("X")


def test_gemini_client_unconfigured_raises() -> None:
    cfg = LLMConfig(api_key_env="DEFINITELY_NOT_SET_12345")
    client = GeminiLLMClient(cfg)
    with pytest.raises(RuntimeError, match="not configured"):
        client.generate("hello")


def test_gemini_client_picks_up_explicit_key() -> None:
    cfg = LLMConfig()
    client = GeminiLLMClient(cfg, api_key="dummy-but-present")
    # Should not raise on construction.
    assert client._api_key == "dummy-but-present"


# ---------------------------------------------------------------------------
# Quota-exhaustion fail-fast (issue: dashboard showed 20 s TTFT because
# the SDK's tenacity retry loop was sleeping on 429s).
# ---------------------------------------------------------------------------


class _FakeQuotaClient:
    """Stand-in for google.genai.Client that raises 429 immediately.

    Records how many HTTP calls the SDK's retry layer made so we can
    verify our application code did NOT trigger a long retry loop.
    """

    def __init__(self, status_code: int = 429, message: str = "RESOURCE_EXHAUSTED") -> None:
        self.status_code = status_code
        self.message = message
        self.call_count = 0
        # Build a minimal SDK-like exception.
        try:
            from google.genai import errors as _sdk_errors  # type: ignore

            err_cls = getattr(_sdk_errors, "ClientError", None)
            if err_cls is None:
                raise ImportError
        except Exception:
            err_cls = None
        if err_cls is not None:
            self._exc = err_cls(
                status_code,
                {
                    "error": {
                        "code": status_code,
                        "message": message + " - retryDelay: '1.5s'",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
            )
        else:  # pragma: no cover - tests run with SDK installed
            raise RuntimeError("google.genai.errors.ClientError unavailable")

        class _Chats:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **_kwargs):
                class _Chat:
                    def __init__(self, outer):
                        self._outer = outer

                    def send_message_stream(self, _msg):
                        self._outer.call_count += 1
                        raise self._outer._exc

                return _Chat(self._outer)

        self.chats = _Chats(self)


def _attach_fake_sdk(monkeypatch) -> tuple[GeminiLLMClient, _FakeQuotaClient]:
    """Build a GeminiLLMClient whose underlying SDK client is the fake."""
    cfg = LLMConfig(api_key_env="DUMMY_FOR_TEST")
    fake = _FakeQuotaClient()
    monkeypatch.setenv("DUMMY_FOR_TEST", "dummy-key")
    # Patch google.genai.Client so the constructor returns our fake.
    import google.genai as genai_mod  # type: ignore

    monkeypatch.setattr(genai_mod, "Client", lambda **kw: fake)
    client = GeminiLLMClient(cfg)
    return client, fake


def test_quota_exhausted_raises_immediately(monkeypatch) -> None:
    """A 429 must raise LLMQuotaExhaustedError on the first attempt."""
    client, fake = _attach_fake_sdk(monkeypatch)
    t0 = time.perf_counter()
    with pytest.raises(LLMQuotaExhaustedError) as excinfo:
        list(client.generate_stream("hi"))
    elapsed = time.perf_counter() - t0
    assert excinfo.value.retry_after_seconds == pytest.approx(1.5)
    # Fail-fast: at most one HTTP-level call should have been made.
    assert fake.call_count == 1, (
        f"expected 1 SDK attempt, got {fake.call_count} -- retry loop leaked"
    )
    # And it must be fast (no exponential backoff sleeps).
    assert elapsed < 2.0, f"quota raise took {elapsed:.2f}s -- too slow"


def test_quota_exhausted_generate_also_translates(monkeypatch) -> None:
    """generate() (non-streaming) must surface quota errors the same way."""
    client, fake = _attach_fake_sdk(monkeypatch)
    with pytest.raises(LLMQuotaExhaustedError):
        client.generate("hi")
    assert fake.call_count == 1


def test_quota_error_detection_helper() -> None:
    """The detector picks up quota errors by code or message body."""

    class _E:
        code = 429

    assert _is_quota_error(_E()) is True
    assert _is_quota_error(Exception("RESOURCE_EXHAUSTED quota")) is True
    assert _is_quota_error(Exception("some 500 server error")) is False
    assert _is_quota_error(Exception("regular network timeout")) is False


def test_quota_failure_tags_benchmark_results() -> None:
    """A failed pipeline must mark succeeded=False and zero LLM latency.

    This is what lets benchmark/aggregate code drop the row without
    polluting the rolling average.
    """
    from backend.config.config import load_config
    from backend.rag.database import VectorDatabase
    from backend.rag.normal_rag import NormalRAG
    from backend.compressor.pipeline import PipelineComponents
    from backend.rag.smart_rag import SmartRAG
    from tests.unit._pipeline_fakes import HashEmbedder, WordOverlapCrossEncoder

    cfg = load_config()
    db = VectorDatabase(embedder=HashEmbedder())
    db.add_document("d1", "Token-Diet enforces a global token budget.")
    components = PipelineComponents(
        embedder=HashEmbedder(),
        cross_encoder=WordOverlapCrossEncoder(),
    )

    class _AlwaysQuota(LLMClient):
        def generate(self, prompt, **kw):  # type: ignore[override]
            raise LLMQuotaExhaustedError(
                "RESOURCE_EXHAUSTED", retry_after_seconds=2.0
            )

        def generate_stream(self, prompt, **kw):
            raise LLMQuotaExhaustedError(
                "RESOURCE_EXHAUSTED", retry_after_seconds=2.0
            )
            yield ""  # pragma: no cover

    failing_llm = _AlwaysQuota()
    normal = NormalRAG(db, failing_llm, cfg).run("What is the budget?")
    smart = SmartRAG(db, failing_llm, cfg, components=components).run("What is the budget?")
    assert normal.succeeded is False
    assert smart.succeeded is False
    # LLM-side timings must be zero so aggregates cannot be polluted.
    assert normal.llm_ttft_ms == 0.0
    assert normal.llm_total_gen_ms == 0.0
    assert smart.llm_ttft_ms == 0.0
    assert smart.llm_total_gen_ms == 0.0

    cmp = SmartRAG.compare(normal, smart)
    assert cmp["quota_failed"] is True
    assert cmp["normal_succeeded"] is False
    assert cmp["smart_succeeded"] is False


def test_quota_error_is_subclass_of_llm_error() -> None:
    """Callers can branch on the LLMError base type."""
    err = LLMQuotaExhaustedError("RESOURCE_EXHAUSTED")
    assert isinstance(err, LLMError)
    assert isinstance(err, LLMQuotaExhaustedError)


def test_gemini_client_model_resolution() -> None:
    from backend.config.config import LLMConfig
    from backend.llm.gemini_client import GeminiLLMClient
    
    cfg = LLMConfig(model="some-unavailable-model")
    
    # Mocking SDK Client and model list
    class MockModel:
        def __init__(self, name: str, actions: list[str]) -> None:
            self.name = name
            self.supported_actions = actions

    class MockModelsService:
        def list(self):
            return [
                MockModel("models/other-model", ["otherAction"]),
                MockModel("models/gemini-2.5-flash-lite", ["generateContent"]),
                MockModel("models/gemini-2.5-flash", ["generateContent"]),
            ]

    class MockGenaiClient:
        def __init__(self, **kwargs) -> None:
            self.models = MockModelsService()

    client = GeminiLLMClient(cfg, api_key="fake-api-key")
    client._client = MockGenaiClient()
    
    # Verify gemini-2.5-flash is preferred as default when available
    assert client._resolve_model("some-unavailable-model") == "gemini-2.5-flash"
    
    # Verify fallback if gemini-2.5-flash is not available
    class MockModelsServiceNo25:
        def list(self):
            return [
                MockModel("models/other-model", ["otherAction"]),
                MockModel("models/gemini-2.5-flash-lite", ["generateContent"]),
            ]
    client._client.models = MockModelsServiceNo25()
    assert client._resolve_model("some-unavailable-model") == "gemini-2.5-flash-lite"