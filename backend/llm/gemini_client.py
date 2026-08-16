"""LLM client abstraction + a Gemini implementation.

Both Normal RAG and Smart RAG call :class:`LLMClient.generate` -- the
implementation is the same instance, so the comparison is fair (plan
§2 "Explicit System Boundary").

Implementations:

* :class:`GeminiLLMClient`  -- uses the official ``google-genai`` SDK
  (plan §6). Falls back to a stub that explains the install when the
  SDK is missing.
* :class:`FakeLLMClient`     -- deterministic for tests.

The streaming surface (``generate_stream``) is provided as a generator
that yields text deltas -- this is what the plan's benchmark loop
(§17 / §18) uses to record Time-to-First-Token.

Quota-exhaustion behaviour (plan: "fail fast, no long retry loops"):

* The official SDK defaults to **5 retry attempts with exponential
  backoff up to 60 s on HTTP 429** (RESOURCE_EXHAUSTED). That is the
  *right* behaviour for transient infra issues but the *wrong* one for
  benchmark workloads: it inflates latency measurements by tens of
  seconds without ever delivering an answer on a free-tier quota cap.
* :class:`GeminiLLMClient` therefore configures
  ``HttpRetryOptions.http_status_codes = [408, 500, 502, 503, 504]``,
  i.e. **keeps retries for transient errors but fails fast on 429**.
* Quota exhaustion surfaces as :class:`LLMQuotaExhaustedError`, a
  subclass of :class:`LLMError`, with ``retry_after_seconds`` so the UI
  can show a clear countdown.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from backend.config.config import LLMConfig


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """Return value of a (non-streaming) LLM call."""

    text: str
    ttft_ms: float
    total_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMError(RuntimeError):
    """Base class for any error originating from an LLM client.

    Distinct from :class:`RuntimeError` so callers can branch on it
    (e.g. the dashboard surfaces a clean message and benchmark code can
    drop the run from the aggregate).
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self._cause = cause


class LLMQuotaExhaustedError(LLMError):
    """Raised when Gemini returns HTTP 429 (RESOURCE_EXHAUSTED).

    Quota errors are **non-retryable from our application code**: the
    server has explicitly told us we are over the limit. The retry
    window is governed by ``retry_after_seconds`` (best-effort parse
    of the SDK response).
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.retry_after_seconds = retry_after_seconds


class LLMClient(ABC):
    """Abstract final-LLM client."""

    @abstractmethod
    def generate(self, prompt: str | list[dict[str, str]], **kwargs) -> LLMResult: ...

    def generate_stream(self, prompt: str | list[dict[str, str]], **kwargs) -> Iterable[str]:
        """Default streaming fallback: yield the full text in one chunk."""
        yield self.generate(prompt, **kwargs).text


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


# Status codes the SDK may retry on. We intentionally OMIT 429 here:
# RESOURCE_EXHAUSTED is a quota signal, not a transient infra blip.
# Retrying it only wastes minutes of latency budget per request.
_RETRYABLE_STATUS_CODES: tuple[int, ...] = (408, 500, 502, 503, 504)


def _is_quota_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` is a Gemini RESOURCE_EXHAUSTED (HTTP 429)."""
    # SDK's google.genai.errors.ClientError carries ``code`` as int.
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    # Older paths or wrapped exceptions: look at the message.
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def _extract_retry_after(exc: BaseException) -> float | None:
    """Best-effort parse of ``retryDelay`` from a Gemini error body."""
    msg = str(exc)
    # SDK echoes the server-suggested retry as e.g. 'retryDelay: "1.5s"'.
    import re

    m = re.search(r"retry[_ ]?delay[\"']?\s*[:=]\s*[\"']?([0-9.]+)s", msg, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


class GeminiLLMClient(LLMClient):
    """Wrapper around Google's official ``google-genai`` SDK.

    The SDK is imported lazily so unit tests can run without it
    installed.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
        *,
        fail_fast_on_quota: bool = True,
    ) -> None:
        """Construct the client.

        Parameters
        ----------
        cfg:
            Resolved :class:`LLMConfig`.
        api_key:
            Optional explicit override. Defaults to ``cfg.api_key_env``.
        fail_fast_on_quota:
            When True (default), the SDK's built-in retry loop is
            restricted so HTTP 429 RESOURCE_EXHAUSTED responses bubble
            up immediately instead of being retried for up to ~60 s.
            Transient 5xx / 408 / network errors still retry normally.
        """

        self._cfg = cfg
        self._api_key = api_key or os.environ.get(cfg.api_key_env, "").strip()
        self._fail_fast_on_quota = fail_fast_on_quota
        self._client = None
        self._resolved_model_name = cfg.model
        if self._api_key:
            try:
                from google import genai  # type: ignore[import-not-found]
                from google.genai import types

                client_kwargs: dict = {"api_key": self._api_key}
                if fail_fast_on_quota:
                    # Pin the SDK's retry policy: keep transient retries,
                    # drop 429. The SDK accepts this via http_options.
                    client_kwargs["http_options"] = _build_http_options()
                else:
                    client_kwargs["http_options"] = types.HttpOptions(api_version="v1")
                self._client = genai.Client(**client_kwargs)
                self._resolved_model_name = self._resolve_model(cfg.model)
            except Exception:  # pragma: no cover -- SDK import errors
                self._client = None

    def _resolve_model(self, requested_model: str) -> str:
        if self._client is None:
            return requested_model
        try:
            models = list(self._client.models.list())
            available_names = []
            for m in models:
                actions = m.supported_actions or []
                supports_gen = any("generateContent" in a or "generate_content" in a or "generate" in a.lower() for a in actions)
                if supports_gen:
                    available_names.append(m.name.replace("models/", ""))
            
            if not available_names:
                return requested_model
            
            # Prefer gemini-2.5-flash if available
            if "gemini-2.5-flash" in available_names:
                return "gemini-2.5-flash"
                
            if requested_model in available_names:
                return requested_model
                
            return available_names[0]
        except Exception:
            return requested_model

    # -- public API ----------------------------------------------------------

    def _chat_stream(self, prompt):
        """Build the chat once and yield token deltas via the SDK's Chats API.

        Uses ``chats.create`` + ``send_message_stream`` per the SDK's
        explicit recommendation over the deprecated modular
        ``generate_content_stream`` path.
        """

        from google.genai import types  # type: ignore[import-not-found]

        messages = _coerce_messages(prompt)
        system_instruction = None
        first_user: str | None = None
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user" and first_user is None:
                first_user = m["content"]

        if first_user is None:
            first_user = ""

        config = types.GenerateContentConfig(
            temperature=self._cfg.temperature,
            max_output_tokens=self._cfg.max_tokens,
            system_instruction=system_instruction,
        )
        chat = self._client.chats.create(  # type: ignore[union-attr]
            model=self._resolved_model_name,
            config=config,
        )
        return chat.send_message_stream(first_user)

    @staticmethod
    def _raise_for_quota(exc: BaseException) -> None:
        """Re-raise a Gemini 429 as :class:`LLMQuotaExhaustedError`.

        Anything else propagates unchanged so the caller (and our
        tests) see the SDK's native error types.
        """
        if _is_quota_error(exc):
            raise LLMQuotaExhaustedError(
                "Gemini quota exhausted (HTTP 429 RESOURCE_EXHAUSTED). "
                "Free-tier per-day cap reached. "
                "Wait for the quota to reset, upgrade to a paid plan, "
                "or switch to a model with a higher free-tier allowance.",
                retry_after_seconds=_extract_retry_after(exc),
                cause=exc,
            ) from exc

    def generate(self, prompt, **kwargs) -> LLMResult:
        if self._client is None:
            raise RuntimeError(
                f"GeminiLLMClient is not configured. Set the {self._cfg.api_key_env} "
                "environment variable and `pip install google-genai`."
            )

        t0 = time.perf_counter()
        first_token_time: float | None = None
        text_parts: list[str] = []

        try:
            for chunk in self._chat_stream(prompt):
                try:
                    piece = chunk.text  # type: ignore[attr-defined]
                except Exception:
                    piece = ""
                if piece:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    text_parts.append(piece)
        except BaseException as exc:
            # Translate quota errors immediately; let everything else
            # (network blip, 5xx, etc.) propagate. Transient errors
            # have already been retried by the SDK retry policy.
            self._raise_for_quota(exc)
            raise
        t1 = time.perf_counter()

        return LLMResult(
            text="".join(text_parts),
            ttft_ms=(first_token_time - t0) * 1000 if first_token_time else (t1 - t0) * 1000,
            total_ms=(t1 - t0) * 1000,
            model=self._resolved_model_name,
        )

    def generate_stream(self, prompt, **kwargs) -> Iterable[str]:
        """Yield text deltas directly from the SDK stream.

        Quota errors are translated to :class:`LLMQuotaExhaustedError`
        and raised on the *first* attempted ``next()`` rather than
        waiting for the SDK's full retry budget. This keeps our
        latency measurements honest: a quota error must show up as a
        fast raise, never as a multi-second TTFT.
        """
        if self._client is None:
            raise RuntimeError("GeminiLLMClient is not configured.")

        # Build the stream. Quota errors raised during construction
        # (e.g. inside send_message_stream's generator-function body
        # before the first yield) must also surface fast.
        try:
            stream = self._chat_stream(prompt)
        except BaseException as exc:
            self._raise_for_quota(exc)
            raise

        # Pull the first chunk eagerly so a 429 surfaced on the first
        # network read also surfaces before the caller iterates.
        try:
            first = next(stream)
        except BaseException as exc:
            self._raise_for_quota(exc)
            raise

        def _iter():
            try:
                yield _safe_chunk_text(first)
                for chunk in stream:
                    yield _safe_chunk_text(chunk)
            except BaseException as exc:
                self._raise_for_quota(exc)
                raise

        return _iter()


def _safe_chunk_text(chunk) -> str:
    try:
        piece = chunk.text  # type: ignore[attr-defined]
    except Exception:
        piece = ""
    return piece if piece else ""


def _build_http_options():
    """Build the SDK's ``HttpOptions`` with quota-aware retries."""
    try:
        from google.genai import types  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover -- SDK import errors
        return None
    retry = types.HttpRetryOptions(
        # 5 attempts is the SDK default; we keep it for transient errors
        # but restrict the retriable set so quota errors are excluded.
        attempts=5,
        initial_delay=1.0,
        max_delay=60.0,
        exp_base=2.0,
        jitter=1.0,
        http_status_codes=list(_RETRYABLE_STATUS_CODES),
    )
    return types.HttpOptions(api_version="v1", retry_options=retry)


# ---------------------------------------------------------------------------
# Test fake
# ---------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """Deterministic LLM client for tests.

    Echoes the prompt back with a small prefix so tests can assert that
    the compressor actually changed the prompt that reached the LLM.
    """

    def __init__(self, response_prefix: str = "ANSWER:", sleep_seconds: float = 0.0) -> None:
        self.response_prefix = response_prefix
        self.sleep_seconds = sleep_seconds
        self.calls: list[str | list[dict[str, str]]] = []

    def generate(self, prompt, **kwargs) -> LLMResult:
        self.calls.append(prompt)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        text = f"{self.response_prefix} {_extract_user_text(prompt)}"
        return LLMResult(text=text, ttft_ms=1.0, total_ms=2.0, model="fake")

    def generate_stream(self, prompt, **kwargs) -> Iterable[str]:
        self.calls.append(prompt)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        for piece in (f"{self.response_prefix} ", _extract_user_text(prompt)):
            yield piece


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_messages(prompt: str | list[dict[str, str]]) -> list[dict[str, str]]:
    """Accept either a single prompt string or a chat messages array."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


def _extract_user_text(prompt: str | list[dict[str, str]]) -> str:
    msgs = _coerce_messages(prompt)
    # Return the last user message, or empty string.
    for m in reversed(msgs):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""