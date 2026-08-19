"""Groq LLM client -- OpenAI-API-compatible wrapper.

The Groq API is intentionally a drop-in replacement for the OpenAI
``/v1/chat/completions`` endpoint, so this client uses the official
``openai`` Python package pointed at Groq's base URL. No Groq SDK is
required.

Why Groq over Gemini free-tier:
- Free-tier Gemini has wildly variable Time-to-First-Token (3-40 s for
  the same prompt) because the inference backend is shared and
  capacity-constrained.
- Groq runs on custom LPU silicon designed for low TTFT, so the
  first-token latency on free-tier is typically ~0.2-0.5 s and the
  inter-token throughput is ~300-800 tok/s.
- For the dashboard's "faster / slower" comparison, Groq removes the
  dominant source of variance (TTFT) so the metric actually reflects
  the compressor's contribution.

The class implements the same ``LLMClient`` interface as
:class:`backend.llm.gemini_client.GeminiLLMClient`, so the rest of
the pipeline (NormalRAG, SmartRAG, dashboard) does not need to know
which provider is in use.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable

from backend.config.config import LLMConfig
from backend.llm.gemini_client import LLMClient, LLMError, LLMQuotaExhaustedError, LLMResult

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_KEY_ENV_DEFAULT = "GROQ_API_KEY"

# HTTP status codes we consider transient (worth retrying) on the
# application side. Mirrors the Gemini client's policy: keep 5xx /
# 408 retries, drop 429 because that's a quota signal, not transient
# infra noise. Note: the openai SDK also has its own internal retry
# loop; we additionally bound it to keep latency honest.
_RETRYABLE_STATUS_CODES = frozenset({408, 500, 502, 503, 504})


def _is_quota_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` looks like a Groq 429 / rate-limit error."""
    msg = (str(exc) or "").lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return True
    # The OpenAI SDK exposes a status_code attribute on its exceptions.
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return False


def _extract_retry_after(exc: BaseException) -> float | None:
    """Best-effort parse of Groq's retry-after hint from an exception."""
    raw = getattr(exc, "retry_after", None)
    if isinstance(raw, (int, float)):
        return float(raw)
    headers = getattr(exc, "headers", None) or {}
    try:
        ra = headers.get("retry-after") if headers else None
    except Exception:
        ra = None
    if ra is not None:
        try:
            return float(ra)
        except (TypeError, ValueError):
            pass
    return None


def _coerce_messages(prompt):
    """Normalize a prompt (str or list[dict]) into OpenAI's messages shape."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not prompt:
        return [{"role": "user", "content": ""}]
    out: list[dict[str, str]] = []
    for m in prompt:
        role = m.get("role") if isinstance(m, dict) else None
        text = m.get("content") if isinstance(m, dict) else None
        if role not in {"system", "user", "assistant"}:
            role = "user"
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        out.append({"role": role, "content": text})
    if not any(m["role"] == "user" for m in out):
        out.append({"role": "user", "content": ""})
    return out


class GroqLLMClient(LLMClient):
    """LLM client that talks to Groq via the OpenAI-compatible API.

    The :mod:`openai` package is imported lazily so tests and offline
    tools can run without it.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
        *,
        max_retries: int = 2,
    ) -> None:
        """Construct the client.

        Parameters
        ----------
        cfg:
            Resolved :class:`LLMConfig`. The client reads ``cfg.model``,
            ``cfg.temperature``, ``cfg.max_tokens``, and
            ``cfg.api_key_env`` (defaults to ``GROQ_API_KEY``).
        api_key:
            Optional explicit override. Falls back to
            ``os.environ[cfg.api_key_env]``.
        max_retries:
            Bound on the SDK's retry loop for transient errors.
            Quota errors are never retried.
        """

        self._cfg = cfg
        # Honour an explicit api_key_env override; default to Groq's
        # own env var so users don't need to change their existing
        # config to swap providers.
        self._api_key = (
            api_key
            or os.environ.get(cfg.api_key_env, "").strip()
            or os.environ.get(_GROQ_KEY_ENV_DEFAULT, "").strip()
        )
        self._max_retries = max(0, int(max_retries))
        self._client: Any = None
        self._resolved_model_name = cfg.model
        if self._api_key:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]

                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=_GROQ_BASE_URL,
                    max_retries=self._max_retries,
                )
            except Exception:  # pragma: no cover -- SDK import errors
                self._client = None

    # -- public API ----------------------------------------------------------

    def _inspect_chunk_usage(self, chunk) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_time = getattr(usage, "prompt_time", None)
            if prompt_time is not None:
                self.last_prompt_time_ms = float(prompt_time) * 1000
            queue_time = getattr(usage, "queue_time", None)
            if queue_time is not None:
                self.last_queue_time_ms = float(queue_time) * 1000

    def _chat_stream(self, prompt):
        """Build the streaming completion and yield OpenAI chunks."""
        messages = _coerce_messages(prompt)
        stream = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._resolved_model_name,
            messages=messages,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        return stream

    def _raise_for_quota(self, exc: BaseException) -> None:
        """Translate a Groq 429 into :class:`LLMQuotaExhaustedError`."""
        if _is_quota_error(exc):
            raise LLMQuotaExhaustedError(
                "Groq rate limit / quota exhausted (HTTP 429). "
                "Free-tier per-minute or per-day cap reached. "
                "Wait briefly (Groq resets faster than Gemini) or "
                "switch back to provider: 'gemini' in default_config.yaml.",
                retry_after_seconds=_extract_retry_after(exc),
                cause=exc,
            ) from exc

    def generate(self, prompt, **kwargs) -> LLMResult:
        if self._client is None:
            raise RuntimeError(
                f"GroqLLMClient is not configured. Set the {self._cfg.api_key_env} "
                f"(or {_GROQ_KEY_ENV_DEFAULT}) environment variable and "
                "`pip install openai>=1.0`."
            )

        t0 = time.perf_counter()
        first_token_time: float | None = None
        text_parts: list[str] = []
        try:
            for piece in self.generate_stream(prompt):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                text_parts.append(piece)
        except LLMQuotaExhaustedError:
            raise
        except Exception as exc:
            self._raise_for_quota(exc)
            raise
        t1 = time.perf_counter()

        return LLMResult(
            text="".join(text_parts),
            ttft_ms=(first_token_time - t0) * 1000 if first_token_time else (t1 - t0) * 1000,
            total_ms=(t1 - t0) * 1000,
            model=self._resolved_model_name,
            server_prompt_time_ms=getattr(self, "last_prompt_time_ms", 0.0),
            server_queue_time_ms=getattr(self, "last_queue_time_ms", 0.0),
        )

    def generate_stream(self, prompt, **kwargs) -> Iterable[str]:
        """Yield text deltas from a Groq streaming completion.

        Quota errors are translated to :class:`LLMQuotaExhaustedError`
        and raised on the *first* attempted ``next()`` so the dashboard
        can show a clean countdown instead of a multi-minute stall.
        """
        if self._client is None:
            raise RuntimeError(
                f"GroqLLMClient is not configured. Set the {self._cfg.api_key_env} "
                f"(or {_GROQ_KEY_ENV_DEFAULT}) environment variable and "
                "`pip install openai>=1.0`."
            )

        self.last_prompt_time_ms = 0.0
        self.last_queue_time_ms = 0.0

        try:
            stream = self._chat_stream(prompt)
        except BaseException as exc:
            self._raise_for_quota(exc)
            raise

        try:
            first = next(stream)
            self._inspect_chunk_usage(first)
        except BaseException as exc:
            self._raise_for_quota(exc)
            raise

        def _iter():
            try:
                yield _safe_chunk_text(first)
                for chunk in stream:
                    self._inspect_chunk_usage(chunk)
                    yield _safe_chunk_text(chunk)
            except BaseException as exc:
                self._raise_for_quota(exc)
                raise

        return _iter()


def _safe_chunk_text(chunk) -> str:
    """Extract the text delta from an OpenAI streaming chunk.

    The shape is ``chunk.choices[0].delta.content`` -- which may be
    ``None`` on tool-call or role-only chunks (the very first chunk
    in a stream often has ``content=None`` and only sets the role).
    Returning "" for non-text chunks lets the caller concatenate
    without explicit branching.
    """
    try:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        text = getattr(delta, "content", None)
        if isinstance(text, str):
            return text
    except Exception:
        return ""
    return ""