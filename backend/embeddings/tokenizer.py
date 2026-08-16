"""Tokenizer utilities.

Per the implementation plan (§9), we use :mod:`tiktoken` for exact token
counts because it matches OpenAI-family tokenizers and is fast.

Strategy (matches §11 of the plan):

* :func:`count_tokens` is the authoritative exact tokenizer. It is
  called *once* on each ``ContextUnit.target_text`` during unit
  formation (storing the result in ``ContextUnit.token_count``), and
  *once* on the final packed context for budget validation. We never
  call it inside the hot selection loop.

A small :class:`Tokenizer` protocol is exposed so a future
``AutoTokenizer``-backed implementation can be dropped in without
touching the rest of the pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import tiktoken


class Tokenizer(Protocol):
    """Anything that can count tokens in a string."""

    def count(self, text: str) -> int: ...


# ---------------------------------------------------------------------------
# tiktoken-backed implementation
# ---------------------------------------------------------------------------

# cl100k_base matches gpt-4 / gpt-4o-mini / text-embedding-3-*; it's a
# reasonable default that lines up with the OpenAI models the plan
# targets. If we later target a different model family we override
# ``DEFAULT_ENCODING`` via the constructor.
DEFAULT_ENCODING = "cl100k_base"


class TiktokenTokenizer:
    """Exact, fast tokenizer backed by ``tiktoken``."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        self._encoding = _get_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


@lru_cache(maxsize=8)
def _get_encoding(name: str):
    """Cache encoded objects so we don't pay the bootstrap cost per call."""
    return tiktoken.get_encoding(name)


# ---------------------------------------------------------------------------
# Module-level default
# ---------------------------------------------------------------------------

_default_tokenizer: Tokenizer = TiktokenTokenizer()


def get_tokenizer() -> Tokenizer:
    """Return the process-wide default tokenizer."""
    return _default_tokenizer


def set_tokenizer(tokenizer: Tokenizer) -> None:
    """Replace the process-wide default tokenizer."""
    global _default_tokenizer
    _default_tokenizer = tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` using the default tokenizer."""
    return _default_tokenizer.count(text)