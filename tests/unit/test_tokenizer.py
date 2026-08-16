"""Tests for Phase 1: tokenizer utilities."""

from __future__ import annotations

import pytest

from backend.embeddings.tokenizer import (
    TiktokenTokenizer,
    count_tokens,
    get_tokenizer,
    set_tokenizer,
)


def test_default_tokenizer_is_tiktoken() -> None:
    tok = get_tokenizer()
    assert isinstance(tok, TiktokenTokenizer)


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0


def test_count_tokens_basic() -> None:
    # Sanity: 1 token per ASCII word (roughly) for cl100k_base.
    n = count_tokens("hello world")
    assert 2 <= n <= 3


def test_count_tokens_idempotent() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)


def test_tiktoken_tokenizer_encode_decode_round_trip() -> None:
    tok = TiktokenTokenizer()
    text = "Token-Diet compresses retrieved context."
    toks = tok.encode(text)
    assert isinstance(toks, list)
    assert all(isinstance(t, int) for t in toks)
    assert tok.decode(toks) == text


def test_set_tokenizer_round_trip() -> None:
    class Stub:
        def count(self, text: str) -> int:
            return len(text)

    stub = Stub()
    original = get_tokenizer()
    try:
        set_tokenizer(stub)  # type: ignore[arg-type]
        assert get_tokenizer() is stub
        assert count_tokens("abc") == 3
    finally:
        set_tokenizer(original)  # restore


def test_count_tokens_handles_unicode() -> None:
    n = count_tokens("café — naïve — 北京")
    assert n > 0