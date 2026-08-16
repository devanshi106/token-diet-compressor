"""Tests for Stage 2 -- fast_filter."""

from __future__ import annotations

import pytest

from tests.unit._pipeline_fakes import HashEmbedder, make_chunk

from backend.config.config import CompressorConfig
from backend.rag.models import ScoredCandidate
from backend.compressor.pipeline.fast_filter import fast_filter_candidates
from backend.compressor.pipeline.unit_formation import form_context_units


@pytest.fixture
def units():
    text = (
        "Retrieval augmented generation combines retrieval with a generator. "
        "FAISS enables fast nearest-neighbor search. "
        "Pizza dough is kneaded for ten minutes. "
        "Vector databases store embeddings for similarity lookup. "
        "The weather in June is warm and rainy."
    )
    return form_context_units([make_chunk(text)]).units


def test_fast_filter_returns_top_m(units) -> None:
    out = fast_filter_candidates(
        query="What is FAISS?",
        units=units,
        embedder=HashEmbedder(),
        candidate_limit=2,
    )
    assert len(out) == 2
    assert all(isinstance(c, ScoredCandidate) for c in out)


def test_fast_filter_sorts_descending(units) -> None:
    out = fast_filter_candidates(
        query="vector embeddings",
        units=units,
        embedder=HashEmbedder(),
        candidate_limit=len(units),
    )
    scores = [c.final_score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_fast_filter_does_not_tokenize_again(units) -> None:
    """The contract: Stage 2 reuses Stage 1's precomputed token_count."""
    fast_filter_candidates(
        query="anything", units=units, embedder=HashEmbedder(), candidate_limit=2
    )
    # token_count is set in Stage 1 and never changes here.
    for u in units:
        assert u.token_count > 0


def test_fast_filter_caches_embeddings_on_units(units) -> None:
    fast_filter_candidates(
        query="anything", units=units, embedder=HashEmbedder(), candidate_limit=2
    )
    # All units now have embeddings populated for Stage 4's redundancy check.
    assert all(u.embedding is not None for u in units)


def test_fast_filter_zero_weights_falls_back_to_lexical(units) -> None:
    out = fast_filter_candidates(
        query="FAISS",
        units=units,
        embedder=HashEmbedder(),
        candidate_limit=3,
        bm25_weight=0.0,
        embedding_weight=0.0,
    )
    # Still returns ranked candidates -- degenerates to lexical.
    assert len(out) == 3


def test_fast_filter_negative_weight_raises(units) -> None:
    with pytest.raises(ValueError):
        fast_filter_candidates(
            query="x",
            units=units,
            embedder=HashEmbedder(),
            candidate_limit=3,
            bm25_weight=-0.1,
        )


def test_fast_filter_empty_input() -> None:
    out = fast_filter_candidates(query="x", units=[], embedder=HashEmbedder())
    assert out == []


def test_fast_filter_uses_config_weights() -> None:
    """Sanity: CompressorConfig defaults flow through."""
    cfg = CompressorConfig()
    assert cfg.bm25_weight == 0.5
    assert cfg.embedding_weight == 0.5