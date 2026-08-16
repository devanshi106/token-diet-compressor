"""Tests for Stage 3 -- reranker."""

from __future__ import annotations

import pytest

from tests.unit._pipeline_fakes import HashEmbedder, WordOverlapCrossEncoder, make_chunk

from backend.rag.interfaces import CrossEncoder
from backend.rag.models import ContextUnit, ScoredCandidate
from backend.compressor.pipeline.fast_filter import fast_filter_candidates
from backend.compressor.pipeline.reranker import rerank_candidates
from backend.compressor.pipeline.unit_formation import form_context_units


@pytest.fixture
def candidates():
    text = (
        "FAISS is a vector database library. "
        "Pizza toppings include cheese and tomato. "
        "Vector databases enable fast similarity search."
    )
    units = form_context_units([make_chunk(text)]).units
    return fast_filter_candidates(
        query="vector database FAISS",
        units=units,
        embedder=HashEmbedder(),
        candidate_limit=10,
    )


def test_reranker_populates_scores(candidates) -> None:
    out = rerank_candidates(
        query="vector database FAISS",
        candidates=candidates,
        cross_encoder=WordOverlapCrossEncoder(),
    )
    assert all(0.0 <= c.rerank_score <= 1.0 for c in out)


def test_reranker_sorts_descending_by_rerank(candidates) -> None:
    out = rerank_candidates(
        query="vector database FAISS",
        candidates=candidates,
        cross_encoder=WordOverlapCrossEncoder(),
    )
    scores = [c.rerank_score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_reranker_empty_input_returns_empty() -> None:
    out = rerank_candidates(query="x", candidates=[], cross_encoder=WordOverlapCrossEncoder())
    assert out == []


def test_reranker_pair_count_mismatch_raises() -> None:
    class WrongLengthCrossEncoder(CrossEncoder):
        def predict(self, pairs, batch_size=32):
            # Always two scores, no matter how many pairs we get.
            return [0.0, 0.0]

    cand = ScoredCandidate(
        unit=ContextUnit(
            unit_id="u",
            doc_id="d",
            chunk_id="c",
            target_text="t",
            scoring_text="t",
            unit_type="prose",
            position_idx=0,
            parent_chunk_id="c",
        )
    )
    with pytest.raises(ValueError):
        rerank_candidates(
            query="q",
            candidates=[cand],
            cross_encoder=WrongLengthCrossEncoder(),
        )