"""Tests for Phase 1: data models and the cosine similarity helper."""

from __future__ import annotations

import math

import pytest

from backend.rag.models import (
    CompressorOutput,
    ContextUnit,
    PROSE,
    RetrievedChunk,
    ScoredCandidate,
    STRUCTURED,
    cosine_similarity,
)


def _make_unit(text: str = "hello world", **overrides) -> ContextUnit:
    base = dict(
        unit_id="d1_c1_0",
        doc_id="d1",
        chunk_id="c1",
        target_text=text,
        scoring_text=text,
        unit_type=PROSE,
        position_idx=0,
        parent_chunk_id="c1",
        token_count=2,
    )
    base.update(overrides)
    return ContextUnit(**base)


def test_context_unit_defaults() -> None:
    u = _make_unit()
    assert u.embedding is None
    assert u.metadata == {}
    assert u.token_count == 2


def test_scored_candidate_defaults() -> None:
    sc = ScoredCandidate(unit=_make_unit())
    assert sc.lexical_score == 0.0
    assert sc.embedding_score == 0.0
    assert sc.rerank_score == 0.0
    assert sc.final_score == 0.0


def test_compressor_output_basic() -> None:
    out = CompressorOutput(
        compressed_text="[1] hello",
        selected_units=[_make_unit()],
        metrics={"foo": 1},
    )
    assert out.compressed_text == "[1] hello"
    assert len(out.selected_units) == 1
    assert out.metrics["foo"] == 1


def test_retrieved_chunk_basic() -> None:
    c = RetrievedChunk(text="t", doc_id="d", chunk_id="c", score=0.5)
    assert c.metadata == {}


def test_cosine_similarity_identical() -> None:
    v = [1.0, 2.0, 3.0]
    assert math.isclose(cosine_similarity(v, v), 1.0, abs_tol=1e-9)


def test_cosine_similarity_orthogonal() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_mismatched_length() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_similarity_none_inputs() -> None:
    assert cosine_similarity(None, [1.0]) == 0.0
    assert cosine_similarity([1.0], None) == 0.0


def test_unit_type_constants() -> None:
    assert PROSE == "prose"
    assert STRUCTURED == "structured"