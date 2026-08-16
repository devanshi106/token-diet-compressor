from __future__ import annotations

import pytest
from backend.config.config import load_config
from backend.embeddings.local_models import SentenceTransformersEmbedder, SentenceTransformersCrossEncoder
from backend.compressor.pipeline.unit_formation import form_context_units
from backend.compressor.pipeline.fast_filter import fast_filter_candidates
from backend.compressor.pipeline.reranker import rerank_candidates
from tests.unit._pipeline_fakes import make_chunk


def test_real_models_pipeline_integration() -> None:
    # 1. Initialize real models on CPU
    embedder = SentenceTransformersEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    cross_encoder = SentenceTransformersCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )

    query = "What does FAISS do?"
    chunks = [
        make_chunk("First sentence. FAISS is a library for similarity search.", doc_id="d1", chunk_id="c1"),
        make_chunk("Unrelated information about domestic cats.", doc_id="d2", chunk_id="c2"),
        make_chunk("FAISS supports dense vector indexing on CPU and GPU.", doc_id="d3", chunk_id="c3"),
    ]

    # Stage 1: Unit Formation
    units_out = form_context_units(chunks)
    assert len(units_out.units) > 0

    # Stage 2: Fast Filter (BM25 + Similarity)
    candidates = fast_filter_candidates(
        query=query,
        units=units_out.units,
        embedder=embedder,
        candidate_limit=2,
    )
    assert len(candidates) <= 2

    # Stage 3: Cross-Encoder Reranking
    reranked = rerank_candidates(
        query=query,
        candidates=candidates,
        cross_encoder=cross_encoder,
        batch_size=2,
    )

    # Verification
    assert len(reranked) > 0
    for cand in reranked:
        assert isinstance(cand.rerank_score, float)
        assert cand.unit.scoring_text
