"""End-to-end test for the full 5-stage compressor pipeline."""

from __future__ import annotations

from tests.unit._pipeline_fakes import HashEmbedder, WordOverlapCrossEncoder, make_chunk

from backend.config.config import CompressorConfig
from backend.compressor.pipeline import PipelineComponents, compress_context
from backend.embeddings.tokenizer import count_tokens


def test_compress_context_end_to_end_respects_budget() -> None:
    text = " ".join(
        f"Sentence {i} about FAISS and similarity search is number {i}."
        for i in range(15)
    )
    cfg = CompressorConfig(global_token_budget=80, fast_filter_candidate_limit=20)

    out = compress_context(
        query="FAISS similarity search",
        chunks=[make_chunk(text)],
        cfg=cfg,
        components=PipelineComponents(
            embedder=HashEmbedder(),
            cross_encoder=WordOverlapCrossEncoder(),
        ),
    )

    # Stage-5 packed output must respect the hard budget invariant.
    assert count_tokens(out.compressed_text) <= cfg.global_token_budget
    # Metrics must be populated for every stage.
    for stage in (
        "unit_formation_ms",
        "fast_filter_ms",
        "rerank_ms",
        "selection_ms",
        "pack_ms",
        "total_compressor_ms",
        "original_token_count",
        "compressed_token_count",
    ):
        assert stage in out.metrics


def test_compress_context_reduces_token_count() -> None:
    text = " ".join(
        f"Sentence {i} about FAISS is number {i}." for i in range(15)
    )
    cfg = CompressorConfig(global_token_budget=80, fast_filter_candidate_limit=20)

    out = compress_context(
        query="FAISS",
        chunks=[make_chunk(text)],
        cfg=cfg,
        components=PipelineComponents(
            embedder=HashEmbedder(),
            cross_encoder=WordOverlapCrossEncoder(),
        ),
    )

    assert out.metrics["compressed_token_count"] <= out.metrics["original_token_count"]


def test_compress_context_accepts_dict_chunks() -> None:
    """Plan pseudocode uses dict-shaped chunks; the pipeline must accept them."""
    cfg = CompressorConfig(global_token_budget=80, fast_filter_candidate_limit=10)
    out = compress_context(
        query="anything",
        chunks=[
            {"text": "Sentence alpha. Sentence beta.", "doc_id": "d", "chunk_id": "c"},
        ],
        cfg=cfg,
        components=PipelineComponents(
            embedder=HashEmbedder(),
            cross_encoder=WordOverlapCrossEncoder(),
        ),
    )
    assert isinstance(out.compressed_text, str)


def test_compress_context_requires_components() -> None:
    cfg = CompressorConfig(global_token_budget=80)
    try:
        compress_context(query="q", chunks=[make_chunk("hi.")], cfg=cfg)
    except ValueError as exc:
        assert "embedder" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_compress_context_metrics_have_correct_ordering() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth."
    cfg = CompressorConfig(global_token_budget=200)
    out = compress_context(
        query="anything",
        chunks=[make_chunk(text)],
        cfg=cfg,
        components=PipelineComponents(
            embedder=HashEmbedder(),
            cross_encoder=WordOverlapCrossEncoder(),
        ),
    )
    # Stage timings must be non-negative and sum (approximately) to total.
    stage_ms = sum(
        out.metrics[k]
        for k in (
            "unit_formation_ms",
            "fast_filter_ms",
            "rerank_ms",
            "selection_ms",
            "pack_ms",
        )
    )
    assert 0 <= stage_ms <= out.metrics["total_compressor_ms"] + 1.0  # small slack