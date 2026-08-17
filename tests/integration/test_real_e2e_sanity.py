"""End-to-end sanity check on the real demo corpus.

Tests the *real* pipeline (real embedder, real cross-encoder, real reranker)
on the demo_company documents. Does NOT need a live Gemini key.

Verifies:
1. 10 documents load successfully and embed cleanly.
2. Retrieval returns non-empty chunks for a known query.
3. The compressor produces a smaller context than the raw retrieval.
4. The compressor PRESERVES a known answer-bearing chunk for a known query.
5. The compressor's output answers the question it was asked (must contain
   the keyword from the answer).
6. No exception is raised anywhere in the pipeline.
7. unit_formation, fast_filter, rerank, selection, pack all complete.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.config.config import load_config  # noqa: E402
from backend.embeddings.local_models import (  # noqa: E402
    SentenceTransformersCrossEncoder,
    SentenceTransformersEmbedder,
)
from backend.compressor.pipeline import (  # noqa: E402
    PipelineComponents,
    RegexSentenceSplitter,
    compress_context,
)
from backend.rag.database import VectorDatabase  # noqa: E402


FIXTURES_DIR = REPO_ROOT / "datasets" / "demo_company" / "documents"


def _load_corpus(db: VectorDatabase) -> int:
    docs = [
        (p.stem, p.read_text(encoding="utf-8"))
        for p in sorted(FIXTURES_DIR.rglob("*.md"))
        if "SAMPLE FIXTURE" in p.read_text(encoding="utf-8")
        or "SYNTHETIC DEVELOPMENT" in p.read_text(encoding="utf-8")
    ]
    for doc_id, text in docs:
        db.add_document(doc_id, text)
    return len(docs)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def components(cfg):
    device = cfg.system.device
    return PipelineComponents(
        splitter=RegexSentenceSplitter(),
        embedder=SentenceTransformersEmbedder(
            model_name=cfg.retriever.embedding_model, device=device
        ),
        cross_encoder=SentenceTransformersCrossEncoder(
            model_name=cfg.compressor.cross_encoder_model, device=device
        ),
    )


@pytest.fixture(scope="module")
def db(components):
    db = VectorDatabase(embedder=components.embedder)
    n = _load_corpus(db)
    assert n >= 10, f"Expected at least 10 docs, got {n}"
    return db


# ---------------------------------------------------------------- tests ----


def test_corpus_loads(db):
    """The demo corpus has at least 10 documents; all embed cleanly."""
    assert db.size >= 10


def test_retrieval_returns_k_chunks(db, cfg):
    query = "data retention and backup policies for NovaDB"
    chunks = db.retrieve(query, top_k=cfg.retriever.top_k)
    assert len(chunks) == cfg.retriever.top_k
    assert all(c.text.strip() for c in chunks)


def test_retrieval_finds_relevant_doc(db, cfg):
    """Top-k for a NovaDB query must include the data_retention doc."""
    chunks = db.retrieve(
        "data retention and backup policies for NovaDB clusters",
        top_k=cfg.retriever.top_k,
    )
    doc_ids = [c.doc_id for c in chunks]
    assert "data_retention" in doc_ids, (
        f"Expected data_retention in top-{cfg.retriever.top_k}, got {doc_ids}"
    )


def test_compressor_runs_without_error(db, components, cfg):
    """Compress must succeed for a known query."""
    query = "data retention and backup policies for NovaDB clusters"
    chunks = db.retrieve(query, top_k=cfg.retriever.top_k)
    compressed = compress_context(query, chunks, cfg=cfg.compressor, components=components)
    assert compressed.compressed_text.strip()
    assert compressed.metrics["compressed_token_count"] > 0


def test_compressor_reduces_token_count(db, components, cfg):
    """Compressed output must be smaller than the raw retrieval."""
    query = "data retention and backup policies for NovaDB clusters"
    chunks = db.retrieve(query, top_k=cfg.retriever.top_k)
    compressed = compress_context(query, chunks, cfg=cfg.compressor, components=components)
    orig = compressed.metrics["original_token_count"]
    comp = compressed.metrics["compressed_token_count"]
    assert comp < orig, f"Compression did nothing: {comp} >= {orig}"
    assert comp <= cfg.compressor.global_token_budget + 50, (
        f"Compressor exceeded budget: {comp} > {cfg.compressor.global_token_budget}+50"
    )


def test_compressor_preserves_answer_keyword(db, components, cfg):
    """The NovaDB backup line must survive compression."""
    query = "data retention and backup policies for NovaDB clusters"
    chunks = db.retrieve(query, top_k=cfg.retriever.top_k)
    compressed = compress_context(query, chunks, cfg=cfg.compressor, components=components)
    text = compressed.compressed_text.lower()
    # The factual answer lives in this sentence in the source corpus.
    assert "novadb" in text, "Compressor dropped 'NovaDB' reference"
    assert "continuous archiving" in text or "30 days" in text or "35 days" in text, (
        "Compressor dropped the NovaDB backup detail"
    )


def test_compressor_reports_all_stages(db, components, cfg):
    """The breakdown must include all five pipeline stages."""
    query = "What are the data retention and backup policies for NovaDB clusters?"
    chunks = db.retrieve(query, top_k=cfg.retriever.top_k)
    compressed = compress_context(query, chunks, cfg=cfg.compressor, components=components)
    metrics = compressed.metrics
    for key in ("unit_formation_ms", "fast_filter_ms", "rerank_ms",
                "selection_ms", "pack_ms"):
        assert key in metrics, f"Missing stage metric: {key}"
        assert metrics[key] >= 0, f"Negative time on {key}: {metrics[key]}"


def test_compressor_handles_unrelated_query(db, components, cfg):
    """A query with no direct match must still compress cleanly (no crash)."""
    query = "tell me about the meaning of life and the universe"
    chunks = db.retrieve(query, top_k=cfg.retriever.top_k)
    compressed = compress_context(query, chunks, cfg=cfg.compressor, components=components)
    assert compressed.compressed_text.strip()
    assert compressed.metrics["compressed_token_count"] > 0


def test_no_pipeline_exceptions_across_queries(db, components, cfg):
    """Smoke test: 5 random-ish queries should all complete without raise."""
    queries = [
        "What are the data retention and backup policies for NovaDB clusters?",
        "How does NovaCompute handle scaling?",
        "What is the maximum message retention for NovaStream?",
        "How are NovaCloud billing and invoices handled?",
        "What is the NovaCloud SLA for uptime?",
    ]
    for q in queries:
        chunks = db.retrieve(q, top_k=cfg.retriever.top_k)
        compressed = compress_context(q, chunks, cfg=cfg.compressor, components=components)
        assert compressed.compressed_text
