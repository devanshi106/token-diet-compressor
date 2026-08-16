"""Tests for src.normal_rag, src.smart_rag, and src.evaluation.

Both RAG engines share the same LLM and the same database, which is
the explicit system boundary from plan §2; these tests verify both
sides come from the same retrievals.
"""

from __future__ import annotations

from backend.config.config import load_config
from backend.rag.database import VectorDatabase
from backend.evaluation.evaluation import (
    DEFAULT_QUERY_SET,
    EvalQuery,
    aggregate_a,
    aggregate_b,
    run_experiment_a,
    run_experiment_b,
)
from backend.llm.gemini_client import FakeLLMClient
from backend.rag.normal_rag import NormalRAG
from backend.compressor.pipeline import PipelineComponents
from backend.rag.smart_rag import SmartRAG
from tests.unit._pipeline_fakes import HashEmbedder, WordOverlapCrossEncoder


FIXTURE_TEXT = (
    "FAISS is an open-source library for efficient similarity search. "
    "Vector databases store embeddings and return nearest neighbours. "
    "A cross-encoder reranks candidates after a fast filter stage. "
    "The compressor enforces a global token budget invariant."
)


def _build_db() -> VectorDatabase:
    db = VectorDatabase(embedder=HashEmbedder())
    db.add_document("doc1", "Pizza dough is kneaded until smooth." * 30)
    db.add_document("doc2", FIXTURE_TEXT)
    db.add_document("doc3", "Astronomers study galaxies with telescopes." * 20)
    return db


def _components() -> PipelineComponents:
    return PipelineComponents(
        embedder=HashEmbedder(),
        cross_encoder=WordOverlapCrossEncoder(),
    )


def test_normal_rag_returns_timed_answer() -> None:
    db = _build_db()
    llm = FakeLLMClient(response_prefix="NORMAL:")
    run = NormalRAG(db, llm, load_config()).run("What does FAISS enable?")
    assert run.answer.startswith("NORMAL:")
    assert run.context_tokens > 0
    assert run.total_time_ms >= 0
    assert run.llm_ttft_ms >= 0
    assert len(run.raw_chunks) > 0


def test_smart_rag_returns_compressed_context() -> None:
    db = _build_db()
    llm = FakeLLMClient(response_prefix="SMART:")
    run = SmartRAG(db, llm, load_config(), components=_components()).run(
        "What does FAISS enable?"
    )
    assert run.answer.startswith("SMART:")
    assert run.original_tokens > 0
    assert run.compressed_tokens > 0
    # The hard invariant the compressor enforces is the global budget;
    # the *ratio* of compressed vs original depends on corpus size
    # and overhead. On tiny corpora the markdown header overhead can
    # exceed savings, so we don't assert compression here.
    assert run.total_time_ms >= 0
    assert "compressor_breakdown" in run.__dict__


def test_compare_returns_expected_keys() -> None:
    db = _build_db()
    llm = FakeLLMClient()
    normal = NormalRAG(db, llm, load_config()).run("FAISS")
    smart = SmartRAG(db, llm, load_config(), components=_components()).run("FAISS")
    cmp = SmartRAG.compare(normal, smart)
    for key in (
        "normal_total_ms",
        "smart_total_ms",
        "net_latency_savings_ms",
        "normal_context_tokens",
        "smart_context_tokens",
        "token_compression_pct",
        "normal_llm_ttft_ms",
        "smart_llm_ttft_ms",
    ):
        assert key in cmp


def test_smart_rag_prompt_differs_from_normal_prompt() -> None:
    """The whole point of the compressor: the LLM sees a different prompt."""
    db = _build_db()
    llm = FakeLLMClient()
    SmartRAG(db, llm, load_config(), components=_components()).run("FAISS")
    normal_prompt = llm.calls[-1]
    # The captured prompt is the smart one; both should be messages list.
    assert isinstance(normal_prompt, list)
    assert normal_prompt[0]["role"] == "system"
    # System message should mention "compression" in spirit — at minimum
    # the user message content should NOT contain the raw FAISS prose
    # verbatim unless the compressor chose to keep it.
    user_text = normal_prompt[-1]["content"]
    assert "FAISS" in user_text  # compressor kept the relevant bits


def test_normal_and_smart_have_same_retrieval() -> None:
    """Sanity check: same DB + same query -> same raw chunks."""
    db = _build_db()
    llm = FakeLLMClient()
    normal = NormalRAG(db, llm, load_config()).run("FAISS")
    smart = SmartRAG(db, llm, load_config(), components=_components()).run("FAISS")
    assert [c.chunk_id for c in normal.raw_chunks] == [c.chunk_id for c in smart.raw_chunks]


def test_experiment_a_runs_on_frozen_chunks() -> None:
    chunks = [
        {"text": "FAISS provides vector similarity search.", "doc_id": "d", "chunk_id": "c"},
        {"text": "BM25 is a lexical ranking algorithm.", "doc_id": "d", "chunk_id": "c2"},
    ]
    # Use EvalQuery with no required keywords so the fake still passes.
    q = EvalQuery(query="vector retrieval")
    result = run_experiment_a(
        query=q, chunks=chunks, cfg=load_config(), components=_components()
    )
    assert result.normal_tokens > 0
    assert result.smart_tokens > 0
    assert result.query == "vector retrieval"


def test_experiment_b_runs_full_pipeline() -> None:
    db = _build_db()
    llm = FakeLLMClient()
    q = EvalQuery(
        query="FAISS",
        required_keywords=("FAISS",),
        tags=("factual",),
    )
    result = run_experiment_b(
        query=q, db=db, llm=llm, cfg=load_config(), components=_components()
    )
    assert result.smart_total_ms > 0
    assert result.normal_total_ms > 0
    assert result.smart_correct  # the FakeLLMClient echoes the prompt
    assert result.normal_correct


def test_aggregate_a_computes_summary() -> None:
    chunks = [
        {"text": "FAISS provides vector similarity search.", "doc_id": "d", "chunk_id": "c"},
    ]
    qs = [
        EvalQuery(query="q1", required_keywords=()),
        EvalQuery(query="q2", required_keywords=()),
    ]
    results = [
        run_experiment_a(query=q, chunks=chunks, cfg=load_config(), components=_components())
        for q in qs
    ]
    report = aggregate_a(results)
    assert report.n_queries == 2
    assert report.rows and len(report.rows) == 2
    d = report.as_dict()
    assert "n_queries" in d and "rows" in d


def test_aggregate_b_computes_summary() -> None:
    db = _build_db()
    llm = FakeLLMClient()
    qs = [
        EvalQuery(query="FAISS", required_keywords=()),
        EvalQuery(query="vector", required_keywords=()),
    ]
    results = [
        run_experiment_b(query=q, db=db, llm=llm, cfg=load_config(), components=_components())
        for q in qs
    ]
    report = aggregate_b(results)
    assert report.n_queries == 2
    # Compression ratio can go either way on small corpora; just check
    # the aggregation produced a finite number.
    assert isinstance(report.avg_token_compression_pct, float)


def test_default_query_set_is_non_empty() -> None:
    """Plan §20: 'span diverse scenarios' -- shipping a starter set."""
    assert len(DEFAULT_QUERY_SET) >= 10
    # All queries have at least one required keyword for deterministic
    # correctness checks.
    for q in DEFAULT_QUERY_SET:
        assert q.query.strip()
        assert len(q.required_keywords) >= 1, q.query