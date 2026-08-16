"""Benchmarking + evaluation engine (plan §19, §20).

Two experiments:

* **Experiment A** -- *Compressor Effectiveness / Controlled Comparison*.
  Both pipelines receive the same pre-cached retrieved chunks so we can
  isolate the compressor's contribution.
* **Experiment B** -- *True End-to-End Performance*. Both pipelines
  execute independently from query to final answer.

A small but representative evaluation set ships with the project
(:data:`DEFAULT_QUERY_SET`). Each query carries optional
``required_keywords`` to enable deterministic correctness checks (plan
§20: "exact factual match constraints").
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable
from backend.config.config import AppConfig, load_config
from backend.rag.database import VectorDatabase
from backend.llm.gemini_client import LLMClient, FakeLLMClient
from backend.rag.models import RetrievedChunk
from backend.rag.normal_rag import NormalRAG
from backend.compressor.pipeline import PipelineComponents
from backend.rag.smart_rag import SmartRAG


# ---------------------------------------------------------------------------
# Evaluation queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query."""

    query: str
    required_keywords: tuple[str, ...] = ()  # must appear in the answer
    tags: tuple[str, ...] = ()


# A small but representative starting set. Plan §20 calls for ~30; this
# is intentionally a *sample* so tests stay fast. Extend as you grow the
# corpus.
DEFAULT_QUERY_SET: tuple[EvalQuery, ...] = (
    EvalQuery(
        query="What does FAISS enable for vector search?",
        required_keywords=("FAISS",),
        tags=("factual",),
    ),
    EvalQuery(
        query="How does Token-Diet reduce LLM token usage?",
        required_keywords=("token",),
        tags=("system",),
    ),
    EvalQuery(
        query="What is the retrieval step in RAG?",
        required_keywords=("retriev",),
        tags=("system",),
    ),
    EvalQuery(
        query="When should the Cross-Encoder be used in the pipeline?",
        required_keywords=("Cross-Encoder", "rerank"),
        tags=("system",),
    ),
    EvalQuery(
        query="How does the compressor handle context-dependent pronouns?",
        required_keywords=("scoring_text", "context"),
        tags=("system", "pronoun"),
    ),
    EvalQuery(
        query="Explain the global token budget invariant.",
        required_keywords=("budget",),
        tags=("system", "budget"),
    ),
    EvalQuery(
        query="Compare BM25 vs embedding similarity for fast filtering.",
        required_keywords=("BM25", "embed"),
        tags=("system",),
    ),
    EvalQuery(
        query="What stages form the Token-Diet compressor?",
        required_keywords=("filter", "rank", "select"),
        tags=("system",),
    ),
    EvalQuery(
        query="Why is it useful to pack context by source document?",
        required_keywords=("source", "document"),
        tags=("system", "format"),
    ),
    EvalQuery(
        query="What happens if no candidates fit the token budget?",
        required_keywords=("fallback",),
        tags=("edge",),
    ),
)


# ---------------------------------------------------------------------------
# Experiment A -- controlled comparison
# ---------------------------------------------------------------------------


@dataclass
class ExperimentAResult:
    """One Experiment A pass."""

    query: str
    required_keywords: tuple[str, ...]
    normal_answer: str
    smart_answer: str
    normal_tokens: int
    smart_tokens: int
    token_compression_pct: float
    normal_correct: bool
    smart_correct: bool
    normal_prompt_tokens: int
    smart_prompt_tokens: int


def run_experiment_a(
    query: EvalQuery,
    chunks: list,
    cfg: AppConfig,
    llm: LLMClient | None = None,
    components: PipelineComponents | None = None,
) -> ExperimentAResult:
    """Run a single query through both pipelines using the *same* chunks.

    This isolates the compressor from retrieval variance. ``chunks`` may
    be either ``RetrievedChunk`` objects or plain dicts with
    ``text`` / ``doc_id`` / ``chunk_id`` keys.
    """
    from backend.compressor.pipeline import compress_context
    from backend.rag.prompt import build_messages
    from backend.embeddings.tokenizer import count_tokens

    llm = llm or FakeLLMClient()

    # Normalize chunks once so both pipelines see the same shape.
    normalized = [_coerce_chunk(c) for c in chunks]
    # Normal pipeline: build the raw prompt and call the LLM.
    raw_context = "\n\n".join(c.text for c in normalized)
    raw_prompt = build_messages(
        query=query.query,
        chunks=normalized,
        system_prompt="You are a helpful assistant. Answer using only the provided context.",
    )
    # Replace the user message with our standard format.
    raw_prompt[-1] = {
        "role": "user",
        "content": f"Context:\n{raw_context}\n\nQuestion: {query.query}\nAnswer:",
    }
    normal_result = llm.generate(raw_prompt)

    # Smart pipeline: compress then call the same LLM.
    smart_compressed = compress_context(
        query=query.query,
        chunks=chunks,
        cfg=cfg.compressor,
        components=components,
    )
    smart_prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer using only the provided context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{smart_compressed.compressed_text}\n\nQuestion: {query.query}\nAnswer:",
        },
    ]
    smart_result = llm.generate(smart_prompt)

    normal_tokens = count_tokens(" ".join(c.text for c in normalized))
    smart_tokens = smart_compressed.metrics["compressed_token_count"]

    compression = (
        100.0 * (1.0 - smart_tokens / normal_tokens) if normal_tokens else 0.0
    )

    return ExperimentAResult(
        query=query.query,
        required_keywords=query.required_keywords,
        normal_answer=normal_result.text,
        smart_answer=smart_result.text,
        normal_tokens=normal_tokens,
        smart_tokens=smart_tokens,
        token_compression_pct=compression,
        normal_correct=_answer_contains_all(normal_result.text, query.required_keywords),
        smart_correct=_answer_contains_all(smart_result.text, query.required_keywords),
        normal_prompt_tokens=count_tokens(raw_prompt[-1]["content"]),
        smart_prompt_tokens=count_tokens(smart_prompt[-1]["content"]),
    )


# ---------------------------------------------------------------------------
# Experiment B -- true end-to-end
# ---------------------------------------------------------------------------


@dataclass
class ExperimentBResult:
    """One Experiment B pass."""

    query: str
    required_keywords: tuple[str, ...]
    normal_total_ms: float
    smart_total_ms: float
    net_latency_savings_ms: float
    normal_context_tokens: int
    smart_context_tokens: int
    token_compression_pct: float
    normal_correct: bool
    smart_correct: bool


def run_experiment_b(
    query: EvalQuery,
    db: VectorDatabase,
    llm: LLMClient,
    cfg: AppConfig,
    components: PipelineComponents | None = None,
) -> ExperimentBResult:
    """Run both pipelines against a real database."""

    normal = NormalRAG(db, llm, cfg).run(query.query)
    smart = SmartRAG(db, llm, cfg, components=components).run(query.query)

    cmp = SmartRAG.compare(normal, smart)

    return ExperimentBResult(
        query=query.query,
        required_keywords=query.required_keywords,
        normal_total_ms=normal.total_time_ms,
        smart_total_ms=smart.total_time_ms,
        net_latency_savings_ms=cmp["net_latency_savings_ms"],
        normal_context_tokens=normal.context_tokens,
        smart_context_tokens=smart.compressed_tokens,
        token_compression_pct=cmp["token_compression_pct"],
        normal_correct=_answer_contains_all(normal.answer, query.required_keywords),
        smart_correct=_answer_contains_all(smart.answer, query.required_keywords),
    )


# ---------------------------------------------------------------------------
# Aggregate runners
# ---------------------------------------------------------------------------


@dataclass
class AggregateReport:
    """Summary statistics across many queries."""

    n_queries: int
    avg_token_compression_pct: float
    avg_net_latency_savings_ms: float
    normal_correctness_pct: float
    smart_correctness_pct: float
    rows: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "avg_token_compression_pct": self.avg_token_compression_pct,
            "avg_net_latency_savings_ms": self.avg_net_latency_savings_ms,
            "normal_correctness_pct": self.normal_correctness_pct,
            "smart_correctness_pct": self.smart_correctness_pct,
            "rows": self.rows,
        }


def aggregate_a(results: Iterable[ExperimentAResult]) -> AggregateReport:
    rs = list(results)
    if not rs:
        return AggregateReport(0, 0.0, 0.0, 0.0, 0.0)
    return AggregateReport(
        n_queries=len(rs),
        avg_token_compression_pct=_mean(r.token_compression_pct for r in rs),
        avg_net_latency_savings_ms=0.0,
        normal_correctness_pct=_pct(r.normal_correct for r in rs),
        smart_correctness_pct=_pct(r.smart_correct for r in rs),
        rows=[asdict(r) for r in rs],
    )


def aggregate_b(results: Iterable[ExperimentBResult]) -> AggregateReport:
    rs = list(results)
    if not rs:
        return AggregateReport(0, 0.0, 0.0, 0.0, 0.0)
    return AggregateReport(
        n_queries=len(rs),
        avg_token_compression_pct=_mean(r.token_compression_pct for r in rs),
        avg_net_latency_savings_ms=_mean(r.net_latency_savings_ms for r in rs),
        normal_correctness_pct=_pct(r.normal_correct for r in rs),
        smart_correctness_pct=_pct(r.smart_correct for r in rs),
        rows=[asdict(r) for r in rs],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _answer_contains_all(answer: str, keywords: tuple[str, ...]) -> bool:
    a = answer.lower()
    return all(kw.lower() in a for kw in keywords)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _pct(values) -> float:
    values = list(values)
    return 100.0 * sum(1 for v in values if v) / len(values) if values else 0.0


def _coerce_chunk(c: object) -> RetrievedChunk:
    """Coerce a chunk-like object (or dict) into a :class:`RetrievedChunk`."""
    if isinstance(c, RetrievedChunk):
        return c
    if hasattr(c, "text"):
        return RetrievedChunk(
            text=str(getattr(c, "text", "")),
            doc_id=str(getattr(c, "doc_id", "doc")),
            chunk_id=str(getattr(c, "chunk_id", "chunk")),
        )
    if isinstance(c, dict):
        return RetrievedChunk(
            text=str(c.get("text", "")),
            doc_id=str(c.get("doc_id", "doc")),
            chunk_id=str(c.get("chunk_id", "chunk")),
        )
    raise TypeError(f"Cannot coerce chunk of type {type(c).__name__} into RetrievedChunk")