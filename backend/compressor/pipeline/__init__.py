"""Compression pipeline (plan §3).

Public orchestrator: :func:`compress_context`. Stage entry points are
exposed for fine-grained testing:

* :func:`unit_formation.form_context_units` -- Stage 1
* :func:`fast_filter.fast_filter_candidates`  -- Stage 2
* :func:`reranker.rerank_candidates`         -- Stage 3
* :func:`selector.select_budgeted_candidates` -- Stage 4
* :func:`packer.pack_and_order_context`      -- Stage 5
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable
from backend.config.config import CompressorConfig, LLMConfig, AppConfig
from backend.rag.interfaces import CrossEncoder, Embedder, SentenceSplitter
from backend.rag.models import CompressorOutput, RetrievedChunk
from backend.embeddings.tokenizer import count_tokens
from .fast_filter import fast_filter_candidates
from .packer import pack_and_order_context
from .reranker import rerank_candidates
from .selector import SelectionConfig, select_budgeted_candidates
from .unit_formation import RegexSentenceSplitter, UnitFormationOutput, form_context_units


@dataclass
class PipelineComponents:
    """Pluggable pieces of the pipeline. Default to deterministic fakes in tests."""

    splitter: SentenceSplitter | None = None
    embedder: Embedder | None = None
    cross_encoder: CrossEncoder | None = None


def compress_context(
    query: str,
    chunks: list[RetrievedChunk] | Iterable[dict],
    cfg: CompressorConfig | None = None,
    components: PipelineComponents | None = None,
) -> CompressorOutput:
    """Run the 5-stage compression pipeline on a query + retrieved chunks.

    Stage 1  Unit Formation    -- split chunks into semantic units (prose/table/code)
    Stage 2  Fast Filter       -- rank units by cheap BM25 + embedding blended score
    Stage 3  Cross-Encoder     -- re-rank with full query-document attention
    Stage 4  Budget Selection  -- greedy pack into token budget
    Stage 5  Pack & Order      -- assemble final context string

    The cross-encoder (Stage 3) is the primary latency cost on CPU (~10-15s)
    but produces meaningfully better selection than the fast blended score,
    especially for queries where lexical and embedding signals disagree.
    """

    cfg = cfg or CompressorConfig()
    components = components or PipelineComponents()

    if components.embedder is None or components.cross_encoder is None:
        raise ValueError(
            "compress_context requires an embedder and a cross_encoder. "
            "Inject both via PipelineComponents."
        )

    chunks = _coerce_chunks(chunks)
    t_start = time.perf_counter()

    # Stage 1 -- Unit Formation
    units_out = form_context_units(
        chunks,
        splitter=components.splitter or RegexSentenceSplitter(),
    )
    t_units = time.perf_counter()

    # Stage 2 -- Fast Relevance Filter
    candidates = fast_filter_candidates(
        query=query,
        units=units_out.units,
        embedder=components.embedder,
        candidate_limit=cfg.fast_filter_candidate_limit,
        bm25_weight=cfg.bm25_weight,
        embedding_weight=cfg.embedding_weight,
    )
    t_filter = time.perf_counter()

    # Stage 3 -- Batched Cross-Encoder Rerank
    candidates = rerank_candidates(
        query=query,
        candidates=candidates,
        cross_encoder=components.cross_encoder,
        batch_size=cfg.cross_encoder_batch_size,
    )
    t_rerank = time.perf_counter()

    # Stage 4 -- Token-Budgeted Selection
    selection_cfg = SelectionConfig(
        global_token_budget=cfg.global_token_budget,
        restoration_window_left=cfg.restoration_window_left,
        restoration_window_right=cfg.restoration_window_right,
        similarity_threshold=cfg.similarity_threshold,
        markdown_header_overhead_tokens=cfg.markdown_header_overhead_tokens,
    )
    selected = select_budgeted_candidates(
        candidates,
        units_out.parent_chunks,
        selection_cfg,
        score_key="rerank_score",
    )
    t_select = time.perf_counter()

    # Stage 5 -- Pack & Order
    compressed = pack_and_order_context(selected, units_out.parent_chunks)
    t_pack = time.perf_counter()

    metrics = {
        "unit_formation_ms": (t_units - t_start) * 1000,
        "fast_filter_ms": (t_filter - t_units) * 1000,
        "rerank_ms": (t_rerank - t_filter) * 1000,
        "selection_ms": (t_select - t_rerank) * 1000,
        "pack_ms": (t_pack - t_select) * 1000,
        "total_compressor_ms": (t_pack - t_start) * 1000,
        "original_token_count": count_tokens(
            " ".join(c.text for c in chunks)
        ),
        "compressed_token_count": count_tokens(compressed),
        "selected_unit_count": len(selected),
        "candidate_count_after_filter": len(candidates),
    }

    return CompressorOutput(
        compressed_text=compressed,
        selected_units=selected,
        metrics=metrics,
    )


def _coerce_chunks(chunks) -> list[RetrievedChunk]:
    """Accept either RetrievedChunk instances or plan-style dicts."""
    out: list[RetrievedChunk] = []
    for c in chunks:
        if isinstance(c, RetrievedChunk):
            out.append(c)
            continue
        out.append(
            RetrievedChunk(
                text=c["text"],
                doc_id=c.get("doc_id", "doc"),
                chunk_id=c.get("chunk_id", "chunk"),
                score=c.get("score", 0.0),
                metadata={k: v for k, v in c.items() if k not in {"text", "doc_id", "chunk_id", "score"}},
            )
        )
    return out