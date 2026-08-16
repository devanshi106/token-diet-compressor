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
    """Run the full 5-stage compressor on a single query + retrieved chunks.

    Parameters
    ----------
    query:
        The user's question.
    chunks:
        Either :class:`RetrievedChunk` objects *or* plain dicts with at
        least ``{"text", "doc_id", "chunk_id"}`` (matches the plan's
        pseudocode shape).
    cfg:
        Compressor configuration. Falls back to defaults.
    components:
        Stage implementations. The pipeline fails fast if ``embedder`` or
        ``cross_encoder`` are not provided -- they have no sensible
        default (sending 800 tokens through a Cross-Encoder with a fake
        embedder would defeat the purpose of the test).
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
    units_out: UnitFormationOutput = form_context_units(
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

    # Stage 4 -- Smart Token-Budgeted Selection
    selection_cfg = SelectionConfig(
        global_token_budget=cfg.global_token_budget,
        restoration_window_left=cfg.restoration_window_left,
        restoration_window_right=cfg.restoration_window_right,
        similarity_threshold=cfg.similarity_threshold,
        markdown_header_overhead_tokens=cfg.markdown_header_overhead_tokens,
    )
    selected = select_budgeted_candidates(
        candidates=candidates,
        parent_chunks=units_out.parent_chunks,
        cfg=selection_cfg,
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