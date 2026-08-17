"""Smart RAG: Normal RAG + the Token-Diet compressor (plan §18).

User Query -> Retriever -> Compressor Middleware -> Compressed Context
            -> LLM -> Answer

Critically, the retriever and LLM are the *same* instances used by
:mod:`src.normal_rag`; the compressor is the only thing that changes.
That is the explicit system boundary from plan §2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from backend.config.config import AppConfig, CompressorConfig, load_config
from backend.rag.database import VectorDatabase
from backend.llm.gemini_client import LLMClient
from backend.rag.normal_rag import NormalRAGResult
from backend.compressor.pipeline import PipelineComponents, compress_context
from backend.embeddings.tokenizer import count_tokens


@dataclass
class SmartRAGResult:
    """Output of a single Smart RAG run."""

    answer: str
    retrieval_time_ms: float
    compressor_time_ms: float
    compressor_breakdown: dict[str, Any]
    llm_ttft_ms: float
    llm_total_gen_ms: float
    total_time_ms: float
    original_tokens: int
    compressed_tokens: int
    raw_chunks: list[Any]
    output_tokens: int = 0  # tokens the LLM actually produced in the answer
    succeeded: bool = True
    error: str | None = None


class SmartRAG:
    """Optimized RAG engine (plan §18)."""

    def __init__(
        self,
        db: VectorDatabase,
        llm: LLMClient,
        cfg: AppConfig | None = None,
        components: PipelineComponents | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.cfg = cfg or load_config()
        self.components = components

    def run(self, query: str) -> SmartRAGResult:
        t_start = time.perf_counter()

        # 1. Same retrieval as Normal RAG.
        raw_chunks = self.db.retrieve(query, top_k=self.cfg.retriever.top_k)
        t_retrieval = time.perf_counter()

        # 2. Run the compressor middleware.
        compressor_cfg: CompressorConfig = self.cfg.compressor
        compressed = compress_context(
            query=query,
            chunks=raw_chunks,
            cfg=compressor_cfg,
            components=self.components,
        )
        t_compressor = time.perf_counter()

        # 3. Same LLM call shape as Normal RAG.
        prompt = _build_prompt(query, compressed.compressed_text)
        t_llm_start = time.perf_counter()
        ttft_ms: float | None = None
        pieces: list[str] = []
        succeeded = True
        error_msg: str | None = None
        try:
            for piece in self.llm.generate_stream(prompt):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_llm_start) * 1000
                pieces.append(piece)
        except Exception as exc:
            # Quota-exhausted errors surface immediately; we still
            # record the compressor timing because that stage ran
            # successfully, but we mark succeeded=False and zero the
            # LLM-side timings so they cannot pollute benchmark
            # aggregates.
            succeeded = False
            error_msg = f"{type(exc).__name__}: {exc}"
            pieces = []
        t_end = time.perf_counter()

        return SmartRAGResult(
            answer="".join(pieces),
            retrieval_time_ms=(t_retrieval - t_start) * 1000,
            compressor_time_ms=(t_compressor - t_retrieval) * 1000,
            compressor_breakdown=compressed.metrics,
            llm_ttft_ms=ttft_ms or 0.0,
            llm_total_gen_ms=(t_end - t_llm_start) * 1000 if succeeded else 0.0,
            total_time_ms=(t_end - t_start) * 1000,
            original_tokens=compressed.metrics["original_token_count"],
            compressed_tokens=compressed.metrics["compressed_token_count"],
            raw_chunks=raw_chunks,
            output_tokens=count_tokens("".join(pieces)) if succeeded else 0,
            succeeded=succeeded,
            error=error_msg,
        )

    # Convenience for benchmark code: compare a Smart run to a Normal one.
    @staticmethod
    def compare(normal: NormalRAGResult, smart: SmartRAGResult) -> dict[str, float]:
        """Return the comparison summary described in plan §19.

        ``quota_failed`` is True iff either side ran into a
        quota-exhausted LLM response; in that case the latency
        numbers must NOT be used as a benchmark signal.
        """
        token_savings = (
            (normal.context_tokens - smart.compressed_tokens) / normal.context_tokens * 100.0
            if normal.context_tokens
            else 0.0
        )
        net_latency_savings_ms = normal.total_time_ms - smart.total_time_ms
        return {
            "normal_total_ms": normal.total_time_ms,
            "smart_total_ms": smart.total_time_ms,
            "net_latency_savings_ms": net_latency_savings_ms,
            "normal_context_tokens": normal.context_tokens,
            "smart_context_tokens": smart.compressed_tokens,
            "token_compression_pct": token_savings,
            "normal_llm_ttft_ms": normal.llm_ttft_ms,
            "smart_llm_ttft_ms": smart.llm_ttft_ms,
            "quota_failed": (not normal.succeeded) or (not smart.succeeded),
            "normal_succeeded": normal.succeeded,
            "smart_succeeded": smart.succeeded,
        }


def _build_prompt(query: str, compressed_context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question "
                "using only the provided context. If the context is "
                "insufficient, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{compressed_context}\n\nQuestion: {query}\nAnswer:",
        },
    ]