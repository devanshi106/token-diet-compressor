"""Normal RAG baseline (plan §17).

User Query -> Retriever -> Raw Top-K Chunks -> LLM -> Answer

No compression is applied. Timing is recorded with millisecond
precision so it can be compared against Smart RAG (plan §19).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from backend.config.config import AppConfig, load_config
from backend.rag.database import VectorDatabase
from backend.llm.gemini_client import LLMClient
from backend.embeddings.tokenizer import count_tokens


@dataclass
class NormalRAGResult:
    """Output of a single Normal RAG run."""

    answer: str
    retrieval_time_ms: float
    llm_ttft_ms: float
    llm_total_gen_ms: float
    total_time_ms: float
    context_tokens: int
    raw_chunks: list[Any]
    output_tokens: int = 0  # tokens the LLM actually produced in the answer
    succeeded: bool = True
    error: str | None = None
    llm_server_prompt_time_ms: float = 0.0
    llm_server_queue_time_ms: float = 0.0


class NormalRAG:
    """Baseline RAG engine (plan §17)."""

    def __init__(
        self,
        db: VectorDatabase,
        llm: LLMClient,
        cfg: AppConfig | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.cfg = cfg or load_config()

    def run(self, query: str) -> NormalRAGResult:
        t_start = time.perf_counter()

        # 1. Retrieve raw top-k paragraphs.
        raw_chunks = self.db.retrieve(query, top_k=self.cfg.retriever.top_k)
        t_retrieval = time.perf_counter()

        # 2. Build prompt.
        raw_context = "\n\n".join(c.text for c in raw_chunks)
        raw_tokens = count_tokens(raw_context)
        prompt = self._build_prompt(query, raw_context)

        # 3. Stream through the LLM, recording TTFT.
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
            # The LLM client has already translated 429 quota errors
            # into LLMQuotaExhaustedError. We record the failure
            # deterministically rather than letting bogus latency
            # (e.g. SDK retry sleeps) leak into benchmark output.
            succeeded = False
            error_msg = f"{type(exc).__name__}: {exc}"
            pieces = []
        t_end = time.perf_counter()

        return NormalRAGResult(
            answer="".join(pieces),
            retrieval_time_ms=(t_retrieval - t_start) * 1000,
            llm_ttft_ms=ttft_ms or 0.0,
            llm_total_gen_ms=(t_end - t_llm_start) * 1000 if succeeded else 0.0,
            total_time_ms=(t_end - t_start) * 1000,
            context_tokens=raw_tokens,
            raw_chunks=raw_chunks,
            output_tokens=count_tokens("".join(pieces)) if succeeded else 0,
            succeeded=succeeded,
            error=error_msg,
            llm_server_prompt_time_ms=getattr(self.llm, "last_prompt_time_ms", 0.0),
            llm_server_queue_time_ms=getattr(self.llm, "last_queue_time_ms", 0.0),
        )

    @staticmethod
    def _build_prompt(query: str, context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using only the context provided below. If the context "
                    "does not contain the answer, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
            },
        ]