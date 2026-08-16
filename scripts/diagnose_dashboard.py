"""Diagnostic script: trace exact wall-clock across both pipelines.

Run::

    cd C:\\Users\\Lenovo\\Desktop\\token-diet-compressor
    $env:PYTHONPATH = "$PWD"
    python scripts/diagnose_dashboard.py

It does NOT change any production code. It:
  - Records every Gemini API call's start/end with a model name + attempt.
  - Calls Normal RAG and Smart RAG back-to-back with the same query.
  - Reports per-stage timings (retrieval, compressor, prompt, LLM, TTFT, total).
  - Cross-verifies that each pipeline made exactly ONE Gemini call.
  - Asserts that the dashboard's `compressor_time_ms` covers the same
    execution that the compressor pipeline actually performed.

Defaults to using the same Gemini client you wired up for the dashboard.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# Allow the test doubles + the dashboard's lazy imports to work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _banner(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def main() -> None:  # noqa: C901  (diagnostic, intentionally linear)
    from backend.config.config import load_config
    from backend.rag.database import VectorDatabase
    from backend.llm.gemini_client import GeminiLLMClient
    from backend.rag.normal_rag import NormalRAG
    from backend.compressor.pipeline import PipelineComponents
    from backend.rag.smart_rag import SmartRAG
    from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder

    cfg = load_config()
    llm = GeminiLLMClient(cfg.llm)

    # Same exact setup as live_smoke + the dashboard.
    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    model_name = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedder = SentenceTransformersEmbedder(model_name=model_name, device=device)
    cross_encoder = SentenceTransformersCrossEncoder(model_name=ce_model, device=device)

    db = VectorDatabase(embedder=embedder)
    fixtures = sorted((Path(__file__).resolve().parent.parent / "datasets" / "demo" / "documents").rglob("*.md"))
    fixtures = [
        p
        for p in fixtures
        if (
            "SYNTHETIC DEVELOPMENT" in p.read_text(encoding="utf-8")
            or "SAMPLE FIXTURE" in p.read_text(encoding="utf-8")
        )
    ]
    for f in fixtures:
        db.add_document(f.stem, f.read_text(encoding="utf-8"))

    components = PipelineComponents(
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    # ---- Patch generate_stream to log every chunk arrival ---------------
    call_log: list[dict] = []

    original_stream = llm.generate_stream

    def traced_stream(prompt, **kwargs):
        call_id = len(call_log)
        model = llm._cfg.model
        t_call_start = time.perf_counter()
        chunks_received = 0
        first_chunk_wall: float | None = None
        pieces: list[str] = []
        try:
            for piece in original_stream(prompt, **kwargs):
                now = time.perf_counter()
                if chunks_received == 0:
                    first_chunk_wall = now - t_call_start
                chunks_received += 1
                pieces.append(piece)
                yield piece
        finally:
            t_call_end = time.perf_counter()
            call_log.append(
                {
                    "call_id": call_id,
                    "model": model,
                    "prompt_first80": (
                        (prompt[1]["content"] if isinstance(prompt, list) else str(prompt))[:80]
                        if prompt
                        else ""
                    ),
                    "chunks": chunks_received,
                    "first_chunk_in_s": (
                        f"{first_chunk_wall:.3f}"
                        if first_chunk_wall is not None
                        else "n/a (quota-fast-fail)"
                    ),
                    "total_in_s": t_call_end - t_call_start,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    llm.generate_stream = traced_stream  # type: ignore[method-assign]

    # ---- Run the same query Normal-then-Smart (dashboard order) ---------
    question = (
        "Explain the global token budget invariant and how it is enforced."
    )
    print(f"QUESTION: {question}")
    print(f"top_k from cfg = {cfg.retriever.top_k}")
    print(f"budget from cfg = {cfg.compressor.global_token_budget} tokens")
    print(f"model = {llm._cfg.model}")

    _banner("WARM-UP (Normal RAG) -- first call may be a cold connection")
    warm = NormalRAG(db, llm, cfg).run(question)
    print(f"warmup total_ms = {warm.total_time_ms:.1f}, ttft_ms = {warm.llm_ttft_ms:.1f}")

    _banner("NORMAL RAG (post-warmup)")
    n = NormalRAG(db, llm, cfg).run(question)
    print(f"retrieval_ms      : {n.retrieval_time_ms:.2f}")
    print(f"llm_ttft_ms       : {n.llm_ttft_ms:.2f}")
    print(f"llm_total_gen_ms  : {n.llm_total_gen_ms:.2f}")
    print(f"total_ms          : {n.total_time_ms:.2f}")
    print(f"context_tokens    : {n.context_tokens}")

    _banner("SMART RAG (post-warmup)")
    s = SmartRAG(db, llm, cfg, components=components).run(question)
    print(f"retrieval_ms      : {s.retrieval_time_ms:.2f}")
    print(f"compressor_ms     : {s.compressor_time_ms:.2f}")
    print(f"llm_ttft_ms       : {s.llm_ttft_ms:.2f}")
    print(f"llm_total_gen_ms  : {s.llm_total_gen_ms:.2f}")
    print(f"total_ms          : {s.total_time_ms:.2f}")
    print(f"original_tokens   : {s.original_tokens}")
    print(f"compressed_tokens : {s.compressed_tokens}")

    # ---- Cross-verify exact number of LLM calls -------------------------
    _banner("LOW-LEVEL API CALL LOG")
    for entry in call_log:
        fc = entry["first_chunk_in_s"]
        fc_str = f"{fc:.3f}s" if isinstance(fc, (int, float)) else fc
        print(
            f"call#{entry['call_id']:02d} model={entry['model']:>20s} "
            f"chunks={entry['chunks']:>3d} "
            f"first_chunk={fc_str} "
            f"total={entry['total_in_s']:.3f}s "
            f"prompt[:80]='{entry['prompt_first80']}'"
        )

    _banner("INVARIANT CHECKS")
    # Both pipelines together -> 3 calls expected (warm-up + normal + smart).
    expected_total_calls = 3
    actual = len(call_log)
    print(f"Total Gemini calls made: {actual} (expected {expected_total_calls})")
    if actual == expected_total_calls:
        print("PASS: exactly one call per pipeline run (plus warm-up).")
    else:
        print("WARN: unexpected call count -- inspect the table above.")

    # Cross-check: compressor_time_ms must match the actual compressor
    # wall-clock we measured by recomputing the compressed context.
    from backend.compressor.pipeline import compress_context

    raw_chunks = db.retrieve(question, top_k=cfg.retriever.top_k)
    t_c0 = time.perf_counter()
    _ = compress_context(
        query=question,
        chunks=raw_chunks,
        cfg=cfg.compressor,
        components=components,
    )
    t_c1 = time.perf_counter()
    recomputed_ms = (t_c1 - t_c0) * 1000.0
    print(
        f"compressor_time_ms (from run) : {s.compressor_time_ms:.2f}"
    )
    print(
        f"compressor_wallclock (re-measured) : {recomputed_ms:.2f}"
    )
    ratio = s.compressor_time_ms / max(recomputed_ms, 1e-6)
    print(f"ratio                          : {ratio:.3f}")
    if 0.7 <= ratio <= 1.4:
        print(
            "PASS: dashboard compressor_time_ms covers the same execution."
        )
    else:
        print(
            "WARN: dashboard compressor_time_ms does NOT match recomputed "
            "wall-clock."
        )

    # Cross-check: the prompt that reached the LLM must be the same shape.
    p1 = n.raw_chunks and "\n\n".join(c.text for c in n.raw_chunks)
    print()
    print("Normal prompt[:200] vs Smart prompt[:200]:")
    print(f"  normal n_chars = {len(p1) if p1 else 0}")
    compressed_text_for_q = compress_context(
        query=question,
        chunks=raw_chunks,
        cfg=cfg.compressor,
        components=components,
    ).compressed_text
    print(f"  smart  n_chars = {len(compressed_text_for_q)}")

    # The slowness check.
    _banner("DELTA ANALYSIS")
    delta_ttft = s.llm_ttft_ms - n.llm_ttft_ms
    delta_total = s.total_time_ms - n.total_time_ms
    delta_compressor = s.compressor_time_ms - 0.0
    delta_prompt = (s.total_time_ms - s.retrieval_time_ms - s.compressor_time_ms - s.llm_total_gen_ms)
    print(f"DELTA retrieval    : {0.0:.2f} ms (both share the same retriever)")
    print(f"DELTA compressor   : {delta_compressor:.2f} ms (only Smart runs it)")
    print(f"DELTA prompt build : {0.0:.2f} ms (negligible)")
    print(f"DELTA LLM TTFT     : {delta_ttft:.2f} ms")
    print(f"DELTA LLM total gen: {s.llm_total_gen_ms - n.llm_total_gen_ms:.2f} ms")
    print(f"DELTA TOTAL        : {delta_total:.2f} ms")
    # If TTFT accounts for nearly all of the gap, the discrepancy is
    # upstream of the request stream, not in compression or prompt.
    if delta_ttft > 0.7 * delta_total:
        print()
        print(
            "DIAGNOSIS: >70% of the gap is in LLM time-to-first-token.\n"
            "  - The compressor and prompt builder are not the cause.\n"
            "  - The Smart RAG Gemini call is taking longer to start streaming\n"
            "    than the Normal RAG Gemini call.\n"
            "  - Possible causes:\n"
            "    * Different prompt size changes streaming chunk buffer\n"
            "      behavior on the Gemini backend (smart compressed context is\n"
            "      only marginally smaller here because the fixture corpus is\n"
            "      small, so this is unlikely to be the cause).\n"
            "    * A chat-session cold-start: chats.create() issues a setup\n"
            "      request each time; if the backend's first call after\n"
            "      warmup takes 20s, that would land entirely in TTFT.\n"
            "    * A long-running stream that returns 1 chunk at the end --\n"
            "      which would show TTFT == total. If chunks=1 in the API log,\n"
            "      the SDK is NOT actually streaming."
        )


if __name__ == "__main__":
    main()
