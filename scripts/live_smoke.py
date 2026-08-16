"""End-to-end smoke test against live Gemini.

Run with::
    cd token-diet-compressor
    python scripts/live_smoke.py
"""

from __future__ import annotations

from pathlib import Path

from backend.config.config import load_config
from backend.rag.database import VectorDatabase
from backend.llm.gemini_client import GeminiLLMClient
from backend.rag.normal_rag import NormalRAG
from backend.compressor.pipeline import PipelineComponents
from backend.rag.smart_rag import SmartRAG
from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder


def main() -> None:
    cfg = load_config()
    llm = GeminiLLMClient(cfg.llm)

    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    model_name = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    embedder = SentenceTransformersEmbedder(model_name=model_name, device=device)
    cross_encoder = SentenceTransformersCrossEncoder(model_name=ce_model, device=device)

    db = VectorDatabase(embedder=embedder)
    # Walk the repo's data/ tree for any markdown fixture; resilient to
    # data/fixtures/*.md vs data/documents/fixtures/*.md layouts.
    fixtures = sorted((Path(__file__).resolve().parent.parent / "datasets" / "demo_company" / "documents").rglob("*.md"))
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
    print(f"Loaded {len(db)} chunks from {len(fixtures)} fixture docs")

    components = PipelineComponents(
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    question = "Explain the global token budget invariant and how it is enforced."

    print("\n=== Normal RAG ===")
    normal = NormalRAG(db, llm, cfg).run(question)
    print(f"context tokens  : {normal.context_tokens}")
    print(f"retrieval_ms    : {normal.retrieval_time_ms:.1f}")
    print(f"llm_ttft_ms     : {normal.llm_ttft_ms:.1f}")
    print(f"llm_total_ms    : {normal.llm_total_gen_ms:.1f}")
    print(f"total_ms        : {normal.total_time_ms:.1f}")
    print(f"answer          : {normal.answer[:280]}{'...' if len(normal.answer) > 280 else ''}")

    print("\n=== Smart RAG ===")
    smart = SmartRAG(db, llm, cfg, components=components).run(question)
    print(f"context tokens  : {smart.compressed_tokens} (orig {smart.original_tokens})")
    compression_pct = (
        (1 - smart.compressed_tokens / smart.original_tokens) * 100
        if smart.original_tokens
        else 0.0
    )
    print(f"compression_%   : {compression_pct:.1f}")
    print(f"retrieval_ms    : {smart.retrieval_time_ms:.1f}")
    print(f"compressor_ms   : {smart.compressor_time_ms:.1f}")
    print(f"llm_ttft_ms     : {smart.llm_ttft_ms:.1f}")
    print(f"llm_total_ms    : {smart.llm_total_gen_ms:.1f}")
    print(f"total_ms        : {smart.total_time_ms:.1f}")
    print(f"answer          : {smart.answer[:280]}{'...' if len(smart.answer) > 280 else ''}")

    print("\n=== Comparison ===")
    cmp = SmartRAG.compare(normal, smart)
    for k, v in cmp.items():
        print(f"  {k:32s} = {v:.2f}")


if __name__ == "__main__":
    main()
