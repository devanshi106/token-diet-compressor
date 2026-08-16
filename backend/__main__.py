"""CLI entry point: ``python -m backend``.

Loads the configured retriever, runs one end-to-end query against
NormalRAG + SmartRAG, and prints a small comparison table.

The CLI is intentionally lightweight -- it's a smoke harness used by
the Phase 1 acceptance criteria. It does NOT change the algorithm;
both pipelines call the same LLMClient instance via the configured
retriever.

Run with::

    set GEMINI_API_KEY=...  (Windows)
    export GEMINI_API_KEY=...  (POSIX)
    python -m backend
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _cli() -> int:
    """Smoke CLI: load config, build retriever, run both pipelines once."""
    from backend.config.config import load_config
    from backend.llm.gemini_client import FakeLLMClient, GeminiLLMClient
    from backend.rag.database import VectorDatabase
    from backend.rag.normal_rag import NormalRAG
    from backend.rag.smart_rag import SmartRAG
    from backend.compressor.pipeline import PipelineComponents
    from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder

    cfg = load_config()
    query = sys.argv[1] if len(sys.argv) > 1 else "What is Token-Diet?"

    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    model_name = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-MiniLM-L-6-v2"

    embedder = SentenceTransformersEmbedder(model_name=model_name, device=device)
    cross_encoder = SentenceTransformersCrossEncoder(model_name=ce_model, device=device)
    db = VectorDatabase(embedder=embedder)
    fixtures_dir = _PROJECT_ROOT / "datasets" / "demo" / "documents"
    if fixtures_dir.exists():
        for p in sorted(fixtures_dir.rglob("*.md")):
            if "SYNTHETIC DEVELOPMENT" in p.read_text(encoding="utf-8") or "SAMPLE FIXTURE" in p.read_text(encoding="utf-8"):
                db.add_document(p.stem, p.read_text(encoding="utf-8"))

    components = PipelineComponents(
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    api_key = os.environ.get(cfg.llm.api_key_env) or ""
    llm = GeminiLLMClient(cfg.llm, api_key=api_key) if api_key else FakeLLMClient()

    print(f"Query: {query}")
    print(f"DB chunks: {len(db)}")
    print(f"LLM: {type(llm).__name__}")

    normal = NormalRAG(db, llm, cfg).run(query)
    smart = SmartRAG(db, llm, cfg, components).run(query)

    print("\nNormal RAG")
    print(f"  succeeded: {normal.succeeded}")
    print(f"  answer   : {(normal.answer or '')[:120]!r}")
    print(f"  ctx_tok  : {normal.context_tokens}")

    print("\nSmart RAG")
    print(f"  succeeded: {smart.succeeded}")
    print(f"  answer   : {(smart.answer or '')[:120]!r}")
    print(f"  ctx_tok  : {smart.compressed_tokens}")

    return 0


raise SystemExit(_cli())
