"""Multi-run diagnostic -- repeat the comparison N times to characterize variance.

Pure diagnostic, no production changes.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from backend.config.config import load_config
    from backend.rag.database import VectorDatabase
    from backend.llm.gemini_client import GeminiLLMClient
    from backend.rag.normal_rag import NormalRAG
    from backend.compressor.pipeline import PipelineComponents
    from backend.rag.smart_rag import SmartRAG
    from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder

    cfg = load_config()
    llm = GeminiLLMClient(cfg.llm)

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
    components = PipelineComponents(embedder=embedder, cross_encoder=cross_encoder)

    question = (
        "Explain the global token budget invariant and how it is enforced."
    )

    normal_totals: list[float] = []
    smart_totals: list[float] = []
    normal_ttfts: list[float] = []
    smart_ttfts: list[float] = []

    for i in range(5):
        n = NormalRAG(db, llm, cfg).run(question)
        s = SmartRAG(db, llm, cfg, components=components).run(question)
        normal_totals.append(n.total_time_ms)
        smart_totals.append(s.total_time_ms)
        normal_ttfts.append(n.llm_ttft_ms)
        smart_ttfts.append(s.llm_ttft_ms)
        print(
            f"run {i}: normal total={n.total_time_ms/1000:6.2f}s ttft={n.llm_ttft_ms/1000:6.2f}s  "
            f"smart total={s.total_time_ms/1000:6.2f}s ttft={s.llm_ttft_ms/1000:6.2f}s"
        )

    print()
    print(f"normal total_ms : min={min(normal_totals):.0f} max={max(normal_totals):.0f} mean={statistics.mean(normal_totals):.0f} stdev={statistics.pstdev(normal_totals):.0f}")
    print(f"smart  total_ms : min={min(smart_totals):.0f} max={max(smart_totals):.0f} mean={statistics.mean(smart_totals):.0f} stdev={statistics.pstdev(smart_totals):.0f}")
    print(f"normal ttft_ms  : min={min(normal_ttfts):.0f} max={max(normal_ttfts):.0f} mean={statistics.mean(normal_ttfts):.0f} stdev={statistics.pstdev(normal_ttfts):.0f}")
    print(f"smart  ttft_ms  : min={min(smart_ttfts):.0f} max={max(smart_ttfts):.0f} mean={statistics.mean(smart_ttfts):.0f} stdev={statistics.pstdev(smart_ttfts):.0f}")


if __name__ == "__main__":
    main()
