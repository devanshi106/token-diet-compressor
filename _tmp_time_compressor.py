"""Time the compressor with the new fast_filter_candidate_limit=10."""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from backend.config.config import load_config
from backend.embeddings.local_models import (
    SentenceTransformersCrossEncoder,
    SentenceTransformersEmbedder,
)
from backend.compressor.pipeline import (
    PipelineComponents,
    RegexSentenceSplitter,
    compress_context,
)
from backend.rag.database import VectorDatabase

FIXTURES = REPO / "datasets" / "demo_company" / "documents"

cfg = load_config()
print(f"fast_filter_candidate_limit = {cfg.compressor.fast_filter_candidate_limit}")

components = PipelineComponents(
    splitter=RegexSentenceSplitter(),
    embedder=SentenceTransformersEmbedder(
        model_name=cfg.retriever.embedding_model, device=cfg.system.device
    ),
    cross_encoder=SentenceTransformersCrossEncoder(
        model_name=cfg.compressor.cross_encoder_model, device=cfg.system.device
    ),
)
db = VectorDatabase(embedder=components.embedder)
for p in sorted(FIXTURES.rglob("*.md")):
    txt = p.read_text(encoding="utf-8")
    if "SAMPLE FIXTURE" in txt or "SYNTHETIC DEVELOPMENT" in txt:
        db.add_document(p.stem, txt)

queries = [
    "What are the data retention and backup policies for NovaDB clusters?",
    "How does NovaCompute handle scaling?",
    "What is the maximum message retention for NovaStream?",
]
for q in queries:
    chunks = db.retrieve(q, top_k=cfg.retriever.top_k)
    t0 = time.perf_counter()
    out = compress_context(q, chunks, cfg=cfg.compressor, components=components)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\nQ: {q}")
    print(f"  total: {elapsed:.0f}ms")
    for k in ("unit_formation_ms", "fast_filter_ms", "rerank_ms", "selection_ms", "pack_ms"):
        print(f"  {k:25s}: {out.metrics.get(k, 0):.0f}ms")
    print(f"  candidates after filter: {out.metrics.get('candidate_count_after_filter')}")
    print(f"  compression: {out.metrics.get('original_token_count')} -> "
          f"{out.metrics.get('compressed_token_count')} tokens")
