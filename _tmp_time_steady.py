"""Time the compressor with warm models — first call vs steady-state."""
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

q = "What are the data retention and backup policies for NovaDB clusters?"
chunks = db.retrieve(q, top_k=cfg.retriever.top_k)

print(f"candidate_limit = {cfg.compressor.fast_filter_candidate_limit}")
for i in range(5):
    t0 = time.perf_counter()
    out = compress_context(q, chunks, cfg=cfg.compressor, components=components)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  call {i+1}: total {elapsed:.0f}ms "
          f"| fast_filter {out.metrics['fast_filter_ms']:.0f}ms "
          f"| rerank {out.metrics['rerank_ms']:.0f}ms "
          f"| tokens {out.metrics['compressed_token_count']}")