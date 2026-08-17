"""Live end-to-end test using the real Gemini API key from .env.

Skips if GEMINI_API_KEY is not set (so CI can run without secrets).

Verifies the actual NormalRAG + SmartRAG paths complete and produce
non-empty answers for a known-answer query.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env so GEMINI_API_KEY is available to the SDK even when
# pytest is launched without an interactive shell.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from backend.config.config import load_config  # noqa: E402
from backend.embeddings.local_models import (  # noqa: E402
    SentenceTransformersCrossEncoder,
    SentenceTransformersEmbedder,
)
from backend.compressor.pipeline import (  # noqa: E402
    PipelineComponents,
    RegexSentenceSplitter,
)
from backend.rag.database import VectorDatabase  # noqa: E402
from backend.rag.normal_rag import NormalRAG  # noqa: E402
from backend.rag.smart_rag import SmartRAG  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "datasets" / "demo_company" / "documents"


pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set in environment",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def components(cfg):
    return PipelineComponents(
        splitter=RegexSentenceSplitter(),
        embedder=SentenceTransformersEmbedder(
            model_name=cfg.retriever.embedding_model, device=cfg.system.device
        ),
        cross_encoder=SentenceTransformersCrossEncoder(
            model_name=cfg.compressor.cross_encoder_model, device=cfg.system.device
        ),
    )


@pytest.fixture(scope="module")
def db(components):
    db = VectorDatabase(embedder=components.embedder)
    docs = [
        (p.stem, p.read_text(encoding="utf-8"))
        for p in sorted(FIXTURES_DIR.rglob("*.md"))
        if "SAMPLE FIXTURE" in p.read_text(encoding="utf-8")
        or "SYNTHETIC DEVELOPMENT" in p.read_text(encoding="utf-8")
    ]
    for doc_id, text in docs:
        db.add_document(doc_id, text)
    return db


def test_normal_rag_returns_real_answer(cfg, db, components):
    """NormalRAG completes with a non-empty answer."""
    from backend.llm.gemini_client import GeminiLLMClient

    llm = GeminiLLMClient(cfg=cfg.llm)
    rag = NormalRAG(db=db, llm=llm, cfg=cfg)
    query = "What are the data retention and backup policies for NovaDB clusters?"

    t0 = time.perf_counter()
    result = rag.run(query)
    elapsed = time.perf_counter() - t0

    if not result.succeeded and "quota" in (result.error or "").lower():
        pytest.skip(f"Gemini quota exhausted: {result.error}")
    assert result.succeeded, f"NormalRAG failed: {result.error}"
    assert result.answer.strip(), "NormalRAG returned empty answer"
    assert result.output_tokens > 0
    print(
        f"\n[NORMAL] {elapsed:.1f}s | TTFT {result.llm_ttft_ms/1000:.1f}s | "
        f"ctx {result.context_tokens} tok | out {result.output_tokens} tok | "
        f"{len(result.answer)} chars"
    )


def test_smart_rag_returns_real_answer(cfg, db, components):
    """SmartRAG completes with a non-empty answer."""
    from backend.llm.gemini_client import GeminiLLMClient

    llm = GeminiLLMClient(cfg=cfg.llm)
    rag = SmartRAG(db=db, llm=llm, cfg=cfg, components=components)
    query = "What are the data retention and backup policies for NovaDB clusters?"

    t0 = time.perf_counter()
    result = rag.run(query)
    elapsed = time.perf_counter() - t0

    if not result.succeeded and "quota" in (result.error or "").lower():
        pytest.skip(f"Gemini quota exhausted: {result.error}")
    assert result.succeeded, f"SmartRAG failed: {result.error}"
    assert result.answer.strip(), "SmartRAG returned empty answer"
    assert result.output_tokens > 0
    # Compressed context should be smaller than raw context
    assert result.compressed_tokens <= result.original_tokens
    print(
        f"\n[SMART]  {elapsed:.1f}s | TTFT {result.llm_ttft_ms/1000:.1f}s | "
        f"ctx {result.compressed_tokens}/{result.original_tokens} tok | "
        f"out {result.output_tokens} tok | {len(result.answer)} chars"
    )


def test_smart_rag_compressor_within_budget(cfg, db, components):
    """SmartRAG compressor stays within the configured token budget."""
    from backend.llm.gemini_client import GeminiLLMClient

    llm = GeminiLLMClient(cfg=cfg.llm)
    rag = SmartRAG(db=db, llm=llm, cfg=cfg, components=components)
    query = "What are the data retention and backup policies for NovaDB clusters?"
    result = rag.run(query)
    if not result.succeeded and "quota" in (result.error or "").lower():
        pytest.skip(f"Gemini quota exhausted: {result.error}")
    assert result.succeeded
    # Allow small overshoot for sentence-boundary units
    assert result.compressed_tokens <= cfg.compressor.global_token_budget + 100, (
        f"Compressor exceeded budget: {result.compressed_tokens} > "
        f"{cfg.compressor.global_token_budget}"
    )