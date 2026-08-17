from __future__ import annotations

import pytest
from dataclasses import asdict
from backend.rag.interfaces import Embedder
from backend.rag.normal_rag import NormalRAG
from backend.rag.smart_rag import SmartRAG
from scripts.run_evaluation import _run_query, RowResult, _summarize
from datasets import EvalQuery


class MockDeterministicEmbedder(Embedder):
    """Embedder returning deterministic 2D vectors for testing metrics."""

    def __init__(self) -> None:
        self.vectors = {
            "ref": [1.0, 0.0],
            "norm": [0.6, 0.8],   # dot product with ref = 0.6
            "smart": [0.8, 0.6],  # dot product with ref = 0.8
        }
        self._dim = 2

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if "lives for 60 minutes" in t or "OAuth endpoint" in t:
                out.append(self.vectors["ref"])
            elif "normal response" in t:
                out.append(self.vectors["norm"])
            elif "smart response" in t:
                out.append(self.vectors["smart"])
            else:
                # Default fallback
                out.append([1.0, 0.0])
        return out

    @property
    def dim(self) -> int:
        return self._dim


class MockRAGEngine:
    def __init__(self, answer: str, succeeded: bool = True) -> None:
        self.answer = answer
        self.succeeded = succeeded
        self.context_tokens = 100
        self.compressed_tokens = 50
        self.original_tokens = 100


def test_cosine_similarity_metric_calculation() -> None:
    # 1. Setup deterministic inputs and fakes
    embedder = MockDeterministicEmbedder()
    
    q = EvalQuery(
        query="How long does an access token issued by CloudSync's OAuth endpoint live?",
        category="easy",
        difficulty="easy",
        required_keywords=("60 minutes",),
        source_doc="doc",
        _expected_answer="An access token lives for 60 minutes via the OAuth endpoint.",
    )
    
    # We create mock run returns carrying distinct answers
    class MockNormal(MockRAGEngine):
        def run(self, query):
            return self

    class MockSmart(MockRAGEngine):
        def run(self, query):
            return self

    normal_rag = MockNormal("normal response")
    smart_rag = MockSmart("smart response")

    # 2. Run query through eval execution helper
    res = _run_query(1, q, normal_rag, smart_rag, embedder)

    # 3. Verifications
    # Dot product [1.0, 0.0] . [0.6, 0.8] = 0.6
    assert abs(res.normal_answer_cosine_similarity - 0.6) < 1e-5
    # Dot product [1.0, 0.0] . [0.8, 0.6] = 0.8
    assert abs(res.smart_answer_cosine_similarity - 0.8) < 1e-5
    # Delta = 0.8 - 0.6 = +0.2
    assert abs(res.cosine_similarity_delta - 0.2) < 1e-5
    
    # Test aggregate summary
    summary = _summarize([res], include_quota_failures=False)
    assert "answer_similarity" in summary
    sim = summary["answer_similarity"]
    assert abs(sim["normal_mean"] - 0.6) < 1e-5
    assert abs(sim["smart_mean"] - 0.8) < 1e-5
    assert abs(sim["mean_delta"] - 0.2) < 1e-5


def test_evaluation_dataset_loads_without_error() -> None:
    from datasets import EVAL_QUERIES
    # At minimum the dataset should be non-empty.
    assert len(EVAL_QUERIES) >= 20, f"Expected at least 20 queries, got {len(EVAL_QUERIES)}"
