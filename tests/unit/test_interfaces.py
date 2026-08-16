"""Tests for Phase 1: abstract interfaces can be implemented by fakes."""

from __future__ import annotations

from backend.rag.interfaces import (
    CrossEncoder,
    Embedder,
    Retriever,
    SentenceSplitter,
)
from backend.rag.models import RetrievedChunk


class FakeSplitter(SentenceSplitter):
    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [s.strip() for s in text.split(".") if s.strip()]


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            # Deterministic bag-of-words style vector.
            vec = [0.0] * self._dim
            for i, ch in enumerate(t.lower()):
                vec[i % self._dim] += ord(ch) / 255.0
            out.append(vec)
        return out

    @property
    def dim(self) -> int:
        return self._dim


class FakeCrossEncoder(CrossEncoder):
    def predict(self, pairs, batch_size: int = 32) -> list[float]:
        out = []
        for q, p in pairs:
            q_words = set(q.lower().split())
            p_words = set(p.lower().split())
            out.append(float(len(q_words & p_words)))
        return out


class FakeRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int | None = None):
        limit = top_k if top_k is not None else len(self._chunks)
        return self._chunks[:limit]


def test_sentence_splitter_interface() -> None:
    s = FakeSplitter()
    assert s.split("a. b. c.") == ["a", "b", "c"]
    assert s.split("") == []
    assert s.split("   \n  ") == []


def test_embedder_interface_dim_and_shape() -> None:
    e = FakeEmbedder(dim=4)
    assert e.dim == 4
    vecs = e.encode(["alpha", "beta"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 4


def test_cross_encoder_predict_scores() -> None:
    ce = FakeCrossEncoder()
    scores = ce.predict([("foo bar", "bar baz"), ("hello", "world")])
    assert scores[0] == 1.0  # one shared word ("bar")
    assert scores[1] == 0.0


def test_retriever_interface_returns_chunks_in_order() -> None:
    chunks = [
        RetrievedChunk(text=f"chunk-{i}", doc_id="d", chunk_id=f"c{i}")
        for i in range(5)
    ]
    r = FakeRetriever(chunks)
    assert len(r.retrieve("anything", top_k=3)) == 3
    assert r.retrieve("anything")[0].text == "chunk-0"