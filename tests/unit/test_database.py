"""Tests for src.database -- chunking, NumPy index, FAISS index, and the
VectorDatabase facade used by both Normal RAG and Smart RAG.

``FaissVectorIndex`` is only tested when FAISS is installed (which it
is in this environment).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit._pipeline_fakes import HashEmbedder

from backend.rag.database import (
    FaissVectorIndex,
    NumpyVectorIndex,
    VectorDatabase,
    chunk_text,
)


def test_chunk_text_splits_long_input_into_windows() -> None:
    text = ("Sentence one. " * 50).strip()
    chunks = chunk_text(text, "d", chunk_size=200, overlap=20)
    # Each chunk should be <= chunk_size + some slack for the overlap
    # carry-over.
    assert 2 <= len(chunks) <= 4


def test_chunk_text_handles_empty_input() -> None:
    assert chunk_text("", "d") == []
    assert chunk_text("   \n  ", "d") == []


def test_chunk_text_assigns_sequential_chunk_ids() -> None:
    chunks = chunk_text("Sentence one. Sentence two. " * 30, "doc")
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"doc_chunk_{i}" for i in range(len(chunks))]


def test_numpy_index_returns_top_k() -> None:
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((10, 4)).astype(np.float64)
    # Query is identical to row 0.
    idx = NumpyVectorIndex()
    idx.add(embs, [{"i": i} for i in range(10)])
    hits = idx.search(embs[0], top_k=3)
    # Top hit must be row 0 (cosine == 1).
    assert hits[0][0] == 0
    assert abs(hits[0][1] - 1.0) < 1e-6


def test_faiss_index_returns_top_k() -> None:
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((10, 4)).astype(np.float32)
    idx = FaissVectorIndex()
    idx.add(embs, [{"i": i} for i in range(10)])
    hits = idx.search(embs[0], top_k=3)
    assert hits[0][0] == 0


def test_vector_database_add_and_retrieve() -> None:
    db = VectorDatabase(embedder=HashEmbedder())
    db.add_document("a", "The quick brown fox jumps over the lazy dog.")
    db.add_document("b", "FAISS enables fast similarity search over vectors.")
    db.add_document(
        "c", "Pizza dough is kneaded until smooth and elastic."
    )
    assert len(db) == 3

    hits = db.retrieve("vector similarity FAISS", top_k=2)
    assert len(hits) == 2
    # The "b" doc is clearly most relevant.
    assert hits[0].doc_id == "b"


def test_vector_database_handles_empty_corpus() -> None:
    db = VectorDatabase(embedder=HashEmbedder())
    assert db.retrieve("anything") == []


def test_vector_database_faiss_backend() -> None:
    pytest.importorskip("faiss")
    db = VectorDatabase(embedder=HashEmbedder(), use_faiss=True)
    db.add_document("d", "Some text about FAISS and vectors.")
    hits = db.retrieve("vectors", top_k=1)
    assert len(hits) == 1