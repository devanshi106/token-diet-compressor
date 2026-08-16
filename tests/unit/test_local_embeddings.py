from __future__ import annotations

import pytest
from backend.embeddings.local_models import SentenceTransformersEmbedder, _MODEL_CACHE
from backend.rag.models import cosine_similarity


def test_real_embedder_construction_and_dim() -> None:
    # 1. Verification: Construction
    embedder = SentenceTransformersEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.device == "cpu"

    # 2. Verification: Output dimension is 384
    assert embedder.dim == 384


def test_real_embedder_encoding_and_compatibility() -> None:
    embedder = SentenceTransformersEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    
    # 3. Verification: Query/document embeddings have compatible dimensions
    query = "What is FAISS?"
    doc = "FAISS is a library for efficient similarity search."
    
    q_emb = embedder.encode([query])
    doc_emb = embedder.encode([doc])
    
    assert len(q_emb) == 1
    assert len(doc_emb) == 1
    assert len(q_emb[0]) == 384
    assert len(doc_emb[0]) == 384
    
    # 4. Verification: Cosine similarity works
    sim = cosine_similarity(q_emb[0], doc_emb[0])
    assert -1.0 <= sim <= 1.0


def test_real_embedder_batching() -> None:
    embedder = SentenceTransformersEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    
    # 5. Verification: Batch embedding works
    texts = [
        "First candidate sentence.",
        "Second sentence to encode.",
        "Third distinct sentence payload.",
    ]
    embs = embedder.encode(texts)
    assert len(embs) == 3
    for emb in embs:
        assert len(emb) == 384


def test_real_embedder_caching() -> None:
    # Clear cache first to ensure test isolation
    _MODEL_CACHE.clear()
    
    embedder1 = SentenceTransformersEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    embedder2 = SentenceTransformersEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    
    model1 = embedder1.model
    model2 = embedder2.model
    
    # 6. Verification: Model is not reloaded for every request (cached in memory)
    assert model1 is model2
    assert len(_MODEL_CACHE) == 1
