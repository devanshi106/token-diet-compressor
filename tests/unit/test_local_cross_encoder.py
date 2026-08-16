from __future__ import annotations

import pytest
from backend.embeddings.local_models import SentenceTransformersCrossEncoder, _MODEL_CACHE


def test_real_cross_encoder_construction_and_caching() -> None:
    # 1. Construction and Device
    encoder = SentenceTransformersCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )
    assert encoder.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert encoder.device == "cpu"

    # Clear cache to verify caching
    _MODEL_CACHE.clear()

    encoder1 = SentenceTransformersCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )
    encoder2 = SentenceTransformersCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )

    model1 = encoder1.model
    model2 = encoder2.model

    # Verification: Model reuse/caching
    assert model1 is model2
    assert len(_MODEL_CACHE) == 1


def test_real_cross_encoder_predict_and_ordering() -> None:
    encoder = SentenceTransformersCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )

    query = "What is FAISS?"
    pairs = [
        (query, "FAISS is a vector search library developed by Meta AI Research."),
        (query, "Dogs are mammals that have been domesticated for thousands of years."),
        (query, "FAISS facilitates fast nearest-neighbor similarity searches in dense spaces."),
    ]

    # Verification: Batched inference
    scores = encoder.predict(pairs, batch_size=2)
    assert len(scores) == 3

    # Verification: Score assignment and preservation of order
    for s in scores:
        assert isinstance(s, float)

    # First and third passages should be more relevant than the second one.
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
