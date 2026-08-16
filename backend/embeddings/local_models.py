from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from backend.rag.interfaces import CrossEncoder, Embedder

logger = logging.getLogger(__name__)

# Single global cache for initialized models to prevent reloading weights.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class SentenceTransformersEmbedder(Embedder):
    """Concrete Embedder utilizing sentence-transformers for local inference.

    Satisfies the requirement to load the model once and reuse it across requests
    using a global cache, and supports cpu/cuda device targeting.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._dim: Optional[int] = None

    @property
    def model(self) -> Any:
        cache_key = (self.model_name, self.device)
        if cache_key not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer model {self.model_name} on {self.device}...")
            _MODEL_CACHE[cache_key] = SentenceTransformer(self.model_name, device=self.device)
        return _MODEL_CACHE[cache_key]

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        # Run batched encoding with L2 normalization enabled.
        embeddings = self.model.encode(
            text_list,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Convert numpy floats/arrays to serializable Python floats
        return [list(map(float, emb)) for emb in embeddings]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.get_sentence_embedding_dimension() or 384)
        return self._dim


class SentenceTransformersCrossEncoder(CrossEncoder):
    """Concrete CrossEncoder utilizing sentence-transformers for local reranking.

    Satisfies loading once, system device targeting, globally caching,
    and batched predictions.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

    @property
    def model(self) -> Any:
        cache_key = (self.model_name, self.device)
        if cache_key not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder as STCrossEncoder

            logger.info(f"Loading CrossEncoder model {self.model_name} on {self.device}...")
            _MODEL_CACHE[cache_key] = STCrossEncoder(self.model_name, device=self.device)
        return _MODEL_CACHE[cache_key]

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]
