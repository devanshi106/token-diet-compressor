from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from backend.rag.interfaces import CrossEncoder, Embedder

logger = logging.getLogger(__name__)

# Ensure HF progress bars stay suppressed even when this module is imported
# outside the Streamlit entry point (e.g. by scripts or tests).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Single global cache for initialized models to prevent reloading weights.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}

# Global cache for text embeddings to prevent re-encoding identical text strings.
_EMBEDDING_CACHE: dict[str, dict[str, list[float]]] = {}


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
            # HF/tqdm writes loading progress to stderr via stdlib's
            # `print(..., file=sys.stderr)` + manual `flush()`. When this
            # process's stderr is a non-standard stream (e.g. Streamlit's
            # tornado log wrapper on Windows), the flush raises
            # OSError: [Errno 22] Invalid argument. Temporarily redirect
            # stderr to devnull for the construction only.
            import contextlib
            import sys as _sys
            import tempfile
            devnull = open(os.devnull, "w", encoding="utf-8")
            try:
                with contextlib.redirect_stderr(devnull):
                    _MODEL_CACHE[cache_key] = SentenceTransformer(
                        self.model_name, device=self.device
                    )
            finally:
                devnull.close()
            # Restore real stderr for any further logging.
            _sys.stderr = _sys.__stderr__
        return _MODEL_CACHE[cache_key]

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        model_cache = _EMBEDDING_CACHE.setdefault(self.model_name, {})
        
        # Identify unique missing texts
        unique_texts = list(set(text_list))
        missing_texts = [t for t in unique_texts if t not in model_cache]

        if missing_texts:
            # Run batched encoding with L2 normalization enabled.
            embeddings = self.model.encode(
                missing_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Convert numpy floats/arrays to serializable Python floats and cache them
            for text, emb in zip(missing_texts, embeddings):
                model_cache[text] = [float(val) for val in emb]

        return [model_cache[t] for t in text_list]

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
        model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
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
            # Same stderr workaround as the embedder -- tqdm's flush()
            # crashes inside Streamlit's non-standard stderr on Windows.
            import contextlib
            import sys as _sys
            devnull = open(os.devnull, "w", encoding="utf-8")
            try:
                with contextlib.redirect_stderr(devnull):
                    _MODEL_CACHE[cache_key] = STCrossEncoder(
                        self.model_name, device=self.device
                    )
            finally:
                devnull.close()
            _sys.stderr = _sys.__stderr__
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
