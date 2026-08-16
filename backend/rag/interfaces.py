"""Abstract interfaces for the compressor pipeline.

Concrete implementations live in:

* :mod:`backend.compressor.pipeline.unit_formation`    -- ``SentenceSplitter``
* :mod:`backend.compressor.pipeline.fast_filter`       -- ``Embedder``
* :mod:`backend.compressor.pipeline.reranker`          -- ``CrossEncoder``
* :mod:`src.retriever` (existing)       -- ``Retriever``

Keeping them in one place makes the contract obvious and lets tests
provide lightweight fakes without dragging in PyTorch / FAISS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
from backend.rag.models import ContextUnit, RetrievedChunk


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


class SentenceSplitter(ABC):
    """Splits a chunk of text into sentences (or, for structured blocks,
    into logical sub-units).
    """

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Return a non-empty list of sentence strings.

        Implementations must not return the empty string. If the input
        is empty or whitespace-only, return an empty list.
        """


# ---------------------------------------------------------------------------
# Stage 2 / Stage 4 helpers
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """Dense vector embedder used by Stage 2 (fast filter) and Stage 4
    (redundancy removal).
    """

    @abstractmethod
    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """Return one embedding per input text, in the same order."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------


class CrossEncoder(ABC):
    """Scores (query, passage) pairs in a single forward pass."""

    @abstractmethod
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        """Return one relevance score per pair."""


# ---------------------------------------------------------------------------
# Retrieval (shared by Normal RAG and Smart RAG)
# ---------------------------------------------------------------------------


class Retriever(ABC):
    """Returns top-k chunks for a query.

    Smart RAG must use *exactly* the same retriever as Normal RAG -- this
    is the explicit system boundary from plan §2. The interface is
    deliberately small so both in-memory and FAISS-backed
    implementations satisfy it.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks ranked by relevance."""
