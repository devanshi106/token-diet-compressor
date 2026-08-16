"""Retriever interface + a hard-coded implementation for testing.

The pipeline only knows about :class:`Retriever` -- a thin abstract base
that returns a list of text chunks for a given query. The current
implementation is :class:`HardcodedRetriever`, which returns a fixed
set of sample chunks regardless of the query. This lets us exercise the
end-to-end RAG flow without a dataset or vector store.

Later, a ``FaissRetriever`` (documents -> chunking -> embeddings -> FAISS
-> top-k lookup) will implement the same interface and be dropped in
without changes to the rest of the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single retrieved chunk of text."""

    text: str
    source: str = "hardcoded"
    score: float = 1.0


class Retriever(ABC):
    """Abstract retriever.

    Implementations must be pure with respect to the query: they return
    chunks relevant to ``query`` from some backing store.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return up to ``top_k`` chunks relevant to ``query``."""


class HardcodedRetriever(Retriever):
    """Returns a fixed set of sample chunks.

    Useful for testing the prompt + LLM wiring before a real retriever
    is in place. The chunks here cover a handful of plausible topics so
    the LLM has something concrete to cite.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = list(chunks) if chunks is not None else self._default_chunks()

    @staticmethod
    def _default_chunks() -> list[Chunk]:
        return [
            Chunk(
                text=(
                    "Retrieval-Augmented Generation (RAG) combines a retriever "
                    "with a generator. The retriever fetches relevant context "
                    "from a knowledge base, and the generator conditions its "
                    "answer on that context."
                ),
                source="sample:rag-overview",
            ),
            Chunk(
                text=(
                    "FAISS is a library for efficient similarity search over "
                    "dense vectors. It supports cosine and L2 distance and can "
                    "scale to billions of vectors."
                ),
                source="sample:faiss",
            ),
            Chunk(
                text=(
                    "Token dieting is the practice of minimizing the number of "
                    "tokens sent to an LLM without sacrificing answer quality. "
                    "Techniques include chunking, deduplication, re-ranking, "
                    "and prompt compression."
                ),
                source="sample:token-diet",
            ),
            Chunk(
                text=(
                    "OpenAI's chat completions API accepts a list of messages "
                    "with roles 'system', 'user', and 'assistant'. The "
                    "response is streamed or returned as a single JSON object."
                ),
                source="sample:openai-api",
            ),
            Chunk(
                text=(
                    "Python's __main__ idiom (`python -m package`) runs a "
                    "package's __main__.py. It's the canonical way to ship "
                    "a CLI entry point that lives inside a package."
                ),
                source="sample:python-main",
            ),
        ]

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        # Naive: return the first N chunks. Good enough to validate the
        # rest of the pipeline; a real retriever will score against the
        # query.
        del query  # unused for now
        limit = top_k if top_k is not None else len(self._chunks)
        return self._chunks[: max(0, limit)]


# ---------------------------------------------------------------------------
# Module-level helper so callers can do `chunks = retriever.retrieve(query)`
# without holding an instance, matching the prompt's API sketch.
# ---------------------------------------------------------------------------


_default_retriever: Retriever = HardcodedRetriever()


def retrieve(query: str, top_k: int | None = None) -> list[Chunk]:
    """Convenience wrapper around the default retriever."""
    return _default_retriever.retrieve(query, top_k=top_k)


def set_default_retriever(retriever: Retriever) -> None:
    """Swap the module-level default retriever."""
    global _default_retriever
    _default_retriever = retriever