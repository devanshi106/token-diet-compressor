"""In-memory document database with vector retrieval (plan §6 + §7).

Responsibilities:

* Ingest raw documents (strings or files on disk).
* Chunk them with a configurable splitter.
* Embed each chunk with a pluggable :class:`Embedder`.
* Store embeddings in an in-memory index.
* Serve top-k nearest-neighbor retrieval via cosine similarity.

Two backends are shipped:

* :class:`NumpyVectorIndex`  -- pure NumPy, fast, no GPU/FAISS dependency
  for tests and tiny corpora.
* :class:`FaissVectorIndex`   -- production; uses ``faiss-cpu`` (plan §6).
  Built lazily so test runs without FAISS installed still pass.

The chunker is intentionally simple: a sentence-aware splitter that
groups sentences into windows up to ``chunk_size`` characters. The plan
calls for pluggable chunking; we keep that seam explicit so future
improvements (semantic chunking, sliding windows, etc.) drop in
without changing the rest of the pipeline.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from backend.rag.interfaces import Embedder
from backend.rag.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """One chunk of text plus its source metadata."""

    text: str
    doc_id: str
    chunk_id: str


@dataclass
class _Doc:
    """Internal representation of an ingested document."""

    doc_id: str
    text: str


def chunk_text(
    text: str,
    doc_id: str,
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[TextChunk]:
    """Split ``text`` into sentence-aware windows of ~``chunk_size`` chars.

    The algorithm packs whole sentences into each chunk until adding the
    next sentence would exceed ``chunk_size``; on overflow it starts a
    new chunk and carries over the trailing ``overlap`` characters so
    context isn't lost at the boundary.
    """

    if not text or not text.strip():
        return []

    # Lightweight sentence splitter -- same regex as the pipeline but
    # exposed here as a free function so the database doesn't import
    # the pipeline module (which would form an import cycle).
    sentences = [s.strip() + "." for s in _split_into_sentences(text)]

    chunks: list[TextChunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(buf).strip()
        if body:
            chunks.append(
                TextChunk(
                    text=body,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                )
            )
        # Carry over overlap characters from the tail of ``buf``.
        if overlap > 0 and chunks:
            tail_text = " ".join(buf)
            if len(tail_text) > overlap:
                carry = tail_text[-overlap:]
                buf = [carry]
                buf_len = len(carry)
            else:
                buf = list(buf)
                buf_len = len(tail_text)
        else:
            buf = []
            buf_len = 0

    for sent in sentences:
        if buf_len + len(sent) > chunk_size and buf:
            flush()
        buf.append(sent)
        buf_len += len(sent)
    flush()

    # Reassign sequential chunk_ids based on final order.
    return [
        TextChunk(text=c.text, doc_id=c.doc_id, chunk_id=f"{doc_id}_chunk_{i}")
        for i, c in enumerate(chunks)
    ]


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter for the database chunker.

    Strips line breaks, splits on `.`/`!`/`?` followed by whitespace
    and a capital letter. Doesn't need to be clever -- it only feeds the
    chunker.
    """

    import re

    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Vector indices
# ---------------------------------------------------------------------------


class VectorIndex(ABC):
    """Cosine-similarity nearest-neighbor index."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]: ...


class NumpyVectorIndex(VectorIndex):
    """Brute-force cosine index, fine for small corpora and tests."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = embeddings / norms
        if self._embeddings is None:
            self._embeddings = normalized
        else:
            self._embeddings = np.vstack([self._embeddings, normalized])
        self._metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._embeddings is None or len(self._metadata) == 0:
            return []
        q = query / max(np.linalg.norm(query), 1e-12)
        scores = self._embeddings @ q
        k = min(top_k, len(self._metadata))
        # argpartition is O(n); full sort is O(k log k). k is tiny.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


class FaissVectorIndex(VectorIndex):
    """FAISS-backed index. Imported lazily so FAISS isn't required for tests."""

    def __init__(self) -> None:
        import faiss  # type: ignore[import-not-found]

        self._faiss = faiss
        self._index: object | None = None
        self._metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        x = embeddings.astype("float32")
        faiss = self._faiss
        if self._index is None:
            self._index = faiss.IndexFlatIP(x.shape[1])
        # L2-normalize so inner-product == cosine.
        faiss.normalize_L2(x)
        self._index.add(x)  # type: ignore[union-attr]
        self._metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._index is None:
            return []
        q = query.astype("float32").reshape(1, -1)
        self._faiss.normalize_L2(q)
        k = min(top_k, len(self._metadata))
        scores, ids = self._index.search(q, k)  # type: ignore[union-attr]
        out: list[tuple[int, float]] = []
        for i, s in zip(ids[0], scores[0]):
            if i < 0:
                continue
            out.append((int(i), float(s)))
        return out


# ---------------------------------------------------------------------------
# VectorDatabase -- the public ingestion + retrieval facade
# ---------------------------------------------------------------------------


@dataclass
class VectorDatabase:
    """In-memory document store with vector retrieval.

    Both Normal RAG and Smart RAG must use the *same* database instance
    -- this is the explicit system boundary from plan §2.
    """

    embedder: Embedder
    chunk_size: int = 800
    chunk_overlap: int = 100
    use_faiss: bool = False
    index: VectorIndex = field(init=False)
    _chunks: list[TextChunk] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.index = FaissVectorIndex() if self.use_faiss else NumpyVectorIndex()

    # -- ingestion -----------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        chunks = chunk_text(text, doc_id, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return
        self._chunks.extend(chunks)
        embeddings = np.array(self.embedder.encode([c.text for c in chunks]), dtype=np.float64)
        self.index.add(embeddings, [{"doc_id": c.doc_id, "chunk_id": c.chunk_id} for c in chunks])

    def add_documents(self, docs: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add_document(doc_id, text)

    def add_files(self, paths: Iterable[Path], *, encoding: str = "utf-8") -> None:
        for p in paths:
            self.add_document(p.stem, p.read_text(encoding=encoding))

    # -- retrieval -----------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or 5
        if not self._chunks:
            return []
        q_emb = np.array(self.embedder.encode([query])[0], dtype=np.float64)
        hits = self.index.search(q_emb, k)
        out: list[RetrievedChunk] = []
        for idx, score in hits:
            c = self._chunks[idx]
            out.append(RetrievedChunk(text=c.text, doc_id=c.doc_id, chunk_id=c.chunk_id, score=score))
        return out

    @property
    def size(self) -> int:
        return len(self._chunks)

    def __len__(self) -> int:
        return self.size