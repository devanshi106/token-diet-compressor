"""Deterministic test doubles for the compressor pipeline.

These live in :mod:`tests._pipeline_fakes` so the Phase 2 test files
can share them without duplicating fake classes.
"""

from __future__ import annotations

import hashlib
import math

from backend.rag.interfaces import CrossEncoder, Embedder, SentenceSplitter
from backend.rag.models import RetrievedChunk


class HashEmbedder(Embedder):
    """Deterministic 64-dim bag-of-words style embedder.

    Two pieces of text that share a word will produce vectors with a
    positive cosine -- strong enough for the fast-filter test to behave
    sensibly without needing sentence-transformers.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self._dim
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self._dim] += 1.0
            # Normalize.
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    @property
    def dim(self) -> int:
        return self._dim


class WordOverlapCrossEncoder(CrossEncoder):
    """Toy Cross-Encoder: scores the lexical overlap between (q, passage)."""

    def predict(self, pairs, batch_size: int = 32) -> list[float]:
        out: list[float] = []
        for q, p in pairs:
            qw = set(q.lower().split())
            pw = set(p.lower().split())
            if not qw or not pw:
                out.append(0.0)
                continue
            jaccard = len(qw & pw) / len(qw | pw)
            out.append(float(jaccard))
        return out


class FixedSentenceSplitter(SentenceSplitter):
    """Splits on '.' and '!' and '?' -- predictable for tests."""

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        out: list[str] = []
        buf = ""
        for ch in text:
            buf += ch
            if ch in ".!?":
                if buf.strip():
                    out.append(buf.strip())
                buf = ""
        tail = buf.strip()
        if tail:
            out.append(tail)
        return out


def make_chunk(text: str, *, doc_id: str = "doc1", chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(text=text, doc_id=doc_id, chunk_id=chunk_id)