"""Stage 2 -- Fast Relevance Filter (plan §4 Stage 2).

Reduces the candidate pool from ``N`` units (potentially hundreds) to
``M`` units (default 50) using a cheap blended score:

    final_score = bm25_weight * bm25_score + embedding_weight * cosine

The BM25 index is built exactly once per request (plan §4: "avoid
unnecessary repeated BM25 index construction within the same
request") and disposed of when the function returns.

Embedding generation is delegated to an :class:`Embedder` -- production
uses a sentence-transformers model, tests use a deterministic hash
embedder.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from backend.rag.interfaces import Embedder
from backend.rag.models import ContextUnit, ScoredCandidate


# ---------------------------------------------------------------------------
# BM25 (pure-Python, dependency-free)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class _BM25Index:
    docs: list[list[str]]
    doc_lens: list[int]
    df: Counter
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str) -> list[float]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.docs:
            return [0.0] * len(self.docs)

        scores = [0.0] * len(self.docs)
        n_docs = len(self.docs)
        for q in q_tokens:
            if q not in self.df:
                continue
            idf = math.log(1 + (n_docs - self.df[q] + 0.5) / (self.df[q] + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.count(q)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                scores[i] += idf * num / den
        return scores


def _build_bm25(corpus: Iterable[str]) -> _BM25Index:
    docs = [_tokenize(t) for t in corpus]
    doc_lens = [len(d) for d in docs]
    df = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    return _BM25Index(docs=docs, doc_lens=doc_lens, df=df, avgdl=avgdl)


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize ``values`` into [0, 1]. Returns 0.5 for all if constant."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fast_filter_candidates(
    query: str,
    units: list[ContextUnit],
    embedder: Embedder,
    *,
    candidate_limit: int = 50,
    bm25_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> list[ScoredCandidate]:
    """Reduce ``units`` to the top ``candidate_limit`` candidates.

    Each returned :class:`ScoredCandidate` carries ``lexical_score`` and
    ``embedding_score`` populated; ``rerank_score`` and ``final_score``
    remain 0.0 (filled in by Stage 3).

    Raises
    ------
    ValueError
        If weights are negative. Weights of zero are allowed (the
        corresponding signal is then ignored).
    """

    if not units:
        return []

    if bm25_weight < 0 or embedding_weight < 0:
        raise ValueError("bm25_weight and embedding_weight must be non-negative")

    # 1. Lexical scores over the unit corpus.
    corpus = [u.target_text for u in units]
    bm25 = _build_bm25(corpus)
    bm25_scores = bm25.score(query)
    bm25_norm = _normalize(bm25_scores)

    # Sort units based on BM25 scores and select top units
    scored_units_bm25 = list(zip(units, bm25_scores, bm25_norm))
    scored_units_bm25.sort(key=lambda item: item[1], reverse=True)
    top_scored_units = scored_units_bm25[: max(0, candidate_limit)]
    filtered_units = [item[0] for item in top_scored_units]

    # 2. Embedding scores (only for the filtered top candidates).
    embedder_input = [query] + [u.target_text for u in filtered_units]
    all_embeddings = embedder.encode(embedder_input)
    q_emb = all_embeddings[0]
    embeddings = all_embeddings[1:]
    for unit, emb in zip(filtered_units, embeddings):
        unit.embedding = emb

    emb_scores: list[float] = []
    for emb in embeddings:
        if not emb or not q_emb:
            emb_scores.append(0.0)
            continue
        dot = sum(a * b for a, b in zip(emb, q_emb))
        na = math.sqrt(sum(a * a for a in emb))
        nb = math.sqrt(sum(b * b for b in q_emb))
        emb_scores.append(dot / (na * nb) if na and nb else 0.0)
    emb_norm = _normalize(emb_scores)

    # 3. Blend and assemble candidates.
    candidates: list[ScoredCandidate] = []
    total_w = bm25_weight + embedding_weight
    if total_w <= 0:
        total_w = 1.0
        bm25_weight, embedding_weight = 1.0, 0.0

    for (unit, bm25_raw, bm25_n), emb in zip(top_scored_units, emb_norm):
        final = (bm25_weight * bm25_n + embedding_weight * emb) / total_w
        candidates.append(
            ScoredCandidate(
                unit=unit,
                lexical_score=bm25_n,
                embedding_score=emb,
                final_score=final,
            )
        )

    # 4. Sort by blended score (desc)
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates