"""Stage 3 -- Batched Cross-Encoder Reranking (plan §4 Stage 3 + §12).

Each candidate unit is paired with the user's query using its
``scoring_text`` (the target sentence plus a ±1 sentence window) so the
model can resolve context-dependent references ("it", "the function
above") at scoring time -- this is the "context vacuum" mitigation from
plan §31.

Inference is done in batches via the abstract :class:`CrossEncoder`
predict method. Tests can plug in a deterministic fake; production uses
``cross-encoder/ms-marco-MiniLM-L-6-v2`` via sentence-transformers.

Output: ``candidates`` sorted by descending ``rerank_score``.
"""

from __future__ import annotations
from backend.rag.interfaces import CrossEncoder
from backend.rag.models import ScoredCandidate


def rerank_candidates(
    query: str,
    candidates: list[ScoredCandidate],
    cross_encoder: CrossEncoder,
    *,
    batch_size: int = 32,
) -> list[ScoredCandidate]:
    """Score each candidate's ``scoring_text`` against ``query``.

    Parameters
    ----------
    query:
        The user's question.
    candidates:
        Output of :func:`backend.compressor.pipeline.fast_filter.fast_filter_candidates`.
    cross_encoder:
        Any :class:`CrossEncoder` implementation.
    batch_size:
        Forward-pass batch size (default 32, matches the plan's
        ``cross_encoder_batch_size`` default).
    """

    if not candidates:
        return []

    pairs = [(query, c.unit.scoring_text) for c in candidates]
    scores = cross_encoder.predict(pairs, batch_size=batch_size)

    if len(scores) != len(candidates):
        raise ValueError(
            f"CrossEncoder returned {len(scores)} scores for {len(candidates)} pairs"
        )

    for cand, score in zip(candidates, scores):
        cand.rerank_score = float(score)

    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    return candidates