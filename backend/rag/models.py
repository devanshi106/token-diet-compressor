"""Data models used across the compressor pipeline.

The schemas here are taken verbatim from the implementation plan
(``data/plan7.md`` §8). Treat them as the contract between stages; do
not pass ad-hoc dicts around the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROSE = "prose"
STRUCTURED = "structured"

UNIT_TYPES = (PROSE, STRUCTURED)


# ---------------------------------------------------------------------------
# Core pipeline models
# ---------------------------------------------------------------------------


@dataclass
class ContextUnit:
    """A single scored unit of context.

    ``target_text`` is what gets emitted if the unit is selected.
    ``scoring_text`` is what the Cross-Encoder sees (target plus a
    small local context window) so that context-dependent references
    ("it", "the function above") can be resolved at scoring time.
    """

    unit_id: str                  # Format: {doc_id}_{chunk_id}_{unit_idx}
    doc_id: str
    chunk_id: str
    target_text: str
    scoring_text: str
    unit_type: str                # "prose" | "structured"
    position_idx: int             # 0-indexed position within its parent chunk
    parent_chunk_id: str
    token_count: int = 0          # Precomputed exact token count of target_text
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    """A candidate unit carrying lexical, embedding, and rerank scores."""

    unit: ContextUnit
    lexical_score: float = 0.0
    embedding_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


@dataclass
class CompressorOutput:
    """Result of running the full compression middleware."""

    compressed_text: str
    selected_units: list[ContextUnit]
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Retrieval / pipeline I/O helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by the retriever.

    Mirrors the dict shape used in the plan's pseudocode
    (``{"text": ..., "doc_id": ..., "chunk_id": ...}``) but stays typed
    so the rest of the code can pass concrete objects around.
    """

    text: str
    doc_id: str
    chunk_id: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tiny math helpers (kept here so tests don't need to import numpy)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` if either vector is missing/empty or the dimensions
    don't match -- the compressor treats those cases as "no signal"
    rather than raising.
    """

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)