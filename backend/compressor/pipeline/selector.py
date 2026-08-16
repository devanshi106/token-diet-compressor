"""Stage 4 -- Smart Token-Budgeted Selection (plan §13 + §14 + §15 + §16).

Greedy selection that enforces the hard global token budget while:

* Restoring neighbouring sentences (default ±1) around each accepted
  target unit so pronouns and references remain resolvable.
* Pruning candidates that are near-duplicates of already-selected units
  (cosine similarity over embeddings).
* Tracking incremental cost via precomputed
  :attr:`ContextUnit.token_count` values plus a constant header estimate
  -- the tokenizer is **not** called inside the selection loop.

Hard-budget invariant
---------------------

After selection, the final packed context is exactly tokenized *once*.
If it exceeds :attr:`CompressorConfig.global_token_budget`, the lowest-
priority selected units are removed one at a time and the context is
rebuilt + retokenized. The function asserts ``final_tokens <= budget``
before returning; in practice the budget is always met because removing
units only reduces the token count. (Plan §16.)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from backend.rag.models import CompressorOutput, ContextUnit, PROSE, ScoredCandidate
from backend.embeddings.tokenizer import count_tokens

from .packer import pack_and_order_context


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class SelectionConfig:
    """Per-call selection config (subset of CompressorConfig)."""

    global_token_budget: int
    restoration_window_left: int = 1
    restoration_window_right: int = 1
    similarity_threshold: float = 0.8
    markdown_header_overhead_tokens: int = 15


def select_budgeted_candidates(
    candidates: list[ScoredCandidate],
    parent_chunks: dict[str, list[ContextUnit]],
    cfg: SelectionConfig,
) -> list[ContextUnit]:
    """Select units that fit inside ``cfg.global_token_budget``.

    Parameters
    ----------
    candidates:
        Output of :func:`backend.compressor.pipeline.reranker.rerank_candidates`,
        sorted by descending ``rerank_score``.
    parent_chunks:
        Per-chunk unit lists, keyed by ``chunk_id``.
    cfg:
        Budget + restoration + redundancy parameters.
    """

    selected_units: list[ContextUnit] = []
    selected_indices_by_chunk: dict[str, set[int]] = defaultdict(set)
    unit_priorities: dict[str, float] = {}
    estimated_total = 0

    for cand in candidates:
        unit = cand.unit
        chunk_id = unit.parent_chunk_id
        pos_idx = unit.position_idx

        # 1. Redundancy removal.
        if _is_redundant(unit, selected_units, cfg.similarity_threshold):
            continue

        # 2. Determine the proposed window.
        siblings = parent_chunks.get(chunk_id, [])
        if not siblings:
            # Orphan candidate -- still try to include it standalone.
            siblings = [unit]
        if unit.unit_type == PROSE:
            start_idx = max(0, pos_idx - cfg.restoration_window_left)
            end_idx = min(len(siblings) - 1, pos_idx + cfg.restoration_window_right)
        else:
            start_idx = pos_idx
            end_idx = pos_idx

        proposed_indices = set(range(start_idx, end_idx + 1))
        current_indices = selected_indices_by_chunk[chunk_id]
        new_indices = proposed_indices - current_indices
        if not new_indices:
            continue

        # 3. O(1) incremental cost estimate.
        net_new = _calculate_net_token_increase(
            sibling_units=siblings,
            current_indices=current_indices,
            new_indices=new_indices,
            include_header=not current_indices,
            header_overhead=cfg.markdown_header_overhead_tokens,
        )

        # 4. Greedy accept / reject.
        if estimated_total + net_new <= cfg.global_token_budget:
            selected_indices_by_chunk[chunk_id].update(proposed_indices)
            for idx in new_indices:
                u = siblings[idx]
                selected_units.append(u)
                unit_priorities[u.unit_id] = cand.rerank_score
            estimated_total += net_new

    # --- Fallback: nothing selected? pick the top candidate that fits ---
    if not selected_units and candidates:
        for cand in candidates:
            unit = cand.unit
            siblings = parent_chunks.get(unit.parent_chunk_id, [unit])
            cost = _calculate_net_token_increase(
                sibling_units=siblings,
                current_indices=set(),
                new_indices={unit.position_idx},
                include_header=True,
                header_overhead=cfg.markdown_header_overhead_tokens,
            )
            if cost <= cfg.global_token_budget:
                selected_units = [unit]
                unit_priorities[unit.unit_id] = cand.rerank_score
                break

    # --- Exact budget validation + priority-aware shrinking ---
    selected_units = _enforce_budget_invariant(
        selected_units,
        unit_priorities,
        cfg,
        parent_chunks,
    )

    return selected_units


# ---------------------------------------------------------------------------
# Cost / redundancy helpers (kept module-private; small + easily testable)
# ---------------------------------------------------------------------------


def _calculate_net_token_increase(
    *,
    sibling_units: list[ContextUnit],
    current_indices: set[int],
    new_indices: set[int],
    include_header: bool,
    header_overhead: int,
) -> int:
    """Sum precomputed token counts of ``new_indices`` (plan §14)."""

    net = sum(sibling_units[i].token_count for i in new_indices if 0 <= i < len(sibling_units))
    if include_header and not current_indices:
        net += header_overhead
    return net


def _is_redundant(
    candidate: ContextUnit,
    selected_units: list[ContextUnit],
    threshold: float,
) -> bool:
    """Return True if any selected unit is too similar to ``candidate``."""

    if not selected_units or candidate.embedding is None:
        return False
    for sel in selected_units:
        if sel.embedding is None:
            continue
        if _cosine(candidate.embedding, sel.embedding) > threshold:
            return True
    return False


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _enforce_budget_invariant(
    selected_units: list[ContextUnit],
    unit_priorities: dict[str, float],
    cfg: SelectionConfig,
    parent_chunks: dict[str, list[ContextUnit]],
) -> list[ContextUnit]:
    """Run exact tokenization on the final packed context.

    If it exceeds the budget, iteratively drop the lowest-priority unit
    and re-pack. The plan's invariant: ``final_tokens <= budget`` is
    asserted before returning (plan §13 step 7).
    """

    if not selected_units:
        return selected_units

    packed = pack_and_order_context(selected_units, parent_chunks)
    final_tokens = count_tokens(packed)

    # Priority-aware removal until budget is respected.
    safety_iters = len(selected_units) + 1
    while final_tokens > cfg.global_token_budget and selected_units and safety_iters > 0:
        safety_iters -= 1
        lowest = min(
            selected_units,
            key=lambda u: unit_priorities.get(u.unit_id, float("-inf")),
        )
        selected_units.remove(lowest)
        packed = pack_and_order_context(selected_units, parent_chunks)
        final_tokens = count_tokens(packed)

    assert final_tokens <= cfg.global_token_budget, (
        f"Hard token budget violated: {final_tokens} > {cfg.global_token_budget}"
    )
    return selected_units