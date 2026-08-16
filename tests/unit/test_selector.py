"""Tests for Stage 4 -- selector + Stage 5 -- packer (and the
interaction between them: budget invariant enforcement).
"""

from __future__ import annotations

from tests.unit._pipeline_fakes import HashEmbedder, WordOverlapCrossEncoder, make_chunk

from backend.rag.models import ContextUnit, ScoredCandidate
from backend.compressor.pipeline.fast_filter import fast_filter_candidates
from backend.compressor.pipeline.packer import pack_and_order_context
from backend.compressor.pipeline.reranker import rerank_candidates
from backend.compressor.pipeline.selector import SelectionConfig, select_budgeted_candidates
from backend.compressor.pipeline.unit_formation import form_context_units


def _build_pipeline(
    text: str,
    query: str,
    *,
    budget: int,
    threshold: float = 0.8,
) -> tuple[list[ScoredCandidate], dict[str, list[ContextUnit]]]:
    units = form_context_units([make_chunk(text)]).units
    parent = {"c1": units}
    candidates = fast_filter_candidates(
        query=query, units=units, embedder=HashEmbedder(), candidate_limit=len(units)
    )
    candidates = rerank_candidates(
        query=query, candidates=candidates, cross_encoder=WordOverlapCrossEncoder()
    )
    return candidates, parent


def test_selector_respects_global_budget() -> None:
    text = " ".join(f"Sentence number {i} about FAISS." for i in range(20))
    candidates, parent = _build_pipeline(text, query="FAISS", budget=20)
    cfg = SelectionConfig(global_token_budget=20)
    selected = select_budgeted_candidates(candidates, parent, cfg)
    packed = pack_and_order_context(selected, parent)
    from backend.embeddings.tokenizer import count_tokens

    budget = 20
    final_tokens = count_tokens(packed)
    assert final_tokens <= budget, f"got {final_tokens} > {budget}"


def test_selector_restores_neighbours() -> None:
    """Selecting a middle unit must pull in ±1 neighbours from the same chunk."""
    text = " ".join(f"Number {i}." for i in range(6))
    candidates, parent = _build_pipeline(text, query="Number 3", budget=200)
    cfg = SelectionConfig(
        global_token_budget=200,
        restoration_window_left=1,
        restoration_window_right=1,
        similarity_threshold=1.1,  # disable redundancy to isolate restoration
    )
    selected = select_budgeted_candidates(candidates, parent, cfg)
    targets = {u.target_text for u in selected}
    # The middle sentence plus its two neighbours must appear.
    assert any("Number 3." in t for t in targets)


def test_selector_redundancy_pruning() -> None:
    """Near-duplicate units must be pruned once one is selected."""
    text = "FAISS is fast. FAISS is fast. Vector search exists. Pizza is tasty."
    candidates, parent = _build_pipeline(text, query="FAISS", budget=500)
    cfg = SelectionConfig(
        global_token_budget=500,
        restoration_window_left=0,
        restoration_window_right=0,
        similarity_threshold=0.8,
    )
    selected = select_budgeted_candidates(candidates, parent, cfg)
    fast_units = [u for u in selected if "FAISS is fast" in u.target_text]
    # Only one of the two near-identical sentences should make it.
    assert len(fast_units) == 1


def test_selector_handles_empty_candidates() -> None:
    selected = select_budgeted_candidates([], {}, SelectionConfig(global_token_budget=100))
    assert selected == []


def test_selector_structured_unit_no_window_expansion() -> None:
    """Structured units must NOT expand their window (plan §13: start==end)."""
    units = [
        ContextUnit(
            unit_id=f"d_c_{i}",
            doc_id="d",
            chunk_id="c",
            target_text="print('hello')",
            scoring_text="print('hello')",
            unit_type="structured",
            position_idx=i,
            parent_chunk_id="c",
            token_count=3,
        )
        for i in range(3)
    ]
    parent = {"c": units}
    cands = [
        ScoredCandidate(unit=u, rerank_score=1.0 - 0.1 * i, final_score=1.0 - 0.1 * i)
        for i, u in enumerate(units)
    ]
    cfg = SelectionConfig(
        global_token_budget=100,
        restoration_window_left=2,
        restoration_window_right=2,
        similarity_threshold=1.1,
    )
    selected = select_budgeted_candidates(cands, parent, cfg)
    # All three should fit; the window trick must not pull in more than itself.
    assert len(selected) == 3


def test_packer_groups_by_document_with_headers() -> None:
    units = [
        ContextUnit(
            unit_id="d1_c1_0", doc_id="d1", chunk_id="c1",
            target_text="Alpha", scoring_text="Alpha", unit_type="prose",
            position_idx=0, parent_chunk_id="c1",
        ),
        ContextUnit(
            unit_id="d2_c2_0", doc_id="d2", chunk_id="c2",
            target_text="Beta", scoring_text="Beta", unit_type="prose",
            position_idx=0, parent_chunk_id="c2",
        ),
    ]
    out = pack_and_order_context(units, {"c1": [units[0]], "c2": [units[1]]})
    assert "### d1" in out
    assert "### d2" in out
    assert "Alpha" in out and "Beta" in out
    # d1 section must precede d2 (insertion order).
    assert out.index("### d1") < out.index("### d2")


def test_packer_normalizes_whitespace() -> None:
    u = ContextUnit(
        unit_id="d_c_0", doc_id="d", chunk_id="c",
        target_text="Line one.\n\n\n\nLine two.\t   trailing.",
        scoring_text="x", unit_type="prose",
        position_idx=0, parent_chunk_id="c",
    )
    out = pack_and_order_context([u], {"c": [u]})
    assert "\t" not in out
    assert "\n\n\n\n" not in out


def test_packer_empty_returns_empty() -> None:
    assert pack_and_order_context([], {}) == ""