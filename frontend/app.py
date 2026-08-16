"""Streamlit dashboard for the Token-Diet Dynamic Context Compressor.

Plan §21 metric priority + §28 interactive UX:

1. Original Retrieved-Context Tokens (Normal RAG)
2. Compressed-Context Tokens (Smart RAG)
3. Compression Percentage
4. Compressor Latency
5. LLM TTFT (both)
6. End-to-End Latency (both)
7. Net Latency Savings
8. Estimated Input-Token / API Cost
9. Answer Correctness (keyword-based factual check)
10. Answer Cosine Similarity (secondary diagnostic)

Run with::

    set GEMINI_API_KEY=...        (Windows) /  export GEMINI_API_KEY=...
    streamlit run app.py
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any

# Make project root importable so Streamlit can find tests.* helpers
# regardless of where the user launches `streamlit run` from.
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force UTF-8 on stdout/stderr so that any non-ASCII character (e.g. an
# ellipsis `…` accidentally pasted into the API key field, or a Gemini
# response with em-dashes) doesn't crash Streamlit's logging path on
# Windows consoles that default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# Heavy deps are imported inside helpers so this module is cheap to
# `python -c "import app"` (used by the Phase 4 tests).


# ---------------------------------------------------------------------------
# Helpers (streamlit-free where possible)
# ---------------------------------------------------------------------------


FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "datasets" / "demo_company" / "documents"
)


def _find_fixtures() -> list[tuple[str, str]]:
    """Find markdown fixtures under ``datasets/demo/documents``.

    Accepts both legacy ``SAMPLE FIXTURE`` and current
    ``SYNTHETIC DEVELOPMENT / EVALUATION DATA`` markers.
    """

    data_dir = FIXTURES_DIR
    if not data_dir.exists():
        return []
    return [
        (p.stem, p.read_text(encoding="utf-8"))
        for p in sorted(data_dir.rglob("*.md"))
        if (
            "SAMPLE FIXTURE" in p.read_text(encoding="utf-8")
            or "SYNTHETIC DEVELOPMENT" in p.read_text(encoding="utf-8")
        )
    ]


def _load_fixtures() -> list[tuple[str, str]]:
    return _find_fixtures()
GIBBERISH_CORRECTNESS = "expected keyword match (default: empty)"


def _estimate_cost(  # very rough USD estimate for input tokens
    input_tokens: int,
    *,
    price_per_million_input: float = 0.30,  # Gemini 2.5 Flash default tier
) -> float:
    return (input_tokens / 1_000_000) * price_per_million_input


def _keyword_correctness(answer: str, required: tuple[str, ...]) -> bool:
    a = answer.lower()
    return all(kw.lower() in a for kw in required) if required else True


def _show_quota_error(exc) -> None:
    """Render a clear 'Gemini quota exhausted' panel."""
    import streamlit as st

    retry_after = getattr(exc, "retry_after_seconds", None)
    wait_hint = (
        f"Try again in about **{retry_after:.0f} s** (server hint)."
        if retry_after
        else "Try again later -- the free-tier quota resets daily."
    )
    st.error(
        "### 🚫 Gemini quota exhausted\n\n"
        "The Gemini API returned **HTTP 429 RESOURCE_EXHAUSTED**. "
        "Free-tier per-day request cap reached.\n\n"
        f"{wait_hint}\n\n"
        "**What you can do:**\n"
        "- Wait for the daily quota to reset.\n"
        "- Upgrade to a paid Gemini API key (no per-day cap).\n"
        "- Switch the default model to one with a higher free-tier "
        "allowance (e.g. `gemini-2.5-flash-lite`).\n\n"
        "_Latency measurements are not recorded for this run._"
    )


def _show_pipeline_failure(normal, smart, llm=None) -> None:
    import streamlit as st

    bits: list[str] = []
    if not normal.succeeded:
        bits.append(f"**Normal RAG**: {normal.error}")
    if not smart.succeeded:
        bits.append(f"**Smart RAG**: {smart.error}")
    st.error(
        "### ⚠️ LLM call failed\n\n"
        + "\n\n".join(bits)
        + "\n\n_Latency measurements are not recorded for this run._"
    )

    has_model_err = any("not found" in str(b).lower() or "no longer available" in str(b).lower() or "not supported" in str(b).lower() for b in bits)
    if has_model_err and llm is not None:
        try:
            if hasattr(llm, "_client") and llm._client is not None:
                models = list(llm._client.models.list())
                st.info("### 🔍 Available models for your API key:")
                st.write([m.name.replace("models/", "") for m in models])
        except Exception as e:
            st.warning(f"Could not retrieve available models list: {e}")


def _build_database(cfg: Any = None):
    """Lazy import + DB build."""
    from backend.rag.database import VectorDatabase
    from backend.embeddings.local_models import SentenceTransformersEmbedder
    from backend.config.config import load_config

    cfg = cfg or load_config()
    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    model_name = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"

    db = VectorDatabase(
        embedder=SentenceTransformersEmbedder(model_name=model_name, device=device)
    )
    for doc_id, text in _load_fixtures():
        db.add_document(doc_id, text)
    return db


def _build_components(cfg: Any = None):
    from backend.compressor.pipeline import PipelineComponents
    from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder
    from backend.config.config import load_config

    cfg = cfg or load_config()
    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    emb_model = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-MiniLM-L-6-v2"

    return PipelineComponents(
        embedder=SentenceTransformersEmbedder(model_name=emb_model, device=device),
        cross_encoder=SentenceTransformersCrossEncoder(model_name=ce_model, device=device),
    )


def _build_llm(api_key: str = ""):
    from backend.config.config import load_config
    from backend.llm.gemini_client import GeminiLLMClient

    cfg = load_config()
    if api_key:
        cfg = _override_api_key_env(cfg, api_key)
    return GeminiLLMClient(cfg.llm), cfg


def _override_api_key_env(cfg, api_key: str):
    """Return a copy of AppConfig with the key set in os.environ."""
    import os

    os.environ[cfg.llm.api_key_env] = api_key
    # Build a fresh client (it re-reads env).
    from backend.llm.gemini_client import GeminiLLMClient

    return cfg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Token-Diet",
        page_icon="🪙",
        layout="wide",
    )
    st.title("🪙 Token-Diet Dynamic Context Compressor")
    st.caption(
        "Side-by-side comparison of Normal RAG and Smart RAG against the "
        "Token-Diet compressor middleware (plan §21)."
    )

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            help="Stored only in the runtime env var; not persisted.",
        )
        if api_key:
            # Common paste mistake: an ellipsis `…` (U+2026) inserted where
            # copy-paste truncated the middle of the key. Reject it loudly
            # instead of letting Gemini return a 400/429 and burying the
            # root cause under a UnicodeEncodeError on Windows.
            stripped = api_key.strip()
            if "\u2026" in stripped or len(stripped) < 20:
                st.error(
                    "That doesn't look like a valid Gemini API key. "
                    "Did you accidentally paste an ellipsis `…` where the "
                    "middle of the key got truncated? Re-copy the full key "
                    "from the Google AI Studio dashboard."
                )
                api_key = ""
            else:
                import os

                os.environ["GEMINI_API_KEY"] = stripped
        top_k = st.slider("Top-K retrieved chunks", 1, 20, 5)
        budget = st.slider("Global token budget", 100, 2000, 800, step=50)
        required_keywords = st.text_input(
            "Required keywords (comma-separated, optional)",
            "",
        ).strip()
        keywords = tuple(k.strip() for k in required_keywords.split(",") if k.strip()) if required_keywords else ()
        ref_ans_input = st.text_area(
            "Reference/Expected answer (optional)",
            "",
            help="If empty, it auto-detects if the query is in the evaluation query set.",
        )

    query = st.text_input(
        "Ask a question",
        placeholder="e.g. How does Token-Diet reduce LLM token usage?",
    )

    run = st.button("Run comparison", type="primary", disabled=not query)

    if not run or not query:
        # Idle state: show a quick "what this dashboard shows" pane.
        st.info(
            "**Dashboard metrics (plan §21)**: Original tokens → "
            "Compressed tokens → Compression % → Compressor latency → "
            "LLM TTFT → End-to-end latency → Net savings → Cost → "
            "Correctness → Semantic similarity."
        )
        return

    # Execute the comparison. We import here to avoid pulling
    # streamlit-time deps if the user only opens the dashboard.
    from backend.config.config import load_config
    from backend.evaluation.evaluation import EvalQuery
    from backend.rag.normal_rag import NormalRAG
    from backend.rag.smart_rag import SmartRAG

    cfg = load_config()
    # Apply UI overrides via env vars (the loader picks them up).
    import os

    os.environ["TOKEN_DIET_TOP_K"] = str(top_k)
    os.environ["TOKEN_DIET_BUDGET"] = str(budget)
    cfg = load_config()

    if not api_key and not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Paste one in the sidebar or export it first.")
        return

    llm, _ = _build_llm(api_key)
    db = _build_database(cfg)
    components = _build_components(cfg)

    # Apply env-var overrides a second time, then run.
    from backend.llm.gemini_client import LLMQuotaExhaustedError

    progress = st.progress(0.25, text="Normal RAG...")
    try:
        normal = NormalRAG(db, llm, cfg).run(query)
    except LLMQuotaExhaustedError as exc:
        progress.empty()
        _show_quota_error(exc)
        return
    progress.progress(0.60, text="Smart RAG...")
    try:
        smart = SmartRAG(db, llm, cfg, components=components).run(query)
    except LLMQuotaExhaustedError as exc:
        progress.empty()
        _show_quota_error(exc)
        return
    progress.progress(1.0, text="Done.")

    # If either side failed but did NOT raise (e.g. a non-quota LLM
    # error that the engines caught), surface that too instead of
    # pretending the latency numbers are real.
    if not normal.succeeded or not smart.succeeded:
        _show_pipeline_failure(normal, smart, llm=llm)
        return

    cmp = SmartRAG.compare(normal, smart)

    # --------------------------------------------------------------- Metrics
    st.subheader("📊 Dashboard Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original tokens", normal.context_tokens)
    col2.metric("Compressed tokens", smart.compressed_tokens)
    col3.metric(
        "Compression %",
        f"{cmp['token_compression_pct']:.1f}%",
        delta=f"{cmp['token_compression_pct']:.1f}% vs baseline",
        delta_color="inverse",
    )

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Normal RAG latency",
        f"{normal.total_time_ms / 1000:.2f} s",
        help="Retrieval + LLM TTFT.",
    )
    col5.metric(
        "Smart RAG latency",
        f"{smart.total_time_ms / 1000:.2f} s",
        delta=f"{cmp['net_latency_savings_ms'] / 1000:+.2f} s vs normal",
    )
    col6.metric(
        "Compressor overhead",
        f"{smart.compressor_time_ms:.1f} ms",
        help="Token-Diet middleware stage timings.",
    )

    col7, col8, col9 = st.columns(3)
    col7.metric(
        "Normal LLM TTFT",
        f"{normal.llm_ttft_ms / 1000:.2f} s",
    )
    col8.metric(
        "Smart LLM TTFT",
        f"{smart.llm_ttft_ms / 1000:.2f} s",
        delta=f"{(normal.llm_ttft_ms - smart.llm_ttft_ms) / 1000:+.2f} s vs normal",
        delta_color="inverse",
    )
    col9.metric(
        "Net input cost savings",
        f"${_estimate_cost(normal.context_tokens) - _estimate_cost(smart.compressed_tokens):.6f}",
        delta=f"normal cost ${_estimate_cost(normal.context_tokens):.6f}",
        delta_color="off",
    )

    # ------------------------------------------------------------ Stage breakdown
    st.subheader("⚙️ Compressor Stage Breakdown")
    breakdown = smart.compressor_breakdown
    rows: list[dict[str, Any]] = [
        {"stage": "unit_formation", "ms": breakdown.get("unit_formation_ms", 0.0)},
        {"stage": "fast_filter", "ms": breakdown.get("fast_filter_ms", 0.0)},
        {"stage": "rerank", "ms": breakdown.get("rerank_ms", 0.0)},
        {"stage": "selection", "ms": breakdown.get("selection_ms", 0.0)},
        {"stage": "packing", "ms": breakdown.get("pack_ms", 0.0)},
    ]
    if any(r["ms"] > 0 for r in rows):
        st.bar_chart({r["stage"]: r["ms"] for r in rows})

    # --------------------------------------------------------------- Answers
    st.subheader("💬 Side-by-Side Answers")
    a, b = st.columns(2)
    with a:
        st.markdown("**Normal RAG (no compression)**")
        st.caption(f"context tokens: {normal.context_tokens}")
        st.write(normal.answer or "(no answer)")
    with b:
        st.markdown("**Smart RAG (Token-Diet compressed)**")
        st.caption(
            f"context tokens: {smart.compressed_tokens} / orig {smart.original_tokens}"
        )
        st.write(smart.answer or "(no answer)")

    # -------------------------------------------------------------- Correctness
    if keywords:
        n_ok = _keyword_correctness(normal.answer, keywords)
        s_ok = _keyword_correctness(smart.answer, keywords)
        st.subheader("✅ Correctness (plan §20)")
        st.write(
            {
                "Normal RAG": "PASS" if n_ok else "FAIL",
                "Smart RAG": "PASS" if s_ok else "FAIL",
                "Required keywords": ", ".join(keywords),
            }
        )

    # ------------------------------------------------------------- Semantic Similarity
    from datasets.demo.queries.evaluation_queries import REFERENCE_ANSWERS
    ref_ans = ref_ans_input.strip() if ref_ans_input.strip() else REFERENCE_ANSWERS.get(query.strip(), "")
    if ref_ans:
        st.subheader("🎯 Answer Cosine Similarity (PRD metric)")
        try:
            texts = [ref_ans]
            indices = []
            if normal.answer:
                texts.append(normal.answer)
                indices.append("normal")
            if smart.answer:
                texts.append(smart.answer)
                indices.append("smart")
            
            embs = components.embedder.encode(texts)
            ref_emb = embs[0]
            
            normal_sim = 0.0
            smart_sim = 0.0
            curr_idx = 1
            if "normal" in indices:
                normal_sim = sum(x * y for x, y in zip(ref_emb, embs[curr_idx]))
                curr_idx += 1
            if "smart" in indices:
                smart_sim = sum(x * y for x, y in zip(ref_emb, embs[curr_idx]))
            
            delta = smart_sim - normal_sim
            
            col_sim1, col_sim2 = st.columns(2)
            col_sim1.metric(
                label="Normal Answer Cosine Similarity",
                value=f"{normal_sim:.4f}",
            )
            col_sim2.metric(
                label="Smart Answer Cosine Similarity",
                value=f"{smart_sim:.4f}",
                delta=f"{delta:+.4f} vs Normal",
            )
            st.caption(f"**Reference/Expected Answer used for comparison:** {ref_ans}")
        except Exception as e:
            st.error(f"Could not compute cosine similarity: {e}")

    # ------------------------------------------------------------ Diagnostics
    with st.expander("Raw compressor metrics"):
        st.json(smart.compressor_breakdown)


if __name__ == "__main__":
    main()
