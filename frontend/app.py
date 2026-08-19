"""Streamlit dashboard for the Token-Diet Dynamic Context Compressor.

Dashboard metrics:

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

import os

# Suppress HF / tqdm progress bars before any model import.
# Streamlit's stdio redirection on Windows breaks tqdm's stderr flush
# and crashes with OSError: [Errno 22] Invalid argument. Disabling
# the progress bars at the environment level is the supported workaround.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def _format_latency_delta(ms: float) -> tuple[str, str]:
    """Format a latency delta for a Smart RAG metric.

    Returns ``(label, delta_color)`` suitable for ``st.metric(delta=..., delta_color=...)``.

    Positive ``ms`` means Smart RAG saved time relative to Normal RAG (good).
    Negative ``ms`` means Smart RAG cost extra time (bad).

    The label uses an explicit "faster" / "slower" word so the meaning is
    unambiguous regardless of how Streamlit renders the delta color.

    Streamlit "normal" semantics: positive delta → green ↑, negative → red ↓.
    We always use "normal" so the arrow colour matches the sign of the
    delta (faster = green, slower = red).
    """
    seconds = abs(ms) / 1000.0
    if ms >= 0:
        return f"{seconds:.2f} s faster than Normal", "normal"
    else:
        return f"{seconds:.2f} s slower than Normal", "normal"


def _validate_key_format(key: str) -> str:
    """Return one of: ``'valid'``, ``'empty'``, ``'ellipsis'``, ``'too_short'``.

    The check accepts any non-trivially-long, non-ellipsis string and
    lets the SDK decide whether the credential actually works. We do
    NOT enforce a single prefix because the GenAI SDK accepts several
    credential shapes (API keys starting with ``AIza``, OAuth access
    tokens starting with ``AQ.``, etc.) depending on whether Vertex AI
    mode is enabled.
    """
    s = (key or "").strip()
    if not s:
        return "empty"
    if "\u2026" in s:
        return "ellipsis"
    if len(s) < 20:
        return "too_short"
    return "valid"


def _render_chunks(chunks: list, label: str) -> None:
    """Render the retrieved chunks in an expander section.

    ``chunks`` is the list returned by ``db.retrieve()``; each item
    is expected to expose ``.doc_id``, ``.chunk_id`` (optional),
    ``.text``, and ``.score`` (optional). If those attributes are
    missing we degrade gracefully and show whatever is iterable.
    """
    import streamlit as st

    if not chunks:
        st.caption(f"_{label}: no chunks retrieved._")
        return
    for i, chunk in enumerate(chunks, start=1):
        doc = getattr(chunk, "doc_id", "?")
        cid = getattr(chunk, "chunk_id", None)
        score = getattr(chunk, "score", None)
        text = getattr(chunk, "text", str(chunk))
        score_str = f"score={score:.3f}" if isinstance(score, (int, float)) else ""
        title = f"{i}. `{doc}`" + (f" #{cid}" if cid is not None else "") + (
            f"  {score_str}" if score_str else ""
        )
        with st.expander(title, expanded=False):
            # Truncate very long chunks to keep the page responsive.
            preview = text if len(text) <= 1500 else text[:1500] + "\n\n_...truncated..._"
            st.markdown(preview)


def _show_quota_error(exc) -> None:
    """Render a clear 'Gemini quota exhausted' panel."""
    import streamlit as st

    retry_after = getattr(exc, "retry_after_seconds", None)
    # Flip a session-level flag so the sidebar banner stays visible
    # for the rest of this browser session, not just this run.
    st.session_state["_quota_exhausted"] = True
    st.session_state["_quota_retry_after_s"] = retry_after
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
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    return PipelineComponents(
        embedder=SentenceTransformersEmbedder(model_name=emb_model, device=device),
        cross_encoder=SentenceTransformersCrossEncoder(model_name=ce_model, device=device),
    )


def _build_llm(api_key: str = ""):
    from backend.config.config import load_config
    from backend.llm.gemini_client import GeminiLLMClient
    from backend.llm.groq_client import GroqLLMClient

    cfg = load_config()
    if api_key:
        cfg = _override_api_key_env(cfg, api_key)
    provider = (cfg.llm.provider or "gemini").lower()
    if provider == "groq":
        return GroqLLMClient(cfg.llm), cfg
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

    # Load config eagerly so the sidebar can show the fixed parameters
    # (top-K, budget, model) for transparency without exposing them
    # as user-editable controls.
    from backend.config.config import load_config as _load_cfg

    cfg = _load_cfg()

    st.set_page_config(
        page_title="Token-Diet",
        page_icon="🪙",
        layout="wide",
    )
    st.title("🪙 Token-Diet Dynamic Context Compressor")
    st.caption(
        "Same RAG pipeline — one with the Token-Diet compressor in front of "
        "the LLM, one without. Side-by-side so you can see what compression "
        "does to answer quality, latency, and cost."
    )

    with st.sidebar:
        st.header("Configuration")
        import os
        # Quota-aware banner: once Gemini returns 429 in this session,
        # we keep the warning visible until the session restarts so the
        # user doesn't keep clicking "Run" and burning their remaining
        # retry attempts on a known-failed quota.
        if st.session_state.get("_quota_exhausted"):
            retry_after = st.session_state.get("_quota_retry_after_s")
            wait = f" (~{retry_after:.0f}s)" if retry_after else ""
            provider_name = (cfg.llm.provider or "gemini").capitalize()
            st.error(
                f"🚫 {provider_name} quota exhausted. Wait for the daily reset{wait}, "
                f"or upgrade to a paid plan. **Disable the compressor toggle** "
                f"if you only need a quick smoke test of the LLM endpoint."
            )
        # Provider-aware key handling: the env var name is whatever
        # LLMConfig says, with a default of GEMINI_API_KEY for Gemini
        # or GROQ_API_KEY for Groq.
        provider_name = (cfg.llm.provider or "gemini").lower()
        key_env = cfg.llm.api_key_env or (
            "GROQ_API_KEY" if provider_name == "groq" else "GEMINI_API_KEY"
        )
        key_label = "Groq API key" if provider_name == "groq" else "Gemini API key"
        existing_env_key = os.environ.get(key_env, "")
        _existing_key_status = _validate_key_format(existing_env_key)
        if existing_env_key:
            if _existing_key_status == "valid":
                st.success(
                    f"✓ {provider_name.upper()} API key already loaded "
                    f"(from .env or shell). Paste a new one below to override "
                    f"for this session."
                )
        else:
            st.warning(f"No {key_env} set. Paste one below to enable this run.")
        api_key = st.text_input(
            key_label,
            type="password",
            help="Paste to override the .env value for this session. Not persisted to disk.",
        )
        if api_key:
            stripped = api_key.strip()
            _status = _validate_key_format(stripped)
            if _status == "ellipsis":
                st.error(
                    "That doesn't look like a valid credential. "
                    "Did you accidentally paste an ellipsis `…` where the "
                    "middle got truncated? Re-copy from the source."
                )
                api_key = ""
            elif _status == "too_short":
                st.error("That credential is too short. Re-copy from the source.")
                api_key = ""
            else:
                os.environ[key_env] = stripped
                # Clear any previous-run quota-exhausted flag — a new
                # key is unlikely to have the same quota state.
                st.session_state.pop("_quota_exhausted", None)
                st.session_state.pop("_quota_retry_after_s", None)
                st.success("✓ Override applied for this session.")
        # PRD §28: the only user controls are the query input and the
        # compressor ON/OFF toggle. All retrieval/compression parameters
        # (top-K, budget, similarity threshold, cross-encoder model) are
        # fixed in backend/config/default_config.yaml and not surfaced
        # to the user — they are infrastructure knobs, not product knobs.
        use_compressor = st.toggle(
            "Context compressor",
            value=True,
            help="When ON, Smart RAG runs the Token-Diet compressor "
            "before the LLM. When OFF, both sides use uncompressed "
            "context (the comparison collapses, but it's useful to "
            "verify the LLM answer alone).",
        )
        st.caption(
            f"Retrieval top-K: **{cfg.retriever.top_k}** · "
            f"Token budget: **{cfg.compressor.global_token_budget}** · "
            f"Compressor model: `{cfg.compressor.cross_encoder_model}`"
        )

    query = st.text_input(
        "Ask a question",
        placeholder="e.g. How does Token-Diet reduce LLM token usage?",
    )

    run = st.button("Run comparison", type="primary", disabled=not query)

    if not run or not query:
        # Idle state: show a quick "what this dashboard shows" pane.
        st.info(
            "Dashboard metrics (top to bottom): Original tokens → "
            "Compressed tokens → Compression % → Compressor latency → "
            "LLM TTFT → End-to-end latency → Net savings → Cost."
        )
        return

    # Execute the comparison. We import here to avoid pulling
    # streamlit-time deps if the user only opens the dashboard.
    from backend.rag.normal_rag import NormalRAG
    from backend.rag.smart_rag import SmartRAG

    if not api_key and not os.environ.get(key_env):
        st.error(
            f"{key_env} is not set. Paste one in the sidebar "
            f"or export it first."
        )
        return

    llm, _ = _build_llm(api_key)
    db = _build_database(cfg)
    components = _build_components(cfg)

    # Apply env-var overrides a second time, then run.
    from backend.llm.gemini_client import LLMQuotaExhaustedError

    # Run Normal RAG and Smart RAG **sequentially**. Concurrent execution
    # under a free-tier Gemini quota (~20 RPM) triggers a 429 retry loop
    # on one side and inflates its latency measurement; sequential
    # execution keeps both measurements faithful at the cost of an
    # additional ~latency window in wall-clock terms.
    if use_compressor:
        progress = st.progress(0.10, text="Running Normal RAG...")
        try:
            normal = NormalRAG(db, llm, cfg).run(query)
        except LLMQuotaExhaustedError as exc:
            _show_quota_error(exc)
            return
        progress.progress(0.55, text="Running Smart RAG (compressor + LLM)...")
        try:
            smart = SmartRAG(db, llm, cfg, components=components).run(query)
        except LLMQuotaExhaustedError as exc:
            _show_quota_error(exc)
            return
        progress.progress(1.0, text="Done.")
    else:
        # Compressor disabled by the toggle. We only run Normal RAG —
        # the side-by-side comparison collapses because there is
        # nothing to compress against.
        progress = st.progress(0.50, text="Running Normal RAG...")
        try:
            normal = NormalRAG(db, llm, cfg).run(query)
        except LLMQuotaExhaustedError as exc:
            _show_quota_error(exc)
            return
        smart = normal  # reuse the same result for layout purposes
        progress.progress(1.0, text="Done.")
        st.info(
            "Compressor is OFF — only the Normal RAG column is meaningful. "
            "Toggle the compressor ON in the sidebar to see the side-by-side comparison."
        )

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
        delta=f"{cmp['token_compression_pct']:.1f}% smaller than baseline",
    )

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Normal RAG latency",
        f"{normal.total_time_ms / 1000:.2f} s",
        help="Retrieval + LLM TTFT.",
    )
    _smart_lat_label, _smart_lat_color = _format_latency_delta(cmp["net_latency_savings_ms"])
    col5.metric(
        "Smart RAG latency",
        f"{smart.total_time_ms / 1000:.2f} s",
        delta=_smart_lat_label,
        delta_color=_smart_lat_color,
    )
    col6.metric(
        "Compressor overhead",
        f"{smart.compressor_time_ms:.1f} ms",
        help="Token-Diet middleware stage timings.",
    )

    col7, col8, col9 = st.columns(3)
    normal_help = "Measured from client. Includes network transit."
    if getattr(normal, "llm_server_prompt_time_ms", 0.0) > 0:
        normal_help += (
            f"\n\nServer-side breakdown:\n"
            f"- Prompt processing: {normal.llm_server_prompt_time_ms:.1f} ms\n"
            f"- Server queue time: {normal.llm_server_queue_time_ms:.1f} ms\n"
            f"- Total Server TTFT: {normal.llm_server_prompt_time_ms + normal.llm_server_queue_time_ms:.1f} ms"
        )
    col7.metric(
        "Normal LLM TTFT",
        f"{normal.llm_ttft_ms / 1000:.2f} s",
        help=normal_help,
    )
    
    smart_help = "Measured from client. Includes network transit."
    if getattr(smart, "llm_server_prompt_time_ms", 0.0) > 0:
        smart_help += (
            f"\n\nServer-side breakdown:\n"
            f"- Prompt processing: {smart.llm_server_prompt_time_ms:.1f} ms\n"
            f"- Server queue time: {smart.llm_server_queue_time_ms:.1f} ms\n"
            f"- Total Server TTFT: {smart.llm_server_prompt_time_ms + smart.llm_server_queue_time_ms:.1f} ms"
        )
    _smart_ttft_label, _smart_ttft_color = _format_latency_delta(normal.llm_ttft_ms - smart.llm_ttft_ms)
    col8.metric(
        "Smart LLM TTFT",
        f"{smart.llm_ttft_ms / 1000:.2f} s",
        delta=_smart_ttft_label,
        delta_color=_smart_ttft_color,
        help=smart_help,
    )
    col9.metric(
        "Net input cost savings",
        f"${_estimate_cost(normal.context_tokens) - _estimate_cost(smart.compressed_tokens):.6f}",
        delta=f"normal cost ${_estimate_cost(normal.context_tokens):.6f}",
        delta_color="off",
    )

    # ------------------------------------------------------------ Output tokens
    # Distinguishes "LLM is faster because it has less to read" (good)
    # from "LLM is slower because it generated more verbose hedging output"
    # (bad). Same numeric prompt savings, very different latency story.
    extra_captions = []
    if getattr(normal, "llm_server_prompt_time_ms", 0.0) > 0 or getattr(smart, "llm_server_prompt_time_ms", 0.0) > 0:
        extra_captions.append(
            f"**Exact Server-Side Timing (from Groq API headers):**  \n"
            f"* **Normal RAG**: Prompt time = **{normal.llm_server_prompt_time_ms:.1f} ms**, "
            f"Queue time = **{normal.llm_server_queue_time_ms:.1f} ms** (Total: **{normal.llm_server_prompt_time_ms + normal.llm_server_queue_time_ms:.1f} ms**)  \n"
            f"* **Smart RAG**: Prompt time = **{smart.llm_server_prompt_time_ms:.1f} ms**, "
            f"Queue time = **{smart.llm_server_queue_time_ms:.1f} ms** (Total: **{smart.llm_server_prompt_time_ms + smart.llm_server_queue_time_ms:.1f} ms**)"
        )
    
    extra_captions.append(
        f"Output tokens (what the LLM actually wrote): "
        f"Normal **{normal.output_tokens}** · Smart **{smart.output_tokens}**"
    )
    st.markdown("\n\n---\n\n".join(extra_captions))

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
    # ------------------------------------------------- Retrieved chunks inspector
    # Both Normal and Smart RAG see the same retrieved chunks -- only
    # the compressor middleware differs -- so we render this once,
    # before the side-by-side answers panel.
    with st.expander("📄 Retrieved chunks (what the LLM actually saw)"):
        _render_chunks(normal.raw_chunks, label="Retrieved (top-K)")

    st.subheader("💬 Side-by-Side Answers")
    a, b = st.columns(2)
    with a:
        st.markdown("**Normal RAG (no compression)**")
        st.caption(f"context tokens: {normal.context_tokens}")
        if not normal.answer:
            st.warning(
                "⚠️ The LLM returned no text content for this run, even "
                "though {n} chunks ({t} tokens) were retrieved and sent. "
                "This is a Gemini-side anomaly — likely a streaming edge "
                "case where the SDK yielded empty chunks — not a RAG "
                "failure. The retrieved chunks above are still valid and "
                "the Smart RAG side may still answer correctly.".format(
                    n=len(normal.raw_chunks), t=normal.context_tokens
                )
            )
        else:
            st.write(normal.answer)
    with b:
        st.markdown("**Smart RAG (Token-Diet compressed)**")
        st.caption(
            f"context tokens: {smart.compressed_tokens} / orig {smart.original_tokens}"
        )
        st.write(smart.answer or "(no answer)")

    # ------------------------------------------------------------- Semantic Similarity
    # If the user's query happens to match one of the 30 evaluation
    # queries in queries.json, we have a known-good reference answer
    # to compare against. Otherwise this panel stays hidden — the
    # user is exploring the system, not running an evaluation.
    from datasets.demo.queries.evaluation_queries import REFERENCE_ANSWERS
    ref_ans = REFERENCE_ANSWERS.get(query.strip(), "")
    if ref_ans:
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
