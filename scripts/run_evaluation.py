"""Offline evaluation harness for the Token-Diet compressor.

Reads the synthetic documents under ``datasets/demo/documents/``, ingests
them through the real :class:`VectorDatabase`, and runs every query in
``datasets/demo/queries/evaluation_queries.py`` through BOTH pipelines:

    Normal RAG   -> same retriever, same LLM, no compressor
    Smart  RAG   -> same retriever, same LLM, +5-stage compressor

Reports for each query:

    query                              -- the question
    category / difficulty              -- taxonomy
    required_keywords                  -- ground truth
    normal_ctx_tokens                  -- Normal RAG context size (tokens)
    smart_ctx_tokens                   -- Smart RAG context size after compressor
    compression_pct                    -- (1 - smart/normal) * 100
    normal_latency_ms / smart_latency_ms
    normal_correct / smart_correct     -- keyword overlap with ground truth
    quota_failed                       -- True if any pipeline hit 429
    skipped                            -- True if run was skipped due to quota

Quota-failed runs are EXCLUDED from aggregates by default (the
``--include-quota-failures`` flag re-includes them; useful for
diagnostics). The script does NOT change the compressor algorithm,
thresholds, or ranking logic.

Run::

    cd token-diet-compressor
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --limit 5
    python scripts/run_evaluation.py --include-quota-failures
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Make the project root importable regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.config import load_config  # noqa: E402
from backend.rag.database import VectorDatabase  # noqa: E402
from backend.llm.gemini_client import (  # noqa: E402
    LLMClient,
    LLMQuotaExhaustedError,
)
from backend.rag.normal_rag import NormalRAG  # noqa: E402
from backend.rag.smart_rag import SmartRAG  # noqa: E402
from backend.compressor.pipeline import PipelineComponents  # noqa: E402
from datasets import EvalQuery, EVAL_QUERIES as _ALL_EVAL_QUERIES

# EVAL_QUERIES is imported directly from the datasets package, which
# loads it from datasets/demo_company/queries.json at import time.
# The 30-query enforcement was removed to avoid exhausting API keys
# during interactive dashboard use.
EVAL_QUERIES = _ALL_EVAL_QUERIES


# ---------------------------------------------------------------------------
# Fixtures loading
# ---------------------------------------------------------------------------

DOCS_DIR = ROOT / "datasets" / "demo_company" / "documents"


def _load_documents() -> list[tuple[str, str]]:
    """Read every markdown fixture under data/fixtures/documents/.

    The same walker used by the Streamlit dashboard picks up any
    file containing the synthetic-data marker.
    """
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing fixtures directory: {DOCS_DIR}")
    out: list[tuple[str, str]] = []
    for p in sorted(DOCS_DIR.rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        if "SYNTHETIC DEVELOPMENT" not in text and "SAMPLE FIXTURE" not in text:
            # Safety net -- skip any non-synthetic file dropped in by mistake.
            continue
        out.append((p.stem, text))
    if not out:
        raise FileNotFoundError(f"No synthetic documents under {DOCS_DIR}")
    return out


# ---------------------------------------------------------------------------
# Token-count helper (matches the compressor's accounting)
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Approximate token count.

    Uses tiktoken if available; falls back to whitespace proxy.
    Falls back to whitespace so the eval script never requires an
    extra install.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# LLM factory -- respects GEMINI_API_KEY but degrades gracefully
# ---------------------------------------------------------------------------


def _make_llm(cfg, *, force_offline: bool = False) -> LLMClient:
    """Build the Gemini LLM client, falling back to FakeLLMClient when no
    API key is set so the eval script remains runnable offline.

    The eval script ALWAYS runs both pipelines against the same client,
    per plan §2 "explicit system boundary".
    """
    import os

    from backend.llm.gemini_client import GeminiLLMClient

    if force_offline:
        from backend.llm.gemini_client import FakeLLMClient

        return FakeLLMClient()

    api_key = os.environ.get(cfg.llm.api_key_env) or ""
    if not api_key:
        from backend.llm.gemini_client import FakeLLMClient

        return FakeLLMClient()
    return GeminiLLMClient(cfg.llm, api_key=api_key, fail_fast_on_quota=True)


# ---------------------------------------------------------------------------
# Run a single query through both pipelines
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    query_id: int
    query: str
    category: str
    difficulty: str
    required_keywords: list[str]
    source_doc: str
    normal_ctx_tokens: int
    smart_ctx_tokens: int
    compression_pct: float
    normal_latency_ms: float
    smart_latency_ms: float
    normal_correct: bool
    smart_correct: bool
    normal_answer_excerpt: str
    smart_answer_excerpt: str
    normal_answer_cosine_similarity: float
    smart_answer_cosine_similarity: float
    cosine_similarity_delta: float
    quota_failed: bool
    skipped: bool


def _grade(answer: str, required: tuple[str, ...]) -> bool:
    a = answer.lower()
    return all(kw.lower() in a for kw in required)


def _excerpt(text: str, n: int = 120) -> str:
    s = " ".join(text.split())
    return s[:n] + ("..." if len(s) > n else "")


def _run_query(idx: int, q: EvalQuery, normal_rag: NormalRAG, smart_rag: SmartRAG, embedder: Any) -> RowResult:
    """Run a single query through both pipelines.

    Quota failures are caught and recorded as ``quota_failed=True``.
    The script never re-raises LLMQuotaExhaustedError to the caller.
    """
    normal: object | None = None
    smart: object | None = None
    quota_failed = False
    skipped = False
    normal_ctx_tokens = 0
    smart_ctx_tokens = 0
    normal_latency_ms = 0.0
    smart_latency_ms = 0.0
    normal_correct = False
    smart_correct = False
    normal_excerpt = ""
    smart_excerpt = ""

    try:
        t0 = time.perf_counter()
        normal = normal_rag.run(q.query)
        normal_latency_ms = (time.perf_counter() - t0) * 1000
    except LLMQuotaExhaustedError:
        quota_failed = True
        skipped = True

    try:
        t0 = time.perf_counter()
        smart = smart_rag.run(q.query)
        smart_latency_ms = (time.perf_counter() - t0) * 1000
    except LLMQuotaExhaustedError:
        quota_failed = True
        skipped = True

    if normal is not None:
        # NormalRAGResult carries context_tokens (count) + raw_chunks.
        # It catches its own LLMQuotaExhaustedError, so we have to
        # ALSO inspect succeeded=False and the error string to detect
        # quota failures that didn't propagate out of run().
        n_succeeded = bool(getattr(normal, "succeeded", True))
        n_error = getattr(normal, "error", None) or ""
        if not n_succeeded and ("Quota" in n_error or "429" in n_error):
            quota_failed = True
            skipped = True
        normal_ctx_tokens = int(getattr(normal, "context_tokens", 0))
        normal_correct = n_succeeded and _grade(
            getattr(normal, "answer", ""), q.required_keywords
        )
        normal_excerpt = _excerpt(getattr(normal, "answer", ""))

    if smart is not None:
        s_succeeded = bool(getattr(smart, "succeeded", True))
        s_error = getattr(smart, "error", None) or ""
        if not s_succeeded and ("Quota" in s_error or "429" in s_error):
            quota_failed = True
            skipped = True
        smart_ctx_tokens = int(getattr(smart, "compressed_tokens", 0))
        smart_correct = s_succeeded and _grade(
            getattr(smart, "answer", ""), q.required_keywords
        )
        smart_excerpt = _excerpt(getattr(smart, "answer", ""))

    if normal_ctx_tokens > 0:
        comp_pct = (1.0 - smart_ctx_tokens / normal_ctx_tokens) * 100
    else:
        comp_pct = 0.0

    normal_ans = getattr(normal, "answer", "") if normal is not None else ""
    smart_ans = getattr(smart, "answer", "") if smart is not None else ""
    ref_ans = q.reference_answer

    normal_sim = 0.0
    smart_sim = 0.0
    if ref_ans:
        texts = [ref_ans]
        indices = []
        if normal_ans:
            texts.append(normal_ans)
            indices.append("normal")
        if smart_ans:
            texts.append(smart_ans)
            indices.append("smart")

        embs = embedder.encode(texts)
        ref_emb = embs[0]
        
        curr_idx = 1
        if "normal" in indices:
            normal_sim = sum(x * y for x, y in zip(ref_emb, embs[curr_idx]))
            curr_idx += 1
        if "smart" in indices:
            smart_sim = sum(x * y for x, y in zip(ref_emb, embs[curr_idx]))

    delta = smart_sim - normal_sim

    return RowResult(
        query_id=idx,
        query=q.query,
        category=q.category,
        difficulty=q.difficulty,
        required_keywords=list(q.required_keywords),
        source_doc=q.source_doc,
        normal_ctx_tokens=normal_ctx_tokens,
        smart_ctx_tokens=smart_ctx_tokens,
        compression_pct=comp_pct,
        normal_latency_ms=normal_latency_ms,
        smart_latency_ms=smart_latency_ms,
        normal_correct=normal_correct,
        smart_correct=smart_correct,
        normal_answer_excerpt=normal_excerpt,
        smart_answer_excerpt=smart_excerpt,
        normal_answer_cosine_similarity=normal_sim,
        smart_answer_cosine_similarity=smart_sim,
        cosine_similarity_delta=delta,
        quota_failed=quota_failed,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _summarize(rows: list[RowResult], *, include_quota_failures: bool) -> dict:
    kept = [r for r in rows if (include_quota_failures or not r.quota_failed)]
    skipped = [r for r in rows if r.quota_failed]

    if not kept:
        return {
            "n_total": len(rows),
            "n_kept": 0,
            "n_skipped_quota": len(skipped),
            "context_tokens": {
                "normal_mean": 0.0,
                "smart_mean": 0.0,
                "normal_median": 0.0,
                "smart_median": 0.0,
            },
            "compression_pct": {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0},
            "latency_ms": {
                "normal_mean": 0.0,
                "smart_mean": 0.0,
                "median_delta_ms": 0.0,
            },
            "correctness": {
                "normal": 0.0,
                "smart": 0.0,
            },
        }

    normal_ctx = [float(r.normal_ctx_tokens) for r in kept]
    smart_ctx = [float(r.smart_ctx_tokens) for r in kept]
    comp_pct = [r.compression_pct for r in kept]
    normal_lat = [r.normal_latency_ms for r in kept]
    smart_lat = [r.smart_latency_ms for r in kept]
    deltas = [n - s for n, s in zip(normal_lat, smart_lat)]
    
    normal_sims = [r.normal_answer_cosine_similarity for r in kept]
    smart_sims = [r.smart_answer_cosine_similarity for r in kept]
    sim_deltas = [r.cosine_similarity_delta for r in kept]

    return {
        "n_total": len(rows),
        "n_kept": len(kept),
        "n_skipped_quota": len(skipped),
        "context_tokens": {
            "normal_mean": _mean(normal_ctx),
            "smart_mean": _mean(smart_ctx),
            "normal_median": _median(normal_ctx),
            "smart_median": _median(smart_ctx),
        },
        "compression_pct": {
            "mean": _mean(comp_pct),
            "median": _median(comp_pct),
            "min": min(comp_pct),
            "max": max(comp_pct),
        },
        "latency_ms": {
            "normal_mean": _mean(normal_lat),
            "smart_mean": _mean(smart_lat),
            "median_delta_ms": _median(deltas),
        },
        "correctness": {
            "normal": sum(1 for r in kept if r.normal_correct) / len(kept),
            "smart": sum(1 for r in kept if r.smart_correct) / len(kept),
        },
        "answer_similarity": {
            "normal_mean": _mean(normal_sims),
            "smart_mean": _mean(smart_sims),
            "mean_delta": _mean(sim_deltas),
        },
    }


def _print_per_category(rows: list[RowResult], *, include_quota_failures: bool) -> None:
    cats = sorted({r.category for r in rows})
    print("\nPer-category breakdown:")
    print(
        f"  {'category':<22} {'n':>4} {'normal_ctx':>11} {'smart_ctx':>10} "
        f"{'comp%':>7} {'normal_acc':>11} {'smart_acc':>10} {'n_sim':>7} {'s_sim':>7}"
    )
    print("  " + "-" * 95)
    for cat in cats:
        kept = [r for r in rows if r.category == cat and (include_quota_failures or not r.quota_failed)]
        if not kept:
            continue
        n = len(kept)
        n_ctx = _mean([float(r.normal_ctx_tokens) for r in kept])
        s_ctx = _mean([float(r.smart_ctx_tokens) for r in kept])
        comp = _mean([r.compression_pct for r in kept])
        n_acc = sum(1 for r in kept if r.normal_correct) / n
        s_acc = sum(1 for r in kept if r.smart_correct) / n
        n_sim = _mean([r.normal_answer_cosine_similarity for r in kept])
        s_sim = _mean([r.smart_answer_cosine_similarity for r in kept])
        print(
            f"  {cat:<22} {n:>4d} {n_ctx:>11.1f} {s_ctx:>10.1f} "
            f"{comp:>6.1f}% {n_acc:>10.0%} {s_acc:>9.0%} {n_sim:>7.3f} {s_sim:>7.3f}"
        )


def _print_per_row(rows: list[RowResult]) -> None:
    print(
        f"\n{'#':>3} {'category':<18} {'normal_ctx':>10} {'smart_ctx':>9} "
        f"{'comp%':>6} {'n_lat_ms':>9} {'s_lat_ms':>9} {'n_acc':>5} {'s_acc':>5} "
        f"{'n_sim':>7} {'s_sim':>7} {'skip':>5}"
    )
    print("-" * 115)
    for r in rows:
        skip = "YES" if r.skipped else ""
        print(
            f"{r.query_id:>3d} {r.category:<18} {r.normal_ctx_tokens:>10d} "
            f"{r.smart_ctx_tokens:>9d} {r.compression_pct:>5.1f}% "
            f"{r.normal_latency_ms:>9.0f} {r.smart_latency_ms:>9.0f} "
            f"{'Y' if r.normal_correct else 'n':>5} {'Y' if r.smart_correct else 'n':>5} "
            f"{r.normal_answer_cosine_similarity:>7.3f} {r.smart_answer_cosine_similarity:>7.3f} "
            f"{skip:>5}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Token-Diet offline eval harness")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N queries (0 = all).",
    )
    parser.add_argument(
        "--include-quota-failures",
        action="store_true",
        help="Include 429-failed runs in aggregates (diagnostics only).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force FakeLLMClient (skip Gemini calls, no quota usage).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write per-row results as JSON to this file.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Token-Diet offline evaluation harness")
    print("=" * 80)

    # -- ingest documents ----------------------------------------------------
    docs = _load_documents()
    print(f"Ingesting {len(docs)} synthetic documents from {DOCS_DIR}")
    cfg = load_config()
    
    from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder
    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    model_name = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    embedder = SentenceTransformersEmbedder(model_name=model_name, device=device)
    cross_encoder = SentenceTransformersCrossEncoder(model_name=ce_model, device=device)
    db = VectorDatabase(embedder=embedder)
    db.add_documents(docs)
    print(f"  database size = {len(db)} chunks")

    # -- build pipelines -----------------------------------------------------
    llm = _make_llm(cfg, force_offline=args.offline)
    components = PipelineComponents(
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    normal_rag = NormalRAG(db, llm, cfg)
    smart_rag = SmartRAG(db, llm, cfg, components=components)

    # -- run ------------------------------------------------------------------
    queries = EVAL_QUERIES if args.limit == 0 else EVAL_QUERIES[: args.limit]
    print(f"Running {len(queries)} evaluation queries through both pipelines...")
    rows: list[RowResult] = []
    for idx, q in enumerate(queries, start=1):
        rows.append(_run_query(idx, q, normal_rag, smart_rag, embedder))
        status = "QUOTA" if rows[-1].quota_failed else "ok"
        print(
            f"  [{idx:>2d}/{len(queries)}] {q.category:<18} "
            f"normal={rows[-1].normal_ctx_tokens:>4}t "
            f"smart={rows[-1].smart_ctx_tokens:>4}t "
            f"comp={rows[-1].compression_pct:>5.1f}% "
            f"n_acc={'Y' if rows[-1].normal_correct else 'n'} "
            f"s_acc={'Y' if rows[-1].smart_correct else 'n'} "
            f"n_sim={rows[-1].normal_answer_cosine_similarity:>5.3f} "
            f"s_sim={rows[-1].smart_answer_cosine_similarity:>5.3f} "
            f"{status}"
        )

    summary = _summarize(rows, include_quota_failures=args.include_quota_failures)

    # -- report ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (quota-failed runs excluded)")
    print("=" * 80)
    s = summary
    print(f"  queries kept         : {s['n_kept']} / {s['n_total']}")
    print(f"  queries skipped (429): {s['n_skipped_quota']}")
    ct = s["context_tokens"]
    print(f"  mean context tokens  : normal={ct['normal_mean']:.1f}  smart={ct['smart_mean']:.1f}")
    print(f"  median context tokens: normal={ct['normal_median']:.1f}  smart={ct['smart_median']:.1f}")
    cp = s["compression_pct"]
    print(
        f"  compression %        : mean={cp['mean']:.1f}%  median={cp['median']:.1f}% "
        f"min={cp['min']:.1f}%  max={cp['max']:.1f}%"
    )
    lt = s["latency_ms"]
    print(
        f"  mean latency (ms)    : normal={lt['normal_mean']:.0f}  smart={lt['smart_mean']:.0f}  "
        f"median delta (normal-smart) = {lt['median_delta_ms']:.0f}"
    )
    cr = s["correctness"]
    print(f"  accuracy             : normal={cr['normal']:.0%}  smart={cr['smart']:.0%}")
    sim = s.get("answer_similarity", {"normal_mean": 0.0, "smart_mean": 0.0, "mean_delta": 0.0})
    print(
        f"  mean answer similarity: normal={sim['normal_mean']:.4f}  smart={sim['smart_mean']:.4f}  "
        f"mean delta (smart-normal) = {sim['mean_delta']:.4f}"
    )

    _print_per_category(rows, include_quota_failures=args.include_quota_failures)
    _print_per_row(rows)

    if args.output is not None:
        payload = {
            "summary": summary,
            "rows": [asdict(r) for r in rows],
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote per-row results to {args.output}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())