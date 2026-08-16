# Token-Diet Dynamic Context Compressor

A 5-stage token compression middleware that sits between a retriever
and a final LLM, enforcing a hard token budget on the prompt without
losing the relevant facts.

```
User Query
    │
    ├──► Normal RAG (baseline)
    │        └─► Retriever ─► Raw Top-K chunks ─► LLM ─► Answer
    │
    └──► Smart RAG (Token-Diet)
             └─► Retriever ─► Compressor middleware ─► LLM ─► Answer
                              │
                              ▼
                  1. Unit formation (prose + structured)
                  2. Fast filter (BM25 + embedding similarity)
                  3. Cross-Encoder rerank
                  4. Greedy budget-aware selection
                  5. Per-document packing
```

Built per the plan in `data/plan7.md`.

## Quick start

```powershell
# from repo root
pip install -r requirements.txt
copy .env.example .env          # then put your key in .env
streamlit run app.py
```

The Streamlit dashboard opens on http://localhost:8501 (or whichever
port you specified). Side-by-side answers, per-stage timings, and the
10 dashboard metrics from plan §21 are displayed in priority order.

## Headless verification

```powershell
# 100 tests, ~2 s, no API key required
pytest tests/ -v

# Live smoke test against Gemini (requires GEMINI_API_KEY in env)
$env:PYTHONPATH = "$PWD"
python scripts/live_smoke.py
```

## Repository layout

```
src/
├── config.py            # YAML + .env loader, AppConfig dataclasses
├── database.py          # VectorDatabase + NumPy / FAISS indices
├── llm.py               # LLMClient ABC + Gemini + Fake implementations
├── normal_rag.py        # Plan §17 baseline
├── smart_rag.py         # Plan §18 optimized (uses the same LLM/db)
├── evaluation.py        # Plan §19/§20 (Experiments A & B, EvalQuery)
├── tokenizer_utils.py   # tiktoken wrapper, count_tokens()
├── interfaces.py        # SentenceSplitter / Embedder / CrossEncoder / Retriever ABCs
├── models.py            # ContextUnit, ScoredCandidate, CompressorOutput, ...
├── prompt.py            # build_messages()
├── pipeline/
│   ├── unit_formation.py    # Stage 1
│   ├── fast_filter.py       # Stage 2 (pure-Python BM25)
│   ├── reranker.py          # Stage 3 (batchable Cross-Encoder)
│   ├── selector.py          # Stage 4 (greedy + redundancy prune)
│   └── packer.py            # Stage 5 (per-document headers)
data/
├── plan7.md             # Implementation plan (the source of truth)
├── documents/fixtures/  # SAMPLE FIXTURE markdown corpus for the dashboard
tests/
├── _pipeline_fakes.py   # Shared test doubles (HashEmbedder, ...)
└── ...                  # 100 tests across 14 files
config/
└── default_config.yaml  # Single source of truth for tunables
scripts/
└── live_smoke.py        # End-to-end smoke against live Gemini
app.py                   # Streamlit dashboard (plan §21, §28)
```

## Configuration

Everything lives in `config/default_config.yaml`. Override via env:

| Env var                          | Section / field                 |
| -------------------------------- | ------------------------------- |
| `TOKEN_DIET_CONFIG`              | alternate YAML path             |
| `TOKEN_DIET_DEVICE`              | `system.device`                 |
| `TOKEN_DIET_TOP_K`               | `retriever.top_k`               |
| `TOKEN_DIET_BUDGET`              | `compressor.global_token_budget`|
| `GEMINI_API_KEY`                 | from `.env` (auto-loaded)       |

### Embedding Model Setup

The production pipeline utilizes the real local embedding model `sentence-transformers/all-MiniLM-L6-v2` for the vector database and the fast filter/redundancy stages. The model weights are loaded lazily upon the first request and cached globally in memory.

To configure target hardware, set `TOKEN_DIET_DEVICE` or modify the `system.device` field in `config/default_config.yaml` to `"cpu"` or `"cuda"`. Output embeddings are automatically L2-normalized to allow fast cosine similarity computations using dot product.


## Faithfulness to `data/plan7.md`

- §2 — Both pipelines share the same `VectorDatabase` and `LLMClient` instances.
- §4 — 5-stage pipeline; budget invariant enforced at the end.
- §6 — Gemma/Gemini via official SDK, FAISS optional, sentence-transformers lazy.
- §9 — `tiktoken` (cl100k_base) as the default tokenizer.
- §13/§14 — Precomputed `token_count` per unit; O(1) incremental cost.
- §15 — Cosine-similarity redundancy pruning.
- §16 — `assert final_tokens <= global_token_budget` invariant.
- §17/§18 — Normal RAG + Smart RAG instrumentation (TTFT, total ms).
- §19/§20 — Experiment A (frozen chunks) + Experiment B (E2E), `DEFAULT_QUERY_SET`.
- §21 — Streamlit dashboard with all 10 metrics in plan priority order.
- §22/§23/§24 — 100 tests covering pipeline stages, integrations, and live smoke.

## Status

All four phases complete. **100/100 tests passing**, end-to-end live
Gemini verified, Streamlit dashboard runs on first `pip install -r requirements.txt`.

Live numbers from the fixture corpus (will improve dramatically with a
real ~50–200 candidate corpus):

| Metric                | Normal RAG | Smart RAG | Delta     |
| --------------------- | ---------- | --------- | --------- |
| Context tokens        | 732        | 688       | −44       |
| LLM TTFT              | 2.78 s     | 1.85 s    | −0.93 s   |
| Total                 | 2.91 s     | 1.97 s    | −0.94 s   |
