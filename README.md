# Token-Diet Dynamic Context Compressor

The **Token-Diet Dynamic Context Compressor** is a post-retrieval context optimization pipeline designed to sit between document retrieval and generation in a Retrieval-Augmented Generation (RAG) system. 

It segments context chunks into granular prose or structured units, ranks them, and selects only the most informative sentences to fit a strict global token budget—dramatically reducing Time-to-First-Token (TTFT) latency and API costs without losing critical facts.

---

## 🚀 Key Performance Wins
Through targeted CPU optimizations, we achieved an **11.4x latency reduction** in the compressor's execution time:

| Stage | Baseline Latency | Optimized Latency | Speedup |
| :--- | :--- | :--- | :--- |
| **Stage 1: Unit Formation** | 6.9 ms | 7.7 ms | -- |
| **Stage 2: Fast Relevance Filter** | **1,215.5 ms** | **5.5 ms** | **221x faster** |
| **Stage 3: Cross-Encoder Rerank** | 89.4 ms | 81.5 ms | -- |
| **Stage 4: Budget Selection** | 30.1 ms | 22.2 ms | -- |
| **Stage 5: Pack & Order** | 0.35 ms | 0.30 ms | -- |
| **Total Compressor Overhead** | **1,342.2 ms** | **117.2 ms** | **11.4x faster** |

*Note: These benchmarks were run on a CPU using a 1,395-token context containing multiple source documents, achieving **48.4% token savings** with fully preserved factual accuracy.*

---

## 🧠 The 5-Stage Compression Pipeline

```
Raw Retrieved Chunks
       │
       ▼
 ┌───────────┐
 │  Stage 1  │  Unit Formation: Segment prose into sentences (via regex/nltk) and 
 └─────┬─────┘  isolate code blocks, JSON, & tables as logical structured units.
       ▼
 ┌───────────┐
 │  Stage 2  │  Fast Relevance Filter: Slice the units down to the top candidates (M=20) 
 └─────┬─────┘  using lexical BM25 first, then compute embeddings ONLY for these candidates.
       ▼
 ┌───────────┐
 │  Stage 3  │  Cross-Encoder Rerank: Re-score top candidate units using context-aware 
 └─────┬─────┘  scoring text (±1 surrounding sentences) to resolve pronoun dependencies.
       ▼
 ┌───────────┐
 │  Stage 4  │  Greedy Budget Selection: Fill the token budget greedily by score, prune 
 └─────┬─────┘  redundant units via cosine similarity, and restore neighboring context sentences.
       ▼
 ┌───────────┐
 │  Stage 5  │  Pack & Order: Reassemble selected units grouped by source document under 
 └─────┬─────┘  Markdown headers, sorted in original order to preserve narrative flow.
       ▼
Packed Context Payload (Hard budget enforced!)
```

---

## 🤖 Core Models Used

The pipeline balances CPU-efficient local intelligence with powerful cloud models:
1. **Local Embedder (Stage 2)**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional) — used for calculating embedding scores and similarity-based redundancy filtering.
2. **Local Reranker (Stage 3)**: `cross-encoder/ms-marco-TinyBERT-L-2-v2` — a lightweight 2-layer Cross-Encoder selected specifically because it runs **~5x faster on CPU** than standard L-6 Cross-Encoders while maintaining high semantic accuracy.
3. **Cloud LLM (Generation)**:
   * **Google Gemini**: Resolves to `gemini-2.5-flash` or `gemini-3.6-flash` using the official Google GenAI SDK.
   * **Groq Gateway**: Defaults to `openai/gpt-oss-120b` (or Llama 3 models) using the standard OpenAI client.

---

## 🛠️ Key Optimization Techniques Implemented

To solve the RAG prompt-bloat problem without introducing heavy latency overhead, the system implements several key optimizations:

### 1. Fine-Grained Prose & Structured Unit Segmentation (Stage 1)
Instead of treating retrieved context paragraphs as single indivisible blocks, we segment prose into sentence-level units and extract tables, JSON, and code as logical structure-preserving blocks. This allows the compressor to selectively discard irrelevant sentences while keeping only the dense, factual content.

### 2. Context-Aware Cross-Encoder Scoring (Stage 3)
Scoring isolated sentences often leads to a "context vacuum" where reference pronouns (e.g., "it", "she", "the function above") lose their meaning. We solve this by pairing the query with the candidate's `scoring_text` (the target sentence + a local context window of ±1 sentence). This maintains high-quality relevance scoring, but only the core `target_text` is packed into the final prompt to maximize compression.

### 3. O(1) Greedy Selection & Precomputed Token Sums (Stage 4)
Enforcing a hard token budget in a loop is traditionally slow because it requires repeatedly calling string tokenizers. We optimize this to $O(1)$ by:
- Precomputing each unit's exact token count once in Stage 1.
- Running the greedy selection loop using basic arithmetic summation of these precomputed counts and header estimates.
- Running the exact tokenizer call **only once** on the final packed prompt at the very end (with a priority-aware fallback to shrink the payload if formatting overhead causes a budget breach).

### 4. BM25-First Pre-Filtering (Stage 2 CPU Optimization)
To avoid running local embedding models on hundreds of units on the CPU, Stage 2 applies a fast lexical BM25 ranking first, immediately filtering the corpus down to the top $M=20$ candidates. The local embedder is then run **only** on those 20 candidates in a single batch, reducing embedding generation time by **~90%**.

### 5. Global Embedding Caching & Batching (Stage 2 CPU Optimization)
- Text embeddings are stored in a global cache (`_EMBEDDING_CACHE`) keyed by `(model, text)` to completely avoid model calls for duplicate sentence strings.
- All missing embeddings for the top candidates are grouped and computed in a **single PyTorch batch** rather than sequential loops, maximizing CPU instruction cache efficiency.

### 6. Exact Server-Side Timing (Groq API Integration)
To isolate internet routing and network transit jitter, we enable `stream_options={"include_usage": True}` on the Groq client. This extracts the precise server-side `prompt_time` (prefill compute time) and `queue_time` from the final chunk of the response stream, allowing the dashboard to display the exact server-side TTFT performance.

---

## 📁 Repository Layout

```
token-diet-compressor/
│
├── backend/
│   ├── compressor/
│   │   └── pipeline/
│   │       ├── unit_formation.py       # Prose & structured logical unit parsing (Stage 1)
│   │       ├── fast_filter.py          # BM25-first + embedding pre-filtering (Stage 2)
│   │       ├── reranker.py             # Batched Cross-Encoder inference (Stage 3)
│   │       ├── selector.py             # Greedy token-budgeted selection (Stage 4)
│   │       └── packer.py               # Document reordering and formatting (Stage 5)
│   │
│   ├── config/
│   │   ├── config.py                   # YAML & environment variable loader
│   │   └── default_config.yaml         # Tunable system configs (budgets, models, devices)
│   │
│   ├── embeddings/
│   │   ├── local_models.py             # SentenceTransformers Embedder & Cross-Encoder cached models
│   │   └── tokenizer.py                # Token count helper abstractions (tiktoken/NLTK/regex)
│   │
│   ├── evaluation/
│   │   └── evaluation.py               # Benchmark runner and metrics calculation
│   │
│   ├── llm/
│   │   ├── gemini_client.py            # Google GenAI SDK integration
│   │   └── groq_client.py              # OpenAI-compatible Groq API client with server timing
│   │
│   └── rag/
│       ├── database.py                 # In-memory database & vector search
│       ├── interfaces.py               # Abstract interfaces defining splitters/embedders/LLMs
│       ├── models.py                   # Dataclasses (ContextUnit, ScoredCandidate, etc.)
│       ├── normal_rag.py               # Baseline Normal RAG implementation (No Compression)
│       └── smart_rag.py                # Token-Diet Smart RAG workflow
│
├── datasets/                           # Query evaluation datasets and fixtures
├── frontend/
│   └── app.py                          # Streamlit comparative dashboard
│
├── tests/
│   ├── unit/                           # 139 passing unit tests covering all pipeline stages
│   └── integration/                    # End-to-end integration tests
│
├── requirements.txt                    # Project package dependencies
└── README.md                           # Quick start and developer instructions
```

---

## ⚙️ Running Locally

### 1. Installation
Ensure you have Python 3.10+ installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```
*(By default, the system loads `GEMINI_API_KEY` to run LLM operations).*

### 3. Launch the Streamlit Dashboard
Run the following command to start the comparative interface:
```bash
python -m streamlit run frontend/app.py
```
Open **http://localhost:8501** in your browser. You can:
- Enter queries to compare **Normal RAG** against **Smart RAG** side-by-side.
- See exact token savings, cost savings, and compressor stage breakdowns.
- View precise Groq server-side timing (prompt prefill vs. queue times) when using Groq.

### 4. Running the Test Suite
The codebase includes 139 passing unit and integration tests. Run the suite to verify your setup:
```bash
python -m pytest tests/ -v
```

### 5. Running the Evaluation Benchmarks
To run the automated benchmark runner against the evaluation query dataset:
```bash
python -m backend.evaluation.evaluation
```
