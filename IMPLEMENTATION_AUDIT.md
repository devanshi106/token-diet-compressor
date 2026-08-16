# Implementation Audit: Token-Diet Dynamic Context Compressor

This document performs a thorough audit comparing the current implementation in the `token-diet-compressor` repository against the requirements specified in `PRD.md`.

---

## 1. Requirement-by-Requirement Audit

### 1.1 `all-MiniLM-L6-v2` Embeddings
* **PRD Section Reference:** §4 Stage 2, §6, §15, §31
* **Corresponding Code:** 
  - [`backend/compressor/pipeline/fast_filter.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/fast_filter.py)
  - [`frontend/app.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/frontend/app.py)
  - [`scripts/run_evaluation.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/scripts/run_evaluation.py)
* **Implementation Status:** **Partially Implemented** (The pipeline logic accepts an `Embedder` interface, but production and evaluation harnesses use a deterministic mock `HashEmbedder` instead of the actual `all-MiniLM-L6-v2` model).
* **Evidence from Code:**
  - In [`app.py` lines 149-156](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/frontend/app.py#L149-L156):
    ```python
    def _build_components():
        return PipelineComponents(
            embedder=HashEmbedder(),
            cross_encoder=WordOverlapCrossEncoder(),
        )
    ```
  - In [`scripts/run_evaluation.py` lines 435-438](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/scripts/run_evaluation.py#L435-L438):
    ```python
    components = PipelineComponents(
        embedder=HashEmbedder(),
        cross_encoder=WordOverlapCrossEncoder(),
    )
    ```
* **Exact Change Required:**
  1. Create a concrete class `SentenceTransformersEmbedder` inside a new file [`backend/embeddings/local_models.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/embeddings/local_models.py) inheriting from `Embedder`.
  2. Implement lazy initialization using `sentence_transformers.SentenceTransformer` targeting the model `sentence-transformers/all-MiniLM-L6-v2` on the device specified in the config (`cpu` or `cuda`).
  3. Swap out `HashEmbedder()` for the real `SentenceTransformersEmbedder` in both `frontend/app.py` and `scripts/run_evaluation.py` when running in non-test mode or when the user requests live/evaluation execution.
* **Tests Needed to Verify:**
  - Create a unit test `test_sentence_transformers_embedder()` in `tests/unit/test_interfaces.py` that instantiates the class, embeds a list of sample strings, and asserts that the shape matches `[N, 384]`.

---

### 1.2 `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **PRD Section Reference:** §4 Stage 3, §6, §12, §31
* **Corresponding Code:** 
  - [`backend/compressor/pipeline/reranker.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/reranker.py)
  - [`frontend/app.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/frontend/app.py)
  - [`scripts/run_evaluation.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/scripts/run_evaluation.py)
* **Implementation Status:** **Partially Implemented** (The pipeline logic accepts a `CrossEncoder` interface, but production and evaluation harnesses use a deterministic mock `WordOverlapCrossEncoder` instead of the actual `ms-marco-MiniLM-L-6-v2` model).
* **Evidence from Code:**
  - In [`app.py` lines 149-156](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/frontend/app.py#L149-L156):
    ```python
    components = PipelineComponents(
        embedder=HashEmbedder(),
        cross_encoder=WordOverlapCrossEncoder(),
    )
    ```
* **Exact Change Required:**
  1. Create a concrete class `SentenceTransformersCrossEncoder` in [`backend/embeddings/local_models.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/embeddings/local_models.py) inheriting from `CrossEncoder`.
  2. Implement lazy initialization using `sentence_transformers.CrossEncoder` targeting `cross-encoder/ms-marco-MiniLM-L-6-v2` on the configured device.
  3. Swap out `WordOverlapCrossEncoder()` for the real `SentenceTransformersCrossEncoder` in `frontend/app.py` and `scripts/run_evaluation.py`.
* **Tests Needed to Verify:**
  - Write a unit test `test_sentence_transformers_cross_encoder()` verifying that it returns correct relevance scores for paired sequences.

---

### 1.3 BM25 Filtering
* **PRD Section Reference:** §4 Stage 2, §11
* **Corresponding Code:** [`backend/compressor/pipeline/fast_filter.py` lines 41-78](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/fast_filter.py#L41-L78)
* **Implementation Status:** **Fully Implemented**
* **Evidence from Code:** Pure-Python BM25 index built inside `fast_filter_candidates()` on the candidate set context. It is built exactly once per request.
* **Exact Change Required:** None.
* **Tests Needed to Verify:** None (Existing tests pass).

---

### 1.4 Batched Cross-Encoder Inference
* **PRD Section Reference:** §4 Stage 3, §12, §31
* **Corresponding Code:** [`backend/compressor/pipeline/reranker.py` lines 21-58](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/reranker.py#L21-L58)
* **Implementation Status:** **Partially Implemented** (The pipeline passes batched inputs, but the mock encoder executes them sequentially. We must verify batch size handles properly on the real model loader).
* **Evidence from Code:**
  - In `reranker.py`:
    ```python
    scores = cross_encoder.predict(pairs, batch_size=batch_size)
    ```
* **Exact Change Required:**
  - Ensure `SentenceTransformersCrossEncoder.predict` propagates the `batch_size` argument to the underlying `SentenceTransformer.CrossEncoder` call.
* **Tests Needed to Verify:**
  - A test passing multiple batch sizes to the model to ensure compatibility.

---

### 1.5 `scoring_text` with ±1 Sentence Context
* **PRD Section Reference:** §4 Stage 1, §8, §31
* **Corresponding Code:** [`backend/compressor/pipeline/unit_formation.py` lines 309-329](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/unit_formation.py#L309-L329)
* **Implementation Status:** **Fully Implemented**
* **Evidence from Code:**
  - Prose units split into sentences, and a sliding context window of `scoring_window_left` and `scoring_window_right` sentences is built as `scoring_text` for reranking.
* **Exact Change Required:** None.
* **Tests Needed to Verify:** None.

---

### 1.6 Structured Block Parsing (JSON, Table, and Code Blocks)
* **PRD Section Reference:** §4 Stage 1, §8, §31
* **Corresponding Code:** [`backend/compressor/pipeline/unit_formation.py` lines 103-167](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/unit_formation.py#L103-L167)
* **Implementation Status:** **Partially Implemented with Bugs** (The parsers extract structured blocks but do not verify whether they overlap with each other. A table or JSON inside a code block will get extracted multiple times, corrupting offsets and duplicating content).
* **Evidence from Code:**
  - In `unit_formation.py`, `_extract_structured_blocks` searches for tables using line prefixes `|` independently of code blocks, and the JSON regex matches greedily. If a Markdown table is inside a code block, both are added to `blocks`, causing offset collision.
* **Exact Change Required:**
  1. Revamp `_extract_structured_blocks` to prevent character range overlap.
  2. Implement an exact offset tracker using `chunk_text.splitlines(keepends=True)` to map character starts of each line.
  3. Extract code blocks first. Then extract JSON blocks that do not overlap with code blocks. Finally, extract tables ensuring they do not overlap with either code blocks or JSON.
* **Tests Needed to Verify:**
  - Add `test_structured_block_no_overlap()` verifying that a Markdown table inside a fenced code block does not get extracted twice.

---

### 1.7 Global Token Budget & Exact Final Token Validation & Priority-Aware Shrinking
* **PRD Section Reference:** §4 Stage 4, §13, §14, §16, §31
* **Corresponding Code:** [`backend/compressor/pipeline/selector.py` lines 50-229](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/selector.py#L50-L229)
* **Implementation Status:** **Fully Implemented**
* **Evidence from Code:** Compete for the same budget; final output string is tokenized using `tiktoken` inside `_enforce_budget_invariant`; drops lowest score/priority units using `min(..., key=lambda u: priorities)`.
* **Exact Change Required:** None.
* **Tests Needed to Verify:** None.

---

### 1.8 Document-Order Packing
* **PRD Section Reference:** §4 Stage 5
* **Corresponding Code:** [`backend/compressor/pipeline/packer.py` lines 40-53](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/compressor/pipeline/packer.py#L40-L53)
* **Implementation Status:** **Fully Implemented**
* **Evidence from Code:** Sorts by document id, chunk id, and original sentence index before assembly.
* **Exact Change Required:** None.
* **Tests Needed to Verify:** None.

---

### 1.9 Normal RAG vs Smart RAG Boundary
* **PRD Section Reference:** §2, §17, §18
* **Corresponding Code:** [`backend/rag/normal_rag.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/rag/normal_rag.py) and [`backend/rag/smart_rag.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/rag/smart_rag.py)
* **Implementation Status:** **Fully Implemented**
* **Evidence from Code:** Shares identical database inputs and LLM stream generation logic.
* **Exact Change Required:** None.
* **Tests Needed to Verify:** None.

---

### 1.10 30-Query Evaluation Set
* **PRD Section Reference:** §20
* **Corresponding Code:** [`datasets/demo/queries/evaluation_queries.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/datasets/demo/queries/evaluation_queries.py)
* **Implementation Status:** **Partially Implemented** (The evaluation query set contains only 25 queries, whereas the PRD specifies a 30-query set).
* **Evidence from Code:**
  - Length of `EVAL_QUERIES` list is exactly 25.
* **Exact Change Required:**
  - Append 5 more high-quality evaluation queries to `EVAL_QUERIES` covering missing technical edge cases (e.g. structured data, distractor-heavy contexts).
* **Tests Needed to Verify:**
  - Assert that `len(EVAL_QUERIES) == 30` in tests.

---

### 1.11 Answer Cosine Similarity
* **PRD Section Reference:** §20, §21 (Metric 10)
* **Corresponding Code:**
  - [`frontend/app.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/frontend/app.py)
  - [`scripts/run_evaluation.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/scripts/run_evaluation.py)
  - [`backend/evaluation/evaluation.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/evaluation/evaluation.py)
* **Implementation Status:** **Missing** (Mentioned in comments/descriptions, but not actually computed or displayed in the dashboard or evaluation outputs).
* **Evidence from Code:**
  - No reference to `cosine_similarity` of generated answers in `app.py` metrics construction (only listed in text string).
  - Missing in `RowResult` and evaluation runners.
* **Exact Change Required:**
  1. Add `answer_cosine_similarity` field to `ExperimentAResult`, `ExperimentBResult`, and `RowResult`.
  2. Implement similarity calculation in `evaluation.py` and `run_evaluation.py` using `cosine_similarity` on embeddings of Normal RAG and Smart RAG answers.
  3. Add a metric card (card #10) in `app.py` showing "Answer Cosine Similarity".
* **Tests Needed to Verify:**
  - Unit test verifying that cosine similarity between two generated strings is correctly calculated using the real embedder.

---

### 1.12 Latency/TTFT Metrics
* **PRD Section Reference:** §17, §18, §19, §21
* **Corresponding Code:**
  - [`backend/rag/normal_rag.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/rag/normal_rag.py)
  - [`backend/rag/smart_rag.py`](file:///C:/Users/Lenovo/Desktop/token-diet-compressor/backend/rag/smart_rag.py)
* **Implementation Status:** **Fully Implemented**
* **Evidence from Code:** Exact streaming block measurement yields correct TTFT on the LLM client stream generator.
* **Exact Change Required:** None.
* **Tests Needed to Verify:** None.

---

## 2. Summary Status Matrix

| ID | PRD Requirement | File / Function | Status | Gaps / Action |
| :--- | :--- | :--- | :--- | :--- |
| 1 | all-MiniLM-L6-v2 Embeddings | `fast_filter.py` | Partially | App & Eval script use `HashEmbedder`. Needs lazy local model class. |
| 2 | ms-marco-MiniLM-L-6-v2 Cross-Encoder | `reranker.py` | Partially | App & Eval script use `WordOverlapCrossEncoder`. Needs lazy local model class. |
| 3 | BM25 Filtering | `fast_filter.py` | Fully | Built once per query. |
| 4 | Batched Cross-Encoder Inference | `reranker.py` | Partially | Propagate batch size parameter to actual model wrapper. |
| 5 | ±1 Sentence context window | `unit_formation.py` | Fully | Prose scoring texts constructed correctly. |
| 6 | Structured block extraction | `unit_formation.py` | Partially | Overlapping range bugs (table/JSON inside code blocks). |
| 7 | Global Token Budget | `selector.py` | Fully | Shared pool constraint. |
| 8 | Exact Final Token Validation | `selector.py` | Fully | Single tiktoken run at end. |
| 9 | Priority-aware shrinking | `selector.py` | Fully | Drops lowest score units. |
| 10 | Document-order packing | `packer.py` | Fully | Sorted and formatted under source headers. |
| 11 | Normal/Smart Boundary | `normal_rag.py`, `smart_rag.py` | Fully | Strict baseline comparisons. |
| 12 | 30-Query Evaluation Set | `evaluation_queries.py` | Partially | Only has 25 queries. Add 5 more. |
| 13 | Answer Cosine Similarity | `app.py`, `run_evaluation.py` | Missing | Not calculated or reported anywhere. Implement. |
| 14 | Latency/TTFT Metrics | `normal_rag.py`, `smart_rag.py` | Fully | Latencies measured precisely. |
