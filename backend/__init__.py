"""Token-Diet Dynamic Context Compressor -- public API.

Top-level ``backend`` package re-exports the most commonly used classes
so callers can write::

    from backend import load_config, NormalRAG, SmartRAG, VectorDatabase

without having to know the internal sub-package layout.
"""

from backend.config.config import (
    AppConfig,
    CompressorConfig,
    LLMConfig,
    RetrieverConfig,
    SystemConfig,
    load_config,
)
from backend.embeddings.tokenizer import (
    TiktokenTokenizer,
    Tokenizer,
    count_tokens,
    get_tokenizer,
    set_tokenizer,
)
from backend.evaluation.evaluation import (
    DEFAULT_QUERY_SET,
    EvalQuery,
    ExperimentAResult,
    ExperimentBResult,
    aggregate_a,
    aggregate_b,
    run_experiment_a,
    run_experiment_b,
)
from backend.llm.gemini_client import (
    FakeLLMClient,
    GeminiLLMClient,
    LLMClient,
    LLMError,
    LLMQuotaExhaustedError,
    LLMResult,
)
from backend.rag.database import VectorDatabase
from backend.rag.interfaces import (
    CrossEncoder,
    Embedder,
    Retriever,
    SentenceSplitter,
)
from backend.rag.models import (
    CompressorOutput,
    ContextUnit,
    PROSE,
    RetrievedChunk,
    ScoredCandidate,
    STRUCTURED,
    cosine_similarity,
)
from backend.rag.normal_rag import NormalRAG, NormalRAGResult
from backend.rag.prompt import build_messages
from backend.rag.smart_rag import SmartRAG, SmartRAGResult
from backend.compressor.pipeline import PipelineComponents, compress_context

__all__ = [
    "AppConfig",
    "CompressorConfig",
    "CompressorOutput",
    "ContextUnit",
    "CrossEncoder",
    "DEFAULT_QUERY_SET",
    "Embedder",
    "EvalQuery",
    "ExperimentAResult",
    "ExperimentBResult",
    "FakeLLMClient",
    "GeminiLLMClient",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMQuotaExhaustedError",
    "LLMResult",
    "NormalRAG",
    "NormalRAGResult",
    "PipelineComponents",
    "PROSE",
    "RetrievedChunk",
    "Retriever",
    "RetrieverConfig",
    "ScoredCandidate",
    "SentenceSplitter",
    "SmartRAG",
    "SmartRAGResult",
    "STRUCTURED",
    "SystemConfig",
    "TiktokenTokenizer",
    "Tokenizer",
    "VectorDatabase",
    "aggregate_a",
    "aggregate_b",
    "build_messages",
    "compress_context",
    "cosine_similarity",
    "count_tokens",
    "get_tokenizer",
    "load_config",
    "run_experiment_a",
    "run_experiment_b",
    "set_tokenizer",
]