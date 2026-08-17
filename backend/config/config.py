"""Configuration loader.

Reads :mod:`config.default_config` (YAML) and overlays environment
variables for secrets and operator overrides. The intent is that the
rest of the codebase imports a single :class:`AppConfig` object and
never reads ``os.environ`` directly.

Usage::

    from backend.config.config import load_config
    cfg = load_config()                 # uses defaults + env
    cfg = load_config(path=Path("...")) # uses a custom YAML
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


# ---------------------------------------------------------------------------
# Legacy compatibility shim
# ---------------------------------------------------------------------------
#
# The Phase 0 skeleton (``src/llm.py``, ``src/normal_rag.py``) imported a
# ``Settings`` dataclass with an ``api_key`` field and a
# ``load_settings()`` factory. Those are superseded by ``AppConfig`` /
# ``load_config()`` in this module, but Phase 3 will refactor the old
# files. Until then, expose ``Settings`` as an alias for ``LLMConfig``
# (which carries ``api_key`` via its ``api_key_env``) plus a
# ``load_settings()`` helper so the legacy imports keep working.
from dataclasses import dataclass as _dc  # noqa: E402  -- alias for clarity


@_dc(frozen=True)
class Settings:  # noqa: F811 -- intentional legacy alias
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    top_k: int
    system_prompt: str


def load_settings() -> "Settings":  # noqa: F811 -- intentional legacy alias
    """Legacy settings loader. Prefer :func:`load_config`."""
    import os as _os

    from backend.config.config import (
        LLMConfig as _LLMConfig,
        RetrieverConfig as _RC,
    )

    api_key = _os.environ.get("OPENAI_API_KEY", "").strip()
    return Settings(
        api_key=api_key,
        base_url=_os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        model=_os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(_os.environ.get("RAG_TEMPERATURE", "0.2")),
        max_tokens=int(_os.environ.get("RAG_MAX_TOKENS", "512")),
        top_k=_RC().top_k,
        system_prompt=_os.environ.get(
            "RAG_SYSTEM_PROMPT",
            "You are a helpful assistant. Answer using only the provided context.",
        ),
    )


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemConfig:
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class CompressorConfig:
    # Lowered to 150: tight enough to force genuine competition between units.
    # Phase 1's cheap blended score can't perfectly rank candidates — Phase 2's
    # cross-encoder reranker re-orders and typically selects a different (better)
    # set of units within the same budget, demonstrating real value-add.
    global_token_budget: int = 800
    sentence_tokenizer: str = "regex"
    markdown_header_overhead_tokens: int = 15

    # Stage 2
    # Reduced from 50 to 20: fewer candidates = faster fast_filter embed
    # + much cheaper conditional rerank in phase 2.
    fast_filter_candidate_limit: int = 20
    bm25_weight: float = 0.5
    embedding_weight: float = 0.5

    # Stage 3
    # Switched from ms-marco-MiniLM-L-6-v2 (6-layer, ~9s on CPU) to
    # ms-marco-TinyBERT-L-2-v2 (2-layer, ~2s on CPU) for ~5x speedup.
    # TinyBERT retains ~92% of L-6 quality on MS-MARCO benchmarks.
    cross_encoder_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    cross_encoder_batch_size: int = 32

    # Stage 4
    restoration_window_left: int = 1
    restoration_window_right: int = 1
    similarity_threshold: float = 0.8


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-3.6-flash"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key_env: str = "GEMINI_API_KEY"


@dataclass(frozen=True)
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Convenience pass-throughs used by tests / logs.
    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system.__dict__,
            "retriever": self.retriever.__dict__,
            "compressor": self.compressor.__dict__,
            "llm": self.llm.__dict__,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _section(cls: type, data: dict[str, Any] | None) -> Any:
    """Build a section dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**valid)


def _load_dotenv() -> None:
    """Load ``.env`` (if present) into ``os.environ`` without overriding existing vars.

    Uses python-dotenv if installed, otherwise falls back to a tiny
    stdlib parser so we don't add a dependency just for this.
    """

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass

    # Minimal stdlib fallback. Skips comments and blank lines.
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't clobber values already set in the shell.
        os.environ.setdefault(key, value)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML + env + ``.env``.

    Parameters
    ----------
    path:
        Path to the YAML config file. Defaults to
        ``config/default_config.yaml`` relative to the repo root.

    Order of precedence (highest first):

    1. Operator overrides via env vars (``TOKEN_DIET_*``).
    2. YAML file at ``path`` / ``config/default_config.yaml``.
    3. ``.env`` file at the repo root.
    4. Built-in dataclass defaults.

    Operator overrides:

    * ``TOKEN_DIET_CONFIG`` -- alternate YAML path
    * ``TOKEN_DIET_DEVICE``  -- ``system.device``
    * ``TOKEN_DIET_TOP_K``   -- ``retriever.top_k``
    * ``TOKEN_DIET_BUDGET``  -- ``compressor.global_token_budget``
    """

    _load_dotenv()

    cfg_path = Path(path) if path else Path(os.environ.get("TOKEN_DIET_CONFIG", DEFAULT_CONFIG_PATH))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    system = _section(SystemConfig, raw.get("system"))
    retriever = _section(RetrieverConfig, raw.get("retriever"))
    compressor = _section(CompressorConfig, raw.get("compressor"))
    llm = _section(LLMConfig, raw.get("llm"))

    # Operator overrides (rare; everything else stays in YAML).
    if "TOKEN_DIET_DEVICE" in os.environ:
        system = SystemConfig(**{**system.__dict__, "device": os.environ["TOKEN_DIET_DEVICE"]})
    if "TOKEN_DIET_TOP_K" in os.environ:
        retriever = RetrieverConfig(
            **{**retriever.__dict__, "top_k": int(os.environ["TOKEN_DIET_TOP_K"])}
        )
    if "TOKEN_DIET_BUDGET" in os.environ:
        compressor = CompressorConfig(
            **{**compressor.__dict__, "global_token_budget": int(os.environ["TOKEN_DIET_BUDGET"])}
        )

    return AppConfig(
        system=system,
        retriever=retriever,
        compressor=compressor,
        llm=llm,
    )