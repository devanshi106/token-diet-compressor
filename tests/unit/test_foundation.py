"""Tests for Phase 1: configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.config.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    CompressorConfig,
    LLMConfig,
    RetrieverConfig,
    SystemConfig,
    load_config,
)


def test_default_config_file_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists(), f"missing {DEFAULT_CONFIG_PATH}"


def test_load_config_returns_app_config() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.system, SystemConfig)
    assert isinstance(cfg.retriever, RetrieverConfig)
    assert isinstance(cfg.compressor, CompressorConfig)
    assert isinstance(cfg.llm, LLMConfig)


def test_load_config_has_expected_defaults() -> None:
    cfg = load_config()
    # Sanity-check a few values from config/default_config.yaml.
    assert cfg.retriever.top_k == 5
    assert cfg.compressor.global_token_budget == 800
    assert cfg.compressor.fast_filter_candidate_limit == 50
    assert cfg.compressor.similarity_threshold == 0.8
    assert cfg.llm.provider == "gemini"
    assert cfg.llm.api_key_env == "GEMINI_API_KEY"


def test_env_override_top_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a fresh YAML so we don't rely on the on-disk default.
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "retriever:\n  top_k: 3\ncompressor:\n  global_token_budget: 200\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TOKEN_DIET_TOP_K", "11")
    cfg = load_config(yaml_path)
    assert cfg.retriever.top_k == 11  # env override wins
    assert cfg.compressor.global_token_budget == 200  # untouched


def test_env_override_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("compressor:\n  global_token_budget: 200\n", encoding="utf-8")
    monkeypatch.setenv("TOKEN_DIET_BUDGET", "1500")
    cfg = load_config(yaml_path)
    assert cfg.compressor.global_token_budget == 1500


def test_as_dict_round_trip() -> None:
    cfg = load_config()
    d = cfg.as_dict()
    assert d["compressor"]["global_token_budget"] == cfg.compressor.global_token_budget
    assert d["llm"]["model"] == cfg.llm.model


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "retriever:\n  top_k: 7\n  unknown_field: ignored\n"
        "future_section:\n  thing: 1\n",
        encoding="utf-8",
    )
    cfg = load_config(yaml_path)
    assert cfg.retriever.top_k == 7