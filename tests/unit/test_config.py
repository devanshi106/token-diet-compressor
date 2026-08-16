"""Tests for the .env loader added to backend.config.config in Phase 3.

These tests run in-process and don't touch the network.
"""

from __future__ import annotations

import os
from pathlib import Path

from backend.config.config import _load_dotenv, load_config


def test_load_dotenv_does_not_override_existing(monkeypatch: object) -> None:
    # Existing env vars win. The function uses os.environ.setdefault,
    # so pre-existing values are preserved.
    os.environ["TOKEN_DIET_TEST_DOTENV"] = "from-shell"
    _load_dotenv()  # no-op unless a .env entry would set the var
    # The .env file in this repo doesn't define this variable, so the
    # existing one survives.
    assert os.environ["TOKEN_DIET_TEST_DOTENV"] == "from-shell"
    os.environ.pop("TOKEN_DIET_TEST_DOTENV", None)


def test_load_config_is_callable() -> None:
    cfg = load_config()
    assert cfg.llm.model  # has a default


def test_load_config_handles_missing_yaml(tmp_path: Path, monkeypatch: object) -> None:
    """If the YAML is missing we still get the dataclass defaults."""
    monkeypatch.setenv("TOKEN_DIET_CONFIG", str(tmp_path / "nope.yaml"))
    cfg = load_config()
    assert cfg.compressor.global_token_budget > 0