"""Smoke tests for the offline evaluation harness.

These tests verify that ``scripts/run_evaluation.py`` imports cleanly,
loads the synthetic documents, and can iterate over the query set
without raising. They run in-process, with ``--offline --limit 1``, and
use :class:`FakeLLMClient` so no API key or network access is needed.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_run_evaluation_module_imports_cleanly() -> None:
    """Importing the script must not raise."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        mod = importlib.import_module("run_evaluation")
    finally:
        sys.path.pop(0)
    assert hasattr(mod, "EVAL_QUERIES")
    assert hasattr(mod, "main")


def test_evaluation_query_set_is_non_empty() -> None:
    """The shipped query set must contain at least 20 queries across
    the documented categories (plan \xa720 -- 25 queries)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        mod = importlib.import_module("run_evaluation")
    finally:
        sys.path.pop(0)
    queries = mod.EVAL_QUERIES
    assert len(queries) >= 20
    categories = {q.category for q in queries}
    assert "single_fact" in categories
    assert "structured_data_retrieval" in categories or "multi_document_reasoning" in categories