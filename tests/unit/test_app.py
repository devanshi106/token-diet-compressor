"""Phase 4 tests for the Streamlit dashboard.

These tests verify that the dashboard module is import-safe outside
the Streamlit runtime (we don't actually start a browser UI here --
that requires port 8765 and a real browser, which is the smoke
verification in the repo README).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

app = importlib.import_module("app")


def test_app_module_imports_without_streamlit_runtime() -> None:
    """`python -c 'import app'` must succeed without launching Streamlit."""
    assert hasattr(app, "main")
    assert callable(app.main)


def test_app_helpers_are_streamlit_free() -> None:
    """Pure-Python helpers used by the dashboard should not require Streamlit."""
    # These helpers must be importable and call-able without spinning up
    # the Streamlit runner.
    assert app._estimate_cost(1_000_000) > 0
    assert app._keyword_correctness("the FAISS library", ("FAISS",)) is True
    assert app._keyword_correctness("nothing relevant", ("FAISS",)) is False
    assert app._keyword_correctness("anything", ()) is True


def test_load_fixtures_returns_markdown_docs() -> None:
    import app

    docs = app._load_fixtures()
    assert len(docs) >= 1
    for doc_id, text in docs:
        assert doc_id
        # Synthetic docs use the new "SYNTHETIC DEVELOPMENT" marker,
        # while legacy fixtures use "SAMPLE FIXTURE". Either is acceptable.
        assert ("SAMPLE FIXTURE" in text) or ("SYNTHETIC DEVELOPMENT" in text)


def test_build_database_uses_fixtures() -> None:
    import app

    db = app._build_database()
    assert len(db) > 0
    # Basic smoke: a retrieval should return something.
    hits = db.retrieve("FAISS", top_k=1)
    assert len(hits) == 1


def test_build_components_creates_pipeline_components() -> None:
    import app

    components = app._build_components()
    assert components.embedder is not None
    assert components.cross_encoder is not None


def test_estimate_cost_handles_zero() -> None:
    import app

    assert app._estimate_cost(0) == 0.0
    assert app._estimate_cost(800) > 0.0


def test_app_main_signature_is_zero_arg() -> None:
    import inspect

    import app

    sig = inspect.signature(app.main)
    assert len(sig.parameters) == 0