"""datasets — synthetic evaluation corpus for Token-Diet.

Provides :class:`EvalQuery` directly from ``datasets/demo_company/queries.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent


class EvalQuery:
    """A single evaluation query loaded from queries.json."""

    __slots__ = ("query", "category", "difficulty", "required_keywords", "source_doc", "notes", "_expected_answer")

    def __init__(
        self,
        query: str,
        *,
        category: str = "",
        difficulty: str = "medium",
        required_keywords: tuple[str, ...] = (),
        source_doc: str = "",
        notes: str = "",
        _expected_answer: str = "",
    ) -> None:
        self.query = query
        self.category = category
        self.difficulty = difficulty
        self.required_keywords = required_keywords
        self.source_doc = source_doc
        self.notes = notes
        self._expected_answer = _expected_answer

    @property
    def reference_answer(self) -> str:
        return self._expected_answer


def _load_eval_queries() -> list[EvalQuery]:
    json_path = _ROOT / "demo_company" / "queries.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    out: list[EvalQuery] = []
    for q in data.get("queries", []):
        req_kws = tuple(q.get("required_keywords", []))
        expected_docs = q.get("expected_documents", [])
        source_doc = expected_docs[0] if expected_docs else ""
        eq = EvalQuery(
            query=q["query"],
            category=q.get("category", ""),
            difficulty="medium",
            required_keywords=req_kws,
            source_doc=source_doc,
            notes=q.get("challenge", ""),
            _expected_answer=q.get("expected_answer", ""),
        )
        out.append(eq)
    return out


EVAL_QUERIES: list[EvalQuery] = _load_eval_queries()
__all__ = ["EvalQuery", "EVAL_QUERIES"]