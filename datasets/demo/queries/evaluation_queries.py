"""Reference answers for the dashboard's cosine-similarity scoring.

Source of truth lives at ``datasets/demo_company/queries.json``. We
materialise it as a ``{query: expected_answer}`` dict so the dashboard
can look up a reference answer for any query the user types. If the
user's query doesn't exactly match any entry, cosine similarity is
skipped silently (the dashboard relies on the optional `Reference
answer` textarea for ad-hoc queries).

This module is generated at import time, not committed to git in
duplicate form, so JSON stays the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

_QUERIES_JSON = Path(__file__).resolve().parents[2] / "demo_company" / "queries.json"

try:
    _raw = json.loads(_QUERIES_JSON.read_text(encoding="utf-8"))
    REFERENCE_ANSWERS: dict[str, str] = {
        item["query"]: item.get("expected_answer", "")
        for item in _raw.get("queries", [])
    }
except Exception:
    # If the JSON file is missing or malformed, fall back to an empty
    # mapping so the dashboard still imports cleanly. Cosine similarity
    # will be skipped (the `if ref_ans:` guard in app.py).
    REFERENCE_ANSWERS = {}


__all__ = ["REFERENCE_ANSWERS"]