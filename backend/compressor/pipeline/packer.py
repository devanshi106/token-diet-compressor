"""Stage 5 -- Pack & Order Context (plan §4 Stage 5).

Reassembles selected units into a single Markdown block that:

* Sorts units by ``(doc_id, chunk_id, position_idx)`` so narrative flow
  is preserved.
* Groups units from the same source under a Markdown header.
* Normalizes whitespace (one blank line between units, no trailing
  whitespace).

Used by both Stage 4 (for budget validation) and the top-level
``compress_context`` to produce the final payload.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from backend.rag.models import ContextUnit

_WS_RE = re.compile(r"[ \t]+")


def pack_and_order_context(
    selected_units: list[ContextUnit],
    parent_chunks: dict[str, list[ContextUnit]] | None = None,
) -> str:
    """Return a Markdown string of the selected units.

    ``parent_chunks`` is currently unused but kept in the signature so
    that future packer revisions (e.g. inserting parent-chunk structural
    separators) can use it without changing call sites.
    """

    del parent_chunks  # reserved

    if not selected_units:
        return ""

    # Group by source document, preserving order of first appearance.
    by_doc: "OrderedDict[str, list[ContextUnit]]" = OrderedDict()
    for u in sorted(
        selected_units,
        key=lambda u: (u.doc_id, u.parent_chunk_id, u.position_idx),
    ):
        by_doc.setdefault(u.doc_id, []).append(u)

    sections: list[str] = []
    for doc_id, units in by_doc.items():
        body = "\n\n".join(_normalize(u.target_text) for u in units)
        sections.append(f"### {doc_id}\n\n{body}")

    return "\n\n---\n\n".join(sections)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    # Collapse runs of blank lines to a single blank line.
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()