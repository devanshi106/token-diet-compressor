"""Prompt builder.

Turns retrieved chunks into a single chat-completion message list. The
shape is deliberately boring:

* system  -- persona + instructions + the cited context
* user    -- the question

Keeping the prompt construction isolated here means the rest of the
codebase can stay LLM-agnostic.

The :func:`build_messages` helper accepts any chunk-like object that
exposes ``.text`` and (optionally) ``.source`` / ``.doc_id``. Both the
Phase 0 :class:`src.retriever.Chunk` and the Phase 1
:class:`src.models.RetrievedChunk` satisfy this duck-typed contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class _HasText(Protocol):
    text: str
    doc_id: str
    chunk_id: str


@dataclass
class _View:
    """Internal view that handles both Chunk and RetrievedChunk uniformly."""

    text: str
    doc_id: str
    chunk_id: str

    @classmethod
    def from_any(cls, c: object) -> "_View | None":
        if c is None:
            return None
        # Duck-typed: anything with ``text``, ``doc_id``, ``chunk_id`` works.
        text = getattr(c, "text", None)
        if not isinstance(text, str):
            return None
        return cls(
            text=text,
            doc_id=str(getattr(c, "doc_id", "doc")),
            chunk_id=str(getattr(c, "chunk_id", "chunk")),
        )


def build_messages(
    query: str,
    chunks: Iterable[object] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build a chat-completion ``messages`` array."""

    system = (
        system_prompt
        or "You are a helpful assistant. Answer using only the provided context. "
        "If the context does not contain the answer, say you don't know."
    )
    context_block = _format_context(chunks or [])
    full_system = f"{system}\n\n{context_block}" if context_block else system

    return [
        {"role": "system", "content": full_system},
        {"role": "user", "content": query.strip()},
    ]


def _format_context(chunks: Iterable[object]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        view = _View.from_any(c)
        if view is None:
            continue
        text = view.text.strip().replace("\n", " ")
        lines.append(f"[{i}] ({view.doc_id}/{view.chunk_id}) {text}")
    if not lines:
        return ""
    return "Context:\n" + "\n".join(lines)