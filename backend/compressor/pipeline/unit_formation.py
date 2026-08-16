"""Stage 1 -- Structure / Context Unit Formation (plan §4 Stage 1).

Takes raw retrieved chunks and produces fine-grained :class:`ContextUnit`
objects, each carrying:

* ``target_text``   -- the actual sentence / logical block that may be
  emitted if the unit is selected.
* ``scoring_text``  -- ``target_text`` plus a small local context window
  (default: ±1 sentence) so the Cross-Encoder can resolve pronouns and
  references at scoring time. For structured units the scoring_text is
  the target itself (it is already structure-preserving).
* ``token_count``   -- the *exact* token count of ``target_text``,
  precomputed once via :func:`src.tokenizer_utils.count_tokens`. This is
  the key perf trick called out in plan §14: subsequent stages sum these
  precomputed counts instead of re-running the tokenizer in the hot loop.

The stage also tags each unit with positional metadata
(``doc_id``, ``chunk_id``, ``parent_chunk_id``, ``position_idx``) and
attaches the per-chunk unit list as ``parent_chunks[chunk_id]`` for
later lookup by the selector.

The structured parser here is deliberately lightweight -- it identifies
fenced code blocks (```` ``` ````), JSON objects, and Markdown tables
and emits each as a single logical unit. If extraction fails (e.g.
malformed JSON), the chunk falls back to prose parsing rather than
raising -- this is the "Fallback Handling" guarantee from plan §4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable
from backend.rag.interfaces import SentenceSplitter
from backend.rag.models import ContextUnit, PROSE, RetrievedChunk, STRUCTURED
from backend.embeddings.tokenizer import count_tokens


# ---------------------------------------------------------------------------
# Sentence splitter (regex-based, NLTK-free default)
# ---------------------------------------------------------------------------


class RegexSentenceSplitter(SentenceSplitter):
    """Lightweight sentence splitter with no external dependencies.

    Splits on terminal punctuation followed by whitespace and a capital
    letter / digit / opening quote / opening bracket. Handles the common
    abbreviations below to avoid over-splitting:

        e.g.  i.e.  mr.  mrs.  dr.  vs.  etc.  cf.  no.  pp.

    This is intentionally the default so the pipeline works on machines
    where NLTK data isn't downloaded; swap to NLTK by passing
    :class:`NLTKSentenceSplitter` (planned) or another implementation
    via :func:`form_context_units`.
    """

    _ABBREVIATIONS = {
        "e.g", "i.e", "mr", "mrs", "ms", "dr", "st", "mt", "vs",
        "etc", "cf", "no", "pp", "fig", "approx",
    }

    _SPLIT_RE = re.compile(
        r"""
        (?<=[.!?])              # after sentence-final punctuation
        \s+                    # one or more whitespace chars
        (?=[A-Z0-9"'(\[])       # followed by capital / digit / quote / bracket
        """,
        re.VERBOSE,
    )

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Protect abbreviations from splitting.
        protected = text
        for abbr in self._ABBREVIATIONS:
            protected = re.sub(
                rf"(?i)\b{re.escape(abbr)}\.",
                abbr + "<DOT>",
                protected,
            )

        # Also protect decimals like "3.14".
        protected = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", protected)

        raw = self._SPLIT_RE.split(protected)
        sentences = []
        for s in raw:
            s = s.replace("<DEC>", ".").replace("<DOT>", ".").strip()
            if s:
                sentences.append(s)
        return sentences


# ---------------------------------------------------------------------------
# Structured block extraction
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class _Block:
    """A logical block extracted from a chunk before splitting."""

    text: str
    kind: str  # "prose", "code", "json", "table"
    start: int  # char offset within the chunk


def find_json_blocks(text: str) -> list[tuple[int, int, str]]:
    """Find all top-level balanced JSON objects/arrays in text.
    Returns list of (start_idx, end_idx, text_content).
    """
    blocks = []
    n = len(text)
    i = 0
    while i < n:
        char = text[i]
        if char in ('{', '['):
            start = i
            brace_stack = [char]
            j = i + 1
            while j < n and brace_stack:
                c = text[j]
                if c == '{' or c == '[':
                    brace_stack.append(c)
                elif c == '}':
                    if brace_stack and brace_stack[-1] == '{':
                        brace_stack.pop()
                    else:
                        brace_stack.append(c)
                elif c == ']':
                    if brace_stack and brace_stack[-1] == '[':
                        brace_stack.pop()
                    else:
                        brace_stack.append(c)
                j += 1
            if not brace_stack:
                candidate = text[start:j]
                try:
                    json.loads(candidate)
                    blocks.append((start, j, candidate))
                    i = j - 1
                except ValueError:
                    pass
        i += 1
    return blocks


def flatten_json(obj: Any, parent_path: str = "") -> list[tuple[str, str]]:
    """Flatten nested dict/list into a list of (path_key, value_string) tuples."""
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{parent_path}.{k}" if parent_path else k
            if isinstance(v, (dict, list)):
                items.extend(flatten_json(v, new_path))
            else:
                items.append((new_path, str(v)))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_path = f"{parent_path}[{idx}]"
            if isinstance(v, (dict, list)):
                items.extend(flatten_json(v, new_path))
            else:
                items.append((new_path, str(v)))
    else:
        items.append((parent_path, str(obj)))
    return items


def _decompose_json(
    block_text: str,
    doc_id: str,
    chunk_id: str,
    parent_chunk_id: str,
    pos_offset: int,
) -> list[ContextUnit]:
    try:
        obj = json.loads(block_text)
    except Exception:
        return []

    flattened = flatten_json(obj)
    if not flattened:
        return []

    chunk_units = []
    group_size = 5
    for group_idx in range(0, len(flattened), group_size):
        sub_items = flattened[group_idx : group_idx + group_size]
        target_lines = [f"{path} = {val}" for path, val in sub_items]
        target_text = "\n".join(target_lines)

        pos = pos_offset + len(chunk_units)
        unit_id = f"{doc_id}_{chunk_id}_{pos}"
        
        unit = ContextUnit(
            unit_id=unit_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            target_text=target_text,
            scoring_text=target_text,
            unit_type=STRUCTURED,
            position_idx=pos,
            parent_chunk_id=parent_chunk_id,
            token_count=count_tokens(target_text),
            metadata={"block_kind": "json", "decomposed": True},
        )
        chunk_units.append(unit)
    return chunk_units


def _decompose_table(
    block_text: str,
    doc_id: str,
    chunk_id: str,
    parent_chunk_id: str,
    pos_offset: int,
) -> list[ContextUnit]:
    lines = [line.strip() for line in block_text.splitlines() if line.strip()]
    if len(lines) < 3:
        return []

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    if not (header.startswith("|") and separator.startswith("|")):
        return []

    chunk_units = []
    for row in rows:
        if not row.startswith("|"):
            continue
        target_text = f"{header}\n{separator}\n{row}"
        pos = pos_offset + len(chunk_units)
        unit_id = f"{doc_id}_{chunk_id}_{pos}"
        
        unit = ContextUnit(
            unit_id=unit_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            target_text=target_text,
            scoring_text=target_text,
            unit_type=STRUCTURED,
            position_idx=pos,
            parent_chunk_id=parent_chunk_id,
            token_count=count_tokens(target_text),
            metadata={"block_kind": "table", "decomposed": True},
        )
        chunk_units.append(unit)
    return chunk_units


def _decompose_code(
    block_text: str,
    doc_id: str,
    chunk_id: str,
    parent_chunk_id: str,
    pos_offset: int,
) -> list[ContextUnit]:
    m = _FENCE_RE.match(block_text)
    if not m:
        return []
    lang = m.group(1).strip()
    body = m.group(2)

    lines = body.splitlines()
    segments: list[list[str]] = []
    current_segment: list[str] = []

    def flush_segment():
        if current_segment:
            seg_text = "\n".join(current_segment).strip()
            if seg_text:
                segments.append(current_segment.copy())
            current_segment.clear()
    for line in lines:
        # Check class/def at top level
        is_decl = line.startswith(("def ", "class ", "async def "))
        is_import = line.startswith(("import ", "from "))

        if is_decl:
            flush_segment()
            current_segment.append(line)
        elif is_import:
            has_non_import = any(not (l.strip().startswith(("import ", "from ")) or not l.strip()) for l in current_segment)
            if has_non_import:
                flush_segment()
            current_segment.append(line)
        else:
            current_segment.append(line)
    flush_segment()

    if not segments:
        return []

    # If only 1 segment, treat as single block via fallback
    if len(segments) == 1:
        return []

    chunk_units = []
    for seg in segments:
        seg_body = "\n".join(seg)
        target_text = f"```{lang}\n{seg_body}\n```"
        pos = pos_offset + len(chunk_units)
        unit_id = f"{doc_id}_{chunk_id}_{pos}"
        
        unit = ContextUnit(
            unit_id=unit_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            target_text=target_text,
            scoring_text=target_text,
            unit_type=STRUCTURED,
            position_idx=pos,
            parent_chunk_id=parent_chunk_id,
            token_count=count_tokens(target_text),
            metadata={"block_kind": "code", "decomposed": True},
        )
        chunk_units.append(unit)
    return chunk_units


def _extract_structured_blocks(chunk_text: str) -> list[_Block]:
    blocks: list[_Block] = []

    def is_overlapping(start: int, end: int) -> bool:
        for b in blocks:
            b_end = b.start + len(b.text)
            if not (end <= b.start or start >= b_end):
                return True
        return False

    # 1. Fenced code blocks
    for m in _FENCE_RE.finditer(chunk_text):
        lang = m.group(1).strip()
        body = m.group(2)
        label = f"```{lang or ''}\n{body}```"
        blocks.append(_Block(text=label, kind="code", start=m.start()))

    # 2. JSON blocks (top-level balanced, non-overlapping)
    for start, end, val in find_json_blocks(chunk_text):
        if not is_overlapping(start, end):
            blocks.append(_Block(text=val, kind="json", start=start))

    # 3. Markdown tables (using exact line offset tracking, non-overlapping)
    line_starts = []
    cursor = 0
    for line in chunk_text.splitlines(keepends=True):
        line_starts.append((cursor, line))
        cursor += len(line)

    i = 0
    while i < len(line_starts):
        start_offset, line_text = line_starts[i]
        if _TABLE_LINE_RE.match(line_text):
            j = i
            run_text = ""
            while j < len(line_starts) and _TABLE_LINE_RE.match(line_starts[j][1]):
                run_text += line_starts[j][1]
                j += 1
            
            end_offset = start_offset + len(run_text)
            if not is_overlapping(start_offset, end_offset):
                blocks.append(_Block(text=run_text, kind="table", start=start_offset))
            i = j
        else:
            i += 1

    return blocks


@dataclass
class UnitFormationOutput:
    """Result of running unit formation."""

    units: list[ContextUnit] = field(default_factory=list)
    parent_chunks: dict[str, list[ContextUnit]] = field(default_factory=dict)


def form_context_units(
    chunks: Iterable[RetrievedChunk],
    splitter: SentenceSplitter | None = None,
    *,
    scoring_window_left: int = 1,
    scoring_window_right: int = 1,
) -> UnitFormationOutput:
    """Convert retrieved chunks into :class:`ContextUnit` objects."""

    splitter = splitter or RegexSentenceSplitter()
    out = UnitFormationOutput()

    for chunk in chunks:
        chunk_units: list[ContextUnit] = []
        structured = _extract_structured_blocks(chunk.text)

        if not structured:
            chunk_units.extend(
                _build_prose_units(
                    chunk=chunk,
                    text=chunk.text,
                    splitter=splitter,
                    scoring_window_left=scoring_window_left,
                    scoring_window_right=scoring_window_right,
                    position_offset=0,
                )
            )
        else:
            cursor = 0
            pos = 0
            for block in sorted(structured, key=lambda b: b.start):
                if block.start > cursor:
                    prose_segment = chunk.text[cursor:block.start]
                    prose_units = _build_prose_units(
                        chunk=chunk,
                        text=prose_segment,
                        splitter=splitter,
                        scoring_window_left=scoring_window_left,
                        scoring_window_right=scoring_window_right,
                        position_offset=pos,
                    )
                    chunk_units.extend(prose_units)
                    pos += len(prose_units)

                # Attempt logical decomposition for structured blocks
                decomposed_units = []
                if block.kind == "json":
                    decomposed_units = _decompose_json(block.text, chunk.doc_id, chunk.chunk_id, chunk.chunk_id, pos)
                elif block.kind == "table":
                    decomposed_units = _decompose_table(block.text, chunk.doc_id, chunk.chunk_id, chunk.chunk_id, pos)
                elif block.kind == "code":
                    decomposed_units = _decompose_code(block.text, chunk.doc_id, chunk.chunk_id, chunk.chunk_id, pos)

                if decomposed_units:
                    chunk_units.extend(decomposed_units)
                    pos += len(decomposed_units)
                else:
                    # Fallback handling
                    if block.kind in ("json", "table"):
                        # Fall back to plain prose splitting
                        prose_units = _build_prose_units(
                            chunk=chunk,
                            text=block.text,
                            splitter=splitter,
                            scoring_window_left=scoring_window_left,
                            scoring_window_right=scoring_window_right,
                            position_offset=pos,
                        )
                        chunk_units.extend(prose_units)
                        pos += len(prose_units)
                    else:
                        # Code fallback: treat as single structured block
                        unit_id = f"{chunk.doc_id}_{chunk.chunk_id}_{pos}"
                        unit = ContextUnit(
                            unit_id=unit_id,
                            doc_id=chunk.doc_id,
                            chunk_id=chunk.chunk_id,
                            target_text=block.text,
                            scoring_text=block.text,
                            unit_type=STRUCTURED,
                            position_idx=pos,
                            parent_chunk_id=chunk.chunk_id,
                            token_count=count_tokens(block.text),
                            metadata={"block_kind": block.kind},
                        )
                        chunk_units.append(unit)
                        pos += 1

                cursor = block.start + len(block.text)

            if cursor < len(chunk.text):
                prose_segment = chunk.text[cursor:]
                prose_units = _build_prose_units(
                    chunk=chunk,
                    text=prose_segment,
                    splitter=splitter,
                    scoring_window_left=scoring_window_left,
                    scoring_window_right=scoring_window_right,
                    position_offset=pos,
                )
                chunk_units.extend(prose_units)

        out.units.extend(chunk_units)
        out.parent_chunks[chunk.chunk_id] = chunk_units

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_sentences_for_count(text: str, splitter: SentenceSplitter) -> list[str]:
    """Split ``text`` into sentences, returning [] for empty/whitespace."""
    return splitter.split(text) if text and text.strip() else []


def _build_prose_units(
    *,
    chunk: RetrievedChunk,
    text: str,
    splitter: SentenceSplitter,
    scoring_window_left: int,
    scoring_window_right: int,
    position_offset: int,
) -> list[ContextUnit]:
    """Turn a prose segment into :class:`ContextUnit` objects with scoring windows."""

    sentences = _split_sentences_for_count(text, splitter)
    if not sentences:
        return []

    units: list[ContextUnit] = []
    n = len(sentences)
    for idx, sentence in enumerate(sentences):
        start = max(0, idx - scoring_window_left)
        end = min(n - 1, idx + scoring_window_right)
        scoring_text = " ".join(sentences[start : end + 1])

        unit_id = f"{chunk.doc_id}_{chunk.chunk_id}_{position_offset + idx}"
        units.append(
            ContextUnit(
                unit_id=unit_id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                target_text=sentence,
                scoring_text=scoring_text,
                unit_type=PROSE,
                position_idx=position_offset + idx,
                parent_chunk_id=chunk.chunk_id,
                token_count=count_tokens(sentence),
                metadata={"sent_start": start, "sent_end": end},
            )
        )
    return units