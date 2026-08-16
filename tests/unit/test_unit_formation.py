"""Tests for Stage 1 -- unit_formation."""

from __future__ import annotations

from tests.unit._pipeline_fakes import make_chunk

from backend.rag.models import PROSE, STRUCTURED
from backend.compressor.pipeline.unit_formation import (
    RegexSentenceSplitter,
    form_context_units,
)


def test_regex_splitter_empty_and_whitespace() -> None:
    s = RegexSentenceSplitter()
    assert s.split("") == []
    assert s.split("   \n  ") == []


def test_regex_splitter_basic_sentences() -> None:
    s = RegexSentenceSplitter()
    out = s.split("The dog barks. The cat meows. Birds fly.")
    assert out == ["The dog barks.", "The cat meows.", "Birds fly."]


def test_regex_splitter_preserves_abbreviations() -> None:
    s = RegexSentenceSplitter()
    out = s.split("Use e.g. the API. It works well.")
    # Must NOT split inside "e.g."
    assert out[0].startswith("Use e.g.")
    assert "the API." in out[0]


def test_regex_splitter_handles_decimal_numbers() -> None:
    s = RegexSentenceSplitter()
    out = s.split("Pi is about 3.14 in value. It is irrational.")
    # "3.14" should not split the sentence.
    assert any("3.14" in x for x in out)


def test_unit_formation_assigns_unique_ids() -> None:
    chunks = [make_chunk("First sentence. Second sentence.", chunk_id="c1")]
    out = form_context_units(chunks)
    ids = [u.unit_id for u in out.units]
    assert len(ids) == len(set(ids))
    assert ids == ["doc1_c1_0", "doc1_c1_1"]


def test_unit_formation_target_and_scoring_text() -> None:
    text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
    chunks = [make_chunk(text)]
    out = form_context_units(chunks)
    assert [u.unit_type for u in out.units] == [PROSE, PROSE, PROSE]
    assert out.units[0].target_text == "Alpha bravo charlie."
    # ±1 sentence window -- middle unit sees all three.
    assert out.units[0].scoring_text == "Alpha bravo charlie. Delta echo foxtrot."
    assert out.units[1].scoring_text == text  # full window covers everything
    assert out.units[2].scoring_text == "Delta echo foxtrot. Golf hotel india."


def test_unit_formation_precomputes_token_counts() -> None:
    chunks = [make_chunk("One two three. Four five six seven.")]
    out = form_context_units(chunks)
    for u in out.units:
        # All token counts must be > 0 and exactly match target_text.
        assert u.token_count > 0
        from backend.embeddings.tokenizer import count_tokens

        assert u.token_count == count_tokens(u.target_text)


def test_unit_formation_extracts_fenced_code() -> None:
    chunk_text = "Intro paragraph.\n\n```python\nprint('hi')\n```\n\nOutro paragraph."
    chunks = [make_chunk(chunk_text)]
    out = form_context_units(chunks)
    types = [u.unit_type for u in out.units]
    assert PROSE in types
    assert STRUCTURED in types
    code = next(u for u in out.units if u.unit_type == STRUCTURED)
    assert "print('hi')" in code.target_text


def test_unit_formation_extracts_json() -> None:
    chunk_text = 'Header. {"alpha": 1, "beta": [2, 3]}. Trailer.'
    chunks = [make_chunk(chunk_text)]
    out = form_context_units(chunks)
    json_units = [u for u in out.units if u.unit_type == STRUCTURED and u.metadata.get("block_kind") == "json"]
    assert len(json_units) == 1
    assert "alpha" in json_units[0].target_text


def test_unit_formation_json_fallback_to_prose() -> None:
    # Malformed JSON: must NOT raise, must end up as prose.
    chunk_text = "Opening. {not: valid json at all. Trailing text."
    chunks = [make_chunk(chunk_text)]
    out = form_context_units(chunks)
    # No structured JSON unit should have been emitted.
    assert not any(u.metadata.get("block_kind") == "json" for u in out.units)
    # Some prose units survived.
    assert any(u.unit_type == PROSE for u in out.units)


def test_unit_formation_extracts_markdown_table() -> None:
    chunk_text = (
        "Top prose sentence.\n"
        "| a | b |\n| - | - |\n| 1 | 2 |\n"
        "Bottom prose sentence."
    )
    chunks = [make_chunk(chunk_text)]
    out = form_context_units(chunks)
    table_units = [u for u in out.units if u.metadata.get("block_kind") == "table"]
    assert len(table_units) == 1
    assert "| a | b |" in table_units[0].target_text


def test_unit_formation_parent_chunks_indexed() -> None:
    chunks = [
        make_chunk("Alpha.", chunk_id="c1"),
        make_chunk("Beta.", chunk_id="c2"),
    ]
    out = form_context_units(chunks)
    assert set(out.parent_chunks.keys()) == {"c1", "c2"}
    assert [u.target_text for u in out.parent_chunks["c1"]] == ["Alpha."]
    assert [u.target_text for u in out.parent_chunks["c2"]] == ["Beta."]


def test_unit_formation_empty_chunk_yields_no_units() -> None:
    out = form_context_units([make_chunk("")])
    assert out.units == []
    # Chunk is indexed even if empty, so the selector can later see it.
    assert out.parent_chunks == {"c1": []}


def test_unit_formation_scenarios_json_and_table_and_code() -> None:
    import json
    from backend.embeddings.tokenizer import count_tokens
    
    # 1 & 2. Small JSON and Nested JSON
    nest_json = '{"user": {"id": 42, "profile": {"name": "Alice", "roles": ["admin", "user"]}}}'
    chunks = [make_chunk(nest_json, chunk_id="c_nested")]
    out = form_context_units(chunks)
    json_units = [u for u in out.units if u.metadata.get("block_kind") == "json"]
    assert len(json_units) > 0
    # Paths should be flattened in the target text
    first_target = json_units[0].target_text
    assert "user.id = 42" in first_target or "user.profile.name = Alice" in first_target
    assert "user.profile.roles[0] = admin" in first_target

    # 3. Large JSON
    large_dict = {f"key_{i}": f"value_{i}" for i in range(12)}
    large_json = json.dumps(large_dict)
    chunks_large = [make_chunk(large_json, chunk_id="c_large_json")]
    out_large = form_context_units(chunks_large)
    large_units = [u for u in out_large.units if u.metadata.get("block_kind") == "json"]
    assert len(large_units) == 3
    for u in large_units:
        assert len(u.target_text.splitlines()) <= 5

    # 4. Malformed JSON
    malformed = '{"user": {"id": 42'
    chunks_mal = [make_chunk(malformed, chunk_id="c_mal_json")]
    out_mal = form_context_units(chunks_mal)
    assert not any(u.metadata.get("block_kind") == "json" for u in out_mal.units)
    assert len(out_mal.units) > 0

    # 5 & 6. Markdown Table and Multi-row Table
    table_text = (
        "| ID | Name | Role |\n"
        "|---|---|---|\n"
        "| 1 | Alice | Admin |\n"
        "| 2 | Bob | User |\n"
        "| 3 | Charlie | Guest |\n"
    )
    chunks_tbl = [make_chunk(table_text, chunk_id="c_table")]
    out_tbl = form_context_units(chunks_tbl)
    tbl_units = [u for u in out_tbl.units if u.metadata.get("block_kind") == "table"]
    assert len(tbl_units) == 3
    for u in tbl_units:
        lines = u.target_text.splitlines()
        assert len(lines) == 3
        assert lines[0] == "| ID | Name | Role |"
        assert lines[1] == "|---|---|---|"
        assert "ID" in u.target_text
        assert u.token_count == count_tokens(u.target_text)

    # 7. Code with Multiple Functions
    code_text = (
        "```python\n"
        "import os\n"
        "import sys\n\n"
        "def func_one():\n"
        "    return 1\n\n"
        "class MyClass:\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "def func_two():\n"
        "    return 2\n"
        "```"
    )
    chunks_code = [make_chunk(code_text, chunk_id="c_code")]
    out_code = form_context_units(chunks_code)
    code_units = [u for u in out_code.units if u.metadata.get("block_kind") == "code"]
    assert len(code_units) == 4
    assert "import os" in code_units[0].target_text
    assert "def func_one" in code_units[1].target_text
    assert "class MyClass" in code_units[2].target_text
    assert "def func_two" in code_units[3].target_text
    for u in code_units:
        assert u.target_text.startswith("```python")
        assert u.target_text.endswith("```")

    # 8. Malformed/Unparseable Code
    short_code = "```python\nprint('hello')\n```"
    chunks_short = [make_chunk(short_code, chunk_id="c_short")]
    out_short = form_context_units(chunks_short)
    short_units = [u for u in out_short.units if u.metadata.get("block_kind") == "code"]
    assert len(short_units) == 1
    assert short_units[0].target_text == short_code

    # 9, 10 & 11. Token Count & Metadata Correctness
    for u in out_tbl.units:
        assert u.unit_id.startswith("doc1_c_table_")
        assert u.doc_id == "doc1"
        assert u.chunk_id == "c_table"
        assert u.parent_chunk_id == "c_table"
        assert isinstance(u.position_idx, int)
        assert u.unit_type == STRUCTURED
        assert u.token_count > 0
        assert u.token_count == count_tokens(u.target_text)