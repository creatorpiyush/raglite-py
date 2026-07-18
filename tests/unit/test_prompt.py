"""
Unit tests for prompt builders — mirrors tests/unit/prompt.test.ts
"""
import pytest
from raglite.llm.prompt import build_system_prompt, build_user_prompt
from raglite.types import SearchResult, ChunkMetadata


def make_result(id_, text, source="doc.txt", chunk=1, total=3, score=0.9):
    return SearchResult(
        id=id_,
        text=text,
        metadata=ChunkMetadata(source=source, chunk=chunk, totalChunks=total),
        score=score,
        distance=1 - score,
    )


class TestBuildSystemPrompt:
    def test_returns_default_base_prompt(self):
        prompt = build_system_prompt()
        assert "precise assistant" in prompt
        assert "I could not find the answer" in prompt

    def test_appends_system_hint_when_provided(self):
        prompt = build_system_prompt(system_hint="Reply in French.")
        assert "Reply in French." in prompt
        assert "precise assistant" in prompt

    def test_hint_via_dict_options(self):
        prompt = build_system_prompt({"systemHint": "Be concise."})
        assert "Be concise." in prompt


class TestBuildUserPrompt:
    def test_includes_context_and_question(self):
        ctx = [make_result("1", "Dogs are loyal animals.")]
        prompt = build_user_prompt("What are dogs?", ctx)
        assert "Dogs are loyal animals." in prompt
        assert "What are dogs?" in prompt

    def test_includes_citation_tag_with_citations_enabled(self):
        ctx = [make_result("1", "Paris is in France.", chunk=2)]
        prompt = build_user_prompt("Where is Paris?", ctx, include_citations=True)
        assert "[1] (doc.txt #2)" in prompt
        assert "Cite the passages" in prompt

    def test_omits_source_info_without_citations(self):
        ctx = [make_result("1", "Some content here.")]
        prompt = build_user_prompt("Question?", ctx, include_citations=False)
        assert "[1]" in prompt
        assert "doc.txt" not in prompt
        assert "Cite the passages" not in prompt

    def test_multiple_chunks_separated_by_hr(self):
        ctx = [
            make_result("1", "First chunk."),
            make_result("2", "Second chunk.", chunk=2),
        ]
        prompt = build_user_prompt("Question?", ctx)
        assert "---" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_empty_context_produces_empty_context_block(self):
        prompt = build_user_prompt("What?", [])
        assert "Context:\n\n" in prompt
        assert "Question: What?" in prompt
