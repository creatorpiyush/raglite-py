"""
Retrieval evaluation. Mirrors raglite/tests/integration/eval.test.ts.
"""
import json
import os
import re

import pytest

from raglite.cli import run_eval
from raglite.core.document import Document
from raglite.errors import ConfigError
from raglite.eval import (
    EvalQuestion,
    evaluate,
    format_report,
    matches_expected,
    parse_questions,
    score_results,
)
from raglite.types import ChunkMetadata, SearchResult

from .test_hybrid import mock_embedder, write  # noqa: F401  (autouse fixture)


def hit(text, **metadata):
    meta = {"source": "a.pdf", "chunk": 1, "totalChunks": 9, **metadata}
    return SearchResult(id=text, text=text, metadata=ChunkMetadata(**meta), score=1, distance=0)


def test_matches_text_ignoring_case_and_whitespace_and_locations():
    h = hit("Refunds are  issued\nwithin 30 days.", chunk=4, page=2, pageEnd=3)
    assert matches_expected(h, "refunds are issued within")
    assert not matches_expected(h, "60 days")
    assert matches_expected(h, {"source": "a.pdf"})
    assert matches_expected(h, {"source": "a.pdf", "chunk": 4})
    assert not matches_expected(h, {"source": "a.pdf", "chunk": 5})
    assert matches_expected(h, {"source": "a.pdf", "page": 3})
    assert not matches_expected(h, {"source": "a.pdf", "page": 4})
    assert not matches_expected(hit("x"), {"source": "a.pdf", "page": 1})


def test_recall_and_reciprocal_rank():
    hits = [hit("alpha"), hit("beta"), hit("gamma")]
    assert score_results(hits, ["beta"]) == (1, 1 / 2)
    assert score_results(hits, ["gamma", "delta"]) == (1 / 2, 1 / 3)
    assert score_results(hits, ["delta"]) == (0, 0)


def test_parse_questions_reads_jsonl_and_skips_blank_lines():
    qs = parse_questions(
        '{"question": "q1", "expected": ["a"]}\n\n'
        '{"question": "q2", "expected": [{"source": "a.pdf", "page": 2}]}\n'
    )
    assert qs == [
        EvalQuestion("q1", ["a"]),
        EvalQuestion("q2", [{"source": "a.pdf", "page": 2}]),
    ]


@pytest.mark.parametrize(
    "line,message",
    [
        ("not json", "line 1: not valid JSON"),
        ('{"expected": ["a"]}', 'line 1: "question"'),
        ('{"question": "q", "expected": []}', 'line 1: "expected"'),
        ('{"question": "q", "expected": [3]}', "line 1: each"),
        ('{"question": "q", "expected": [{"source": "a", "page": "2"}]}', "line 1: each"),
    ],
)
def test_parse_questions_rejects(line, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        parse_questions(line)


def notes(tmp_dir):
    lines = [f"Note {i + 1} covers topic {i + 1} in one short line." for i in range(20)]
    lines[6] = "The error code ERR_4021 means the gateway timed out."
    doc = Document(
        write(tmp_dir, "notes.txt", "\n".join(lines)),
        {"storeDir": os.path.join(tmp_dir, ".raglite"), "logLevel": "silent", "scoreThreshold": -1},
    )
    doc.build({"chunkSize": 9, "overlap": 0})
    return doc


QUESTIONS = [
    EvalQuestion("ERR_4021", ["gateway timed out"]),
    EvalQuestion("nothing matches this", ["no such passage"]),
]


def test_evaluate_scores_each_mode_and_each_mode_with_rerank(tmp_dir):
    report = evaluate(
        notes(tmp_dir),
        QUESTIONS,
        top_k=3,
        modes=["vector", "keyword"],
        rerank=lambda q, hits: list(reversed(hits)),
    )
    assert [r.mode for r in report.results] == ["vector", "keyword", "vector+rerank", "keyword+rerank"]
    keyword = report.results[1]
    assert (keyword.recall, keyword.mrr, keyword.hitRate) == (0.5, 0.5, 0.5)
    assert keyword.missed == ["nothing matches this"]

    table = format_report(report)
    assert "2 questions, top 3" in table
    assert re.search(r"keyword\s+0\.500\s+0\.500\s+0\.500", table)


def test_cli_prints_the_table_or_json(tmp_dir, capsys):
    notes(tmp_dir)
    questions = write(
        tmp_dir,
        "q.jsonl",
        "\n".join(json.dumps({"question": q.question, "expected": q.expected}) for q in QUESTIONS),
    )
    store = os.path.join(tmp_dir, ".raglite")
    run_eval([os.path.join(tmp_dir, "notes.txt"), questions, "--modes", "keyword", "--store-dir", store])
    assert re.search(r"keyword\s+0\.500", capsys.readouterr().out)

    run_eval([os.path.join(tmp_dir, "notes.txt"), questions, "--json", "--store-dir", store])
    report = json.loads(capsys.readouterr().out)
    assert report["topK"] == 5 and [r["mode"] for r in report["results"]] == ["vector", "keyword", "hybrid"]

    with pytest.raises(ValueError, match="--modes"):
        run_eval([os.path.join(tmp_dir, "notes.txt"), questions, "--modes", "fuzzy"])
