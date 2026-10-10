"""
Cross-SDK fixtures: the TypeScript SDK generates these expected outputs and
both SDKs must reproduce them. Refresh with scripts/sync_shared_fixtures.py.
"""
import json
from pathlib import Path

import pytest

from raglite.chunking import RecursiveChunker
from raglite.errors import ChunkingError
from raglite.retrieval.fusion import RankedList, reciprocal_rank_fusion
from raglite.retrieval.keyword_index import KeywordHit, KeywordIndex
from raglite.text import tokenize
from raglite.types import ChunkMetadata
from raglite.utils.hash import hash_string, namespace_from_path
from raglite.vectordb.base import IndexedChunk

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "shared"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CHUNKER = _load("chunker.json")
HASH = _load("hash.json")
TOKENIZER = _load("tokenizer.json")
BM25 = _load("bm25.json")
RRF = _load("rrf.json")


@pytest.mark.parametrize("case", CHUNKER, ids=[c["name"] for c in CHUNKER])
def test_chunker_matches_typescript(case):
    chunker = RecursiveChunker(case["chunkSize"], case["overlap"])
    if case.get("error"):
        with pytest.raises(ChunkingError):
            chunker.split(case["text"])
    else:
        assert chunker.split(case["text"]) == case["chunks"]


@pytest.mark.parametrize("case", HASH["hashString"], ids=lambda c: repr(c["input"]))
def test_hash_string_matches_typescript(case):
    assert hash_string(case["input"]) == case["sha256"]


@pytest.mark.parametrize("case", HASH["namespaceFromPath"], ids=lambda c: c["input"])
def test_namespace_matches_typescript(case):
    assert namespace_from_path(case["input"]) == case["namespace"]


@pytest.mark.parametrize("case", TOKENIZER, ids=[c["name"] for c in TOKENIZER])
def test_tokenizer_matches_typescript(case):
    assert tokenize(case["text"]) == case["tokens"]


@pytest.fixture(scope="module")
def bm25_index():
    chunks = BM25["chunks"]
    return KeywordIndex.build(
        [
            IndexedChunk(
                id=c["id"],
                text=c["text"],
                metadata=ChunkMetadata(source="corpus.txt", chunk=i + 1, totalChunks=len(chunks)),
            )
            for i, c in enumerate(chunks)
        ]
    )


@pytest.mark.parametrize("case", BM25["queries"], ids=lambda c: repr(c["query"]))
def test_bm25_matches_typescript(case, bm25_index):
    actual = bm25_index.search(case["query"], case["topK"])
    assert [h.id for h in actual] == [h["id"] for h in case["hits"]]
    for got, want in zip(actual, case["hits"]):
        assert got.score == pytest.approx(want["score"], abs=1e-6)


def _ranked(name, spec):
    hits = [
        KeywordHit(id=h["id"], text=h["text"], metadata=ChunkMetadata(**h["metadata"]), score=h["score"])
        for h in spec["hits"]
    ]
    return RankedList(name, spec["weight"], hits)


@pytest.mark.parametrize("case", RRF, ids=[c["name"] for c in RRF])
def test_rrf_matches_typescript(case):
    actual = reciprocal_rank_fusion(
        [_ranked("vector", case["vector"]), _ranked("keyword", case["keyword"])],
        case["rrfK"],
        case["topK"],
    )
    assert [r.id for r in actual] == [r["id"] for r in case["results"]]
    for got, want in zip(actual, case["results"]):
        assert got.score == pytest.approx(want["score"], abs=1e-9)
        assert got.scores.model_dump() == pytest.approx(want["scores"], abs=1e-9)


_LOCATIONS = _load("locations.json")


@pytest.mark.parametrize("case", _LOCATIONS["chunks"], ids=lambda c: c["name"])
def test_chunk_locations(case):
    from raglite.chunking.locations import chunk_location, markdown_headings, page_starts

    if "pages" in case:
        text = "\n".join(case["pages"])
        layout = {"pageStarts": page_starts(case["pages"])}
    else:
        text = case["markdown"]
        layout = {"headings": markdown_headings(text)}
    spans = RecursiveChunker(case["chunkSize"], case["overlap"]).spans(text)
    got = [{**span._asdict(), **chunk_location(layout, span)} for span in spans]
    assert got == case["chunks"]


@pytest.mark.parametrize("case", _LOCATIONS["citations"], ids=lambda c: c["answer"])
def test_citations(case):
    from raglite.llm.prompt import extract_citations
    from raglite.types import SearchResult

    context = [
        SearchResult(
            id=f"c{n}",
            text=f"passage {n}",
            metadata=ChunkMetadata(source="doc.pdf", chunk=n, totalChunks=3, page=n),
            score=1.0,
            distance=0.0,
        )
        for n in (1, 2, 3)
    ]
    assert [c.n for c in extract_citations(case["answer"], context)] == case["cited"]
