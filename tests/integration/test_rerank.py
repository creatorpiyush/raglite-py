"""
Reranking with a function, Cohere, Voyage and an LLM. Mirrors
raglite/tests/integration/rerank.test.ts.
"""
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from raglite.core.collection import DocumentCollection
from raglite.core.document import Document
from raglite.errors import ConfigError
from raglite.retrieval.rerank import parse_ranking

from .test_hybrid import mock_embedder, write  # noqa: F401  (autouse fixture)


def options(tmp_dir, **extra):
    return {
        "storeDir": os.path.join(tmp_dir, ".raglite"),
        "logLevel": "silent",
        "scoreThreshold": -1,
        **extra,
    }


def facts(tmp_dir, **extra):
    """Forty one-sentence chunks: "Fact 1 ..." to "Fact 40 ..."."""
    text = "\n".join(f"Fact {i + 1} is here." for i in range(40))
    doc = Document(write(tmp_dir, "facts.txt", text), options(tmp_dir, **extra))
    doc.build({"chunkSize": 4, "overlap": 0})
    return doc


def fact(hit):
    return int(re.search(r"Fact (\d+)", hit.text).group(1))


def by_fact_desc(query, hits):
    return sorted(hits, key=fact, reverse=True)


def test_function_reorders_candidates_and_keeps_top_k(tmp_dir):
    doc = facts(tmp_dir)
    seen = []

    def rerank(query, hits):
        seen.append(len(hits))
        return by_fact_desc(query, hits)

    hits = doc.search("fact", top_k=3, rerank=rerank)
    assert seen == [30]
    numbers = [fact(h) for h in hits]
    assert len(numbers) == 3 and numbers == sorted(numbers, reverse=True)


def test_constructor_default_and_rerank_false(tmp_dir):
    doc = facts(tmp_dir, rerank=by_fact_desc)
    assert [fact(h) for h in doc.search("fact", top_k=40)] == list(range(40, 0, -1))
    plain = doc.search("fact", {"topK": 40, "rerank": False})
    assert [fact(h) for h in plain] != list(range(40, 0, -1))


def test_failing_reranker_keeps_retrieval_order_and_warns(tmp_dir, capsys):
    doc = facts(tmp_dir, logLevel="warn")
    plain = doc.search("fact", top_k=5)

    def broken(query, hits):
        raise RuntimeError("rate limited")

    hits = doc.search("fact", top_k=5, rerank=broken)
    assert [h.id for h in hits] == [h.id for h in plain]
    assert "Reranking failed, keeping the retrieval order: rate limited" in capsys.readouterr().err


def test_collection_reranks_once_over_merged_results(tmp_dir):
    os.makedirs(os.path.join(tmp_dir, "docs"))
    write(tmp_dir, "docs/a.txt", "Fact 1 is here.\nFact 2 is here.")
    write(tmp_dir, "docs/b.txt", "Fact 3 is here.\nFact 4 is here.")
    calls = []

    def rerank(query, hits):
        calls.append(len(hits))
        return by_fact_desc(query, hits)

    collection = DocumentCollection(
        os.path.join(tmp_dir, "docs"), options(tmp_dir, chunkSize=4, overlap=0)
    )
    hits = collection.search("fact", top_k=2, rerank=rerank)
    assert calls == [4]
    assert [fact(h) for h in hits] == [4, 3]


def fake_sdk(results, calls):
    def rerank(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            results=[SimpleNamespace(index=i, relevance_score=s) for i, s in results(args, kwargs)]
        )

    client = lambda **settings: SimpleNamespace(rerank=rerank, settings=settings)  # noqa: E731
    return SimpleNamespace(ClientV2=client, Client=client)


def test_cohere_reranks_and_records_the_score(tmp_dir):
    doc = facts(tmp_dir)
    calls = []
    sdk = fake_sdk(lambda a, kw: [(len(kw["documents"]) - 1, 0.9), (0, 0.4)], calls)
    plain = doc.search("fact", top_k=30)
    with patch("raglite.retrieval.rerank.import_optional", return_value=sdk):
        hits = doc.search("fact", top_k=2, rerank={"provider": "cohere", "apiKey": "k"})
    assert calls[0][1]["model"] == "rerank-v3.5" and calls[0][1]["query"] == "fact"
    assert [h.id for h in hits] == [plain[29].id, plain[0].id]
    assert hits[0].score == 0.9 and hits[0].scores.rerank == 0.9
    assert hits[0].scores.vector == pytest.approx(plain[29].score)
    assert hits[0].distance == pytest.approx(0.1)


def test_voyage_reranks(tmp_dir):
    doc = facts(tmp_dir)
    calls = []
    sdk = fake_sdk(lambda a, kw: [(1, 0.7)], calls)
    plain = doc.search("fact", top_k=30)
    with patch("raglite.retrieval.rerank.import_optional", return_value=sdk):
        hits = doc.search("fact", top_k=1, rerank={"provider": "voyage", "apiKey": "k"})
    assert calls[0][0][0] == "fact" and calls[0][1]["model"] == "rerank-2.5"
    assert [h.id for h in hits] == [plain[1].id]


def test_llm_orders_passages_and_scores_by_rank(tmp_dir):
    doc = facts(tmp_dir, llm={"provider": "openai", "apiKey": "k"})
    plain = doc.search("fact", top_k=3)
    with patch("raglite.llm.answer._generate", return_value={"text": "Sure: [3, 1]"}) as gen:
        hits = doc.search("fact", top_k=3, rerank={"provider": "llm", "candidates": 3})
    gen.assert_called_once()
    assert [h.id for h in hits] == [plain[2].id, plain[0].id, plain[1].id]
    assert [h.scores.rerank for h in hits] == pytest.approx([1, 2 / 3, 1 / 3])


def test_rejects_unknown_provider_bad_candidates_and_llm_without_llm(tmp_dir):
    doc = facts(tmp_dir)
    with pytest.raises(ConfigError):
        doc.search("fact", rerank={"provider": "jina"})
    with pytest.raises(ConfigError):
        doc.search("fact", rerank={"provider": "cohere", "candidates": 0})
    with pytest.raises(ConfigError):
        doc.search("fact", rerank={"provider": "llm"})


@pytest.mark.parametrize(
    "text,count,expected",
    [("[2, 1]", 3, [1, 0]), ("Here you go:\n[3,3, 9, 0, 1]", 3, [2, 0]), ("[]", 3, [])],
)
def test_parse_ranking(text, count, expected):
    assert parse_ranking(text, count) == expected


def test_parse_ranking_needs_an_array():
    with pytest.raises(ValueError):
        parse_ranking("passage 2 is best", 3)
