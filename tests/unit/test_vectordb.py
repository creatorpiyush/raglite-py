"""
Unit tests for MemoryVectorStore — mirrors tests/unit/vectordb.test.ts
"""
import math
import os
import shutil
import tempfile

import pytest

from raglite.types import ChunkMetadata, StoredChunk
from raglite.vectordb.memory import MemoryVectorStore


def unit_vec(vec):
    s = sum(v * v for v in vec)
    n = math.sqrt(s)
    return [v / n for v in vec]


def make_chunk(id_: str, text: str, embedding):
    return StoredChunk(
        id=id_,
        text=text,
        embedding=unit_vec(embedding),
        metadata=ChunkMetadata(source="t", chunk=1, totalChunks=3),
    )


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestMemoryVectorStore:
    def test_starts_empty_and_can_add(self, tmp_dir):
        store = MemoryVectorStore(tmp_dir, "ns1")
        store.load()
        assert store.count() == 0
        store.add([make_chunk("a", "a", [1, 0, 0])])
        assert store.count() == 1

    def test_returns_hits_ordered_by_cosine_similarity(self, tmp_dir):
        store = MemoryVectorStore(tmp_dir, "ns2")
        store.load()
        store.add([
            make_chunk("north", "n", [1, 0, 0]),
            make_chunk("east",  "e", [0, 1, 0]),
            make_chunk("down",  "d", [0, 0, -1]),
        ])

        hits = store.search(unit_vec([0.9, 0.1, 0]), 3)
        assert [h.id for h in hits] == ["north", "east", "down"]
        assert hits[0].score > hits[1].score
        for h in hits:
            assert abs(h.distance - (1 - h.score)) < 1e-6

    def test_top_k_limits_result_count(self, tmp_dir):
        store = MemoryVectorStore(tmp_dir, "ns3")
        store.load()
        store.add([
            make_chunk("a", "a", [1, 0, 0]),
            make_chunk("b", "b", [0.9, 0.1, 0]),
            make_chunk("c", "c", [0, 1, 0]),
        ])
        hits = store.search(unit_vec([1, 0, 0]), 2)
        assert len(hits) == 2
        assert hits[0].id == "a"

    def test_persists_to_disk_and_reloads(self, tmp_dir):
        store = MemoryVectorStore(tmp_dir, "ns4")
        store.load()
        store.add([make_chunk("x", "x", [1, 0, 0])])
        from datetime import datetime, timezone

        from raglite.types import IndexMetadata
        store.save_index_metadata(IndexMetadata(
            version="0.1.0",
            source="/tmp/foo.txt",
            sourceHash="abc",
            chunkSize=100,
            overlap=10,
            embeddingProvider="local",
            embeddingModel="m",
            embeddingDimensions=3,
            chunkCount=1,
            createdAt=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        ))

        assert os.path.exists(os.path.join(tmp_dir, "ns4", "chunks.json"))
        assert os.path.exists(os.path.join(tmp_dir, "ns4", "metadata.json"))

        store2 = MemoryVectorStore(tmp_dir, "ns4")
        store2.load()
        assert store2.count() == 1
        meta = store2.read_index_metadata()
        assert meta.sourceHash == "abc"
        assert meta.chunkCount == 1

    def test_reset_clears_state_and_disk(self, tmp_dir):
        store = MemoryVectorStore(tmp_dir, "ns5")
        store.load()
        store.add([make_chunk("y", "y", [1, 0, 0])])
        store.reset()
        assert store.count() == 0
        assert not os.path.exists(os.path.join(tmp_dir, "ns5"))

    def test_isolates_namespaces(self, tmp_dir):
        a = MemoryVectorStore(tmp_dir, "left")
        b = MemoryVectorStore(tmp_dir, "right")
        a.load()
        b.load()
        a.add([make_chunk("1", "left-chunk", [1, 0, 0])])
        assert a.count() == 1
        assert b.count() == 0
