"""
Cross-SDK fixtures: the TypeScript SDK generates these expected outputs and
both SDKs must reproduce them. Refresh with scripts/sync_shared_fixtures.py.
"""
import json
from pathlib import Path

import pytest

from raglite.chunking import RecursiveChunker
from raglite.errors import ChunkingError
from raglite.utils.hash import hash_string, namespace_from_path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "shared"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CHUNKER = _load("chunker.json")
HASH = _load("hash.json")


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
