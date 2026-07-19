"""
Unit tests for hash utilities — mirrors tests/unit/hash.test.ts
"""
import os
import shutil
import tempfile

import pytest

from raglite.utils.hash import hash_file, hash_string, namespace_from_path


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestHashString:
    def test_returns_hex_string(self):
        result = hash_string("hello")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_is_deterministic(self):
        assert hash_string("raglite") == hash_string("raglite")

    def test_different_inputs_produce_different_hashes(self):
        assert hash_string("abc") != hash_string("xyz")


class TestHashFile:
    def test_returns_hex_string_for_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.txt")
        with open(path, "w") as f:
            f.write("file contents here")
        result = hash_file(path)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_changes_when_content_changes(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.txt")
        with open(path, "w") as f:
            f.write("version 1")
        h1 = hash_file(path)
        with open(path, "w") as f:
            f.write("version 2")
        h2 = hash_file(path)
        assert h1 != h2

    def test_matches_hash_string_for_same_content(self, tmp_dir):
        content = "deterministic test"
        path = os.path.join(tmp_dir, "data.txt")
        with open(path, "w") as f:
            f.write(content)
        assert hash_file(path) == hash_string(content)


class TestNamespaceFromPath:
    def test_returns_16_char_hex_string(self):
        ns = namespace_from_path("/absolute/path/to/doc.pdf")
        assert isinstance(ns, str)
        assert len(ns) == 16
        assert all(c in "0123456789abcdef" for c in ns)

    def test_different_paths_produce_different_namespaces(self):
        ns1 = namespace_from_path("/path/to/a.pdf")
        ns2 = namespace_from_path("/path/to/b.pdf")
        assert ns1 != ns2

    def test_same_path_always_produces_same_namespace(self):
        path = "/Users/test/documents/policy.pdf"
        assert namespace_from_path(path) == namespace_from_path(path)
