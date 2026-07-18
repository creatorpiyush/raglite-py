"""
Unit tests for file loaders — mirrors tests/unit/loaders.test.ts
"""
import json
import os
import shutil
import tempfile
import pytest
from raglite.loaders import get_loader, TxtLoader, MarkdownLoader, JsonLoader
from raglite.errors import UnsupportedFileTypeError, LoaderError


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def write_file(directory, name, content, binary=False):
    path = os.path.join(directory, name)
    if binary:
        with open(path, "wb") as f:
            f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return path


class TestGetLoader:
    def test_raises_for_unsupported_extension(self, tmp_dir):
        path = write_file(tmp_dir, "file.xyz", "content")
        with pytest.raises(UnsupportedFileTypeError):
            get_loader(path)

    def test_returns_correct_loader_for_each_extension(self, tmp_dir):
        from raglite.loaders import TxtLoader, MarkdownLoader, JsonLoader, PdfLoader, DocxLoader
        path_txt = write_file(tmp_dir, "a.txt", "hello")
        path_md = write_file(tmp_dir, "b.md", "# Hi")
        path_json = write_file(tmp_dir, "c.json", "{}")
        assert isinstance(get_loader(path_txt), TxtLoader)
        assert isinstance(get_loader(path_md), MarkdownLoader)
        assert isinstance(get_loader(path_json), JsonLoader)


class TestTxtLoader:
    def test_loads_utf8_text(self, tmp_dir):
        path = write_file(tmp_dir, "hello.txt", "  Hello World  ")
        loader = TxtLoader(path)
        assert loader.load() == "Hello World"

    def test_raises_for_missing_file(self, tmp_dir):
        loader = TxtLoader(os.path.join(tmp_dir, "nonexistent.txt"))
        with pytest.raises(LoaderError):
            loader.load()


class TestMarkdownLoader:
    def test_loads_markdown_and_strips(self, tmp_dir):
        content = "  # Title\n\nSome **text** here.  "
        path = write_file(tmp_dir, "doc.md", content)
        loader = MarkdownLoader(path)
        assert loader.load() == "# Title\n\nSome **text** here."


class TestJsonLoader:
    def test_flattens_flat_object(self, tmp_dir):
        data = {"name": "Alice", "age": 30}
        path = write_file(tmp_dir, "data.json", json.dumps(data))
        loader = JsonLoader(path)
        result = loader.load()
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_flattens_nested_object(self, tmp_dir):
        data = {"person": {"name": "Bob", "scores": [10, 20]}}
        path = write_file(tmp_dir, "nested.json", json.dumps(data))
        loader = JsonLoader(path)
        result = loader.load()
        assert "person.name: Bob" in result
        assert "person.scores[0]: 10" in result
        assert "person.scores[1]: 20" in result

    def test_handles_null_values(self, tmp_dir):
        data = {"key": None}
        path = write_file(tmp_dir, "null.json", json.dumps(data))
        loader = JsonLoader(path)
        assert "key: null" in loader.load()

    def test_handles_empty_array(self, tmp_dir):
        data = {"items": []}
        path = write_file(tmp_dir, "empty.json", json.dumps(data))
        loader = JsonLoader(path)
        assert "items: []" in loader.load()

    def test_raises_for_invalid_json(self, tmp_dir):
        path = write_file(tmp_dir, "bad.json", "{not valid json}")
        with pytest.raises(LoaderError):
            JsonLoader(path).load()
