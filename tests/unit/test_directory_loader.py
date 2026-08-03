import json
import shutil
import tempfile
from pathlib import Path

from raglite.loaders.directory import DirectoryLoader


def test_directory_loader_scans_and_loads():
    temp_dir = tempfile.mkdtemp()
    try:
        doc1 = Path(temp_dir) / "doc1.md"
        doc2 = Path(temp_dir) / "doc2.txt"
        sub_dir = Path(temp_dir) / "nested"
        sub_dir.mkdir(parents=True, exist_ok=True)
        doc3 = sub_dir / "doc3.json"
        unsupported = Path(temp_dir) / "unsupported.raw"

        doc1.write_text("# Policy\nRefunds within 30 days.", encoding="utf-8")
        doc2.write_text("Shipping takes 2 days.", encoding="utf-8")
        doc3.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        unsupported.write_text("unknown format data", encoding="utf-8")

        loader = DirectoryLoader(temp_dir)
        res = loader.load_files()

        assert len(res.loaded) == 3
        loaded_paths = [item.file_path for item in res.loaded]
        assert any(p.endswith("doc1.md") for p in loaded_paths)
        assert any(p.endswith("doc2.txt") for p in loaded_paths)
        assert any(p.endswith("doc3.json") for p in loaded_paths)

        assert len(res.errors) >= 1
        assert any(e.file_path.endswith("unsupported.raw") for e in res.errors)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_directory_loader_glob_filter():
    temp_dir = tempfile.mkdtemp()
    try:
        doc1 = Path(temp_dir) / "doc1.md"
        doc2 = Path(temp_dir) / "doc2.txt"
        doc1.write_text("MD content", encoding="utf-8")
        doc2.write_text("TXT content", encoding="utf-8")

        loader = DirectoryLoader(temp_dir, glob="*.md")
        res = loader.load_files()

        assert len(res.loaded) == 1
        assert res.loaded[0].file_path.endswith("doc1.md")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
