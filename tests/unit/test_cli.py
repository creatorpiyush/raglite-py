"""
Unit tests for the CLI — mirrors tests/unit/cli.test.ts
"""
import json
import os
import shutil
import sys
import tempfile
import pytest
from io import StringIO
from unittest.mock import MagicMock, patch


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="raglite_cli_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_txt(tmp_dir):
    path = os.path.join(tmp_dir, "doc.txt")
    content = " ".join(f"word{i}" for i in range(150))
    with open(path, "w") as f:
        f.write(content)
    return path


def run_main(argv, expect_exit=None):
    """Run cli.main() with the given argv, capturing stdout/stderr."""
    from raglite.cli import main

    captured_out = StringIO()
    captured_err = StringIO()

    with patch("sys.argv", ["raglite"] + argv), \
         patch("sys.stdout", captured_out), \
         patch("sys.stderr", captured_err):
        if expect_exit is not None:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == expect_exit
        else:
            main()

    return captured_out.getvalue(), captured_err.getvalue()


class TestCliHelp:
    def test_no_args_prints_help(self):
        out, _ = run_main([], expect_exit=0)
        assert "raglite v" in out
        assert "index" in out
        assert "search" in out
        assert "ask" in out
        assert "serve" in out

    def test_help_flag(self):
        out, _ = run_main(["--help"], expect_exit=0)
        assert "raglite v" in out

    def test_version_flag(self):
        from raglite.constants import PACKAGE_VERSION
        out, _ = run_main(["--version"], expect_exit=0)
        assert PACKAGE_VERSION in out

    def test_unknown_command_exits_2(self):
        _, err = run_main(["unknowncmd"], expect_exit=2)
        assert "Unknown command" in err


class TestCliIndex:
    def unit_vec_384(self):
        return [1.0] + [0.0] * 383

    def test_index_command_outputs_json(self, sample_txt, tmp_dir):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        ):
            out, _ = run_main([
                "index", sample_txt,
                "--embed-provider", "local",
            ])
        data = json.loads(out)
        assert data["chunkCount"] > 0
        assert data["cached"] is False
        assert data["embeddingProvider"] == "local"

    def test_index_rebuild_flag(self, sample_txt, tmp_dir):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        ):
            # First build
            run_main(["index", sample_txt, "--embed-provider", "local"])
            # Second build with rebuild flag forces fresh index
            out, _ = run_main(["index", sample_txt, "--embed-provider", "local", "--rebuild"])
        data = json.loads(out)
        assert data["cached"] is False


class TestCliSearch:
    def test_search_command_returns_json_array(self, sample_txt, tmp_dir):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=lambda text: [1.0] + [0.0] * 383,
        ):
            # Build index first
            run_main(["index", sample_txt, "--embed-provider", "local"])
            # Search
            out, _ = run_main(["search", sample_txt, "word5", "--embed-provider", "local"])
        results = json.loads(out)
        assert isinstance(results, list)
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r


class TestCliParsing:
    def test_parse_common_embedding_defaults(self):
        from raglite.cli import parse_common_embedding
        result = parse_common_embedding({})
        assert result["provider"] == "local"

    def test_parse_common_embedding_with_key(self):
        from raglite.cli import parse_common_embedding
        result = parse_common_embedding({
            "embed_provider": "openai",
            "embed_model": "text-embedding-3-small",
            "embed_key": "sk-test",
        })
        assert result["provider"] == "openai"
        assert result["model"] == "text-embedding-3-small"
        assert result["apiKey"] == "sk-test"

    def test_parse_llm_returns_none_without_provider(self):
        from raglite.cli import parse_llm
        assert parse_llm({}) is None

    def test_parse_llm_with_provider(self):
        from raglite.cli import parse_llm
        result = parse_llm({
            "llm_provider": "anthropic",
            "llm_model": "claude-3-5-sonnet-20241022",
            "llm_key": "sk-ant-test",
        })
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert result["apiKey"] == "sk-ant-test"
