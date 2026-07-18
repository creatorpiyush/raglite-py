"""
Shared pytest configuration, fixtures, and helpers.
"""
import math
import os
import shutil
import tempfile
import pytest

# ── Ollama connectivity constants ────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_LLM_MODEL   = "gemma3:latest"
OLLAMA_EMBED_MODEL = "embeddinggemma:latest"


def _ollama_running() -> bool:
    """Return True if Ollama is reachable and both required models are pulled."""
    try:
        import urllib.request, json
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        available = {m["name"] for m in data.get("models", [])}
        return (OLLAMA_LLM_MODEL in available) and (OLLAMA_EMBED_MODEL in available)
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_available():
    """Session-scoped fixture — skips the test if Ollama is not reachable."""
    if not _ollama_running():
        pytest.skip(
            f"Ollama not reachable at {OLLAMA_BASE_URL} "
            f"or models '{OLLAMA_LLM_MODEL}' / '{OLLAMA_EMBED_MODEL}' not pulled."
        )
    return True


# ── Common fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleanup after each test."""
    d = tempfile.mkdtemp(prefix="raglite_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_txt(tmp_dir):
    """Create a sample .txt file."""
    path = os.path.join(tmp_dir, "sample.txt")
    content = " ".join(f"word{i}" for i in range(100))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_md(tmp_dir):
    """Create a sample .md file."""
    path = os.path.join(tmp_dir, "sample.md")
    content = "# Title\n\nThis is a sample markdown document with some content to chunk and embed."
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def unit_vec(vec):
    """Return L2-normalized vector."""
    s = sum(v * v for v in vec)
    n = math.sqrt(s)
    if n == 0:
        return list(vec)
    return [v / n for v in vec]
