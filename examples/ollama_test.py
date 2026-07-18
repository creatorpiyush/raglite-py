"""
Ollama end-to-end test — uses locally running Ollama instance.

Models used (auto-detected from your Ollama install):
  Embeddings : embeddinggemma:latest  (307M, offline)
  LLM        : gemma3:latest          (4.3B, offline)

Requirements:
  - Ollama must be running: `ollama serve`
  - Models already pulled (detected on your machine):
      ollama pull embeddinggemma
      ollama pull gemma3

Run:
  python examples/ollama_test.py
"""

import os
import sys
import time

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "embeddinggemma:latest"
LLM_MODEL       = "gemma3:latest"
SAMPLE_FILE     = os.path.join(os.path.dirname(__file__), "sample.txt")


def check_ollama():
    """Verify Ollama is reachable before running tests."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            import json
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception as e:
        print(f"[ERROR] Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        print("       Make sure Ollama is running: ollama serve")
        sys.exit(1)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── 0. Preflight ─────────────────────────────────────────────
    separator("PREFLIGHT — Checking Ollama")
    available = check_ollama()
    print(f"✓ Ollama is running. Available models: {available}")

    for required in [EMBED_MODEL, LLM_MODEL]:
        if required not in available:
            print(f"[WARN] Model '{required}' not found. Pull it with: ollama pull {required.split(':')[0]}")

    # ── 1. Imports ───────────────────────────────────────────────
    from raglite import Document

    # ── 2. Build Index ───────────────────────────────────────────
    separator("TEST 1 — Build index with Ollama embeddings")
    print(f"File      : {SAMPLE_FILE}")
    print(f"Embed model: {EMBED_MODEL}")

    doc = Document(SAMPLE_FILE, {
        "embeddings": {
            "provider": "ollama",
            "model":    EMBED_MODEL,
            "baseURL":  OLLAMA_BASE_URL,
        },
        "llm": {
            "provider": "ollama",
            "model":    LLM_MODEL,
            "baseURL":  OLLAMA_BASE_URL,
        },
        "chunkSize": 100,
        "overlap":   10,
        "storeDir":  ".raglite_ollama_test",
        "rebuild":   True,   # always fresh for the test
    })

    t0 = time.time()
    result = doc.build({"rebuild": True})
    elapsed = round(time.time() - t0, 2)

    print(f"\n✓ Index built in {elapsed}s")
    print(f"  chunkCount       : {result['chunkCount']}")
    print(f"  embeddingProvider: {result['embeddingProvider']}")
    print(f"  embeddingModel   : {result['embeddingModel']}")
    print(f"  dimensions       : {result['dimensions']}")
    print(f"  cached           : {result['cached']}")

    assert result["chunkCount"] > 0,            "Expected at least 1 chunk"
    assert result["embeddingProvider"] == "ollama", "Expected 'ollama' provider"
    assert result["cached"] is False,           "Expected fresh index (rebuild=True)"
    assert result["dimensions"] is not None,    "Expected dimension count"
    print("\n  ✅ PASS")

    # ── 3. Cache Reuse ───────────────────────────────────────────
    separator("TEST 2 — Cache reuse (no rebuild)")
    doc2 = Document(SAMPLE_FILE, {
        "embeddings": {"provider": "ollama", "model": EMBED_MODEL, "baseURL": OLLAMA_BASE_URL},
        "llm":        {"provider": "ollama", "model": LLM_MODEL,   "baseURL": OLLAMA_BASE_URL},
        "chunkSize":  100,
        "overlap":    10,
        "storeDir":   ".raglite_ollama_test",
    })
    result2 = doc2.build()
    print(f"  cached     : {result2['cached']}")
    print(f"  chunkCount : {result2['chunkCount']}")
    assert result2["cached"] is True, "Expected cache hit on second build"
    print("\n  ✅ PASS")

    # ── 4. Semantic Search ────────────────────────────────────────
    separator("TEST 3 — Semantic search")
    queries = [
        "refund policy",
        "shipping delivery time",
        "warranty coverage",
        "privacy personal data",
    ]

    for q in queries:
        t0 = time.time()
        hits = doc.search(q, top_k=2)
        elapsed = round(time.time() - t0, 2)
        print(f"\n  Query: \"{q}\"  ({elapsed}s, {len(hits)} hit(s))")
        for hit in hits:
            print(f"    [{hit.metadata.chunk}/{hit.metadata.totalChunks}]"
                  f" score={hit.score:.3f}  — {hit.text[:80]}...")

    assert len(doc.search("refund policy", top_k=3)) > 0, "Expected search results"
    print("\n  ✅ PASS")

    # ── 5. Ask (RAG Q&A) ─────────────────────────────────────────
    separator("TEST 4 — Ask (RAG Q&A with Ollama LLM)")
    print(f"LLM: {LLM_MODEL}\n")

    questions = [
        "What is the refund policy?",
        "How long does standard shipping take?",
        "What does the warranty cover?",
    ]

    for question in questions:
        print(f"  Q: {question}")
        t0 = time.time()
        answer = doc.ask(question, {"topK": 3})
        elapsed = round(time.time() - t0, 2)
        print(f"  A: {answer.text.strip()}")
        print(f"     ({elapsed}s | model={answer.model} | tokens={answer.usage})")
        print()

    assert answer.provider == "ollama", "Expected ollama provider in answer"
    assert len(answer.text.strip()) > 0, "Expected non-empty answer"
    print("  ✅ PASS")

    # ── 6. Streaming ─────────────────────────────────────────────
    separator("TEST 5 — Streaming answer")
    print("  Q: Give me a one-sentence summary of this document.\n  A: ", end="", flush=True)
    t0 = time.time()
    chunks_received = 0
    for chunk in doc.ask_stream("Give me a one-sentence summary of this document.", {"topK": 3}):
        print(chunk, end="", flush=True)
        chunks_received += 1
    elapsed = round(time.time() - t0, 2)
    print(f"\n\n     ({elapsed}s | {chunks_received} stream chunks received)")
    assert chunks_received > 0, "Expected streaming chunks"
    print("\n  ✅ PASS")

    # ── Summary ───────────────────────────────────────────────────
    separator("ALL TESTS PASSED ✅")
    print(f"  Embed model : {EMBED_MODEL}")
    print(f"  LLM model   : {LLM_MODEL}")
    print(f"  Chunks      : {doc.chunk_count}")
    print()

    # Cleanup test store
    import shutil
    shutil.rmtree(".raglite_ollama_test", ignore_errors=True)


if __name__ == "__main__":
    main()
