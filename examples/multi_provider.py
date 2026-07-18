"""
Multi-provider example — port of examples/multi-provider.ts

Runs the same question against every configured LLM provider, skipping any
that don't have an API key set.  Local offline embeddings are always used so
no embedding key is required.

Usage:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_GENERATIVE_AI_API_KEY=AI...
    export MISTRAL_API_KEY=...
    export COHERE_API_KEY=...
    export GROQ_API_KEY=...

    python examples/multi_provider.py
"""
import os
import time
from raglite import Document, LLMProviderConfig

QUESTION = "Summarize the document in three bullet points."

PROVIDERS: list[dict] = [
    {"provider": "openai",    "model": "gpt-4o-mini",                 "apiKey": os.environ.get("OPENAI_API_KEY")},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022",  "apiKey": os.environ.get("ANTHROPIC_API_KEY")},
    {"provider": "google",    "model": "gemini-2.0-flash",            "apiKey": os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")},
    {"provider": "mistral",   "model": "mistral-large-latest",        "apiKey": os.environ.get("MISTRAL_API_KEY")},
    {"provider": "cohere",    "model": "command-r-plus",              "apiKey": os.environ.get("COHERE_API_KEY")},
    {"provider": "groq",      "model": "llama-3.3-70b-versatile",     "apiKey": os.environ.get("GROQ_API_KEY")},
]


def main():
    sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")

    # Build once with local offline embeddings
    doc = Document(sample_path, {"embeddings": {"provider": "local"}})
    print("Building index with local embeddings...")
    doc.build()
    print("Index ready.\n")

    for llm in PROVIDERS:
        label = f"{llm['provider']} ({llm['model']})"
        if not llm.get("apiKey"):
            print(f"== {label} skipped (no API key) ==")
            continue
        try:
            start = time.time()
            answer = doc.ask(QUESTION, {"llm": llm, "topK": 5})
            elapsed = int((time.time() - start) * 1000)
            print(f"\n== {label} in {elapsed}ms ==")
            print(answer.text)
        except Exception as exc:
            print(f"\n== {label} FAILED ==")
            print(str(exc))


if __name__ == "__main__":
    main()
