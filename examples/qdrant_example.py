"""
Qdrant Vector Store Example
============================
Run a local Qdrant instance first:
    docker run -p 6333:6333 qdrant/qdrant

Then run this example:
    python examples/qdrant_example.py
"""

import os

from raglite import Document

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")  # optional — for Qdrant Cloud


def main() -> None:
    print("=== Qdrant Vector Store Example ===")
    print(f"Targeting Qdrant at: {QDRANT_URL}\n")

    doc = Document(
        "./examples/sample.txt",
        {
            "embeddings": {"provider": "local"},
            "vectorStore": {
                "provider": "qdrant",
                "url": QDRANT_URL,
                **({"apiKey": QDRANT_KEY} if QDRANT_KEY else {}),
                "indexName": "raglite_demo",
            },
            "chunkSize": 50,
            "overlap": 10,
            "logLevel": "info",
        },
    )

    print("Building index in Qdrant...")
    result = doc.build(rebuild=True)
    print(f"Built: {result}\n")

    query = "shipping policy"
    print(f'Searching for: "{query}"')
    hits = doc.search(query, top_k=3)
    for i, h in enumerate(hits, 1):
        preview = h.text.replace("\n", " ")[:90]
        print(f"  #{i} score={h.score:.4f}  {preview}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
