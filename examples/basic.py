import os

from raglite import Document


def main():
    # Resolve the file path to sample.txt relative to this script
    sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")

    # Initialize Document with local embeddings and OpenAI LLM (falls back to mock if key is missing)
    doc = Document(
        sample_path,
        {
            "embeddings": {
                "provider": "local"  # runs locally offline using sentence-transformers
            },
            "llm": {
                "provider": "openai",
                "apiKey": os.environ.get("OPENAI_API_KEY", "mock-key"),
            },
        },
    )

    print("Building semantic index...")
    build = doc.build()
    print("Index built:", build)

    print("\nSearching for 'refund policy'...")
    hits = doc.search("refund policy", top_k=3)
    for hit in hits:
        print(f"  [{hit.metadata.chunk}] score={hit.score:.3f}")
        print(f"    {hit.text[:120]}...")

    # Only attempt to ask if OPENAI_API_KEY is actually configured
    if "OPENAI_API_KEY" in os.environ:
        print("\nAsking: 'What is the refund policy?'")
        answer = doc.ask("What is the refund policy?")
        print("Answer:", answer.text)
        print("Usage:", answer.usage)
    else:
        print(
            "\nNote: Set the OPENAI_API_KEY environment variable to test the LLM 'ask' question-answering."
        )


if __name__ == "__main__":
    main()
