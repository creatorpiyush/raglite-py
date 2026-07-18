import os
import sys
import time
from raglite import Document


def main():
    # Resolve sample.txt path
    sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")

    # Document index with local offline embeddings
    doc = Document(sample_path, {"embeddings": {"provider": "local"}})

    print("Building semantic index...")
    doc.build()

    # Launch REST API server via FastAPI
    handle = doc.serve(
        {
            "port": 8085,
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "apiKey": os.environ.get("OPENAI_API_KEY", "mock-key"),
            },
            "bearerToken": os.environ.get("RAGFORGE_TOKEN", "change-me"),
            "requestLogging": True,
        }
    )

    print(f"\nServing on {handle.url}")
    print("Try running:")
    print(f"  curl {handle.url}/health")
    print(
        f"  curl -X POST {handle.url}/search "
        f"-H 'authorization: Bearer change-me' "
        f"-H 'content-type: application/json' "
        f"-d '{{\"query\":\"refunds\"}}'"
    )
    print("\nPress Ctrl+C to stop the server.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        handle.close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
