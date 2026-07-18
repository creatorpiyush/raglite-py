import sys
import json
import argparse
import time
from typing import Optional

from .constants import PACKAGE_VERSION
from .core.document import Document

HELP = f"""raglite v{PACKAGE_VERSION}

Usage:
  raglite index <file>   [--chunk-size N] [--overlap N] [--embed-provider P] [--embed-model M] [--embed-key K] [--rebuild]
  raglite search <file> "query"   [--top-k N]
  raglite ask <file> "question"   --llm-provider P [--llm-model M] [--llm-key K] [--stream]
  raglite serve <file>            --llm-provider P [--llm-key K] [--host H] [--port N] [--token T]
  raglite --help
  raglite --version

Providers:
  LLM:        openai, anthropic, google, mistral, cohere, groq, xai, ollama
  Embeddings: openai, google, mistral, cohere, voyage, ollama, local
"""


def parse_common_embedding(args_dict: dict) -> dict:
    provider = args_dict.get("embed_provider") or "local"
    config = {"provider": provider}
    if args_dict.get("embed_model"):
        config["model"] = args_dict["embed_model"]
    if args_dict.get("embed_key"):
        config["apiKey"] = args_dict["embed_key"]
    return config


def parse_llm(args_dict: dict) -> Optional[dict]:
    provider = args_dict.get("llm_provider")
    if not provider:
        return None
    config = {"provider": provider}
    if args_dict.get("llm_model"):
        config["model"] = args_dict["llm_model"]
    if args_dict.get("llm_key"):
        config["apiKey"] = args_dict["llm_key"]
    return config


def run_index(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("file")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--embed-provider")
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-key")
    parser.add_argument("--rebuild", action="store_true")

    parsed = parser.parse_args(args)

    embeddings = parse_common_embedding(vars(parsed))
    doc = Document(parsed.file, {"embeddings": embeddings})

    build_opts = {}
    if parsed.chunk_size is not None:
        build_opts["chunkSize"] = parsed.chunk_size
    if parsed.overlap is not None:
        build_opts["overlap"] = parsed.overlap
    if parsed.rebuild:
        build_opts["rebuild"] = True

    result = doc.build(build_opts)
    sys.stdout.write(f"{json.dumps(result, indent=2)}\n")


def run_search(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("file")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--embed-provider")
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-key")

    parsed = parser.parse_args(args)

    embeddings = parse_common_embedding(vars(parsed))
    doc = Document(parsed.file, {"embeddings": embeddings})

    search_opts = {}
    if parsed.top_k is not None:
        search_opts["topK"] = parsed.top_k

    results = doc.search(parsed.query, search_opts)
    serialized = [r.model_dump(by_alias=True) for r in results]
    sys.stdout.write(f"{json.dumps(serialized, indent=2)}\n")


def run_ask(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("file")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--embed-provider")
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-key")
    parser.add_argument("--llm-provider", required=True)
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-key")
    parser.add_argument("--stream", action="store_true")

    parsed = parser.parse_args(args)

    embeddings = parse_common_embedding(vars(parsed))
    llm = parse_llm(vars(parsed))

    doc = Document(parsed.file, {"embeddings": embeddings, "llm": llm})

    opts = {}
    if parsed.top_k is not None:
        opts["topK"] = parsed.top_k

    if parsed.stream:
        for chunk in doc.ask_stream(parsed.question, opts):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        answer = doc.ask(parsed.question, opts)
        sys.stdout.write(f"{answer.text}\n")


def run_serve(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("file")
    parser.add_argument("--embed-provider")
    parser.add_argument("--embed-model")
    parser.add_argument("--embed-key")
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-key")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--token")

    parsed = parser.parse_args(args)

    embeddings = parse_common_embedding(vars(parsed))
    llm = parse_llm(vars(parsed))

    doc = Document(
        parsed.file,
        {"embeddings": embeddings, **({"llm": llm} if llm else {})},
    )
    doc.build()

    serve_opts = {}
    if llm:
        serve_opts["llm"] = llm
    if parsed.host:
        serve_opts["host"] = parsed.host
    if parsed.port is not None:
        serve_opts["port"] = parsed.port
    if parsed.token:
        serve_opts["bearerToken"] = parsed.token

    handle = doc.serve(serve_opts)
    sys.stdout.write(f"RagLite listening on {handle.url}\n")
    sys.stdout.flush()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handle.close()
        sys.exit(0)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("--help", "-h", "help"):
        sys.stdout.write(HELP)
        sys.stdout.flush()
        sys.exit(0)
    if argv[0] in ("--version", "-v", "version"):
        sys.stdout.write(f"{PACKAGE_VERSION}\n")
        sys.stdout.flush()
        sys.exit(0)

    command = argv[0]
    args = argv[1:]

    try:
        if command == "index":
            run_index(args)
        elif command == "search":
            run_search(args)
        elif command == "ask":
            run_ask(args)
        elif command == "serve":
            run_serve(args)
        else:
            sys.stderr.write(f"Unknown command: {command}\n\n{HELP}")
            sys.stderr.flush()
            sys.exit(2)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
