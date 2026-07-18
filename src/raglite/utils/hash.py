import hashlib


def hash_string(input_str: str) -> str:
    """SHA-256 hash of an arbitrary string."""
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()


def hash_file(file_path: str) -> str:
    """Streaming SHA-256 of a file's contents."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def namespace_from_path(absolute_path: str) -> str:
    """A short, filesystem-safe namespace derived from a document's absolute path."""
    return hash_string(absolute_path)[:16]


# Aliases for TypeScript parity
hashString = hash_string
hashFile = hash_file
namespaceFromPath = namespace_from_path
