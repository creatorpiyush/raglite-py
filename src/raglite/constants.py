import importlib.metadata

try:
    PACKAGE_NAME = "raglite-toolkit"
    PACKAGE_VERSION = importlib.metadata.version("raglite-toolkit")
except importlib.metadata.PackageNotFoundError:
    PACKAGE_NAME = "raglite-toolkit"
    PACKAGE_VERSION = "1.2.2"  # local development fallback

# Version of the stored index layout (chunking, ids, payloads). The build cache
# is keyed on this instead of PACKAGE_VERSION so upgrading the package does not
# re-embed every index. Bump it only when a change makes existing indexes
# incompatible.
INDEX_FORMAT_VERSION = 1

# Releases that wrote format-1 indexes before ``formatVersion`` was recorded.
LEGACY_FORMAT_1_VERSIONS = frozenset({"1.2.1"})


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".json", ".md", ".markdown", ".docx"}

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.0

DEFAULT_TEMPERATURE = 0.0

DEFAULT_STORE_DIRNAME = ".raglite"
DEFAULT_COLLECTION_NAME = "default"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8085
