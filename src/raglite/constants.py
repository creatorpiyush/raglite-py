import importlib.metadata

try:
    PACKAGE_NAME = "raglite-toolkit"
    PACKAGE_VERSION = importlib.metadata.version("raglite-toolkit")
except importlib.metadata.PackageNotFoundError:
    PACKAGE_NAME = "raglite-toolkit"
    PACKAGE_VERSION = "1.0.2"  # local development fallback


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
