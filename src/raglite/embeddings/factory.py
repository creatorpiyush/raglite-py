from typing import Union
from .base import Embedder
from .local import LocalEmbedder
from .remote import RemoteEmbedder
from ..types import EmbeddingProviderConfig


def create_embedder(config: Union[EmbeddingProviderConfig, dict]) -> Embedder:
    """Create an embedder instance from configuration."""
    if isinstance(config, dict):
        config = EmbeddingProviderConfig.model_validate(config)
    if config.provider == "local":
        return LocalEmbedder.create(config)
    return RemoteEmbedder.create(config)


# Alias for TS parity
createEmbedder = create_embedder
