"""
Unit tests for config — mirrors tests/unit/config.test.ts
"""
import pytest
from raglite.config import resolve_config, DocumentOptions
from raglite.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_STORE_DIRNAME,
)


class TestResolveConfig:
    def test_returns_defaults_with_no_options(self):
        cfg = resolve_config()
        assert cfg.chunkSize == DEFAULT_CHUNK_SIZE
        assert cfg.overlap == DEFAULT_CHUNK_OVERLAP
        assert cfg.topK == DEFAULT_TOP_K
        assert cfg.scoreThreshold == DEFAULT_SCORE_THRESHOLD
        assert cfg.storeDir == DEFAULT_STORE_DIRNAME
        assert cfg.embeddings.provider == "local"
        assert cfg.logLevel == "info"
        assert cfg.llm is None

    def test_overrides_are_applied(self):
        cfg = resolve_config({
            "chunkSize": 200,
            "overlap": 20,
            "topK": 3,
            "scoreThreshold": 0.5,
            "storeDir": "/tmp/mystore",
            "logLevel": "debug",
        })
        assert cfg.chunkSize == 200
        assert cfg.overlap == 20
        assert cfg.topK == 3
        assert cfg.scoreThreshold == 0.5
        assert cfg.storeDir == "/tmp/mystore"
        assert cfg.logLevel == "debug"
