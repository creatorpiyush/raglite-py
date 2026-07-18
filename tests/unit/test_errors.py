"""
Unit tests for errors — mirrors tests/unit/errors.test.ts
"""
import pytest
from raglite.errors import (
    RagLiteError,
    UnsupportedFileTypeError,
    FileNotIndexedError,
    LoaderError,
    ChunkingError,
    EmbeddingError,
    VectorDBError,
    LLMError,
    ConfigError,
)


class TestErrors:
    def test_raglite_error_is_exception(self):
        e = RagLiteError("base error")
        assert isinstance(e, Exception)
        assert str(e) == "base error"

    def test_subclasses_inherit_from_raglite_error(self):
        subclasses = [
            UnsupportedFileTypeError,
            FileNotIndexedError,
            LoaderError,
            ChunkingError,
            EmbeddingError,
            VectorDBError,
            LLMError,
            ConfigError,
        ]
        for cls in subclasses:
            e = cls("test message")
            assert isinstance(e, RagLiteError)
            assert isinstance(e, Exception)

    def test_cause_is_attached(self):
        cause = ValueError("original cause")
        e = LoaderError("wrapper", cause=cause)
        assert e.__cause__ is cause
