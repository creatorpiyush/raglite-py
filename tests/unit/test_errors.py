"""
Unit tests for errors — mirrors tests/unit/errors.test.ts
"""
from raglite.errors import (
    ChunkingError,
    ConfigError,
    EmbeddingError,
    FileNotIndexedError,
    LLMError,
    LoaderError,
    RagLiteError,
    UnsupportedFileTypeError,
    VectorDBError,
    import_optional,
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


class TestImportOptional:
    def test_missing_module_raises_config_error_naming_the_extra(self):
        import pytest

        with pytest.raises(ConfigError, match=r"pip install 'raglite-toolkit\[thing\]'"):
            import_optional("raglite_no_such_module", "thing", "Thing")

    def test_returns_installed_module(self):
        import json

        assert import_optional("json", "x", "x") is json
