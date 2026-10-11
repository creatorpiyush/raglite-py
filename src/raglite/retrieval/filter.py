"""Metadata filters for search().

A filter maps field names to conditions; every field must match. A plain
value means equality, and a dict combines its operators (``in``, ``gt``,
``gte``, ``lt``, ``lte``) with AND. A chunk without the field never matches::

    {"source": "policy.pdf", "page": {"gte": 3, "lte": 5}, "tenant": {"in": ["acme", "globex"]}}
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..errors import ConfigError

# Metadata the chunker sets per chunk; filters on these run inside the store.
CHUNK_FIELDS = frozenset({"chunk", "totalChunks", "page", "pageEnd", "section"})
# Fields Document sets itself, so custom metadata may not use them.
RESERVED_FIELDS = CHUNK_FIELDS | {"source"}

_RANGE_OPS = ("gt", "gte", "lt", "lte")
_OPS = frozenset({"in", *_RANGE_OPS})


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bool)) or _is_number(value)


def _equals(value: Any, expected: Any) -> bool:
    # Python treats True == 1; the TypeScript SDK does not, so neither do we.
    if isinstance(value, bool) or isinstance(expected, bool):
        return isinstance(value, bool) and isinstance(expected, bool) and value == expected
    if isinstance(expected, str):
        return isinstance(value, str) and value == expected
    return _is_number(value) and value == expected


def validate_filter(filter: Any) -> Dict[str, Any]:
    """Raise ConfigError for anything that is not a valid metadata filter."""
    if not isinstance(filter, Mapping):
        raise ConfigError("filter must be an object of field conditions")
    for field, cond in filter.items():
        if _is_scalar(cond):
            continue
        if not isinstance(cond, Mapping):
            raise ConfigError(
                f"filter.{field} must be a string, number, boolean or an object of operators"
            )
        if not cond:
            raise ConfigError(f"filter.{field} has no operators")
        for op, value in cond.items():
            if op not in _OPS:
                raise ConfigError(
                    f'Unknown filter operator "{op}" in filter.{field}. Use in, gt, gte, lt or lte.'
                )
            if op == "in":
                if not isinstance(value, (list, tuple)) or not all(_is_scalar(v) for v in value):
                    raise ConfigError(
                        f"filter.{field}.in must be an array of strings, numbers or booleans"
                    )
            elif not _is_number(value):
                raise ConfigError(f"filter.{field}.{op} must be a number")
    return dict(filter)


def validate_custom_metadata(metadata: Any) -> Dict[str, Any]:
    """Raise ConfigError unless ``metadata`` is flat, scalar-valued and avoids reserved fields."""
    if not isinstance(metadata, Mapping):
        raise ConfigError("metadata must be an object")
    for field, value in metadata.items():
        if field in RESERVED_FIELDS:
            raise ConfigError(f"metadata.{field} is set by RAGLite; use another field name")
        if not _is_scalar(value):
            raise ConfigError(f"metadata.{field} must be a string, number or boolean")
    return dict(metadata)


def _condition_matches(value: Any, cond: Any) -> bool:
    if value is None:
        return False
    if _is_scalar(cond):
        return _equals(value, cond)
    if "in" in cond and not any(_equals(value, v) for v in cond["in"]):
        return False
    for op in _RANGE_OPS:
        if op not in cond:
            continue
        if not _is_number(value):
            return False
        bound = cond[op]
        if op == "gt" and not value > bound:
            return False
        if op == "gte" and not value >= bound:
            return False
        if op == "lt" and not value < bound:
            return False
        if op == "lte" and not value <= bound:
            return False
    return True


def matches_filter(metadata: Any, filter: Mapping[str, Any]) -> bool:
    """True when ``metadata`` (a dict or a ChunkMetadata) matches every field of ``filter``."""
    get = metadata.get if isinstance(metadata, Mapping) else (lambda f: getattr(metadata, f, None))
    return all(_condition_matches(get(field), cond) for field, cond in filter.items())


def split_filter(filter: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Split a filter into per-chunk fields and per-document fields (source and custom metadata)."""
    chunk = {f: c for f, c in filter.items() if f in CHUNK_FIELDS}
    document = {f: c for f, c in filter.items() if f not in CHUNK_FIELDS}
    return (chunk or None), document


def filter_conditions(filter: Mapping[str, Any]) -> List[Tuple[str, str, Any]]:
    """Each field's conditions as (field, operator, value), with "eq" for a plain value."""
    out: List[Tuple[str, str, Any]] = []
    for field, cond in filter.items():
        if _is_scalar(cond):
            out.append((field, "eq", cond))
            continue
        if "in" in cond:
            out.append((field, "in", list(cond["in"])))
        out.extend((field, op, cond[op]) for op in _RANGE_OPS if op in cond)
    return out
