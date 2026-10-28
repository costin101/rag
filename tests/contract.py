"""Shared contract check: every path's output must pass `assert_valid_rag_response` (rule 1)."""

import json
import math
from collections.abc import Mapping

from normalizer import RAGResponse, is_blank

TOP_LEVEL = {"answer", "chunks", "sources", "usage", "metrics"}
CHUNK_KEYS = {"id", "text", "score", "source_id"}
SOURCE_KEYS = {"id", "title", "uri"}
USAGE_KEYS = {"input_tokens", "output_tokens"}
METRICS_KEYS = {"latency_ms"}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _keys(obj, expected: set, where: str) -> None:
    assert isinstance(obj, Mapping), f"{where} must be an object"
    assert set(obj) == expected, f"{where} keys {sorted(obj)} != {sorted(expected)}"


def _text(value, where: str) -> None:
    assert isinstance(value, str) and not is_blank(value), f"{where} must be non-blank text"


def assert_valid_rag_response(obj) -> None:
    """Raise AssertionError unless `obj` (RAGResponse or its dict form) honours the contract."""
    if isinstance(obj, RAGResponse):
        # dump, never trust attributes: model_copy/model_construct skip validation.
        data = obj.model_dump(mode="python", warnings=False)
    else:
        assert isinstance(obj, Mapping), "response must be a RAGResponse or a mapping"
        data = dict(obj)
    _keys(data, TOP_LEVEL, "response")

    _text(data["answer"], "answer")

    chunks = data["chunks"]
    assert isinstance(chunks, list) and chunks, "chunks must be a non-empty list"
    for i, chunk in enumerate(chunks):
        _keys(chunk, CHUNK_KEYS, f"chunks[{i}]")
        for key in ("id", "text", "source_id"):
            assert isinstance(chunk[key], str), f"chunks[{i}].{key} must be text"
        assert _is_real(chunk["score"]) and math.isfinite(chunk["score"]), (
            f"chunks[{i}].score must be a finite number"
        )

    sources = data["sources"]
    assert isinstance(sources, list) and sources, "sources must be a non-empty list"
    for i, source in enumerate(sources):
        _keys(source, SOURCE_KEYS, f"sources[{i}]")
        assert isinstance(source["id"], str), f"sources[{i}].id must be text"
        for key in ("title", "uri"):
            assert source[key] is None or isinstance(source[key], str), (
                f"sources[{i}].{key} must be text or null"
            )
    source_ids = [s["id"] for s in sources]
    assert len(set(source_ids)) == len(source_ids), "source ids must be unique"
    missing = {c["source_id"] for c in chunks} - set(source_ids)
    assert not missing, "every chunk source_id must appear in sources"

    _keys(data["usage"], USAGE_KEYS, "usage")
    for key in USAGE_KEYS:
        value = data["usage"][key]
        assert _is_int(value) and value >= 0, f"usage.{key} must be a non-negative integer"

    _keys(data["metrics"], METRICS_KEYS, "metrics")
    latency = data["metrics"]["latency_ms"]
    assert _is_real(latency) and math.isfinite(latency) and latency >= 0, (
        "metrics.latency_ms must be a finite number >= 0"
    )

    # The wire form must re-validate through the one model: catches whatever the checks above
    # do not spell out, and anything a validation bypass slipped in.
    try:
        wire = json.dumps(data, allow_nan=False)
        round_tripped = RAGResponse.model_validate_json(wire).model_dump(mode="python")
    except (ValueError, TypeError) as exc:
        raise AssertionError(f"response does not survive a JSON round trip: {type(exc).__name__}")
    assert round_tripped == json.loads(wire), "response changed in a JSON round trip"
