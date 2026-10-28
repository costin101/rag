import copy

import pytest

from normalizer import (
    ContextChunk,
    Generation,
    Metrics,
    RAGResponse,
    Source,
    Usage,
    from_generation,
)
from tests.contract import assert_valid_rag_response


def chunk(cid="c1", source_id="s1", score=0.5):
    return ContextChunk(id=cid, text="text", score=score, source_id=source_id)


def valid_response():
    return from_generation(
        Generation(text="answer", input_tokens=3, output_tokens=2),
        [chunk("c1", "s1"), chunk("c2", "s2"), chunk("c3", "s1")],
        12.5,
    )


def valid_dict():
    return valid_response().model_dump()


def test_accepts_a_model():
    assert_valid_rag_response(valid_response())


def test_accepts_a_dict():
    assert_valid_rag_response(valid_dict())


def test_accepts_the_json_wire_form():
    import json

    assert_valid_rag_response(json.loads(valid_response().model_dump_json()))


def test_accepts_integer_valued_json_numbers():
    d = valid_dict()
    d["chunks"][0]["score"] = 1
    d["metrics"]["latency_ms"] = 0
    assert_valid_rag_response(d)


def mutate(fn):
    d = copy.deepcopy(valid_dict())
    fn(d)
    return d


def _set(path, value):
    def apply(d):
        cur = d
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = value

    return apply


def _drop(path):
    def apply(d):
        cur = d
        for key in path[:-1]:
            cur = cur[key]
        del cur[path[-1]]

    return apply


BAD = {
    "extra top-level path": _set(("path",), "api"),
    "extra top-level provider": _set(("provider",), "anthropic"),
    "missing answer": _drop(("answer",)),
    "missing metrics": _drop(("metrics",)),
    "blank answer": _set(("answer",), "   "),
    "invisible answer": _set(("answer",), "​ㅤ"),
    "non-string answer": _set(("answer",), 5),
    "no chunks": _set(("chunks",), []),
    "chunks not a list": _set(("chunks",), "x"),
    "chunk extra key": _set(("chunks", 0, "extra"), 1),
    "chunk missing key": _drop(("chunks", 0, "text")),
    "chunk id not text": _set(("chunks", 0, "id"), 7),
    "chunk text not text": _set(("chunks", 0, "text"), None),
    "chunk score nan": _set(("chunks", 0, "score"), float("nan")),
    "chunk score inf": _set(("chunks", 0, "score"), float("inf")),
    "chunk score bool": _set(("chunks", 0, "score"), True),
    "chunk score text": _set(("chunks", 0, "score"), "0.5"),
    "no sources": _set(("sources",), []),
    "sources not a list": _set(("sources",), None),
    "source extra key": _set(("sources", 0, "extra"), 1),
    "source missing id": _drop(("sources", 0, "id")),
    "source id not text": _set(("sources", 0, "id"), 1),
    "source with title": _set(("sources", 0, "title"), "Doc"),
    "source with uri": _set(("sources", 0, "uri"), "https://example.com/doc"),
    "duplicate source ids": lambda d: d["sources"].append(dict(d["sources"][0])),
    "chunk source not in sources": _set(("chunks", 0, "source_id"), "ghost"),
    "usage extra key": _set(("usage", "total"), 5),
    "usage missing key": _drop(("usage", "output_tokens")),
    "usage negative input": _set(("usage", "input_tokens"), -1),
    "usage negative output": _set(("usage", "output_tokens"), -1),
    "usage bool input": _set(("usage", "input_tokens"), True),
    "usage float input": _set(("usage", "input_tokens"), 1.5),
    "usage text output": _set(("usage", "output_tokens"), "2"),
    "metrics extra key": _set(("metrics", "x"), 1),
    "latency negative": _set(("metrics", "latency_ms"), -0.1),
    "latency nan": _set(("metrics", "latency_ms"), float("nan")),
    "latency inf": _set(("metrics", "latency_ms"), float("inf")),
    "latency bool": _set(("metrics", "latency_ms"), False),
    "latency text": _set(("metrics", "latency_ms"), "1"),
    "latency beyond the model bound": _set(("metrics", "latency_ms"), 1e12),
    "usage beyond the model bound": _set(("usage", "input_tokens"), 10**13),
}


# Only the final round trip can see the model's own upper bounds.
ROUND_TRIP_ONLY = {"latency beyond the model bound", "usage beyond the model bound"}


@pytest.mark.parametrize("name", sorted(BAD))
def test_rejects(name):
    with pytest.raises(AssertionError) as excinfo:
        assert_valid_rag_response(mutate(BAD[name]))
    # Each explicit check must fire on its own, not be masked by the round trip behind it.
    assert ("round trip" in str(excinfo.value)) == (name in ROUND_TRIP_ONLY)


@pytest.mark.parametrize("junk", [None, "x", 5, [], [("answer", "a")]])
def test_rejects_non_responses(junk):
    with pytest.raises(AssertionError):
        assert_valid_rag_response(junk)


# --- validation bypasses on real models --------------------------------------------


def test_rejects_a_model_copy_that_bypasses_validation():
    bad = valid_response().model_copy(update={"answer": "  "})
    with pytest.raises(AssertionError):
        assert_valid_rag_response(bad)


def test_rejects_a_model_copy_with_negative_usage():
    bad = valid_response().model_copy(update={"usage": Usage.model_construct(input_tokens=-1, output_tokens=0)})
    with pytest.raises(AssertionError):
        assert_valid_rag_response(bad)


def test_rejects_a_model_construct_with_nan_latency():
    bad = valid_response().model_copy(
        update={"metrics": Metrics.model_construct(latency_ms=float("nan"))}
    )
    with pytest.raises(AssertionError):
        assert_valid_rag_response(bad)


def test_rejects_a_model_construct_with_no_chunks():
    bad = RAGResponse.model_construct(
        answer="a", chunks=[], sources=[Source(id="s")],
        usage=Usage(input_tokens=1, output_tokens=1), metrics=Metrics(latency_ms=1.0),
    )
    with pytest.raises(AssertionError):
        assert_valid_rag_response(bad)


def test_rejects_a_model_with_an_orphan_chunk_source():
    bad = valid_response().model_copy(update={"sources": [Source(id="s1")]})
    with pytest.raises(AssertionError):
        assert_valid_rag_response(bad)


def test_rejects_a_model_whose_fields_hold_wrong_types():
    bad = valid_response().model_copy(update={"chunks": ["not a chunk"]})
    with pytest.raises(AssertionError):
        assert_valid_rag_response(bad)


def test_failure_messages_never_echo_values():
    d = valid_dict()
    d["answer"] = "LEAK-secret"
    d["usage"]["input_tokens"] = "LEAK-tokens"
    with pytest.raises(AssertionError) as excinfo:
        assert_valid_rag_response(d)
    assert "LEAK" not in str(excinfo.value)
