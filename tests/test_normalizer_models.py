import copy
import inspect
import json
import math
import pickle

import pytest
from pydantic import BaseModel, ValidationError

from normalizer import (
    ConfigError,
    ContextChunk,
    EmptyRetrievalError,
    ErrorCode,
    InvalidQueryError,
    Metrics,
    ProviderError,
    ProviderPayloadError,
    RAGError,
    RAGResponse,
    RetrievalError,
    Source,
    Usage,
)


def chunk(**kw):
    return {"id": "c1", "text": "some text", "score": 0.5, "source_id": "s1", **kw}


def response(**kw):
    data = {
        "answer": "an answer",
        "chunks": [chunk()],
        "sources": [{"id": "s1"}],
        "usage": {"input_tokens": 3, "output_tokens": 4},
        "metrics": {"latency_ms": 12.5},
    }
    data.update(kw)
    return data


# --- RAGResponse -------------------------------------------------------------


def test_valid_response_roundtrips_through_json():
    r = RAGResponse.model_validate(response())
    assert RAGResponse.model_validate_json(r.model_dump_json()) == r
    assert json.loads(r.model_dump_json())["usage"] == {"input_tokens": 3, "output_tokens": 4}


def test_response_has_exactly_the_contract_keys():
    assert set(RAGResponse.model_fields) == {"answer", "chunks", "sources", "usage", "metrics"}
    assert set(RAGResponse.model_validate(response()).model_dump()) == {
        "answer",
        "chunks",
        "sources",
        "usage",
        "metrics",
    }


@pytest.mark.parametrize("field", ["path", "provider"])
def test_path_specific_fields_are_rejected(field):
    with pytest.raises(ValidationError):
        RAGResponse.model_validate(response(**{field: "api"}))


@pytest.mark.parametrize(
    "blank",
    [
        "",
        " ",
        "\n\t ",
        "\u200b",  # zero-width space
        "\ufeff",  # BOM / zero-width no-break space
        "\u00a0",  # no-break space
        "\u2003\u3000",  # em space, ideographic space
        "\u200b \u200d\u2060",
        "\x00",  # Cc
        "\x7f",  # Cc
        "\u0085",  # Cc (NEL)
        "\u3164",  # Hangul filler
        "\u2800",  # Braille blank
        "\u034f",  # combining grapheme joiner
        "\u115f\u1160",  # Hangul choseong/jungseong fillers
        "\uffa0",  # halfwidth Hangul filler
        "\u17b4\u17b5",  # Khmer inherent vowels
        "\u180e",  # Mongolian vowel separator
        "\u2062\u2063\u2064",  # invisible operators
        "\ufe0f",  # variation selector
        "\u2028\u2029",  # line/paragraph separators
    ],
)
def test_blank_answer_rejected(blank):
    with pytest.raises(ValidationError):
        RAGResponse.model_validate(response(answer=blank))


def test_answer_with_visible_text_among_invisibles_is_kept():
    r = RAGResponse.model_validate(response(answer="\u200b hi \u200b"))
    assert r.answer == "\u200b hi \u200b"


def test_empty_chunks_and_sources_are_valid_at_model_level():
    # Emptiness is a pipeline concern (empty_retrieval); the model only pins shape.
    r = RAGResponse.model_validate(response(chunks=[], sources=[]))
    assert r.chunks == [] and r.sources == []


def test_freezing_is_shallow_lists_stay_mutable():
    # Documents (does not endorse) the limitation: frozen blocks attribute assignment only.
    r = RAGResponse.model_validate(response())
    r.chunks.append(ContextChunk(**chunk(id="c2")))
    assert len(r.chunks) == 2


def test_non_string_answer_rejected():
    with pytest.raises(ValidationError):
        RAGResponse.model_validate(response(answer=None))


@pytest.mark.parametrize("missing", ["answer", "chunks", "sources", "usage", "metrics"])
def test_missing_top_level_field_rejected(missing):
    data = response()
    del data[missing]
    with pytest.raises(ValidationError):
        RAGResponse.model_validate(data)


def test_response_is_frozen():
    r = RAGResponse.model_validate(response())
    with pytest.raises(ValidationError):
        r.answer = "other"


def test_nested_models_are_validated():
    with pytest.raises(ValidationError):
        RAGResponse.model_validate(response(chunks=[chunk(score=math.nan)]))


# --- ContextChunk ------------------------------------------------------------


def test_chunk_valid():
    c = ContextChunk(**chunk())
    assert (c.id, c.text, c.score, c.source_id) == ("c1", "some text", 0.5, "s1")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_chunk_score_must_be_finite(bad):
    with pytest.raises(ValidationError):
        ContextChunk(**chunk(score=bad))


def test_chunk_accepts_negative_and_integer_scores():
    assert ContextChunk(**chunk(score=-0.25)).score == -0.25
    assert ContextChunk(**chunk(score=1)).score == 1.0


def test_chunk_score_rejects_bool():
    with pytest.raises(ValidationError):
        ContextChunk(**chunk(score=True))


def test_chunk_extra_forbidden_and_frozen():
    with pytest.raises(ValidationError):
        ContextChunk(**chunk(extra="x"))
    with pytest.raises(ValidationError):
        ContextChunk(**chunk()).score = 1.0


@pytest.mark.parametrize("missing", ["id", "text", "score", "source_id"])
def test_chunk_required_fields(missing):
    data = chunk()
    del data[missing]
    with pytest.raises(ValidationError):
        ContextChunk(**data)


# --- Source ------------------------------------------------------------------


def test_source_is_only_an_id():
    assert Source(id="s1").model_dump() == {"id": "s1"}


@pytest.mark.parametrize("field", ["title", "uri"])
def test_source_rejects_dropped_title_and_uri(field):
    with pytest.raises(ValidationError):
        Source(id="s1", **{field: "https://example.com/a"})


def test_source_extra_forbidden_and_frozen():
    with pytest.raises(ValidationError):
        Source(id="s1", url="https://example.com")
    with pytest.raises(ValidationError):
        Source(id="s1").id = "x"


# --- Usage -------------------------------------------------------------------


def test_usage_valid_including_zero():
    u = Usage(input_tokens=0, output_tokens=7)
    assert (u.input_tokens, u.output_tokens) == (0, 7)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("bad", [True, False, -1, 1.5, 2.0, "3", None, math.nan])
def test_usage_rejects_non_strict_or_negative_ints(field, bad):
    data = {"input_tokens": 1, "output_tokens": 1, field: bad}
    with pytest.raises(ValidationError):
        Usage(**data)


def test_usage_upper_bound_inclusive():
    assert Usage(input_tokens=10**12, output_tokens=10**12).input_tokens == 10**12
    with pytest.raises(ValidationError):
        Usage(input_tokens=10**12 + 1, output_tokens=0)
    with pytest.raises(ValidationError):
        Usage(input_tokens=0, output_tokens=10**12 + 1)


def test_usage_extra_forbidden_and_frozen():
    with pytest.raises(ValidationError):
        Usage(input_tokens=1, output_tokens=1, total_tokens=2)
    with pytest.raises(ValidationError):
        Usage(input_tokens=1, output_tokens=1).input_tokens = 2


def test_usage_json_roundtrip_stays_strict():
    assert Usage.model_validate_json('{"input_tokens": 1, "output_tokens": 2}')
    with pytest.raises(ValidationError):
        Usage.model_validate_json('{"input_tokens": true, "output_tokens": 2}')


# --- Metrics -----------------------------------------------------------------


def test_metrics_valid():
    assert Metrics(latency_ms=0).latency_ms == 0.0
    assert Metrics(latency_ms=1.5).latency_ms == 1.5


@pytest.mark.parametrize(
    "bad", [-0.1, math.nan, math.inf, -math.inf, "fast", "5", None, True, False]
)
def test_metrics_rejects_negative_nonfinite_or_non_numeric(bad):
    with pytest.raises(ValidationError):
        Metrics(latency_ms=bad)


@pytest.mark.parametrize("raw", ["true", "false", '"5"', "null"])
def test_metrics_json_is_strict(raw):
    with pytest.raises(ValidationError):
        Metrics.model_validate_json('{"latency_ms": ' + raw + "}")


def test_metrics_json_accepts_numbers():
    assert Metrics.model_validate_json('{"latency_ms": 5}').latency_ms == 5.0


def test_metrics_upper_bound_inclusive():
    assert Metrics(latency_ms=1e9).latency_ms == 1e9
    with pytest.raises(ValidationError):
        Metrics(latency_ms=1e9 + 1)


def test_metrics_extra_forbidden_and_frozen():
    with pytest.raises(ValidationError):
        Metrics(latency_ms=1, path="api")
    with pytest.raises(ValidationError):
        Metrics(latency_ms=1).latency_ms = 2


# --- ErrorCode and RAGError --------------------------------------------------


def test_error_code_is_the_closed_public_set():
    assert {c.name: c.value for c in ErrorCode} == {
        "CONFIG_INVALID": "config_invalid",
        "INVALID_QUERY": "invalid_query",
        "RETRIEVAL_FAILED": "retrieval_failed",
        "EMPTY_RETRIEVAL": "empty_retrieval",
        "PROVIDER_FAILED": "provider_failed",
        "PROVIDER_PAYLOAD_INVALID": "provider_payload_invalid",
    }


def test_error_code_is_a_str():
    assert ErrorCode.CONFIG_INVALID == "config_invalid"
    assert f"{ErrorCode.CONFIG_INVALID}" == "config_invalid"


@pytest.mark.parametrize(
    "cls,code",
    [
        (ConfigError, ErrorCode.CONFIG_INVALID),
        (InvalidQueryError, ErrorCode.INVALID_QUERY),
        (RetrievalError, ErrorCode.RETRIEVAL_FAILED),
        (EmptyRetrievalError, ErrorCode.EMPTY_RETRIEVAL),
        (ProviderError, ErrorCode.PROVIDER_FAILED),
        (ProviderPayloadError, ErrorCode.PROVIDER_PAYLOAD_INVALID),
    ],
)
def test_each_error_has_its_fixed_code_and_format(cls, code):
    e = cls("something went wrong")
    assert isinstance(e, RAGError) and isinstance(e, Exception)
    assert e.code is code
    assert e.message == "something went wrong"
    assert str(e) == f"{code.value}: something went wrong"


def test_every_error_code_has_exactly_one_subclass():
    subclasses = RAGError.__subclasses__()
    assert sorted(c("x").code.value for c in subclasses) == sorted(c.value for c in ErrorCode)


def test_base_error_takes_explicit_code():
    e = RAGError(ErrorCode.RETRIEVAL_FAILED, "boom")
    assert e.code is ErrorCode.RETRIEVAL_FAILED
    assert str(e) == "retrieval_failed: boom"


# --- structural guards over every model --------------------------------------

EXPECTED_FIELDS = {
    "ContextChunk": {"id", "text", "score", "source_id"},
    "Source": {"id"},
    "Usage": {"input_tokens", "output_tokens"},
    "Metrics": {"latency_ms"},
    "RAGResponse": {"answer", "chunks", "sources", "usage", "metrics"},
    "Generation": {"text", "input_tokens", "output_tokens"},
}


def normalizer_models():
    import normalizer

    return {
        name: obj
        for name, obj in vars(normalizer).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseModel)
        and obj.__module__ == "normalizer"
        and not name.startswith("_")
    }


def test_every_model_has_exactly_its_contract_fields():
    assert {n: set(m.model_fields) for n, m in normalizer_models().items()} == EXPECTED_FIELDS


def property_names(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties":
                yield from value
            yield from property_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from property_names(item)


def test_no_path_or_provider_property_anywhere_in_the_schema():
    names = set(property_names(RAGResponse.model_json_schema()))
    assert names.isdisjoint({"path", "provider"})
    assert names == set().union(*EXPECTED_FIELDS.values())


def test_every_model_forbids_extras_is_frozen_and_hides_input():
    for name, model in normalizer_models().items():
        cfg = model.model_config
        assert cfg.get("extra") == "forbid", name
        assert cfg.get("frozen") is True, name
        assert cfg.get("hide_input_in_errors") is True, name


@pytest.mark.parametrize(
    "factory",
    [
        lambda m: ContextChunk(**chunk(score=m)),
        lambda m: Usage(input_tokens=m, output_tokens=1),
        lambda m: Metrics(latency_ms=m),
        lambda m: Source(id=5, title=m),
        lambda m: RAGResponse.model_validate(response(answer=m, usage=m)),
    ],
)
def test_validation_errors_never_echo_input(factory):
    with pytest.raises(ValidationError) as excinfo:
        factory("MARKERVALUE")
    assert "MARKERVALUE" not in str(excinfo.value)


# --- copy / pickle round trip (review PR #17) ----------------------------------


@pytest.mark.parametrize(
    "error",
    [
        RAGError(ErrorCode.INVALID_QUERY, "x"),
        ConfigError("x"),
        InvalidQueryError("x"),
        RetrievalError("x"),
        EmptyRetrievalError("x"),
        ProviderError("x"),
        ProviderPayloadError("x"),
    ],
)
@pytest.mark.parametrize("roundtrip", [copy.copy, copy.deepcopy, lambda e: pickle.loads(pickle.dumps(e))])
def test_errors_survive_copy_and_pickle(error, roundtrip):
    clone = roundtrip(error)
    assert type(clone) is type(error)
    assert str(clone) == str(error)
    assert clone.code == error.code
    assert clone.message == error.message
