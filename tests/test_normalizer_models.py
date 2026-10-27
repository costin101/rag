import json
import math

import pytest
from pydantic import ValidationError

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
        "sources": [{"id": "s1", "title": "Doc", "uri": "https://example.com/doc"}],
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


@pytest.mark.parametrize("blank", ["", " ", "\n\t "])
def test_blank_answer_rejected(blank):
    with pytest.raises(ValidationError):
        RAGResponse.model_validate(response(answer=blank))


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


def test_source_optional_fields_default_to_none():
    s = Source(id="s1")
    assert s.title is None and s.uri is None


@pytest.mark.parametrize("uri", ["http://example.com/a", "https://example.com/a?q=1"])
def test_source_keeps_http_uris(uri):
    assert Source(id="s1", uri=uri).uri == uri


@pytest.mark.parametrize(
    "uri",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "data:text/html,hi",
        "example.com/path",
        "/relative/path",
        "https://",
        "http:///nohost",
        "",
    ],
)
def test_source_coerces_other_uris_to_none(uri):
    assert Source(id="s1", title="t", uri=uri).uri is None


def test_source_non_string_uri_rejected():
    with pytest.raises(ValidationError):
        Source(id="s1", uri=5)


def test_source_extra_forbidden_and_frozen():
    with pytest.raises(ValidationError):
        Source(id="s1", url="https://example.com")
    with pytest.raises(ValidationError):
        Source(id="s1").title = "x"


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


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf, "fast", None])
def test_metrics_rejects_negative_nonfinite_or_non_numeric(bad):
    with pytest.raises(ValidationError):
        Metrics(latency_ms=bad)


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
