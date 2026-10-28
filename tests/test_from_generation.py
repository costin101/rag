import pytest
from pydantic import ValidationError

from normalizer import (
    ContextChunk,
    EmptyRetrievalError,
    Generation,
    ProviderPayloadError,
    RAGResponse,
    from_generation,
    is_blank,
)


def gen(**kw):
    data = {"text": "an answer", "input_tokens": 7, "output_tokens": 3}
    data.update(kw)
    return Generation(**data)


def chunk(cid="c1", source_id="s1", **kw):
    data = {"id": cid, "text": f"text of {cid}", "score": 0.5, "source_id": source_id}
    data.update(kw)
    return ContextChunk(**data)


# --- Generation ----------------------------------------------------------------


def test_generation_holds_text_and_counts():
    g = gen()
    assert (g.text, g.input_tokens, g.output_tokens) == ("an answer", 7, 3)


def test_generation_is_frozen():
    with pytest.raises(ValidationError):
        gen().text = "other"


def test_generation_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Generation(text="a", input_tokens=1, output_tokens=1, provider="anthropic")


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("bad", [-1, True, False, 1.0, "3", None])
def test_generation_rejects_bad_token_counts(field, bad):
    with pytest.raises(ValidationError):
        gen(**{field: bad})


def test_generation_accepts_zero_tokens():
    g = gen(input_tokens=0, output_tokens=0)
    assert (g.input_tokens, g.output_tokens) == (0, 0)


def test_generation_validation_error_hides_input():
    with pytest.raises(ValidationError) as excinfo:
        gen(input_tokens="LEAKME")
    assert "LEAKME" not in str(excinfo.value)


# --- is_blank --------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\t", "​", "ㅤ", "⠀ "])
def test_is_blank_true(text):
    assert is_blank(text)


@pytest.mark.parametrize("text", ["a", " a ", "0", "​x"])
def test_is_blank_false(text):
    assert not is_blank(text)


# --- from_generation -----------------------------------------------------------


def test_builds_a_valid_rag_response():
    r = from_generation(gen(), [chunk()], 12.5)
    assert isinstance(r, RAGResponse)
    assert r.answer == "an answer"
    assert r.usage.input_tokens == 7
    assert r.usage.output_tokens == 3
    assert r.metrics.latency_ms == 12.5
    assert r.chunks == [chunk()]


def test_sources_are_deduplicated_and_order_stable():
    chunks = [
        chunk("c1", "b"),
        chunk("c2", "a"),
        chunk("c3", "b"),
        chunk("c4", "c"),
        chunk("c5", "a"),
    ]
    r = from_generation(gen(), chunks, 1.0)
    assert [s.id for s in r.sources] == ["b", "a", "c"]
    assert [c.id for c in r.chunks] == ["c1", "c2", "c3", "c4", "c5"]


def test_sources_carry_only_the_id():
    (source,) = from_generation(gen(), [chunk()], 1.0).sources
    assert source.model_dump() == {"id": "s1"}


def test_every_chunk_source_id_is_a_source():
    chunks = [chunk("c1", "x"), chunk("c2", "y")]
    r = from_generation(gen(), chunks, 1.0)
    assert {c.source_id for c in r.chunks} <= {s.id for s in r.sources}


def test_accepts_any_sequence_of_chunks():
    r = from_generation(gen(), (chunk(),), 1.0)
    assert len(r.chunks) == 1


def test_empty_chunks_is_empty_retrieval():
    with pytest.raises(EmptyRetrievalError):
        from_generation(gen(), [], 1.0)


@pytest.mark.parametrize("text", ["", "  \n", "​"])
def test_blank_generation_is_a_payload_error(text):
    with pytest.raises(ProviderPayloadError):
        from_generation(gen(text=text), [chunk()], 1.0)


def test_response_round_trips_through_json():
    r = from_generation(gen(), [chunk()], 3.0)
    assert RAGResponse.model_validate_json(r.model_dump_json()) == r


def test_negative_latency_is_rejected():
    with pytest.raises(ValidationError):
        from_generation(gen(), [chunk()], -1.0)
