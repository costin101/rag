import traceback

import pytest

import pipeline
from normalizer import (
    ConfigError,
    ContextChunk,
    EmptyRetrievalError,
    ErrorCode,
    Generation,
    InvalidQueryError,
    ProviderError,
    ProviderPayloadError,
    RAGError,
    RAGResponse,
    RetrievalError,
)
from tests.contract import assert_valid_rag_response
from tests.test_retriever import point, retriever_over

SECRET = "test-not-a-key-LEAK"


def chunk(cid="c1", source_id="s1"):
    return ContextChunk(id=cid, text="the sky is blue", score=0.5, source_id=source_id)


class StubRetriever:
    def __init__(self, result=None, error=None):
        self.result = [chunk("c1", "s1"), chunk("c2", "s2")] if result is None else result
        self.error = error
        self.calls = []

    def retrieve(self, query, k=None):
        self.calls.append((query, k))
        if self.error:
            raise self.error
        return self.result


class StubGenerator:
    def __init__(self, result=None, error=None):
        self.result = result or Generation(text="Blue.", input_tokens=11, output_tokens=4)
        self.error = error
        self.calls = []

    def generate(self, query, chunks):
        self.calls.append((query, chunks))
        if self.error:
            raise self.error
        return self.result


def failure(retriever, generator, query="why?", **kw):
    with pytest.raises(RAGError) as excinfo:
        pipeline.answer(query, retriever, generator, **kw)
    return excinfo.value


PRIVATE_QUERY = "why LEAK?"


def leaky_chunk():
    return ContextChunk(id="c1", text="private LEAK text", score=0.5, source_id="s1")


def failure_on_leaky_input(retriever, generator):
    # Reads the query from a module constant so no local of this frame holds it.
    with pytest.raises(RAGError) as excinfo:
        pipeline.answer(PRIVATE_QUERY, retriever, generator)
    return excinfo.value


def assert_no_payload_in_pipeline_frames(err):
    """Locals of the pipeline's own frames, as an error tracker would capture them."""
    frames = [f for f, _ in traceback.walk_tb(err.__traceback__) if f.f_code.co_filename == pipeline.__file__]
    assert frames
    assert "LEAK" not in "".join(repr(v) for f in frames for v in f.f_locals.values())


def assert_clean(err):
    assert err.__cause__ is None and err.__context__ is None
    text = f"{err} {err!r} {err.args} " + "".join(
        traceback.TracebackException.from_exception(err, capture_locals=True).format()
    )
    assert "LEAK" not in text


# --- happy path -------------------------------------------------------------------


def test_returns_a_contract_valid_response():
    r = pipeline.answer("why?", StubRetriever(), StubGenerator())
    assert type(r) is RAGResponse
    assert_valid_rag_response(r)
    assert r.answer == "Blue."
    assert [c.id for c in r.chunks] == ["c1", "c2"]
    assert [s.id for s in r.sources] == ["s1", "s2"]
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 4)


def test_generator_receives_the_query_and_the_retrieved_chunks():
    retriever, generator = StubRetriever(), StubGenerator()
    pipeline.answer("why?", retriever, generator)
    assert generator.calls == [("why?", retriever.result)]


def test_retriever_receives_the_query_and_k():
    retriever = StubRetriever()
    pipeline.answer("why?", retriever, StubGenerator(), k=7)
    pipeline.answer("again", retriever, StubGenerator())
    assert retriever.calls == [("why?", 7), ("again", None)]


def test_query_is_passed_through_unchanged():
    retriever, generator = StubRetriever(), StubGenerator()
    pipeline.answer("  spaced \n", retriever, generator)
    assert retriever.calls[0][0] == generator.calls[0][0] == "  spaced \n"


def test_latency_is_measured_here_in_milliseconds(monkeypatch):
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr(pipeline, "perf_counter", lambda: next(ticks))
    r = pipeline.answer("why?", StubRetriever(), StubGenerator())
    assert r.metrics.latency_ms == pytest.approx(250.0)


def test_latency_covers_retrieval_and_generation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pipeline, "perf_counter", lambda: clock[0])

    class Slow(StubRetriever):
        def retrieve(self, query, k=None):
            clock[0] += 0.1
            return super().retrieve(query, k)

    class SlowGen(StubGenerator):
        def generate(self, query, chunks):
            clock[0] += 0.2
            return super().generate(query, chunks)

    r = pipeline.answer("why?", Slow(), SlowGen())
    assert r.metrics.latency_ms == pytest.approx(300.0)


# --- invalid query / empty retrieval: nothing downstream runs -------------------------


@pytest.mark.parametrize("bad", ["", "  ", "\n", "​", None, 3])
def test_blank_query_is_rejected_before_retrieval(bad):
    retriever, generator = StubRetriever(), StubGenerator()
    err = failure(retriever, generator, bad)
    assert isinstance(err, InvalidQueryError) and err.code is ErrorCode.INVALID_QUERY
    assert retriever.calls == [] and generator.calls == []


def test_empty_retrieval_never_calls_the_generator():
    generator = StubGenerator()
    err = failure(StubRetriever(result=[]), generator)
    assert isinstance(err, EmptyRetrievalError)
    assert generator.calls == []


def test_a_retriever_raising_empty_retrieval_never_calls_the_generator():
    generator = StubGenerator()
    err = failure(StubRetriever(error=EmptyRetrievalError("none")), generator)
    assert isinstance(err, EmptyRetrievalError) and generator.calls == []


# --- RAGError passes through unwrapped -------------------------------------------------


@pytest.mark.parametrize(
    "side,error",
    [
        ("retriever", RetrievalError("store down")),
        ("retriever", EmptyRetrievalError("none")),
        ("retriever", ConfigError("bad config")),
        ("generator", ProviderError("rate limited")),
        ("generator", ProviderPayloadError("truncated")),
    ],
)
def test_rag_errors_are_not_wrapped(side, error):
    retriever = StubRetriever(error=error if side == "retriever" else None)
    generator = StubGenerator(error=error if side == "generator" else None)
    rebuilt = failure(retriever, generator)
    # Rebuilt so the component's own frames stay out of the traceback: same type, code, message.
    assert rebuilt is not error
    assert (type(rebuilt), rebuilt.code, rebuilt.message) == (type(error), error.code, error.message)
    assert rebuilt.__cause__ is None and rebuilt.__context__ is None


class DetailedRetrievalError(RetrievalError):
    """A component's own RAGError subclass whose constructor is not `(message)` (rule 5)."""

    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail

    def __str__(self):
        return f"{self.message} [{self.detail}]"


class OSRetrievalError(RetrievalError, OSError):
    """Inherits a built-in with its own `__new__`."""


class NewRetrievalError(RetrievalError):
    def __new__(cls, message, detail):
        return super().__new__(cls)

    def __init__(self, message, detail):
        super().__init__(message)


class NoSuperError(RetrievalError):
    def __init__(self, message, detail):
        Exception.__init__(self, message)


class BadCodeError(RetrievalError):
    def __init__(self, message):
        super().__init__(message)
        self.code = "custom"


class BadMessageError(RetrievalError):
    def __init__(self, message):
        super().__init__(message)
        self.message = 42


@pytest.mark.parametrize(
    "error",
    [
        DetailedRetrievalError("store down", f"{SECRET} LEAK"),
        OSRetrievalError("store down"),
        NewRetrievalError("store down", "x"),
    ],
)
def test_a_component_error_subclass_is_replaced_by_the_public_class_for_its_code(error):
    rebuilt = failure(StubRetriever(error=error), StubGenerator())
    assert type(rebuilt) is RetrievalError
    assert (rebuilt.code, rebuilt.message) == (ErrorCode.RETRIEVAL_FAILED, "store down")
    assert str(rebuilt) == "retrieval_failed: store down"
    assert_clean(rebuilt)


class RaisingCodeError(RetrievalError):
    @property
    def code(self):
        raise ValueError(f"{SECRET} LEAK")

    @code.setter
    def code(self, value):
        pass


class StrCodeError(RetrievalError):
    def __init__(self, message):
        super().__init__(message)
        self.code = "retrieval_failed"


class StrSubclassMessage(str):
    def __str__(self):
        return "LEAK"


class StrSubclassMessageError(RetrievalError):
    def __init__(self, message):
        super().__init__(message)
        self.message = StrSubclassMessage(message)


@pytest.mark.parametrize(
    "error",
    [
        NoSuperError("store down", "x"),
        BadCodeError("store down"),
        BadMessageError("store down"),
        RaisingCodeError("store down"),
        StrCodeError("store down"),
        StrSubclassMessageError("store down"),
    ],
)
def test_a_malformed_component_error_becomes_the_stage_error_and_never_escapes_bare(error):
    rebuilt = failure_on_leaky_input(StubRetriever(error=error), StubGenerator())
    assert type(rebuilt) is RetrievalError
    assert rebuilt.message == "retrieval failed unexpectedly"
    assert_clean(rebuilt)


def test_a_malformed_generator_error_becomes_the_provider_stage_error():
    rebuilt = failure(StubRetriever(), StubGenerator(error=NoSuperError("x", "y")))
    assert type(rebuilt) is ProviderError and rebuilt.message == "generation failed unexpectedly"
    assert_clean(rebuilt)


# --- anything else becomes the right RAGError, cleanly ---------------------------------


@pytest.mark.parametrize(
    "boom", [RuntimeError(SECRET), KeyError(SECRET), ValueError(SECRET), OSError(SECRET), ZeroDivisionError()]
)
def test_unexpected_retriever_errors_become_retrieval_failed(boom):
    generator = StubGenerator()
    err = failure(StubRetriever(error=boom), generator)
    assert type(err) is RetrievalError and err.code is ErrorCode.RETRIEVAL_FAILED
    assert_clean(err)
    assert generator.calls == []


@pytest.mark.parametrize("boom", [RuntimeError(SECRET), KeyError(SECRET), OSError(SECRET)])
def test_unexpected_generator_errors_become_provider_failed(boom):
    err = failure(StubRetriever(), StubGenerator(error=boom))
    assert type(err) is ProviderError and err.code is ErrorCode.PROVIDER_FAILED
    assert_clean(err)


def test_a_generator_returning_the_wrong_type_is_a_payload_error():
    err = failure(StubRetriever(), StubGenerator(result="just text"))
    assert type(err) is ProviderPayloadError
    assert_clean(err)


def test_a_blank_generation_is_a_payload_error():
    blank = Generation(text=" ", input_tokens=1, output_tokens=1)
    err = failure_on_leaky_input(StubRetriever(result=[leaky_chunk()]), StubGenerator(result=blank))
    assert type(err) is ProviderPayloadError
    assert_clean(err)


def test_a_generation_that_bypassed_validation_is_a_payload_error():
    bad = Generation.model_construct(text="answer LEAK", input_tokens=-5, output_tokens=1)
    err = failure_on_leaky_input(StubRetriever(result=[leaky_chunk()]), StubGenerator(result=bad))
    assert type(err) is ProviderPayloadError
    assert_clean(err)


@pytest.mark.parametrize(
    "boom", [RuntimeError("x"), ProviderError("rate limited"), ProviderPayloadError("truncated")]
)
def test_generator_failures_keep_the_query_and_context_out_of_captured_locals(boom):
    err = failure_on_leaky_input(StubRetriever(result=[leaky_chunk()]), StubGenerator(error=boom))
    assert_no_payload_in_pipeline_frames(err)


@pytest.mark.parametrize(
    "points,kind",
    [
        ((), EmptyRetrievalError),
        # A hit without source_id fails while its payload (the chunk text) is a local.
        ((point(1, [1.0, 0.0, 0.0], text="private LEAK text"),), RetrievalError),
    ],
)
def test_a_real_retriever_failure_leaves_no_query_or_text_in_the_whole_traceback(points, kind):
    err = failure_on_leaky_input(retriever_over(points=points), StubGenerator())
    assert type(err) is kind
    assert_clean(err)
    frames = [f.f_code.co_filename for f, _ in traceback.walk_tb(err.__traceback__)]
    assert not any(name.endswith("retriever.py") for name in frames)


def test_retriever_failures_keep_the_query_out_of_captured_locals():
    err = failure_on_leaky_input(StubRetriever(error=RetrievalError("store down")), StubGenerator())
    assert_no_payload_in_pipeline_frames(err)


@pytest.mark.parametrize(
    "result", [None, "chunks", [{"id": "c1"}], [chunk(), "x"], 5, iter([chunk()]), {chunk(): 1}]
)
def test_malformed_retriever_output_is_retrieval_failed(result):
    generator = StubGenerator()
    retriever = StubRetriever(result=result)
    if result is None:
        retriever.result = None
    err = failure(retriever, generator)
    assert type(err) is RetrievalError and "unexpected" in err.message
    assert generator.calls == []
    assert_clean(err)


def test_a_retriever_returning_a_tuple_of_chunks_is_accepted():
    generator = StubGenerator()
    r = pipeline.answer("why?", StubRetriever(result=(chunk(),)), generator)
    assert_valid_rag_response(r)
    assert type(generator.calls[0][1]) is list


def test_traceback_locals_hold_no_secret_from_a_failing_generator():
    err = failure(StubRetriever(), StubGenerator(error=RuntimeError(SECRET)))
    assert_clean(err)


def test_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        pipeline.answer("why?", StubRetriever(error=KeyboardInterrupt()), StubGenerator())


def test_no_bare_exception_escapes_for_arbitrary_failures():
    for retriever, generator in (
        (StubRetriever(error=RuntimeError("x")), StubGenerator()),
        (StubRetriever(), StubGenerator(error=RuntimeError("x"))),
        (object(), StubGenerator()),
        (StubRetriever(), object()),
    ):
        with pytest.raises(RAGError):
            pipeline.answer("why?", retriever, generator)


def test_every_error_code_maps_to_a_public_class_carrying_that_code():
    assert set(pipeline._PUBLIC) == set(ErrorCode)
    for code, error_type in pipeline._PUBLIC.items():
        assert error_type("m").code is code
