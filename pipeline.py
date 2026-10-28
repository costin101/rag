"""The path-neutral answer pipeline (rules 1, 3, 5, 6): validate, retrieve, generate, normalize.

Knows neither the vector store nor the provider: both arrive as interfaces. Latency is measured
here so every path reports the same quantity. Callers see a `RAGResponse` or a `RAGError`; any
other failure is converted to the `RAGError` for the stage it came from, with a fixed message
and no cause, context or secret attached (the new error is raised outside the `except` block).
A `RAGError` from a component is rebuilt the same way as the public class for its code (message
only; a malformed one becomes the stage's fixed error), so no component frame, none of its
locals and none of its own subclass state rides along in the traceback.
"""

from time import perf_counter

from normalizer import (
    ConfigError,
    ContextChunk,
    EmptyRetrievalError,
    ErrorCode,
    InvalidQueryError,
    ProviderError,
    ProviderPayloadError,
    RAGError,
    RAGResponse,
    RetrievalError,
    from_generation,
    is_blank,
)
from provider import Provider
from retriever import Retriever

# stage -> (error type, fixed message) for a failure that is not already a RAGError.
_STAGES = {
    "retrieve": (RetrievalError, "retrieval failed unexpectedly"),
    "generate": (ProviderError, "generation failed unexpectedly"),
    "normalize": (ProviderPayloadError, "the generation could not be turned into a response"),
}


# code -> the public class callers see, whatever subclass a component raised.
_PUBLIC = {
    ErrorCode.CONFIG_INVALID: ConfigError,
    ErrorCode.INVALID_QUERY: InvalidQueryError,
    ErrorCode.RETRIEVAL_FAILED: RetrievalError,
    ErrorCode.EMPTY_RETRIEVAL: EmptyRetrievalError,
    ErrorCode.PROVIDER_FAILED: ProviderError,
    ErrorCode.PROVIDER_PAYLOAD_INVALID: ProviderPayloadError,
}


def answer(
    query: str, retriever: Retriever, generator: Provider, *, k: int | None = None
) -> RAGResponse:
    """Answer `query` from retrieved context; `k=None` leaves top-k to the retriever."""
    started = perf_counter()
    stage = "retrieve"
    chunks = generation = response = None
    try:
        if not isinstance(query, str) or is_blank(query):
            raise InvalidQueryError("the query must not be blank")
        chunks = _chunks(retriever.retrieve(query, k))
        if not chunks:
            raise EmptyRetrievalError("no context was retrieved for the query")
        stage = "generate"
        generation = generator.generate(query, chunks)
        stage = "normalize"
        # from_generation validates: a wrong type or a model_construct bypass fails right here.
        response = from_generation(generation, chunks, (perf_counter() - started) * 1000.0)
    except RAGError as exc:
        # Rebuilt, not re-raised: the original traceback would carry the frames (and locals)
        # of whichever component raised it.
        failure = _public(exc, stage)
    except Exception:
        error_type, message = _STAGES[stage]
        failure = error_type(message)
    else:
        return response
    finally:
        # A traceback with captured locals (error trackers do this) must not carry the query,
        # the context or the generated text; the names are cleared on every exit path.
        query = chunks = generation = response = None
    raise failure


def _public(exc: RAGError, stage: str) -> RAGError:
    """The public error for `exc`: the class for its code, built from its message alone.

    A component's own subclass is never instantiated or returned, so its constructor, `__str__`
    and extra state cannot fail or leak here. A code or message that is missing or malformed
    degrades to the stage's fixed error.
    """
    try:
        code, message = exc.code, exc.message
        if isinstance(code, ErrorCode) and isinstance(message, str):
            return _PUBLIC[code](message)
    except Exception:
        pass
    error_type, fixed = _STAGES[stage]
    return error_type(fixed)


def _chunks(result) -> list[ContextChunk]:
    if not isinstance(result, (list, tuple)) or not all(isinstance(c, ContextChunk) for c in result):
        raise RetrievalError("the retriever returned an unexpected result")
    return list(result)
