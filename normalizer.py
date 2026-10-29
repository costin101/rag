"""The single caller-visible response contract (rule 1) and the public error set.

Path adapters (generation -> RAGResponse) live here; vendor payload formats never do. The public
error set is `RAGError` and its subclasses; `ContainerError` is internal to the container path
and never reaches a caller.
"""

import json
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

_HTTP_SCHEMES = ("http", "https")


def has_hidden_chars(text: str) -> bool:
    """True if `text` contains whitespace or a control/format character anywhere."""
    return any(c.isspace() or unicodedata.category(c) in ("Cc", "Cf") for c in text)


def url_problem(url: str, *, allow_query: bool) -> str | None:
    """Reason `url` is not a safe plain http(s) URL, or None. Never echoes the URL.

    Rejects whitespace/control characters, other schemes, missing host, embedded
    userinfo, an empty or invalid port, and (unless allow_query) a query string,
    path parameter (';') or fragment.
    """
    if has_hidden_chars(url):
        return "must not contain whitespace or control characters"
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "has an invalid port or host"
    if parts.scheme not in _HTTP_SCHEMES or not parts.hostname:
        return "must be an http(s) URL with a host"
    if "@" in parts.netloc:
        return "must not embed credentials"
    if port == 0 or parts.netloc.endswith(":"):
        return "has an invalid port or host"
    if not allow_query and any(c in url for c in "?#;"):
        return "must not contain a query string, ';' or fragment"
    return None


TokenCount = Annotated[StrictInt, Field(ge=0, le=10**12)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class ContextChunk(_Model):
    id: str
    text: str
    score: Annotated[StrictFloat, Field(allow_inf_nan=False)]
    source_id: str


class Source(_Model):
    id: str


class Usage(_Model):
    input_tokens: TokenCount
    output_tokens: TokenCount


class Metrics(_Model):
    latency_ms: Annotated[StrictFloat, Field(ge=0, le=10**9, allow_inf_nan=False)]


# Visible-looking-blank characters outside the Zs/Zl/Zp/Cc/Cf categories: Hangul and
# Khmer fillers, braille blank, combining grapheme joiner, variation selectors.
_BLANK_LOOKALIKES = frozenset("\u034f\u115f\u1160\u17b4\u17b5\u2800\u3164\uffa0") | frozenset(
    chr(c) for c in range(0xFE00, 0xFE10)
)


def _invisible(char: str) -> bool:
    return (
        char.isspace()
        or unicodedata.category(char) in ("Cc", "Cf", "Zs", "Zl", "Zp")
        or char in _BLANK_LOOKALIKES
    )


def is_blank(text: str) -> bool:
    """True if `text` has no visible character (empty, whitespace, invisible or filler only)."""
    return all(_invisible(c) for c in text)


class RAGResponse(_Model):
    answer: str
    # frozen=True blocks attribute assignment only; the lists themselves stay mutable.
    chunks: list[ContextChunk]
    sources: list[Source]
    usage: Usage
    metrics: Metrics

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, answer: str) -> str:
        if is_blank(answer):
            raise ValueError("answer must not be blank")
        return answer


class ErrorCode(StrEnum):
    CONFIG_INVALID = "config_invalid"
    INVALID_QUERY = "invalid_query"
    RETRIEVAL_FAILED = "retrieval_failed"
    EMPTY_RETRIEVAL = "empty_retrieval"
    PROVIDER_FAILED = "provider_failed"
    PROVIDER_PAYLOAD_INVALID = "provider_payload_invalid"


class RAGError(Exception):
    """Public failure carrying a stable code; str() is '<code>: <message>'."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message

    def __reduce__(self):
        # Subclasses fix their code and take only the message; the base takes both.
        args = (self.code, self.message) if type(self) is RAGError else (self.message,)
        return (type(self), args)


class ConfigError(RAGError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.CONFIG_INVALID, message)


class InvalidQueryError(RAGError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.INVALID_QUERY, message)


class RetrievalError(RAGError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.RETRIEVAL_FAILED, message)


class EmptyRetrievalError(RAGError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.EMPTY_RETRIEVAL, message)


class ProviderError(RAGError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.PROVIDER_FAILED, message)


class ProviderPayloadError(RAGError):
    def __init__(self, message: str):
        super().__init__(ErrorCode.PROVIDER_PAYLOAD_INVALID, message)


class ContainerError(Exception):
    """The container path failed. Internal, NOT a `RAGError`: callers never see it.

    Messages are fixed text: never a Nomad body, host, token or any container output.
    """


class Generation(_Model):
    """Provider-neutral generation result: what any path hands to its adapter."""

    text: str
    input_tokens: TokenCount
    output_tokens: TokenCount


def from_generation(
    generation: Generation, chunks: Sequence[ContextChunk], latency_ms: float
) -> RAGResponse:
    """Adapt a generation plus the chunks it was grounded on into the caller-visible response.

    Sources are derived from the chunks' source_id, deduplicated in first-seen order; chunks
    carry no title or uri, and a source is only its id. Zero chunks means the answer would
    be ungrounded: that is an EmptyRetrievalError, never a response.
    """
    try:
        if not chunks:
            raise EmptyRetrievalError("no context chunks to ground the answer")
        if is_blank(generation.text):
            raise ProviderPayloadError("the provider returned a blank answer")
        source_ids = dict.fromkeys(c.source_id for c in chunks)
        return RAGResponse(
            answer=generation.text,
            chunks=list(chunks),
            sources=[Source(id=source_id) for source_id in source_ids],
            usage=Usage(input_tokens=generation.input_tokens, output_tokens=generation.output_tokens),
            metrics=Metrics(latency_ms=latency_ms),
        )
    finally:
        # Error trackers capture locals: the context and the generated text stay out of them.
        generation = chunks = None


_CONTAINER_KEYS = frozenset({"v", "answer", "input_tokens", "output_tokens"})
_TOO_LARGE = "container output too large"
_NOT_V1 = "container output is not a valid v1 result"


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _refuse_constant(_name: str):
    raise ValueError("not a JSON number")


def _read_container_output(stdout, max_bytes: int) -> Generation | str:
    """The generation, or the fixed refusal text. Never raises on bad input, never echoes it."""
    if type(stdout) is not bytes:
        return _NOT_V1
    if len(stdout) > max_bytes:
        return _TOO_LARGE
    # One line, optionally ended by one newline; whitespace, a BOM or a second line is a refusal.
    body = stdout[:-1] if stdout.endswith(b"\n") else stdout
    if not (body.startswith(b"{") and body.endswith(b"}")) or b"\n" in body or b"\r" in body:
        return _NOT_V1
    try:
        data = json.loads(
            body.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_refuse_constant
        )
    except (ValueError, RecursionError):
        return _NOT_V1
    if type(data) is not dict or data.keys() != _CONTAINER_KEYS or type(data["v"]) is not int:
        return _NOT_V1
    answer, tokens = data["answer"], (data["input_tokens"], data["output_tokens"])
    if data["v"] != 1 or type(answer) is not str or any(type(n) is not int for n in tokens):
        return _NOT_V1
    if not all(0 <= n <= 10**12 for n in tokens) or is_blank(answer):
        return _NOT_V1
    try:
        answer.encode("utf-8")  # a lone surrogate escape decodes but cannot be sent on
    except UnicodeEncodeError:
        return _NOT_V1
    return Generation(text=answer, input_tokens=tokens[0], output_tokens=tokens[1])


def parse_container_output(stdout: bytes, *, max_bytes: int) -> Generation:
    """Parse the container's stdout v1 line into a Generation, or raise a fixed ContainerError.

    Strict: at most `max_bytes` bytes, exactly one JSON object line (an optional final newline)
    `{"v":1,"answer":str,"input_tokens":int,"output_tokens":int}` with exactly those keys, no
    duplicates, NaN or Infinity, plain ints (no bool) within the token range and a visible answer.
    The output is untrusted: the error is raised outside any `except` (no cause or context) and
    after the names are cleared, so neither it nor a traceback with captured locals carries it.
    """
    try:
        result = _read_container_output(stdout, max_bytes)
    except Exception:
        result = _NOT_V1
    finally:
        stdout = None
    if isinstance(result, str):
        raise ContainerError(result)
    return result
