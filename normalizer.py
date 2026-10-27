"""The single caller-visible response contract (rule 1) and the public error set.

Path adapters (generation -> RAGResponse) are added here by later stories; vendor
payload formats never live in this module.
"""

import unicodedata
from enum import StrEnum
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

_HTTP_SCHEMES = ("http", "https")


def url_problem(url: str, *, allow_query: bool) -> str | None:
    """Reason `url` is not a safe plain http(s) URL, or None. Never echoes the URL.

    Rejects whitespace/control characters, other schemes, missing host, embedded
    userinfo, an invalid port, and (unless allow_query) a query string or fragment.
    """
    if any(c.isspace() or unicodedata.category(c) == "Cc" for c in url):
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
    if port == 0:
        return "has an invalid port or host"
    if not allow_query and ("?" in url or "#" in url):
        return "must not contain a query string or fragment"
    return None


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


TokenCount = Annotated[StrictInt, Field(ge=0)]


class ContextChunk(_Model):
    id: str
    text: str
    score: Annotated[StrictFloat, Field(allow_inf_nan=False)]
    source_id: str


class Source(_Model):
    id: str
    title: str | None = None
    uri: str | None = None

    @field_validator("uri")
    @classmethod
    def _http_uri_only(cls, uri: str | None) -> str | None:
        if uri is None:
            return None
        return None if url_problem(uri, allow_query=True) else uri


class Usage(_Model):
    input_tokens: TokenCount
    output_tokens: TokenCount


class Metrics(_Model):
    latency_ms: Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]


def _invisible(char: str) -> bool:
    # Whitespace plus Unicode format characters (zero-width space, BOM, joiners).
    return char.isspace() or unicodedata.category(char) == "Cf"


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
        if all(_invisible(c) for c in answer):
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
