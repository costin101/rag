"""Environment-validated settings (rule 4): the only place hosts, limits and keys come from.

Errors name every bad variable and the reason, never a value.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictFloat,
    StrictInt,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from normalizer import ConfigError, has_hidden_chars, url_problem

TimeoutS = Annotated[StrictFloat, Field(gt=0, le=600, allow_inf_nan=False)]

# API keys travel in HTTP headers, so they must be printable ASCII.
_ASCII_KEY = re.compile(r"[\x21-\x7e]+")

# Env numbers are plain decimals only: no signs, underscores, exponents or non-ASCII digits.
_WHOLE = re.compile(r"[0-9]{1,9}")
_DECIMAL = re.compile(r"[0-9]{1,9}(\.[0-9]{1,9})?")

# The collection name is interpolated into Qdrant REST paths: no separators, '..' or escapes.
_COLLECTION = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")

TextField = Annotated[str, Field(min_length=1)]

# The default embedding model and its pinned Hugging Face commit (checked 2026-10-28).
# fastembed downloads this model from Qdrant/bge-small-en-v1.5-onnx-Q, not from BAAI, so the
# pinned commit is that repo's. A different RAG_EMBED_MODEL must name its own revision.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBED_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    ANTHROPIC_API_KEY: SecretStr
    RAG_PROVIDER: Literal["anthropic"] = "anthropic"
    RAG_MODEL: TextField = "claude-sonnet-5-5"
    RAG_MAX_TOKENS: StrictInt = Field(default=1024, gt=0, le=200_000)
    RAG_PROVIDER_TIMEOUT_S: TimeoutS = 30.0

    # Declared before QDRANT_URL so its validator can see whether a key is set.
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_URL: str
    QDRANT_COLLECTION: TextField
    QDRANT_TIMEOUT_S: TimeoutS = 10.0

    RAG_TOP_K: StrictInt = Field(default=5, ge=1, le=50)
    RAG_EMBED_MODEL: TextField = DEFAULT_EMBED_MODEL
    RAG_EMBED_REVISION: TextField = Field(default=None, validate_default=True)
    RAG_EMBED_CACHE_DIR: Path | None = None

    @field_validator("RAG_MAX_TOKENS", "RAG_TOP_K", mode="before")
    @classmethod
    def _whole_number(cls, value):
        if isinstance(value, str):
            if not _WHOLE.fullmatch(value):
                raise ValueError("must be a whole number")
            return int(value)
        return value

    @field_validator("RAG_PROVIDER_TIMEOUT_S", "QDRANT_TIMEOUT_S", mode="before")
    @classmethod
    def _decimal_number(cls, value):
        if isinstance(value, str):
            if not _DECIMAL.fullmatch(value):
                raise ValueError("must be a plain decimal number")
            return float(value)
        return value

    @field_validator("ANTHROPIC_API_KEY", "QDRANT_API_KEY", mode="before")
    @classmethod
    def _clean_key(cls, value):
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw, str):
            if not raw.strip():
                raise ValueError("must not be empty")
            if has_hidden_chars(raw):
                raise ValueError("must not contain whitespace or control characters")
            if not _ASCII_KEY.fullmatch(raw):
                raise ValueError("must contain only printable ASCII characters")
        return value

    @field_validator("RAG_MODEL", "RAG_EMBED_MODEL", "RAG_EMBED_REVISION", "QDRANT_COLLECTION")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        if value is not None and has_hidden_chars(value):
            raise ValueError("must not contain whitespace or control characters")
        return value

    @field_validator("QDRANT_COLLECTION")
    @classmethod
    def _collection_name(cls, value: str) -> str:
        if not _COLLECTION.fullmatch(value):
            raise ValueError("must contain only letters, digits, '_', '-' and inner '.'")
        return value

    @field_validator("RAG_EMBED_CACHE_DIR", mode="before")
    @classmethod
    def _no_nul(cls, value):
        if isinstance(value, str) and "\0" in value:
            raise ValueError("must not contain a NUL byte")
        return value

    @field_validator("RAG_EMBED_REVISION", mode="before")
    @classmethod
    def _pinned_revision(cls, value, info: ValidationInfo):
        if value is not None:
            return value
        model = info.data.get("RAG_EMBED_MODEL")
        if model is None:  # the model itself failed validation; that error is reported
            return DEFAULT_EMBED_REVISION
        if model != DEFAULT_EMBED_MODEL:
            raise ValueError("required when RAG_EMBED_MODEL is not the default model")
        return DEFAULT_EMBED_REVISION

    @field_validator("QDRANT_URL")
    @classmethod
    def _http_url(cls, url: str, info: ValidationInfo) -> str:
        problem = url_problem(url, allow_query=False)
        if problem:
            raise ValueError(problem)
        if info.data.get("QDRANT_API_KEY") is not None and not url.lower().startswith("https://"):
            raise ValueError("must be https when QDRANT_API_KEY is set")
        return url


def _reason(error) -> str:
    if error["type"] == "missing":
        return "required but not set"
    # Pydantic messages describe the constraint, never the input (hide_input_in_errors).
    return error["msg"].removeprefix("Value error, ")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (default: os.environ); blank values count as unset."""
    source = os.environ if env is None else env
    values = {
        name: source[name].strip()
        for name in Settings.model_fields
        if name in source and source[name].strip()
    }
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        problems = "; ".join(f"{e['loc'][0]}: {_reason(e)}" for e in exc.errors())
    # Raised outside the except block so the error carries no __context__ to trace back.
    raise ConfigError(f"invalid configuration: {problems}")
