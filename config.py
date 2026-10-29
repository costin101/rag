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

# A pinned revision is a full commit hash: branches and tags can move.
_COMMIT = re.compile(r"[0-9a-f]{40}")

# Env numbers are plain decimals only: no signs, underscores, exponents or non-ASCII digits.
_WHOLE = re.compile(r"[0-9]{1,9}")
_DECIMAL = re.compile(r"[0-9]{1,9}(\.[0-9]{1,9})?")

# The collection name is interpolated into Qdrant REST paths: no separators, '..' or escapes.
_COLLECTION = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*")

# A digest-pinned image: optional registry host[:port], repository path, full sha256. A tag-only
# reference can move and a leading '-' would read as a docker flag, so neither is accepted.
_IMAGE_PART = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
_IMAGE = re.compile(
    rf"(?:{_IMAGE_PART}(?::[0-9]{{1,5}})?/)?{_IMAGE_PART}(?:/{_IMAGE_PART})*@sha256:[0-9a-f]{{64}}"
)

# Nomad namespace names go into a query parameter and the job spec: letters, digits, '-' only.
_NAMESPACE = re.compile(r"[A-Za-z0-9-]{1,128}")

TextField = Annotated[str, Field(min_length=1)]

# The default embedding model and its pinned Hugging Face commit (checked 2026-10-28).
# fastembed downloads this model from Qdrant/bge-small-en-v1.5-onnx-Q, not from BAAI, so the
# pinned commit is that repo's. A different RAG_EMBED_MODEL must name its own revision.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBED_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
# Passed explicitly to the download so the ambient HF_ENDPOINT never picks the host.
DEFAULT_EMBED_ENDPOINT = "https://huggingface.co"
# Passed explicitly to the SDK so the ambient ANTHROPIC_BASE_URL never picks the key's host.
DEFAULT_PROVIDER_BASE_URL = "https://api.anthropic.com"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    ANTHROPIC_API_KEY: SecretStr
    RAG_PROVIDER: Literal["anthropic"] = "anthropic"
    RAG_MODEL: TextField = "claude-sonnet-5-5"
    RAG_MAX_TOKENS: StrictInt = Field(default=1024, gt=0, le=200_000)
    RAG_PROVIDER_TIMEOUT_S: TimeoutS = 30.0
    RAG_PROVIDER_MAX_RETRIES: StrictInt = Field(default=2, ge=0, le=10)
    # Longest the SDK may sleep before one retry, whatever Retry-After the server sends.
    RAG_PROVIDER_MAX_RETRY_WAIT_S: TimeoutS = 60.0
    RAG_PROVIDER_BASE_URL: str = DEFAULT_PROVIDER_BASE_URL

    # Declared before QDRANT_URL so its validator can see whether a key is set.
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_URL: str
    QDRANT_COLLECTION: TextField
    QDRANT_TIMEOUT_S: TimeoutS = 10.0

    RAG_TOP_K: StrictInt = Field(default=5, ge=1, le=50)
    RAG_EMBED_MODEL: TextField = DEFAULT_EMBED_MODEL
    RAG_EMBED_REVISION: TextField = Field(default=None, validate_default=True)
    RAG_EMBED_CACHE_DIR: Path | None = None
    RAG_EMBED_ENDPOINT: str = DEFAULT_EMBED_ENDPOINT

    # The container path (all optional: the API path runs without Nomad; ContainerPath checks
    # that what it needs is set). Declared token-first so NOMAD_ADDR's validator can see it.
    NOMAD_TOKEN: SecretStr | None = None
    NOMAD_ADDR: str | None = None
    NOMAD_CACERT: Path | None = None
    NOMAD_NAMESPACE: str | None = None
    RAG_CONTAINER_IMAGE: str | None = None
    RAG_CONTAINER_CPU_MHZ: StrictInt = Field(default=500, ge=1, le=100_000)
    # Above Nomad's floor and the interpreter's own need.
    RAG_CONTAINER_MEMORY_MB: StrictInt = Field(default=128, ge=64, le=8192)
    RAG_CONTAINER_TIMEOUT_S: TimeoutS = 60.0
    # Caps on the input the host sends in the job spec and the stdout it reads back.
    RAG_CONTAINER_MAX_INPUT_BYTES: StrictInt = Field(default=262_144, ge=1024, le=1_048_576)
    RAG_CONTAINER_MAX_OUTPUT_BYTES: StrictInt = Field(default=65_536, ge=1024, le=1_048_576)

    @field_validator(
        "RAG_MAX_TOKENS",
        "RAG_PROVIDER_MAX_RETRIES",
        "RAG_TOP_K",
        "RAG_CONTAINER_CPU_MHZ",
        "RAG_CONTAINER_MEMORY_MB",
        "RAG_CONTAINER_MAX_INPUT_BYTES",
        "RAG_CONTAINER_MAX_OUTPUT_BYTES",
        mode="before",
    )
    @classmethod
    def _whole_number(cls, value):
        if isinstance(value, str):
            if not _WHOLE.fullmatch(value):
                raise ValueError("must be a whole number")
            return int(value)
        return value

    @field_validator(
        "RAG_PROVIDER_TIMEOUT_S",
        "RAG_PROVIDER_MAX_RETRY_WAIT_S",
        "QDRANT_TIMEOUT_S",
        "RAG_CONTAINER_TIMEOUT_S",
        mode="before",
    )
    @classmethod
    def _decimal_number(cls, value):
        if isinstance(value, str):
            if not _DECIMAL.fullmatch(value):
                raise ValueError("must be a plain decimal number")
            return float(value)
        return value

    @field_validator("ANTHROPIC_API_KEY", "QDRANT_API_KEY", "NOMAD_TOKEN", mode="before")
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

    @field_validator("RAG_EMBED_CACHE_DIR", "NOMAD_CACERT", mode="before")
    @classmethod
    def _no_nul(cls, value):
        if isinstance(value, str) and "\0" in value:
            raise ValueError("must not contain a NUL byte")
        return value

    @field_validator("RAG_EMBED_REVISION", mode="before")
    @classmethod
    def _pinned_revision(cls, value, info: ValidationInfo):
        if value is not None:
            if isinstance(value, str) and not _COMMIT.fullmatch(value):
                raise ValueError("must be a 40-character lowercase hex commit hash")
            return value
        model = info.data.get("RAG_EMBED_MODEL")
        if model is None:  # the model itself failed validation; that error is reported
            return DEFAULT_EMBED_REVISION
        if model != DEFAULT_EMBED_MODEL:
            raise ValueError("required when RAG_EMBED_MODEL is not the default model")
        return DEFAULT_EMBED_REVISION

    @field_validator("RAG_PROVIDER_BASE_URL", "RAG_EMBED_ENDPOINT")
    @classmethod
    def _https_endpoint(cls, url: str) -> str:
        problem = url_problem(url, allow_query=False)
        if problem:
            raise ValueError(problem)
        if not url.lower().startswith("https://"):
            raise ValueError("must be an https URL")
        return url

    @field_validator("QDRANT_URL")
    @classmethod
    def _http_url(cls, url: str, info: ValidationInfo) -> str:
        problem = url_problem(url, allow_query=False)
        if problem:
            raise ValueError(problem)
        if info.data.get("QDRANT_API_KEY") is not None and not url.lower().startswith("https://"):
            raise ValueError("must be https when QDRANT_API_KEY is set")
        return url

    @field_validator("NOMAD_ADDR")
    @classmethod
    def _nomad_url(cls, url: str | None, info: ValidationInfo) -> str | None:
        if url is None:
            return None
        problem = url_problem(url, allow_query=False)
        if problem:
            raise ValueError(problem)
        if info.data.get("NOMAD_TOKEN") is not None and not url.lower().startswith("https://"):
            raise ValueError("must be https when NOMAD_TOKEN is set")
        return url

    @field_validator("NOMAD_NAMESPACE")
    @classmethod
    def _namespace(cls, name: str | None) -> str | None:
        if name is not None and not _NAMESPACE.fullmatch(name):
            raise ValueError("must be 1-128 letters, digits or '-'")
        return name

    @field_validator("RAG_CONTAINER_IMAGE")
    @classmethod
    def _pinned_image(cls, image: str | None) -> str | None:
        if image is not None and not _IMAGE.fullmatch(image):
            raise ValueError("must be name@sha256:<64 hex digits>")
        return image


def _reason(error) -> str:
    if error["type"] == "missing":
        return "required but not set"
    # Pydantic messages describe the constraint, never the input (hide_input_in_errors).
    return error["msg"].removeprefix("Value error, ")


def _validate(source: Mapping[str, str]) -> tuple[Settings | None, str]:
    """Return (settings, "") or (None, problems); secrets stay in this frame, never in a raise."""
    values = {
        name: source[name].strip()
        for name in Settings.model_fields
        if name in source and source[name].strip()
    }
    try:
        return Settings.model_validate(values), ""
    except ValidationError as exc:
        found = [f"{e['loc'][0]}: {_reason(e)}" for e in exc.errors()]
    # A key that failed validation never reaches the https check; report both together.
    for key_name, url_name in (("QDRANT_API_KEY", "QDRANT_URL"), ("NOMAD_TOKEN", "NOMAD_ADDR")):
        url = values.get(url_name)
        if (
            any(m.startswith(f"{key_name}:") for m in found)
            and not any(m.startswith(f"{url_name}:") for m in found)
            and isinstance(url, str)
            and not url_problem(url, allow_query=False)
            and not url.lower().startswith("https://")
        ):
            found.append(f"{url_name}: must be https when {key_name} is set")
    return None, "; ".join(found)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (default: os.environ); blank values count as unset."""
    settings, problems = _validate(os.environ if env is None else env)
    if settings is None:
        # No secret-bearing local here, so traceback locals capture cannot reach a key.
        raise ConfigError(f"invalid configuration: {problems}")
    return settings
