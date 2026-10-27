"""Environment-validated settings (rule 4): the only place hosts, limits and keys come from.

Errors name every bad variable and the reason, never a value.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from normalizer import ConfigError

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    ANTHROPIC_API_KEY: SecretStr
    RAG_PROVIDER: Literal["anthropic"] = "anthropic"
    RAG_MODEL: str = "claude-sonnet-5-5"
    RAG_MAX_TOKENS: int = Field(default=1024, gt=0)
    RAG_PROVIDER_TIMEOUT_S: PositiveFloat = 30

    # Declared before QDRANT_URL so its validator can see whether a key is set.
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_URL: str
    QDRANT_COLLECTION: str
    QDRANT_TIMEOUT_S: PositiveFloat = 10

    RAG_TOP_K: int = Field(default=5, ge=1, le=50)
    RAG_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    RAG_EMBED_CACHE_DIR: Path | None = None

    @field_validator("QDRANT_URL")
    @classmethod
    def _http_url(cls, url: str, info: ValidationInfo) -> str:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be an http(s) URL with a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("must not embed credentials")
        if info.data.get("QDRANT_API_KEY") is not None and parts.scheme != "https":
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
        raise ConfigError(f"invalid configuration: {problems}") from None
