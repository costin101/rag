import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import ConfigError, Settings, load_settings
from normalizer import ErrorCode, RAGError

API_KEY = "test-not-a-key-anthropic"
QDRANT_KEY = "test-not-a-key-qdrant"

REQUIRED = ("ANTHROPIC_API_KEY", "QDRANT_URL", "QDRANT_COLLECTION")


def valid_env(**overrides):
    env = {
        "ANTHROPIC_API_KEY": API_KEY,
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "docs",
    }
    env.update(overrides)
    return env


def error_for(env):
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env)
    return excinfo.value


# --- defaults and happy path -------------------------------------------------


def test_minimal_env_applies_defaults():
    s = load_settings(valid_env())
    assert s.RAG_PROVIDER == "anthropic"
    assert s.RAG_MODEL == "claude-sonnet-5-5"
    assert s.RAG_MAX_TOKENS == 1024
    assert s.RAG_PROVIDER_TIMEOUT_S == 30
    assert s.QDRANT_TIMEOUT_S == 10
    assert s.RAG_TOP_K == 5
    assert s.RAG_EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert s.RAG_EMBED_CACHE_DIR is None
    assert s.QDRANT_API_KEY is None
    assert s.ANTHROPIC_API_KEY.get_secret_value() == API_KEY
    assert s.QDRANT_URL == "http://localhost:6333"
    assert s.QDRANT_COLLECTION == "docs"


def test_all_fields_overridable():
    s = load_settings(
        valid_env(
            QDRANT_URL="https://qdrant.example.com:6333",
            QDRANT_API_KEY=QDRANT_KEY,
            RAG_MODEL="other-model",
            RAG_MAX_TOKENS="256",
            RAG_PROVIDER_TIMEOUT_S="2.5",
            QDRANT_TIMEOUT_S="1.5",
            RAG_TOP_K="7",
            RAG_EMBED_MODEL="some/embedder",
            RAG_EMBED_CACHE_DIR="/tmp/embed-cache",
        )
    )
    assert s.QDRANT_API_KEY.get_secret_value() == QDRANT_KEY
    assert s.RAG_MODEL == "other-model"
    assert s.RAG_MAX_TOKENS == 256
    assert s.RAG_PROVIDER_TIMEOUT_S == 2.5
    assert s.QDRANT_TIMEOUT_S == 1.5
    assert s.RAG_TOP_K == 7
    assert s.RAG_EMBED_MODEL == "some/embedder"
    assert s.RAG_EMBED_CACHE_DIR == Path("/tmp/embed-cache")


def test_unrelated_env_vars_are_ignored():
    s = load_settings(valid_env(PATH="/usr/bin", HOME="/home/x"))
    assert isinstance(s, Settings)


def test_load_settings_defaults_to_process_environment(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "from-os-environ")
    assert load_settings().QDRANT_COLLECTION == "from-os-environ"


def test_values_are_stripped():
    s = load_settings(valid_env(QDRANT_COLLECTION="  docs\n", RAG_TOP_K=" 3 "))
    assert s.QDRANT_COLLECTION == "docs"
    assert s.RAG_TOP_K == 3


def test_settings_are_frozen():
    s = load_settings(valid_env())
    with pytest.raises(ValidationError):
        s.RAG_TOP_K = 9


# --- missing / empty ---------------------------------------------------------


def test_all_missing_required_vars_reported_together():
    err = error_for({})
    for name in REQUIRED:
        assert f"{name}: required but not set" in str(err)
    assert isinstance(err, RAGError)
    assert err.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("blank", ["", " ", "\t\n  "])
def test_blank_required_counts_as_missing_and_all_reported(blank):
    err = error_for({name: blank for name in REQUIRED})
    for name in REQUIRED:
        assert name in str(err)


@pytest.mark.parametrize("missing", REQUIRED)
def test_each_required_var_individually_reported(missing):
    env = valid_env()
    del env[missing]
    err = error_for(env)
    assert missing in str(err)
    others = [n for n in REQUIRED if n != missing]
    assert not any(n in str(err) for n in others)


def test_blank_optional_falls_back_to_default():
    s = load_settings(valid_env(RAG_TOP_K="  ", RAG_MODEL="", QDRANT_API_KEY=" "))
    assert s.RAG_TOP_K == 5
    assert s.RAG_MODEL == "claude-sonnet-5-5"
    assert s.QDRANT_API_KEY is None


def test_missing_and_invalid_reported_in_one_error():
    err = error_for({"QDRANT_URL": "ftp://nope", "RAG_TOP_K": "0"})
    for name in ("ANTHROPIC_API_KEY", "QDRANT_COLLECTION", "QDRANT_URL", "RAG_TOP_K"):
        assert name in str(err)


# --- numeric bounds ----------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", "51", "true"])
def test_invalid_top_k(bad):
    err = error_for(valid_env(RAG_TOP_K=bad))
    assert "RAG_TOP_K" in str(err)


@pytest.mark.parametrize("good", ["1", "50"])
def test_top_k_bounds_inclusive(good):
    assert load_settings(valid_env(RAG_TOP_K=good)).RAG_TOP_K == int(good)


@pytest.mark.parametrize("name", ["RAG_MAX_TOKENS"])
@pytest.mark.parametrize("bad", ["0", "-5", "abc", "1.5"])
def test_invalid_positive_int(name, bad):
    assert name in str(error_for(valid_env(**{name: bad})))


@pytest.mark.parametrize("name", ["RAG_PROVIDER_TIMEOUT_S", "QDRANT_TIMEOUT_S"])
@pytest.mark.parametrize("bad", ["0", "-1", "abc", "nan", "inf", "-inf"])
def test_invalid_positive_float(name, bad):
    assert name in str(error_for(valid_env(**{name: bad})))


def test_timeouts_must_be_finite_numbers():
    s = load_settings(valid_env(RAG_PROVIDER_TIMEOUT_S="0.5"))
    assert math.isfinite(s.RAG_PROVIDER_TIMEOUT_S)


def test_unknown_provider_rejected():
    assert "RAG_PROVIDER" in str(error_for(valid_env(RAG_PROVIDER="openai")))


# --- URLs --------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "ftp://host",
        "localhost:6333",
        "qdrant.local:6333",
        "http://",
        "https://",
        "//host",
        "not a url",
        "file:///etc/passwd",
        "http://user:pw@host:6333",
    ],
)
def test_invalid_qdrant_url(bad):
    assert "QDRANT_URL" in str(error_for(valid_env(QDRANT_URL=bad)))


@pytest.mark.parametrize("good", ["http://localhost:6333", "https://q.example.com"])
def test_valid_qdrant_url(good):
    assert load_settings(valid_env(QDRANT_URL=good)).QDRANT_URL == good


def test_https_required_when_qdrant_api_key_set():
    err = error_for(valid_env(QDRANT_URL="http://q.example.com", QDRANT_API_KEY=QDRANT_KEY))
    assert "QDRANT_URL: must be https when QDRANT_API_KEY is set" in str(err)


def test_https_with_qdrant_api_key_accepted():
    s = load_settings(valid_env(QDRANT_URL="https://q.example.com", QDRANT_API_KEY=QDRANT_KEY))
    assert s.QDRANT_API_KEY.get_secret_value() == QDRANT_KEY


def test_http_without_qdrant_api_key_accepted():
    assert load_settings(valid_env(QDRANT_URL="http://q.example.com")).QDRANT_API_KEY is None


# --- no value leakage --------------------------------------------------------


def test_secret_values_never_in_error():
    env = valid_env(
        ANTHROPIC_API_KEY="test-not-a-key-LEAK-anthropic",
        QDRANT_API_KEY="test-not-a-key-LEAK-qdrant",
        QDRANT_URL="http://q.example.com",  # invalid: http while a key is set
        RAG_TOP_K="LEAK-top-k",
    )
    err = error_for(env)
    text = f"{err} {err!r} {err.args}"
    assert "LEAK" not in text
    assert err.__cause__ is None


@pytest.mark.parametrize(
    "name,bad",
    [
        ("QDRANT_URL", "ftp://LEAK-host"),
        ("RAG_TOP_K", "LEAK"),
        ("RAG_MAX_TOKENS", "LEAK"),
        ("RAG_PROVIDER_TIMEOUT_S", "LEAK"),
        ("QDRANT_TIMEOUT_S", "LEAK"),
        ("RAG_PROVIDER", "LEAK"),
    ],
)
def test_bad_values_never_echoed(name, bad):
    err = error_for(valid_env(**{name: bad}))
    assert name in str(err)
    assert "LEAK" not in f"{err} {err!r}"


def test_url_credentials_never_echoed():
    err = error_for(valid_env(QDRANT_URL="http://user:LEAKPW@host:6333"))
    assert "LEAKPW" not in f"{err} {err!r}"


def test_settings_repr_and_str_hide_secrets():
    s = load_settings(
        valid_env(
            ANTHROPIC_API_KEY="test-not-a-key-LEAK-anthropic",
            QDRANT_URL="https://q.example.com",
            QDRANT_API_KEY="test-not-a-key-LEAK-qdrant",
        )
    )
    assert "LEAK" not in repr(s)
    assert "LEAK" not in str(s)


def test_config_error_is_config_invalid():
    err = error_for({})
    assert str(err).startswith("config_invalid: ")
