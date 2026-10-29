import os
import traceback
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_REVISION, ConfigError, Settings, load_settings
from normalizer import ErrorCode, RAGError

API_KEY = "test-not-a-key-anthropic"
QDRANT_KEY = "test-not-a-key-qdrant"

PIN_A = "a" * 40
PIN_B = "b" * 40

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
    assert s.RAG_PROVIDER_MAX_RETRIES == 2
    assert s.RAG_PROVIDER_MAX_RETRY_WAIT_S == 60
    assert s.QDRANT_TIMEOUT_S == 10
    assert s.RAG_TOP_K == 5
    assert s.RAG_EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert s.RAG_EMBED_CACHE_DIR is None
    assert s.RAG_EMBED_REVISION == DEFAULT_EMBED_REVISION
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
            RAG_PROVIDER_MAX_RETRIES="4",
            RAG_PROVIDER_MAX_RETRY_WAIT_S="7.5",
            QDRANT_TIMEOUT_S="1.5",
            RAG_TOP_K="7",
            RAG_EMBED_MODEL="some/embedder",
            RAG_EMBED_CACHE_DIR="/tmp/embed-cache",
            RAG_EMBED_REVISION=PIN_A,
        )
    )
    assert s.QDRANT_API_KEY.get_secret_value() == QDRANT_KEY
    assert s.RAG_MODEL == "other-model"
    assert s.RAG_MAX_TOKENS == 256
    assert s.RAG_PROVIDER_TIMEOUT_S == 2.5
    assert s.RAG_PROVIDER_MAX_RETRIES == 4
    assert s.RAG_PROVIDER_MAX_RETRY_WAIT_S == 7.5
    assert s.QDRANT_TIMEOUT_S == 1.5
    assert s.RAG_TOP_K == 7
    assert s.RAG_EMBED_MODEL == "some/embedder"
    assert s.RAG_EMBED_CACHE_DIR == Path("/tmp/embed-cache")
    assert s.RAG_EMBED_REVISION == PIN_A


def test_unrelated_env_vars_are_filtered_out_before_validation():
    # Settings forbids extras, so an unfiltered environment would raise.
    s = load_settings(valid_env(PATH="/usr/bin", RAG_BOGUS="x", QDRANT_URL_EXTRA="y"))
    assert set(s.model_dump()) == set(Settings.model_fields)


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
    s = load_settings(
        valid_env(RAG_TOP_K="  ", RAG_MODEL="", QDRANT_API_KEY=" ", RAG_EMBED_REVISION="\t")
    )
    assert s.RAG_EMBED_REVISION == DEFAULT_EMBED_REVISION
    assert s.RAG_TOP_K == 5
    assert s.RAG_MODEL == "claude-sonnet-5-5"
    assert s.QDRANT_API_KEY is None


def test_missing_and_invalid_reported_in_one_error():
    err = error_for({"QDRANT_URL": "ftp://nope", "RAG_TOP_K": "0"})
    for name in ("ANTHROPIC_API_KEY", "QDRANT_COLLECTION", "QDRANT_URL", "RAG_TOP_K"):
        assert name in str(err)


# --- numeric bounds ----------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", "51", "true", "1_0", "+5", "１０", "0x5"])
def test_invalid_top_k(bad):
    err = error_for(valid_env(RAG_TOP_K=bad))
    assert "RAG_TOP_K" in str(err)


@pytest.mark.parametrize("good", ["1", "50"])
def test_top_k_bounds_inclusive(good):
    assert load_settings(valid_env(RAG_TOP_K=good)).RAG_TOP_K == int(good)


@pytest.mark.parametrize("name", ["RAG_MAX_TOKENS"])
@pytest.mark.parametrize("bad", ["0", "-5", "abc", "1.5", "1_0", "1_000", "200001", "99999999999"])
def test_invalid_positive_int(name, bad):
    assert name in str(error_for(valid_env(**{name: bad})))


@pytest.mark.parametrize("bad", ["-1", "abc", "1.5", "1_0", "+5", "11", "true", "１", "99999999999"])
def test_invalid_max_retries(bad):
    assert "RAG_PROVIDER_MAX_RETRIES" in str(error_for(valid_env(RAG_PROVIDER_MAX_RETRIES=bad)))


@pytest.mark.parametrize("good", ["0", "10"])
def test_max_retries_bounds_inclusive(good):
    s = load_settings(valid_env(RAG_PROVIDER_MAX_RETRIES=good))
    assert s.RAG_PROVIDER_MAX_RETRIES == int(good)


@pytest.mark.parametrize(
    "name", ["RAG_PROVIDER_TIMEOUT_S", "RAG_PROVIDER_MAX_RETRY_WAIT_S", "QDRANT_TIMEOUT_S"]
)
@pytest.mark.parametrize(
    "bad",
    ["0", "-1", "abc", "nan", "inf", "-inf", "1_0", "1_0.5", "1e3", "1e0", "1e-1", "+5", "5.", ".5", "601"],
)
def test_invalid_positive_float(name, bad):
    assert name in str(error_for(valid_env(**{name: bad})))


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
        "https://TOKEN@host",
        "https://@host",
        "https://host:6333/?api-key=KEY",
        "https://host:6333?",
        "https://host#frag",
        "http://host:abc",
        "http://host:99999",
        "http://host:0",
        "http://host :6333",
        "http://ho st",
        "http://host:6333,http://other:6333",
        "http://[::1",
        "http://host\x00",
        "http://host:",
        "http://host:/path",
        "https://host:6333/path;param",
        "https://host;x:6333",
        "https://host\u200b:6333",
    ],
)
def test_invalid_qdrant_url(bad):
    assert "QDRANT_URL" in str(error_for(valid_env(QDRANT_URL=bad)))


@pytest.mark.parametrize("good", ["http://localhost:6333", "https://q.example.com"])
def test_valid_qdrant_url(good):
    assert load_settings(valid_env(QDRANT_URL=good)).QDRANT_URL == good


def test_url_reasons_are_distinct_per_problem():
    cases = {
        "ftp://host": "http(s)",
        "https://TOKEN@host": "credentials",
        "https://host/?k=v": "query",
        "http://host:abc": "port",
    }
    for url, word in cases.items():
        assert word in str(error_for(valid_env(QDRANT_URL=url)))


def test_https_required_when_qdrant_api_key_set():
    err = error_for(valid_env(QDRANT_URL="http://q.example.com", QDRANT_API_KEY=QDRANT_KEY))
    assert "QDRANT_URL: must be https when QDRANT_API_KEY is set" in str(err)


def test_https_with_qdrant_api_key_accepted():
    s = load_settings(valid_env(QDRANT_URL="https://q.example.com", QDRANT_API_KEY=QDRANT_KEY))
    assert s.QDRANT_API_KEY.get_secret_value() == QDRANT_KEY


def test_https_required_even_when_host_contains_https():
    err = error_for(valid_env(QDRANT_URL="http://https.internal:6333", QDRANT_API_KEY=QDRANT_KEY))
    assert "QDRANT_URL: must be https when QDRANT_API_KEY is set" in str(err)


def test_https_check_is_scheme_case_insensitive():
    s = load_settings(valid_env(QDRANT_URL="HTTPS://q.example.com", QDRANT_API_KEY=QDRANT_KEY))
    assert s.QDRANT_URL == "HTTPS://q.example.com"


def test_http_without_qdrant_api_key_accepted():
    assert load_settings(valid_env(QDRANT_URL="http://q.example.com")).QDRANT_API_KEY is None


# --- no value leakage --------------------------------------------------------


def full_trace(err):
    return "".join(traceback.format_exception(err))


def test_secret_values_never_in_error():
    env = valid_env(
        QDRANT_API_KEY="test-not-a-key-LEAK-qdrant",
        QDRANT_URL="http://q.example.com",  # invalid: http while a key is set
        RAG_TOP_K="LEAK-top-k",
    )
    err = error_for(env)
    text = f"{err} {err!r} {err.args} {full_trace(err)}"
    assert "LEAK" not in text


def test_secrets_not_reachable_from_traceback_locals(monkeypatch):
    # Locals-capturing reporters (Sentry, rich show_locals, pytest --showlocals) walk the frames.
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key-LEAK-anthropic")
    monkeypatch.setenv("QDRANT_URL", "http://q.example.com")
    monkeypatch.setenv("QDRANT_COLLECTION", "docs")
    monkeypatch.setenv("QDRANT_API_KEY", "test-not-a-key-LEAK-qdrant")
    with pytest.raises(ConfigError) as excinfo:
        load_settings()
    trace = traceback.TracebackException.from_exception(excinfo.value, capture_locals=True)
    assert "LEAK" not in "".join(trace.format())
    assert os.environ["ANTHROPIC_API_KEY"]  # the env was really populated


def test_error_carries_no_exception_context():
    err = error_for(valid_env(RAG_TOP_K="LEAK"))
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest.mark.parametrize(
    "url",
    [
        "http://user:LEAKPW@host:6333",
        "https://LEAKTOKEN@host",
        "https://host:6333/?api-key=LEAKQ",
        "https://host#LEAKFRAG",
        "http://host:LEAKPORT",
    ],
)
def test_bad_url_never_echoed_anywhere_in_traceback(url):
    err = error_for(valid_env(QDRANT_URL=url, QDRANT_API_KEY="test-not-a-key-LEAK-key"))
    text = f"{err} {err!r} {full_trace(err)}"
    assert "LEAK" not in text


@pytest.mark.parametrize(
    "name,bad",
    [
        ("QDRANT_URL", "ftp://LEAK-host"),
        ("RAG_TOP_K", "LEAK"),
        ("RAG_MAX_TOKENS", "LEAK"),
        ("RAG_PROVIDER_MAX_RETRIES", "LEAK"),
        ("RAG_PROVIDER_TIMEOUT_S", "LEAK"),
        ("RAG_PROVIDER_MAX_RETRY_WAIT_S", "LEAK"),
        ("QDRANT_TIMEOUT_S", "LEAK"),
        ("RAG_PROVIDER", "LEAK"),
        ("RAG_EMBED_CACHE_DIR", "LEAK\x00"),
    ],
)
def test_bad_values_never_echoed(name, bad):
    err = error_for(valid_env(**{name: bad}))
    assert name in str(err)
    assert "LEAK" not in f"{err} {err!r} {full_trace(err)}"


def test_settings_validation_error_hides_input_when_validated_directly():
    # hide_input_in_errors must be set on Settings itself, not only via load_settings.
    with pytest.raises(ValidationError) as excinfo:
        Settings.model_validate({**valid_env(), "RAG_TOP_K": "MARKERVALUE"})
    assert "MARKERVALUE" not in str(excinfo.value)
    assert "MARKERVALUE" not in repr(excinfo.value)


def test_settings_config_hides_input():
    assert Settings.model_config["hide_input_in_errors"] is True


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


# --- strict numeric parsing, cache dir, revision -----------------------------


def test_cache_dir_rejects_nul_byte():
    assert "RAG_EMBED_CACHE_DIR" in str(error_for(valid_env(RAG_EMBED_CACHE_DIR="/tmp/a\x00b")))


def test_embed_revision_defaults_to_pinned_hash_for_default_model():
    assert DEFAULT_EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert len(DEFAULT_EMBED_REVISION) == 40
    assert set(DEFAULT_EMBED_REVISION) <= set("0123456789abcdef")
    s = load_settings(valid_env())
    assert s.RAG_EMBED_MODEL == DEFAULT_EMBED_MODEL
    assert s.RAG_EMBED_REVISION == DEFAULT_EMBED_REVISION


def test_embed_revision_required_for_non_default_model():
    err = error_for(valid_env(RAG_EMBED_MODEL="some/embedder"))
    assert "RAG_EMBED_REVISION" in str(err)
    assert "some/embedder" not in str(err)
    assert load_settings(
        valid_env(RAG_EMBED_MODEL="some/embedder", RAG_EMBED_REVISION=PIN_A)
    ).RAG_EMBED_REVISION == PIN_A


def test_embed_revision_required_on_direct_construction_for_non_default_model():
    with pytest.raises(ValidationError):
        Settings(
            ANTHROPIC_API_KEY=API_KEY,
            QDRANT_URL="http://localhost:6333",
            QDRANT_COLLECTION="docs",
            RAG_EMBED_MODEL="some/embedder",
        )


def test_embed_revision_can_override_the_default():
    assert load_settings(valid_env(RAG_EMBED_REVISION=PIN_A)).RAG_EMBED_REVISION == PIN_A
    assert load_settings(valid_env(RAG_EMBED_REVISION=f" {PIN_B} ")).RAG_EMBED_REVISION == PIN_B


@pytest.mark.parametrize("good", ["0.5", "2", "30.25"])
def test_plain_decimal_timeouts_accepted(good):
    assert load_settings(valid_env(QDRANT_TIMEOUT_S=good)).QDRANT_TIMEOUT_S == float(good)


@pytest.mark.parametrize("good", ["0.5", "2", "30.25"])
def test_plain_decimal_retry_wait_accepted(good):
    s = load_settings(valid_env(RAG_PROVIDER_MAX_RETRY_WAIT_S=good))
    assert s.RAG_PROVIDER_MAX_RETRY_WAIT_S == float(good)


# --- upper bounds ------------------------------------------------------------


def test_max_tokens_upper_bound_inclusive():
    assert load_settings(valid_env(RAG_MAX_TOKENS="200000")).RAG_MAX_TOKENS == 200000


@pytest.mark.parametrize(
    "name", ["RAG_PROVIDER_TIMEOUT_S", "RAG_PROVIDER_MAX_RETRY_WAIT_S", "QDRANT_TIMEOUT_S"]
)
def test_timeout_upper_bound_inclusive(name):
    assert getattr(load_settings(valid_env(**{name: "600"})), name) == 600
    assert name in str(error_for(valid_env(**{name: "600.01"})))


# --- API keys: no hidden characters ------------------------------------------

KEY_NAMES = ("ANTHROPIC_API_KEY", "QDRANT_API_KEY")
HIDDEN_CHARS = {
    "newline": "\n",
    "tab": "\t",
    "carriage return": "\r",
    "space": " ",
    "nbsp": "\u00a0",
    "line separator": "\u2028",
    "nul (Cc)": "\x00",
    "del (Cc)": "\x7f",
    "zero-width space (Cf)": "\u200b",
    "bom (Cf)": "\ufeff",
}


@pytest.mark.parametrize("key_name", KEY_NAMES)
@pytest.mark.parametrize("char", HIDDEN_CHARS.values(), ids=HIDDEN_CHARS.keys())
def test_api_key_with_interior_hidden_character_rejected(key_name, char):
    env = valid_env(QDRANT_URL="https://q.example.com", QDRANT_API_KEY="test-not-a-key-ok")
    env[key_name] = f"LEAK{char}SECRET"
    err = error_for(env)
    assert key_name in str(err)
    assert "must not contain whitespace or control characters" in str(err)
    other = next(n for n in KEY_NAMES if n != key_name)
    assert other not in str(err)
    text = f"{err} {err!r} {full_trace(err)}"
    assert "LEAK" not in text and "SECRET" not in text


@pytest.mark.parametrize("key_name", KEY_NAMES)
@pytest.mark.parametrize("char", HIDDEN_CHARS.values(), ids=HIDDEN_CHARS.keys())
def test_api_key_hidden_character_rejected_on_direct_construction(key_name, char):
    fields = {
        "ANTHROPIC_API_KEY": "test-not-a-key",
        "QDRANT_API_KEY": "test-not-a-key",
        "QDRANT_URL": "https://q.example.com",
        "QDRANT_COLLECTION": "docs",
    }
    fields[key_name] = f"LEAK{char}SECRET"
    with pytest.raises(ValidationError) as excinfo:
        Settings(**fields)
    assert "LEAK" not in str(excinfo.value)


@pytest.mark.parametrize("key_name", KEY_NAMES)
@pytest.mark.parametrize("blank", ["", " ", "\u200b"])
def test_api_key_empty_on_direct_construction_rejected(key_name, blank):
    fields = {
        "ANTHROPIC_API_KEY": "test-not-a-key",
        "QDRANT_API_KEY": "test-not-a-key",
        "QDRANT_URL": "https://q.example.com",
        "QDRANT_COLLECTION": "docs",
    }
    fields[key_name] = blank
    with pytest.raises(ValidationError):
        Settings(**fields)


def test_api_key_edge_whitespace_is_stripped_not_rejected():
    s = load_settings(valid_env(ANTHROPIC_API_KEY="  test-not-a-key\n"))
    assert s.ANTHROPIC_API_KEY.get_secret_value() == "test-not-a-key"


# --- direct construction is strict -------------------------------------------

DIRECT = {
    "ANTHROPIC_API_KEY": "test-not-a-key",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION": "docs",
}


@pytest.mark.parametrize(
    "name,bad",
    [
        ("RAG_TOP_K", True),
        ("RAG_TOP_K", 2.0),
        ("RAG_MAX_TOKENS", True),
        ("RAG_MAX_TOKENS", 2.0),
        ("RAG_PROVIDER_MAX_RETRIES", True),
        ("RAG_PROVIDER_MAX_RETRIES", 2.0),
        ("RAG_PROVIDER_TIMEOUT_S", True),
        ("RAG_PROVIDER_MAX_RETRY_WAIT_S", True),
        ("QDRANT_TIMEOUT_S", True),
        ("QDRANT_TIMEOUT_S", False),
    ],
)
def test_direct_construction_rejects_bool_and_float_for_int(name, bad):
    with pytest.raises(ValidationError):
        Settings(**DIRECT, **{name: bad})


def test_direct_construction_accepts_native_numbers_and_numeric_strings():
    s = Settings(**DIRECT, RAG_TOP_K=7, QDRANT_TIMEOUT_S=2, RAG_MAX_TOKENS="9")
    assert (s.RAG_TOP_K, s.QDRANT_TIMEOUT_S, s.RAG_MAX_TOKENS) == (7, 2.0, 9)


# --- text fields (review PR #17) ----------------------------------------------

TEXT_FIELDS = ("RAG_MODEL", "QDRANT_COLLECTION", "RAG_EMBED_MODEL", "RAG_EMBED_REVISION")


@pytest.mark.parametrize("name", TEXT_FIELDS)
@pytest.mark.parametrize("bad", ["a\r\nX-Injected: 1", "a b", "a\x00b", "a​b"])
def test_text_fields_reject_hidden_characters(name, bad):
    err = error_for(valid_env(**{name: bad}))
    assert name in str(err)
    assert bad not in str(err)


@pytest.mark.parametrize("name", TEXT_FIELDS)
def test_text_fields_reject_empty_on_direct_construction(name):
    values = {
        "ANTHROPIC_API_KEY": API_KEY,
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "docs",
        name: "",
    }
    with pytest.raises(ValidationError):
        Settings(**values)


@pytest.mark.parametrize("bad", ["../points", "a/b", "a?x=1", "a#b", "..", ".", "a..b", "a%2e"])
def test_collection_rejects_path_unsafe_names(bad):
    err = error_for(valid_env(QDRANT_COLLECTION=bad))
    assert "QDRANT_COLLECTION" in str(err)


@pytest.mark.parametrize("ok", ["docs", "my-docs_v2", "docs.v1", "Docs123"])
def test_collection_accepts_plain_names(ok):
    assert load_settings(valid_env(QDRANT_COLLECTION=ok)).QDRANT_COLLECTION == ok


# --- API keys: printable ASCII only ------------------------------------------

NON_ASCII_CHARS = {
    "left curly quote (Pi)": "\u201c",
    "private use (Co)": "\ue000",
    "lone surrogate (Cs)": "\udcff",
    "cyrillic a": "\u0430",
    "e acute": "\u00e9",
}


@pytest.mark.parametrize("key_name", KEY_NAMES)
@pytest.mark.parametrize("char", NON_ASCII_CHARS.values(), ids=NON_ASCII_CHARS.keys())
def test_api_key_with_non_ascii_character_rejected(key_name, char):
    env = valid_env(QDRANT_URL="https://q.example.com", QDRANT_API_KEY="test-not-a-key-ok")
    env[key_name] = f"LEAK{char}SECRET"
    err = error_for(env)
    assert f"{key_name}: must contain only printable ASCII characters" in str(err)
    text = f"{err} {err!r} {full_trace(err)}"
    assert "LEAK" not in text and "SECRET" not in text


def test_settings_reject_unknown_fields_on_direct_construction():
    with pytest.raises(ValidationError):
        Settings(
            ANTHROPIC_API_KEY=API_KEY,
            QDRANT_URL="http://localhost:6333",
            QDRANT_COLLECTION="docs",
            QDRANT_APIKEY="typo",
        )


def test_timeout_defaults_are_floats():
    s = load_settings(valid_env())
    assert isinstance(s.RAG_PROVIDER_TIMEOUT_S, float)
    assert isinstance(s.RAG_PROVIDER_MAX_RETRY_WAIT_S, float)
    assert isinstance(s.QDRANT_TIMEOUT_S, float)


@pytest.mark.parametrize("bad", ["main", "v1.0", "abc123", "A" * 40, "a" * 39, "a" * 41, "g" * 40])
def test_embed_revision_rejects_non_commit_values(bad):
    err = error_for(valid_env(RAG_EMBED_MODEL="some/embedder", RAG_EMBED_REVISION=bad))
    assert "RAG_EMBED_REVISION: must be a 40-character lowercase hex commit hash" in str(err)
    assert bad not in str(err)


def test_invalid_embed_model_does_not_add_a_revision_error():
    err = error_for(valid_env(RAG_EMBED_MODEL="a b"))
    assert "RAG_EMBED_MODEL" in str(err)
    assert "RAG_EMBED_REVISION" not in str(err)


def test_bad_qdrant_key_and_plain_http_url_are_reported_together():
    err = error_for(valid_env(QDRANT_URL="http://h.example.com", QDRANT_API_KEY="bad key"))
    assert "QDRANT_API_KEY" in str(err)
    assert "QDRANT_URL: must be https when QDRANT_API_KEY is set" in str(err)


# --- embed endpoint (red team: ambient HF_ENDPOINT must not pick the download host) ---


def test_embed_endpoint_defaults_to_the_public_hub():
    assert load_settings(valid_env()).RAG_EMBED_ENDPOINT == "https://huggingface.co"


def test_embed_endpoint_can_be_overridden_with_https():
    s = load_settings(valid_env(RAG_EMBED_ENDPOINT="https://hub.example.com"))
    assert s.RAG_EMBED_ENDPOINT == "https://hub.example.com"


@pytest.mark.parametrize(
    "bad",
    [
        "http://hub.example.com",
        "ftp://hub.example.com",
        "hub.example.com",
        "https://TOKEN@hub.example.com",
        "https://hub.example.com/?k=v",
        "https://hub.example.com#frag",
        "https://hub.example.com:abc",
        "https://hub​.example.com",
    ],
)
def test_invalid_embed_endpoint(bad):
    err = error_for(valid_env(RAG_EMBED_ENDPOINT=bad))
    assert "RAG_EMBED_ENDPOINT" in str(err)
    assert "TOKEN" not in f"{err} {full_trace(err)}"


def test_embed_endpoint_must_be_https_case_insensitively():
    s = load_settings(valid_env(RAG_EMBED_ENDPOINT="HTTPS://hub.example.com"))
    assert s.RAG_EMBED_ENDPOINT == "HTTPS://hub.example.com"


# --- provider base URL (red team: ambient ANTHROPIC_BASE_URL must not pick the key's host) ---


def test_provider_base_url_defaults_to_the_public_api():
    assert load_settings(valid_env()).RAG_PROVIDER_BASE_URL == "https://api.anthropic.com"


def test_provider_base_url_can_be_overridden_with_https():
    s = load_settings(valid_env(RAG_PROVIDER_BASE_URL="https://gateway.example.com/anthropic"))
    assert s.RAG_PROVIDER_BASE_URL == "https://gateway.example.com/anthropic"


@pytest.mark.parametrize(
    "bad",
    [
        "http://gateway.example.com",
        "ftp://gateway.example.com",
        "gateway.example.com",
        "https://TOKEN@gateway.example.com",
        "https://gateway.example.com/?k=v",
        "https://gateway.example.com#frag",
        "https://gateway.example.com:abc",
    ],
)
def test_invalid_provider_base_url(bad):
    err = error_for(valid_env(RAG_PROVIDER_BASE_URL=bad))
    assert "RAG_PROVIDER_BASE_URL" in str(err)
    assert "TOKEN" not in f"{err} {full_trace(err)}"


def test_provider_base_url_must_be_https_case_insensitively():
    s = load_settings(valid_env(RAG_PROVIDER_BASE_URL="HTTPS://gateway.example.com"))
    assert s.RAG_PROVIDER_BASE_URL == "HTTPS://gateway.example.com"


# --- container path (all optional: the API path runs without any of them) ---------

DIGEST = "a" * 64
IMAGE = f"registry.example.com:5000/rag/session@sha256:{DIGEST}"
NOMAD_TOKEN = "test-not-a-key-nomad"

CONTAINER_NAMES = (
    "NOMAD_ADDR",
    "NOMAD_TOKEN",
    "NOMAD_CACERT",
    "NOMAD_NAMESPACE",
    "RAG_CONTAINER_IMAGE",
    "RAG_CONTAINER_CPU_MHZ",
    "RAG_CONTAINER_MEMORY_MB",
    "RAG_CONTAINER_TIMEOUT_S",
    "RAG_CONTAINER_MAX_INPUT_BYTES",
    "RAG_CONTAINER_MAX_OUTPUT_BYTES",
)


def test_container_settings_are_optional_and_default_to_unset_or_bounded_values():
    s = load_settings(valid_env())
    assert (s.NOMAD_ADDR, s.NOMAD_TOKEN, s.NOMAD_CACERT, s.NOMAD_NAMESPACE) == (None,) * 4
    assert s.RAG_CONTAINER_IMAGE is None
    assert s.RAG_CONTAINER_CPU_MHZ == 500
    assert s.RAG_CONTAINER_MEMORY_MB == 128
    assert s.RAG_CONTAINER_TIMEOUT_S == 60.0
    assert s.RAG_CONTAINER_MAX_INPUT_BYTES == 262_144
    assert s.RAG_CONTAINER_MAX_OUTPUT_BYTES == 65_536
    assert isinstance(s.RAG_CONTAINER_TIMEOUT_S, float)


@pytest.mark.parametrize("name", CONTAINER_NAMES)
def test_no_container_setting_is_required(name):
    assert name not in REQUIRED
    assert name in Settings.model_fields
    assert not Settings.model_fields[name].is_required()


def test_container_settings_overridable():
    s = load_settings(
        valid_env(
            NOMAD_ADDR="https://nomad.example.com:4646",
            NOMAD_TOKEN=NOMAD_TOKEN,
            NOMAD_CACERT="/etc/nomad/ca.pem",
            NOMAD_NAMESPACE="rag-sessions",
            RAG_CONTAINER_IMAGE=IMAGE,
            RAG_CONTAINER_CPU_MHZ="250",
            RAG_CONTAINER_MEMORY_MB="256",
            RAG_CONTAINER_TIMEOUT_S="12.5",
            RAG_CONTAINER_MAX_INPUT_BYTES="4096",
            RAG_CONTAINER_MAX_OUTPUT_BYTES="2048",
        )
    )
    assert s.NOMAD_ADDR == "https://nomad.example.com:4646"
    assert s.NOMAD_TOKEN.get_secret_value() == NOMAD_TOKEN
    assert s.NOMAD_CACERT == Path("/etc/nomad/ca.pem")
    assert s.NOMAD_NAMESPACE == "rag-sessions"
    assert s.RAG_CONTAINER_IMAGE == IMAGE
    assert (s.RAG_CONTAINER_CPU_MHZ, s.RAG_CONTAINER_MEMORY_MB) == (250, 256)
    assert s.RAG_CONTAINER_TIMEOUT_S == 12.5
    assert (s.RAG_CONTAINER_MAX_INPUT_BYTES, s.RAG_CONTAINER_MAX_OUTPUT_BYTES) == (4096, 2048)


def test_blank_container_settings_count_as_unset():
    s = load_settings(valid_env(**{name: " " for name in CONTAINER_NAMES}))
    assert s.NOMAD_ADDR is None and s.RAG_CONTAINER_IMAGE is None and s.NOMAD_TOKEN is None
    assert s.RAG_CONTAINER_MEMORY_MB == 128


@pytest.mark.parametrize(
    "bad",
    [
        "ftp://nomad",
        "nomad:4646",
        "http://",
        "https://TOKEN@nomad",
        "https://nomad:4646/?k=v",
        "https://nomad#frag",
        "http://nomad:abc",
        "http://no mad",
        "https://nomad​",
    ],
)
def test_invalid_nomad_addr(bad):
    err = error_for(valid_env(NOMAD_ADDR=bad))
    assert "NOMAD_ADDR" in str(err)
    assert "TOKEN@" not in f"{err} {full_trace(err)}"


@pytest.mark.parametrize("good", ["http://127.0.0.1:4646", "https://nomad.example.com:4646"])
def test_valid_nomad_addr_without_token(good):
    assert load_settings(valid_env(NOMAD_ADDR=good)).NOMAD_ADDR == good


def test_https_required_when_nomad_token_set():
    err = error_for(valid_env(NOMAD_ADDR="http://nomad.example.com:4646", NOMAD_TOKEN=NOMAD_TOKEN))
    assert "NOMAD_ADDR: must be https when NOMAD_TOKEN is set" in str(err)
    assert NOMAD_TOKEN not in f"{err} {full_trace(err)}"


def test_https_with_nomad_token_accepted_case_insensitively():
    s = load_settings(valid_env(NOMAD_ADDR="HTTPS://nomad.example.com", NOMAD_TOKEN=NOMAD_TOKEN))
    assert s.NOMAD_TOKEN.get_secret_value() == NOMAD_TOKEN


def test_bad_nomad_token_and_plain_http_addr_are_reported_together():
    err = error_for(valid_env(NOMAD_ADDR="http://nomad.example.com", NOMAD_TOKEN="bad token"))
    assert "NOMAD_TOKEN" in str(err)
    assert "NOMAD_ADDR: must be https when NOMAD_TOKEN is set" in str(err)


@pytest.mark.parametrize("char", HIDDEN_CHARS.values(), ids=HIDDEN_CHARS.keys())
def test_nomad_token_with_interior_hidden_character_rejected(char):
    err = error_for(valid_env(NOMAD_TOKEN=f"LEAK{char}SECRET"))
    assert "NOMAD_TOKEN: must not contain whitespace or control characters" in str(err)
    assert "LEAK" not in f"{err} {err!r} {full_trace(err)}"


@pytest.mark.parametrize("char", NON_ASCII_CHARS.values(), ids=NON_ASCII_CHARS.keys())
def test_nomad_token_with_non_ascii_character_rejected(char):
    err = error_for(valid_env(NOMAD_TOKEN=f"LEAK{char}SECRET"))
    assert "NOMAD_TOKEN: must contain only printable ASCII characters" in str(err)
    assert "LEAK" not in f"{err} {err!r} {full_trace(err)}"


def test_nomad_token_repr_hides_the_secret():
    s = load_settings(
        valid_env(NOMAD_ADDR="https://nomad.example.com", NOMAD_TOKEN="test-not-a-key-LEAK-nomad")
    )
    assert "LEAK" not in f"{s!r} {s}"


def test_nomad_token_without_addr_is_a_config_matter_for_the_container_path_not_for_load():
    assert load_settings(valid_env(NOMAD_TOKEN=NOMAD_TOKEN)).NOMAD_ADDR is None


def test_nomad_cacert_rejects_nul_byte():
    assert "NOMAD_CACERT" in str(error_for(valid_env(NOMAD_CACERT="/etc/a\x00b")))


@pytest.mark.parametrize("good", ["default", "rag-sessions", "A1", "a" * 128, "-", "9-9"])
def test_nomad_namespace_accepts_letters_digits_and_hyphen(good):
    assert load_settings(valid_env(NOMAD_NAMESPACE=good)).NOMAD_NAMESPACE == good


@pytest.mark.parametrize(
    "bad", ["a" * 129, "a b", "a/b", "a_b", "a.b", "a?x=1", "a#b", "a%2f", "..", "é", "a\x00b", "a​"]
)
def test_nomad_namespace_rejects_everything_else(bad):
    err = error_for(valid_env(NOMAD_NAMESPACE=bad))
    assert "NOMAD_NAMESPACE" in str(err)
    assert bad not in str(err)


@pytest.mark.parametrize("good", [IMAGE, f"python@sha256:{DIGEST}", f"127.0.0.1:5000/rag@sha256:{DIGEST}"])
def test_container_image_accepts_a_digest_pinned_reference(good):
    assert load_settings(valid_env(RAG_CONTAINER_IMAGE=good)).RAG_CONTAINER_IMAGE == good


@pytest.mark.parametrize(
    "bad",
    [
        "rag/session",
        "rag/session:latest",
        "rag/session:1.2.3",
        f"rag/session:latest@sha256:{DIGEST}",
        f"@sha256:{DIGEST}",
        "rag/session@sha256:" + "a" * 63,
        "rag/session@sha256:" + "a" * 65,
        "rag/session@sha256:" + "A" * 64,
        "rag/session@sha256:" + "g" * 64,
        "rag/session@sha512:" + "a" * 128,
        "rag/session@" + "a" * 64,
        f"-rag@sha256:{DIGEST}",
        f"Rag/Session@sha256:{DIGEST}",
        f"rag//session@sha256:{DIGEST}",
        f"rag/session @sha256:{DIGEST}",
        f"rag/session@sha256:{DIGEST}\n--privileged",
        f"https://rag/session@sha256:{DIGEST}",
        f"rag/session@sha256:{DIGEST}@sha256:{DIGEST}",
    ],
)
def test_container_image_rejects_anything_but_a_full_digest_pin(bad):
    err = error_for(valid_env(RAG_CONTAINER_IMAGE=bad))
    assert "RAG_CONTAINER_IMAGE: must be name@sha256:<64 hex digits>" in str(err)


@pytest.mark.parametrize(
    "name,low,high",
    [
        ("RAG_CONTAINER_CPU_MHZ", 1, 100_000),
        ("RAG_CONTAINER_MEMORY_MB", 64, 8192),
        ("RAG_CONTAINER_MAX_INPUT_BYTES", 1024, 1_048_576),
        ("RAG_CONTAINER_MAX_OUTPUT_BYTES", 1024, 1_048_576),
    ],
)
def test_container_whole_number_bounds_are_inclusive(name, low, high):
    assert getattr(load_settings(valid_env(**{name: str(low)})), name) == low
    assert getattr(load_settings(valid_env(**{name: str(high)})), name) == high
    for bad in (str(low - 1), str(high + 1)):
        assert name in str(error_for(valid_env(**{name: bad})))


@pytest.mark.parametrize(
    "name",
    [
        "RAG_CONTAINER_CPU_MHZ",
        "RAG_CONTAINER_MEMORY_MB",
        "RAG_CONTAINER_MAX_INPUT_BYTES",
        "RAG_CONTAINER_MAX_OUTPUT_BYTES",
    ],
)
@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5", "1_000", "+500", "１００", "1e3", "true"])
def test_container_whole_numbers_are_plain_decimals(name, bad):
    assert name in str(error_for(valid_env(**{name: bad})))


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "nan", "inf", "1e3", "+5", "5.", ".5", "601"])
def test_invalid_container_timeout(bad):
    assert "RAG_CONTAINER_TIMEOUT_S" in str(error_for(valid_env(RAG_CONTAINER_TIMEOUT_S=bad)))


def test_container_timeout_upper_bound_inclusive():
    assert load_settings(valid_env(RAG_CONTAINER_TIMEOUT_S="600")).RAG_CONTAINER_TIMEOUT_S == 600


@pytest.mark.parametrize(
    "name,value",
    [
        ("RAG_CONTAINER_CPU_MHZ", True),
        ("RAG_CONTAINER_CPU_MHZ", 500.0),
        ("RAG_CONTAINER_MEMORY_MB", True),
        ("RAG_CONTAINER_MEMORY_MB", 128.0),
        ("RAG_CONTAINER_TIMEOUT_S", True),
        ("RAG_CONTAINER_MAX_INPUT_BYTES", True),
        ("RAG_CONTAINER_MAX_OUTPUT_BYTES", 2048.0),
    ],
)
def test_container_direct_construction_rejects_bool_and_float_for_int(name, value):
    with pytest.raises(ValidationError):
        Settings(**DIRECT, **{name: value})


@pytest.mark.parametrize("name", ["RAG_CONTAINER_IMAGE", "NOMAD_NAMESPACE", "NOMAD_ADDR"])
def test_container_direct_construction_validates_text(name):
    with pytest.raises(ValidationError):
        Settings(**DIRECT, **{name: "not valid"})


@pytest.mark.parametrize(
    "name,bad",
    [
        ("NOMAD_ADDR", "ftp://LEAK-host"),
        ("NOMAD_NAMESPACE", "LEAK/ns"),
        ("RAG_CONTAINER_IMAGE", "LEAK:latest"),
        ("RAG_CONTAINER_CPU_MHZ", "LEAK"),
        ("RAG_CONTAINER_MEMORY_MB", "LEAK"),
        ("RAG_CONTAINER_TIMEOUT_S", "LEAK"),
        ("RAG_CONTAINER_MAX_INPUT_BYTES", "LEAK"),
        ("RAG_CONTAINER_MAX_OUTPUT_BYTES", "LEAK"),
        ("NOMAD_CACERT", "LEAK\x00"),
    ],
)
def test_bad_container_values_never_echoed(name, bad):
    err = error_for(valid_env(**{name: bad}))
    assert name in str(err)
    assert "LEAK" not in f"{err} {err!r} {full_trace(err)}"


def test_container_problems_are_reported_together_with_the_api_ones():
    err = error_for({"NOMAD_NAMESPACE": "a b", "RAG_CONTAINER_IMAGE": "x"})
    for name in (*REQUIRED, "NOMAD_NAMESPACE", "RAG_CONTAINER_IMAGE"):
        assert name in str(err)


def test_nomad_token_not_reachable_from_traceback_locals(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "docs")
    monkeypatch.setenv("NOMAD_ADDR", "http://nomad.example.com")
    monkeypatch.setenv("NOMAD_TOKEN", "test-not-a-key-LEAK-nomad")
    with pytest.raises(ConfigError) as excinfo:
        load_settings()
    trace = traceback.TracebackException.from_exception(excinfo.value, capture_locals=True)
    assert "LEAK" not in "".join(trace.format())
