import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import retriever as retriever_module
from config import DEFAULT_EMBED_REVISION, load_settings
from normalizer import (
    ContextChunk,
    EmptyRetrievalError,
    ErrorCode,
    InvalidQueryError,
    RAGError,
    RetrievalError,
)
from retriever import FastEmbedder, QdrantRetriever, Retriever

QDRANT_KEY = "test-not-a-key-LEAK-qdrant"
URL = "https://qdrant.example.invalid:6333"
HF_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"


def settings(**overrides):
    env = {
        "ANTHROPIC_API_KEY": "test-not-a-key-LEAK-anthropic",
        "QDRANT_URL": URL,
        "QDRANT_API_KEY": QDRANT_KEY,
        "QDRANT_COLLECTION": "docs",
    }
    env.update(overrides)
    return load_settings(env)


class StubEmbedder:
    """Maps a query to a fixed vector; records what it was asked."""

    def __init__(self, vectors=None, default=(1.0, 0.0, 0.0)):
        self.vectors = vectors or {}
        self.default = default
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.vectors.get(text, self.default)


class Reached(BaseException):
    """Not an Exception: the code under test cannot swallow it, so reaching it fails the test."""


class ExplodingEmbedder:
    def embed_query(self, text):
        raise Reached("the embedder must not be reached")


def memory_client(dim=3, points=(), collection="docs", distance=Distance.COSINE):
    client = QdrantClient(":memory:")
    client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=distance))
    if points:
        client.upsert(collection, points=list(points))
    return client


def point(pid, vector, **payload):
    return PointStruct(id=pid, vector=vector, payload=payload)


SKY = point(1, [1.0, 0.0, 0.0], text="The sky is blue.", source_id="weather")
SEA = point(2, [0.8, 0.6, 0.0], text="The sea is deep.", source_id="ocean")
GRASS = point(3, [0.0, 1.0, 0.0], text="Grass is green.", source_id="plants")


def retriever_over(points=(SKY, SEA, GRASS), embedder=None, cfg=None, **kw):
    return QdrantRetriever(
        cfg or settings(),
        client=memory_client(points=points),
        embedder=embedder or StubEmbedder(),
        **kw,
    )


class FakeClient:
    """Boundary stub for failures a healthy in-memory client cannot produce."""

    def __init__(self, hits=(), vectors=None, collection_error=None, query_error=None):
        self.hits = list(hits)
        self.vectors = vectors if vectors is not None else VectorParams(
            size=3, distance=Distance.COSINE
        )
        self.collection_error = collection_error
        self.query_error = query_error
        self.get_collection_calls = 0
        self.queries = []

    def get_collection(self, collection_name, **kw):
        self.get_collection_calls += 1
        if self.collection_error:
            raise self.collection_error
        params = SimpleNamespace(vectors=self.vectors)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def query_points(self, collection_name, query=None, limit=10, **kw):
        self.queries.append((collection_name, query, limit, kw))
        if self.query_error:
            raise self.query_error
        return SimpleNamespace(points=self.hits)


_UNSET = object()


def hit(pid="p1", score=0.9, payload=_UNSET):
    if payload is _UNSET:
        payload = {"text": "some text", "source_id": "s1"}
    return SimpleNamespace(id=pid, score=score, payload=payload)


def retriever_with(client, embedder=None, cfg=None):
    return QdrantRetriever(cfg or settings(), client=client, embedder=embedder or StubEmbedder())


def failure_of(fn, *args, kind=RetrievalError):
    with pytest.raises(kind) as excinfo:
        fn(*args)
    return excinfo.value


def full_trace(err):
    return "".join(traceback.TracebackException.from_exception(err, capture_locals=True).format())


# --- interface and happy path ----------------------------------------------------


def test_qdrant_retriever_satisfies_the_retriever_protocol():
    r: Retriever = retriever_over()
    assert callable(r.retrieve)


def test_returns_context_chunks_best_first():
    chunks = retriever_over().retrieve("sky?", 3)
    assert all(type(c) is ContextChunk for c in chunks)
    assert [c.id for c in chunks] == ["1", "2", "3"]
    assert [c.source_id for c in chunks] == ["weather", "ocean", "plants"]
    assert chunks[0].text == "The sky is blue."
    assert chunks[0].score == pytest.approx(1.0)
    assert chunks[0].score > chunks[1].score > chunks[2].score


def test_k_limits_the_number_of_chunks():
    assert [c.id for c in retriever_over().retrieve("sky?", 2)] == ["1", "2"]


def test_k_defaults_to_the_configured_top_k():
    r = retriever_over(cfg=settings(RAG_TOP_K="1"))
    assert [c.id for c in r.retrieve("sky?")] == ["1"]


def test_explicit_k_overrides_the_configured_top_k():
    r = retriever_over(cfg=settings(RAG_TOP_K="1"))
    assert len(r.retrieve("sky?", 3)) == 3


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "2"])
def test_invalid_k_is_a_retrieval_error(bad):
    embedder = ExplodingEmbedder()
    err = failure_of(retriever_over(embedder=embedder).retrieve, "sky?", bad)
    assert err.code is ErrorCode.RETRIEVAL_FAILED


def test_query_is_embedded_once_and_the_vector_is_searched():
    embedder = StubEmbedder({"sea?": [0.0, 1.0, 0.0]})
    chunks = retriever_over(embedder=embedder).retrieve("sea?", 1)
    assert embedder.queries == ["sea?"]
    assert chunks[0].id == "3"  # nearest to [0, 1, 0]


def test_integer_and_uuid_point_ids_become_string_chunk_ids():
    uid = "11111111-2222-3333-4444-555555555555"
    uuid_point = point(uid, [1.0, 0.0, 0.0], text="t", source_id="s")
    chunks = retriever_over(points=[uuid_point, SKY]).retrieve("q", 2)
    assert {c.id for c in chunks} == {uid, "1"}


def test_optional_title_and_uri_payload_keys_are_accepted():
    p = point(1, [1.0, 0.0, 0.0], text="t", source_id="s", title="A title", uri="https://x.test/a")
    (chunk,) = retriever_over(points=[p]).retrieve("q", 1)
    assert chunk.source_id == "s"


# --- invalid query: before any network -----------------------------------------


@pytest.mark.parametrize("bad", ["", " ", "\n\t", "​", "ㅤ", None, 5, b"q"])
def test_invalid_query_is_rejected_before_embedding_or_network(bad):
    client = FakeClient()
    r = QdrantRetriever(settings(), client=client, embedder=ExplodingEmbedder())
    err = failure_of(r.retrieve, bad, 3, kind=InvalidQueryError)
    assert err.code is ErrorCode.INVALID_QUERY
    assert client.get_collection_calls == 0 and client.queries == []


def test_invalid_query_wins_over_an_invalid_k():
    failure_of(retriever_over().retrieve, "  ", 0, kind=InvalidQueryError)


def test_invalid_query_never_loads_the_real_embedder_or_builds_a_client(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not construct")

    monkeypatch.setattr("qdrant_client.QdrantClient", boom)
    monkeypatch.setattr(retriever_module.FastEmbedder, "_load", boom)
    failure_of(QdrantRetriever(settings()).retrieve, " ", 1, kind=InvalidQueryError)


# --- empty / failure ----------------------------------------------------------------


def test_zero_hits_is_an_empty_retrieval_error():
    err = failure_of(retriever_over(points=()).retrieve, "q", 3, kind=EmptyRetrievalError)
    assert err.code is ErrorCode.EMPTY_RETRIEVAL


def test_empty_retrieval_is_a_retrieval_distinct_from_failure():
    assert not issubclass(EmptyRetrievalError, RetrievalError)


@pytest.mark.parametrize(
    "boom",
    [
        ConnectionError(f"connect to {URL} api-key {QDRANT_KEY} refused"),
        TimeoutError(QDRANT_KEY),
        RuntimeError(QDRANT_KEY),
        ValueError(QDRANT_KEY),
    ],
)
def test_unreachable_server_is_a_retrieval_error_without_leaks(boom):
    client = FakeClient(collection_error=boom)
    err = failure_of(retriever_with(client).retrieve, "q", 3)
    assert err.code is ErrorCode.RETRIEVAL_FAILED
    text = f"{err} {err!r} {err.args} {full_trace(err)}"
    assert "LEAK" not in text and "example.invalid" not in text
    assert err.__cause__ is None and err.__context__ is None


def test_search_failure_is_a_retrieval_error_without_leaks():
    client = FakeClient(query_error=ConnectionError(f"{URL} {QDRANT_KEY}"))
    err = failure_of(retriever_with(client).retrieve, "q", 3)
    assert "LEAK" not in f"{err} {full_trace(err)}"
    assert err.__cause__ is None and err.__context__ is None


def test_embedder_failure_is_a_retrieval_error_without_leaks():
    class Broken:
        def embed_query(self, text):
            raise RuntimeError(f"model path /secret {QDRANT_KEY}")

    err = failure_of(retriever_with(FakeClient(), Broken()).retrieve, "q", 3)
    assert "LEAK" not in f"{err} {full_trace(err)}" and "/secret" not in str(err)
    assert err.__cause__ is None and err.__context__ is None


@pytest.mark.parametrize(
    "vector",
    [[], [float("nan"), 0.0, 0.0], [float("inf"), 0, 0], "abc", None, [True, 0.0, 0.0], ["a", 0, 0]],
)
def test_unusable_query_vectors_are_retrieval_errors(vector):
    client = FakeClient(hits=[hit()])
    err = failure_of(retriever_with(client, StubEmbedder(default=vector)).retrieve, "q", 3)
    assert err.code is ErrorCode.RETRIEVAL_FAILED
    assert "vector" in err.message and "embedder" in err.message
    assert client.queries == []


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_real_client_unreachable_is_a_retrieval_error():
    # With sockets disabled the real client cannot connect: the error must still be ours.
    r = QdrantRetriever(settings(QDRANT_TIMEOUT_S="1"), embedder=StubEmbedder())
    err = failure_of(r.retrieve, "q", 1)
    assert "LEAK" not in f"{err} {full_trace(err)}" and "example.invalid" not in str(err)
    assert err.__cause__ is None and err.__context__ is None


def test_traceback_locals_hold_no_key_when_retrieval_fails():
    client = FakeClient(collection_error=ConnectionError("down"))
    err = failure_of(retriever_with(client).retrieve, "q", 3)
    assert "LEAK" not in full_trace(err)


# --- payload validation --------------------------------------------------------------


GOOD = {"text": "some text", "source_id": "s1"}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "text",
        {},
        {"text": "t"},
        {"source_id": "s"},
        {"text": 5, "source_id": "s"},
        {"text": "t", "source_id": 5},
        {"text": "", "source_id": "s"},
        {"text": "   ", "source_id": "s"},
        {"text": "t", "source_id": ""},
        {"text": "t", "source_id": " ​"},
        {**GOOD, "title": 5},
        {**GOOD, "uri": ["x"]},
        {**GOOD, "title": {"a": 1}},
    ],
)
def test_bad_payload_is_a_retrieval_error(payload):
    client = FakeClient(hits=[hit(payload=payload)])
    err = failure_of(retriever_with(client).retrieve, "q", 3)
    assert err.code is ErrorCode.RETRIEVAL_FAILED


def test_one_bad_hit_fails_the_whole_retrieval():
    client = FakeClient(hits=[hit("a"), hit("b", payload={"text": "t"})])
    failure_of(retriever_with(client).retrieve, "q", 3)


def test_payload_values_are_not_echoed_in_the_error():
    payload = {"text": 5, "source_id": "LEAK-source", "extra": "LEAK-extra"}
    err = failure_of(retriever_with(FakeClient(hits=[hit(payload=payload)])).retrieve, "q", 3)
    assert "LEAK" not in f"{err} {err!r} {err.args}"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), None, "0.5", True])
def test_non_finite_or_missing_score_is_a_retrieval_error(score):
    client = FakeClient(hits=[hit(score=score)])
    err = failure_of(retriever_with(client).retrieve, "q", 3)
    assert err.code is ErrorCode.RETRIEVAL_FAILED


def test_integer_score_is_accepted_as_a_float():
    (chunk,) = retriever_with(FakeClient(hits=[hit(score=1)])).retrieve("q", 3)
    assert chunk.score == 1.0 and isinstance(chunk.score, float)


@pytest.mark.parametrize("pid", [None, True, "", "  ", 1.5, ("a",)])
def test_unusable_point_id_is_a_retrieval_error(pid):
    err = failure_of(retriever_with(FakeClient(hits=[hit(pid=pid)])).retrieve, "q", 3)
    assert "id" in err.message


@pytest.mark.parametrize("payload", [None, [], "text", 5])
def test_non_object_payload_is_reported_as_such(payload):
    err = failure_of(retriever_with(FakeClient(hits=[hit(payload=payload)])).retrieve, "q", 3)
    assert "no payload" in err.message


def test_a_rag_error_from_the_embedder_passes_through_unchanged():
    class Failing:
        def embed_query(self, text):
            raise original

    original = RetrievalError("the embedding model is unavailable")
    with pytest.raises(RetrievalError) as excinfo:
        retriever_with(FakeClient(), Failing()).retrieve("q", 3)
    assert excinfo.value is original


# --- collection compatibility (verified at first use) -----------------------------


def test_wrong_vector_size_is_a_retrieval_error():
    r = QdrantRetriever(
        settings(), client=memory_client(dim=5, points=[]), embedder=StubEmbedder()
    )
    err = failure_of(r.retrieve, "q", 3)
    assert err.code is ErrorCode.RETRIEVAL_FAILED and "vector" in err.message


@pytest.mark.parametrize("distance", [Distance.EUCLID, Distance.MANHATTAN])
def test_incompatible_distance_is_a_retrieval_error(distance):
    client = FakeClient(hits=[hit()], vectors=VectorParams(size=3, distance=distance))
    failure_of(retriever_with(client).retrieve, "q", 3)


@pytest.mark.parametrize("distance", [Distance.COSINE, Distance.DOT])
def test_cosine_and_dot_are_compatible(distance):
    client = FakeClient(hits=[hit()], vectors=VectorParams(size=3, distance=distance))
    assert len(retriever_with(client).retrieve("q", 3)) == 1


@pytest.mark.parametrize(
    "vectors",
    [
        {"dense": VectorParams(size=3, distance=Distance.COSINE)},
        None,
        {},
    ],
)
def test_named_or_missing_vector_config_is_a_retrieval_error(vectors):
    client = FakeClient(hits=[hit()], vectors=vectors)
    client.vectors = vectors
    err = failure_of(retriever_with(client).retrieve, "q", 3)
    assert "unnamed" in err.message


def test_missing_collection_is_a_retrieval_error():
    client = QdrantClient(":memory:")
    r = QdrantRetriever(settings(), client=client, embedder=StubEmbedder())
    err = failure_of(r.retrieve, "q", 3)
    assert err.code is ErrorCode.RETRIEVAL_FAILED


def test_compatibility_is_checked_once_then_cached():
    client = FakeClient(hits=[hit()])
    r = retriever_with(client)
    r.retrieve("q", 3)
    r.retrieve("q", 3)
    assert client.get_collection_calls == 1


def test_failed_compatibility_check_is_not_cached():
    client = FakeClient(hits=[hit()], collection_error=ConnectionError("down"))
    r = retriever_with(client)
    failure_of(r.retrieve, "q", 3)
    client.collection_error = None
    assert len(r.retrieve("q", 3)) == 1
    assert client.get_collection_calls == 2


def test_incompatible_collection_is_rechecked_not_cached_as_good():
    client = FakeClient(hits=[hit()], vectors=VectorParams(size=9, distance=Distance.COSINE))
    r = retriever_with(client)
    failure_of(r.retrieve, "q", 3)
    failure_of(r.retrieve, "q", 3)
    assert client.get_collection_calls == 2 and client.queries == []


def test_search_uses_the_configured_collection_and_requests_payload():
    client = FakeClient(hits=[hit()])
    retriever_with(client, cfg=settings(QDRANT_COLLECTION="other_docs")).retrieve("q", 4)
    ((collection, vector, limit, kw),) = client.queries
    assert (collection, vector, limit) == ("other_docs", [1.0, 0.0, 0.0], 4)
    assert kw.get("with_payload") is True


# --- the real client is built from Settings only --------------------------------------


def test_client_is_built_explicitly_from_settings(monkeypatch):
    seen = {}

    class Recorder(FakeClient):
        def __init__(self, **kwargs):
            super().__init__(hits=[hit()])
            seen.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", Recorder)
    cfg = settings(QDRANT_TIMEOUT_S="2.2")
    QdrantRetriever(cfg, embedder=StubEmbedder()).retrieve("q", 1)
    assert seen["url"] == URL
    assert seen["api_key"] == QDRANT_KEY
    assert seen["timeout"] == 3  # whole seconds, rounded up: the client takes an int
    assert seen["check_compatibility"] is False
    assert seen.get("prefer_grpc", False) is False


def test_client_without_api_key_passes_none(monkeypatch):
    seen = {}

    class Recorder(FakeClient):
        def __init__(self, **kwargs):
            super().__init__(hits=[hit()])
            seen.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", Recorder)
    cfg = settings(QDRANT_URL="http://localhost:6333", QDRANT_API_KEY="")
    QdrantRetriever(cfg, embedder=StubEmbedder()).retrieve("q", 1)
    assert seen["api_key"] is None and seen["url"] == "http://localhost:6333"


def test_client_is_built_lazily_and_reused(monkeypatch):
    built = []

    class Recorder(FakeClient):
        def __init__(self, **kwargs):
            super().__init__(hits=[hit()])
            built.append(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", Recorder)
    r = QdrantRetriever(settings(), embedder=StubEmbedder())
    assert built == []
    r.retrieve("q", 1)
    r.retrieve("q", 1)
    assert len(built) == 1


def test_retriever_repr_does_not_expose_keys():
    r = QdrantRetriever(settings(), embedder=StubEmbedder())
    assert "LEAK" not in f"{r!r} {r.__dict__!r}" and "LEAK" not in str(vars(r))


# --- FastEmbedder: the pinned revision is requested from the configured endpoint -----


def install_vendor(monkeypatch, tmp_path, *, models=None, revision_dir=None, embed=None):
    """Fake `huggingface_hub` and `fastembed` modules so no download or onnx load happens."""
    calls = SimpleNamespace(snapshot=[], text_embedding=[])
    snapshot = tmp_path / "models--x" / "snapshots" / (revision_dir or DEFAULT_EMBED_REVISION)
    snapshot.mkdir(parents=True)
    vec = [0.1, 0.2, 0.3] if embed is None else embed

    def snapshot_download(repo_id, **kwargs):
        calls.snapshot.append({"repo_id": repo_id, **kwargs})
        return str(snapshot)

    class TextEmbedding:
        def __init__(self, model_name, **kwargs):
            calls.text_embedding.append({"model_name": model_name, **kwargs})

        @staticmethod
        def list_supported_models():
            return models if models is not None else [DEFAULT_MODEL]

        def query_embed(self, text):
            return iter([vec])

        def embed(self, texts):
            return iter([vec for _ in texts])

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    fe = types.ModuleType("fastembed")
    fe.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "fastembed", fe)
    return calls, snapshot


DEFAULT_MODEL = {
    "model": "BAAI/bge-small-en-v1.5",
    "sources": {"hf": HF_REPO, "url": None},
    "model_file": "model_optimized.onnx",
    "additional_files": [],
}


def test_fast_embedder_downloads_the_pinned_revision(monkeypatch, tmp_path):
    calls, snapshot = install_vendor(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    cfg = settings(RAG_EMBED_CACHE_DIR=str(cache))
    assert FastEmbedder(cfg).embed_query("q") == [0.1, 0.2, 0.3]
    (dl,) = calls.snapshot
    assert dl["repo_id"] == HF_REPO
    assert dl["revision"] == DEFAULT_EMBED_REVISION == cfg.RAG_EMBED_REVISION
    assert dl["cache_dir"] == str(cache)
    assert "model_optimized.onnx" in dl["allow_patterns"]
    assert "tokenizer.json" in dl["allow_patterns"]
    (te,) = calls.text_embedding
    assert te["model_name"] == "BAAI/bge-small-en-v1.5"
    assert te["specific_model_path"] == str(snapshot)


def test_fast_embedder_honours_an_overridden_revision(monkeypatch, tmp_path):
    other = "b" * 40
    calls, _ = install_vendor(monkeypatch, tmp_path, revision_dir=other)
    FastEmbedder(settings(RAG_EMBED_REVISION=other)).embed_query("q")
    assert calls.snapshot[0]["revision"] == other


def test_fast_embedder_without_cache_dir_passes_none(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path)
    FastEmbedder(settings()).embed_query("q")
    assert calls.snapshot[0]["cache_dir"] is None


def test_fast_embedder_passes_the_configured_endpoint_and_no_token(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path)
    FastEmbedder(settings(RAG_EMBED_ENDPOINT="https://hub.example.com")).embed_query("q")
    (dl,) = calls.snapshot
    assert dl["endpoint"] == "https://hub.example.com"
    assert dl["token"] is False


def test_fast_embedder_default_endpoint_is_the_public_hub(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path)
    FastEmbedder(settings()).embed_query("q")
    assert calls.snapshot[0]["endpoint"] == "https://huggingface.co"


def test_ambient_hub_environment_does_not_change_the_download_call(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path)
    FastEmbedder(settings()).embed_query("q")
    monkeypatch.setenv("HF_ENDPOINT", "https://evil.invalid")
    monkeypatch.setenv("HF_TOKEN", "LEAK-hf-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "LEAK-hf-token")
    ambient_calls, _ = install_vendor(monkeypatch, tmp_path / "again")
    FastEmbedder(settings()).embed_query("q")
    assert ambient_calls.snapshot == calls.snapshot
    assert "evil.invalid" not in str(ambient_calls.snapshot)
    assert "LEAK" not in str(ambient_calls.snapshot)


def test_fast_embedder_rejects_a_snapshot_dir_that_is_missing(monkeypatch, tmp_path):
    _, snapshot = install_vendor(monkeypatch, tmp_path)
    snapshot.rmdir()
    failure_of(FastEmbedder(settings()).embed_query, "q")


def test_fast_embedder_rejects_a_model_fastembed_does_not_know(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path, models=[])
    err = failure_of(FastEmbedder(settings()).embed_query, "q")
    assert calls.snapshot == [] and "not supported" in err.message


def test_fast_embedder_rejects_a_model_without_a_hub_source(monkeypatch, tmp_path):
    no_hf = {**DEFAULT_MODEL, "sources": {"hf": None, "url": "https://x.invalid/m.tgz"}}
    calls, _ = install_vendor(monkeypatch, tmp_path, models=[no_hf])
    err = failure_of(FastEmbedder(settings()).embed_query, "q")
    assert calls.snapshot == [] and "source" in err.message


def test_fast_embedder_matches_model_names_case_insensitively(monkeypatch, tmp_path):
    lower = {**DEFAULT_MODEL, "model": "baai/bge-small-en-v1.5"}
    install_vendor(monkeypatch, tmp_path, models=[lower])
    assert FastEmbedder(settings()).embed_query("q") == [0.1, 0.2, 0.3]


def test_fast_embedder_download_failure_is_a_retrieval_error_without_leaks(monkeypatch, tmp_path):
    install_vendor(monkeypatch, tmp_path)

    def boom(repo_id, **kwargs):
        raise OSError("HF_TOKEN=LEAK-hf offline")

    sys.modules["huggingface_hub"].snapshot_download = boom
    err = failure_of(FastEmbedder(settings()).embed_query, "q")
    assert "LEAK" not in f"{err} {full_trace(err)}"
    assert err.__cause__ is None and err.__context__ is None


def test_fast_embedder_load_failure_is_a_retrieval_error(monkeypatch, tmp_path):
    install_vendor(monkeypatch, tmp_path)

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("corrupt onnx LEAK")

        list_supported_models = staticmethod(lambda: [DEFAULT_MODEL])

    sys.modules["fastembed"].TextEmbedding = Broken
    err = failure_of(FastEmbedder(settings()).embed_query, "q")
    assert "LEAK" not in f"{err} {full_trace(err)}"
    assert err.__cause__ is None and err.__context__ is None


def test_fast_embedder_loads_once(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path)
    e = FastEmbedder(settings())
    e.embed_query("a")
    e.embed_query("b")
    e.embed_documents(["c"])
    assert len(calls.snapshot) == 1 and len(calls.text_embedding) == 1


def test_fast_embedder_is_lazy(monkeypatch, tmp_path):
    calls, _ = install_vendor(monkeypatch, tmp_path)
    FastEmbedder(settings())
    assert calls.snapshot == [] and calls.text_embedding == []


def test_fast_embedder_returns_plain_float_lists(monkeypatch, tmp_path):
    numpy = pytest.importorskip("numpy")
    install_vendor(monkeypatch, tmp_path, embed=numpy.array([0.5, 0.25], dtype="float32"))
    vec = FastEmbedder(settings()).embed_query("q")
    assert vec == [0.5, 0.25] and all(type(x) is float for x in vec)
    docs = FastEmbedder(settings()).embed_documents(["a", "b"])
    assert docs == [[0.5, 0.25], [0.5, 0.25]]


def test_fast_embedder_embedding_failure_is_a_retrieval_error(monkeypatch, tmp_path):
    install_vendor(monkeypatch, tmp_path)
    e = FastEmbedder(settings())
    e.embed_query("warm")

    def boom(text):
        raise RuntimeError("onnx LEAK")

    e._model.query_embed = boom
    err = failure_of(e.embed_query, "q")
    assert "LEAK" not in f"{err} {full_trace(err)}"


def test_default_retriever_embedder_is_the_pinned_fast_embedder():
    r = QdrantRetriever(settings(), client=FakeClient())
    assert isinstance(r._embedder, FastEmbedder)


def test_vendor_imports_are_lazy():
    # The unit lane must not pay for onnxruntime/fastembed at import time of this module.
    source = Path(retriever_module.__file__).read_text()
    top_level = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("fastembed" in ln or "huggingface_hub" in ln for ln in top_level)


def test_numpy_scalar_vectors_reach_the_client_as_plain_floats():
    numpy = pytest.importorskip("numpy")
    vec = [numpy.float32(0.5), numpy.float32(0.5), numpy.float32(0.0)]
    client = FakeClient(hits=[hit()])
    retriever_with(client, StubEmbedder(default=vec)).retrieve("q", 1)
    sent = client.queries[0][1]
    assert sent == [0.5, 0.5, 0.0] and all(type(x) is float for x in sent)


def test_every_failure_is_a_rag_error():
    for client in (
        FakeClient(collection_error=KeyError("x")),
        FakeClient(query_error=OSError("x")),
        FakeClient(hits=[hit(payload={})]),
    ):
        with pytest.raises(RAGError):
            retriever_with(client).retrieve("q", 1)
