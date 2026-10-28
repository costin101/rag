import httpx2
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import rag_api_fallback
from config import load_settings
from normalizer import ConfigError, InvalidQueryError, RAGResponse, RetrievalError
from provider import AnthropicProvider
from rag_api_fallback import ApiPath, answer_via_api
from retriever import QdrantRetriever
from tests.contract import assert_valid_rag_response

ENV = {
    "ANTHROPIC_API_KEY": "test-not-a-key-LEAK-anthropic",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION": "docs",
    "RAG_TOP_K": "2",
}


def settings(**kw):
    return load_settings({**ENV, **kw})


class Embedder:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def memory_client():
    client = QdrantClient(":memory:")
    client.create_collection("docs", vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    client.upsert(
        "docs",
        points=[
            PointStruct(id=i, vector=v, payload={"text": t, "source_id": s})
            for i, (v, t, s) in enumerate(
                [
                    ([1.0, 0.0, 0.0], "The sky is blue.", "weather"),
                    ([0.9, 0.1, 0.0], "Air scatters light.", "physics"),
                    ([0.0, 1.0, 0.0], "Grass is green.", "plants"),
                ],
                start=1,
            )
        ],
    )
    return client


def anthropic_reply(handler_log):
    def handler(request):
        import json

        handler_log.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": "Because air scatters light."}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 6},
            },
        )

    return handler


def api_path(log, **kw):
    cfg = settings(**kw)
    http = httpx2.Client(transport=httpx2.MockTransport(anthropic_reply(log)))
    return ApiPath(
        cfg,
        retriever=QdrantRetriever(cfg, client=memory_client(), embedder=Embedder()),
        provider=AnthropicProvider(cfg, http_client=http, max_retries=0),
    )


def test_answers_through_retrieval_and_the_anthropic_provider():
    log = []
    r = api_path(log).answer("why is the sky blue?")
    assert isinstance(r, RAGResponse)
    assert_valid_rag_response(r)
    assert r.answer == "Because air scatters light."
    assert (r.usage.input_tokens, r.usage.output_tokens) == (20, 6)
    assert [s.id for s in r.sources] == ["weather", "physics"]


def test_retrieved_chunks_reach_the_provider_request():
    log = []
    api_path(log).answer("why is the sky blue?")
    (sent,) = log
    turn = sent["messages"][0]["content"]
    assert "The sky is blue." in turn and "Air scatters light." in turn
    assert "Grass is green." not in turn  # RAG_TOP_K=2


def test_top_k_comes_from_settings():
    assert len(api_path([], RAG_TOP_K="1").answer("q").chunks) == 1
    assert len(api_path([], RAG_TOP_K="3").answer("q").chunks) == 3


def test_blank_query_is_invalid_and_calls_nothing():
    log = []
    with pytest.raises(InvalidQueryError):
        api_path(log).answer("  ")
    assert log == []


def test_default_wiring_builds_the_real_retriever_and_provider_from_settings():
    path = ApiPath(settings())
    assert isinstance(path._retriever, QdrantRetriever)
    assert isinstance(path._provider, AnthropicProvider)


def test_retrieval_failure_surfaces_as_a_rag_error_and_the_provider_is_not_called():
    log = []
    cfg = settings(QDRANT_COLLECTION="missing")
    http = httpx2.Client(transport=httpx2.MockTransport(anthropic_reply(log)))
    path = ApiPath(
        cfg,
        retriever=QdrantRetriever(cfg, client=memory_client(), embedder=Embedder()),
        provider=AnthropicProvider(cfg, http_client=http, max_retries=0),
    )
    with pytest.raises(RetrievalError):
        path.answer("q")
    assert log == []


def test_answer_via_api_loads_settings_from_the_environment(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    seen = {}

    def fake_answer(query, retriever, provider, *, k):
        seen.update(query=query, k=k, retriever=retriever, provider=provider)
        return "sentinel"

    monkeypatch.setattr(rag_api_fallback, "answer", fake_answer)
    assert answer_via_api("hello") == "sentinel"
    assert seen["k"] == 2 and seen["query"] == "hello"
    assert isinstance(seen["retriever"], QdrantRetriever)


def test_answer_via_api_with_bad_environment_is_a_config_error(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError):
        answer_via_api("hello")


def test_answer_via_api_uses_given_settings(monkeypatch):
    monkeypatch.setattr(rag_api_fallback, "answer", lambda q, r, p, *, k: k)
    assert answer_via_api("q", settings(RAG_TOP_K="4")) == 4
