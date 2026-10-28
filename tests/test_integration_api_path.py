"""Integration lane: the whole API path on a real Qdrant, with the provider stubbed.

Proves #7's first acceptance criterion in CI on every PR: real retrieval, the real pipeline and
CLI, a contract-valid `RAGResponse`. Only the Anthropic call is replaced (no key needed).
"""

import io
import json
import os

import pytest
from qdrant_client import QdrantClient

import main
from normalizer import Generation, RAGResponse
from rag_api_fallback import ApiPath
from retriever import FastEmbedder, QdrantRetriever
from tests.contract import assert_valid_rag_response
from tests.seed_qdrant import drop, seed, unique_test_collection
from tests.test_integration_retrieval import settings_for

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class StubProvider:
    def __init__(self):
        self.calls = []

    def generate(self, query, chunks):
        self.calls.append((query, list(chunks)))
        return Generation(text="The sky looks blue.", input_tokens=12, output_tokens=5)


@pytest.fixture(scope="module")
def embedder():
    return FastEmbedder(settings_for("unused"))


@pytest.fixture(scope="module")
def collection(embedder):
    cfg = settings_for("unused")
    key = cfg.QDRANT_API_KEY.get_secret_value() if cfg.QDRANT_API_KEY else None
    client = QdrantClient(url=cfg.QDRANT_URL, api_key=key, check_compatibility=False)
    name = unique_test_collection()
    try:
        seed(client, name, embedder.embed_documents)
        yield name
    finally:
        drop(client, name)


def test_a_query_returns_a_contract_valid_response_from_real_retrieval(collection, embedder):
    settings = settings_for(collection)
    provider = StubProvider()
    path = ApiPath(settings, retriever=QdrantRetriever(settings, embedder=embedder), provider=provider)
    response = path.answer("Why is the sky blue?")
    assert isinstance(response, RAGResponse)
    assert_valid_rag_response(response)
    assert response.chunks and response.chunks[0].source_id == "weather"
    assert {s.id for s in response.sources} == {c.source_id for c in response.chunks}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 5)
    assert provider.calls[0][1] == response.chunks  # the provider saw the retrieved context


def test_the_cli_prints_the_same_contract_over_real_retrieval(collection, embedder):
    settings = settings_for(collection)
    out, err = io.StringIO(), io.StringIO()

    def factory(s):
        return ApiPath(s, retriever=QdrantRetriever(s, embedder=embedder), provider=StubProvider())

    env = {"ANTHROPIC_API_KEY": "test-not-a-key", "QDRANT_URL": settings.QDRANT_URL, "QDRANT_COLLECTION": collection}
    if os.environ.get("QDRANT_API_KEY"):
        env["QDRANT_API_KEY"] = os.environ["QDRANT_API_KEY"]
    assert main.run(["Why is the sky blue?"], env=env, out=out, err=err, api_path=factory) == 0
    assert err.getvalue() == ""
    assert_valid_rag_response(json.loads(out.getvalue()))
