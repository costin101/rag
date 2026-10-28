"""Integration lane: real Qdrant (QDRANT_URL) and the real pinned embedder (needs the Hub)."""

import os

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import load_settings
from normalizer import ContextChunk, EmptyRetrievalError, RetrievalError
from retriever import FastEmbedder, QdrantRetriever
from tests.seed_qdrant import drop, seed, unique_test_collection

pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


def settings_for(collection):
    env = {
        "ANTHROPIC_API_KEY": "test-not-a-key",
        "QDRANT_URL": os.environ["QDRANT_URL"],
        "QDRANT_COLLECTION": collection,
    }
    if os.environ.get("QDRANT_API_KEY"):
        env["QDRANT_API_KEY"] = os.environ["QDRANT_API_KEY"]
    return load_settings(env)


@pytest.fixture(scope="module")
def embedder():
    return FastEmbedder(settings_for("unused"))


@pytest.fixture(scope="module")
def client():
    cfg = settings_for("unused")
    key = cfg.QDRANT_API_KEY.get_secret_value() if cfg.QDRANT_API_KEY else None
    return QdrantClient(url=cfg.QDRANT_URL, api_key=key, check_compatibility=False)


@pytest.fixture(scope="module")
def collection(client, embedder):
    name = unique_test_collection()
    seed(client, name, embedder.embed_documents)
    yield name
    drop(client, name)


def test_real_retrieval_ranks_the_relevant_document_first(collection, embedder):
    r = QdrantRetriever(settings_for(collection), embedder=embedder)
    chunks = r.retrieve("Why is the sky blue?", 3)
    assert all(isinstance(c, ContextChunk) for c in chunks)
    assert chunks[0].source_id == "weather"
    assert len(chunks) == 3


def test_identical_queries_see_identical_context(collection, embedder):
    a = QdrantRetriever(settings_for(collection), embedder=embedder)
    b = QdrantRetriever(settings_for(collection), embedder=embedder)
    assert a.retrieve("deepest ocean", 2) == b.retrieve("deepest ocean", 2)


def test_missing_collection_is_a_retrieval_error(embedder):
    r = QdrantRetriever(settings_for(unique_test_collection()), embedder=embedder)
    with pytest.raises(RetrievalError):
        r.retrieve("anything", 1)


def test_incompatible_vector_size_is_a_retrieval_error(client, embedder):
    name = unique_test_collection()
    client.create_collection(name, vectors_config=VectorParams(size=5, distance=Distance.COSINE))
    try:
        with pytest.raises(RetrievalError):
            QdrantRetriever(settings_for(name), embedder=embedder).retrieve("anything", 1)
    finally:
        drop(client, name)


def test_empty_collection_is_an_empty_retrieval_error(client, embedder):
    name = unique_test_collection()
    client.create_collection(name, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    try:
        with pytest.raises(EmptyRetrievalError):
            QdrantRetriever(settings_for(name), embedder=embedder).retrieve("anything", 1)
    finally:
        drop(client, name)
