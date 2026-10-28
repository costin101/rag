import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import load_settings
from retriever import QdrantRetriever
from tests.seed_qdrant import (
    FIXTURE_DOCS,
    TEST_PREFIX,
    RefusedCollectionError,
    drop,
    require_test_collection,
    seed,
    unique_test_collection,
)


def embed(texts):
    # Distinct deterministic 3-d vectors keyed on the document text.
    return [[1.0 + len(t) % 7, 1.0 + len(t) % 5, 1.0 + i] for i, t in enumerate(texts)]


class RecordingClient:
    def __getattr__(self, name):
        raise AssertionError(f"client.{name} called for a refused collection")


@pytest.mark.parametrize(
    "name", ["docs", "", "rag_test_", "rag_test", "RAG_TEST_x", "xrag_test_a", "rag_test_a/b",
             "rag_test_a b", "rag_test_..", "prod_rag_test_x", None, 5]
)
def test_non_test_collections_are_refused_before_any_client_call(name):
    client = RecordingClient()
    with pytest.raises(RefusedCollectionError):
        seed(client, name, embed)
    with pytest.raises(RefusedCollectionError):
        drop(client, name)
    with pytest.raises(RefusedCollectionError):
        require_test_collection(name)


def test_unique_names_carry_the_prefix_and_differ():
    a, b = unique_test_collection(), unique_test_collection()
    assert a.startswith(TEST_PREFIX) and a != b
    assert require_test_collection(a) == a


def test_seed_builds_a_searchable_collection():
    client = QdrantClient(":memory:")
    name = unique_test_collection()
    assert seed(client, name, embed) == len(FIXTURE_DOCS)
    info = client.get_collection(name)
    assert info.config.params.vectors.size == 3
    assert client.count(name).count == len(FIXTURE_DOCS)
    cfg = load_settings(
        {"ANTHROPIC_API_KEY": "test-not-a-key", "QDRANT_URL": "http://localhost:6333",
         "QDRANT_COLLECTION": name}
    )

    class Q:
        def embed_query(self, text):
            return embed([FIXTURE_DOCS[0]["text"]])[0]

    chunks = QdrantRetriever(cfg, client=client, embedder=Q()).retrieve("sky", 1)
    assert chunks[0].source_id == "weather"


def test_seed_replaces_an_existing_test_collection_and_other_collections_survive():
    client = QdrantClient(":memory:")
    client.create_collection("docs", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    name = unique_test_collection()
    seed(client, name, embed)
    seed(client, name, embed, docs=FIXTURE_DOCS[:2])
    assert client.count(name).count == 2
    drop(client, name)
    assert not client.collection_exists(name)
    assert client.collection_exists("docs")
