"""Fixture collection for the integration lane.

Every collection this module touches must carry TEST_PREFIX; any other name is refused before the
client is called, so a mis-set QDRANT_COLLECTION can never wipe real data.
"""

import re
import uuid
from collections.abc import Callable, Sequence

from qdrant_client.models import Distance, PointStruct, VectorParams

TEST_PREFIX = "rag_test_"

_TEST_NAME = re.compile(re.escape(TEST_PREFIX) + r"[A-Za-z0-9_-]+")

FIXTURE_DOCS = (
    {"text": "The sky appears blue because air scatters short wavelengths of sunlight.",
     "source_id": "weather", "title": "Why the sky is blue", "uri": "https://example.test/sky"},
    {"text": "Photosynthesis lets green plants turn sunlight, water and carbon dioxide into sugar.",
     "source_id": "plants", "title": "Photosynthesis"},
    {"text": "The Pacific is the deepest and largest ocean on Earth.",
     "source_id": "ocean", "title": "Oceans"},
    {"text": "Python is a programming language known for readable syntax.",
     "source_id": "code"},
)


class RefusedCollectionError(ValueError):
    pass


def unique_test_collection() -> str:
    return f"{TEST_PREFIX}{uuid.uuid4().hex[:12]}"


def require_test_collection(name: str) -> str:
    if not isinstance(name, str) or not _TEST_NAME.fullmatch(name):
        raise RefusedCollectionError(f"refusing to touch a collection without the {TEST_PREFIX!r} prefix")
    return name


def seed(
    client,
    collection: str,
    embed_documents: Callable[[list[str]], Sequence[Sequence[float]]],
    docs: Sequence[dict] = FIXTURE_DOCS,
) -> int:
    """(Re)create `collection` with one unnamed cosine vector and one point per doc."""
    require_test_collection(collection)
    vectors = embed_documents([d["text"] for d in docs])
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        collection,
        points=[
            PointStruct(id=i, vector=list(v), payload=dict(d))
            for i, (d, v) in enumerate(zip(docs, vectors), start=1)
        ],
    )
    return len(docs)


def drop(client, collection: str) -> None:
    require_test_collection(collection)
    if client.collection_exists(collection):
        client.delete_collection(collection)
