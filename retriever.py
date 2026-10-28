"""Retrievers (rules 5 and 6): the only module that knows the vector store or the embedder.

A retriever turns a query into `ContextChunk`s from the shared, networked vector store, so every
path sees the same context for the same query. Failures surface as `RetrievalError` (the store
or embedder failed, or returned something unusable) or `EmptyRetrievalError` (nothing matched);
messages quote nothing from the store, the SDK or the settings, and are raised outside the
`except` block that caught the vendor's error, so no cause, context or key is attached.
Heavy vendor packages are imported on first use to keep imports and the unit lane fast.
"""

import math
from collections.abc import Iterable, Sequence
from numbers import Real
from pathlib import Path
from typing import Protocol

from config import Settings
from normalizer import (
    ContextChunk,
    EmptyRetrievalError,
    InvalidQueryError,
    RAGError,
    RetrievalError,
    is_blank,
)

# Vectors are normalized by the embedder, so cosine and dot product rank identically.
_COMPATIBLE_DISTANCES = frozenset({"cosine", "dot"})

# Files the embedder needs besides the model weights (the same set fastembed downloads).
_EMBEDDER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)


class Retriever(Protocol):
    """`k=None` means the retriever's configured top-k."""

    def retrieve(self, query: str, k: int | None = None) -> list[ContextChunk]: ...


class _Unusable(Exception):
    """Internal: carries a fixed, value-free reason; never escapes this module."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _plain_floats(raw) -> list[float]:
    """A finite, non-empty vector of real numbers as plain floats, or _Unusable."""
    try:
        values = list(raw)
    except TypeError:
        raise _Unusable("the embedder returned an unusable vector") from None
    if not values or not all(isinstance(x, Real) and not isinstance(x, bool) for x in values):
        raise _Unusable("the embedder returned an unusable vector")
    floats = [float(x) for x in values]
    if not all(math.isfinite(x) for x in floats):
        raise _Unusable("the embedder returned a non-finite vector")
    return floats


class FastEmbedder:
    """Host-side fastembed model, loaded at first use at the pinned revision.

    `TextEmbedding` takes no revision, so the pinned revision is requested with
    `snapshot_download(revision=...)` from the configured endpoint (`RAG_EMBED_ENDPOINT`, passed
    explicitly, with `token=False`, so the ambient HF_ENDPOINT and HF_TOKEN are never used) and
    handed over as `specific_model_path`. Known limit: the content hashes of the downloaded files
    are not verified against anything beyond what the hub client does itself.
    """

    def __init__(self, settings: Settings):
        self._name = settings.RAG_EMBED_MODEL
        self._revision = settings.RAG_EMBED_REVISION
        self._cache_dir = settings.RAG_EMBED_CACHE_DIR
        self._endpoint = settings.RAG_EMBED_ENDPOINT
        self._model = None

    def embed_query(self, text: str) -> list[float]:
        model = self._loaded()
        problem = None
        vector = None
        try:
            vector = _plain_floats(next(iter(model.query_embed(text))))
        except _Unusable as exc:
            problem = exc.reason
        except Exception:
            problem = "could not embed the query"
        if problem is not None:
            raise RetrievalError(problem)
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._loaded()
        problem = None
        vectors = None
        try:
            vectors = [_plain_floats(v) for v in model.embed(list(texts))]
        except _Unusable as exc:
            problem = exc.reason
        except Exception:
            problem = "could not embed the documents"
        if problem is not None:
            raise RetrievalError(problem)
        return vectors

    def _loaded(self):
        if self._model is None:
            problem = None
            model = None
            try:
                model = self._load()
            except _Unusable as exc:
                problem = exc.reason
            except Exception:
                problem = "could not load the embedding model"
            if problem is not None:
                raise RetrievalError(problem)
            self._model = model
        return self._model

    def _load(self):
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download

        wanted = self._name.lower()
        description = next(
            (m for m in TextEmbedding.list_supported_models() if m["model"].lower() == wanted),
            None,
        )
        if description is None:
            raise _Unusable("the embedding model is not supported by the embedder")
        repo = (description.get("sources") or {}).get("hf")
        if not repo:
            raise _Unusable("the embedding model has no pinned hub source")
        patterns = [*_EMBEDDER_FILES, description["model_file"], *description["additional_files"]]
        snapshot = Path(
            snapshot_download(
                repo,
                revision=self._revision,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
                allow_patterns=patterns,
                endpoint=self._endpoint,
                token=False,  # never send an ambient HF_TOKEN
            )
        )
        if not snapshot.is_dir():
            raise _Unusable("the embedding model snapshot is missing")
        extra = {"cache_dir": str(self._cache_dir)} if self._cache_dir else {}
        return TextEmbedding(model_name=self._name, specific_model_path=str(snapshot), **extra)


class QdrantRetriever:
    """Qdrant collection searched with the host-side embedding of the query."""

    def __init__(self, settings: Settings, *, client=None, embedder=None):
        self._url = settings.QDRANT_URL
        self._api_key = settings.QDRANT_API_KEY
        self._timeout = settings.QDRANT_TIMEOUT_S
        self._collection = settings.QDRANT_COLLECTION
        self._top_k = settings.RAG_TOP_K
        self._client = client
        self._embedder = embedder if embedder is not None else FastEmbedder(settings)
        self._verified = False

    def retrieve(self, query: str, k: int | None = None) -> list[ContextChunk]:
        # Cheap local checks first: a bad request never costs a model load or a round trip.
        if not isinstance(query, str) or is_blank(query):
            raise InvalidQueryError("the query must not be blank")
        k = self._top_k if k is None else k
        if type(k) is not int or k < 1:
            raise RetrievalError("k must be a positive whole number")
        vector = self._embed(query)
        self._verify_collection(len(vector))
        points = self._search(vector, k)
        if not points:
            raise EmptyRetrievalError("the vector store returned no context for the query")
        return [self._chunk(p) for p in points]

    def _embed(self, query: str) -> list[float]:
        problem = None
        vector = None
        try:
            vector = _plain_floats(self._embedder.embed_query(query))
        except RAGError:
            raise
        except _Unusable as exc:
            problem = exc.reason
        except Exception:
            problem = "could not embed the query"
        if problem is not None:
            raise RetrievalError(problem)
        return vector

    def _store(self):
        if self._client is None:
            problem = None
            client = None
            try:
                from qdrant_client import QdrantClient

                client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key.get_secret_value() if self._api_key else None,
                    timeout=math.ceil(self._timeout),  # the client takes whole seconds
                    prefer_grpc=False,
                    check_compatibility=False,  # no extra round trip; the collection check is ours
                )
            except Exception:
                problem = "could not create the vector store client"
            if problem is not None:
                raise RetrievalError(problem)
            self._client = client
        return self._client

    def _verify_collection(self, dim: int) -> None:
        """First use only: the collection must hold one unnamed vector the embedder can search."""
        if self._verified:
            return
        problem = None
        vectors = None
        try:
            info = self._store().get_collection(self._collection)
            vectors = info.config.params.vectors
        except RAGError:
            raise
        except Exception:
            problem = "could not read the vector store collection"
        if problem is None:
            problem = _vector_config_problem(vectors, dim)
        if problem is not None:
            raise RetrievalError(problem)
        self._verified = True

    def _search(self, vector: list[float], k: int) -> Sequence:
        problem = None
        points = ()
        try:
            response = self._store().query_points(
                collection_name=self._collection, query=vector, limit=k, with_payload=True
            )
            points = list(response.points)
        except RAGError:
            raise
        except Exception:
            problem = "the vector store search failed"
        if problem is not None:
            raise RetrievalError(problem)
        return points

    @staticmethod
    def _chunk(point) -> ContextChunk:
        problem = None
        chunk = None
        try:
            chunk = _parse_point(point)
        except _Unusable as exc:
            problem = exc.reason
        except Exception:
            problem = "the vector store returned an unusable hit"
        if problem is not None:
            raise RetrievalError(problem)
        return chunk


def _vector_config_problem(vectors, dim: int) -> str | None:
    size = getattr(vectors, "size", None)
    if size is None:  # missing config or a mapping of named vectors
        return "the collection must hold exactly one unnamed vector"
    if size != dim:
        return "the collection vector size does not match the embedding model"
    distance = getattr(vectors, "distance", None)
    if str(getattr(distance, "value", distance)).lower() not in _COMPATIBLE_DISTANCES:
        return "the collection vector distance is not cosine or dot"
    return None


def _text_field(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or is_blank(value):
        raise _Unusable(f"a hit has no usable '{key}' payload")
    return value


def _parse_point(point) -> ContextChunk:
    """A scored Qdrant point -> ContextChunk, or _Unusable with a reason quoting nothing."""
    payload = getattr(point, "payload", None)
    if not isinstance(payload, dict):
        raise _Unusable("a hit has no payload")
    text = _text_field(payload, "text")
    source_id = _text_field(payload, "source_id")
    point_id = getattr(point, "id", None)
    if isinstance(point_id, bool) or not isinstance(point_id, (int, str)) or is_blank(str(point_id)):
        raise _Unusable("a hit has no usable id")
    score = getattr(point, "score", None)
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
        raise _Unusable("a hit has no finite score")
    return ContextChunk(id=str(point_id), text=text, score=float(score), source_id=source_id)
