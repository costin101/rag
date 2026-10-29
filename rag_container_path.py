"""The container path: Qdrant retrieval plus a Nomad session container as the generator.

Wiring only, like `ApiPath`, and it returns the same `RAGResponse`. Whether a query takes this
path is routing's decision (#12), not made here. Each query gets its own session, purged when
the answer (or the failure) is done; #9 will serve many queries from one.
"""

from collections.abc import Callable

from config import Settings
from container_generator import NomadGenerator
from container_session import ContainerSession
from normalizer import ConfigError, RAGResponse
from pipeline import answer
from retriever import QdrantRetriever, Retriever

_REQUIRED = ("NOMAD_ADDR", "RAG_CONTAINER_IMAGE")


class ContainerPath:
    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Retriever | None = None,
        session_factory: Callable[[Settings], ContainerSession] = ContainerSession,
    ):
        missing = [name for name in _REQUIRED if getattr(settings, name) is None]
        if missing:
            problems = "; ".join(f"{name}: required but not set" for name in missing)
            raise ConfigError(f"invalid configuration for the container path: {problems}")
        self._settings = settings
        self._top_k = settings.RAG_TOP_K
        self._retriever = retriever if retriever is not None else QdrantRetriever(settings)
        self._session_factory = session_factory

    def answer(self, query: str) -> RAGResponse:
        with self._session_factory(self._settings) as session:
            return answer(query, self._retriever, NomadGenerator(session), k=self._top_k)
