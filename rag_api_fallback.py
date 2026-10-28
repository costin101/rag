"""The API path (rule 3's fallback): Qdrant retrieval plus the configured provider, from Settings.

Wiring only. Whether a query takes this path is routing's decision (#8), not made here.
"""

from config import Settings, load_settings
from normalizer import RAGResponse
from pipeline import answer
from provider import Provider, build_provider
from retriever import QdrantRetriever, Retriever


class ApiPath:
    """Built once and reused, so the embedding model and clients load once per process."""

    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Retriever | None = None,
        provider: Provider | None = None,
    ):
        self._top_k = settings.RAG_TOP_K
        self._retriever = retriever if retriever is not None else QdrantRetriever(settings)
        self._provider = provider if provider is not None else build_provider(settings)

    def answer(self, query: str) -> RAGResponse:
        return answer(query, self._retriever, self._provider, k=self._top_k)


def answer_via_api(query: str, settings: Settings | None = None) -> RAGResponse:
    """One-shot: settings from the environment unless given."""
    return ApiPath(settings if settings is not None else load_settings()).answer(query)
