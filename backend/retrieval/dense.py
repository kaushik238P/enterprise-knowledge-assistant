from dataclasses import dataclass
import logging
from typing import Any
from uuid import UUID

from backend.config.settings import settings
from backend.embeddings.dense_embedder import DenseEmbedder, get_dense_embedder
from backend.vectorstore.qdrant import QdrantVectorStore, get_vector_store


logger = logging.getLogger(__name__)

__all__ = [
    "DenseSearchResult",
    "DenseRetriever",
    "get_dense_retriever",
]


@dataclass(slots=True, frozen=True)
class DenseSearchResult:
    chunk_id: UUID
    score: float
    content: str
    metadata: dict[str, Any]


class DenseRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        dense_embedder: DenseEmbedder | None = None,
    ) -> None:
        if self._vector_store is None:
            raise RuntimeError("Vector store initialization failed.")

        if self._dense_embedder is None:
            raise RuntimeError("Dense embedder initialization failed.")
        
        self._vector_store = vector_store or get_vector_store()
        self._dense_embedder = dense_embedder or get_dense_embedder()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[DenseSearchResult]:
        if query is None or not query.strip():
            raise ValueError("query cannot be empty or whitespace-only.")

        resolved_top_k = top_k if top_k is not None else settings.dense_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        logger.info("Dense retrieval started | top_k=%d", resolved_top_k)

        query_embedding = self._dense_embedder.embed_query(query)
        logger.info("Dense query embedding generated | dimension=%d", len(query_embedding))

        raw_results = self._vector_store.dense_search(
            query_vector=query_embedding,
            limit=resolved_top_k,
        )

        results = [
            DenseSearchResult(
                chunk_id=result.chunk_id,
                score=result.score,
                content=result.content,
                metadata=result.metadata,
            )
            for result in raw_results
        ]

       

        logger.info(
            "Dense retrieval complete | requested_top_k=%d | results=%d",
            resolved_top_k,
            len(results),
        )

        return results


_DEFAULT_DENSE_RETRIEVER: DenseRetriever | None = None


def get_dense_retriever() -> DenseRetriever:
    global _DEFAULT_DENSE_RETRIEVER
    if _DEFAULT_DENSE_RETRIEVER is None:
        _DEFAULT_DENSE_RETRIEVER = DenseRetriever()
    return _DEFAULT_DENSE_RETRIEVER