from dataclasses import dataclass

from backend.embeddings.dense_embedder import (
    DenseEmbedder,
    get_dense_embedder,
)
from backend.embeddings.sparse_embedder import (
    SparseEmbedder,
    SparseVector,
    get_sparse_embedder,
)
from backend.ingestion.metadata import ChunkData

__all__ = [
    "HybridEmbeddingResult",
    "HybridQueryEmbedding",
    "HybridEmbedder",
]

@dataclass(frozen=True)
class HybridEmbeddingResult:
    dense_vectors: list[list[float]]
    sparse_vectors: list[SparseVector]


@dataclass(frozen=True)
class HybridQueryEmbedding:
    dense_vector: list[float]
    sparse_vector: SparseVector


class HybridEmbedder:
    def __init__(
        self,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self._dense = dense_embedder or get_dense_embedder()
        self._sparse = sparse_embedder or get_sparse_embedder()

    def embed_chunks_hybrid(
        self,
        chunks: list[ChunkData],
    ) -> HybridEmbeddingResult:
        if not chunks:
            raise ValueError("chunks list cannot be empty.")

        dense_vectors = self._dense.embed_chunks(chunks)
        sparse_vectors = self._sparse.embed_chunks(chunks)

        if len(dense_vectors) != len(sparse_vectors):
            raise RuntimeError(
                "Dense/sparse embedding count mismatch."
            )

        return HybridEmbeddingResult(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )

    def embed_query_hybrid(
        self,
        query: str,
    ) -> HybridQueryEmbedding:
        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty or whitespace-only."
            )

        dense_vector = self._dense.embed_query(query)
        sparse_vector = self._sparse.embed_query(query)

        return HybridQueryEmbedding(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )

    def is_loaded(self) -> bool:
        return (
            self._dense.is_loaded()
            and self._sparse.is_loaded()
        )