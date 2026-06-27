from dataclasses import dataclass
from uuid import UUID
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from backend.config.settings import settings
from backend.embeddings.hybride_embedder import HybridEmbeddingResult, HybridQueryEmbedding
from backend.ingestion.metadata import ChunkData

__all__ = [
    "SearchResult",
    "CollectionStats",
    "QdrantVectorStore",
    "get_vector_store",
    "list_documents",
]

@dataclass(slots=True, frozen=True)
class SearchResult:
    chunk_id: UUID
    score: float
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class CollectionStats:
    points_count: int
    indexed_vectors_count: int


class QdrantVectorStore:
    def __init__(self, collection_name: str = settings.qdrant_collection_name) -> None:
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name cannot be empty.")
        if settings.embedding_dimension <= 0:
            raise ValueError("settings.embedding_dimension must be a positive integer.")
        self._collection_name = collection_name
        self._client = QdrantClient(path=settings.qdrant_path)

    def collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
            return True
        except Exception:
            return False

    def create_collection(self) -> None:
        if self.collection_exists():
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": rest.VectorParams(
                    size=settings.embedding_dimension,
                    distance=rest.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": rest.SparseVectorParams()
            },
        )

    def delete_collection(self) -> None:
        if self.collection_exists():
            self._client.delete_collection(self._collection_name)

    def recreate_collection(self) -> None:
        self.delete_collection()
        self.create_collection()

    def upsert_chunks(
        self,
        chunks: list[ChunkData],
        embeddings: HybridEmbeddingResult,
    ) -> None:
        if not chunks:
            raise ValueError("chunks list cannot be empty.")
        if len(chunks) != len(embeddings.dense_vectors):
            raise RuntimeError("Chunk/dense embedding count mismatch.")
        if len(chunks) != len(embeddings.sparse_vectors):
            raise RuntimeError("Chunk/sparse embedding count mismatch.")

        points: list[rest.PointStruct] = []

        for idx, chunk in enumerate(chunks):
            dense_vector = embeddings.dense_vectors[idx]
            sparse_vector = embeddings.sparse_vectors[idx]

            if chunk.chunk_id != chunk.metadata.chunk_id:
                raise ValueError("ChunkData chunk_id must match ChunkMetadata chunk_id.")
            if len(dense_vector) != settings.embedding_dimension:
                raise RuntimeError(
                    f"Dense vector dimension mismatch for chunk {chunk.chunk_id}."
                )
            if len(sparse_vector.indices) != len(sparse_vector.values):
                raise RuntimeError(
                    f"Sparse vector index/value mismatch for chunk {chunk.chunk_id}."
                )

            payload = {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.metadata.document_id),
                "document_name": chunk.metadata.document_name,
                "page_number": chunk.metadata.page_number,
                "section_title": chunk.metadata.section_title,
                "section_path": chunk.metadata.section_path,
                "content_type": chunk.metadata.content_type,
                "source_hash": chunk.metadata.source_hash,
                "document_version": chunk.metadata.document_version,
                "content_length": chunk.metadata.content_length,
                "ingestion_timestamp": chunk.metadata.ingestion_timestamp.isoformat() if chunk.metadata.ingestion_timestamp else None,
                "content": chunk.content,
            }

            points.append(
                rest.PointStruct(
                    id=str(chunk.chunk_id),
                    vector={
                        "dense": dense_vector,
                        "sparse": rest.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                    },
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    def dense_search(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        if len(query_vector) != settings.embedding_dimension:
            raise ValueError(
                f"Expected dense vector dimension "
                f"{settings.embedding_dimension}, "
                f"got {len(query_vector)}."
            )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )

        results: list[SearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                SearchResult(
                    chunk_id=UUID(str(point.id)),
                    score=float(point.score),
                    content=str(payload.get("content", "")),
                    metadata=dict(payload),
                )
            )

        return results

    def hybrid_search(
        self,
        query_embedding: HybridQueryEmbedding,
        limit: int,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        if len(query_embedding.dense_vector) != settings.embedding_dimension:
            raise ValueError(
                f"Expected dense vector dimension "
                f"{settings.embedding_dimension}, "
                f"got {len(query_embedding.dense_vector)}."
            )

        if (
            len(query_embedding.sparse_vector.indices)
            != len(query_embedding.sparse_vector.values)
        ):
            raise ValueError("Sparse query vector indices/values length mismatch.")

        dense_response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding.dense_vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )

        dense_results = dense_response.points
        
        print("\n" + "=" * 100)
        print("DENSE RESULTS")
        print("=" * 100)

        for i, point in enumerate(dense_results, 1):
            payload = point.payload or {}
            print(
                f"{i:2d}. "
                f"Page={payload.get('page_number')} "
                f"Score={point.score:.4f}"
            )

        sparse_response = self._client.query_points(
            collection_name=self._collection_name,
            query=rest.SparseVector(
                indices=query_embedding.sparse_vector.indices,
                values=query_embedding.sparse_vector.values,
            ),
            using="sparse",
            limit=limit,
            with_payload=True,
        )

        sparse_results = sparse_response.points
        
        print("\n" + "=" * 100)
        print("SPARSE RESULTS")
        print("=" * 100)

        for i, point in enumerate(sparse_results, 1):
            payload = point.payload or {}
            print(
                f"{i:2d}. "
                f"Page={payload.get('page_number')} "
                f"Score={point.score:.4f}"
            )

        fused: dict[str, SearchResult] = {}

        def add_results(results, source_weight: float = 1.0) -> None:
            for rank, point in enumerate(results, start=1):
                point_id = str(point.id)
                payload = point.payload or {}

                existing = fused.get(point_id)
                score = source_weight / (60 + rank)

                if existing is None:
                    fused[point_id] = SearchResult(
                        chunk_id=UUID(point_id),
                        score=score,
                        content=str(payload.get("content", "")),
                        metadata=dict(payload),
                    )
                else:
                    fused[point_id] = SearchResult(
                        chunk_id=existing.chunk_id,
                        score=existing.score + score,
                        content=existing.content,
                        metadata=existing.metadata,
                    )

        add_results(dense_results, source_weight=1.0)
        add_results(sparse_results, source_weight=1.0)

        return sorted(fused.values(), key=lambda r: r.score, reverse=True)[:limit]

    def get_collection_stats(self) -> CollectionStats:
        info = self._client.get_collection(self._collection_name)
        points_count = getattr(info, "points_count", 0) or 0
        indexed_vectors_count = getattr(info, "indexed_vectors_count", 0) or 0
        return CollectionStats(
            points_count=points_count,
            indexed_vectors_count=indexed_vectors_count,
        )
        
    def list_documents(self) -> list[str]:
        """
        Returns the unique document names stored in the collection.
        """
        if not self.collection_exists():
            return []

        document_names: set[str] = set()

        offset = None

        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=["document_name"],
                with_vectors=False,
                limit=100,
                offset=offset,
            )       

            for point in points:
                payload = point.payload or {}

                name = payload.get("document_name")

                if isinstance(name, str) and name.strip():
                    document_names.add(name)

            if offset is None:
                break

        return sorted(document_names)

    def get_document_info(self) -> list[dict[str, Any]]:
        """
        Aggregate document information from Qdrant payloads.
        Return one record per document containing:
        - document_id
        - document_name
        - chunk_count
        - page_count
        - ingestion_timestamp (if available)
        """
        if not self.collection_exists():
            return []

        docs: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection_name,
                with_payload=["document_id", "document_name", "page_number", "ingestion_timestamp"],
                with_vectors=False,
                limit=100,
                offset=offset,
            )

            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("document_id")
                doc_name = payload.get("document_name")
                page_number = payload.get("page_number")
                ing_timestamp = payload.get("ingestion_timestamp")

                if not doc_id:
                    continue

                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "document_name": doc_name or "Unknown",
                        "chunk_count": 0,
                        "pages": set(),
                        "ingestion_timestamp": ing_timestamp,
                    }

                entry = docs[doc_id]
                entry["chunk_count"] += 1
                if page_number is not None:
                    try:
                        entry["pages"].add(int(page_number))
                    except (ValueError, TypeError):
                        pass
                
                if ing_timestamp and not entry["ingestion_timestamp"]:
                    entry["ingestion_timestamp"] = ing_timestamp

            if offset is None:
                break

        results = []
        for doc_id, entry in docs.items():
            pages = entry["pages"]
            page_count = max(pages) if pages else 0
            
            results.append({
                "document_id": doc_id,
                "document_name": entry["document_name"],
                "chunk_count": entry["chunk_count"],
                "page_count": page_count,
                "ingestion_timestamp": entry["ingestion_timestamp"],
            })

        return results

    def delete_document(self, document_id: str) -> int:
        """
        Delete every point whose payload contains:
        payload["document_id"] == document_id

        Use Qdrant payload filtering.
        Do not recreate the collection.
        Return the number of deleted chunks.
        """
        if not self.collection_exists():
            return 0

        # Count chunks before deletion
        count_result = self._client.count(
            collection_name=self._collection_name,
            count_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        chunk_count = count_result.count

        if chunk_count == 0:
            return 0

        # Perform deletion
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(value=document_id),
                    )
                ]
            )
        )

        # Verify deletion
        verify_result = self._client.count(
            collection_name=self._collection_name,
            count_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="document_id",
                        match=rest.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        if verify_result.count > 0:
            import logging
            logging.getLogger(__name__).error(
                "Deletion verification failed: %d chunks remain for document %s",
                verify_result.count,
                document_id,
            )
            raise RuntimeError(f"Deletion verification failed. {verify_result.count} chunks still remain.")

        return chunk_count



_DEFAULT_VECTOR_STORE: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    global _DEFAULT_VECTOR_STORE
    if _DEFAULT_VECTOR_STORE is None:
        _DEFAULT_VECTOR_STORE = QdrantVectorStore()
    return _DEFAULT_VECTOR_STORE
