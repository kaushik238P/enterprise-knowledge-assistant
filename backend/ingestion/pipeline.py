# backend/ingestion/pipeline.py

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID

from backend.config.settings import settings
from backend.embeddings.hybride_embedder import HybridEmbedder, HybridEmbeddingResult
from backend.ingestion.chunker import chunk_document
from backend.ingestion.metadata import ChunkData, DocumentData
from backend.ingestion.parser import parse_document
from backend.vectorstore.qdrant import QdrantVectorStore, get_vector_store

# Create module logger
logger = logging.getLogger(__name__)

__all__ = [
    "IngestionResult",
    "DocumentIngestionPipeline",
    "get_ingestion_pipeline",
]


@dataclass(slots=True, frozen=True)
class IngestionResult:
    document_id: UUID
    document_name: str
    chunk_count: int
    stored_vectors: int


class DocumentIngestionPipeline:
    """
    Orchestrates the full document ingestion pipeline:
        parse → chunk → embed → upsert into Qdrant
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedder: Optional[HybridEmbedder] = None,
    ) -> None:
        self._vector_store: QdrantVectorStore = (
            vector_store if vector_store is not None else get_vector_store()
        )
        self._embedder: HybridEmbedder = (
            embedder if embedder is not None else HybridEmbedder()
        )

        logger.info(
            "DocumentIngestionPipeline initialized | vector_store=%s | embedder=%s",
            type(self._vector_store).__name__,
            type(self._embedder).__name__,
        )

    def ingest(self, file_path: str, document_name: Optional[str] = None) -> IngestionResult:
        """
        Runs the full ingestion pipeline for a single document.

        Steps:
            1. Validate file_path
            2. Parse document into structured elements
            3. Chunk elements into ChunkData objects
            4. Generate dense + sparse embeddings
            5. Create Qdrant collection if missing
            6. Upsert chunks and embeddings into Qdrant
            7. Return IngestionResult

        Raises:
            ValueError: If file_path is empty, parser returns no elements,
                        or chunker returns no chunks.
            RuntimeError: If embedding count mismatches or Qdrant upsert fails.
        """
        # ── Step 1: Validate file_path ────────────────────────────────────────
        if not file_path or not file_path.strip():
            raise ValueError("file_path cannot be empty or whitespace.")

        resolved_path = Path(file_path.strip()).resolve()
        
        if not resolved_path.exists():
            raise FileNotFoundError(
            f"Document not found: {resolved_path}"
        )

        logger.info(
            "Ingestion started | file=%s",
            resolved_path.name,
        )

        # ── Step 2: Parse document ────────────────────────────────────────────
        document_data: DocumentData = parse_document(resolved_path, document_name=document_name)

        if not document_data.elements:
            raise ValueError(
                f"Parser returned no elements for document '{resolved_path.name}'. "
                "The file may be empty, corrupt, or in an unsupported format."
            )

        logger.info(
            "Document parsed | file=%s | elements=%d | pages=%d",
            document_data.document_name,
            len(document_data.elements),
            document_data.total_pages,
        )

        # ── Step 3: Chunk document ────────────────────────────────────────────
        chunks: list[ChunkData] = chunk_document(document_data)

        if not chunks:
            raise ValueError(
                f"Chunker returned no chunks for document '{resolved_path.name}'. "
                "All elements may have been filtered out as too short or duplicate."
            )

        logger.info(
            "Chunks created | file=%s | chunk_count=%d",
            document_data.document_name,
            len(chunks),
        )

        # ── Step 4: Generate embeddings ───────────────────────────────────────
        logger.info(
            "Generating embeddings | chunks=%d",
            len(chunks),
        )
        embeddings: HybridEmbeddingResult = self._embedder.embed_chunks_hybrid(chunks)

        if len(embeddings.dense_vectors) != len(chunks):
            raise RuntimeError(
                f"Dense embedding count mismatch: "
                f"expected {len(chunks)}, got {len(embeddings.dense_vectors)}."
            )

        if len(embeddings.sparse_vectors) != len(chunks):
            raise RuntimeError(
                f"Sparse embedding count mismatch: "
                f"expected {len(chunks)}, got {len(embeddings.sparse_vectors)}."
            )

        logger.info(
            "Embeddings created | file=%s | dense=%d | sparse=%d",
            document_data.document_name,
            len(embeddings.dense_vectors),
            len(embeddings.sparse_vectors),
        )

        # ── Step 5: Create collection if missing ──────────────────────────────
        if not self._vector_store.collection_exists():
            logger.info(
                "Qdrant collection '%s' not found. Creating...",
                settings.qdrant_collection_name,
            )
            self._vector_store.create_collection()

        # ── Step 6: Upsert chunks into Qdrant ────────────────────────────────
        try:
            self._vector_store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
            )
            stats = self._vector_store.get_collection_stats()

            logger.info(
                "Collection stats | points=%d | indexed_vectors=%d",
                stats.points_count,
                stats.indexed_vectors_count,
            )
            
        except Exception as exc:
            logger.exception(
                "Qdrant upsert failed | file=%s | error=%s",
                document_data.document_name,
                exc,
            )
            raise RuntimeError(
                f"Qdrant upsert failed for '{document_data.document_name}': {exc}"
            ) from exc

        logger.info(
            "Qdrant upsert completed | file=%s | stored_vectors=%d",
            document_data.document_name,
            len(chunks),
        )

        # ── Step 7: Return IngestionResult ────────────────────────────────────
        result = IngestionResult(
            document_id=document_data.document_id,
            document_name=document_data.document_name,
            chunk_count=len(chunks),
            stored_vectors=len(chunks),
        )

        logger.info(
            "Ingestion completed | file=%s | chunk_count=%d | stored_vectors=%d",
            result.document_name,
            result.chunk_count,
            result.stored_vectors,
        )

        return result


# Singleton pattern
_DEFAULT_PIPELINE: Optional[DocumentIngestionPipeline] = None


def get_ingestion_pipeline() -> DocumentIngestionPipeline:
    """
    Returns the default DocumentIngestionPipeline instance, performing lazy
    initialization and reusing the singleton instance on subsequent calls.
    """
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = DocumentIngestionPipeline()
    return _DEFAULT_PIPELINE