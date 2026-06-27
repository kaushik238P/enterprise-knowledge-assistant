import logging
import time
from typing import Any

from backend.vectorstore.qdrant import (
    QdrantVectorStore,
    get_vector_store,
)
from backend.schemas.documents import DeletionResult

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentService",
    "get_document_service",
    "DocumentDeletionService",
    "get_document_deletion_service",
]


class DocumentService:
    """
    Provides document-related operations.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._vector_store = vector_store or get_vector_store()

    def list_documents(self) -> list[str]:
        """
        Returns all unique document names stored in Qdrant.
        """
        logger.info("Listing ingested documents.")

        documents = self._vector_store.list_documents()

        logger.info(
            "Found %d documents.",
            len(documents),
        )

        return documents

    def get_document_info(self) -> list[dict[str, Any]]:
        """
        Returns rich metadata for all ingested documents.
        """
        logger.info("Getting document info for all ingested documents.")
        return self._vector_store.get_document_info()


class DocumentDeletionService:
    """
    Provides document deletion services with auditing and validation.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._vector_store = vector_store or get_vector_store()

    def delete_document(self, document_id: str) -> DeletionResult:
        logger.info("Audit: User action - deletion requested for document_id=%s", document_id)
        start_time = time.perf_counter()

        # 1. Verify existence and get document name
        all_docs = self._vector_store.get_document_info()
        target_doc = next((d for d in all_docs if d["document_id"] == document_id), None)

        if not target_doc:
            logger.error("Audit: Deletion failed - document_id=%s not found", document_id)
            raise ValueError(f"Document with ID '{document_id}' not found.")

        doc_name = target_doc["document_name"]
        logger.info("Audit: Document found - document_name='%s'", doc_name)

        # 2. Call vector store deletion (this also verifies deletion internally)
        try:
            chunks_deleted = self._vector_store.delete_document(document_id)
        except Exception as exc:
            logger.error("Audit: Deletion failed - Qdrant error: %s", exc)
            raise

        # 3. Calculate execution time
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # 4. Log full audit info
        logger.info(
            "Audit: Deletion successful | document_id=%s | document_name=%s | chunks_deleted=%d | elapsed_ms=%d | verification=passed",
            document_id,
            doc_name,
            chunks_deleted,
            elapsed_ms,
        )

        return DeletionResult(
            status="success",
            document_id=document_id,
            document_name=doc_name,
            chunks_deleted=chunks_deleted,
            elapsed_ms=elapsed_ms,
        )


_DEFAULT_DOCUMENT_SERVICE: DocumentService | None = None
_DEFAULT_DOCUMENT_DELETION_SERVICE: DocumentDeletionService | None = None


def get_document_service() -> DocumentService:
    global _DEFAULT_DOCUMENT_SERVICE

    if _DEFAULT_DOCUMENT_SERVICE is None:
        _DEFAULT_DOCUMENT_SERVICE = DocumentService()

    return _DEFAULT_DOCUMENT_SERVICE


def get_document_deletion_service() -> DocumentDeletionService:
    global _DEFAULT_DOCUMENT_DELETION_SERVICE

    if _DEFAULT_DOCUMENT_DELETION_SERVICE is None:
        _DEFAULT_DOCUMENT_DELETION_SERVICE = DocumentDeletionService()

    return _DEFAULT_DOCUMENT_DELETION_SERVICE