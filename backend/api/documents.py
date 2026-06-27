import logging
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.documents import DocumentInfo, DeleteDocumentResponse
from backend.services.document_service import (
    DocumentService,
    DocumentDeletionService,
    get_document_service,
    get_document_deletion_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get("", response_model=Union[list[DocumentInfo], dict[str, list[str]]])
def list_documents(
    legacy: bool = False,
    document_service: DocumentService = Depends(get_document_service),
) -> Any:
    """
    Returns the names of all ingested documents or rich metadata.
    """
    logger.info("Incoming request: GET /documents | legacy=%s", legacy)

    if legacy:
        documents = document_service.list_documents()
        logger.info("Returning %d documents in legacy format.", len(documents))
        return {
            "documents": documents,
        }

    docs_info = document_service.get_document_info()
    logger.info("Returning %d documents in detailed format.", len(docs_info))
    return docs_info


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    deletion_service: DocumentDeletionService = Depends(get_document_deletion_service),
) -> Any:
    """
    Permanently deletes a document by ID.
    """
    logger.info("Incoming request: DELETE /documents/%s", document_id)
    try:
        result = deletion_service.delete_document(document_id)
        return result
    except ValueError as exc:
        logger.warning("Document not found for deletion: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during deletion of document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )