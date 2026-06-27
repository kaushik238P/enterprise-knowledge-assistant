# backend/api/ingest.py

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from backend.config.settings import settings
from backend.ingestion.pipeline import (
    DocumentIngestionPipeline,
    get_ingestion_pipeline,
)
from backend.schemas.ingest import IngestResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)


@router.post(
    "",
    response_model=IngestResponse,
    summary="Ingest a document",
    description=(
        "Uploads a document, parses it, creates chunks, generates "
        "hybrid embeddings, and stores them in Qdrant."
    ),
)
async def ingest_document(
    file: UploadFile = File(...),
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """
    Upload and ingest a supported document.
    """

    filename = file.filename or "unknown"

    logger.info(
        "Incoming request | endpoint=/ingest | filename=%s",
        filename,
    )

    start_time = time.perf_counter()

    suffix = Path(filename).suffix.lower()

    if suffix not in settings.supported_document_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: "
                f"{', '.join(settings.supported_document_types)}"
            ),
        )

    temp_file_path: str | None = None

    try:
        logger.info("Creating temporary upload file")

        with tempfile.NamedTemporaryFile(
            delete=False,
            prefix="eka_",
            suffix=suffix,
        ) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_file_path = tmp_file.name

        file_size = os.path.getsize(temp_file_path)

        max_size = settings.max_document_size_mb * 1024 * 1024

        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File exceeds maximum allowed size "
                    f"({settings.max_document_size_mb} MB)."
                ),
            )

        logger.info(
            "Temporary file created | path=%s | size=%.2f MB",
            temp_file_path,
            file_size / (1024 * 1024),
        )

        logger.info("Starting ingestion pipeline")

        result = pipeline.ingest(
            file_path=temp_file_path,
            document_name=filename,
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Ingestion completed | filename=%s | "
            "chunks=%d | vectors=%d | elapsed=%.3fs",
            filename,
            result.chunk_count,
            result.stored_vectors,
            elapsed,
        )

        return IngestResponse(
            filename=filename,
            chunk_count=result.chunk_count,
            stored_vectors=result.stored_vectors,
            status="success",
        )

    except ValueError as exc:
        logger.exception(
            "Validation error during ingestion"
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "Pipeline error during ingestion"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected ingestion error"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while ingesting document.",
        ) from exc

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(
                    "Temporary file removed | %s",
                    temp_file_path,
                )
            except Exception:
                logger.exception(
                    "Failed to remove temporary file"
                )

        await file.close()