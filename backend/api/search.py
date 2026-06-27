# backend/api/search.py

import logging
import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from backend.retrieval.retriever import Retriever, get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search document chunks",
    description="Retrieve relevant document chunks using hybrid retrieval and cross-encoder reranking.",
)

def search_documents(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    """
    Search document chunks based on a query.
    """
    logger.info(
        "Incoming request: POST /search | query='%s' | top_k=%s",
        request.query,
        request.top_k,
    )
    start_time = time.perf_counter()
    logger.info("Search operation started")

    try:
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
        )

        items = [
            SearchResultItem(
                chunk_id=str(result.chunk_id),
                score=result.score,
                content=result.content,
                metadata=result.metadata,
            )
            for result in results
        ]

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Search operation completed | results_returned=%d | elapsed_time=%.4fs",
            len(items),
            elapsed,
        )
        return SearchResponse(results=items)

    except ValueError as exc:
        elapsed = time.perf_counter() - start_time
        logger.error(
            "Search operation failed with ValueError: %s | elapsed_time=%.4fs",
            exc,
            elapsed,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception(
            "Search failed: %s",
            exc,
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
        
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception(
            "Search failed: %s",
            exc,
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during search.",
        )