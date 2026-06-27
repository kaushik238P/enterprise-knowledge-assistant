# backend/api/chat.py
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.chat import ChatRequest, ChatResponse, SourceResponse, EvaluationResponse
from backend.llm.rag_chain import RAGChain, get_rag_chain

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the assistant",
    description="Generate a grounded answer and evaluate it using the RAG pipeline.",
)
def chat_with_assistant(
    request: ChatRequest,
    use_agent: bool = False,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> ChatResponse:
    """
    Generate an answer to the query using retrieval, LLM, and evaluate it.
    """
    logger.info("Incoming request: POST /chat | query='%s' | use_agent=%s", request.query, use_agent)
    start_time = time.perf_counter()
    logger.info("Chat operation started")

    try:
        rag_result = rag_chain.invoke(query=request.query, use_agent=use_agent)
        logger.info(
            "Generated answer | length=%d characters",
            len(rag_result.answer),
        )

        # Parse evaluation response
        if rag_result.evaluation is not None:
            eval_res = rag_result.evaluation
            evaluation_resp = EvaluationResponse(
                passed=eval_res.passed,
                overall_score=eval_res.confidence_score,
                grounding_score=eval_res.grounding_score,
                coverage_score=eval_res.coverage_score,
                citation_score=eval_res.citation_score,
                numerical_score=eval_res.numerical_score,
                table_score=eval_res.table_score,
                hallucination_score=eval_res.hallucination_score,
                hallucination_risk=eval_res.hallucination_risk,
                confidence=eval_res.confidence_score,
                explanation=eval_res.reasoning or eval_res.summary or "Evaluation completed.",
                warnings=[],
            )
        else:
            evaluation_resp = EvaluationResponse(
                passed=True,
                overall_score=1.0,
                grounding_score=1.0,
                coverage_score=1.0,
                citation_score=1.0,
                numerical_score=1.0,
                table_score=1.0,
                hallucination_score=1.0,
                hallucination_risk="Low",
                confidence=1.0,
                explanation="Evaluation skipped per configuration.",
                warnings=[],
            )

        # Parse sources response using CitationService
        from backend.services.citation_service import get_citation_service

        sufficiency = True
        if rag_result.evaluation and getattr(rag_result.evaluation, "status", None) == "INSUFFICIENT_EVIDENCE":
            sufficiency = False

        formatted_cits = get_citation_service().filter_and_format_citations(
            results=rag_result.retrieval_results,
            answer=rag_result.answer,
            retrieval_sufficiency=sufficiency,
        )

        sources = []
        for s in formatted_cits:
            score_val = s["score"]
            if score_val < 0.0:
                score_val = 0.0
            elif score_val > 1.0:
                score_val = 1.0

            sources.append(
                SourceResponse(
                    document=s["document"],
                    page=s["page"],
                    section=s["section"],
                    section_path=s["section_path"],
                    score=score_val,
                    content_type=s.get("content_type"),
                    chunk_id=s["chunk_id"],
                )
            )

        response = ChatResponse(
            answer=rag_result.answer,
            evaluation=evaluation_resp,
            sources=sources,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Chat operation completed | passed=%s | grounding=%.3f | coverage=%.3f | elapsed_time=%.4fs",
            response.evaluation.passed,
            response.evaluation.grounding_score,
            response.evaluation.coverage_score,
            elapsed,
        )
        return response

    except ValueError as exc:
        elapsed = time.perf_counter() - start_time
        logger.error(
            "Chat operation failed with ValueError: %s | elapsed_time=%.4fs",
            exc,
            elapsed,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except TimeoutError as exc:
        elapsed = time.perf_counter() - start_time
        logger.error(
            "Chat operation timed out: %s | elapsed_time=%.4fs",
            exc,
            elapsed,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The request timed out waiting for the search provider or model response.",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception(
            "Chat operation failed | elapsed_time=%.4fs",
            elapsed,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the chat session.",
        )