# backend/agents/nodes.py
import logging
import time
from typing import Any
from uuid import uuid4

from backend.agents.state import EnterpriseState
from backend.agents.router import get_router
from backend.llm.generator import get_generator
from backend.llm.evaluator import get_evaluator
from backend.agents.tools import (
    document_search_tool,
    web_search_tool,
    _format_document_results,
    _format_web_results,
)
from backend.config.settings import settings
from backend.retrieval.retriever import get_retriever, RetrievalResult
from backend.agents.web_search import get_web_search_service

logger = logging.getLogger(__name__)

# Initialize singletons
_ROUTER = get_router()
_GENERATOR = get_generator()
_EVALUATOR = get_evaluator()

__all__ = [
    "route_node",
    "document_search_node",
    "web_search_node",
    "hybrid_search_node",
    "generate_answer_node",
    "evaluate_answer_node",
]

def _require(
    state: EnterpriseState,
    key: str,
) -> str:
    """
    Returns a required string value from the workflow state.
    """
    value = state.get(key)

    if not isinstance(value, str):
        raise ValueError(f"State field '{key}' is missing.")

    value = value.strip()

    if not value:
        raise ValueError(f"State field '{key}' cannot be empty.")

    return value

def route_node(state: EnterpriseState) -> dict[str, str]:
    """
    Evaluates the user query to decide the routing path: documents, web, or hybrid.
    """
    logger.info("route_node started")
    start_time = time.perf_counter()

    try:
        query = _require(state, "query")
        route = _ROUTER.route(query)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "route_node finished | Selected route: '%s' | duration=%.4fs",
            route,
            elapsed,
        )
        return {"route": route}

    except Exception as exc:
        logger.exception("Unexpected error in route_node: %s", exc)
        raise RuntimeError("Route node failed") from exc


def document_search_node(state: EnterpriseState) -> dict[str, str]:
    """
    Searches the Enterprise Knowledge Base for relevant documents using hybrid retrieval.
    """
    start_time = time.perf_counter()

    try:
        query = _require(state, "query")
        logger.info(
            "document search node started | query='%s'",
            query[:100],
        )
        context = document_search_tool.invoke({"query": query})

        elapsed = time.perf_counter() - start_time
        logger.info(
            "document search node finished | duration=%.4fs | context_chars=%d", 
            elapsed,
            len(context)
        )
        return {"context": context}

    except Exception as exc:
        logger.exception("Unexpected error in document_search_node: %s", exc)
        raise RuntimeError("Document search node failed.") from exc


def web_search_node(state: EnterpriseState) -> dict[str, str]:
    """
    Searches the web for public information using Exa.
    """
    start_time = time.perf_counter()

    try:
        query = _require(state, "query")
        logger.info("web search node started | query='%s'", query[:100])
        context = web_search_tool.invoke({"query": query})

        elapsed = time.perf_counter() - start_time
        logger.info(
            "web search node finished | duration=%.4fs | context_chars=%d", 
            elapsed,
            len(context)
        )
        return {"context": context}

    except Exception as exc:
        logger.exception("Unexpected error in web search node: %s", exc)
        raise RuntimeError("Web search node failed.") from exc

def hybrid_search_node(state: EnterpriseState) -> dict[str, str]:
    """
    Executes both document and web searches, combining the results.
    """
    start_time = time.perf_counter()

    try:
        query = _require(state, "query")
        logger.info("hybrid_search_node started | query='%s'", query[:100])
        doc_context = document_search_tool.invoke({"query": query})
        web_context = web_search_tool.invoke({"query": query})

        combined_context = (
            "=== DOCUMENT RESULTS ===\n\n"
            f"{doc_context}\n\n"
            "=== WEB RESULTS ===\n\n"
            f"{web_context}"
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "hybrid_search_node finished | duration=%.4fs | context_chars=%d", 
            elapsed,
            len(combined_context)
        )
        return {"context": combined_context}

    except Exception as exc:
        logger.exception("Unexpected error in hybrid search node: %s", exc)
        raise RuntimeError("Hybrid search node failed.") from exc


def generate_answer_node(state: EnterpriseState) -> dict[str, str]:
    """
    Generates a grounded answer from the compiled context.
    """
    start_time = time.perf_counter()

    try:
        query = _require(state, "query")
        context = _require(state, "context")
        route = state.get("route", "documents")

        # Check sufficiency based on route
        sufficiency = True
        if route == "documents":
            sufficiency = getattr(get_retriever(), "last_sufficiency", True)
        elif route == "web":
            sufficiency = getattr(get_web_search_service(), "last_sufficiency", True)
        elif route == "hybrid":
            sufficiency = getattr(get_retriever(), "last_sufficiency", True) or getattr(get_web_search_service(), "last_sufficiency", True)

        if not sufficiency:
            logger.info("generate_answer_node: retrieval sufficiency is False, returning refusal answer")
            return {"answer": "The retrieved context does not contain this information."}

        logger.info(
            "generate_answer_node started | query='%s' | context_chars=%d", 
            query[:100], 
            len(context)
        )

        generation = _GENERATOR.generate_from_context(query=query, context=context)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "generate answer node finished | duration=%.4fs | answer_chars=%d", 
            elapsed,
            len(generation.answer)
        )
        return {"answer": generation.answer}

    except Exception as exc:
        logger.exception("Unexpected error in generate answer node: %s", exc)
        raise RuntimeError("Generate answer node failed.") from exc


def evaluate_answer_node(state: EnterpriseState) -> dict[str, Any]:
    """
    Evaluates the generated answer against the query and context (optional).
    """
    start_time = time.perf_counter()

    try:
        if not settings.enable_answer_evaluation:
            logger.info("Answer evaluation is disabled. Skipping evaluate_answer_node.")
            return {"evaluation": None}

        query = _require(state, "query")
        context = _require(state, "context")
        answer = _require(state, "answer")
        route = state.get("route", "documents")

        logger.info(
            "evaluate answer node started | query='%s' | answer_chars=%d", 
            query[:100],
            len(answer)
        )

        # Get results and sufficiency based on route
        from unittest.mock import Mock
        retriever = get_retriever()
        web_search = get_web_search_service()

        retrieval_results = []
        sufficiency = True

        if route == "documents":
            if not isinstance(retriever, Mock):
                retrieval_results = getattr(retriever, "last_results", [])
                sufficiency = getattr(retriever, "last_sufficiency", True)
        elif route == "web":
            if not isinstance(web_search, Mock):
                retrieval_results = getattr(web_search, "last_results", [])
                sufficiency = getattr(web_search, "last_sufficiency", True)
        elif route == "hybrid":
            ret_res = []
            suff = True
            if not isinstance(retriever, Mock):
                ret_res.extend(getattr(retriever, "last_results", []))
                suff = getattr(retriever, "last_sufficiency", True)
            if not isinstance(web_search, Mock):
                ret_res.extend(getattr(web_search, "last_results", []))
                suff = suff or getattr(web_search, "last_sufficiency", True)
            retrieval_results = ret_res
            sufficiency = suff
        else:
            retrieval_results = []
            sufficiency = True

        kwargs = {}
        if retrieval_results:
            kwargs["retrieval_results"] = retrieval_results
        if not sufficiency:
            kwargs["metadata"] = {"retrieval_sufficiency": False}

        evaluation = _EVALUATOR.evaluate(
            query=query,
            context=context,
            answer=answer,
            **kwargs
        )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "evaluate answer node finished | duration=%.4fs | passed=%s", 
            elapsed,
            evaluation.passed
        )
        return {"evaluation": evaluation}

    except Exception as exc:
        logger.exception("Unexpected error in evaluate answer node: %s", exc)
        raise RuntimeError("Evaluate answer node failed.") from exc