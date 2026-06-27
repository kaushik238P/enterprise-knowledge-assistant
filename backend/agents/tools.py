# backend/agents/tools.py
import logging
import time
from langchain_core.tools import tool

from backend.retrieval.retriever import (
    RetrievalResult,
    Retriever,
    get_retriever,
)

from backend.agents.web_search import (
    WebSearchResult,
    get_web_search_service,
    WebSearchService,
)

logger = logging.getLogger(__name__)

__all__ = [
    "document_search_tool",
    "web_search_tool",
]

def _get_retriever()-> Retriever:
    """
    Returns the singleton Retriever instance.
    """
    return get_retriever()


def _get_web_search()-> WebSearchService:
    """
    Returns the singleton WebSearchService instance.
    """
    return get_web_search_service()




def _validate_query(query: str) -> str:
    """
    Validates and normalizes the input query.

    Args:
        query: The raw query string.

    Returns:
        The stripped query string.

    Raises:
        TypeError:
            If the query is not a string.
        ValueError:
            If the query is empty or contains only whitespace.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty or whitespace-only.")

    return query


def _format_document_results(results: list[RetrievalResult]) -> str:
    """ 
    Formats retrieved document chunks into a readable string.
    """
    if not results:
        return "No relevant documents found."

    formatted = []

    for i, result in enumerate(results, start=1):
        formatted.append(
    (
        f"Document {i}\n"
        f"-----------\n"
        f"Document Name : {result.metadata.get('document_name', 'Unknown')}\n"
        f"Page Number   : {result.metadata.get('page_number', 'Unknown')}\n"
        f"Chunk ID      : {result.chunk_id}\n"
        f"Score         : {result.score:.4f}\n\n"
        f"{result.content}\n"
    )
)

    return "\n".join(formatted)


def _format_web_results(results: list[WebSearchResult]) -> str:
    """
    Formats Tavily search results into a readable string.
    """
    if not results:
        return "No relevant web results found."

    formatted = []

    for i, result in enumerate(results, start=1):
        formatted.append(
            (
                f"Result {i}\n"
                f"--------\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n\n"
                f"{result.content}\n"
            )
        )

    return "\n".join(formatted)
    

@tool("document_search")
def document_search_tool(query: str) -> str:
    """
    Searches the Enterprise Knowledge Base using hybrid retrieval and returns the most relevant document chunks.

    Args:
        query: The search query string.

    Returns:
        A formatted string containing the content, document name, and page number of the retrieved documents,
        or a default message if no results are found.

    Raises:
        ValueError: If the input query is empty or whitespace-only.
        RuntimeError: If retrieval fails due to an unexpected error.
    """
    logger.info("document search tool started | query='%s'", query)
    start_time = time.perf_counter()

    query = _validate_query(query)

    try:
        logger.info("Executing retrieval | query='%s'", query[:100],
                    )
        results = _get_retriever().retrieve(query=query)

        logger.info("document search tool retrieved %d results", len(results))


        elapsed = time.perf_counter() - start_time

        logger.info(
            "Document search tool finished | duration=%.4fs",
        elapsed,
        )

        return _format_document_results(results)
    

    except ValueError:
        # Re-raise validation exception
        raise
    except Exception as exc:
        
        logger.exception(
            "Unexpected error in document search tool for query '%s': %s",
            query,
            exc,
        )
        raise RuntimeError(
            "Document search failed."
        ) from exc


@tool("web_search")
def web_search_tool(query: str) -> str:
    """
    Searches the internet using Tavily.

    Args:
        query: The search query string.

    Returns:
        A formatted string containing titles, URLs, and snippets of the web results,
        or a default message if no results are found.

    Raises:
        ValueError: If the input query is empty or whitespace-only.
        RuntimeError: If web search fails due to an unexpected error.
    """
    logger.info("web search tool started | query='%s'", query[:100])
    start_time = time.perf_counter()

    query = _validate_query(query)

    try:
        logger.info(
            "Executing web search | query='%s'",
        query[:100],
        )
        response = _get_web_search().search(query=query)

        logger.info("web search tool retrieved %d results", len(response.results))

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Web search tool finished | duration=%.4fs",
        elapsed,
        )

        return _format_web_results(response.results)

    except ValueError:
        # Re-raise validation exception
        raise
    except Exception as exc:
        logger.exception(
            "Unexpected error in web search tool for query '%s': %s", query, exc
        )
        raise RuntimeError(
            "Web search failed."
        ) from exc