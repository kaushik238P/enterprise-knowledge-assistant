# backend/agents/web_search.py
import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agents.providers.base import BaseWebSearchProvider

logger = logging.getLogger(__name__)

__all__ = [
    "WebSearchResult",
    "WebSearchResponse",
    "WebSearchService",
    "create_web_search_service",
    "get_web_search_service",
]

MAX_RESULTS = 20  # Maximum number of results allowed per query

@dataclass(frozen=True)
class WebSearchResult:
    """Immutable representation of a single web search result."""
    title: str
    url: str
    content: str
    source: str = "web"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source": self.source,
        }

@dataclass(frozen=True)
class WebSearchResponse:
    """Immutable structured response from a web search query."""
    query: str
    results: list[WebSearchResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
        }

class WebSearchService:
    """
    Production-grade provider-agnostic web search service.
    Delegates all search operations to the injected provider.
    """

    def __init__(self, provider: "BaseWebSearchProvider") -> None:
        if provider is None:
            raise ValueError("Provider cannot be None.")
        self._provider = provider
        self.last_results: list = []
        self.last_sufficiency: bool = True
        logger.info(
            "WebSearchService initialized | provider=%s",
            type(provider).__name__,
        )

    def is_ready(self) -> bool:
        """Returns whether the injected provider is initialized."""
        return self._provider is not None

    def search(
        self,
        query: str,
        max_results: int | None = None,
        topic: str = "general",  # Retained parameter for backward compatibility
    ) -> WebSearchResponse:
        """
        Executes a web search through the injected provider.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        limit = max_results if max_results is not None else 5
        if limit <= 0:
            raise ValueError(f"max_results must be > 0. Got: {limit}.")
        if limit > MAX_RESULTS:
            raise ValueError(f"max_results cannot exceed {MAX_RESULTS}.")

        response = self._provider.search(query=query, max_results=limit)
        
        # Convert and cache results
        from uuid import uuid4
        from backend.retrieval.retriever import RetrievalResult
        
        retrieval_results = []
        for r in response.results:
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=uuid4(),
                    score=1.0,
                    content=r.content,
                    metadata={
                        "document_name": r.title,
                        "source_file": r.title,
                        "url": r.url,
                        "page_number": 1,
                        "section_title": r.title,
                        "content_type": "text",
                        "source": "web",
                    },
                )
            )
        self.last_results = retrieval_results
        self.last_sufficiency = len(response.results) > 0
        
        return response

# Singleton instance management for WebSearchService
_WEB_SEARCH_SERVICE: WebSearchService | None = None

def create_web_search_service(provider: Optional["BaseWebSearchProvider"] = None) -> WebSearchService:
    """
    Factory that assembles and returns a production-ready WebSearchService.
    Dynamically loads the configured provider at runtime to prevent circular imports.
    """
    if provider is None:
        from backend.config.settings import settings
        from backend.agents.providers.exa import ExaWebSearchProvider
        
        # Load configured provider settings
        provider_name = settings.web_search_provider.lower().strip()
        if provider_name == "exa":
            provider = ExaWebSearchProvider()
        else:
            raise ValueError(f"Unsupported web search provider: {provider_name}")

    return WebSearchService(provider=provider)

def get_web_search_service() -> WebSearchService:
    global _WEB_SEARCH_SERVICE
    if _WEB_SEARCH_SERVICE is None:
        _WEB_SEARCH_SERVICE = create_web_search_service()
    return _WEB_SEARCH_SERVICE

def _reset_web_search_service() -> None:
    """
    Resets the singleton instance. Used strictly in unit tests.
    """
    global _WEB_SEARCH_SERVICE
    _WEB_SEARCH_SERVICE = None