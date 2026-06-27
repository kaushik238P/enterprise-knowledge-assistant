# backend/agents/providers/exa.py
import logging
import time
from backend.agents.providers.base import BaseWebSearchProvider
from backend.agents.web_search import WebSearchResult, WebSearchResponse
from backend.config.settings import settings

logger = logging.getLogger(__name__)

class ExaWebSearchProvider(BaseWebSearchProvider):
    """
    Exa search provider implementation.
    Insulates the application from Exa-specific data models and raw SDK exceptions.
    """

    def __init__(self) -> None:
        api_key = settings.exa_api_key
        if not api_key:
            raise ValueError(
                "Required environment variable 'EXA_API_KEY' is not set."
            )

        try:
            from exa_py import Exa
        except ImportError as exc:
            raise ImportError(
                "exa-py is not installed. Run: uv add exa-py"
            ) from exc

        self._client = Exa(api_key=api_key)
        logger.info("ExaWebSearchProvider initialized successfully.")

    def search(
        self,
        query: str,
        max_results: int,
    ) -> WebSearchResponse:
        """
        Performs web search and extracts content via Exa SDK.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if max_results <= 0:
            raise ValueError(f"max_results must be > 0. Got: {max_results}.")

        logger.info(
            "Exa search started | query='%s' | max_results=%d",
            query[:100],
            max_results,
        )
        start_time = time.perf_counter()

        try:
            response = self._client.search_and_contents(
                query=query,
                num_results=max_results,
                text=True,
            )
        except Exception as exc:
            logger.exception(
                "Exa SDK call failed | query='%s' | error=%s",
                query[:100],
                exc,
            )
            # Differentiate timeout issues from generic failures
            exc_str = str(exc).lower()
            if "timeout" in exc_str or "time out" in exc_str or "timed out" in exc_str:
                raise TimeoutError(f"Web search provider timed out: {exc}") from exc
            raise RuntimeError(f"Web search provider failed: {exc}") from exc

        elapsed = time.perf_counter() - start_time

        # Validate that the response contains the expected format
        if not hasattr(response, "results"):
            logger.error("Exa response payload is missing a 'results' attribute.")
            raise RuntimeError("Malformed provider response: missing results list.")

        results: list[WebSearchResult] = []
        for item in response.results:
            title = getattr(item, "title", "")
            url = getattr(item, "url", "")
            text = getattr(item, "text", "")

            # Normalize values to ensure no None values are passed to dataclass
            title = str(title).strip() if title is not None else ""
            url = str(url).strip() if url is not None else ""
            content = str(text).strip() if text is not None else ""

            # Truncate content to reduce token load and processing latency
            max_chars = settings.max_web_result_chars
            if len(content) > max_chars:
                content = content[:max_chars] + "... [truncated]"

            # Standard filtering rules
            if not title and not content:
                continue
            if not url:
                continue

            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    content=content,
                    source="web",
                )
            )

        logger.info(
            "Exa search complete | query='%s' | results=%d | duration=%.3fs",
            query[:100],
            len(results),
            elapsed,
        )

        return WebSearchResponse(query=query, results=results)