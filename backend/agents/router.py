# backend/agents/router.py

"""
This module implements the QueryRouter class, which uses the application's configured
LLM to classify incoming user queries into one of three routes: 'documents', 'web', or 'hybrid'.
"""

import logging
import time
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm.model import get_llm
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = [
    "QueryRouter",
    "get_router",
    "ROUTER_SYSTEM_PROMPT"
]

_ALLOWED_ROUTES = frozenset(
    {
        "documents",
        "web",
        "hybrid",
    }
)

ROUTER_SYSTEM_PROMPT = (
            "You are a routing classifier.\n\n"
            "Choose exactly one route.\n\n"
            "documents\n"
            "The answer is likely contained in the enterprise knowledge base.\n"
            "Examples:\n"
            "employee handbook\n"
            "internal policy\n"
            "uploaded documents\n"
            "technical documentation\n"
            "project files\n"
            "database schema\n"
            "meeting notes\n"
            "company reports\n\n"
            "web\n"
            "The answer requires recent or public internet information.\n"
            "Examples:\n"
            "today's news\n"
            "weather\n"
            "stock price\n"
            "latest AI model\n"
            "current CEO\n"
            "sports score\n\n"
            "hybrid\n"
            "The answer benefits from BOTH enterprise documents and web information.\n"
            "Examples:\n"
            "Compare our company policy with industry standards.\n"
            "Compare uploaded report with latest news.\n"
            "Explain our architecture and compare it with current best practices.\n\n"
            "Return ONLY one word:\n"
            "documents\n"
            "or\n"
            "web\n"
            "or\n"
            "hybrid\n"
            "No explanation."
        )

def _parse_response(response: object) -> str:
    """
    Extracts and normalizes the text content from an LLM response.

    Args:
        response: The raw response returned by the language model.

    Returns:
        The normalized response text in lowercase.
    """
    content = getattr(response, "content", response)

    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    return str(content).strip().lower()


class QueryRouter:
    """
    Decides the optimal routing path ('documents', 'web', or 'hybrid') for a query
    using the configured chat model.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """
        Initializes the QueryRouter.

        Args:
            llm: Optional chat model to use for classification. Defaults to the global LLM.
        """
        self._llm = llm or get_llm()

    def route(self, query: str) -> Literal["documents", "web", "hybrid"]:
        """
        Classifies the query into one of three routes: 'documents', 'web', or 'hybrid'.

        Args:
            query: The user query string to classify.

        Returns:
            The selected route string: 'documents', 'web', or 'hybrid'.

        Raises:
            ValueError: If the query is empty or contains only whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        logger.info("Router started | query='%s'", query[:100])
        start_time = time.perf_counter()

        try:
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            response = self._llm.invoke(messages)

            # Extract content from response
            decision = _parse_response(response)

            # Validate the route decision
            if decision not in _ALLOWED_ROUTES:
                logger.warning(
                    "LLM returned unexpected route classification: '%s'. Falling back to 'documents'.",
                    decision,
                )
                decision = "documents"

        except Exception as exc:
            logger.exception(
                "LLM classification failed during routing. Falling back to 'documents'. Error: %s",
                exc,
            )
            decision = "documents"

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Router finished | Selected route: '%s' | duration=%.4fs",
            decision,
            elapsed,
        )
        return decision


# Singleton instance management for QueryRouter
_DEFAULT_ROUTER: QueryRouter | None = None


def get_router() -> QueryRouter:
    """
    Returns the default lazy-initialized QueryRouter instance.
    """
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = QueryRouter()
    return _DEFAULT_ROUTER