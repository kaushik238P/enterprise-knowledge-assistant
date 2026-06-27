# backend/agents/providers/base.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agents.web_search import WebSearchResponse

class BaseWebSearchProvider(ABC):
    """
    Abstract interface for all Web Search providers (Exa, Tavily, Brave, etc.).
    """

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int,
    ) -> "WebSearchResponse":
        """
        Executes a search query and returns a normalized WebSearchResponse.
        """
        pass