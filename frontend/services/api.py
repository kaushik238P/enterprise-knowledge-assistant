# frontend/services/api.py
import logging
import os
import requests
from typing import Any, Optional

from backend.config.settings import settings

logger = logging.getLogger(__name__)

__all__ = [
    "APIClient",
    "BACKEND_URL",
    "set_backend_status",
    "get_backend_status",
    "is_backend_online",
    "clear_backend_status",
    "get_backend_url",
]

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

_HEALTH_ENDPOINT = "/health"
_CHAT_ENDPOINT = "/chat"
_UPLOAD_ENDPOINT = "/ingest"
_DOCUMENTS_ENDPOINT = "/documents"

_HEALTH_TIMEOUT = 5
# Configured via settings.py
_CHAT_TIMEOUT = (10, settings.frontend_read_timeout)
_UPLOAD_TIMEOUT = (10, settings.upload_timeout)

class APIClient:
    """
    Client for interacting with the FastAPI backend.
    """

    def __init__(self, base_url: str = BACKEND_URL) -> None:
        self.base_url = base_url.rstrip("/")
    
    def _handle_request_exception(
        self,
        exc: requests.RequestException,
        ) -> dict[str, Any]:
        """
        Converts a requests exception into a consistent error response.
        """
        logger.error("HTTP request failed: %s", exc)

        response = getattr(exc, "response", None)

        if response is not None:
            try:
                detail = response.json().get("detail", str(exc))
                return {"error": detail}
            except ValueError:
                return {
                    "error": response.text or str(exc),
                }

        # Differentiate between timeout issues
        if isinstance(exc, requests.exceptions.ConnectTimeout):
            return {"error": "Connection timed out. Please check if the server is running."}
        elif isinstance(exc, requests.exceptions.ReadTimeout):
            return {"error": "Request timed out waiting for the server to respond."}

        return {"error": str(exc)}
    
    def _post(
        self,
        endpoint: str,
        **kwargs: Any,
        ) -> dict[str, Any]:
        """
        Executes a POST request and returns the JSON response.
        """
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            return self._handle_request_exception(exc)
        
    def _get(
        self,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Executes a GET request and returns the JSON response.
        """
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            return self._handle_request_exception(exc)

    def _delete(
        self,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Executes a DELETE request and returns the JSON response.
        """
        try:
            response = requests.delete(
                f"{self.base_url}{endpoint}",
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as exc:
            return self._handle_request_exception(exc)
        
    def _validate_query(
        self,
        query: str,
    ) -> str:
        """
        Validates and normalizes a chat query.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        return query
    
    def _validate_upload(
        self,
        file_name: str,
        file_bytes: bytes,
    ) -> None:
        """
        Validates upload inputs.
        """
        if not isinstance(file_name, str):
            raise TypeError(
                "file_name must be a string."
            )

        if not file_name.strip():
            raise ValueError(
                "file_name cannot be empty."
            )

        if not isinstance(file_bytes, bytes):
            raise TypeError(
                "file_bytes must be bytes."
            )

        if not file_bytes:
            raise ValueError(
                "file_bytes cannot be empty."
            )

    def health(self) -> bool:
        """
        Check if the FastAPI backend is running and healthy.
        """
        try:
            response = requests.get(f"{self.base_url}{_HEALTH_ENDPOINT}", timeout=_HEALTH_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except requests.RequestException as exc:
            logger.error("Backend health check failed: %s", exc)
            return False

    def chat(self, query: str, use_agent: bool = False) -> dict[str, Any]:
        """
        Send a query to the chat endpoint.
        """
        query = self._validate_query(query)
        payload = {"query": query}

        params = {
            "use_agent": str(use_agent).lower(),
        }

        return self._post(
            _CHAT_ENDPOINT,
            json=payload,
            params=params,
            timeout=_CHAT_TIMEOUT,
        )

    def upload_document(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
        """
        Upload a document file to the ingestion endpoint.
        """
        self._validate_upload(file_name, file_bytes)
        files = {
           "file": (
                file_name,
                file_bytes,
                "application/octet-stream",
            )
        }

        return self._post(
            _UPLOAD_ENDPOINT,
            files=files,
            timeout=_UPLOAD_TIMEOUT,
        )

    def list_documents(self) -> list[str]:
        """
        Returns the list of ingested documents from the backend.
        """
        response = self._get(
            _DOCUMENTS_ENDPOINT,
            timeout=_HEALTH_TIMEOUT,
        )

        if isinstance(response, dict) and "error" in response:
            logger.error(
                "Failed to retrieve documents: %s",
                 response["error"],
            )
            return []

        if isinstance(response, list):
            return [doc.get("document_name") for doc in response if isinstance(doc, dict) and doc.get("document_name")]
        elif isinstance(response, dict):
            return response.get("documents", [])
        return []

    def list_documents_detailed(self) -> list[dict[str, Any]]:
        """
        Returns a detailed list of ingested documents from the backend.
        """
        response = self._get(
            _DOCUMENTS_ENDPOINT,
            timeout=_HEALTH_TIMEOUT,
        )

        if isinstance(response, dict) and "error" in response:
            logger.error(
                "Failed to retrieve documents detailed: %s",
                 response["error"],
            )
            return []

        if isinstance(response, list):
            return response
        return []

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """
        Permanently deletes a document by ID.
        """
        return self._delete(
            f"{_DOCUMENTS_ENDPOINT}/{document_id}",
            timeout=_HEALTH_TIMEOUT,
        )