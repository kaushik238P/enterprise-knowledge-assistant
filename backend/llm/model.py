# backend/llm/model.py
import logging
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

from backend.config.settings import settings

# Create module logger
logger = logging.getLogger(__name__)

__all__ = [
    "LLMFactory",
    "get_llm",
]


class LLMFactory:
    """
    Factory class responsible for selecting the provider, creating the LLM client,
    and returning a configured language model instance.
    """

    @staticmethod
    def create_llm() -> BaseChatModel:
        """
        Creates and returns a configured BaseChatModel instance based on the application settings.

        Raises:
            ValueError: If the provider is unsupported or the required API key is missing.
        """
        provider = settings.llm_provider.lower()
        model = settings.llm_model
        temperature = settings.llm_temperature

        logger.info(
            "Creating LLM | provider=%s | model=%s",
            provider,
            model,
        )

        if provider == "gemini":
            if not settings.google_api_key:
                raise ValueError("google_api_key is not configured for Gemini provider.")
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=settings.google_api_key,
            )
        elif provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("groq_api_key is not configured for Groq provider.")
            llm = ChatGroq(
                model=model,
                temperature=temperature,
                groq_api_key=settings.groq_api_key,
            )
        elif provider == "mistral":
            if not settings.mistral_api_key:
                raise ValueError("mistral_api_key is not configured for Mistral provider.")
            llm = ChatMistralAI(
                model=model,
                temperature=temperature,
                mistral_api_key=settings.mistral_api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info("LLM factory successfully created instance for model: %s", model)
        return llm


# Singleton pattern
_DEFAULT_LLM: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """
    Returns the default LLM instance, performing lazy initialization
    and reusing the singleton instance on subsequent calls.
    """
    global _DEFAULT_LLM
    if _DEFAULT_LLM is None:
        _DEFAULT_LLM = LLMFactory.create_llm()
    return _DEFAULT_LLM
