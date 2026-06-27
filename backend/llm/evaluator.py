# backend/llm/evaluator.py
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.llm.model import get_llm
from backend.llm.evaluation import (
    EvaluationInput,
    EvaluationResult,
    EvaluationManager,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationResult",
    "AnswerEvaluator",
    "get_evaluator",
]


class AnswerEvaluator:
    """
    Evaluates generated answers against the query and retrieved context.
    Delegates to EvaluationManager for modular evaluation.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
    ) -> None:
        self._llm = llm or get_llm()
        self._manager = EvaluationManager(self._llm)

    def evaluate(
        self,
        query: str,
        context: str,
        answer: str,
        retrieval_results: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        if not context or not context.strip():
            raise ValueError("context cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("answer cannot be empty.")

        input_data = EvaluationInput(
            query=query,
            retrieved_context=context,
            answer=answer,
            retrieval_results=retrieval_results or [],
            metadata=metadata or {},
        )
        return self._manager.evaluate(input_data)


_DEFAULT_EVALUATOR: AnswerEvaluator | None = None


def get_evaluator() -> AnswerEvaluator:
    global _DEFAULT_EVALUATOR

    if _DEFAULT_EVALUATOR is None:
        _DEFAULT_EVALUATOR = AnswerEvaluator()

    return _DEFAULT_EVALUATOR
