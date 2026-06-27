# backend/llm/generator.py
from dataclasses import dataclass
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.config.settings import settings
from backend.llm.model import get_llm
from backend.llm.prompts import ANSWER_PROMPT
from backend.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerGenerator",
    "get_generator",
    "ContextBuildResult",
    "GenerationResult",
    "ContextDiagnostics",
    "PromptInput",
]

_CONTEXT_SEPARATOR = "\n\n---\n\n"

# Headings that delimit sections in the LLM's structured output.
# The answer extraction stops at the first of these headings found
# after the ### Answer heading.
_SECTION_HEADING_PATTERN = re.compile(
    r"^###\s+(?:Supporting Evidence|Sources|Reason|Available Evidence)",
    re.MULTILINE,
)
_ANSWER_HEADING_PATTERN = re.compile(
    r"^###\s+Answer\s*$",
    re.MULTILINE,
)


@dataclass(slots=True, frozen=True)
class ContextBuildResult:
    context: str
    source_count: int


@dataclass(slots=True, frozen=True)
class GenerationResult:
    answer: str
    context_used: str
    source_count: int
    context_characters: int = 0
    generation_time_ms: float = 0.0
    model_name: str = ""


@dataclass(slots=True, frozen=True)
class ContextDiagnostics:
    original_chunks: int
    duplicates_removed: int
    merged_chunks: int
    table_chunks: int
    final_chunks: int
    context_characters: int
    context_build_time: float


@dataclass(slots=True, frozen=True)
class PromptInput:
    query: str
    context: str
    source_count: int
    context_length: int


class ContextBuilder:
    def __init__(self) -> None:
        pass

    def build_context(self, results: list[RetrievalResult]) -> ContextBuildResult:
        logger.info("Context Build Started | input_chunks=%d", len(results))
        start_time = time.perf_counter()
        
        # 1. Validation & Duplicate Removal
        original_count = len(results)
        unique_results = []
        seen_ids = set()
        seen_contents = set()
        duplicates_removed = 0
        
        for r in results:
            if r.content is None or not r.content.strip():
                continue
            if r.chunk_id in seen_ids:
                duplicates_removed += 1
                continue
            content_norm = " ".join(r.content.lower().split())
            if content_norm in seen_contents:
                duplicates_removed += 1
                continue
            seen_ids.add(r.chunk_id)
            seen_contents.add(content_norm)
            unique_results.append(r)
            
        logger.info(
            "Duplicates Removed | original=%d | duplicates_removed=%d | remaining=%d",
            original_count,
            duplicates_removed,
            len(unique_results),
        )
        
        # 2. Merge Adjacent Chunks
        merged_results = self._merge_adjacent(unique_results)
        merged_count = len(unique_results) - len(merged_results)
        logger.info("Chunks Merged | count=%d", merged_count)
        
        # 3. Evidence Prioritization for Budgeting
        # Keep track of original index to preserve rank order later
        indexed_chunks = [(idx, chunk) for idx, chunk in enumerate(merged_results)]
        
        # Sort prioritized chunks for budget selection
        prioritized = sorted(indexed_chunks, key=lambda x: self._get_priority(x[1]))
        
        selected_indexed = []
        current_chars = 0
        table_count = 0
        source_count = 0
        
        max_chars = settings.max_context_chars
        max_sources = settings.max_context_sources
        max_tables = settings.max_context_tables
        
        for idx, chunk in prioritized:
            is_table = (chunk.metadata.get("content_type") == "table") or ("|---" in (chunk.content or ""))
            
            if is_table and table_count >= max_tables:
                continue
            if source_count >= max_sources:
                break
                
            chunk_len = len(chunk.content or "")
            # Stop before exceeding max_chars
            if current_chars + chunk_len > max_chars:
                continue
                
            selected_indexed.append((idx, chunk))
            current_chars += chunk_len
            source_count += 1
            if is_table:
                table_count += 1
                
        # Re-sort to preserve original retrieval rank order
        selected_indexed.sort(key=lambda x: x[0])
        final_chunks = [x[1] for x in selected_indexed]
        
        # Format context
        context_parts = [c.content for c in final_chunks]
        context = _CONTEXT_SEPARATOR.join(context_parts)
        
        build_time = (time.perf_counter() - start_time) * 1000.0
        
        # 4. Context Diagnostics Logging
        diagnostics = ContextDiagnostics(
            original_chunks=original_count,
            duplicates_removed=duplicates_removed,
            merged_chunks=merged_count,
            table_chunks=table_count,
            final_chunks=len(final_chunks),
            context_characters=len(context),
            context_build_time=build_time,
        )
        logger.info("Context build completed: %s", diagnostics)
        
        return ContextBuildResult(
            context=context,
            source_count=source_count,
        )
        
    def _merge_adjacent(self, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        if not chunks:
            return []
            
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            meta_curr = current.metadata
            meta_next = next_chunk.metadata
            
            same_doc = meta_curr.get("document_name") == meta_next.get("document_name")
            same_page = meta_curr.get("page_number") == meta_next.get("page_number")
            same_type = meta_curr.get("content_type") == meta_next.get("content_type")
            
            curr_idx = meta_curr.get("chunk_index")
            next_idx = meta_next.get("chunk_index")
            adjacent = (curr_idx is not None) and (next_idx is not None) and (abs(curr_idx - next_idx) == 1)
            
            merged_len = len(current.content or "") + len(next_chunk.content or "") + 2
            under_limit = merged_len <= settings.max_merged_chunk_size
            
            if same_doc and same_page and same_type and adjacent and under_limit:
                merged_content = (current.content or "") + "\n\n" + (next_chunk.content or "")
                current = RetrievalResult(
                    chunk_id=current.chunk_id,
                    score=current.score,
                    content=merged_content,
                    metadata=current.metadata
                )
            else:
                merged.append(current)
                current = next_chunk
                
        merged.append(current)
        return merged
        
    def _get_priority(self, r: RetrievalResult) -> int:
        content_lower = (r.content or "").lower()
        is_table = (r.metadata.get("content_type") == "table") or ("|---" in content_lower)
        if is_table:
            return 1
            
        is_fin = any(w in content_lower for w in ["pat", "revenue", "eps", "ebitda", "income", "profit", "loss", "borrowings", "debt", "assets", "liabilities"])
        if is_fin:
            return 2
            
        is_num = any(c.isdigit() for c in content_lower) or "%" in content_lower or "₹" in content_lower or "rs" in content_lower
        if is_num:
            return 3
            
        is_text = r.metadata.get("content_type") == "text"
        if is_text:
            return 4
            
        is_note = "note" in content_lower or "note" in str(r.metadata.get("section_title", "")).lower()
        if is_note:
            return 5
            
        return 6


class AnswerGenerator:
    """
    Generates grounded answers from retrieved context.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm or get_llm()
        self._context_builder = ContextBuilder()

    def build_context(self, results: list[RetrievalResult]) -> ContextBuildResult:
        """
        Build context string from retrieval results while respecting
        the configured context size limit.
        """
        if not results:
            raise ValueError("Retrieval results cannot be empty.")
        return self._context_builder.build_context(results)

    def _generate_answer(
        self,
        query: str,
        context: str,
        source_count: int,
    ) -> GenerationResult:
        """
        Generates an answer from an already constructed context string.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty or whitespace-only.")

        start_time = time.perf_counter()
        
        prompt_input = PromptInput(
            query=query,
            context=context,
            source_count=source_count,
            context_length=len(context),
        )
        
        answer = self._invoke_llm(prompt_input)
        processed_answer = self._post_process_answer(answer)
        
        generation_time = (time.perf_counter() - start_time) * 1000.0
        
        logger.info(
            "Generation Completed | latency=%.2fms | answer_chars=%d",
            generation_time,
            len(processed_answer),
        )
        
        return GenerationResult(
            answer=processed_answer,
            context_used=context,
            source_count=source_count,
            context_characters=len(context),
            generation_time_ms=generation_time,
            model_name=settings.llm_model,
        )

    def generate(self, query: str, results: list[RetrievalResult]) -> GenerationResult:
        """
        Generate an answer from retrieved context.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        if not results:
            raise ValueError("Retrieval results cannot be empty.")

        logger.info(
            "Generation Started | model=%s | chunks=%d",
            settings.llm_model,
            len(results),
        )

        context_result = self.build_context(results)

        return self._generate_answer(
            query=query,
            context=context_result.context,
            source_count=context_result.source_count,
        )

    def generate_from_context(
        self,
        query: str,
        context: str,
    ) -> GenerationResult:
        """
        Generates an answer from an already prepared context string.
        """
        if not query or not query.strip():
           raise ValueError("Query cannot be empty or whitespace-only.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty or whitespace-only.")

        # Ensure context does not exceed max_context_chars setting
        max_chars = settings.max_context_chars
        if len(context) > max_chars:
            logger.warning(
                "Truncating incoming context size from %d to max limit of %d characters",
                len(context),
                max_chars,
            )
            context = context[:max_chars]

        return self._generate_answer(
            query=query,
            context=context,
            source_count=0,
        )

    def _invoke_llm(self, prompt_input: PromptInput) -> str:
        logger.info(
            "LLM Invoked | model=%s | context_chars=%d | sources=%d",
            settings.llm_model,
            prompt_input.context_length,
            prompt_input.source_count,
        )
        
        try:
            messages = ANSWER_PROMPT.format_messages(
                context=prompt_input.context,
                query=prompt_input.query,
            )
            response = self._llm.invoke(messages)
        except Exception as exc:
            logger.exception("LLM invocation failed | error=%s", exc)
            raise RuntimeError("LLM invocation failed.") from exc

        answer = getattr(response, "content", response)

        if isinstance(answer, list):
            answer = "".join(
                (
                    item.get("text", "")
                    if isinstance(item, dict)
                    else str(item)
                )
                for item in answer
            )

        answer_str = str(answer).strip()
        if not answer_str:
            raise RuntimeError("LLM returned an empty response.")

        return answer_str

    @staticmethod
    def _extract_answer_section(raw: str) -> str:
        """
        Extracts only the content beneath the '### Answer' heading from the
        LLM's structured output, discarding Supporting Evidence, Sources, and
        any other trailing sections produced by the prompt template.

        If the heading is absent the full text is returned unchanged so that
        responses that do not follow the template are never silently discarded.
        """
        match = _ANSWER_HEADING_PATTERN.search(raw)
        if not match:
            # No structured heading found — return the raw text as-is.
            return raw

        # Slice from just after the '### Answer' heading.
        after_heading = raw[match.end():]

        # Find the start of the next section heading (if any).
        next_section = _SECTION_HEADING_PATTERN.search(after_heading)
        if next_section:
            after_heading = after_heading[: next_section.start()]

        return after_heading.strip()

    def _post_process_answer(self, answer: str) -> str:
        if not answer:
            return ""
        # Extract only the Answer section, dropping Supporting Evidence / Sources.
        answer = self._extract_answer_section(answer)
        # Trim each line
        lines = [line.strip() for line in answer.splitlines()]
        answer = "\n".join(lines)
        # Collapse 3+ newlines to 2 newlines
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        # Trim overall leading/trailing whitespace
        answer = answer.strip()
        return answer


_DEFAULT_GENERATOR: AnswerGenerator | None = None


def get_generator() -> AnswerGenerator:
    global _DEFAULT_GENERATOR
    if _DEFAULT_GENERATOR is None:
        _DEFAULT_GENERATOR = AnswerGenerator()
    return _DEFAULT_GENERATOR