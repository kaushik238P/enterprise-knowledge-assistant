# backend/services/citation_service.py
import logging
import re
from typing import Any
from backend.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

__all__ = [
    "CitationService",
    "get_citation_service",
]


class CitationService:
    """
    Service responsible for filtering, deduplicating, limiting, and formatting citations.
    """

    def filter_and_format_citations(
        self,
        results: list[RetrievalResult],
        answer: str,
        retrieval_sufficiency: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Filters, deduplicates, and limits retrieval results to produce formatted citations.
        """
        if not retrieval_sufficiency or not results:
            return []

        logger.info("Filtering and formatting citations | input_chunks=%d", len(results))

        # 1. Deduplication & Same-Page Merging
        # Group strictly by document and page number to merge multiple chunks originating from the same page.
        # Keep the one with the highest score.
        seen = {}
        for r in results:
            doc_id = (
                r.metadata.get("document_id")
                or r.metadata.get("document_name")
                or r.metadata.get("source_file")
                or "Unknown"
            )
            page_num = r.metadata.get("page_number")

            # Normalize values to ensure accurate deduplication by page
            key = (str(doc_id).strip(), page_num)

            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        # 2. Ranking: Preserve the existing Cross-Encoder (or retrieval) score ordering
        unique_results = list(seen.values())
        unique_results.sort(key=lambda r: r.score, reverse=True)

        # 3. Limit citation count based on answer structure complexity
        limit = self.get_max_sources_limit(answer)
        final_chunks = unique_results[:limit]

        # 4. Format citation metadata
        formatted_citations = []
        for r in final_chunks:
            doc_name = (
                r.metadata.get("document_name")
                or r.metadata.get("source_file")
                or "Unknown"
            )
            page_num = r.metadata.get("page_number")

            try:
                page_val = int(page_num or 1)
                if page_val < 1:
                    page_val = 1
            except (ValueError, TypeError):
                page_val = 1

            formatted_citations.append({
                "document": doc_name,
                "page": page_val,
                "section": r.metadata.get("section_title"),
                "section_path": r.metadata.get("section_path"),
                "score": float(r.score),
                "chunk_id": str(r.chunk_id) if r.chunk_id else None,
            })

        logger.info(
            "Citations filtered and formatted | remaining=%d | limit=%d | original=%d",
            len(formatted_citations),
            limit,
            len(results),
        )
        return formatted_citations

    def get_max_sources_limit(self, answer: str) -> int:
        """
        Determines the maximum number of citations based on the answer structure.
        - 1 citation: Single fact / definition (1 sentence / 0-1 list items).
        - 2 citations: Multiple facts or comparison (2-3 sentences / 2 list items).
        - 3 citations: Multi-section explanation or long technical answer (>= 4 sentences / >= 3 list items).
        """
        if not answer or not answer.strip():
            return 1

        # Count bullet points / numbered list items
        bullet_patterns = [r"^\s*[-*+•]\s+", r"^\s*\d+\.\s+"]
        bullets_count = 0
        for line in answer.splitlines():
            if any(re.match(pat, line) for pat in bullet_patterns):
                bullets_count += 1

        # Count sentences by splitting on punctuation followed by space or end of string
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        sentence_count = len(sentences)

        if bullets_count >= 3 or sentence_count >= 4:
            return 3
        elif bullets_count >= 2 or sentence_count >= 2:
            return 2
        else:
            return 1


_DEFAULT_CITATION_SERVICE: CitationService | None = None


def get_citation_service() -> CitationService:
    global _DEFAULT_CITATION_SERVICE
    if _DEFAULT_CITATION_SERVICE is None:
        _DEFAULT_CITATION_SERVICE = CitationService()
    return _DEFAULT_CITATION_SERVICE
