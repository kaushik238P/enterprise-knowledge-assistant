from dataclasses import dataclass
import logging
import re
import time
from typing import Any
from uuid import UUID

from backend.config.settings import settings
from backend.retrieval.hybrid import (
    HybridRetriever,
    HybridSearchResult,
    get_hybrid_retriever,
)
from backend.retrieval.reranker import (
    CrossEncoderReranker,
    RerankResult,
    get_reranker,
)

__all__ = [
    "RetrievalResult",
    "Retriever",
    "get_retriever",
    "QueryAnalysis",
    "RetrievalDiagnostics",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class QueryAnalysis:
    category: str
    quarters: list[str]
    years: list[str]
    currencies: list[str]
    percentages: list[str]
    companies: list[str]
    reports: list[str]


@dataclass(slots=True, frozen=True)
class RetrievalDiagnostics:
    query_type: str
    hybrid_candidates: int
    validated_candidates: int
    reranked_candidates: int
    duplicates_removed: int
    average_score: float
    retrieval_latency_ms: float
    timings: dict[str, float]


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    chunk_id: UUID
    score: float
    content: str
    metadata: dict[str, Any]


class Retriever:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever or get_hybrid_retriever()
        self._reranker = reranker or get_reranker()
        self.last_diagnostics: RetrievalDiagnostics | None = None
        self.last_sufficiency: bool = True
        self.last_results: list = []

        if self._hybrid_retriever is None:
            raise RuntimeError("HybridRetriever initialization failed.")

        if self._reranker is None:
            raise RuntimeError("Reranker initialization failed.")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if query is None or not query.strip():
            raise ValueError("query cannot be empty or whitespace-only.")

        start_total = time.perf_counter()
        timings = {}

        # 1. Query Analysis
        start_stage = time.perf_counter()
        analysis = self._analyze_query(query)
        timings["query_analysis"] = (time.perf_counter() - start_stage) * 1000.0
        logger.debug("Query analysis completed: %s", analysis)

        # 2. Adaptive Retrieval Configuration
        resolved_hybrid_k, resolved_pool_size, resolved_final_k = self._get_adaptive_config(
            analysis.category, top_k
        )
        logger.info(
            "Adaptive retrieval configuration | category=%s | hybrid_k=%d | pool_size=%d | final_k=%d",
            analysis.category,
            resolved_hybrid_k,
            resolved_pool_size,
            resolved_final_k,
        )

        # 3. Hybrid Retrieval
        start_stage = time.perf_counter()
        hybrid_results: list[HybridSearchResult] = self._hybrid_retriever.retrieve(
            query=query,
            top_k=resolved_hybrid_k,
        )
        timings["hybrid_retrieval"] = (time.perf_counter() - start_stage) * 1000.0

        if not hybrid_results:
            logger.warning("Hybrid retrieval returned zero results.")
            self._save_diagnostics(
                category=analysis.category,
                hybrid_count=0,
                valid_count=0,
                reranked_count=0,
                dups_removed=0,
                avg_score=0.0,
                latency_ms=(time.perf_counter() - start_total) * 1000.0,
                timings=timings,
            )
            return []

        # 4. Candidate Quality Validation
        start_stage = time.perf_counter()
        validated_candidates, dups_removed = self._validate_candidates(hybrid_results)
        timings["validation"] = (time.perf_counter() - start_stage) * 1000.0

        # Fallback: if validation removes every candidate, return the original hybrid candidates
        if not validated_candidates:
            logger.warning("Validation removed all candidates. Falling back to original hybrid candidates.")
            validated_candidates = hybrid_results
            dups_removed = 0

        # Slice to candidate pool size for reranking
        candidates_to_rerank = validated_candidates[:resolved_pool_size]

        # 5. Cross Encoder Reranking
        start_stage = time.perf_counter()
        reranked_results = []
        reranking_failed = False

        try:
            reranked_results = self._reranker.rerank(
                query=query,
                results=candidates_to_rerank,
                top_k=resolved_final_k,
            )
        except Exception as exc:
            logger.exception("CrossEncoder reranking failed, falling back to hybrid results: %s", exc)
            reranking_failed = True

        timings["cross_encoder"] = (time.perf_counter() - start_stage) * 1000.0

        # Fallback: if Cross Encoder fails, use validated candidates
        if reranking_failed or not reranked_results:
            reranked_results = [
                RerankResult(
                    chunk_id=result.chunk_id,
                    score=result.score,
                    content=result.content,
                    metadata=result.metadata,
                )
                for result in candidates_to_rerank[:resolved_final_k]
            ]

        # Check retrieval sufficiency
        self.last_sufficiency = True
        if reranked_results:
            top_score = reranked_results[0].score
            if top_score < settings.retrieval_min_score:
                logger.warning(
                    "Retrieval sufficiency check failed | top_score=%.4f < min_score=%.4f",
                    top_score,
                    settings.retrieval_min_score,
                )
                self.last_sufficiency = False
        else:
            logger.warning("Retrieval sufficiency check failed | no reranked results returned.")
            self.last_sufficiency = False

        # 6. Context Optimization
        start_stage = time.perf_counter()
        final_results = [
            RetrievalResult(
                chunk_id=result.chunk_id,
                score=result.score,
                content=result.content,
                metadata=result.metadata,
            )
            for result in reranked_results
        ]

        if settings.enable_context_optimization:
            # Sort results by document name and page number to maintain reading continuity
            final_results.sort(
                key=lambda r: (
                    str(r.metadata.get("document_name") or ""),
                    int(r.metadata.get("page_number") or 0),
                )
            )

        timings["context_optimization"] = (time.perf_counter() - start_stage) * 1000.0

        total_latency_ms = (time.perf_counter() - start_total) * 1000.0
        timings["total"] = total_latency_ms

        # 7. Diagnostics Collection
        avg_score = sum(r.score for r in final_results) / len(final_results) if final_results else 0.0
        self._save_diagnostics(
            category=analysis.category,
            hybrid_count=len(hybrid_results),
            valid_count=len(validated_candidates),
            reranked_count=len(reranked_results),
            dups_removed=dups_removed,
            avg_score=avg_score,
            latency_ms=total_latency_ms,
            timings=timings,
        )

        context_char_size = sum(len(r.content) for r in final_results)

        # 8. Structured Logging
        logger.info(
            "Retrieval completed | QueryType=%s | HybridCount=%d | ValidatedCount=%d | RerankedCount=%d | ContextCharSize=%d | Latency=%.2fms",
            analysis.category,
            len(hybrid_results),
            len(validated_candidates),
            len(final_results),
            context_char_size,
            total_latency_ms,
        )
        logger.info(
            "Retrieval performance breakdown (ms) | analysis=%.2f | hybrid=%.2f | validation=%.2f | cross_encoder=%.2f | context_opt=%.2f",
            timings["query_analysis"],
            timings["hybrid_retrieval"],
            timings["validation"],
            timings["cross_encoder"],
            timings["context_optimization"],
        )
        self.last_results = final_results
        return final_results

    def _analyze_query(self, query: str) -> QueryAnalysis:
        query_lower = query.lower()
        
        # Heuristics classification
        category = "general"
        if any(w in query_lower for w in ["compare", "versus", "vs", "difference", "comparison", "both", "relative to"]):
            category = "comparison"
        elif any(w in query_lower for w in ["summarize", "summary", "overview", "outline", "brief"]):
            category = "summarization"
        elif any(w in query_lower for w in ["what is", "define", "definition", "meaning of", "explain"]):
            category = "definition"
        elif any(w in query_lower for w in ["table", "statement", "schedule", "results", "sheet"]):
            category = "table_lookup"
        elif any(w in query_lower for w in ["section", "notes", "note to", "disclosure", "annexure"]):
            category = "section_specific"
        elif any(re.search(r"\b(q[1-4]|h[1-2]|fy\s*\d{2,4}|20\d{2})\b", query_lower) for query_lower in [query_lower]):
            category = "time_period"
        elif any(w in query_lower for w in ["pat", "revenue", "eps", "ebitda", "income", "profit", "loss", "borrowings", "debt", "assets", "liabilities"]):
            category = "financial_metric"

        # Entity Extraction
        quarters = re.findall(r"\b(Q[1-4]|H[1-2])\b", query, re.IGNORECASE)
        years = re.findall(r"\b(FY\s*\d{2,4}|20\d{2}|19\d{2})\b", query, re.IGNORECASE)
        
        currencies = re.findall(
            r"(\brs\.?\s*\d+[\d,.]*|\b₹\s*\d+[\d,.]*|\d+[\d,.]*\s*(?:crore|lakh|million|billion|percent|%|usd|inr|rupees))\b",
            query,
            re.IGNORECASE,
        )
        for sym in ["₹", "Rs", "rupees", "USD", "INR", "crore", "lakh", "million", "billion"]:
            if sym.lower() in query_lower:
                currencies.append(sym)
        currencies = list(set(currencies))
        
        percentages = re.findall(
            r"(\d+[\d,.]*\s*%\s*|\d+[\d,.]*\s*percent\s*|\d+[\d,.]*\s*percentage\s*)",
            query,
            re.IGNORECASE,
        )
        
        companies = re.findall(r"\b([A-Z]{2,5}|Adani|Reliance|Tata|TCS|Infosys|Wipro|HDFC|ICICI|SBI)\b", query)
        companies = list(set(companies))
        
        reports = re.findall(
            r"\b(Annual Report|Financial Results|Balance Sheet|Cash Flow|P&L|Profit and Loss|Income Statement)\b",
            query,
            re.IGNORECASE,
        )
        
        return QueryAnalysis(
            category=category,
            quarters=list(set(quarters)),
            years=list(set(years)),
            currencies=currencies,
            percentages=list(set(percentages)),
            companies=companies,
            reports=list(set(reports)),
        )

    def _get_adaptive_config(self, category: str, top_k: int | None = None) -> tuple[int, int, int]:
        if not settings.adaptive_retrieval_enabled:
            # Fixed settings fallback
            final_k = top_k if top_k is not None else settings.final_top_k
            return settings.hybrid_top_k, settings.candidate_pool_size, final_k

        if category in ["comparison", "summarization"]:
            hybrid_k = settings.adaptive_complex_hybrid_k
            pool_size = settings.adaptive_complex_pool_size
            final_k = settings.adaptive_complex_final_k
        elif category in ["financial_metric", "table_lookup", "time_period", "section_specific"]:
            hybrid_k = settings.adaptive_medium_hybrid_k
            pool_size = settings.adaptive_medium_pool_size
            final_k = settings.adaptive_medium_final_k
        else:
            hybrid_k = settings.adaptive_simple_hybrid_k
            pool_size = settings.adaptive_simple_pool_size
            final_k = settings.adaptive_simple_final_k

        if top_k is not None:
            # Overwrite final_k if top_k is explicitly requested by caller
            final_k = top_k

        return hybrid_k, pool_size, final_k

    def _validate_candidates(self, results: list[HybridSearchResult]) -> tuple[list[HybridSearchResult], int]:
        valid = []
        seen_ids = set()
        seen_contents = set()
        removed = 0
        min_len = settings.min_chunk_length

        for r in results:
            if r.content is None:
                removed += 1
                continue
                
            clean_content = r.content.strip()
            if not clean_content or len(clean_content) < min_len:
                removed += 1
                continue

            if r.chunk_id in seen_ids:
                removed += 1
                continue

            content_norm = " ".join(clean_content.lower().split())
            if content_norm in seen_contents:
                removed += 1
                continue

            seen_ids.add(r.chunk_id)
            seen_contents.add(content_norm)
            valid.append(r)

        return valid, removed

    def _save_diagnostics(
        self,
        category: str,
        hybrid_count: int,
        valid_count: int,
        reranked_count: int,
        dups_removed: int,
        avg_score: float,
        latency_ms: float,
        timings: dict[str, float],
    ) -> None:
        if settings.enable_retrieval_diagnostics:
            self.last_diagnostics = RetrievalDiagnostics(
                query_type=category,
                hybrid_candidates=hybrid_count,
                validated_candidates=valid_count,
                reranked_candidates=reranked_count,
                duplicates_removed=dups_removed,
                average_score=avg_score,
                retrieval_latency_ms=latency_ms,
                timings=timings,
            )


_DEFAULT_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = Retriever()
    return _DEFAULT_RETRIEVER