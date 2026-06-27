from dataclasses import dataclass
import logging
import re
from typing import Any
from uuid import UUID

from qdrant_client.http import models as rest

from backend.config.settings import settings
from backend.embeddings.hybride_embedder import HybridEmbedder, HybridQueryEmbedding
from backend.vectorstore.qdrant import QdrantVectorStore, SearchResult, get_vector_store

logger = logging.getLogger(__name__)

__all__ = [
    "HybridSearchResult",
    "HybridRetriever",
    "get_hybrid_retriever",
]


@dataclass(slots=True, frozen=True)
class HybridSearchResult:
    chunk_id: UUID
    score: float
    content: str
    metadata: dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        hybrid_embedder: HybridEmbedder | None = None,
    ) -> None:
        self._vector_store = vector_store or get_vector_store()
        self._hybrid_embedder = hybrid_embedder or HybridEmbedder()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[HybridSearchResult]:
        if query is None or not query.strip():
            raise ValueError("query cannot be empty or whitespace-only.")

        resolved_top_k = top_k if top_k is not None else settings.hybrid_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # 1. Financial Query Expansion
        expanded_query = self._expand_query(query)
        logger.info(
            "Hybrid retrieval started | original_query='%s' | expanded_query='%s' | top_k=%d",
            query,
            expanded_query,
            resolved_top_k,
        )

        # 2. Dense & Sparse Query Embeddings
        query_embedding: HybridQueryEmbedding = self._hybrid_embedder.embed_query_hybrid(expanded_query)
        logger.info("Hybrid query embedding generated")

        # 3. Dense & Sparse Search Candidate Retrieval
        dense_results = []
        sparse_results = []

        try:
            dense_response = self._vector_store._client.query_points(
                collection_name=self._vector_store._collection_name,
                query=query_embedding.dense_vector,
                using="dense",
                limit=resolved_top_k,
                with_payload=True,
            )
            dense_results = dense_response.points
        except Exception as exc:
            logger.warning("Dense vector search failed: %s", exc)

        try:
            sparse_response = self._vector_store._client.query_points(
                collection_name=self._vector_store._collection_name,
                query=rest.SparseVector(
                    indices=query_embedding.sparse_vector.indices,
                    values=query_embedding.sparse_vector.values,
                ),
                using="sparse",
                limit=resolved_top_k,
                with_payload=True,
            )
            sparse_results = sparse_response.points
        except Exception as exc:
            logger.warning("Sparse vector search failed: %s", exc)

        logger.info(
            "Candidate retrieval complete | Dense Result Count: %d | Sparse Result Count: %d",
            len(dense_results),
            len(sparse_results),
        )

        # 4. Weighted Reciprocal Rank Fusion (RRF)
        fused: dict[UUID, dict[str, Any]] = {}
        dense_weight = settings.dense_rrf_weight
        sparse_weight = settings.sparse_rrf_weight

        for rank, point in enumerate(dense_results, start=1):
            point_id = UUID(str(point.id))
            payload = point.payload or {}
            score = dense_weight / (60.0 + rank)
            fused[point_id] = {
                "chunk_id": point_id,
                "score": score,
                "content": str(payload.get("content", "")),
                "metadata": dict(payload),
            }

        for rank, point in enumerate(sparse_results, start=1):
            point_id = UUID(str(point.id))
            payload = point.payload or {}
            score = sparse_weight / (60.0 + rank)
            if point_id in fused:
                fused[point_id]["score"] += score
            else:
                fused[point_id] = {
                    "chunk_id": point_id,
                    "score": score,
                    "content": str(payload.get("content", "")),
                    "metadata": dict(payload),
                }

        # 5. Metadata-aware Score Boost
        boosts_applied = 0
        bonus = settings.metadata_bonus
        query_lower = query.lower()

        # Rules mapping keywords in query to preferred categories/titles/headings
        meta_rules = [
            {
                "keywords": [r"\brevenue\b", r"\bpat\b", r"\beps\b", r"\bincome\b", r"\bprofit\b", r"\bloss\b"],
                "categories": ["profit & loss", "income statement", "statement of financial results", "statement of profit"]
            },
            {
                "keywords": [r"\bassets\b", r"\bborrowings\b", r"\bdebt\b", r"\bliabilities\b", r"\bequity\b"],
                "categories": ["balance sheet"]
            },
            {
                "keywords": [r"\boperating\s+cash\s+flow\b", r"\bcash\s+generated\b", r"\bcash\s+flow\b"],
                "categories": ["cash flow", "cash flow statement"]
            }
        ]

        active_categories = []
        for rule in meta_rules:
            if any(re.search(kw, query_lower) for kw in rule["keywords"]):
                active_categories.extend(rule["categories"])

        if active_categories and bonus > 0:
            for item in fused.values():
                metadata = item["metadata"]
                table_category = str(metadata.get("table_category", "")).lower()
                section_title = str(metadata.get("section_title", "")).lower()
                section_path = str(metadata.get("section_path", "")).lower()

                boost = False
                for cat in active_categories:
                    if (cat in table_category) or (cat in section_title) or (cat in section_path):
                        boost = True
                        break

                if boost:
                    item["score"] += bonus
                    boosts_applied += 1

        logger.info("Metadata Boost Applied: %d chunks boosted", boosts_applied)

        # 6. Sorting Candidates
        sorted_candidates = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        # 7. Duplicate Removal
        unique_results = []
        seen_chunk_ids = set()
        seen_contents = []
        duplicates_removed = 0
        threshold = settings.duplicate_similarity_threshold

        for item in sorted_candidates:
            chunk_id = item["chunk_id"]
            content = item["content"]

            if chunk_id in seen_chunk_ids:
                duplicates_removed += 1
                continue

            content_clean = " ".join(content.lower().split())
            is_dup = False
            for seen in seen_contents:
                if content_clean == seen:
                    is_dup = True
                    break
                
                # Jaccard overlap similarity
                w1 = set(content_clean.split())
                w2 = set(seen.split())
                if w1 and w2:
                    intersect = w1.intersection(w2)
                    union = w1.union(w2)
                    similarity = len(intersect) / len(union)
                    if similarity >= threshold:
                        is_dup = True
                        break

            if is_dup:
                duplicates_removed += 1
                continue

            seen_chunk_ids.add(chunk_id)
            seen_contents.append(content_clean)
            unique_results.append(item)

        logger.info("Duplicate Chunks Removed: %d duplicates discarded", duplicates_removed)

        final_results = [
            HybridSearchResult(
                chunk_id=item["chunk_id"],
                score=item["score"],
                content=item["content"],
                metadata=item["metadata"],
            )
            for item in unique_results[:resolved_top_k]
        ]

        logger.info(
            "Final Hybrid Results: retrieved=%d | returned=%d | best_score=%.4f",
            len(fused),
            len(final_results),
            final_results[0].score if final_results else 0.0,
        )

        return final_results

    def _expand_query(self, query: str) -> str:
        if not settings.query_expansion_enabled:
            return query

        expanded_parts = []
        query_lower = query.lower()

        # Domain expansions
        domain_mappings = {
            r"\bpat\b": "PAT Profit After Tax Net Profit Profit for the Period",
            r"\brevenue\b": "Revenue Revenue from Operations Operating Revenue Sales",
            r"\beps\b": "EPS Earnings Per Share Basic EPS Diluted EPS",
            r"\boci\b": "Other Comprehensive Income OCI",
            r"\bborrowings\b": "Debt Loans Borrowings Financial Liabilities",
            r"\bcash\s+flow\b": "Operating Cash Flow Investing Cash Flow Financing Cash Flow",
        }

        for pattern, expansion in domain_mappings.items():
            if re.search(pattern, query_lower):
                expanded_parts.append(expansion)

        # Company aliases
        company_mappings = {
            r"\bael\b": "Adani Enterprises Limited",
        }
        for pattern, expansion in company_mappings.items():
            if re.search(pattern, query_lower):
                expanded_parts.append(expansion)

        # Fiscal periods expansions
        fiscal_match = re.search(r"\b(Q[1-4]|H[1-2])\s+FY(\d{2})\b", query, re.IGNORECASE)
        if fiscal_match:
            period_type = fiscal_match.group(1).upper()
            yy = int(fiscal_match.group(2))
            
            cal_year_prev = 2000 + yy - 1
            cal_year_curr = 2000 + yy

            if period_type == "Q1":
                expanded_parts.append(f"Quarter Ended 30 June {cal_year_prev}")
            elif period_type == "Q2":
                expanded_parts.append(f"Quarter Ended 30 September {cal_year_prev}")
            elif period_type == "Q3":
                expanded_parts.append(f"Quarter Ended 31 December {cal_year_prev}")
            elif period_type == "Q4":
                expanded_parts.append(f"Quarter Ended 31 March {cal_year_curr}")
            elif period_type == "H1":
                expanded_parts.append(f"Half Year Ended 30 September {cal_year_prev}")
            elif period_type == "H2":
                expanded_parts.append(f"Half Year Ended 31 March {cal_year_curr}")

        elif re.search(r"\bFY\s*(\d{2})\b", query, re.IGNORECASE):
            fy_match = re.search(r"\bFY\s*(\d{2})\b", query, re.IGNORECASE)
            yy = int(fy_match.group(1))
            cal_year_curr = 2000 + yy
            expanded_parts.append(f"Year Ended 31 March {cal_year_curr}")

        if expanded_parts:
            expanded_query = query + " " + " ".join(expanded_parts)
            return " ".join(expanded_query.split())

        return query



_DEFAULT_HYBRID_RETRIEVER: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _DEFAULT_HYBRID_RETRIEVER
    if _DEFAULT_HYBRID_RETRIEVER is None:
        _DEFAULT_HYBRID_RETRIEVER = HybridRetriever()
    return _DEFAULT_HYBRID_RETRIEVER