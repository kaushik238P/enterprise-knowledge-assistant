from dataclasses import dataclass
import logging
import re
from typing import Any, Protocol
from uuid import UUID

from sentence_transformers import CrossEncoder

from backend.config.settings import settings


logger = logging.getLogger(__name__)

__all__ = [
    "CrossEncoderReranker",
    "get_reranker",
    "RerankResult",
    "RetrievedResultLike"
]


class RetrievedResultLike(Protocol):
    chunk_id: UUID
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class RerankResult:
    chunk_id: UUID
    score: float
    content: str
    metadata: dict[str, Any]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = settings.reranker_model,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model

        try:
            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker model loaded | model=%s", self._model_name)
        except Exception as exc:
            logger.exception("Failed to load reranker model | model=%s | error=%s", self._model_name, exc)
            raise RuntimeError(f"Failed to load CrossEncoder model '{self._model_name}': {exc}") from exc

        return self._model
    
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        results: list[RetrievedResultLike],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if query is None or not query.strip():
            raise ValueError("query cannot be empty or whitespace-only.")
        if not results:
            return []

        resolved_top_k = top_k if top_k is not None else settings.final_top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # 1. Candidate Preparation
        valid_candidates = []
        seen_chunk_ids = set()
        seen_contents = set()
        duplicates_removed = 0

        # Sort incoming by score descending so we keep highest score instance
        sorted_inputs = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)

        for result in sorted_inputs:
            content = result.content
            if content is None or not content.strip():
                continue
            
            chunk_id = result.chunk_id
            if chunk_id in seen_chunk_ids:
                duplicates_removed += 1
                continue
                
            content_clean = " ".join(content.lower().split())
            if content_clean in seen_contents:
                duplicates_removed += 1
                continue

            seen_chunk_ids.add(chunk_id)
            seen_contents.add(content_clean)
            valid_candidates.append(result)

        logger.info(
            "Candidate validation complete | original=%d | valid=%d | duplicates_removed=%d",
            len(results),
            len(valid_candidates),
            duplicates_removed,
        )

        if not valid_candidates:
            return []

        # 2. Score Normalization of Hybrid Scores
        hybrid_scores = [getattr(r, "score", 0.0) for r in valid_candidates]
        strategy = settings.reranker_normalization_strategy
        normalized_hybrid_scores = self._normalize_scores(hybrid_scores, strategy)

        # 3. Cross Encoder Batch Scoring
        try:
            model = self._load_model()
            pairs = [(query, r.content) for r in valid_candidates]
            
            # Predict in batches
            batch_size = settings.reranker_batch_size
            ce_scores = model.predict(pairs, batch_size=batch_size)
            ce_scores = [float(s) for s in ce_scores]
        except Exception as exc:
            logger.exception("CrossEncoder scoring failed, falling back to hybrid scores: %s", exc)
            ce_scores = [0.0] * len(valid_candidates)

        # Normalize Cross Encoder scores
        normalized_ce_scores = self._normalize_scores(ce_scores, strategy)

        # 4. Metadata-aware Scoring
        query_lower = query.lower()
        bonus = settings.metadata_bonus
        
        # Meta rules mapping keywords to categories
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
            },
            {
                "keywords": [r"\beps\b", r"\bearnings\s+per\s+share\b"],
                "categories": ["earnings per share"]
            }
        ]

        active_categories = []
        for rule in meta_rules:
            if any(re.search(kw, query_lower) for kw in rule["keywords"]):
                active_categories.extend(rule["categories"])

        # 5. Combined Score and Stable Ranking
        ce_weight = settings.cross_encoder_weight
        h_weight = settings.hybrid_weight
        m_weight = settings.metadata_weight

        candidate_items = []
        for idx, result in enumerate(valid_candidates):
            norm_hybrid = normalized_hybrid_scores[idx]
            norm_ce = normalized_ce_scores[idx]
            
            # Check metadata bonus
            metadata = result.metadata
            table_category = str(metadata.get("table_category", "")).lower()
            section_title = str(metadata.get("section_title", "")).lower()
            section_path = str(metadata.get("section_path", "")).lower()

            meta_boost = 0.0
            if active_categories:
                for cat in active_categories:
                    if (cat in table_category) or (cat in section_title) or (cat in section_path):
                        meta_boost = 1.0
                        break

            combined_score = (ce_weight * norm_ce) + (h_weight * norm_hybrid) + (m_weight * meta_boost * bonus)
            
            candidate_items.append({
                "result": result,
                "combined_score": combined_score,
                "norm_ce": norm_ce,
                "norm_hybrid": norm_hybrid,
                "meta_boost": meta_boost * bonus
            })

        # 6. Diversity Selection (Greedy MMR-like)
        selected_pages = set()
        selected_sections = set()
        selected_categories = set()
        final_selection = []
        
        div_weight = settings.diversity_weight

        while len(final_selection) < resolved_top_k and candidate_items:
            best_item = None
            best_score = -float("inf")
            best_idx = -1
            best_penalty = 0.0

            for idx, item in enumerate(candidate_items):
                result = item["result"]
                metadata = result.metadata
                page = metadata.get("page_number")
                section = metadata.get("section_title")
                category = metadata.get("table_category")

                penalty = 0.0
                if page is not None and page in selected_pages:
                    penalty += 0.5
                if section and section in selected_sections:
                    penalty += 0.3
                if category and category in selected_categories:
                    penalty += 0.2

                adjusted_score = item["combined_score"] - div_weight * penalty
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_item = item
                    best_idx = idx
                    best_penalty = penalty

            if best_item:
                result = best_item["result"]
                metadata = result.metadata
                page = metadata.get("page_number")
                section = metadata.get("section_title")
                category = metadata.get("table_category")

                if page is not None:
                    selected_pages.add(page)
                if section:
                    selected_sections.add(section)
                if category:
                    selected_categories.add(category)

                final_selection.append(
                    RerankResult(
                        chunk_id=result.chunk_id,
                        score=best_score,
                        content=result.content,
                        metadata=result.metadata
                    )
                )
                candidate_items.pop(best_idx)
            else:
                break

        logger.info(
            "Reranking complete | returned=%d | best_score=%.4f",
            len(final_selection),
            final_selection[0].score if final_selection else 0.0
        )

        return final_selection

    def _normalize_scores(self, scores: list[float], strategy: str) -> list[float]:
        if not scores:
            return []
        if strategy == "min-max":
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return [1.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        elif strategy == "z-score":
            mean = sum(scores) / len(scores)
            variance = sum((s - mean)**2 for s in scores) / len(scores)
            std = variance**0.5
            if std == 0:
                return [0.0] * len(scores)
            return [(s - mean) / std for s in scores]
        return scores


_DEFAULT_RERANKER: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _DEFAULT_RERANKER
    if _DEFAULT_RERANKER is None:
        _DEFAULT_RERANKER = CrossEncoderReranker()
    return _DEFAULT_RERANKER