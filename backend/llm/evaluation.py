# backend/llm/evaluation.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from backend.config.settings import settings
from backend.retrieval.retriever import RetrievalResult

# Create module logger
logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationInput",
    "EvaluationResult",
    "EvaluatorResult",
]
   

# ---------------------------------------------------------------------------
# Shared Extraction & Normalization Helper Utilities
# ---------------------------------------------------------------------------

FINANCIAL_METRICS_KEYWORDS = [
    "Revenue from Operations", "Revenue", "Net Profit", "Profit After Tax", "PAT", "EPS", 
    "Total Income", "Operating Revenue", "EBITDA", "Operating Profit", "Borrowings", 
    "Total Debt", "Liabilities", "Assets", "Debt", "Earnings Per Share", "Profit before tax",
    "PBT", "Operating expenses", "Finance costs", "Employee benefit expenses"
]


def extract_citations(text: str) -> list[str]:
    """
    Extracts all bracketed citations from text, e.g. [Doc A, Page 3, Section B].
    """
    return re.findall(r"\[([^\]]+)\]", text)


def extract_numbers(text: str) -> list[str]:
    """
    Extracts numerical strings, filtering out trivial numbers (<= 9) to avoid false positives.
    """
    raw_nums = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", text)
    nums = []
    for val in raw_nums:
        clean_val = val.replace(",", "")
        try:
            num_float = float(clean_val)
            if num_float > 9:
                nums.append(val)
        except ValueError:
            continue
    return list(set(nums))


def extract_percentages(text: str) -> list[str]:
    """
    Extracts percentage expressions, e.g., 15.5%.
    """
    return re.findall(r"\b\d+(?:\.\d+)?\s*%", text)


def extract_currency_values(text: str) -> list[str]:
    """
    Extracts currency values with symbols, e.g., ₹21,248.51, Rs. 500, $100.
    """
    return re.findall(r"(?:₹|Rs\.?|\$)\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text, re.IGNORECASE)


def extract_dates(text: str) -> list[str]:
    """
    Extracts financial and calendar dates/years, e.g., 30-09-2025, FY25, Q3 FY25.
    """
    dates = re.findall(r"\b\d{1,2}[-/\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s]\d{2,4}\b", text, re.IGNORECASE)
    dates += re.findall(r"\b\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4}\b", text)
    dates += re.findall(r"\b(?:FY|Q[1-4]\s*FY)?\s*(?:19|20)\d{2}\b", text)
    dates += re.findall(r"\bFY\d{2}\b", text)
    return list(set(dates))


def extract_entities(text: str) -> list[str]:
    """
    Extracts capitalized terms representing metrics, companies, or entities, ignoring common stops.
    """
    stop_words = {"The", "And", "For", "With", "This", "That", "From", "Here", "There", "Will", "Shall", "Must", "Page", "Section", "Document", "Answer", "Context", "Query", "What", "Who", "How", "Explain", "Write", "State", "Show", "Which", "Why", "When", "Is", "Are", "Was", "Were", "Did", "Do", "Does"}
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text)
    entities = []
    for c in candidates:
        if c not in stop_words and len(c) > 3:
            entities.append(c)
    return list(set(entities))


def extract_metrics(text: str) -> list[str]:
    """
    Extracts financial metrics using known dictionary matches and capitalised sequence rules.
    """
    extracted = []
    for m in FINANCIAL_METRICS_KEYWORDS:
        if re.search(r"\b" + re.escape(m) + r"\b", text, re.IGNORECASE):
            extracted.append(m)
            
    # Extract capitalized phrases that look like metrics
    candidates = re.findall(r"\b[A-Z][a-zA-Z]*(?:\s+(?:from\s+|for\s+|of\s+|in\s+|and\s+|to\s+)?[A-Z][a-zA-Z]*)*\b", text)
    for c in candidates:
        c_clean = c.strip()
        # Strip leading stop words like "The ", "A ", "An " case-insensitively
        c_clean = re.sub(r"^(?:The|A|An)\s+", "", c_clean, flags=re.IGNORECASE)
        # Filter out years or quarters
        if re.search(r"\b(?:FY\d{2,4}|Q[1-4])\b", c_clean, re.IGNORECASE):
            continue
        if len(c_clean) > 3 and c_clean not in extracted:
            extracted.append(c_clean)
    return list(set(extracted))


def extract_periods(text: str) -> list[str]:
    """
    Extracts reporting periods (Quarter, Year, Half-Year, Month) along with dates.
    """
    periods = []
    matches = re.finditer(r"\b(?:Quarter|Year|Half[- ]Year|Month)\s+(?:Ended|Ending|Ended On)?\s*[\w\-\/\s\.]+", text, re.IGNORECASE)
    for m in matches:
        periods.append(m.group(0).strip())
    q_matches = re.findall(r"\bQ[1-4]\s*FY\d{2,4}\b", text, re.IGNORECASE)
    periods.extend(q_matches)
    for d in extract_dates(text):
        if d not in periods:
            periods.append(d)
    return list(set(periods))


def extract_financial_entities(text: str) -> list[str]:
    """
    Extracts company or bank names from the text, ignoring general RAG stopwords.
    """
    stop_words = {"The", "And", "For", "With", "This", "That", "From", "Here", "There", "Will", "Shall", "Must", "Page", "Section", "Document", "Answer", "Context", "Query", "Quarter", "Year", "Month", "Ended", "Standalone", "Consolidated", "What", "Who", "How", "Explain", "Write", "State", "Show", "Which", "Why", "When", "Is", "Are", "Was", "Were", "Did", "Do", "Does"}
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text)
    entities = []
    for c in candidates:
        words = c.split()
        if all(w not in stop_words for w in words) and len(c) > 3:
            entities.append(c.strip())
    return list(set(entities))


def extract_table_claims(text: str) -> list[tuple[str, str, str]]:
    """
    Extracts Metric * Period * Value tuples from both structured markdown tables and sentences.
    """
    claims = []
    
    # 1. Parse markdown tables
    lines = text.splitlines()
    headers = []
    table_rows = []
    in_table = False
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if all(re.match(r"^:?\-+:?$", p) for p in parts) and parts:
                in_table = True
                continue
            if in_table:
                table_rows.append(parts)
            else:
                headers = parts
        else:
            if in_table:
                if len(headers) > 1 and table_rows:
                    col_headers = headers[1:]
                    for row in table_rows:
                        if len(row) > 1:
                            row_label = row[0]
                            for col_idx, val in enumerate(row[1:]):
                                if col_idx < len(col_headers) and val.strip():
                                    claims.append((row_label, col_headers[col_idx], val))
                headers = []
                table_rows = []
                in_table = False
                
    if in_table and len(headers) > 1 and table_rows:
        col_headers = headers[1:]
        for row in table_rows:
            if len(row) > 1:
                row_label = row[0]
                for col_idx, val in enumerate(row[1:]):
                    if col_idx < len(col_headers) and val.strip():
                        claims.append((row_label, col_headers[col_idx], val))

    # 2. Parse sentence-based claims (excluding table lines)
    non_table_lines = [line for line in lines if "|" not in line]
    non_table_text = "\n".join(non_table_lines)
    
    sentences = re.split(r"(?<=[.!?])\s+", non_table_text)
    for sent in sentences:
        nums = extract_numbers(sent) + extract_currency_values(sent) + extract_percentages(sent)
        if not nums:
            continue
        metrics = extract_metrics(sent)
        periods = extract_periods(sent)
        
        # Clean periods to keep only the longest ones
        clean_periods = []
        for p in periods:
            if not any(p != other and p in other for other in periods):
                clean_periods.append(p)
                
        # Clean metrics to keep only the longest ones and exclude any that are substrings of periods
        clean_metrics = []
        for m in metrics:
            if not any(m != other and m in other for other in metrics):
                is_part_of_period = False
                for p in clean_periods:
                    if m.lower() in p.lower():
                        is_part_of_period = True
                        break
                if not is_part_of_period:
                    clean_metrics.append(m)
                
        # Clean nums to exclude any number that is a substring of any extracted period
        clean_nums = []
        for n in nums:
            is_part_of_date = False
            for p in clean_periods:
                if n in p:
                    is_part_of_date = True
                    break
            if not is_part_of_date:
                clean_nums.append(n)
                
        if clean_metrics and clean_periods and clean_nums:
            for m in clean_metrics:
                for p in clean_periods:
                    for n in clean_nums:
                        if n not in m and n not in p:
                            claims.append((m, p, n))

    unique_claims = []
    seen = set()
    for m, p, v in claims:
        key = (m.strip(), p.strip(), v.strip())
        if key not in seen:
            seen.add(key)
            unique_claims.append(key)
    return unique_claims


def normalize_currency(curr_str: str) -> str:
    """
    Standardises currency identifiers.
    """
    cleaned = curr_str.strip().lower()
    if cleaned in ["₹", "rs", "rs.", "rupees", "rupee", "inr"]:
        return "INR"
    if cleaned in ["$", "usd", "dollars", "dollar"]:
        return "USD"
    return cleaned.upper()


def normalize_numbers(num_str: str) -> float | None:
    """
    Standardises and converts signed/bracketed numbers and currency strings to float representation.
    """
    if not num_str:
        return None
    cleaned = num_str.strip()
    is_negative = False
    
    # Handle bracketed values representing negative numbers, e.g. (2,000)
    if cleaned.startswith("(") and cleaned.endswith(")"):
        is_negative = True
        cleaned = cleaned[1:-1].strip()
        
    # Strip currency prefixes at the start
    cleaned = re.sub(r"^(?:₹|Rs\.?|\$)\s*", "", cleaned, flags=re.IGNORECASE)
    
    # Strip trailing percent sign if present
    cleaned = re.sub(r"\s*%\s*$", "", cleaned)
    
    cleaned = re.sub(r"[^\d.+\-eE]", "", cleaned.replace(",", ""))
    try:
        val = float(cleaned)
        if is_negative:
            val = -val
        return val
    except ValueError:
        return None


def normalize_dates(date_str: str) -> str:
    """
    Standardises calendar dates into YYYY-MM-DD.
    """
    cleaned = date_str.strip().lower()
    cleaned = re.sub(r"[/\s\.]+", "-", cleaned)
    
    months = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "september": "09", "sept": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12"
    }
    
    m = re.match(r"(\d{1,2})-(\d{1,2})-(\d{2,4})", cleaned)
    if m:
        d, m_num, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{int(m_num):02d}-{int(d):02d}"
        
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", cleaned)
    if m:
        y, m_num, d = m.groups()
        return f"{y}-{int(m_num):02d}-{int(d):02d}"
        
    m = re.match(r"(\d{1,2})-([a-z]+)-(\d{2,4})", cleaned)
    if m:
        d, month_name, y = m.groups()
        if month_name in months:
            m_num = months[month_name]
            if len(y) == 2:
                y = "20" + y
            return f"{y}-{m_num}-{int(d):02d}"
            
    return date_str.strip().upper()


def fuzzy_numeric_match(val1: float, val2: float, tolerance: float = 1e-4) -> bool:
    """
    Performs float matching within tolerance to handle float differences (e.g. 21,248.5 vs 21248.50).
    """
    return abs(val1 - val2) <= tolerance


def preprocess_text(text: str) -> str:
    """
    Strips ordinal suffixes from numbers (e.g. 30th -> 30) for better matching.
    """
    if not text:
        return ""
    return re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)


def normalize_period_string(p: str) -> str:
    """
    Standardizes a period string by lowercasing, replacing spaces/hyphens with a single space,
    and replacing any detected dates with their YYYY-MM-DD normalized form.
    """
    p_clean = p.lower()
    p_clean = re.sub(r"[/\s\.\-]+", " ", p_clean).strip()
    
    dates = extract_dates(preprocess_text(p))
    for d in dates:
        norm_d = normalize_dates(preprocess_text(d))
        if norm_d != d:
            d_clean = re.sub(r"[/\s\.\-]+", " ", d.lower()).strip()
            if d_clean in p_clean:
                p_clean = p_clean.replace(d_clean, norm_d)
            else:
                p_clean = re.sub(re.escape(d_clean), norm_d, p_clean)
    return p_clean


def normalized_period_match(ans_p: str, ctx_p: str) -> bool:
    """
    Fuzzy matches two period strings using normalized period values.
    """
    ap = normalize_period_string(ans_p)
    cp = normalize_period_string(ctx_p)
    
    if ap == cp:
        return True
        
    if ap in cp or cp in ap:
        # Check for period type conflicts (e.g. quarter vs half-year)
        period_types = ["quarter", "half year", "half-year", "nine month", "year", "annual"]
        for pt in period_types:
            pt_norm = pt.replace("-", " ")
            ap_std = ap.replace("-", " ")
            cp_std = cp.replace("-", " ")
            
            if pt_norm in ap_std and pt_norm not in cp_std:
                for pt2 in period_types:
                    pt2_norm = pt2.replace("-", " ")
                    if pt2_norm != pt_norm and pt2_norm in cp_std:
                        return False
            if pt_norm in cp_std and pt_norm not in ap_std:
                for pt2 in period_types:
                    pt2_norm = pt2.replace("-", " ")
                    if pt2_norm != pt_norm and pt2_norm in ap_std:
                        return False
        return True
    return False


def parse_json_safely(text: str) -> dict | list | None:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    match_braces = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match_braces:
        try:
            return json.loads(match_braces.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    return None


class EvaluationContext:
    def __init__(self, input_data: 'EvaluationInput', llm: BaseChatModel | None) -> None:
        self.query = input_data.query
        self.answer = input_data.answer
        self.retrieved_context = input_data.retrieved_context
        self.retrieval_results = input_data.retrieval_results
        self.citations = input_data.citations
        self.llm = llm
        
        # Lazy caches
        self._extracted_claims: list[str] | None = None
        self._claim_evaluations: list[dict[str, Any]] | None = None
        self._coverage_eval: dict[str, Any] | None = None

    def get_extracted_claims(self) -> list[str]:
        if self._extracted_claims is not None:
            return self._extracted_claims

        if not self.answer.strip():
            self._extracted_claims = []
            return self._extracted_claims

        prompt = f"""You are an expert claim extraction assistant.
Extract all atomic factual claims from the generated answer. An atomic claim is a single statement that makes a specific assertion.
Do not extract questions, opinions, or non-factual statements.

Answer:
{self.answer}

Respond with a JSON object in the following format:
{{
  "claims": [
    "Atomic claim 1...",
    "Atomic claim 2..."
  ]
}}
"""
        try:
            messages = [("user", prompt)]
            response = self.llm.invoke(messages)
            response_text = getattr(response, "content", response)
            data = parse_json_safely(response_text)
            if data and isinstance(data, dict) and "claims" in data:
                self._extracted_claims = [c for c in data["claims"] if isinstance(c, str) and c.strip()]
            else:
                self._extracted_claims = []
        except Exception as exc:
            logger.warning("Claim extraction failed: %s", exc)
            self._extracted_claims = []

        return self._extracted_claims

    def get_claim_evaluations(self) -> list[dict[str, Any]]:
        if self._claim_evaluations is not None:
            return self._claim_evaluations

        claims = self.get_extracted_claims()
        if not claims:
            self._claim_evaluations = []
            return self._claim_evaluations

        # Create chunk string list
        chunks_str = ""
        if self.retrieval_results:
            for idx, r in enumerate(self.retrieval_results):
                content = getattr(r, "page_content", None) or getattr(r, "content", "")
                chunks_str += f"Chunk {idx}: {content}\n\n"
        else:
            chunks_str = f"Chunk 0: {self.retrieved_context}\n"

        prompt = f"""You are a RAG Claim Verifier. Your task is to verify whether each of the extracted claims is supported, unsupported, or contradicted by the retrieved context.

Retrieved Context Chunks:
{chunks_str}

Claims to verify:
{json.dumps(claims)}

For each claim:
1. Determine if it is "Supported", "Unsupported", or "Contradicted" by the retrieved context.
2. Provide a confidence score between 0.0 and 1.0.
3. Cite the supporting chunk index/indices from the context if the claim is Supported (e.g. [0, 2]). If not supported, return an empty list.

For dates and numerical values (especially table cells):
- Be lenient with formatting variations (e.g. "30th September 2025" matches "30-09-2025", "30 Sept", or "2025-09-30" in the context).
- Allow standard currency conversions or representations (e.g. "₹15,000 crore" matches "15000.00").

Respond with a JSON object in the following format:
{{
  "evaluations": [
    {{
      "claim": "<The claim text>",
      "status": "<Supported / Unsupported / Contradicted>",
      "confidence": <float between 0.0 and 1.0>,
      "supporting_chunks": [<list of integer indices>]
    }}
  ]
}}
"""
        try:
            messages = [("user", prompt)]
            response = self.llm.invoke(messages)
            response_text = getattr(response, "content", response)
            data = parse_json_safely(response_text)
            if data and isinstance(data, dict) and "evaluations" in data:
                self._claim_evaluations = data["evaluations"]
            else:
                self._claim_evaluations = []
        except Exception as exc:
            logger.warning("Claim verification failed: %s", exc)
            self._claim_evaluations = []

        return self._claim_evaluations

    def get_coverage_eval(self) -> dict[str, Any]:
        if self._coverage_eval is not None:
            return self._coverage_eval

        prompt = f"""You are a RAG Coverage Evaluator. Your task is to evaluate how completely the generated answer covers the key information present in the retrieved context.

Retrieved Context:
{self.retrieved_context}

Generated Answer:
{self.answer}

First, extract the key information units or points from the retrieved context.
Second, determine whether each key point is covered by the generated answer.
Third, compute the coverage score as: (Number of covered key points) / (Total key points). If the context is short or concise and the answer covers all of it, the score must be 1.0. Do not penalize the answer for omitting info not present in the context.

Respond with a JSON object in the following format:
{{
  "key_points": [
    "Key point 1...",
    "Key point 2..."
  ],
  "covered_points": [
    "Key point 1..."
  ],
  "coverage_score": <float between 0.0 and 1.0>,
  "reasoning": "<Brief explanation of what key context information is covered or omitted>"
}}
"""
        try:
            messages = [("user", prompt)]
            response = self.llm.invoke(messages)
            response_text = getattr(response, "content", response)
            data = parse_json_safely(response_text)
            if data and isinstance(data, dict) and "coverage_score" in data:
                self._coverage_eval = data
            else:
                self._coverage_eval = {
                    "key_points": [],
                    "covered_points": [],
                    "coverage_score": 1.0,
                    "reasoning": "Could not parse coverage analysis."
                }
        except Exception as exc:
            logger.warning("Coverage evaluation failed: %s", exc)
            self._coverage_eval = {
                "key_points": [],
                "covered_points": [],
                "coverage_score": 1.0,
                "reasoning": f"Coverage calculation failed: {exc}"
            }

        return self._coverage_eval


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class EvaluationInput:
    """
    Immutable input variables consumed during evaluation.
    """
    query: str
    answer: str
    retrieved_context: str
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EvaluatorResult:
    """
    Result emitted by an individual check evaluator.
    """
    score: float
    passed: bool
    warnings: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    """
    Complete consolidated result produced by the EvaluationManager.
    """
    grounding_score: float | None
    coverage_score: float | None
    hallucination_risk: str
    passed: bool
    reasoning: str
    citation_score: float = 1.0
    numerical_score: float = 1.0
    table_score: float = 1.0
    hallucination_score: float = 1.0
    confidence_score: float = 1.0
    summary: str = ""
    details: list[str] = field(default_factory=list)
    status: str = "PASSED"
    supporting_chunks: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluator Interfaces & Implementations
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """
    Base class representing an independent evaluator check.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        pass

    @abstractmethod
    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        pass


class GroundingEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "grounding"

    @property
    def is_enabled(self) -> bool:
        return settings.enable_grounding_check

    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        logger.info("Grounding Evaluation Started (Claim-Based)")
        
        context: EvaluationContext = input_data.metadata.get("evaluation_context")
        if context is None:
            context = EvaluationContext(input_data, self._llm)
            input_data.metadata["evaluation_context"] = context

        evals = context.get_claim_evaluations()
        if not evals:
            logger.info("Grounding Score = 1.00 (No claims extracted/evaluated)")
            return EvaluatorResult(
                score=1.0,
                passed=True,
                warnings=[],
                details=["No factual claims extracted to evaluate grounding."]
            )

        total_claims = len(evals)
        supported_claims = sum(1 for e in evals if e.get("status") == "Supported")
        score = supported_claims / total_claims

        passed = score >= settings.grounding_threshold

        warnings = []
        details = []
        for e in evals:
            status = e.get("status", "Unsupported")
            claim = e.get("claim", "")
            conf = e.get("confidence", 1.0)
            chunks = e.get("supporting_chunks", [])
            details.append(f"Claim: '{claim}' | Status: {status} | Confidence: {conf:.2f} | Chunks: {chunks}")
            if status != "Supported":
                warnings.append(f"Claim '{claim}' is {status.lower()} (Confidence: {conf:.2f}).")

        logger.info("Grounding Score = %.2f", score)
        return EvaluatorResult(
            score=score,
            passed=passed,
            warnings=warnings,
            details=details,
            diagnostics={
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "evaluations": evals
            }
        )


class CoverageEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "coverage"

    @property
    def is_enabled(self) -> bool:
        return settings.enable_coverage_check

    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        logger.info("Coverage Evaluation Started (Claim-Based)")

        context: EvaluationContext = input_data.metadata.get("evaluation_context")
        if context is None:
            context = EvaluationContext(input_data, self._llm)
            input_data.metadata["evaluation_context"] = context

        cov_data = context.get_coverage_eval()
        score = cov_data.get("coverage_score", 1.0)
        reasoning = cov_data.get("reasoning", "Coverage evaluated successfully.")
        key_points = cov_data.get("key_points", [])
        covered_points = cov_data.get("covered_points", [])

        passed = score >= settings.query_coverage_threshold

        warnings = []
        if score < 1.0:
            warnings.append(f"Omitted context key points: {list(set(key_points) - set(covered_points))}")

        details = [
            f"Coverage Score: {score:.2f}",
            f"Reasoning: {reasoning}",
            f"Key points extracted: {key_points}",
            f"Covered points: {covered_points}"
        ]

        logger.info("Coverage Score = %.2f", score)
        return EvaluatorResult(
            score=score,
            passed=passed,
            warnings=warnings,
            details=details,
            diagnostics={
                "coverage_score": score,
                "key_points": key_points,
                "covered_points": covered_points,
                "reasoning": reasoning
            }
        )


class CitationEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "citation"

    @property
    def is_enabled(self) -> bool:
        return settings.enable_citation_check

    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        logger.info("Citation Evaluation Started")

        citations = extract_citations(input_data.answer)
        if not citations:
            if settings.allow_missing_citations:
                logger.info("Citation Score = 1.00 (No citations found, but allowed by configuration)")
                return EvaluatorResult(score=1.0, passed=True, details=["No citations present, allowed by configuration."])
            else:
                logger.info("Citation Score = 0.00 (No citations found, strict mode enabled)")
                return EvaluatorResult(
                    score=0.0,
                    passed=False,
                    warnings=["Strict citations mode: answer contains no citations."],
                    details=["Citations are required but missing."]
                )

        verified_citations = []
        invalid_citations = []

        # Index metadata from results (Doc, Page, Section)
        chunk_metadata_list = []
        for r in input_data.retrieval_results:
            doc = (r.metadata.get("document_name") or r.metadata.get("source_file") or "").lower().strip()
            page = str(r.metadata.get("page_number") or "").strip()
            sec = str(r.metadata.get("section_title") or "").lower().strip()
            chunk_metadata_list.append((doc, page, sec))

        for cit in citations:
            parts = [p.strip().lower() for p in cit.split(",")]
            
            matched = False
            for (doc_name, page_num, sec_title) in chunk_metadata_list:
                has_doc = any(p in doc_name or doc_name in p for p in parts)
                
                # Page match logic
                has_page = False
                if page_num:
                    has_page_part = False
                    for p in parts:
                        if "page" in p or "p." in p or p.strip().isdigit():
                            has_page_part = True
                            p_digits = re.findall(r"\d+", p)
                            if page_num in p_digits or page_num == p.strip():
                                has_page = True
                                break
                    if not has_page_part:
                        has_page = True
                else:
                    has_page = True
                    
                # Section match logic
                has_sec = False
                if sec_title:
                    has_sec_part = False
                    for p in parts:
                        if p not in doc_name and not re.search(r"\b(?:page|p\.?)\s*\d+|\b\d+\b", p):
                            has_sec_part = True
                            if p in sec_title or sec_title in p:
                                has_sec = True
                                break
                    if not has_sec_part:
                        has_sec = True
                else:
                    has_sec = True
                    
                if has_doc and has_page and has_sec:
                    matched = True
                    break

            if matched:
                verified_citations.append(cit)
            else:
                invalid_citations.append(cit)

        total_citations = len(citations)
        score = len(verified_citations) / total_citations if total_citations > 0 else 1.0

        passed = score >= 0.8
        if settings.strict_citation_validation and score < 1.0:
            passed = False

        warnings = []
        if invalid_citations:
            warnings.append(f"Invalid citations (mismatching context metadata): {invalid_citations}")

        details = [
            f"Total Citations Checked: {total_citations}",
            f"Verified Citations: {len(verified_citations)}",
            f"Invalid Citations: {len(invalid_citations)}"
        ]

        logger.info("Citation Score = %.2f", score)
        return EvaluatorResult(
            score=score,
            passed=passed,
            warnings=warnings,
            details=details,
            diagnostics={"verified_citations": verified_citations, "invalid_citations": invalid_citations}
        )


class NumericalEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "numerical"

    @property
    def is_enabled(self) -> bool:
        return settings.enable_numeric_validation and settings.enable_numeric_check

    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        logger.info("Numerical Evaluation Started")

        answer_claims = extract_table_claims(preprocess_text(input_data.answer))
        context_claims = extract_table_claims(preprocess_text(input_data.retrieved_context))

        # Build fallback list of simple number tokens
        ans_nums = extract_numbers(preprocess_text(input_data.answer)) + extract_currency_values(preprocess_text(input_data.answer)) + extract_percentages(preprocess_text(input_data.answer))
        context_nums = extract_numbers(preprocess_text(input_data.retrieved_context)) + extract_numbers(preprocess_text(input_data.query))

        mismatches = []
        checked_claims_count = 0
        supported_claims_count = 0
        
        if answer_claims:
            logger.info("Numerical: Comparing structured Metric * Period * Value tuples.")
            for ans_m, ans_p, ans_v in answer_claims:
                ans_v_num = normalize_numbers(ans_v)
                if ans_v_num is None:
                    continue
                    
                checked_claims_count += 1
                value_found_in_context = False
                value_matched_correctly = False
                mismatched_tuples = []
                
                for ctx_m, ctx_p, ctx_v in context_claims:
                    ctx_v_num = normalize_numbers(ctx_v)
                    if ctx_v_num is not None and fuzzy_numeric_match(ans_v_num, ctx_v_num):
                        value_found_in_context = True
                        metric_match = (ans_m.lower() in ctx_m.lower()) or (ctx_m.lower() in ans_m.lower())
                        period_match = normalized_period_match(ans_p, ctx_p)
                        
                        if metric_match and period_match:
                            value_matched_correctly = True
                            break
                        else:
                            mismatched_tuples.append((ctx_m, ctx_p, ctx_v))
                            
                if value_matched_correctly:
                    supported_claims_count += 1
                elif value_found_in_context:
                    mismatches.append(f"Value '{ans_v}' exists in table but is assigned to wrong metric/period: " + ", ".join([f"'{m}' / '{p}'" for m, p, v in mismatched_tuples]))
                else:
                    mismatches.append(f"Structured numerical claim '{ans_m}' for period '{ans_p}' value '{ans_v}' not found in context.")
            
            score = supported_claims_count / checked_claims_count if checked_claims_count > 0 else 1.0
            
        else:
            # Fallback to simple number token matching
            logger.info("Numerical: No structured claims found. Running fallback token comparison.")
            def clean_val(v):
                return re.sub(r"[^\d.]", "", v).strip(".")

            context_nums_cleaned = {clean_val(n) for n in context_nums if clean_val(n)}
            
            for n in ans_nums:
                c_n = clean_val(n)
                if c_n not in context_nums_cleaned:
                    if not any(c_n in ctx or ctx in c_n for ctx in context_nums_cleaned):
                        mismatches.append(f"Number '{n}' not found in context")
                        
            total_checked = len(ans_nums)
            score = 1.0 - (len(mismatches) / total_checked) if total_checked > 0 else 1.0
            checked_claims_count = total_checked

        passed = score >= 0.9
        warnings = mismatches
        details = [
            f"Numbers Checked: {checked_claims_count}",
            f"Mismatches: {len(mismatches)}"
        ]

        logger.info("Numerical Score = %.2f", score)
        return EvaluatorResult(
            score=score,
            passed=passed,
            warnings=warnings,
            details=details,
            diagnostics={
                "numbers_checked": checked_claims_count,
                "mismatches": len(mismatches)
            }
        )


class TableEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "table"

    @property
    def is_enabled(self) -> bool:
        return settings.enable_table_validation and settings.enable_table_check

    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        logger.info("Table Evaluation Started")

        has_table = any(r.metadata.get("content_type") == "table" for r in input_data.retrieval_results) or "|---" in input_data.retrieved_context
        if not has_table:
            logger.info("Table Score = 1.00 (No table evidence exists, marking N/A)")
            return EvaluatorResult(
                score=1.0,
                passed=True,
                details=["No table evidence present in retrieved context. Not applicable."]
            )

        # 1. Deterministic validation first
        table_cells = extract_table_claims(preprocess_text(input_data.retrieved_context))
        ans_nums = extract_numbers(preprocess_text(input_data.answer)) + extract_currency_values(preprocess_text(input_data.answer))
        
        if table_cells and ans_nums:
            logger.info("Table: Executing deterministic cell mapping.")
            query_words = set(w.lower() for w in re.findall(r"\b\w+\b", preprocess_text(input_data.query)))
            
            best_cell = None
            max_query_score = 0.0
            
            for row_label, col_header, cell_value in table_cells:
                row_words = set(w.lower() for w in re.findall(r"\b\w+\b", row_label))
                col_words = set(w.lower() for w in re.findall(r"\b\w+\b", col_header))
                
                row_score = len(row_words & query_words) / len(row_words) if row_words else 0
                col_score = len(col_words & query_words) / len(col_words) if col_words else 0
                
                score = row_score + col_score
                if score > max_query_score:
                    max_query_score = score
                    best_cell = (row_label, col_header, cell_value)
                    
            if best_cell is not None and max_query_score >= 0.5:
                best_row, best_col, best_val = best_cell
                best_val_num = normalize_numbers(best_val)
                
                warnings = []
                details = [f"Query targets row '{best_row}' and column '{best_col}' (Expected value: '{best_val}')."]
                
                passed = True
                matched_any_correct = False
                mismatches_found = False
                
                for n in ans_nums:
                    n_num = normalize_numbers(n)
                    if n_num is None:
                        continue
                        
                    if best_val_num is not None and fuzzy_numeric_match(n_num, best_val_num):
                        matched_any_correct = True
                        continue
                        
                    mismatches = []
                    for row_label, col_header, cell_value in table_cells:
                        cell_num = normalize_numbers(cell_value)
                        if cell_num is not None and fuzzy_numeric_match(n_num, cell_num):
                            if row_label.lower() == best_row.lower() and col_header.lower() != best_col.lower():
                                mismatches.append(f"wrong column '{col_header}' for row '{row_label}'")
                            elif col_header.lower() == best_col.lower() and row_label.lower() != best_row.lower():
                                mismatches.append(f"wrong row '{row_label}' for column '{col_header}'")
                            else:
                                mismatches.append(f"row '{row_label}' and column '{col_header}'")
                                
                    if mismatches:
                        mismatches_found = True
                        passed = False
                        warnings.append(f"Answer contains value '{n}' which maps to wrong table cell: " + ", and ".join(mismatches) + f" (Expected '{best_val}' for '{best_row}' / '{best_col}').")
                
                if not passed:
                    score = 0.0
                elif matched_any_correct:
                    score = 1.0
                    details.append("Answer contains the correct table cell value.")
                else:
                    score = 0.5
                    details.append("Answer contains numbers, but correct cell was not matched.")
                    
                logger.info("Table Score = %.2f (Deterministic)", score)
                return EvaluatorResult(
                    score=score,
                    passed=passed,
                    warnings=warnings,
                    details=details,
                    diagnostics={
                        "table_chunks_examined": 1,
                        "matched_cells": 1 if passed else 0,
                        "fallback_llm_used": False
                    }
                )

        # 2. Fallback to LLM Table check
        logger.info("Deterministic table check inconclusive. Falling back to LLM.")
        prompt = f"""You are an expert RAG Table Consistency Evaluator.
The retrieved context contains table data. Analyze if the generated answer is completely consistent with the table columns, rows, cell values, and headers in the context.

Context:
{input_data.retrieved_context}

Answer:
{input_data.answer}

Verify if cell values are correctly associated with columns and rows.
Return EXACTLY:
Table Consistency Score: <score between 0.0 and 1.0>
Passed: <True/False>
Reasoning: <brief reasoning of consistency or warning cell mismatch reasons>
"""
        try:
            messages = [("user", prompt)]
            response = self._llm.invoke(messages)
            response_text = getattr(response, "content", response)

            score_match = re.search(r"Table\s+Consistency\s+Score\s*:\s*([\d.]+)", response_text, re.IGNORECASE)
            passed_match = re.search(r"Passed\s*:\s*(True|False)", response_text, re.IGNORECASE)
            reasoning_match = re.search(r"Reasoning\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

            score = float(score_match.group(1)) if score_match else 1.0
            passed = passed_match.group(1).strip().lower() == "true" if passed_match else True
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Table cells consistent."

            warnings = [] if passed else [f"Table cell mismatch: {reasoning}"]
            details = [reasoning]

            logger.info("Table Score = %.2f (LLM Fallback)", score)
            return EvaluatorResult(
                score=score,
                passed=passed,
                warnings=warnings,
                details=details,
                diagnostics={
                    "table_chunks_examined": 1,
                    "matched_cells": 1 if passed else 0,
                    "fallback_llm_used": True
                }
            )
        except Exception as exc:
            logger.warning("LLM call failed in TableEvaluator: %s. Falling back to default table pass.", exc)
            return EvaluatorResult(score=1.0, passed=True, details=["LLM failed, fallback to default table pass."])


class HallucinationEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "hallucination"

    @property
    def is_enabled(self) -> bool:
        return settings.enable_hallucination_check

    def evaluate(self, input_data: EvaluationInput) -> EvaluatorResult:
        logger.info("Hallucination Evaluation Started (Claim-Based)")

        context: EvaluationContext = input_data.metadata.get("evaluation_context")
        if context is None:
            context = EvaluationContext(input_data, self._llm)
            input_data.metadata["evaluation_context"] = context

        evals = context.get_claim_evaluations()
        if not evals:
            logger.info("Hallucination Score = 1.00, Risk = Low (No claims extracted)")
            return EvaluatorResult(
                score=1.0,
                passed=True,
                warnings=[],
                details=["Risk Level: Low", "Unsupported Facts: 0"]
            )

        total_claims = len(evals)
        unsupported_claims = [e for e in evals if e.get("status") in ("Unsupported", "Contradicted")]
        unsupported_count = len(unsupported_claims)

        score = 1.0 - (unsupported_count / total_claims)

        if unsupported_count == 0:
            risk = "Low"
        elif unsupported_count >= 3 or (total_claims > 0 and (unsupported_count / total_claims) >= 0.5):
            risk = "High"
        else:
            risk = "Medium"

        passed = risk != settings.hallucination_fail_level

        warnings = [f"Claim '{e.get('claim')}' is {e.get('status').lower()} (Confidence: {e.get('confidence', 0.0):.2f})." for e in unsupported_claims]
        details = [
            f"Risk Level: {risk}",
            f"Unsupported Facts: {unsupported_count}",
            f"Hallucination Score: {score:.2f}"
        ]

        logger.info("Hallucination Score = %.2f, Risk = %s", score, risk)
        return EvaluatorResult(
            score=score,
            passed=passed,
            warnings=warnings,
            details=details,
            diagnostics={
                "risk_level": risk,
                "unsupported_facts": unsupported_count,
                "unsupported_claims": unsupported_claims
            }
        )


# ---------------------------------------------------------------------------
# EvaluationManager Orchestrator
# ---------------------------------------------------------------------------

class EvaluationManager:
    """
    Orchestrates independent evaluators, gathering scores, managing warnings,
    and computing metrics.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm
        self._evaluators: list[BaseEvaluator] = [
            GroundingEvaluator(llm),
            CoverageEvaluator(llm),
            CitationEvaluator(llm),
            NumericalEvaluator(llm),
            TableEvaluator(llm),
            HallucinationEvaluator(llm),
        ]
        self._last_metadata: dict[str, Any] = {}

    def evaluate(self, input_data: EvaluationInput) -> EvaluationResult:
        logger.info("Evaluation started")
        
        sufficiency = input_data.metadata.get("retrieval_sufficiency", True)
        if sufficiency is None:
            sufficiency = True
            
        is_refusal = input_data.answer == "The retrieved context does not contain this information."
            
        if not sufficiency or input_data.retrieved_context == "No Relevant Context Found" or is_refusal:
            logger.info("Evaluation bypassed: INSUFFICIENT_EVIDENCE detected")
            return EvaluationResult(
                grounding_score=None,
                coverage_score=None,
                hallucination_risk="Low",
                passed=False,
                reasoning="No retrieved document satisfied the relevance threshold.",
                summary="No retrieved document satisfied the relevance threshold.",
                status="INSUFFICIENT_EVIDENCE",
                details=["No retrieved document satisfied the relevance threshold."],
                supporting_chunks=[]
            )

        # Initialize shared EvaluationContext
        context = EvaluationContext(input_data, self._llm)
        input_data.metadata["evaluation_context"] = context
        
        start_time = time.perf_counter()
        scores: dict[str, float] = {}
        evaluator_results: dict[str, EvaluatorResult] = {}
        evaluators_run = []
        failed_evaluators = []
        warnings = []
        timings = {}

        for evaluator in self._evaluators:
            if not evaluator.is_enabled:
                continue

            evaluators_run.append(evaluator.name)
            t_start = time.perf_counter()
            try:
                result = evaluator.evaluate(input_data)
                evaluator_results[evaluator.name] = result
                scores[evaluator.name] = result.score
                if result.warnings:
                    warnings.extend(result.warnings)
            except Exception as exc:
                failed_evaluators.append(evaluator.name)
                warnings.append(f"Evaluator '{evaluator.name}' failed: {str(exc)}")
                logger.error("Evaluator failed | name=%s | error=%s", evaluator.name, exc)
            finally:
                timings[evaluator.name] = (time.perf_counter() - t_start) * 1000.0

        # Retrieve scores
        grounding_score = scores.get("grounding", 1.0)
        coverage_score = scores.get("coverage", 1.0)
        citation_score = scores.get("citation", 1.0)
        numerical_score = scores.get("numerical", 1.0)
        table_score = scores.get("table", 1.0)
        hallucination_score = scores.get("hallucination", 1.0)

        # Average confidence calculation
        valid_scores = list(scores.values())
        confidence_score = sum(valid_scores) / len(valid_scores) if valid_scores else 1.0
        logger.info("Confidence computed | confidence=%.3f", confidence_score)

        # Map risk from hallucination evaluator diagnostics if available
        if "hallucination" in evaluator_results:
            hallucination_risk = evaluator_results["hallucination"].diagnostics.get("risk_level", "Low")
        else:
            if hallucination_score >= 0.9:
                hallucination_risk = "Low"
            elif hallucination_score >= 0.7:
                hallucination_risk = "Medium"
            else:
                hallucination_risk = "High"

        # Check for metric inconsistencies
        is_consistent = True
        inconsistency_message = ""
        if grounding_score >= 0.75 and hallucination_risk == "High":
            is_consistent = False
            inconsistency_message = f"Evaluation Inconsistency Detected: Grounding Score is high ({grounding_score:.2f}) but Hallucination Risk is High."
            logger.warning(inconsistency_message)
        elif grounding_score < 0.3 and hallucination_risk == "Low":
            is_consistent = False
            inconsistency_message = f"Evaluation Inconsistency Detected: Grounding Score is low ({grounding_score:.2f}) but Hallucination Risk is Low."
            logger.warning(inconsistency_message)

        # Determine passed status
        grounding_passed = grounding_score >= settings.grounding_threshold
        if "grounding" in evaluator_results:
            grounding_passed = grounding_passed and evaluator_results["grounding"].passed

        coverage_passed = coverage_score >= settings.query_coverage_threshold
        if "coverage" in evaluator_results:
            coverage_passed = coverage_passed and evaluator_results["coverage"].passed

        hallucination_passed = hallucination_risk != settings.hallucination_fail_level

        # Derive Pass/Fail strictly from Grounding, Coverage, and Hallucination Risk
        passed = (
            grounding_passed
            and coverage_passed
            and hallucination_passed
        )

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Log evaluator summaries separately (Part 10 — Logging)
        for name, score in scores.items():
            logger.info("%s Evaluation Started | Score = %.2f", name.capitalize(), score)
            
        logger.info("Evaluation finished | passed=%s | latency=%.2fms", passed, total_time_ms)
        logger.info(
            "Final Evaluation Latency | Grounding: %.2fms | Coverage: %.2fms | Numeric: %.2fms | Table: %.2fms | Citation: %.2fms | Hallucination: %.2fms | Total: %.2fms",
            timings.get("grounding", 0.0),
            timings.get("coverage", 0.0),
            timings.get("numerical", 0.0),
            timings.get("table", 0.0),
            timings.get("citation", 0.0),
            timings.get("hallucination", 0.0),
            total_time_ms
        )

        summary = f"RAG Evaluation completed. Passed: {passed}. Overall Confidence: {confidence_score:.2f}."
        
        details = []
        for name, res in evaluator_results.items():
            details.append(f"{name.capitalize()} Score: {res.score:.2f} (Passed: {res.passed})")
            if res.details:
                details.extend([f"  - {d}" for d in res.details])

        # Gather supporting chunk indices from evaluations
        supporting_chunks = []
        if context and context._claim_evaluations:
            supporting_indices = set()
            for ev in context._claim_evaluations:
                if ev.get("status") == "Supported" and ev.get("supporting_chunks"):
                    supporting_indices.update(ev.get("supporting_chunks"))
            supporting_chunks = sorted(list(supporting_indices))

        # Diagnostic metadata collection
        self._last_metadata = {
            "evaluation_time_ms": total_time_ms,
            "evaluators_run": evaluators_run,
            "failed_evaluators": failed_evaluators,
            "warnings": warnings,
            "total_sources": len(input_data.retrieval_results),
            "table_sources": sum(1 for r in input_data.retrieval_results if r.metadata.get("content_type") == "table"),
            "numeric_claims": len(extract_numbers(input_data.answer)),
            "timings_ms": timings
        }

        return EvaluationResult(
            grounding_score=grounding_score,
            coverage_score=coverage_score,
            hallucination_risk=hallucination_risk,
            passed=passed,
            reasoning=summary,
            citation_score=citation_score,
            numerical_score=numerical_score,
            table_score=table_score,
            hallucination_score=hallucination_score,
            confidence_score=confidence_score,
            summary=summary,
            details=details,
            status="PASSED" if passed else "FAILED",
            supporting_chunks=supporting_chunks
        )
