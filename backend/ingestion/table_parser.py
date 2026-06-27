# backend/ingestion/table_parser.py
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "parse_markdown_table",
    "generate_table_facts",
    "TableCell",
    "TableRow",
    "ParsedTable",
    "TableFact",
    "FinancialTableParser",
]

# Month mapping for expanding abbreviations (e.g. Sep -> September)
_MONTH_MAPPING = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December"
}

_FINANCIAL_KEYWORDS = {
    "revenue", "income", "profit", "loss", "expense", "expenses", "assets", "liabilities",
    "cash", "debt", "borrowings", "equity", "capital", "dividend", "ebitda", "ebit", "pat",
    "expenditure", "receivables", "payables", "goodwill", "inventories", "retained earnings"
}

_EXCLUDED_KEYWORDS = {"eps", "ratio", "percentage", "shares", "margin"}

_FILTERED_ROW_LABELS = {
    "particulars", "notes", "sr no", "sr. no.", "total", "page", "continuation",
    "s. no.", "s.no.", "serial no", "serial number"
}


@dataclass(slots=True, frozen=True)
class TableCell:
    value: str


@dataclass(slots=True, frozen=True)
class TableRow:
    label: str
    values: list[TableCell]


@dataclass(slots=True, frozen=True)
class ParsedTable:
    headers: list[str]
    rows: list[TableRow]

    def to_markdown(self) -> str:
        """
        Reconstructs a clean markdown table representation from headers and rows.
        """
        if not self.headers:
            return ""
        header_line = "| " + " | ".join(self.headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        row_lines = []
        for r in self.rows:
            row_vals = [r.label] + [cell.value for cell in r.values]
            if len(row_vals) < len(self.headers):
                row_vals += [""] * (len(self.headers) - len(row_vals))
            elif len(row_vals) > len(self.headers):
                row_vals = row_vals[:len(self.headers)]
            row_line = "| " + " | ".join(row_vals) + " |"
            row_lines.append(row_line)
        return "\n".join([header_line, sep_line] + row_lines)


class FinancialTableParser:
    """
    Parser wrapper for extracting and processing structured facts from markdown tables.
    """
    def parse(self, markdown: str) -> ParsedTable:
        return parse_markdown_table(markdown)


@dataclass(slots=True, frozen=True)
class TableFact:
    row_label: str
    column_label: str
    value: str
    sentence: str
    table_category: str = "Generic Table"
    table_index: int = 0
    row_index: int = 0
    column_index: int = 0
    normalized_value: Optional[str] = None
    detected_unit: Optional[str] = None
    value_type: str = "text"
    is_numeric: bool = False
    semantic_key: str = ""


def _is_separator_row(line: str) -> bool:
    """
    Checks if a markdown line is a separator row.
    Contains only pipes, hyphens, colons, and spaces.
    """
    line_clean = line.strip()
    if not (line_clean.startswith("|") and line_clean.endswith("|")):
        return False
    chars = set(line_clean)
    permitted = {'|', '-', ':', ' ', '\t'}
    if '-' not in chars:
        return False
    return chars.issubset(permitted)


def _split_row(line: str) -> list[str]:
    """
    Splits a pipe-separated markdown row and trims cell whitespace.
    """
    line_stripped = line.strip()
    if line_stripped.startswith("|"):
        line_stripped = line_stripped[1:]
    if line_stripped.endswith("|"):
        line_stripped = line_stripped[:-1]
    return [col.strip() for col in line_stripped.split("|")]


def _propagate_headers(header_rows: list[list[str]]) -> list[list[str]]:
    """
    Propagates empty cell headers horizontally to preserve hierarchy.
    """
    propagated_rows = []
    for row in header_rows:
        new_row = list(row)
        for col_idx in range(1, len(new_row)):
            if not new_row[col_idx].strip() and new_row[col_idx - 1].strip():
                new_row[col_idx] = new_row[col_idx - 1]
        propagated_rows.append(new_row)
    return propagated_rows


def _flatten_headers(header_rows: list[list[str]]) -> list[str]:
    """
    Flattens multi-row table headers by joining non-empty values vertically.
    """
    if not header_rows:
        return []
    propagated = _propagate_headers(header_rows)
    num_cols = len(propagated[0])
    flattened = []
    for col_idx in range(num_cols):
        col_parts = []
        for row in propagated:
            if col_idx < len(row):
                val = " ".join(row[col_idx].strip().split())
                if val:
                    col_parts.append(val)
        
        joined = " ".join(col_parts)
        # Deduplicate identical adjacent words
        words = joined.split()
        deduped_words = []
        for word in words:
            if not deduped_words or deduped_words[-1] != word:
                deduped_words.append(word)
        flattened.append(" ".join(deduped_words))
    return flattened


def _detect_table_category(headers: list[str], row_labels: list[str]) -> str:
    """
    Classifies the table type based on header and row label keywords.
    """
    text = " ".join(headers + row_labels).lower()
    if "cash flow" in text or "operating activities" in text or "investing activities" in text or "financing activities" in text:
        return "Cash Flow"
    elif "balance sheet" in text or ("assets" in text and "liabilities" in text):
        return "Balance Sheet"
    elif "ratio" in text or "margin" in text or "return on" in text:
        return "Financial Ratios"
    elif "shareholding" in text or "promoter" in text or "public shareholding" in text:
        return "Shareholding"
    elif "revenue" in text or "profit" in text or "loss" in text or "income" in text or "expenses" in text or "eps" in text:
        return "Profit & Loss"
    else:
        return "Generic Table"


def _detect_unit(text_to_scan: str) -> Optional[str]:
    """
    Identifies financial units inside headers, row labels, or captions.
    """
    t = text_to_scan.lower()
    if "crore" in t:
        return "Crores"
    elif "lakh" in t:
        return "Lakhs"
    elif "million" in t:
        return "Million"
    elif "billion" in t:
        return "Billion"
    elif "basis points" in t or " bps" in t:
        return "Basis Points"
    elif "percentage" in t or "%" in t:
        return "Percentage"
    elif "₹" in t or "rs." in t or "rupees" in t:
        return "Rupees"
    return None


def _normalize_numeric_value(val: str) -> tuple[Optional[str], str, bool]:
    """
    Cleans and normalizes a string into a standard float/integer representation.
    """
    val_strip = val.strip()
    if not val_strip:
        return None, "text", False
        
    missing_placeholders = {"—", "-", "n/a", "nil", "null", "none", "not available"}
    if val_strip.lower() in missing_placeholders:
        return None, "missing", False
        
    has_currency = "₹" in val_strip or "rs." in val_strip.lower() or "$" in val_strip
    val_clean = val_strip.replace("₹", "").replace("$", "").replace("Rs.", "").replace("rs.", "").strip()
    
    is_percent = val_clean.endswith("%")
    if is_percent:
        val_clean = val_clean[:-1].strip()
        
    is_negative = False
    if val_clean.startswith("(") and val_clean.endswith(")"):
        is_negative = True
        val_clean = val_clean[1:-1].strip()
    elif val_clean.startswith("-"):
        is_negative = True
        val_clean = val_clean[1:].strip()
        
    val_clean = val_clean.replace(",", "")
    
    try:
        float_val = float(val_clean)
        if is_negative:
            float_val = -float_val
            
        norm_str = f"{float_val:.4f}".rstrip("0").rstrip(".")
        if norm_str == "-0" or norm_str == "-0.0":
            norm_str = "0"
            
        if is_percent:
            return f"{norm_str}%", "percentage", True
        elif has_currency:
            return norm_str, "currency", True
        else:
            return norm_str, "number", True
    except ValueError:
        return None, "text", False


def _format_financial_value(val: str, currency_prefix: str = "₹") -> str:
    """
    Formats a normalized numeric value string with commas.
    """
    val_clean = val.replace("₹", "").replace(",", "").strip()
    is_negative = val_clean.startswith("-")
    if is_negative:
        val_clean = val_clean[1:]
        
    try:
        if "." in val_clean:
            val_float = float(val_clean)
            parts = val_clean.split(".")
            dec_len = len(parts[1])
            formatted = f"{val_float:,.{dec_len}f}"
        else:
            val_int = int(val_clean)
            formatted = f"{val_int:,}"
        
        result = f"{currency_prefix}{formatted}"
        if is_negative:
            result = f"-{result}"
        return result
    except ValueError:
        if not val.startswith(currency_prefix):
            return f"{currency_prefix}{val}"
        return val


def _normalize_date_string(date_str: str) -> str:
    """
    Converts standard financial report dates to 'DD Month YYYY' format.
    """
    clean_date = date_str.replace("-", " ").replace("/", " ").replace(",", " ").strip()
    words = clean_date.split()
    
    if len(words) != 3:
        return date_str
        
    day, month_part, year = None, None, None
    numeric_parts = []
    word_parts = []
    for w in words:
        if w.isdigit():
            numeric_parts.append(int(w))
        else:
            word_parts.append(w)
            
    if len(numeric_parts) == 3:
        part0, part1, part2 = numeric_parts
        if part0 > 31:  # YYYY MM DD
            year = part0
            month_num = part1
            day = part2
        else:  # DD MM YYYY
            day = part0
            month_num = part1
            year = part2
            
        if 1 <= month_num <= 12:
            months = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
            month_part = months[month_num - 1]
            
    elif len(numeric_parts) == 2 and len(word_parts) == 1:
        num0, num1 = numeric_parts
        word = word_parts[0].lower()
        
        month_found = None
        for k, v in _MONTH_MAPPING.items():
            if word.startswith(k):
                month_found = v
                break
                
        if month_found:
            month_part = month_found
            if num0 > 31:
                year = num0
                day = num1
            elif num1 > 31 or num1 > num0:
                year = num1
                day = num0
            else:
                day = num0
                year = num1
                
    if day is not None and month_part is not None and year is not None:
        if year < 100:
            year += 2000 if year <= 50 else 1900
        return f"{day:02d} {month_part} {year}"
        
    return date_str


def _was_or_were(row_label: str) -> str:
    """
    Decides plural verb forms for balance sheet items.
    """
    rl = row_label.lower().strip()
    plurals = ["assets", "liabilities", "equivalents", "receivables", "payables", "borrowings", "shares", "reserves", "activities"]
    if any(p in rl for p in plurals) or rl.endswith("s"):
        return "were"
    return "was"


def _to_semantic_key_part(text: str) -> str:
    """
    Normalizes string parts to build a semantic key.
    """
    clean = "".join(c if c.isalnum() or c == " " else "" for c in text)
    words = clean.split()
    return "_".join(words)


def _generate_semantic_key(row_label: str, column_label: str) -> str:
    """
    Generates a unique semantic key for fact deduplication.
    """
    r_key = _to_semantic_key_part(row_label)
    c_key = _to_semantic_key_part(column_label)
    return f"{r_key}__{c_key}"


def _is_filtered_row(label: str) -> bool:
    """
    Filters non-factual row items out of the pipeline.
    """
    l = label.lower().strip()
    if l in _FILTERED_ROW_LABELS:
        return True
    if all(not c.isalnum() or c.isdigit() for c in l):
        return True
    return False


def parse_markdown_table(markdown: str) -> ParsedTable:
    """
    Parses a markdown table string into a structured ParsedTable object.
    """
    start_time = time.perf_counter()
    logger.info("Table parsing started.")

    if not markdown or not markdown.strip():
        logger.warning("Malformed table skipped | Reason: empty markdown.")
        raise ValueError("Markdown content is empty.")

    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        logger.warning("Malformed table skipped | Reason: too few lines.")
        raise ValueError("Malformed markdown table: must have at least 2 rows.")

    separator_idx = -1
    for idx, line in enumerate(lines):
        if _is_separator_row(line):
            separator_idx = idx
            break

    if separator_idx == -1:
        logger.warning("Malformed table skipped | Reason: missing separator.")
        raise ValueError("Malformed markdown: missing separator row.")

    header_lines = lines[:separator_idx]
    if not header_lines:
        logger.warning("Malformed table skipped | Reason: missing headers.")
        raise ValueError("Malformed markdown: missing header rows.")

    try:
        header_rows_cols = [_split_row(h) for h in header_lines]
        num_cols = len(header_rows_cols[0])

        for h_cols in header_rows_cols:
            if len(h_cols) != num_cols:
                raise ValueError("Inconsistent column count in headers.")

        # Reconstruct hierarchical headers
        try:
            logger.info("Header reconstruction started using propagation strategy.")
            headers = _flatten_headers(header_rows_cols)
        except Exception as exc:
            logger.warning("Header reconstruction failed, falling back to standard concatenation: %s", exc)
            headers = header_rows_cols[-1]

        logger.info("Headers detected | count=%d | values=%s", len(headers), headers)
        if len(headers) < 2:
            raise ValueError("Table must contain at least a row label and a value column.")

        body_lines = lines[separator_idx + 1:]
        if not body_lines:
            raise ValueError("Table contains only headers (header-only table).")

        rows: list[TableRow] = []
        for line in body_lines:
            cols = _split_row(line)
            if not cols:
                continue

            label = cols[0]
            raw_values = cols[1:]

            # Ignore empty trailing columns
            while raw_values and not raw_values[-1]:
                raw_values.pop()

            values = [TableCell(value=val) for val in raw_values]
            rows.append(TableRow(label=label, values=values))

        logger.info("Rows detected | count=%d", len(rows))
        
        elapsed = time.perf_counter() - start_time
        logger.info("Table parsing completed | duration=%.4fs", elapsed)

        return ParsedTable(headers=headers, rows=rows)

    except Exception as exc:
        logger.warning("Malformed table skipped | Reason: %s", exc)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Malformed markdown table structure: {exc}") from exc


def generate_table_facts(table: ParsedTable) -> list[TableFact]:
    """
    Converts a ParsedTable into a flat list of sentence-based semantic TableFacts.
    Supports row label checks, date normalization, unit checks, and deduplication.
    """
    start_time = time.perf_counter()
    logger.info("Facts generation started.")
    facts: list[TableFact] = []
    seen_keys: set[str] = set()

    row_labels = [row.label for row in table.rows]
    category = _detect_table_category(table.headers, row_labels)
    logger.info("Detected table category: %s", category)

    # Detect global table unit from combined header/row context
    combined_header_text = " ".join(table.headers)
    global_unit = _detect_unit(combined_header_text)
    if global_unit:
        logger.info("Global unit detected: %s", global_unit)

    duplicate_count = 0
    skipped_count = 0

    for r_idx, row in enumerate(table.rows):
        row_label = row.label
        
        # Skip rows that are meta-rows or headers (Particulars, Notes, Sr No, etc.)
        if _is_filtered_row(row_label):
            skipped_count += 1
            logger.info("Row filtered out: %s", row_label)
            continue

        row_lower = row_label.lower()
        is_financial = any(kw in row_lower for kw in _FINANCIAL_KEYWORDS)
        is_excluded = any(kw in row_lower for kw in _EXCLUDED_KEYWORDS)

        # Detect row-level unit
        row_unit = _detect_unit(row_label)

        for col_idx, cell in enumerate(row.values):
            val_idx = col_idx + 1
            if val_idx >= len(table.headers):
                break
            
            column_label = table.headers[val_idx]
            raw_val = cell.value

            # Skip empty cells
            if not raw_val or raw_val.strip() in {"", "—", "-"}:
                continue

            # Normalized numeric processing
            norm_val, val_type, is_num = _normalize_numeric_value(raw_val)

            # Detect column-level unit
            col_unit = _detect_unit(column_label)
            unit = col_unit or row_unit or global_unit

            # Date Normalization
            normalized_date = _normalize_date_string(column_label)

            # Generate semantic sentence based on category
            if is_financial and not is_excluded and is_num:
                formatted_val = _format_financial_value(norm_val or raw_val)
                unit_suffix = ""
                if unit == "Crores":
                    unit_suffix = " crore"
                elif unit == "Lakhs":
                    unit_suffix = " lakh"
                elif unit == "Million":
                    unit_suffix = " million"
                elif unit == "Billion":
                    unit_suffix = " billion"
                
                # Context-aware sentence generation
                if category == "Balance Sheet":
                    verb = _was_or_were(row_label)
                    sentence = f"{row_label} as of {normalized_date} {verb} {formatted_val}{unit_suffix}."
                elif category == "Cash Flow":
                    sentence = f"{row_label} for {normalized_date} was {formatted_val}{unit_suffix}."
                else:  # Profit & Loss, Generic
                    sentence = f"{row_label} for {normalized_date} was {formatted_val}{unit_suffix}."
            else:
                # Fallback for ratios, percentages, non-currency metrics
                formatted_val = raw_val
                if unit == "Percentage" and "%" not in formatted_val:
                    formatted_val = f"{formatted_val}%"
                
                if category == "Financial Ratios":
                    sentence = f"{row_label} for {normalized_date} was {formatted_val}."
                else:
                    sentence = f"{row_label} for {normalized_date} is {formatted_val}."

            # Semantic key generation and duplication check
            semantic_key = _generate_semantic_key(row_label, column_label)
            if semantic_key in seen_keys:
                duplicate_count += 1
                logger.info("Duplicate fact removed | key=%s", semantic_key)
                continue
            seen_keys.add(semantic_key)

            facts.append(
                TableFact(
                    row_label=row_label,
                    column_label=column_label,
                    value=raw_val,
                    sentence=sentence,
                    table_category=category,
                    table_index=0,
                    row_index=r_idx,
                    column_index=val_idx,
                    normalized_value=norm_val,
                    detected_unit=unit,
                    value_type=val_type,
                    is_numeric=is_num,
                    semantic_key=semantic_key,
                )
            )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Facts generation completed | count=%d | duplicates_removed=%d | skipped_rows=%d | duration=%.4fs",
        len(facts),
        duplicate_count,
        skipped_count,
        elapsed,
    )
    return facts