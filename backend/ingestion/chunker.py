# backend/ingestion/chunker.py
import hashlib
import uuid
import logging
import re


from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.ingestion.metadata import (
    ChunkData,
    ChunkMetadata,
    DocumentData,
    DocumentElement,
)
from backend.config.settings import settings

logger = logging.getLogger(__name__)

__all__ = [
    "chunk_document",
    "DocumentChunker",
    "estimate_tokens",
    "DocumentData",
    "DocumentElement",
]

DEFAULT_CHUNK_SIZE: int = settings.chunk_size
DEFAULT_CHUNK_OVERLAP: int = settings.chunk_overlap
MIN_WORDS_PER_ELEMENT: int = settings.min_words_per_element

# Regex to detect standard markdown table separator lines of any column count
_MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")


def estimate_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _make_chunk_id(source_file: str, page_number: int, content: str) -> uuid.UUID:
    fingerprint = f"{source_file}::{page_number}::{content}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).digest()
    return uuid.UUID(bytes=digest[:16])


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        logger.info(
            "DocumentChunker initialized | chunk_size=%d | chunk_overlap=%d | min_words=%d",
            chunk_size,
            chunk_overlap,
            MIN_WORDS_PER_ELEMENT,
        )

    def chunk_document(self, document_data: DocumentData) -> list[ChunkData]:
        all_chunks: list[ChunkData] = []
        chunk_index: int = 0
        seen_content_hashes: set[str] = set()

        heading_elements_skipped: int = 0
        tiny_elements_skipped: int = 0
        duplicate_chunks_skipped: int = 0
        unique_chunks_created: int = 0

        logger.info(
            "Starting chunking | document=%s | elements=%d",
            document_data.document_name,
            len(document_data.elements),
        )

        for element in document_data.elements:
            if not element.content or not element.content.strip():
                continue

            content = element.content.strip()
            word_count = len(content.split())

            if element.content_type != "heading" and word_count < MIN_WORDS_PER_ELEMENT:
                tiny_elements_skipped += 1
                continue

            if element.content_type == "heading":
                heading_elements_skipped += 1
                continue

            produced = self._chunk_element(
                element=element,
                document_data=document_data,
                start_chunk_index=chunk_index,
            )

            for chunk in produced:
                fingerprint = hashlib.sha256(
                    f"{document_data.document_id}::{chunk.content}".encode("utf-8")
                ).hexdigest()

                if fingerprint in seen_content_hashes:
                    duplicate_chunks_skipped += 1
                    continue

                seen_content_hashes.add(fingerprint)
                all_chunks.append(chunk)
                unique_chunks_created += 1
                chunk_index += 1

        self._log_chunk_statistics(
            document_name=document_data.document_name,
            total_elements=len(document_data.elements),
            chunks=all_chunks,
            duplicate_chunks_skipped=duplicate_chunks_skipped,
            tiny_elements_skipped=tiny_elements_skipped,
            heading_elements_skipped=heading_elements_skipped,
            unique_chunks_created=unique_chunks_created,
        )

        return all_chunks

    def _chunk_element(
        self,
        element: DocumentElement,
        document_data: DocumentData,
        start_chunk_index: int,
    ) -> list[ChunkData]:
        content_type = element.content_type

        if content_type == "text":
            return self._chunk_text(element, document_data, start_chunk_index)

        if content_type == "table":
            return self._chunk_table(element, document_data, start_chunk_index)

        if content_type == "list":
            return self._chunk_list(element, document_data, start_chunk_index)

        logger.warning(
            "Unknown content_type '%s' encountered. Treating as text. | page=%d",
            content_type,
            element.page_number,
        )
        return self._chunk_text(element, document_data, start_chunk_index)

    def _chunk_text(
        self,
        element: DocumentElement,
        document_data: DocumentData,
        start_chunk_index: int,
    ) -> list[ChunkData]:
        raw_splits = self._splitter.split_text(element.content.strip())
        splits = [s.strip() for s in raw_splits if s.strip()]

        chunks: list[ChunkData] = []
        for i, split_text in enumerate(splits):
            enriched = self._enrich_content(split_text, element)
            chunk = self._build_chunk(
                content=enriched,
                element=element,
                document_data=document_data,
                chunk_index=start_chunk_index + i,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_table(
        self,
        element: DocumentElement,
        document_data: DocumentData,
        start_chunk_index: int,
    ) -> list[ChunkData]:
        raw_content = element.content.strip()

        # Dedicated markdown table chunker check
        if self._is_markdown_table(raw_content):
            return self._split_markdown_table(element, document_data, start_chunk_index)

        if len(raw_content) <= self._chunk_size:
            enriched = self._enrich_content(raw_content, element)
            chunk = self._build_chunk(
                content=enriched,
                element=element,
                document_data=document_data,
                chunk_index=start_chunk_index,
            )
            return [chunk]

        raw_splits = self._splitter.split_text(raw_content)
        splits = [s.strip() for s in raw_splits if s.strip()]

        chunks: list[ChunkData] = []
        for i, split_text in enumerate(splits):
            enriched = self._enrich_content(split_text, element)
            chunk = self._build_chunk(
                content=enriched,
                element=element,
                document_data=document_data,
                chunk_index=start_chunk_index + i,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_list(
        self,
        element: DocumentElement,
        document_data: DocumentData,
        start_chunk_index: int,
    ) -> list[ChunkData]:
        raw_content = element.content.strip()

        if len(raw_content) <= self._chunk_size:
            enriched = self._enrich_content(raw_content, element)
            chunk = self._build_chunk(
                content=enriched,
                element=element,
                document_data=document_data,
                chunk_index=start_chunk_index,
            )
            return [chunk]

        raw_splits = self._splitter.split_text(raw_content)
        splits = [s.strip() for s in raw_splits if s.strip()]

        chunks: list[ChunkData] = []
        for i, split_text in enumerate(splits):
            enriched = self._enrich_content(split_text, element)
            chunk = self._build_chunk(
                content=enriched,
                element=element,
                document_data=document_data,
                chunk_index=start_chunk_index + i,
            )
            chunks.append(chunk)

        return chunks

    def _enrich_content(
        self,
        content: str,
        element: DocumentElement,
    ) -> str:
        header_lines: list[str] = []

        if element.section_path:
            header_lines.append(f"Section Path: {element.section_path}")

        if element.section_title:
            header_lines.append(f"Section: {element.section_title}")

        if not header_lines:
            return content

        header = "\n".join(header_lines)
        return f"{header}\n\n{content}"

    def _is_markdown_table(self, content: str) -> bool:
        """
        Detects if the given text is a markdown table.
        A markdown table has at least:
        - One header row starting and ending with '|'
        - One separator row containing only pipes, hyphens, colons, and spaces
        """
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        # Header row check
        if not (lines[0].startswith("|") and lines[0].endswith("|")):
            return False

        # Separator row check
        for line in lines[1:]:
            if _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(line):
                return True

        return False

    def _split_markdown_table(
        self,
        element: DocumentElement,
        document_data: DocumentData,
        start_chunk_index: int,
    ) -> list[ChunkData]:
        """
        Splits a markdown table row-by-row ensuring that each chunk includes
        the original header block and separator rows, and does not exceed chunk_size.
        """
        raw_content = element.content.strip()
        lines = [line.strip() for line in raw_content.splitlines() if line.strip()]

        # Locate the separator row
        sep_idx = -1
        for idx, line in enumerate(lines):
            if _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(line):
                sep_idx = idx
                break

        # Fallback if separator is malformed (should not occur if _is_markdown_table returned True)
        if sep_idx == -1 or sep_idx == 0 or sep_idx == len(lines) - 1:
            header_block = lines[0]
            separator_row = "|---|"
            body_rows = lines[1:]
            header_count = 1
        else:
            header_block = "\n".join(lines[:sep_idx])
            separator_row = lines[sep_idx]
            body_rows = lines[sep_idx + 1:]
            header_count = sep_idx

        # Log markdown table detection details
        logger.info(
            "Markdown table detected | page=%d | document=%s | total_rows=%d | header_rows=%d | body_rows=%d",
            element.page_number,
            document_data.document_name,
            len(lines),
            header_count,
            len(body_rows),
        )

        # If the entire table fits (and contains no individual oversized rows), return it in a single chunk
        if len(raw_content) <= self._chunk_size:
            enriched = self._enrich_content(raw_content, element)
            chunk = self._build_chunk(
                content=enriched,
                element=element,
                document_data=document_data,
                chunk_index=start_chunk_index,
            )
            created_chunks = [chunk]
        else:
            created_chunks = []
            current_group: list[str] = []
            current_len = len(header_block) + len(separator_row) + 2  # account for newlines
            chunk_idx = start_chunk_index

            for row in body_rows:
                row_len = len(row) + 1  # account for newline
                full_row_chunk_len = len(header_block) + len(separator_row) + 2 + len(row)

                # Check for oversized row
                if full_row_chunk_len > self._chunk_size:
                    # Flush the current group first
                    if current_group:
                        chunk_content = f"{header_block}\n{separator_row}\n" + "\n".join(current_group)
                        enriched = self._enrich_content(chunk_content, element)
                        created_chunks.append(self._build_chunk(
                            content=enriched,
                            element=element,
                            document_data=document_data,
                            chunk_index=chunk_idx,
                        ))
                        chunk_idx += 1
                        current_group = []
                        current_len = len(header_block) + len(separator_row) + 2

                    # Log warning for oversized row
                    logger.warning(
                        "Oversized table row detected | page=%d | document=%s | row_len=%d | chunk_size=%d",
                        element.page_number,
                        document_data.document_name,
                        len(row),
                        self._chunk_size,
                    )

                    # Place the oversized row into its own chunk
                    chunk_content = f"{header_block}\n{separator_row}\n{row}"
                    enriched = self._enrich_content(chunk_content, element)
                    created_chunks.append(self._build_chunk(
                        content=enriched,
                        element=element,
                        document_data=document_data,
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1
                    continue

                # If adding this row exceeds the chunk size limit, flush current group
                if len(current_group) > 0 and current_len + row_len > self._chunk_size:
                    chunk_content = f"{header_block}\n{separator_row}\n" + "\n".join(current_group)
                    enriched = self._enrich_content(chunk_content, element)
                    created_chunks.append(self._build_chunk(
                        content=enriched,
                        element=element,
                        document_data=document_data,
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1

                    # Start a new group with the current row
                    current_group = [row]
                    current_len = len(header_block) + len(separator_row) + 2 + len(row) + 1
                else:
                    current_group.append(row)
                    current_len += row_len

            # Flush the final group
            if current_group:
                chunk_content = f"{header_block}\n{separator_row}\n" + "\n".join(current_group)
                enriched = self._enrich_content(chunk_content, element)
                created_chunks.append(self._build_chunk(
                    content=enriched,
                    element=element,
                    document_data=document_data,
                    chunk_index=chunk_idx,
                ))

        # Log chunk statistics after chunking
        chunk_lengths = [len(c.content) for c in created_chunks]
        chunk_rows = [len(c.content.splitlines()) for c in created_chunks]
        avg_rows = sum(chunk_rows) / len(chunk_rows) if chunk_rows else 0.0
        max_chunk_size = max(chunk_lengths) if chunk_lengths else 0
        min_chunk_size = min(chunk_lengths) if chunk_lengths else 0

        logger.info(
            "Markdown table chunking complete | page=%d | document=%s | chunks_created=%d | avg_rows_per_chunk=%.1f | largest_chunk_chars=%d | smallest_chunk_chars=%d",
            element.page_number,
            document_data.document_name,
            len(created_chunks),
            avg_rows,
            max_chunk_size,
            min_chunk_size,
        )

        return created_chunks

    def _build_chunk(
        self,
        content: str,
        element: DocumentElement,
        document_data: DocumentData,
        chunk_index: int,
    ) -> ChunkData:
        chunk_id = _make_chunk_id(
            source_file=document_data.source_file,
            page_number=element.page_number,
            content=content,
        )

        from datetime import datetime, timezone
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_data.document_id,
            source_file=element.source_file,
            document_name=document_data.document_name,
            document_type=document_data.document_type,
            page_number=element.page_number,
            chunk_index=chunk_index,
            content_type=element.content_type,
            section_title=element.section_title,
            section_path=element.section_path,
            ingestion_timestamp=datetime.now(timezone.utc),
        )

        return ChunkData(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            token_estimate=self._estimate_tokens(content),
        )

    def _estimate_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def _log_chunk_statistics(
        self,
        document_name: str,
        total_elements: int,
        chunks: list[ChunkData],
        duplicate_chunks_skipped: int,
        tiny_elements_skipped: int,
        heading_elements_skipped: int,
        unique_chunks_created: int,
    ) -> None:
        if not chunks:
            logger.warning(
                "Chunking produced 0 chunks | document=%s | elements=%d | "
                "headings_skipped=%d | tiny_skipped=%d",
                document_name,
                total_elements,
                heading_elements_skipped,
                tiny_elements_skipped,
            )
            return

        lengths = [len(c.content) for c in chunks]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)

        logger.info(
            "Chunking complete | document=%s | "
            "total_elements=%d | "
            "heading_elements_skipped=%d | "
            "tiny_elements_skipped=%d | "
            "unique_chunks_created=%d | "
            "duplicate_chunks_skipped=%d | "
            "avg_chars=%.1f | max_chars=%d | min_chars=%d",
            document_name,
            total_elements,
            heading_elements_skipped,
            tiny_elements_skipped,
            unique_chunks_created,
            duplicate_chunks_skipped,
            avg_length,
            max_length,
            min_length,
        )


_module_chunker = DocumentChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


def chunk_document(document_data: DocumentData) -> list[ChunkData]:
    return _module_chunker.chunk_document(document_data)