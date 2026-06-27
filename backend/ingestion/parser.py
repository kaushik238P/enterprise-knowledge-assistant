import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Union

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel

from backend.config.settings import settings
from backend.ingestion.metadata import (
    ContentTypeLiteral,
    DocumentData,
    DocumentElement,
)
from backend.ingestion.table_parser import FinancialTableParser, generate_table_facts

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentParser",
    "parse_document",
    "parser_configuration",
    "SUPPORTED_DOCUMENT_TYPES",
    "DOCLING_LABEL_MAP",
    "DocumentData",
    "DocumentElement",
    "ContentTypeLiteral",
    "DocItemLabel",
]
SUPPORTED_DOCUMENT_TYPES: frozenset[str] = frozenset(settings.supported_document_types)

_MD_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
_TXT_PARAGRAPH_SEPARATOR = re.compile(r"\n{2,}")

DOCLING_LABEL_MAP: dict[DocItemLabel, ContentTypeLiteral] = {
    DocItemLabel.TEXT: "text",
    DocItemLabel.PARAGRAPH: "text",
    DocItemLabel.TABLE: "table",
    DocItemLabel.SECTION_HEADER: "heading",
    DocItemLabel.TITLE: "heading",
    DocItemLabel.LIST_ITEM: "list",
}


def _make_document_id(file_path: Path, document_name: Optional[str] = None) -> uuid.UUID:
    if document_name:
        return uuid.uuid5(uuid.NAMESPACE_URL, document_name)
    return uuid.uuid5(uuid.NAMESPACE_URL, str(file_path.resolve()))


def parser_configuration() -> dict[str, bool]:
    return {
        "ocr_enabled": settings.ocr_enabled,
        "table_extraction_enabled": settings.table_extraction_enabled,
    }


class DocumentParser:
    def __init__(self, enable_table_extraction: bool = settings.table_extraction_enabled) -> None:
        self._enable_table_extraction = enable_table_extraction
        self._converter = self._build_converter()
        self._table_parser = FinancialTableParser()

        logger.info(
            "Docling configured | OCR=%s | Table Extraction=%s",
            settings.ocr_enabled,
            enable_table_extraction,
        )
        logger.info(
            "DocumentParser initialized | table_extraction=%s | ocr=%s",
            enable_table_extraction,
            settings.ocr_enabled,
        )

    def _build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions(
            do_ocr=settings.ocr_enabled,
            do_picture_classification=False,
            do_picture_description=False,
            do_chart_extraction=False,
            generate_page_images=False,
            generate_picture_images=False,
            do_table_structure=self._enable_table_extraction,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

        logger.info(
            "DocumentConverter created | do_ocr=%s | do_table_structure=%s",
            settings.ocr_enabled,
            self._enable_table_extraction,
        )

        return converter

    def parse(self, file_path: Union[str, Path], document_name: Optional[str] = None) -> DocumentData:
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_size = file_path.stat().st_size
        max_file_size_bytes = settings.max_document_size_mb * 1024 * 1024
        if file_size > max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            raise ValueError(
                f"Document exceeds supported ingestion size limit. "
                f"File '{file_path.name}' is {size_mb:.1f} MB; "
                f"maximum allowed is {settings.max_document_size_mb} MB."
            )

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_DOCUMENT_TYPES:
            supported = ", ".join(sorted(SUPPORTED_DOCUMENT_TYPES))
            raise ValueError(
                f"Unsupported file format '{extension}' for file '{file_path.name}'. "
                f"Currently supported formats are: {supported}."
            )

        logger.info(
            "Routing document | file=%s | format=%s | size=%.2f KB",
            file_path.name,
            extension,
            file_size / 1024,
        )

        if extension == ".pdf":
            return self._parse_pdf(file_path, document_name=document_name)

        if extension == ".txt":
            return self._parse_txt(file_path, document_name=document_name)

        if extension == ".md":
            return self._parse_markdown(file_path, document_name=document_name)

        raise ValueError(f"No parser implemented for format '{extension}'.")

    def _parse_pdf(self, file_path: Path, document_name: Optional[str] = None) -> DocumentData:
        logger.info(
            "Starting PDF parse | file=%s | OCR=%s | tables=%s",
            file_path.name,
            settings.ocr_enabled,
            self._enable_table_extraction,
        )
        start_time = time.perf_counter()

        try:
            result = self._converter.convert(str(file_path))
        except Exception as exc:
            logger.exception(
                "Docling conversion failed | file=%s | error=%s",
                file_path.name,
                exc,
            )
            raise RuntimeError(
                f"Docling failed to convert '{file_path.name}': {exc}"
            ) from exc

        docling_doc: DoclingDocument = result.document

        elements = self._extract_elements(
            docling_doc=docling_doc,
            source_file=str(file_path),
        )


        total_pages = self._resolve_total_pages(docling_doc=docling_doc)

        logger.warning(
            "Document returned 0 extracted elements | "
            "file=%s | pages=%d | ocr_enabled=%s",
            file_path.name,
            total_pages,
            settings.ocr_enabled,
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "PDF parse complete | file=%s | pages=%d | elements=%d | duration=%.2fs",
            file_path.name,
            total_pages,
            len(elements),
            elapsed,
        )

        element_stats = self._compute_element_statistics(elements)
        doc_stats = self._compute_document_statistics(elements)
        logger.info(
            "Document health | file=%s | total_chars=%d | avg_element_length=%.1f | empty_skipped=%d",
            file_path.name,
            doc_stats["total_characters"],
            doc_stats["average_element_length"],
            doc_stats["empty_elements_skipped"],
        )

        return DocumentData(
            document_id=_make_document_id(file_path, document_name=document_name),
            document_name=document_name if document_name else file_path.name,
            document_type="pdf",
            source_file=str(file_path),
            total_pages=total_pages,
            elements=elements,
            **element_stats,
        )

    def _parse_txt(self, file_path: Path, document_name: Optional[str] = None) -> DocumentData:
        logger.info("Starting TXT parse | file=%s", file_path.name)
        start_time = time.perf_counter()

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            logger.exception(
                "UTF-8 decode failed | file=%s | error=%s",
                file_path.name,
                exc,
            )
            raise RuntimeError(
                f"Failed to read '{file_path.name}' as UTF-8 text."
            ) from exc

        raw_paragraphs = _TXT_PARAGRAPH_SEPARATOR.split(raw_text)
        elements: list[DocumentElement] = []
        empty_skipped = 0

        for paragraph in raw_paragraphs:
            content = paragraph.strip()
            if not content:
                empty_skipped += 1
                continue
            elements.append(DocumentElement(
                content=content,
                source_file=str(file_path),
                page_number=1,
                content_type="text",
                section_title=None,
                section_path=None,
            ))

        elapsed = time.perf_counter() - start_time
        logger.info(
            "TXT parse complete | file=%s | paragraphs=%d | empty_skipped=%d | duration=%.2fs",
            file_path.name,
            len(elements),
            empty_skipped,
            elapsed,
        )

        element_stats = self._compute_element_statistics(elements)
        doc_stats = self._compute_document_statistics(elements, empty_skipped=empty_skipped)
        logger.info(
            "Document health | file=%s | total_chars=%d | avg_element_length=%.1f | empty_skipped=%d",
            file_path.name,
            doc_stats["total_characters"],
            doc_stats["average_element_length"],
            doc_stats["empty_elements_skipped"],
        )

        return DocumentData(
            document_id=_make_document_id(file_path, document_name=document_name),
            document_name=document_name if document_name else file_path.name,
            document_type="txt",
            source_file=str(file_path),
            total_pages=1,
            elements=elements,
            **element_stats,
        )

    def _parse_markdown(self, file_path: Path, document_name: Optional[str] = None) -> DocumentData:
        logger.info("Starting Markdown parse | file=%s", file_path.name)
        start_time = time.perf_counter()

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            logger.exception(
                "UTF-8 decode failed | file=%s | error=%s",
                file_path.name,
                exc,
            )
            raise RuntimeError(
                f"Failed to read '{file_path.name}' as UTF-8 Markdown."
            ) from exc

        lines = raw_text.splitlines()
        elements: list[DocumentElement] = []
        empty_skipped = 0

        current_section_title: Optional[str] = None
        section_path_parts: list[str] = []
        pending_text_lines: list[str] = []

        def flush_text_buffer() -> None:
            nonlocal pending_text_lines, empty_skipped
            content = " ".join(pending_text_lines).strip()
            pending_text_lines = []
            if not content:
                empty_skipped += 1
                return
            elements.append(DocumentElement(
                content=content,
                source_file=str(file_path),
                page_number=1,
                content_type="text",
                section_title=current_section_title,
                section_path=" > ".join(section_path_parts) if section_path_parts else None,
            ))

        for line in lines:
            heading_match = _MD_HEADING_PATTERN.match(line)
            if heading_match:
                flush_text_buffer()
                hashes, heading_text = heading_match.group(1), heading_match.group(2).strip()
                heading_level = len(hashes)
                depth = heading_level - 1
                section_path_parts = section_path_parts[:depth]
                section_path_parts.append(heading_text)
                current_section_title = heading_text
                section_path = " > ".join(section_path_parts) if section_path_parts else None
                elements.append(DocumentElement(
                    content=heading_text,
                    source_file=str(file_path),
                    page_number=1,
                    content_type="heading",
                    section_title=current_section_title,
                    section_path=section_path,
                ))
            else:
                stripped = line.strip()
                if stripped:
                    pending_text_lines.append(stripped)
                else:
                    flush_text_buffer()

        flush_text_buffer()

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Markdown parse complete | file=%s | elements=%d | empty_skipped=%d | duration=%.2fs",
            file_path.name,
            len(elements),
            empty_skipped,
            elapsed,
        )

        element_stats = self._compute_element_statistics(elements)
        doc_stats = self._compute_document_statistics(elements, empty_skipped=empty_skipped)
        logger.info(
            "Document health | file=%s | total_chars=%d | avg_element_length=%.1f | empty_skipped=%d",
            file_path.name,
            doc_stats["total_characters"],
            doc_stats["average_element_length"],
            doc_stats["empty_elements_skipped"],
        )

        return DocumentData(
            document_id=_make_document_id(file_path, document_name=document_name),
            document_name=document_name if document_name else file_path.name,
            document_type="md",
            source_file=str(file_path),
            total_pages=1,
            elements=elements,
            **element_stats,
        )

    def _extract_elements(
        self,
        docling_doc: DoclingDocument,
        source_file: str,
    ) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        current_section_title: Optional[str] = None
        section_path_parts: list[str] = []

        for item, _ in docling_doc.iterate_items():
            content_type = self._resolve_content_type(item)
            if content_type is None:
                continue

            page_number = self._resolve_page_number(item)

            if content_type == "heading":
                heading_text = self._safe_text(item, docling_doc)
                current_section_title = heading_text
                section_path_parts = self._update_section_path(
                    section_path_parts, heading_text, item
                )

            section_path = " > ".join(section_path_parts) if section_path_parts else None

            if content_type == "table":
                original_markdown = self._extract_table_content(item, docling_doc)
                if not original_markdown or not original_markdown.strip():
                    continue

                try:
                    parsed_table = self._table_parser.parse(original_markdown)
                    facts = generate_table_facts(parsed_table)
                    reconstructed_markdown = parsed_table.to_markdown()

                    num_rows = len(parsed_table.rows)
                    num_cols = len(parsed_table.headers)
                    table_category = "Generic Table"
                    if facts:
                        table_category = facts[0].table_category

                    logger.info(
                        "Parsed financial table | page=%d | rows=%d | columns=%d | facts_generated=%d | category=%s",
                        page_number,
                        num_rows,
                        num_cols,
                        len(facts),
                        table_category,
                    )

                    # Build Content for Element A (Table Element)
                    content_parts = []
                    if section_path:
                        content_parts.append(f"Section Context: {section_path}")
                    elif current_section_title:
                        content_parts.append(f"Section Context: {current_section_title}")
                    
                    content_parts.append(reconstructed_markdown)
                    content_parts.append("Original Table Representation:")
                    content_parts.append(original_markdown)

                    element_a_content = "\n\n".join(content_parts)
                    elements.append(DocumentElement(
                        content=element_a_content,
                        source_file=source_file,
                        page_number=page_number,
                        content_type="table",
                        section_title=current_section_title,
                        section_path=section_path,
                    ))

                    # Build Content for Element B (Semantic Text Element)
                    facts_text = "\n\n".join(fact.sentence for fact in facts if fact.sentence)
                    if facts_text:
                        elements.append(DocumentElement(
                            content=facts_text,
                            source_file=source_file,
                            page_number=page_number,
                            content_type="text",
                            section_title=current_section_title,
                            section_path=section_path,
                        ))

                except Exception as exc:
                    logger.exception(
                        "Failed to parse table | page=%d | doc=%s | falling back to original extraction | error=%s",
                        page_number,
                        Path(source_file).name,
                        exc,
                    )
                    elements.append(DocumentElement(
                        content=original_markdown,
                        source_file=source_file,
                        page_number=page_number,
                        content_type="table",
                        section_title=current_section_title,
                        section_path=section_path,
                    ))
                continue

            content = self._safe_text(item, docling_doc)

            if not content or not content.strip():
                continue

            elements.append(DocumentElement(
                content=content,
                source_file=source_file,
                page_number=page_number,
                content_type=content_type,
                section_title=current_section_title,
                section_path=section_path,
            ))

        logger.info("Extracted %d elements total.", len(elements))
        return elements

    def _compute_element_statistics(
        self,
        elements: list[DocumentElement],
    ) -> dict[str, int]:
        stats: dict[str, int] = {
            "text_elements": 0,
            "table_elements": 0,
            "heading_elements": 0,
            "list_elements": 0,
        }
        for element in elements:
            if element.content_type == "text":
                stats["text_elements"] += 1
            elif element.content_type == "table":
                stats["table_elements"] += 1
            elif element.content_type == "heading":
                stats["heading_elements"] += 1
            elif element.content_type == "list":
                stats["list_elements"] += 1

        return stats

    def _compute_document_statistics(
        self,
        elements: list[DocumentElement],
        empty_skipped: int = 0,
    ) -> dict[str, int | float]:
        total_chars = sum(len(e.content) for e in elements)
        avg_length = (total_chars / len(elements)) if elements else 0.0
        return {
            "total_characters": total_chars,
            "average_element_length": round(avg_length, 2),
            "empty_elements_skipped": empty_skipped,
        }

    def _resolve_content_type(self, item) -> Optional[ContentTypeLiteral]:
        label = getattr(item, "label", None)
        if label is None:
            return None
        return DOCLING_LABEL_MAP.get(label, None)

    def _resolve_page_number(self, item) -> int:
        try:
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                return int(prov[0].page_no)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            logger.debug("Could not resolve page number: %s", exc)
        return 1

    def _safe_text(
        self,
        item,
        docling_doc: DoclingDocument | None = None,
    ) -> str:
        try:
            text = getattr(item, "text", None)
            if text:
                return str(text).strip()
            export = getattr(item, "export_to_markdown", None)
            if callable(export) and docling_doc is not None:
                return export(doc=docling_doc).strip()
        except Exception as exc:
            logger.debug("Failed to extract text from item: %s", exc)
        return ""

    def _extract_table_content(self, item, docling_doc: DoclingDocument) -> str:
        try:
            export = getattr(item, "export_to_markdown", None)
            if callable(export):
                return export(doc=docling_doc).strip()
        except Exception as exc:
            logger.debug("Markdown table export failed, falling back to text: %s", exc)
        return self._safe_text(item, docling_doc)

    def _update_section_path(
        self,
        current_parts: list[str],
        heading_text: str,
        item,
    ) -> list[str]:
        level = getattr(item, "level", None)
        if level is not None:
            try:
                depth = max(0, int(level) - 1)
                truncated = current_parts[:depth]
                truncated.append(heading_text)
                return truncated
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "item.level not coercible to int | value=%s | error=%s. "
                    "Falling back to flat section replacement.",
                    level,
                    exc,
                )
        if current_parts:
            return current_parts[:-1] + [heading_text]
        return [heading_text]

    def _resolve_total_pages(self, docling_doc: DoclingDocument) -> int:
        try:
            pages = getattr(docling_doc, "pages", None)
            if pages:
                return len(pages)
        except Exception as exc:
            logger.debug("Failed to resolve total pages: %s", exc)
        return 0


_module_parser = DocumentParser()


def parse_document(
    file_path: Union[str, Path],
    document_name: Optional[str] = None,
) -> DocumentData:
    return _module_parser.parse(file_path, document_name=document_name)