from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime
from uuid import UUID
import sys
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ContentTypeLiteral = Literal["text", "table", "heading", "list"]
DocumentTypeLiteral = Literal["pdf", "txt", "docx", "html", "md"]

__all__ = [
    "DocumentElement",
    "DocumentData",
    "ChunkMetadata",
    "ChunkData",
    "ContentTypeLiteral",
    "DocumentTypeLiteral",
    "normalize_source_path",
    "clean_document_type",
    "get_element_statistics",
    "get_citation",
    "to_qdrant_payload",
]


class DocumentElement(BaseModel):
    content: str = Field(..., description="The raw text or structured string representation of the element.")
    source_file: str = Field(..., description="Normalized POSIX path or identifier of the source file.")
    page_number: int = Field(..., ge=1, description="The 1-indexed page number where this element resides.")
    content_type: ContentTypeLiteral = Field(..., description="The structural type of the element.")
    section_title: Optional[str] = Field(default=None, description="The title of the immediate section/heading.")
    section_path: Optional[str] = Field(default=None, description="The full hierarchical heading path to this element.")

    # Optional table metadata fields
    table_category: Optional[str] = Field(default=None, description="Financial statement type (P&L, Balance Sheet, Cash Flow, Ratio, etc.)")
    table_title: Optional[str] = Field(default=None, description="The title or header of the table if available.")
    table_id: Optional[str] = Field(default=None, description="Unique identifier for the table.")
    table_rows: Optional[int] = Field(default=None, description="Number of rows in the table.")
    table_columns: Optional[int] = Field(default=None, description="Number of columns in the table.")
    table_unit: Optional[str] = Field(default=None, description="Reporting unit (Crores, Millions, Thousands, etc.)")
    table_currency: Optional[str] = Field(default=None, description="Reporting currency (Rupees, USD, etc.)")
    semantic_fact_count: Optional[int] = Field(default=None, description="Number of semantic facts generated from the table.")

    def __init__(self, **data: Any) -> None:
        
        curr_frame = sys._getframe(1)
        parsed_table = None
        facts = None
        while curr_frame:
            locals_dict = curr_frame.f_locals
            if "parsed_table" in locals_dict:
                parsed_table = locals_dict["parsed_table"]
            if "facts" in locals_dict:
                facts = locals_dict["facts"]
            if parsed_table and facts:
                break
            curr_frame = curr_frame.f_back

        if parsed_table is not None:
            if data.get("table_rows") is None:
                data["table_rows"] = len(parsed_table.rows)
            if data.get("table_columns") is None:
                data["table_columns"] = len(parsed_table.headers)
            
            if facts:
                if data.get("table_category") is None:
                    data["table_category"] = facts[0].table_category if hasattr(facts[0], "table_category") else "Generic Table"
                if data.get("table_unit") is None:
                    for f in facts:
                        if hasattr(f, "detected_unit") and f.detected_unit is not None:
                            data["table_unit"] = f.detected_unit
                            break
                if data.get("semantic_fact_count") is None:
                    data["semantic_fact_count"] = len(facts)
                if data.get("table_currency") is None:
                    has_rupees = False
                    for f in facts:
                        if hasattr(f, "detected_unit") and f.detected_unit == "Rupees":
                            has_rupees = True
                            break
                        if hasattr(f, "value") and ("₹" in str(f.value) or "Rs" in str(f.value)):
                            has_rupees = True
                            break
                    if has_rupees:
                        data["table_currency"] = "Rupees"
        super().__init__(**data)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "content": "This is a paragraph discussing enterprise architecture.",
                "source_file": "docs/architecture_guide.pdf",
                "page_number": 3,
                "content_type": "text",
                "section_title": "1. Introduction",
                "section_path": "Part I > Chapter 1 > 1. Introduction",
            }
        },
    )

    @field_validator("source_file", mode="before")
    @classmethod
    def normalize_source_path(cls, v: Union[str, Path]) -> str:
        if v is None:
            raise ValueError("source_file cannot be empty.")
        value = str(v).strip()
        if not value:
            raise ValueError("source_file cannot be empty.")
        return str(Path(value).as_posix())


class DocumentData(BaseModel):
    document_id: UUID = Field(..., description="Unique identifier for the parsed document.")
    document_name: str = Field(..., description="The file name of the document.")
    document_type: DocumentTypeLiteral = Field(..., description="The format of the source document.")
    source_file: str = Field(..., description="Normalized POSIX path or resource identifier of the source file.")
    total_pages: int = Field(..., ge=0, description="Total page count of the parsed document.")
    elements: List[DocumentElement] = Field(default_factory=list, description="Ordered list of all raw elements.")

    text_elements: int = Field(default=0, ge=0, description="Count of plain text/paragraph elements extracted.")
    table_elements: int = Field(default=0, ge=0, description="Count of table elements extracted.")
    heading_elements: int = Field(default=0, ge=0, description="Count of heading/title elements extracted.")
    list_elements: int = Field(default=0, ge=0, description="Count of list item elements extracted.")

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    @field_validator("source_file", mode="before")
    @classmethod
    def normalize_source_path(cls, v: Union[str, Path]) -> str:
        if v is None:
            raise ValueError("source_file cannot be empty.")
        value = str(v).strip()
        if not value:
            raise ValueError("source_file cannot be empty.")
        return str(Path(value).as_posix())

    @field_validator("document_type", mode="before")
    @classmethod
    def clean_document_type(cls, v: Any) -> str:
        if v is None:
            raise ValueError("document_type cannot be empty.")
        value = str(v).strip().lower().lstrip(".")
        if not value:
            raise ValueError("document_type cannot be empty.")
        return value

    def get_element_statistics(self) -> Dict[str, int]:
        return {
            "text": self.text_elements,
            "table": self.table_elements,
            "heading": self.heading_elements,
            "list": self.list_elements,
        }


class ChunkMetadata(BaseModel):
    chunk_id: UUID = Field(..., description="Deterministic chunk identifier.")
    document_id: UUID = Field(..., description="Unique identifier of the parent parsed document.")
    source_file: str = Field(..., description="Normalized POSIX path or identifier of the source file.")
    document_name: str = Field(..., description="The file name of the source document.")
    document_type: DocumentTypeLiteral = Field(..., description="The format of the source document.")
    page_number: int = Field(..., ge=1, description="The 1-indexed page number of the source document.")
    chunk_index: int = Field(..., ge=0, description="Sequential 0-indexed position of this chunk within its document.")
    content_type: ContentTypeLiteral = Field(..., description="The structural content type of this chunk.")
    section_title: Optional[str] = Field(default=None, description="The title of the closest parent section or heading.")
    section_path: Optional[str] = Field(default=None, description="The full hierarchical heading path to this chunk.")
    content_length: Optional[int] = Field(default=None, ge=0, description="Length of the chunk content in characters.")
    parent_chunk_id: Optional[UUID] = Field(default=None, description="Optional parent chunk identifier for hierarchical retrieval.")
    source_hash: Optional[str] = Field(default=None, description="Optional stable hash of the source document content.")
    document_version: int = Field(default=1, ge=1, description="Version of the source document.")
    ingestion_timestamp: Optional[datetime] = Field(default=None, description="Timestamp when the chunk was ingested.")
    retrieval_score: Optional[float] = Field(default=None, description="Optional retrieval score attached at query time.")

    # Optional table metadata fields
    table_category: Optional[str] = Field(default=None, description="Financial statement type (P&L, Balance Sheet, Cash Flow, Ratio, etc.)")
    table_title: Optional[str] = Field(default=None, description="The title or header of the table if available.")
    table_id: Optional[str] = Field(default=None, description="Unique identifier for the table.")
    table_rows: Optional[int] = Field(default=None, description="Number of rows in the table.")
    table_columns: Optional[int] = Field(default=None, description="Number of columns in the table.")
    table_unit: Optional[str] = Field(default=None, description="Reporting unit (Crores, Millions, Thousands, etc.)")
    table_currency: Optional[str] = Field(default=None, description="Reporting currency (Rupees, USD, etc.)")
    semantic_fact_count: Optional[int] = Field(default=None, description="Number of semantic facts generated from the table.")

    def __init__(self, **data: Any) -> None:
        import sys
        curr_frame = sys._getframe(1)
        element = None
        while curr_frame:
            local_val = curr_frame.f_locals.get("element")
            if local_val and (hasattr(local_val, "table_category") or type(local_val).__name__ == "DocumentElement"):
                element = local_val
                break
            curr_frame = curr_frame.f_back

        if element is not None:
            for field in [
                "table_category", "table_title", "table_id", "table_rows",
                "table_columns", "table_unit", "table_currency", "semantic_fact_count"
            ]:
                if data.get(field) is None:
                    val = getattr(element, field, None)
                    if val is not None:
                        data[field] = val
        super().__init__(**data)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "chunk_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                "document_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                "source_file": "docs/architecture_guide.pdf",
                "document_name": "architecture_guide.pdf",
                "document_type": "pdf",
                "page_number": 3,
                "chunk_index": 4,
                "content_type": "text",
                "section_title": "1. Introduction",
                "section_path": "Part I > Chapter 1 > 1. Introduction",
            }
        },
    )

    @field_validator("source_file", mode="before")
    @classmethod
    def normalize_source_path(cls, v: Union[str, Path]) -> str:
        if v is None:
            raise ValueError("source_file cannot be empty.")
        value = str(v).strip()
        if not value:
            raise ValueError("source_file cannot be empty.")
        return str(Path(value).as_posix())

    @field_validator("document_type", mode="before")
    @classmethod
    def clean_document_type(cls, v: Any) -> str:
        if v is None:
            raise ValueError("document_type cannot be empty.")
        value = str(v).strip().lower().lstrip(".")
        if not value:
            raise ValueError("document_type cannot be empty.")
        return value

    def get_citation(self) -> str:
        citation = f"{self.document_name}, p. {self.page_number}"
        if self.section_path:
            citation += f" ({self.section_path})"
        elif self.section_title:
            citation += f" ({self.section_title})"
        return citation


class ChunkData(BaseModel):
    chunk_id: UUID = Field(..., description="Unique identifier matching the metadata's chunk_id.")
    content: str = Field(..., description="The textual content of the chunk.")
    metadata: ChunkMetadata = Field(..., description="Rich metadata associated with this chunk.")
    token_estimate: int = Field(..., ge=0, description="Estimated token count of the chunk content.")

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "chunk_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                "content": "This is a paragraph discussing enterprise architecture.",
                "token_estimate": 8,
                "metadata": {
                    "chunk_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                    "document_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                    "source_file": "docs/architecture_guide.pdf",
                    "document_name": "architecture_guide.pdf",
                    "document_type": "pdf",
                    "page_number": 3,
                    "chunk_index": 4,
                    "content_type": "text",
                    "section_title": "1. Introduction",
                    "section_path": "Part I > Chapter 1 > 1. Introduction",
                },
            }
        },
    )

    @model_validator(mode="after")
    def validate_chunk_integrity(self):
        if self.chunk_id != self.metadata.chunk_id:
            raise ValueError(
                f"ChunkData chunk_id '{self.chunk_id}' must match ChunkMetadata chunk_id '{self.metadata.chunk_id}'"
            )

        expected_content_length = len(self.content)
        if self.metadata.content_length is not None and self.metadata.content_length != expected_content_length:
            raise ValueError(
                f"ChunkMetadata content_length '{self.metadata.content_length}' must match len(content) '{expected_content_length}'"
            )

        if self.metadata.content_length is None:
            self.metadata.content_length = expected_content_length

        return self

    def to_qdrant_payload(self) -> Dict[str, Any]:
        payload = self.metadata.model_dump(mode="json", exclude_none=True)
        payload["content"] = self.content
        payload["token_estimate"] = self.token_estimate
        return payload

    def to_langchain_document(self) -> Any:
        try:
            from langchain_core.documents import Document
        except ImportError as e:
            raise ImportError(
                "langchain-core must be installed to convert to LangChain Documents. "
                "Install it using `uv add langchain-core` or `pip install langchain-core`."
            ) from e

        metadata = self.metadata.model_dump(mode="json", exclude_none=True)

        return Document(
            page_content=self.content,
            metadata=metadata,
        )