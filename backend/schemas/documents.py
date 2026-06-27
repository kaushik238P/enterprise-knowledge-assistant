# backend/schemas/documents.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class DocumentInfo(BaseModel):
    document_id: str = Field(..., description="Unique UUID of the document")
    document_name: str = Field(..., description="Original name of the document")
    chunk_count: int = Field(..., description="Total number of chunks")
    page_count: int = Field(..., description="Total number of pages")
    ingestion_timestamp: Optional[datetime] = Field(default=None, description="Ingestion timestamp if available")

class DeleteDocumentResponse(BaseModel):
    status: str = Field(..., description="Deletion status, e.g. success")
    document_id: str = Field(..., description="UUID of the deleted document")
    document_name: str = Field(..., description="Original name of the deleted document")
    chunks_deleted: int = Field(..., description="Number of deleted chunks")
    elapsed_ms: int = Field(..., description="Time taken to delete in milliseconds")

class DeletionResult(BaseModel):
    status: str
    document_id: str
    document_name: str
    chunks_deleted: int
    elapsed_ms: int
