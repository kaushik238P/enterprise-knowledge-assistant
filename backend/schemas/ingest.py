from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str = Field(
        ...,
        description="Name of the ingested document."
    )

    chunk_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks created."
    )

    stored_vectors: int = Field(
        ...,
        ge=0,
        description="Number of vectors stored in Qdrant."
    )

    status: Literal[
        "success",
        "failed",
    ] = Field(
        ...,
        description="Ingestion status."
    )