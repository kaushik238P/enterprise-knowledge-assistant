from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="The query string to search for.",
        min_length=1,
    )
    top_k: Optional[int] = Field(
        None,
        description="The maximum number of search results to return. Must be greater than 0 if provided.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_and_strip_query(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("query must be a string.")

        v = v.strip()

        if not v:
            raise ValueError(
                "Query cannot be empty or only whitespace."
            )

        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("top_k must be greater than 0.")
        return v


class SearchResultItem(BaseModel):
    chunk_id: str = Field(
        ...,
        description="The unique identifier of the document chunk."
    )
    score: float = Field(
        ...,
        
        description="The similarity search score of the result."
    )
    content: str = Field(
        ...,
        description="The text content of the document chunk."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dictionary associated with the chunk."
    )
    


class SearchResponse(BaseModel):
    results: list[SearchResultItem] = Field(
        ...,
        description="List of search result items matching the query."
    )