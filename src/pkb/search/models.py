"""Search query and result models."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SearchMode(str, Enum):
    """Search mode selection."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchQuery(BaseModel):
    """Search query parameters."""

    query: str = Field(min_length=1)
    mode: SearchMode = SearchMode.HYBRID
    domains: list[str] = []
    topics: list[str] = []
    kb: str | None = None
    after: date | None = None
    before: date | None = None
    limit: int = Field(default=10, gt=0)
    stance: str | None = None
    has_consensus: bool | None = None
    has_synthesis: bool | None = None


class BundleSearchResult(BaseModel):
    """A single search result representing a matched bundle."""

    bundle_id: str
    question: str
    summary: str | None
    domains: list[str]
    topics: list[str]
    score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    source: str  # "fts", "semantic", "both"

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ("fts", "semantic", "both"):
            raise ValueError(f"source must be 'fts', 'semantic', or 'both', got '{v}'")
        return v
