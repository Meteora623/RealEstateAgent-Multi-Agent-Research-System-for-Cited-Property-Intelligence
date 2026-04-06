from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    property_id: str | None = None
    city: str | None = None
    neighborhood: str | None = None
    property_type: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    building_class: str | None = None
    min_year: int | None = None
    max_year: int | None = None

    def as_match_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.property_id:
            payload["property_id"] = self.property_id
        if self.city:
            payload["city"] = self.city
        if self.neighborhood:
            payload["neighborhood"] = self.neighborhood
        if self.property_type:
            payload["property_type"] = self.property_type
        if self.building_class:
            payload["building_class"] = self.building_class
        return payload


class RetrievalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    property_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    doc_id: str
    property_id: str
    title: str | None = None
    source: str | None = None
    snippet: str


class RetrieverDebugInfo(BaseModel):
    retrieval_mode: str
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    rerank_applied: bool = False
    reranker_provider: str = "none"
    filter_applied: dict[str, Any] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    dense_rankings: list[dict[str, Any]] = Field(default_factory=list)
    sparse_rankings: list[dict[str, Any]] = Field(default_factory=list)
    fused_rankings: list[dict[str, Any]] = Field(default_factory=list)
    reranked_rankings: list[dict[str, Any]] = Field(default_factory=list)
