from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PropertyDocument(BaseModel):
    doc_id: str
    property_id: str
    doc_type: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    property_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionStats(BaseModel):
    total_documents: int
    total_chunks: int
    dense_index_backend: str
    bm25_documents: int
    semantic_chunking: bool
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    metadata_fields_used: list[str]
    average_chunk_length: float = 0.0
    median_chunk_length: float = 0.0
    p95_chunk_length: float = 0.0
    chunks_by_doc_type: dict[str, int] = Field(default_factory=dict)
    metadata_coverage_pct: dict[str, float] = Field(default_factory=dict)
    citation_map_integrity: dict[str, int | bool] = Field(default_factory=dict)
