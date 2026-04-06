from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationQuery(BaseModel):
    query_id: str
    query: str
    reference_answer: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_property_ids: list[str] = Field(default_factory=list)
    expected_doc_types: list[str] = Field(default_factory=list)
    difficulty: str = "medium"


class ConfigEvaluationResult(BaseModel):
    config_name: str
    retrieval_mode: str
    rerank: bool
    semantic_chunking: bool
    chunking_strategy: str = "fixed"
    top_k: int = 0
    candidate_pool_size: int = 0
    fusion_alpha: float = 0.0
    metric_source: str = "unknown"
    ragas_metrics: dict[str, float | None] = Field(default_factory=dict)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    latency_metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EvaluationRunSummary(BaseModel):
    run_dir: str
    scale_metrics: dict[str, float | int]
    config_results: list[ConfigEvaluationResult]
    best_configuration: str
