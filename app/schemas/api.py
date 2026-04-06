from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.agent import FinalAnswer, IntentType
from app.schemas.retrieval import MetadataFilter, RetrieverDebugInfo


class QueryRequest(BaseModel):
    query: str = Field(min_length=3)
    filters: MetadataFilter | None = None
    top_k: int | None = None
    candidate_pool_size: int | None = None
    rerank: bool | None = None
    rerank_top_n: int | None = None
    retrieval_mode: str | None = None
    fusion_alpha: float | None = None
    async_tools: bool = True
    include_debug: bool = False


class QueryResponse(BaseModel):
    request_id: str
    intent: IntentType
    concise_answer: str
    reasoning: list[str]
    citations: list[dict]
    confidence_level: float
    unsupported_or_uncertain: list[str]
    retrieval_debug: RetrieverDebugInfo | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class StructuredQueryResponse(BaseModel):
    request_id: str
    intent: IntentType
    answer: FinalAnswer
    retrieval_debug: RetrieverDebugInfo | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace_events: list[dict] = Field(default_factory=list)


class IngestRequest(BaseModel):
    semantic_chunking: bool | None = None
    chunking_strategy: str | None = None
    dataset_profile: str | None = None
    force_reindex: bool = False


class EvaluateRequest(BaseModel):
    run_ragas: bool = True
    ragas_mode: str | None = None
    dataset_profile: str | None = None
    max_queries: int | None = None
    output_tag: str | None = None


class EvaluateResponse(BaseModel):
    run_dir: str
    best_configuration: str
    corpus_stats_path: str
    quality_metrics_path: str
    latency_metrics_path: str
    comparison_table_path: str
    resume_claim_report_path: str
    report_path: str
    experiment_metadata: dict = Field(default_factory=dict)
