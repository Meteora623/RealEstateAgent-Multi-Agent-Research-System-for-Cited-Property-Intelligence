from __future__ import annotations

from typing import TypedDict

from app.schemas.agent import AnalystDraft, FactCheckReport, FinalAnswer, IntentType
from app.schemas.retrieval import MetadataFilter, RetrievalChunk, RetrieverDebugInfo


class GraphState(TypedDict, total=False):
    request_id: str
    query: str
    filters: MetadataFilter | None
    top_k: int
    candidate_pool_size: int | None
    rerank: bool
    rerank_top_n: int | None
    retrieval_mode: str
    fusion_alpha: float | None
    async_tools: bool
    intent: IntentType
    plan: list[str]
    retrieval_round: int
    retrieved_chunks: list[RetrievalChunk]
    retrieval_debug: RetrieverDebugInfo
    analyst_draft: AnalystDraft
    fact_check_report: FactCheckReport
    final_answer: FinalAnswer
    timings_ms: dict[str, float]
    trace_events: list[dict]
    errors: list[str]
