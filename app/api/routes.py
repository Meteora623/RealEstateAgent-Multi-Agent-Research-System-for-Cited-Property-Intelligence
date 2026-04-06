from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_pipeline, get_settings_dep, get_system, run_evaluation
from app.core.config import Settings
from app.graph.workflow import RealEstateResearchSystem
from app.ingestion.pipeline import IngestionPipeline
from app.schemas.api import (
    EvaluateRequest,
    EvaluateResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
    StructuredQueryResponse,
)
from app.schemas.retrieval import MetadataFilter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
async def get_config(settings: Settings = Depends(get_settings_dep)) -> dict:
    return {
        "app_name": settings.app_name,
        "env": settings.env,
        "default_retrieval_mode": settings.default_retrieval_mode,
        "default_top_k": settings.default_top_k,
        "default_candidate_pool_multiplier": settings.default_candidate_pool_multiplier,
        "fusion_alpha": settings.fusion_alpha,
        "rerank_top_n": settings.rerank_top_n,
        "semantic_chunking": settings.semantic_chunking,
        "chunking_strategy": settings.chunking_strategy,
        "dataset_profile": settings.dataset_profile,
        "ragas_mode": settings.ragas_mode,
        "use_pinecone": settings.use_pinecone,
        "langsmith_enabled": settings.langsmith_enabled,
    }


@router.post("/ingest")
async def ingest(
    request: IngestRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    try:
        stats = pipeline.run(
            semantic_chunking=request.semantic_chunking,
            chunking_strategy=request.chunking_strategy,
            dataset_profile=request.dataset_profile,
            force_reindex=request.force_reindex,
        )
        return stats.model_dump()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    system: RealEstateResearchSystem = Depends(get_system),
) -> QueryResponse:
    try:
        return await system.run_query(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.post("/query/structured", response_model=StructuredQueryResponse)
async def query_structured(
    request: QueryRequest,
    system: RealEstateResearchSystem = Depends(get_system),
) -> StructuredQueryResponse:
    try:
        return await system.run_query_structured(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Structured query failed: {exc}") from exc


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: EvaluateRequest,
    settings: Settings = Depends(get_settings_dep),
) -> EvaluateResponse:
    try:
        summary = run_evaluation(
            settings=settings,
            run_ragas=request.run_ragas,
            ragas_mode=request.ragas_mode,
            dataset_profile=request.dataset_profile,
            max_queries=request.max_queries,
            output_tag=request.output_tag,
        )
        run_dir = summary.run_dir
        return EvaluateResponse(
            run_dir=run_dir,
            best_configuration=summary.best_configuration,
            corpus_stats_path=f"{run_dir}/corpus_stats.json",
            quality_metrics_path=f"{run_dir}/quality_metrics.json",
            latency_metrics_path=f"{run_dir}/latency_metrics.json",
            comparison_table_path=f"{run_dir}/comparison_table.csv",
            resume_claim_report_path=f"{run_dir}/resume_claim_support.md",
            report_path=f"{run_dir}/evaluation_report.md",
            experiment_metadata={
                "scale_metrics": summary.scale_metrics,
                "configs": [
                    {
                        "config_name": item.config_name,
                        "retrieval_mode": item.retrieval_mode,
                        "chunking_strategy": item.chunking_strategy,
                        "rerank": item.rerank,
                        "metric_source": item.metric_source,
                    }
                    for item in summary.config_results
                ],
                "best_config_rationale": "Selected by weighted quality + retrieval score in evaluation runner.",
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


@router.get("/debug/retrieval")
async def debug_retrieval(
    q: str = Query(..., min_length=3),
    top_k: int = Query(8, ge=1, le=50),
    candidate_pool_size: int = Query(32, ge=4, le=200),
    retrieval_mode: str = Query("hybrid"),
    rerank: bool = Query(True),
    rerank_top_n: int = Query(20, ge=1, le=200),
    fusion_alpha: float = Query(0.65, ge=0.0, le=1.0),
    city: str | None = Query(None),
    neighborhood: str | None = Query(None),
    property_type: str | None = Query(None),
    system: RealEstateResearchSystem = Depends(get_system),
):
    filters = MetadataFilter(city=city, neighborhood=neighborhood, property_type=property_type)
    request = QueryRequest(
        query=q,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        retrieval_mode=retrieval_mode,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        fusion_alpha=fusion_alpha,
        filters=filters if any([city, neighborhood, property_type]) else None,
        include_debug=True,
    )
    state = await system.run_query_state(request)
    return {
        "request_id": state["request_id"],
        "retrieval_round": state.get("retrieval_round", 0),
        "retrieval_debug": state.get("retrieval_debug").model_dump() if state.get("retrieval_debug") else {},
        "rank_transitions": {
            "dense": state.get("retrieval_debug").dense_rankings if state.get("retrieval_debug") else [],
            "sparse": state.get("retrieval_debug").sparse_rankings if state.get("retrieval_debug") else [],
            "fused": state.get("retrieval_debug").fused_rankings if state.get("retrieval_debug") else [],
            "reranked": state.get("retrieval_debug").reranked_rankings if state.get("retrieval_debug") else [],
        },
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "property_id": chunk.property_id,
                "dense_score": chunk.dense_score,
                "sparse_score": chunk.sparse_score,
                "fused_score": chunk.fused_score,
                "rerank_score": chunk.rerank_score,
                "metadata": chunk.metadata,
                "preview": chunk.text[:220],
            }
            for chunk in state.get("retrieved_chunks", [])
        ],
        "timings_ms": state.get("timings_ms", {}),
    }
